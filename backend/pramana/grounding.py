"""Source-grounded tutoring.

Grounding method ("quote-lock"):
1. Retrieve units with hybrid search. If nothing is similar enough, decline before generating.
2. The generator must attach to every source-backed statement a verbatim quote and the id
   of the unit it came from.
3. Code, not a model, checks each quote is really in that unit. A statement whose quotes all
   fail is shown as unverified, never as source-backed.
4. Anything from the model's general knowledge goes in a separate, visibly different lane.
"""
from __future__ import annotations

import difflib
import json
import re
import time

from . import config, llm
from .db import DB
from .embed import HybridIndex
from .llm import INT, NSTR, STR, arr, enum, nullable, obj
from .service import load_graph, load_model, save_updates, today, unit_view

# --------------------------------------------------------------------------- quote-lock


def _norm(s: str) -> str:
    s = s.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\[figure\]", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def locate_quote(quote: str, text: str, min_len: int = 12, fuzzy: float = 0.9) -> bool:
    """True if `quote` appears in `text`, ignoring case, punctuation and line breaks.

    A near-match (>= `fuzzy` similarity against the best window) is accepted so that caption
    line breaks and PDF ligatures do not reject honest quotes.
    """
    q, t = _norm(quote), _norm(text)
    if len(q) < min_len:
        return False
    if q in t:
        return True
    qw, tw = q.split(), t.split()
    if len(qw) > len(tw):
        return False
    best = 0.0
    for i in range(0, len(tw) - len(qw) + 1):
        window = " ".join(tw[i : i + len(qw)])
        best = max(best, difflib.SequenceMatcher(None, q, window, autojunk=False).ratio())
        if best >= fuzzy:
            return True
    return False


# --------------------------------------------------------------------------- prompts

_REWRITE_SYSTEM = "You turn a student's message into an English search query for their course material."
_REWRITE_USER = """Conversation so far:
{history}

Latest student message: {message}

Write one self-contained English search query for the latest message. Resolve pronouns and follow-ups
using the conversation. If the message is in Hindi or Hinglish, translate it. Keep technical terms."""
_REWRITE_SCHEMA = obj(search_query=STR)

_TUTOR_SYSTEM = """You are Pramana, a study companion. You teach strictly from the student's own course material.

Rules for every answer:
- Split your answer into segments. Each segment has a lane.
  - lane "source": every claim in it is supported by the EXCERPTS. Attach at least one citation: the excerpt's
    unit_id and a quote copied character-for-character from that excerpt (8 to 40 words). Do not paraphrase
    inside a quote, do not stitch two places together, and do not quote from an excerpt you did not cite.
  - lane "outside": anything the excerpts do not support. Use it sparingly, only to bridge a gap, and never to
    answer a question the material does not cover. No citations here.
- coverage: "full" if the excerpts answer the question, "partial" if they answer part of it, "none" if they do
  not address it. With "none", write one short source-free segment in lane "outside" saying the material does
  not cover it, and nothing else.
- Excerpts marked [figure description] were produced by reading a diagram. You may cite them like any excerpt.
- Write plain text. Math in Unicode (P(A|B), ∩, Σ, ≤). No Markdown headings, no LaTeX.
- Adapt to the STUDENT STATE: if a prerequisite of the topic is weak, open with a one- or two-sentence bridge to
  it; match depth to their level; if they hold a known misconception, address it directly.
- Reply in the student's language setting. Quotes always stay in the original language of the excerpt.
- If PENDING CHECK is present and the student's message attempts to answer it, set check_result to "correct",
  "partial" or "incorrect" and start your answer by responding to their attempt. Otherwise "not_an_answer".
- End with a check_question when the answer taught something testable: one short question the student can
  answer in a sentence, about the main concept, with the expected answer. Otherwise null.
- concept_ids: ids from CONCEPTS that this exchange is about (at most 3).
- misconception: if the student's message reveals a specific wrong belief, state it in a short sentence. Else null."""

_TUTOR_SCHEMA = obj(
    coverage=enum("full", "partial", "none"),
    segments=arr(obj(lane=enum("source", "outside"), text=STR, citations=arr(obj(unit_id=INT, quote=STR)))),
    concept_ids=arr(STR),
    check_result=enum("correct", "partial", "incorrect", "not_an_answer"),
    check_question=nullable(obj(question=STR, expected_answer=STR, concept_id=STR)),
    misconception=NSTR,
)

_OUTSIDE_SYSTEM = (
    "You are a tutor. The student's course material does not cover their question and they have asked for an "
    "answer from general knowledge anyway. Answer briefly and accurately in plain text with Unicode math. "
    "Reply in the student's language setting."
)
_OUTSIDE_SCHEMA = obj(answer=STR)

LANG = {"en": "English", "hi": "Hindi (Devanagari script)", "hinglish": "Hinglish (Hindi in Latin script mixed with English terms)"}
CHECK_WEIGHT = 0.5          # a chat check answer counts half as much as a quiz answer
MISCONCEPTION_WEIGHT = 0.3  # a misconception voiced in chat is soft negative evidence


def _history(db: DB, student_id: str, n: int = 6) -> str:
    rows = db.q("SELECT role, content FROM chats WHERE student_id=? ORDER BY id DESC LIMIT ?", (student_id, n))
    lines = []
    for r in reversed(rows):
        c = json.loads(r["content"])
        text = c["text"] if r["role"] == "user" else " ".join(s["text"] for s in c.get("segments", []))
        lines.append(f"{r['role']}: {text[:500]}")
    return "\n".join(lines) or "(none)"


def retrieve(index: HybridIndex, query: str, k: int = 8, allowed: set[int] | None = None, mode: str = "hybrid"):
    return index.search(query, k=k, allowed=allowed, mode=mode)


def coverage_gate(index: HybridIndex, query: str, hits, threshold: float | None = None) -> dict:
    """First refusal layer, before any generation.

    Pass if the best embedding match clears the threshold, or if the query contains a
    distinctive term the corpus uses. Tuned to almost never block an answerable question;
    in-domain questions the material does not cover are left to the generator's own
    coverage judgement, the second layer.
    """
    threshold = config.COVERAGE_THRESHOLD if threshold is None else threshold
    top = max((h.dense for h in hits), default=0.0)
    rare = index.rare_terms(query) if top < threshold else []
    return {"pass": top >= threshold or bool(rare), "top_score": round(top, 3), "threshold": threshold,
            "rescued_by": rare}


def _student_state(db: DB, student: dict, model, concept_rows: dict, concept_ids: list[str], t: float) -> str:
    lines = [f"Level: {student.get('level') or 'unknown'}. Language setting: {LANG.get(student.get('language'), 'English')}."]
    for c in concept_ids:
        if c not in concept_rows:
            continue
        lines.append(f"- {concept_rows[c]['name']}: estimated mastery {model.p_now(c, t):.2f}")
        for p in model.g.prereqs[c]:
            if model.p_now(p, t) < model.params.weak:
                lines.append(f"  - weak prerequisite: {concept_rows[p]['name']} ({model.p_now(p, t):.2f})")
    seen = db.q("SELECT misconception, COUNT(*) AS n FROM assessment_items ai JOIN assessments a ON a.id=ai.assessment_id "
                "WHERE a.student_id=? AND misconception IS NOT NULL GROUP BY misconception ORDER BY n DESC LIMIT 3", (student["id"],))
    for m in seen:
        lines.append(f"- misconception seen in quizzes ({m['n']}x): {m['misconception']}")
    return "\n".join(lines)


def answer(db: DB, index: HybridIndex, student_id: str, message: str, allow_outside: bool = False) -> dict:
    """Answer one chat message and update the learner model from it."""
    student = db.one("SELECT * FROM students WHERE id=?", (student_id,))
    if not student:
        raise KeyError(student_id)
    t = today()
    history = _history(db, student_id)
    db.x("INSERT INTO chats(student_id, ts, role, content) VALUES(?,?,?,?)",
         (student_id, time.time(), "user", json.dumps({"text": message})))

    needs_rewrite = history != "(none)" or student.get("language") != "en" or not message.isascii()
    query = message
    if needs_rewrite:
        query = llm.ask("bulk", _REWRITE_SYSTEM, _REWRITE_USER.format(history=history, message=message),
                        _REWRITE_SCHEMA, max_tokens=500)["search_query"]

    hits = retrieve(index, query, k=8)
    gate = coverage_gate(index, query, hits)
    pending = db.get_kv(f"check:{student_id}")
    retrieval = {"query": query, **gate, "unit_ids": [h.unit_id for h in hits]}

    def finish(result: dict) -> dict:
        result["retrieval"] = retrieval
        db.x("INSERT INTO chats(student_id, ts, role, content) VALUES(?,?,?,?)",
             (student_id, time.time(), "assistant", json.dumps(result)))
        return result

    def decline(reason: str) -> dict:
        nearest = [unit_view(db, h.unit_id, False) for h in hits[:3]]
        return finish({"status": "declined", "coverage": "none", "segments": [], "reason": reason,
                       "nearest": nearest, "check_question": None, "mastery_updates": []})

    if allow_outside:
        out = llm.ask("generator", _OUTSIDE_SYSTEM + f" Language setting: {LANG.get(student.get('language'), 'English')}.",
                      message, _OUTSIDE_SCHEMA, max_tokens=4000)
        return finish({"status": "outside", "coverage": "none", "check_question": None, "mastery_updates": [],
                       "segments": [{"lane": "outside", "text": out["answer"], "citations": []}]})

    # Coverage gate: skip generation when nothing retrieved is close, unless the student is
    # answering a pending check question (their reply need not resemble the material).
    if not gate["pass"] and not pending:
        return decline("Nothing in your course material is close enough to this question.")

    graph, concept_rows = load_graph(db)
    model = load_model(db, student_id, graph)
    views = {h.unit_id: unit_view(db, h.unit_id, True) for h in hits}
    related: list[str] = []
    for v in views.values():
        for c in v["concepts"][:1]:
            if c not in related:
                related.append(c)
    excerpts = "\n\n".join(
        f"[unit_id {uid}] {v['label']}{' [figure description]' if v['derived'] else ''}\n{v['text']}" for uid, v in views.items())
    concepts_block = "\n".join(f"- {c}: {concept_rows[c]['name']}" for c in related[:8] if c in concept_rows) or "(none tagged)"
    user = (f"STUDENT STATE\n{_student_state(db, student, model, concept_rows, related[:4], t)}\n\n"
            f"CONCEPTS\n{concepts_block}\n\n"
            f"PENDING CHECK\n{json.dumps(pending) if pending else '(none)'}\n\n"
            f"CONVERSATION\n{history}\n\nEXCERPTS\n{excerpts}\n\nSTUDENT MESSAGE\n{message}")
    out = llm.ask("generator", _TUTOR_SYSTEM, user, _TUTOR_SCHEMA, max_tokens=8000)

    if out["coverage"] == "none" and out["check_result"] == "not_an_answer":
        return decline("Your course material does not cover this.")

    segments, n_cite, n_ok = [], 0, 0
    for seg in out["segments"]:
        cites = []
        for c in seg["citations"] if seg["lane"] == "source" else []:
            n_cite += 1
            v = views.get(c["unit_id"])
            ok = bool(v) and locate_quote(c["quote"], v["text"])
            n_ok += ok
            if ok:
                cites.append({"unit": v, "quote": c["quote"]})
        lane = seg["lane"]
        if lane == "source" and not cites:
            lane = "unverified"       # claimed a source but no quote survived the check
        segments.append({"lane": lane, "text": seg["text"], "citations": cites})

    updates = []
    score = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}.get(out["check_result"])
    if pending and score is not None and pending.get("concept_id") in model.s:
        ups = model.observe(pending["concept_id"], score, "short", 2, t, weight=CHECK_WEIGHT)
        updates += save_updates(db, student_id, model, ups, "chat_check", score, pending["question"][:80])
    if out["misconception"]:
        target = next((c for c in out["concept_ids"] if c in model.s), None)
        if target:
            ups = model.observe(target, 0.0, "short", 2, t, weight=MISCONCEPTION_WEIGHT, learn=False)
            updates += save_updates(db, student_id, model, ups, "chat_misconception", 0.0, out["misconception"][:120])
    check = out["check_question"] if out["check_question"] and out["check_question"]["concept_id"] in model.s else None
    db.set_kv(f"check:{student_id}", check)

    return finish({
        "status": "answered", "coverage": out["coverage"], "segments": segments,
        "check_question": {"question": check["question"], "concept_id": check["concept_id"]} if check else None,
        "check_result": out["check_result"] if pending else None,
        "misconception": out["misconception"], "concept_ids": [c for c in out["concept_ids"] if c in model.s],
        "mastery_updates": updates,
        "citation_check": {"claimed": n_cite, "verified": n_ok},
    })
