"""Question generation and assessment assembly.

Questions live in a bank. An assessment first draws verified questions the student has
never seen, and generates (then verifies) new ones only for the gaps. A student is never
served the same question twice.
"""
from __future__ import annotations

import json
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from .. import llm
from ..db import DB, to_blob
from ..embed import embed_passages
from ..llm import BOOL, INT, NNUM, NSTR, STR, arr, obj
from ..service import load_graph, load_model, today, unit_view
from . import verify as V

QTYPES = ("mcq", "numeric", "short")

_GEN_SYSTEM = """You write exam questions for a university course, strictly from the course's own material.

Requirements:
- The question tests the stated CONCEPT and can be answered from the EXCERPTS plus the course's standard
  definitions. Pick the one excerpt it rests on most and return its unit_id with source_quote: 8 to 40 words
  copied character-for-character from that excerpt.
- Do not ask about the wording of the material ("according to the lecture..."). Ask about the idea.
- Self-contained: define every quantity. Exactly one defensible answer.
- Plain text, math in Unicode. No Markdown, no LaTeX.
- difficulty 1 = recall or a one-step use of a definition; 2 = apply the idea to a new situation;
  3 = multi-step reasoning or combining it with a prerequisite.

By type:
- mcq: exactly four options, one correct. Each wrong option must be what a student holding a specific wrong
  belief would choose; put that belief in `misconception` (prefer the KNOWN MISCONCEPTIONS, one short sentence).
  `misconception` is null for the correct option. Put the correct option in a random position. If the answer is
  a number, also give numeric_answer and check_code.
- numeric: a problem with one numerical answer. Give numeric_answer as a decimal, a tolerance (absolute), and
  check_code. options is empty.
- short: answerable in one to three sentences. answer_text is the model answer. options is empty,
  numeric_answer and check_code are null.

check_code is Python that independently recomputes the answer:
- def exact(): return the answer using exact arithmetic (fractions.Fraction, math.comb, sympy). Compute it from
  the problem's givens. Never hard-code the final answer.
- def simulate(rng, n): optional but strongly preferred for probability and expectation questions. Estimate the
  same quantity by simulating the experiment n times using only `rng` (a random.Random). Return the estimate.
- Allowed imports: math, fractions, itertools, random, statistics, sympy, functools, collections, decimal.

solution: a worked explanation a student can learn from, 2 to 6 sentences."""

_GEN_USER = """CONCEPT: {name}
Definition: {definition}
KNOWN MISCONCEPTIONS: {misconceptions}

TYPE: {qtype}
DIFFICULTY: {difficulty}

EXCERPTS
{excerpts}

Questions that already exist for this concept. Write something clearly different in scenario and in what is asked:
{avoid}"""

_GEN_SCHEMA = obj(
    stem=STR,
    options=arr(obj(text=STR, correct=BOOL, misconception=NSTR)),
    answer_text=STR, numeric_answer=NNUM, tolerance=NNUM, solution=STR,
    unit_id=INT, source_quote=STR, check_code=NSTR,
)


def _evidence(db: DB, concept_id: str, k: int = 4) -> list[dict]:
    """Best-tagged units for a concept, spread across sources and rotated so questions vary."""
    rows = db.q("SELECT uc.unit_id, uc.score, u.source_id, u.kind FROM unit_concepts uc JOIN units u ON u.id=uc.unit_id "
                "WHERE uc.concept_id=? AND length(u.text) > 120 ORDER BY uc.score DESC LIMIT 24", (concept_id,))
    used = {r["unit_id"]: r["n"] for r in db.q(
        "SELECT unit_id, COUNT(*) AS n FROM questions WHERE concept_id=? GROUP BY unit_id", (concept_id,))}
    rows.sort(key=lambda r: (used.get(r["unit_id"], 0), -r["score"]))
    picked, per_source = [], {}
    for r in rows:
        if per_source.get(r["source_id"], 0) >= 2:
            continue
        per_source[r["source_id"]] = per_source.get(r["source_id"], 0) + 1
        picked.append(unit_view(db, r["unit_id"], False))
        if len(picked) == k:
            break
    return picked


def make_question(db: DB, concept: dict, qtype: str, difficulty: int, attempts: int = 2) -> int | None:
    """Generate and verify one question. Returns its id, or None if no attempt passed the gate.

    Rejected attempts are stored too, so the rejection rate can be reported.
    """
    units = _evidence(db, concept["id"])
    if not units:
        return None
    by_id = {u["id"]: u for u in units}
    excerpts = "\n\n".join(f"[unit_id {u['id']}] {u['label']}{' [figure description]' if u['derived'] else ''}\n{u['text']}"
                           for u in units)
    for _ in range(attempts):
        existing = db.q("SELECT stem FROM questions WHERE concept_id=? AND status='verified' ORDER BY id DESC LIMIT 12",
                        (concept["id"],))
        avoid = "\n".join(f"- {r['stem'][:200]}" for r in existing) or "(none yet)"
        q = llm.ask("generator", _GEN_SYSTEM, _GEN_USER.format(
            name=concept["name"], definition=concept["definition"],
            misconceptions="; ".join(concept["misconceptions"]) or "(none listed)",
            qtype=qtype, difficulty=difficulty, excerpts=excerpts, avoid=avoid), _GEN_SCHEMA, max_tokens=8000)
        q.update(type=qtype, concept_id=concept["id"], difficulty=difficulty)
        if qtype == "mcq":
            random.shuffle(q["options"])     # do not rely on the model to randomise the key position
        unit = by_id.get(q["unit_id"])
        emb = embed_passages([q["stem"]])[0]
        if unit is None:
            ver = {"checks": {"quote_lock": False}, "notes": {"quote_lock": "cited a unit that was not provided"}, "passed": False}
        else:
            ver = V.verify(db, q, unit["text"], emb)
        qid = db.x(
            "INSERT INTO questions(concept_id, unit_id, qtype, difficulty, stem, payload, verification, status, emb, created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (concept["id"], q["unit_id"] if unit else None, qtype, difficulty, q["stem"], json.dumps(q), json.dumps(ver),
             "verified" if ver["passed"] else "rejected", to_blob(emb), time.time()))
        if ver["passed"]:
            return qid
    return None


def _unseen(db: DB, student_id: str, concept_id: str, qtype: str | None, difficulty: int | None,
            taken: set[int]) -> int | None:
    """A verified bank question this student has never been served, least-exposed first."""
    rows = db.q(
        "SELECT q.id, q.qtype, q.difficulty, (SELECT COUNT(*) FROM assessment_items ai WHERE ai.question_id=q.id) AS exposure "
        "FROM questions q WHERE q.status='verified' AND q.concept_id=? AND q.id NOT IN ("
        " SELECT ai.question_id FROM assessment_items ai JOIN assessments a ON a.id=ai.assessment_id WHERE a.student_id=?)",
        (concept_id, student_id))
    rows = [r for r in rows if r["id"] not in taken and (qtype is None or r["qtype"] == qtype)]
    if not rows:
        return None
    rows.sort(key=lambda r: (abs(r["difficulty"] - (difficulty or r["difficulty"])), r["exposure"], random.random()))
    return rows[0]["id"]


def public_question(db: DB, qid: int, idx: int, reveal: bool = False) -> dict:
    row = db.one("SELECT q.*, c.name AS cname, c.topic FROM questions q JOIN concepts c ON c.id=q.concept_id WHERE q.id=?", (qid,))
    p = json.loads(row["payload"])
    out = {"idx": idx, "question_id": qid, "type": row["qtype"], "stem": row["stem"], "difficulty": row["difficulty"],
           "concept": {"id": row["concept_id"], "name": row["cname"], "topic": row["topic"]},
           "options": [o["text"] for o in p["options"]] if row["qtype"] == "mcq" else []}
    if reveal:
        out["source"] = unit_view(db, row["unit_id"], True)
        out["source_quote"] = p["source_quote"]
        out["solution"] = p["solution"]
        out["answer"] = (next(o["text"] for o in p["options"] if o["correct"]) if row["qtype"] == "mcq"
                         else (f"{p['numeric_answer']:g}" if row["qtype"] == "numeric" else p["answer_text"]))
        out["verification"] = json.loads(row["verification"])["checks"]
    return out


def _fill_slot(db: DB, student_id: str, concept: dict, qtype: str, difficulty: int, taken: set[int]) -> int | None:
    qid = _unseen(db, student_id, concept["id"], qtype, difficulty, taken)
    if qid is None and llm.available("generator"):
        qid = make_question(db, concept, qtype, difficulty)
    if qid is None:                                   # fall back to any unseen type for this concept
        qid = _unseen(db, student_id, concept["id"], None, difficulty, taken)
    return qid


def next_slot(db: DB, assessment: dict, model, concept_rows: dict, taken: set[int], served: dict[str, int],
              position: int) -> int | None:
    """Pick concept, type and difficulty for one slot from the current learner state, then fill it."""
    scope = json.loads(assessment["scope"])
    candidates = [c for c in scope["concept_ids"] if c in model.s]
    t = today()
    mode = "diagnostic" if assessment["mode"] == "diagnostic" else "practice"
    tried: set[str] = set()
    while len(tried) < len(candidates):
        c = model.select([x for x in candidates if x not in tried], t, served, mode)
        qtype = "mcq" if mode == "diagnostic" else scope["types"][position % len(scope["types"])]
        difficulty = 2 if mode == "diagnostic" else model.pick_difficulty(c, qtype, t)
        qid = _fill_slot(db, assessment["student_id"], concept_rows[c], qtype, difficulty, taken)
        if qid is not None:
            served[c] = served.get(c, 0) + 1
            return qid
        tried.add(c)
    return None


def create_assessment(db: DB, student_id: str, mode: str, concept_ids: list[str], n: int, types: list[str]) -> dict:
    """mode: quiz (feedback after each answer), mock (feedback at the end), diagnostic (adaptive, one at a time)."""
    graph, concept_rows = load_graph(db)
    concept_ids = [c for c in concept_ids if c in concept_rows] or list(concept_rows)
    types = [t for t in types if t in QTYPES] or ["mcq"]
    aid = uuid.uuid4().hex[:12]
    scope = {"concept_ids": concept_ids, "types": types, "n": n}
    db.x("INSERT INTO assessments(id, student_id, mode, scope, created_at) VALUES(?,?,?,?,?)",
         (aid, student_id, mode, json.dumps(scope), time.time()))
    assessment = db.one("SELECT * FROM assessments WHERE id=?", (aid,))
    model = load_model(db, student_id, graph)
    taken: set[int] = set()
    served: dict[str, int] = {}
    first = 1 if mode == "diagnostic" else n      # diagnostics choose each question after the previous answer

    # Plan the slots first (cheap), then fill them in parallel (generation is the slow part).
    t = today()
    plan = []
    for i in range(first):
        c = model.select(concept_ids, t, served, "diagnostic" if mode == "diagnostic" else "practice")
        served[c] = served.get(c, 0) + 1
        qtype = "mcq" if mode == "diagnostic" else types[i % len(types)]
        plan.append((c, qtype, 2 if mode == "diagnostic" else model.pick_difficulty(c, qtype, t)))

    def fill(slot):
        c, qtype, difficulty = slot
        return _fill_slot(db, student_id, concept_rows[c], qtype, difficulty, set())

    with ThreadPoolExecutor(max_workers=4) as pool:
        qids = list(pool.map(fill, plan))
    idx = 0
    for qid in qids:
        if qid is None or qid in taken:
            continue
        taken.add(qid)
        db.x("INSERT INTO assessment_items(assessment_id, idx, question_id) VALUES(?,?,?)", (aid, idx, qid))
        idx += 1
    while idx < first:                            # top up slots that collided or could not be filled
        qid = next_slot(db, assessment, model, concept_rows, taken, served, idx)
        if qid is None:
            break
        taken.add(qid)
        db.x("INSERT INTO assessment_items(assessment_id, idx, question_id) VALUES(?,?,?)", (aid, idx, qid))
        idx += 1
    return {"id": aid, "mode": mode, "total": n,
            "questions": [public_question(db, r["question_id"], r["idx"]) for r in db.q(
                "SELECT idx, question_id FROM assessment_items WHERE assessment_id=? ORDER BY idx", (aid,))]}


def prebuild_bank(db: DB, per_concept: int = 2, types: list[str] | None = None) -> dict:
    """Generate verified questions ahead of time so quizzes start instantly."""
    _, concept_rows = load_graph(db)
    types = types or list(QTYPES)
    jobs = []
    for c in concept_rows.values():
        have = db.one("SELECT COUNT(*) AS n FROM questions WHERE concept_id=? AND status='verified'", (c["id"],))["n"]
        for i in range(have, per_concept):
            jobs.append((c, types[i % len(types)], 1 + i % 3))
    with ThreadPoolExecutor(max_workers=4) as pool:
        made = list(pool.map(lambda j: make_question(db, *j), jobs))
    return {"requested": len(jobs), "verified": sum(m is not None for m in made), "bank": bank_stats(db)}


def bank_stats(db: DB) -> dict:
    """Verification gate outcomes: how many generated questions each check rejected."""
    rows = db.q("SELECT status, verification FROM questions")
    failed: dict[str, int] = {}
    for r in rows:
        if r["status"] == "rejected":
            for name, ok in json.loads(r["verification"])["checks"].items():
                if ok is False:
                    failed[name] = failed.get(name, 0) + 1
    total = len(rows)
    verified = sum(r["status"] == "verified" for r in rows)
    return {"generated": total, "verified": verified, "rejected": total - verified,
            "pass_rate": round(verified / total, 3) if total else None, "rejections_by_check": failed}
