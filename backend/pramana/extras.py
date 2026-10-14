"""Revision material: cited flashcards for weak concepts and a study schedule."""
from __future__ import annotations

import datetime as dt

from . import llm
from .db import DB
from .grounding import LANG, locate_quote
from .learner import schedule
from .llm import INT, STR, arr, obj
from .service import load_graph, load_model, today, unit_view

_CARD_SYSTEM = (
    "You write revision flashcards strictly from the provided course excerpts. Each card tests one idea. "
    "The back must be supported by a quote copied character-for-character (8 to 40 words) from the excerpt "
    "whose unit_id you give. Plain text, Unicode math."
)
_CARD_USER = """Student language setting: {language}. Write the front and back in that language; quotes stay as in the source.

Write {n} flashcards for the concept "{name}" ({definition}).
{misconception}

EXCERPTS
{excerpts}"""
_CARD_SCHEMA = obj(cards=arr(obj(front=STR, back=STR, unit_id=INT, quote=STR)))


def flashcards(db: DB, student_id: str, per_concept: int = 3, max_concepts: int = 4) -> list[dict]:
    """Flashcards for the student's weakest concepts. A card is kept only if its quote checks out."""
    student = db.one("SELECT * FROM students WHERE id=?", (student_id,))
    graph, concept_rows = load_graph(db)
    model = load_model(db, student_id, graph)
    t = today()
    weak = [c for c, _ in model.weak_concepts(t)]
    # Prefer concepts the student has actually been tested on; fall back to the lowest estimates.
    weak.sort(key=lambda c: (model.s[c].n == 0, model.p_now(c, t)))
    out = []
    for cid in weak[:max_concepts]:
        c = concept_rows[cid]
        rows = db.q("SELECT unit_id FROM unit_concepts uc JOIN units u ON u.id=uc.unit_id WHERE uc.concept_id=? "
                    "AND length(u.text) > 120 ORDER BY uc.score DESC LIMIT 4", (cid,))
        units = {r["unit_id"]: unit_view(db, r["unit_id"], True) for r in rows}
        if not units:
            continue
        seen = db.q("SELECT ai.misconception FROM assessment_items ai JOIN assessments a ON a.id=ai.assessment_id "
                    "JOIN questions q ON q.id=ai.question_id WHERE a.student_id=? AND q.concept_id=? AND ai.misconception IS NOT NULL",
                    (student_id, cid))
        mis = ("This student has shown this misconception; make one card confront it: " + seen[0]["misconception"]) if seen else ""
        excerpts = "\n\n".join(f"[unit_id {uid}] {u['label']}\n{u['text']}" for uid, u in units.items())
        res = llm.ask("generator", _CARD_SYSTEM, _CARD_USER.format(
            language=LANG.get(student["language"], "English"), n=per_concept, name=c["name"],
            definition=c["definition"], misconception=mis, excerpts=excerpts), _CARD_SCHEMA, max_tokens=4000)
        for card in res["cards"]:
            u = units.get(card["unit_id"])
            if u and locate_quote(card["quote"], u["text"]):
                out.append({"concept_id": cid, "concept": c["name"], "mastery": round(model.p_now(cid, t), 2),
                            "front": card["front"], "back": card["back"], "citation": {"unit": u, "quote": card["quote"]}})
    return out


def study_schedule(db: DB, student_id: str, minutes_per_day: int = 45) -> dict:
    student = db.one("SELECT * FROM students WHERE id=?", (student_id,))
    graph, concept_rows = load_graph(db)
    model = load_model(db, student_id, graph)
    start = dt.date.today()
    try:
        days = (dt.date.fromisoformat(student["exam_date"]) - start).days
    except (TypeError, ValueError):
        days = 14
    days = max(1, days)
    plan = schedule(model, today(), days, minutes_per_day)
    for day in plan:
        day["date"] = (start + dt.timedelta(days=day["day"])).isoformat()
        for b in day["blocks"]:
            b["name"] = concept_rows[b["concept"]]["name"]
            b["topic"] = concept_rows[b["concept"]]["topic"]
    return {"exam_date": student.get("exam_date"), "days_to_exam": days, "minutes_per_day": minutes_per_day,
            "truncated": days > 60, "plan": plan}
