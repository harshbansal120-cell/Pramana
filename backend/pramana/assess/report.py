"""Answer handling and the post-assessment report."""
from __future__ import annotations

import json
import time

from ..db import DB
from ..grounding import LANG
from ..service import load_graph, load_model, save_updates, today, unit_view
from . import generate as G
from . import verify as V


def submit_answer(db: DB, assessment_id: str, idx: int, response: str) -> dict:
    """Grade one answer, update mastery, and return cited feedback."""
    a = db.one("SELECT * FROM assessments WHERE id=?", (assessment_id,))
    item = db.one("SELECT * FROM assessment_items WHERE assessment_id=? AND idx=?", (assessment_id, idx))
    if not a or not item:
        raise KeyError("assessment item not found")
    if item["answered_at"]:
        return json.loads(item["feedback"])
    row = db.one("SELECT * FROM questions WHERE id=?", (item["question_id"],))
    q = json.loads(row["payload"])
    graph, concept_rows = load_graph(db)
    concept = concept_rows[row["concept_id"]]
    student = db.one("SELECT * FROM students WHERE id=?", (a["student_id"],))
    g = V.grade(q, response, concept["misconceptions"], LANG.get(student["language"], "English"))

    model = load_model(db, a["student_id"], graph)
    diagnostic = a["mode"] == "diagnostic"
    ups = model.observe(row["concept_id"], g["score"], row["qtype"], row["difficulty"], today(), learn=not diagnostic)
    updates = save_updates(db, a["student_id"], model, ups, a["mode"], g["score"], f"q{row['id']}")

    revealed = G.public_question(db, row["id"], idx, reveal=True)
    feedback = {
        "idx": idx, "score": g["score"], "note": g["note"], "misconception": g["misconception"],
        "correct_answer": revealed["answer"], "solution": revealed["solution"],
        "citation": {"unit": revealed["source"], "quote": revealed["source_quote"]},
        "mastery_updates": updates, "verification": revealed["verification"],
    }
    db.x("UPDATE assessment_items SET response=?, score=?, misconception=?, feedback=?, answered_at=? "
         "WHERE assessment_id=? AND idx=?",
         (response, g["score"], g["misconception"], json.dumps(feedback), time.time(), assessment_id, idx))

    # Diagnostics are adaptive: choose the next question now that the model has moved.
    scope = json.loads(a["scope"])
    done = db.one("SELECT COUNT(*) AS n FROM assessment_items WHERE assessment_id=?", (assessment_id,))["n"]
    if diagnostic and done < scope["n"]:
        items = db.q("SELECT ai.question_id, q.concept_id FROM assessment_items ai JOIN questions q ON q.id=ai.question_id "
                     "WHERE ai.assessment_id=?", (assessment_id,))
        served: dict[str, int] = {}
        for it in items:
            served[it["concept_id"]] = served.get(it["concept_id"], 0) + 1
        qid = G.next_slot(db, a, model, concept_rows, {it["question_id"] for it in items}, served, done)
        if qid is not None:
            db.x("INSERT INTO assessment_items(assessment_id, idx, question_id) VALUES(?,?,?)", (assessment_id, done, qid))
            feedback["next"] = G.public_question(db, qid, done)
    if a["mode"] in ("mock", "diagnostic"):      # these modes hold feedback back until the report
        return {"idx": idx, "recorded": True, "next": feedback.get("next")}
    return feedback


def build_report(db: DB, assessment_id: str) -> dict:
    """Scores by concept and topic, likely misconceptions, root causes, and what to do next."""
    a = db.one("SELECT * FROM assessments WHERE id=?", (assessment_id,))
    if not a:
        raise KeyError(assessment_id)
    if not a["finished_at"]:
        db.x("UPDATE assessments SET finished_at=? WHERE id=?", (time.time(), assessment_id))
        if a["mode"] == "diagnostic":
            db.x("UPDATE students SET diagnosed=1 WHERE id=?", (a["student_id"],))
    graph, concept_rows = load_graph(db)
    model = load_model(db, a["student_id"], graph)
    t = today()
    items = db.q("SELECT ai.*, q.concept_id, q.qtype, q.difficulty FROM assessment_items ai JOIN questions q ON q.id=ai.question_id "
                 "WHERE ai.assessment_id=? ORDER BY ai.idx", (assessment_id,))
    answered = [i for i in items if i["answered_at"]]

    by_concept: dict[str, dict] = {}
    for i in answered:
        c = by_concept.setdefault(i["concept_id"], {"n": 0, "score": 0.0})
        c["n"] += 1
        c["score"] += i["score"]
    concepts = []
    for cid, agg in by_concept.items():
        first = db.one("SELECT p_before FROM events WHERE student_id=? AND concept_id=? AND ref IN (%s) ORDER BY id LIMIT 1"
                       % ",".join("?" * len(answered)),
                       (a["student_id"], cid, *[f"q{i['question_id']}" for i in answered]))
        concepts.append({"concept_id": cid, "name": concept_rows[cid]["name"], "topic": concept_rows[cid]["topic"],
                         "questions": agg["n"], "accuracy": round(agg["score"] / agg["n"], 2),
                         "mastery_before": round(first["p_before"], 3) if first else None,
                         "mastery_after": round(model.p_now(cid, t), 3)})
    concepts.sort(key=lambda c: c["mastery_after"])

    topics: dict[str, dict] = {}
    for c in concepts:
        tp = topics.setdefault(c["topic"], {"topic": c["topic"], "questions": 0, "score": 0.0})
        tp["questions"] += c["questions"]
        tp["score"] += c["accuracy"] * c["questions"]
    for tp in topics.values():
        tp["accuracy"] = round(tp.pop("score") / tp["questions"], 2)

    mis: dict[str, dict] = {}
    for i in answered:
        if i["misconception"]:
            m = mis.setdefault(i["misconception"], {"misconception": i["misconception"], "count": 0, "concepts": set()})
            m["count"] += 1
            m["concepts"].add(concept_rows[i["concept_id"]]["name"])
    # One wrong option can be a slip. Call a misconception "likely" only when it recurs.
    misconceptions = sorted(({**m, "concepts": sorted(m["concepts"]), "strength": "likely" if m["count"] >= 2 else "possible"}
                             for m in mis.values()), key=lambda m: -m["count"])

    weak = [c for c in concepts if c["mastery_after"] < 0.6]
    causes = []
    for d in model.diagnose(t):
        if d["concept"] in by_concept:
            causes.append({"concept": concept_rows[d["concept"]]["name"], "root_cause": concept_rows[d["root_cause"]]["name"],
                           "root_cause_id": d["root_cause"], "p_root": round(d["p_root"], 2)})

    # Point each weak concept at the best place in the material to re-study it.
    actions = []
    for c in weak[:4]:
        target = next((x["root_cause_id"] for x in causes if x["concept"] == c["name"]), c["concept_id"])
        best = db.one("SELECT unit_id FROM unit_concepts WHERE concept_id=? ORDER BY score DESC LIMIT 1", (target,))
        actions.append({"concept": concept_rows[target]["name"], "concept_id": target,
                        "why": f"needed for {c['name']}" if target != c["concept_id"] else "low mastery after this assessment",
                        "study": unit_view(db, best["unit_id"], True) if best else None})

    total = sum(i["score"] for i in answered)
    return {
        "assessment_id": assessment_id, "mode": a["mode"], "answered": len(answered), "total": len(items),
        "score": round(total, 1), "percent": round(100 * total / len(answered)) if answered else 0,
        "concepts": concepts, "topics": list(topics.values()), "weak_topics": [c["name"] for c in weak],
        "misconceptions": misconceptions, "root_causes": causes, "next_steps": actions,
        "items": [{"question": G.public_question(db, i["question_id"], i["idx"], reveal=True),
                   "response": i["response"], "score": i["score"], "misconception": i["misconception"]} for i in answered],
    }
