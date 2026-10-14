"""End-to-end paths with the model calls replaced by scripted responses.

These tests exercise the real retrieval, quote-lock, verification gate, grading, learner
updates and reporting code. Only `llm.ask` is faked, so they say nothing about model quality.
"""
import json

import pytest

from pramana import grounding, llm
from pramana.assess import generate as G
from pramana.assess import report as R
from pramana.db import DB, to_blob
from pramana.embed import HybridIndex, embed_missing_units, embed_passages
from pramana.service import load_model, today

TPT = ("Total probability theorem. Divide and conquer. Partition of sample space into A1, A2, A3. "
       "Have P(B | Ai), for every i. One way of computing P(B): P(B) = P(A1)P(B | A1) + P(A2)P(B | A2) + P(A3)P(B | A3).")
COND = ("Conditional probability. P(A | B) = probability of A, given that B occurred. B is our new universe. "
        "Definition: assuming P(B) > 0, P(A | B) = P(A and B) / P(B).")


@pytest.fixture()
def world(tmp_path, monkeypatch):
    db = DB(tmp_path / "t.sqlite")
    db.x("INSERT INTO sources(id, kind, title, uri, order_idx, meta) VALUES('deck','slides','Lecture 2 slides','x',0,'{}')")
    for i, text in enumerate((COND, TPT), start=1):
        db.x("INSERT INTO units(source_id, kind, loc_type, loc_start, loc_end, text, image_path, order_idx, meta) "
             "VALUES('deck','slide','slide',?,?,?,?,?,'{}')", (i, i, text, f"renders/deck/slide_{i:03d}.png", i))
    embed_missing_units(db)
    concepts = [("conditional-probability", "Conditional probability", "P(A|B) = P(A and B)/P(B)."),
                ("total-probability", "Total probability theorem", "P(B) as a weighted sum over a partition.")]
    vecs = embed_passages([f"{n}: {d}" for _, n, d in concepts])
    for i, ((cid, name, definition), v) in enumerate(zip(concepts, vecs)):
        db.x("INSERT INTO concepts(id, name, definition, topic, subtopic, order_idx, misconceptions, emb) VALUES(?,?,?,?,?,?,?,?)",
             (cid, name, definition, "Conditioning", "Tools", i, json.dumps(["Confuses P(A|B) with P(B|A)"]), to_blob(v)))
    db.x("INSERT INTO prereqs VALUES('total-probability','conditional-probability')")
    db.many("INSERT INTO unit_concepts VALUES(?,?,?)", [(1, "conditional-probability", 0.9), (2, "total-probability", 0.9)])
    db.x("INSERT INTO students(id, name, level, language, created_at) VALUES('s1','Asha','beginner','en',0)")
    monkeypatch.setattr(llm, "available", lambda role="generator": True)
    return db, HybridIndex(db)


def script(monkeypatch, responses):
    """Make llm.ask return the next scripted response; record the prompts it was given."""
    calls = []

    def fake(role, system, user, schema, **kw):
        calls.append((role, user))
        r = responses.pop(0)
        return r(user) if callable(r) else r

    monkeypatch.setattr(llm, "ask", fake)
    return calls


def tutor_reply(segments, check=None, result="not_an_answer", coverage="full", misconception=None):
    return {"coverage": coverage, "segments": segments, "concept_ids": ["total-probability"],
            "check_result": result, "check_question": check, "misconception": misconception}


def test_off_topic_question_is_declined_without_calling_the_model(world, monkeypatch):
    db, index = world
    calls = script(monkeypatch, [])
    res = grounding.answer(db, index, "s1", "How do mRNA vaccines work?")
    assert res["status"] == "declined" and calls == []
    assert res["retrieval"]["top_score"] < res["retrieval"]["threshold"]


def test_fabricated_quote_is_demoted_and_real_quote_is_cited(world, monkeypatch):
    db, index = world
    script(monkeypatch, [tutor_reply([
        {"lane": "source", "text": "It splits P(B) over a partition.",
         "citations": [{"unit_id": 2, "quote": "Partition of sample space into A1, A2, A3"}]},
        {"lane": "source", "text": "It was proved by Laplace in 1812.",
         "citations": [{"unit_id": 2, "quote": "first proved by Laplace in the year 1812 in Paris"}]},
        {"lane": "outside", "text": "It is also called the law of total probability.", "citations": []},
    ])])
    res = grounding.answer(db, index, "s1", "What is the total probability theorem?")
    lanes = [s["lane"] for s in res["segments"]]
    assert lanes == ["source", "unverified", "outside"]
    assert res["citation_check"] == {"claimed": 2, "verified": 1}
    cited = res["segments"][0]["citations"][0]["unit"]
    assert cited["label"] == "Slide 2 of Lecture 2 slides" and cited["open"]["type"] == "image"


def test_chat_check_question_updates_mastery(world, monkeypatch):
    db, index = world
    check = {"question": "What must the events A1..An form?", "expected_answer": "A partition", "concept_id": "total-probability"}
    seg = [{"lane": "source", "text": "Use a partition.", "citations": [{"unit_id": 2, "quote": "Partition of sample space into A1, A2, A3"}]}]
    script(monkeypatch, [tutor_reply(seg, check=check),
                         {"search_query": "partition of the sample space total probability theorem"},
                         tutor_reply(seg, result="correct")])
    first = grounding.answer(db, index, "s1", "What is the total probability theorem?")
    assert first["check_question"]["question"].startswith("What must") and first["mastery_updates"] == []
    before = load_model(db, "s1").p_now("total-probability", today())
    second = grounding.answer(db, index, "s1", "a partition of the sample space")
    assert second["check_result"] == "correct"
    moved = {u["concept_id"]: u for u in second["mastery_updates"]}
    assert moved["total-probability"]["after"] > before
    assert db.one("SELECT COUNT(*) AS n FROM events WHERE kind='chat_check'")["n"] == 1


def numeric_question(key, code_value):
    return {
        "stem": "Events A1, A2 partition the sample space with P(A1) = 0.3. If P(B|A1) = 0.5 and P(B|A2) = 0.2, find P(B).",
        "options": [], "answer_text": str(key), "numeric_answer": key, "tolerance": 0.001,
        "solution": "P(B) = 0.3*0.5 + 0.7*0.2 = 0.29.", "unit_id": 2,
        "source_quote": "One way of computing P(B): P(B) = P(A1)P(B | A1)",
        "check_code": f"from fractions import Fraction as F\ndef exact():\n    return {code_value}\n",
    }


SOLVED = {"chosen_option": None, "numeric_answer": 0.29, "short_answer": None, "answerable": True, "ambiguous": False, "issue": None}
GOOD_CODE = "F(3,10)*F(1,2) + F(7,10)*F(1,5)"


def test_gate_rejects_a_wrong_answer_key_and_accepts_the_corrected_question(world, monkeypatch):
    db, _ = world
    concept = {"id": "total-probability", "name": "Total probability theorem", "definition": "d", "misconceptions": []}
    # Attempt 1: key says 0.35 but the executed code gives 0.29. Attempt 2 is consistent.
    script(monkeypatch, [numeric_question(0.35, GOOD_CODE), numeric_question(0.29, GOOD_CODE), SOLVED])
    qid = G.make_question(db, concept, "numeric", 2)
    rows = db.q("SELECT id, status, verification FROM questions ORDER BY id")
    assert [r["status"] for r in rows] == ["rejected", "verified"] and qid == rows[1]["id"]
    assert json.loads(rows[0]["verification"])["checks"]["code_exact"] is False
    checks = json.loads(rows[1]["verification"])["checks"]
    assert checks == {"structure": True, "quote_lock": True, "code_exact": True, "novel": True,
                      "blind_solve": True, "answerable": True}
    assert G.bank_stats(db)["rejections_by_check"] == {"code_exact": 1}


def test_gate_rejects_when_the_second_model_disagrees_or_the_quote_is_invented(world, monkeypatch):
    db, _ = world
    concept = {"id": "total-probability", "name": "Total probability theorem", "definition": "d", "misconceptions": []}
    invented = dict(numeric_question(0.29, GOOD_CODE), source_quote="this sentence does not appear anywhere in the slide")
    script(monkeypatch, [numeric_question(0.29, GOOD_CODE), dict(SOLVED, numeric_answer=0.41), invented])
    assert G.make_question(db, concept, "numeric", 2) is None
    failed = [json.loads(r["verification"])["checks"] for r in db.q("SELECT verification FROM questions ORDER BY id")]
    assert failed[0]["blind_solve"] is False and failed[1]["quote_lock"] is False


def test_quiz_serves_no_question_twice_and_reports_weak_concepts(world, monkeypatch):
    db, _ = world
    concept = {"id": "total-probability", "name": "Total probability theorem", "definition": "d", "misconceptions": []}
    script(monkeypatch, [numeric_question(0.29, GOOD_CODE), SOLVED])
    banked = G.make_question(db, concept, "numeric", 2)
    monkeypatch.setattr(llm, "available", lambda role="generator": False)      # bank only from here on
    db.x("UPDATE students SET level='intermediate' WHERE id='s1'")              # a prior with room to fall

    quiz = G.create_assessment(db, "s1", "quiz", ["total-probability"], 3, ["numeric"])
    assert [q["question_id"] for q in quiz["questions"]] == [banked]            # one question exists, so one is served
    assert "answer" not in quiz["questions"][0] and "solution" not in quiz["questions"][0]

    fb = R.submit_answer(db, quiz["id"], 0, "0.5")
    assert fb["score"] == 0 and fb["correct_answer"] == "0.29"
    assert fb["citation"]["unit"]["label"] == "Slide 2 of Lecture 2 slides"
    touched = {u["concept_id"]: u for u in fb["mastery_updates"]}
    assert touched["total-probability"]["after"] < touched["total-probability"]["before"]
    assert not touched["conditional-probability"]["direct"]                     # the prerequisite shares the blame

    report = R.build_report(db, quiz["id"])
    assert report["percent"] == 0 and report["weak_topics"] == ["Total probability theorem"]
    assert report["next_steps"][0]["study"]["label"].startswith("Slide")

    again = G.create_assessment(db, "s1", "quiz", ["total-probability"], 3, ["numeric"])
    assert again["questions"] == []                                             # already seen: never repeated
