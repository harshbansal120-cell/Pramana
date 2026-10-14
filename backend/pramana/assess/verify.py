"""The verification gate, and grading.

A generated question enters the bank only if every applicable check passes:

  structure    - well-formed for its type (one correct option, four distinct options, ...)
  quote_lock   - its source quote really appears in the cited unit
  code_exact   - (numerical) executed checking code reproduces the answer key
  code_sim     - (numerical, when a simulation is supplied) Monte Carlo agrees with the key
  blind_solve  - a second model, never shown the key, arrives at the same answer
  answerable   - that model judges the question answerable from the cited material and unambiguous
  novel        - not a near-duplicate of a question already in the bank
"""
from __future__ import annotations

import math
import re
from fractions import Fraction

import numpy as np

from .. import llm
from ..db import DB, from_blob
from ..grounding import locate_quote
from ..llm import BOOL, NNUM, NSTR, STR, enum, nullable, obj
from .sandbox import run_check

DUPLICATE_SIM = 0.92

_SOLVE_SYSTEM = (
    "You are checking an exam question written by someone else. Solve it yourself, carefully and independently. "
    "You are not told the intended answer."
)
_SOLVE_USER = """COURSE EXCERPT the question is based on:
{excerpt}

QUESTION ({qtype}):
{stem}
{options}

Solve the question. Then judge it:
- chosen_option: for multiple choice, the letter of the single best option, else null.
- numeric_answer: for a numerical question, your final number as a decimal, else null.
- short_answer: for a short-answer question, your answer in one or two sentences, else null.
- answerable: true if the excerpt, plus the course's standard definitions, is enough to answer it.
- ambiguous: true if more than one answer is defensible, an option other than yours is also correct,
  or the wording is unclear.
- issue: a short note if anything is wrong with the question, else null."""
_SOLVE_SCHEMA = obj(chosen_option=nullable(enum("A", "B", "C", "D")), numeric_answer=NNUM, short_answer=NSTR,
                    answerable=BOOL, ambiguous=BOOL, issue=NSTR)

_EQUIV_SYSTEM = "You compare two answers to the same question and decide whether they agree in substance."
_EQUIV_SCHEMA = obj(equivalent=BOOL, note=STR)

_GRADE_SYSTEM = (
    "You grade a student's short answer against a reference answer from their course. Be fair: credit correct "
    "ideas however they are worded, in any language. Do not credit vague or partly wrong answers in full."
)
_GRADE_USER = """QUESTION: {stem}
REFERENCE ANSWER: {answer}
KNOWN MISCONCEPTIONS for this concept: {misconceptions}
STUDENT ANSWER: {response}

score: "correct", "partial" or "incorrect".
misconception: if the answer shows one of the known misconceptions, or another specific wrong belief, state it
in a short sentence. Else null.
feedback: one or two sentences to the student, in {language}."""
_GRADE_SCHEMA = obj(score=enum("correct", "partial", "incorrect"), misconception=NSTR, feedback=STR)


def close(a: float, b: float, tol: float | None) -> bool:
    return abs(a - b) <= max(tol or 0.0, 1e-6, 1e-4 * abs(b))


def parse_number(text: str) -> float | None:
    """Read a student's numeric entry: decimals, fractions like 3/8, percentages."""
    s = text.strip().replace(",", "").replace(" ", "")
    pct = s.endswith("%")
    s = s.rstrip("%")
    try:
        v = float(Fraction(s)) if "/" in s else float(s)
    except (ValueError, ZeroDivisionError):
        m = re.search(r"-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", s)
        if not m:
            return None
        try:
            v = float(Fraction(m.group())) if "/" in m.group() else float(m.group())
        except (ValueError, ZeroDivisionError):
            return None
    return v / 100 if pct else v


def verify(db: DB, q: dict, unit_text: str, stem_emb: np.ndarray) -> dict:
    """Run every applicable check on a generated question. Returns {"checks": {...}, "passed": bool, ...}."""
    checks: dict[str, bool | None] = {}
    notes: dict[str, str] = {}
    qtype = q["type"]

    # structure
    ok = len(q["stem"].strip()) > 15
    if qtype == "mcq":
        texts = [o["text"].strip().lower() for o in q["options"]]
        ok = ok and len(texts) == 4 and len(set(texts)) == 4 and sum(o["correct"] for o in q["options"]) == 1
    elif qtype == "numeric":
        ok = ok and q.get("numeric_answer") is not None and math.isfinite(q["numeric_answer"])
    else:
        ok = ok and len(q.get("answer_text", "").strip()) > 3
    checks["structure"] = ok

    checks["quote_lock"] = locate_quote(q["source_quote"], unit_text)

    # executed answer check
    if q.get("check_code") and q.get("numeric_answer") is not None:
        res = run_check(q["check_code"])
        if "exact" in res:
            checks["code_exact"] = close(res["exact"], q["numeric_answer"], q.get("tolerance"))
            notes["code_exact"] = f"code {res['exact']:.6g} vs key {q['numeric_answer']:.6g}"
        else:
            checks["code_exact"] = False
            notes["code_exact"] = res.get("error") or res.get("exact_error", "no result")
        if "simulated" in res:
            key = q["numeric_answer"]
            # 4 standard errors for a probability; 3% relative for other quantities.
            band = 4 * math.sqrt(max(key * (1 - key), 1e-4) / res["n"]) if 0 <= key <= 1 else 0.03 * max(1.0, abs(key))
            checks["code_sim"] = abs(res["simulated"] - key) <= max(band, q.get("tolerance") or 0)
            notes["code_sim"] = f"simulated {res['simulated']:.5g} vs key {key:.5g} (band {band:.2g})"
    elif qtype == "numeric":
        checks["code_exact"] = False
        notes["code_exact"] = "numerical question without checking code"

    # novelty
    rows = db.q("SELECT id, emb FROM questions WHERE status='verified' AND concept_id=? AND emb IS NOT NULL", (q["concept_id"],))
    sims = [(float(from_blob(r["emb"]) @ stem_emb), r["id"]) for r in rows]
    worst = max(sims, default=(0.0, None))
    checks["novel"] = worst[0] < DUPLICATE_SIM
    if not checks["novel"]:
        notes["novel"] = f"similarity {worst[0]:.2f} to question {worst[1]}"

    # Cross-model checks cost a call; skip them if the cheap checks already failed.
    if all(v is not False for v in checks.values()):
        letters = "ABCD"
        options = "\n".join(f"{letters[i]}. {o['text']}" for i, o in enumerate(q["options"])) if qtype == "mcq" else ""
        solved = llm.ask("verifier", _SOLVE_SYSTEM, _SOLVE_USER.format(
            excerpt=unit_text, qtype=qtype, stem=q["stem"], options=options), _SOLVE_SCHEMA, max_tokens=8000)
        if qtype == "mcq":
            key = letters[[o["correct"] for o in q["options"]].index(True)]
            checks["blind_solve"] = solved["chosen_option"] == key
            notes["blind_solve"] = f"verifier chose {solved['chosen_option']}, key {key}"
        elif qtype == "numeric":
            got = solved["numeric_answer"]
            checks["blind_solve"] = got is not None and close(got, q["numeric_answer"], max(q.get("tolerance") or 0, 0.005))
            notes["blind_solve"] = f"verifier got {got}, key {q['numeric_answer']}"
        else:
            eq = llm.ask("verifier", _EQUIV_SYSTEM,
                         f"QUESTION: {q['stem']}\nANSWER 1: {q['answer_text']}\nANSWER 2: {solved['short_answer']}",
                         _EQUIV_SCHEMA, max_tokens=2000)
            checks["blind_solve"] = eq["equivalent"]
            notes["blind_solve"] = eq["note"][:200]
        checks["answerable"] = solved["answerable"] and not solved["ambiguous"]
        if solved["issue"]:
            notes["answerable"] = solved["issue"][:300]

    return {"checks": checks, "notes": notes, "passed": all(v is not False for v in checks.values())
            and "blind_solve" in checks, "verifier": llm.spec_for("verifier"), "generator": llm.spec_for("generator")}


def grade(q: dict, response: str, misconceptions: list[str], language: str = "English") -> dict:
    """Score a response. Returns {"score": 0..1, "misconception": str|None, "note": str}."""
    qtype = q["type"]
    if not response.strip():
        return {"score": 0.0, "misconception": None, "note": "No answer given."}
    if qtype == "mcq":
        try:
            chosen = q["options"][int(response)]
        except (ValueError, IndexError):
            return {"score": 0.0, "misconception": None, "note": "Invalid option."}
        return {"score": 1.0 if chosen["correct"] else 0.0,
                "misconception": None if chosen["correct"] else chosen.get("misconception"), "note": ""}
    if qtype == "numeric":
        v = parse_number(response)
        if v is None:
            return {"score": 0.0, "misconception": None, "note": "Could not read a number from the answer."}
        ok = close(v, q["numeric_answer"], max(q.get("tolerance") or 0, 0.005 * max(1.0, abs(q["numeric_answer"]))))
        return {"score": 1.0 if ok else 0.0, "misconception": None, "note": ""}
    out = llm.ask("verifier", _GRADE_SYSTEM, _GRADE_USER.format(
        stem=q["stem"], answer=q["answer_text"], misconceptions="; ".join(misconceptions) or "(none listed)",
        response=response, language=language), _GRADE_SCHEMA, max_tokens=2000)
    return {"score": {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}[out["score"]],
            "misconception": out["misconception"], "note": out["feedback"]}
