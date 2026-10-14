"""Generation-quality evaluation of the tutor with RAGAS, plus grounding-specific checks.

Runs the production chat pipeline (retrieve -> gate -> generate -> quote-lock) on
eval/testset.json and reports:

  RAGAS, on answerable questions   faithfulness, answer relevancy, context precision, context recall
  grounding, on all questions      refusal correctness, quote-lock pass rate, citation location accuracy

The RAGAS judge is the *verifier* model, which should be a different model from the generator
that wrote the answers (set PRAMANA_VERIFIER). Needs API keys.

    python eval/run_ragas.py [--limit N]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
warnings.filterwarnings("ignore")

from langchain_core.outputs import Generation, LLMResult  # noqa: E402
from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: E402
from ragas.embeddings import BaseRagasEmbeddings  # noqa: E402
from ragas.llms.base import BaseRagasLLM  # noqa: E402
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy  # noqa: E402

from pramana import grounding, llm  # noqa: E402
from pramana.db import DB  # noqa: E402
from pramana.embed import HybridIndex, embed_passages, embed_query  # noqa: E402

EVAL_STUDENT = "ragas-eval"


class JudgeLLM(BaseRagasLLM):
    """RAGAS judge backed by Pramana's verifier role.

    Goes through pramana.llm rather than a LangChain chat model so that no sampling
    parameters are sent (current Claude models reject non-default temperature).
    """

    def generate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None) -> LLMResult:
        out = [Generation(text=llm.text("verifier", "", prompt.to_string())) for _ in range(n)]
        return LLMResult(generations=[out])

    async def agenerate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None) -> LLMResult:
        return await asyncio.to_thread(self.generate_text, prompt, n, temperature, stop, callbacks)

    def is_finished(self, response: LLMResult) -> bool:
        return True


class LocalEmbeddings(BaseRagasEmbeddings):
    """The same local embedding model the retriever uses."""

    def embed_query(self, text: str) -> list[float]:
        return embed_query(text).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_passages(texts).tolist()

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)


def ask_tutor(db: DB, index: HybridIndex, question: str) -> dict:
    """One independent turn: clear history so questions do not influence each other."""
    db.x("DELETE FROM chats WHERE student_id=?", (EVAL_STUDENT,))
    db.set_kv(f"check:{EVAL_STUDENT}", None)
    return grounding.answer(db, index, EVAL_STUDENT, question)


def in_gold(unit: dict, gold: list[dict]) -> bool:
    return any(unit["source_id"] == g["source"] and unit["loc_end"] >= g["from"] and unit["loc_start"] <= g["to"] for g in gold)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N answerable questions")
    args = ap.parse_args()
    for role in ("generator", "verifier"):
        if not llm.available(role):
            raise SystemExit(f"No API key for the {role} model ({llm.spec_for(role)}). Add it to .env.")
    if llm.spec_for("generator") == llm.spec_for("verifier"):
        print("WARNING: generator and verifier are the same model, so the judge is grading its own answers.")

    db = DB()
    index = HybridIndex(db)
    if not db.one("SELECT 1 FROM students WHERE id=?", (EVAL_STUDENT,)):
        db.x("INSERT INTO students(id, name, level, language, created_at) VALUES(?,?,?,?,0)",
             (EVAL_STUDENT, "RAGAS evaluation", "intermediate", "en"))
    tests = json.loads((ROOT / "eval" / "testset.json").read_text(encoding="utf-8"))
    answerable = tests["in_scope"][: args.limit] if args.limit else tests["in_scope"]

    samples, rows = [], []
    claimed = verified = located = 0
    seg_counts = {"source": 0, "outside": 0, "unverified": 0}
    false_refusals = []
    for q in answerable:
        res = ask_tutor(db, index, q["question"])
        if res["status"] != "answered":
            false_refusals.append(q["id"])
            continue
        contexts = [db.one("SELECT text FROM units WHERE id=?", (uid,))["text"] for uid in res["retrieval"]["unit_ids"]]
        response = " ".join(s["text"] for s in res["segments"])
        samples.append(SingleTurnSample(user_input=q["question"], response=response,
                                        retrieved_contexts=contexts, reference=q["reference"]))
        claimed += res["citation_check"]["claimed"]
        verified += res["citation_check"]["verified"]
        for s in res["segments"]:
            seg_counts[s["lane"]] += 1
            located += sum(in_gold(c["unit"], q["gold"]) for c in s["citations"])
        rows.append({"id": q["id"], "coverage": res["coverage"]})

    def declined(group: str) -> list[bool]:
        out = []
        for q in tests[group]:
            res = ask_tutor(db, index, q["question"])
            out.append(res["status"] == "declined" or res["coverage"] == "none")
        return out

    off, unc = declined("off_topic"), declined("uncovered")

    result = evaluate(
        EvaluationDataset(samples=samples),
        metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()],
        llm=JudgeLLM(), embeddings=LocalEmbeddings(), raise_exceptions=False)
    frame = result.to_pandas()
    names = {"faithfulness": "faithfulness", "answer_relevancy": "answer_relevancy",
             "llm_context_precision_with_reference": "context_precision", "context_recall": "context_recall"}
    metrics = {}
    for col, name in names.items():
        vals = [v for v in frame[col].tolist() if v == v]      # drop NaN from judge failures
        metrics[name] = round(statistics.fmean(vals), 3) if vals else None
        metrics[f"{name}_n"] = len(vals)

    n_seg = sum(seg_counts.values()) or 1
    n_bad = len(off) + len(unc)
    wrongly_declined = len(false_refusals)
    rightly_declined = sum(off) + sum(unc)
    out = {
        "description": (f"{len(samples)} answerable questions answered by the production pipeline. "
                        f"Generator {llm.spec_for('generator')}, judge {llm.spec_for('verifier')}, RAGAS {__import__('ragas').__version__}."),
        "metrics": metrics,
        "grounding": {
            "quotes_claimed": claimed,
            "quote_lock_pass_rate": round(verified / claimed, 3) if claimed else None,
            "verified_citations_at_gold_location": round(located / verified, 3) if verified else None,
            "segments_source_backed": round(seg_counts["source"] / n_seg, 3),
            "segments_outside_knowledge": round(seg_counts["outside"] / n_seg, 3),
            "segments_unverified": round(seg_counts["unverified"] / n_seg, 3),
            "answerable_wrongly_declined": round(wrongly_declined / len(answerable), 3),
            "off_topic_declined": round(sum(off) / len(off), 3),
            "uncovered_in_domain_declined": round(sum(unc) / len(unc), 3),
            "refusal_precision": round(rightly_declined / (rightly_declined + wrongly_declined), 3)
            if rightly_declined + wrongly_declined else None,
            "refusal_recall": round(rightly_declined / n_bad, 3),
        },
        "wrongly_declined_ids": false_refusals,
        "caveats": [
            "RAGAS scores come from an LLM judge and carry its errors; spot-check eval/results/ragas_rows.json by hand.",
            "Gold locations list verified places only, so 'citations at gold location' is a lower bound.",
            f"{len(answerable)} answerable, {len(off)} off-topic and {len(unc)} uncovered questions: small samples, wide intervals.",
        ],
    }
    results = ROOT / "eval" / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "ragas.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    frame.to_json(results / "ragas_rows.json", orient="records", indent=1)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
