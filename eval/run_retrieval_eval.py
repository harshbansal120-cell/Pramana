"""Retrieval and refusal-gate evaluation. No LLM calls, so it is cheap and exactly reproducible.

Measures, on eval/testset.json:
  - whether retrieved units land on the known source locations (hit@k, MRR, source recall),
    for hybrid, dense-only and BM25-only retrieval;
  - how well the similarity gate separates answerable questions from off-topic ones and from
    in-domain questions the material does not cover. The threshold is fitted on half of the
    questions and scored on the other half.

    python eval/run_retrieval_eval.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from pramana.db import DB  # noqa: E402
from pramana.embed import HybridIndex  # noqa: E402
from pramana.grounding import coverage_gate  # noqa: E402

K = 8


def in_gold(unit: dict, gold: list[dict]) -> bool:
    return any(unit["source_id"] == g["source"] and unit["loc_end"] >= g["from"] and unit["loc_start"] <= g["to"] for g in gold)


def main() -> None:
    db = DB()
    index = HybridIndex(db)
    units = {u["id"]: u for u in db.q("SELECT id, source_id, kind, loc_start, loc_end FROM units")}
    tests = json.loads((ROOT / "eval" / "testset.json").read_text(encoding="utf-8"))
    has_figures = any(u["kind"] == "figure" for u in units.values())

    ablation, per_question = [], {}
    for mode in ("hybrid", "dense", "bm25"):
        rows = []
        for q in tests["in_scope"]:
            hits = index.search(q["question"], k=K, mode=mode)
            flags = [in_gold(units[h.unit_id], q["gold"]) for h in hits]
            rank = flags.index(True) + 1 if any(flags) else None
            found_sources = {units[h.unit_id]["source_id"] for h, f in zip(hits, flags) if f}
            rows.append({"id": q["id"], "modality": q["modality"], "rank": rank,
                         "precision": sum(flags) / K,
                         "source_recall": len(found_sources) / len({g["source"] for g in q["gold"]})})
        if mode == "hybrid":
            per_question = {r["id"]: r for r in rows}

        def agg(sel):
            sel = list(sel)
            return {
                "n": len(sel),
                "hit@1": round(statistics.fmean(r["rank"] == 1 for r in sel), 3),
                "hit@3": round(statistics.fmean(bool(r["rank"]) and r["rank"] <= 3 for r in sel), 3),
                f"hit@{K}": round(statistics.fmean(bool(r["rank"]) for r in sel), 3),
                "mrr": round(statistics.fmean(1 / r["rank"] if r["rank"] else 0 for r in sel), 3),
                "gold_source_recall": round(statistics.fmean(r["source_recall"] for r in sel), 3),
                "location_precision_lower_bound": round(statistics.fmean(r["precision"] for r in sel), 3),
            }

        ablation.append({"retriever": mode, **agg(rows)})
        if mode == "hybrid":
            by_modality = [{"gold_modality": m, **agg(r for r in rows if r["modality"] == m)}
                           for m in ("slides", "textbook", "video", "figure")]

    # ---- refusal gate: top dense similarity vs a threshold
    def top(q):
        """(best similarity, has a rare-term match) - the two inputs of the production gate."""
        hits = index.search(q["question"], k=K)
        return max((h.dense for h in hits), default=0.0), bool(index.rare_terms(q["question"]))

    scored = ([("in_scope", top(q)) for q in tests["in_scope"]] + [("off_topic", top(q)) for q in tests["off_topic"]]
              + [("uncovered", top(q)) for q in tests["uncovered"]])
    assert all(coverage_gate(index, q["question"], index.search(q["question"], k=K), 0.7)["pass"] == (s >= 0.7 or r)
               for q, (_, (s, r)) in zip(tests["in_scope"], scored))      # the eval mirrors the production gate
    # Deterministic split: even positions fit the threshold, odd positions test it.
    fit, held = scored[0::2], scored[1::2]

    def rates(rows, thr, rescue=True):
        g = {k: [s if not (rescue and r) else 1.0 for kind, (s, r) in rows if kind == k]
             for k in ("in_scope", "off_topic", "uncovered")}
        answered = sum(s >= thr for s in g["in_scope"])
        declined_bad = sum(s < thr for s in g["off_topic"]) + sum(s < thr for s in g["uncovered"])
        declined_all = declined_bad + (len(g["in_scope"]) - answered)
        n_bad = len(g["off_topic"]) + len(g["uncovered"])
        return {
            "in_scope_answered": round(answered / len(g["in_scope"]), 3),
            "off_topic_declined": round(sum(s < thr for s in g["off_topic"]) / len(g["off_topic"]), 3),
            "uncovered_in_domain_declined": round(sum(s < thr for s in g["uncovered"]) / len(g["uncovered"]), 3),
            "precision": round(declined_bad / declined_all, 3) if declined_all else None,   # of declines, how many were right
            "recall": round(declined_bad / n_bad, 3),                                        # of unanswerable, how many declined
        }

    # Prefer never refusing an answerable question: the generator is a second refusal layer for what slips through.
    candidates = sorted({round(s, 3) for _, (s, _) in fit})
    ok = [t for t in candidates if rates(fit, t)["in_scope_answered"] >= 0.97] or candidates[:1]
    threshold = max(ok, key=lambda t: (rates(fit, t)["recall"], t))
    dist = {k: {"min": round(min(v), 3), "median": round(statistics.median(v), 3), "max": round(max(v), 3)}
            for k in ("in_scope", "off_topic", "uncovered") for v in [[s for kind, (s, _) in scored if kind == k]]}
    sweep = [{"threshold": t, **rates(scored, t)} for t in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80)]

    caveats = [
        "Gold locations list the places we verified, not every place an answer appears, so location precision is a lower bound.",
        "The similarity gate is only the first refusal layer. In-domain questions the material does not cover score "
        "close to answerable ones and mostly pass it; they are meant to be caught by the generator's coverage judgement, "
        "which eval/run_ragas.py measures.",
        f"Small test set ({len(tests['in_scope'])} answerable, {len(tests['off_topic'])} off-topic, "
        f"{len(tests['uncovered'])} uncovered); the threshold was fitted on half and scored on the other half.",
    ]
    caveats.append("The gate's rare-term rule was designed while looking at this test set, so even the held-out half "
                   "is not a fully independent test of it.")
    if not has_figures:
        caveats.append("Figure units were not built for this run (no vision model key), so figure-targeted questions "
                       "could only be matched through slide text.")
    result = {
        "description": f"{len(tests['in_scope'])} questions with known source locations, top {K} retrieved units, "
                       f"{len(units)} units in the index. Embeddings: {index.mat.shape[1]}-d local model. No LLM involved.",
        "ablation": ablation,
        "by_gold_modality": by_modality,
        "top_similarity": dist,
        "refusal": {"threshold": threshold, "scored_on": "held-out half", **rates(held, threshold)},
        "refusal_all_questions": {"threshold": threshold, **rates(scored, threshold)},
        "refusal_without_rare_term_rescue": {"threshold": threshold, **rates(scored, threshold, rescue=False)},
        "refusal_threshold_sweep": sweep,
        "missed": [qid for qid, r in per_question.items() if not r["rank"]],
        "caveats": caveats,
    }
    out = ROOT / "eval" / "results" / "retrieval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
