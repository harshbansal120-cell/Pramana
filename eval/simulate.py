"""Personalisation evaluation with simulated students.

Each simulated student has a hidden true knowledge state and answers questions according
to it. We run the production learner model (backend/pramana/learner.py, unmodified) against
these students over several sessions and compare:

  policies   - how the next question is chosen: Pramana's selector vs random vs syllabus order
  estimators - how well the model's mastery estimates match the hidden truth, with the
               prerequisite and forgetting components ablated on identical answer logs

Assumptions of the simulated world are listed in `CAVEATS` and written into the results
file. They matter: the world is built on the same pedagogical premise as the model (a skill
is hard to use or learn without its prerequisites), so this measures whether the system
exploits that premise correctly, not whether the premise holds for real students.

    python eval/simulate.py [--students-per-profile 6] [--sessions 8] [--questions 10] [--graph auto|db|fixture]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from pramana.learner import Graph, LearnerModel, Params  # noqa: E402

RESULTS = ROOT / "eval" / "results"
GAP_DAYS = 2.0            # default days between sessions
ITEMS_PER_CELL = 4        # bank size per concept per difficulty
GUESS = 0.25              # four-option multiple choice

CAVEATS = [
    "Students are simulated. These numbers show the system behaves as designed under stated assumptions; "
    "they are not evidence of learning gains in real students.",
    "World assumption: a question is answered correctly (up to slip) only if the concept and all its direct "
    "prerequisites are known, and a concept is learned from practice ten times more slowly while a prerequisite "
    "is missing. The learner model makes the same structural assumption, which favours it over baselines.",
    "Simulated slip, guess, learning and forgetting rates differ per student and are hidden from the model, "
    "but the model's fixed parameters were chosen by us with this kind of world in mind.",
    "Mastery is a binary hidden state per concept; real knowledge is graded.",
    "Pramana's repetition rate of zero is a property of the selector, not a discovery: it never serves a seen "
    "question while an unseen one exists in scope. Here the bank holds 12 questions per concept; the product "
    "generates new ones on demand.",
    "Misconception detection from distractor choice alone was unreliable at this data volume (see recall and "
    "precision columns). 'Observable' is the share of seeded misconceptions whose concept drew two or more wrong "
    "answers, an upper bound on recall.",
]


@dataclass
class Profile:
    name: str
    known: float          # initial P(known) for a concept whose prerequisites are known
    learn: float          # P(learn) from one practised question with feedback, prerequisites in place
    slip: float
    stability: float      # days; true forgetting time constant of newly learned material before any review
    holes: float = 0.0    # fraction of foundational concepts knocked out despite later knowledge


PROFILES = [
    Profile("novice", known=0.10, learn=0.30, slip=0.12, stability=12),
    Profile("average", known=0.45, learn=0.30, slip=0.10, stability=14),
    Profile("gaps in prerequisites", known=0.75, learn=0.30, slip=0.10, stability=14, holes=0.35),
    Profile("forgetful", known=0.55, learn=0.35, slip=0.10, stability=4),
    Profile("strong reviser", known=0.80, learn=0.40, slip=0.06, stability=25),
]


class Student:
    """Hidden ground truth for one simulated learner."""

    def __init__(self, graph: Graph, profile: Profile, seed: int):
        self.g, self.profile, self.rng = graph, profile, random.Random(seed)
        self.known: dict[str, bool] = {}
        order = sorted(graph.ids, key=graph.depth)
        for c in order:
            ready = all(self.known[p] for p in graph.prereqs[c])
            self.known[c] = self.rng.random() < (profile.known if ready else profile.known * 0.3)
        if profile.holes:      # knows later material by rote but is missing foundations
            for c in order:
                if graph.children[c] and self.rng.random() < profile.holes:
                    self.known[c] = False
        # Knowledge a student arrives with is consolidated (4x); newly learned material is fragile.
        self.stability = {c: profile.stability * self.rng.uniform(0.7, 1.3) * (4 if self.known[c] else 1)
                          for c in graph.ids}
        # Two seeded misconceptions on concepts the student does not know: (concept, tag index).
        unknown = [c for c in graph.ids if not self.known[c]] or graph.ids
        self.misconceptions = {c: self.rng.randrange(2) for c in self.rng.sample(unknown, min(2, len(unknown)))}
        self.streak = {c: 0 for c in graph.ids}

    def able(self, c: str) -> bool:
        return self.known[c] and all(self.known[p] for p in self.g.prereqs[c])

    def answer(self, c: str, difficulty: int) -> tuple[bool, int | None]:
        """Returns (correct, misconception tag chosen or None)."""
        slip = min(0.4, self.profile.slip + 0.06 * (difficulty - 2))
        correct = self.rng.random() < ((1 - slip) if self.able(c) else GUESS)
        tag = None
        if not correct:
            if c in self.misconceptions and self.rng.random() < 0.75:
                tag = self.misconceptions[c]
            else:                                   # three distractors: two carry tags, one does not
                tag = self.rng.choice([0, 1, None])
        return correct, tag

    def practise(self, c: str, correct: bool) -> None:
        """Feedback was shown: maybe learn, and consolidate if already known."""
        ready = all(self.known[p] for p in self.g.prereqs[c])
        if not self.known[c]:
            if self.rng.random() < self.profile.learn * (1.0 if ready else 0.1):
                self.known[c] = True
        else:
            self.stability[c] = min(120, self.stability[c] * 1.6)
        self.streak[c] = self.streak[c] + 1 if correct else 0
        if self.streak[c] >= 2:
            self.misconceptions.pop(c, None)

    def forget(self, days: float) -> None:
        for c in self.g.ids:
            if self.known[c] and self.rng.random() > math.exp(-days / self.stability[c]):
                self.known[c] = False

    def mastery(self) -> float:
        return sum(self.known.values()) / len(self.known)


def load_graph(which: str) -> tuple[Graph, str]:
    if which in ("auto", "db"):
        try:
            from pramana.db import DB
            from pramana.service import load_graph as load_db_graph

            graph, _ = load_db_graph(DB())
            if graph.ids:
                return graph, "concept graph extracted from the ingested course"
        except Exception as e:
            if which == "db":
                raise SystemExit(f"cannot load graph from the database: {e}")
        if which == "db":
            raise SystemExit("the database has no concept graph yet")
    data = json.loads((ROOT / "eval" / "fixtures" / "probability_graph.json").read_text(encoding="utf-8"))
    ids = [c["id"] for c in data["concepts"]]
    return Graph(ids, {c["id"]: c["prereqs"] for c in data["concepts"]}), "fixture graph (hand-written, 6.041SC lectures 1-8)"


def rmse(model: LearnerModel, student: Student, t: float) -> float:
    return math.sqrt(statistics.fmean((model.p_now(c, t) - student.known[c]) ** 2 for c in model.g.ids))


def auc(model: LearnerModel, student: Student, t: float) -> float | None:
    pos = [model.p_now(c, t) for c in model.g.ids if student.known[c]]
    neg = [model.p_now(c, t) for c in model.g.ids if not student.known[c]]
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


# Pramana's selector with parts of the learner model switched off, to see what each part buys.
POLICY_FLAGS = {
    "pramana": {},
    "pramana, no prerequisite structure": {"use_prereqs": False},
    "pramana, no forgetting": {"use_forgetting": False},
}


def run_student(graph: Graph, profile: Profile, seed: int, policy: str, sessions: int, questions: int,
                gap: float = GAP_DAYS) -> dict:
    student = Student(graph, profile, seed)
    rng = random.Random(seed * 7919 + sum(map(ord, policy)))
    adaptive = policy in POLICY_FLAGS
    model = LearnerModel(graph, Params(), **POLICY_FLAGS.get(policy, {}))
    seen: set[tuple] = set()
    log: list[tuple] = []                  # (concept, score, difficulty, t, learn) for estimator replay
    truth: list[dict[str, bool]] = []      # hidden state at the start of each session, and at the end
    tag_counts: dict[tuple, int] = {}
    wrong_counts: dict[str, int] = {}
    seeded = dict(student.misconceptions)
    out = {"mastery": [], "rmse": [], "auc": [], "served": 0, "repeats": 0, "in_zone": [0] * sessions}
    syllabus_i = 0
    t = 0.0
    for s in range(sessions):
        out["mastery"].append(student.mastery())
        out["rmse"].append(rmse(model, student, t))
        out["auc"].append(auc(model, student, t))
        truth.append(dict(student.known))
        served: dict[str, int] = {}
        diagnostic = adaptive and s == 0                 # cold start: adaptive diagnostic, no feedback
        for _ in range(questions):
            if adaptive:
                # As in the product (assess/generate.py next_slot): a concept with no unseen question left is
                # skipped while any other concept still has one.
                open_ids = [x for x in graph.ids if sum((x, dd, k) in seen for dd in (1, 2, 3)
                                                        for k in range(ITEMS_PER_CELL)) < 3 * ITEMS_PER_CELL]
                c = model.select(open_ids or graph.ids, t, served, "diagnostic" if diagnostic else "practice")
                d = 2 if diagnostic else model.pick_difficulty(c, "mcq", t)
            elif policy == "random":
                c, d = rng.choice(graph.ids), rng.choice((1, 2, 3))
            else:                                        # syllabus order, round and round
                c, d = graph.ids[syllabus_i % len(graph.ids)], 2
                syllabus_i += 1
            served[c] = served.get(c, 0) + 1
            if policy == "random":                       # a static quiz generator has no memory of what it served
                item = (c, d, rng.randrange(ITEMS_PER_CELL))
            else:
                # As in the product: an unseen question on this concept, nearest difficulty first. The product
                # would then generate a new question; this finite bank has to repeat one instead.
                fresh = sorted(((abs(dd - d), dd, k) for dd in (1, 2, 3) for k in range(ITEMS_PER_CELL)
                                if (c, dd, k) not in seen))
                item = (c, fresh[0][1], fresh[0][2]) if fresh else (c, d, rng.randrange(ITEMS_PER_CELL))
                d = item[1]
            out["served"] += 1
            out["repeats"] += item in seen
            seen.add(item)
            # "In the zone": the concept is not yet known but its prerequisites are, so practice can teach it.
            out["in_zone"][s] += (not student.known[c]) and all(student.known[p] for p in graph.prereqs[c])
            correct, tag = student.answer(c, d)
            if not correct:
                wrong_counts[c] = wrong_counts.get(c, 0) + 1
            if tag is not None:
                tag_counts[(c, tag)] = tag_counts.get((c, tag), 0) + 1
            model.observe(c, float(correct), "mcq", d, t, learn=not diagnostic)
            log.append((c, float(correct), d, t, not diagnostic))
            if not diagnostic:
                student.practise(c, correct)
        t += gap
        student.forget(gap)
    out["mastery"].append(student.mastery())
    out["rmse"].append(rmse(model, student, t))
    out["auc"].append(auc(model, student, t))
    truth.append(dict(student.known))
    # Flag a misconception when its distractor was chosen at least twice and in most wrong answers on that concept.
    detected = {k for k, n in tag_counts.items() if n >= 2 and n / wrong_counts[k[0]] > 0.5}
    out["misc_observable"] = statistics.fmean(wrong_counts.get(c, 0) >= 2 for c in seeded) if seeded else None
    seeded_set = set(seeded.items())
    out["misc_recall"] = len(detected & seeded_set) / len(seeded_set) if seeded_set else None
    out["misc_precision"] = len(detected & seeded_set) / len(detected) if detected else None
    out["log"], out["truth"], out["questions"], out["gap"] = log, truth, questions, gap
    return out


def replay(graph: Graph, run: dict, **flags) -> list[float]:
    """Score an estimator variant on a recorded answer log against the recorded truth."""
    model = LearnerModel(graph, Params(), **flags)
    errors, q = [], run["questions"]
    for s, truth in enumerate(run["truth"]):
        t = s * run["gap"]
        errors.append(math.sqrt(statistics.fmean((model.p_now(c, t) - truth[c]) ** 2 for c in graph.ids)))
        for c, score, d, tt, learn in run["log"][s * q:(s + 1) * q]:
            model.observe(c, score, "mcq", d, tt, learn=learn)
    return errors


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def boot_ci(diffs: list[float], rng: random.Random, n: int = 4000) -> tuple[float, float]:
    means = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def evaluate(graph: Graph, graph_name: str, per_profile: int, sessions: int, questions: int, gap: float) -> dict:
    policies = list(POLICY_FLAGS) + ["random", "syllabus"]
    runs = {p: [] for p in policies}
    for pi, profile in enumerate(PROFILES):
        for k in range(per_profile):
            seed = 1000 * (pi + 1) + k
            for p in policies:          # same seed: every policy meets the identical student
                r = run_student(graph, profile, seed, p, sessions, questions, gap)
                r["profile"] = profile.name
                runs[p].append(r)
    n = len(runs["pramana"])
    rng = random.Random(0)

    def gain(r: dict) -> float:
        return r["mastery"][-1] - r["mastery"][0]

    rows = []
    for p in policies:
        diffs = [gain(a) - gain(b) for a, b in zip(runs["pramana"], runs[p])]
        lo, hi = boot_ci(diffs, rng) if p != "pramana" else (0.0, 0.0)
        served = sum(r["served"] for r in runs[p])
        rows.append({
            "policy": p,
            "mastery_start": round(mean(r["mastery"][0] for r in runs[p]), 3),
            "mastery_end": round(mean(r["mastery"][-1] for r in runs[p]), 3),
            "mastery_gain": round(mean(gain(r) for r in runs[p]), 3),
            "pramana_minus_this_gain_95ci": "-" if p == "pramana" else f"{statistics.fmean(diffs):+.3f} [{lo:+.3f}, {hi:+.3f}]",
            "question_repetition_rate": round(sum(r["repeats"] for r in runs[p]) / served, 3),
            "in_zone_rate": round(sum(sum(r["in_zone"]) for r in runs[p]) / served, 3),
            "final_estimate_rmse": round(mean(r["rmse"][-1] for r in runs[p]), 3),
            "final_estimate_auc": round(mean(r["auc"][-1] for r in runs[p]), 3),
            "misconception_recall": round(mean(r["misc_recall"] for r in runs[p]), 2),
            "misconception_observable": round(mean(r["misc_observable"] for r in runs[p]), 2),
            "misconception_precision": round(mean(r["misc_precision"] for r in runs[p]) or 0, 2),
        })

    by_profile = [dict(profile=profile.name, **{p: round(mean(gain(r) for r in runs[p] if r["profile"] == profile.name), 3)
                                                for p in policies}) for profile in PROFILES]

    variants = {
        "full model": {},
        "no prerequisite structure": {"use_prereqs": False},
        "no forgetting": {"use_forgetting": False},
        "plain BKT": {"use_prereqs": False, "use_forgetting": False},
    }
    curves = {name: [mean(col) for col in zip(*(replay(graph, r, **flags) for r in runs["pramana"]))]
              for name, flags in variants.items()}
    ablation = [{"estimator": name, "rmse_before_any_data": round(c[0], 3), "rmse_after_2_sessions": round(c[2], 3),
                 "rmse_final": round(c[-1], 3), "mean_rmse_over_sessions": round(statistics.fmean(c[1:]), 3)}
                for name, c in curves.items()]
    shown = ["pramana", "random", "syllabus"]
    return {
        "description": (
            f"{n} simulated students ({len(PROFILES)} profiles x {per_profile} seeds), "
            f"{sessions} sessions of {questions} questions, {gap:g} days apart, on the {graph_name} "
            f"({len(graph.ids)} concepts). Every policy meets the identical students. Pramana spends session 0 on a "
            f"no-feedback diagnostic, so it gets one fewer teaching session than the baselines."),
        "graph": graph_name,
        "n_students": n,
        "gap_days": gap,
        "true_mastery_by_session": {p: [round(mean(r["mastery"][s] for r in runs[p]), 4) for s in range(sessions + 1)]
                                    for p in shown},
        "estimator_rmse_by_session": {k: [round(v, 4) for v in c] for k, c in curves.items()},
        "estimator_auc_by_session": [round(mean(r["auc"][s] for r in runs["pramana"]), 3) for s in range(sessions + 1)],
        "in_zone_rate_by_session": {p: [round(sum(r["in_zone"][s] for r in runs[p]) / (n * questions), 3)
                                        for s in range(sessions)] for p in shown},
        "policies": rows,
        "mastery_gain_by_profile": by_profile,
        "estimator_ablation": ablation,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--students-per-profile", type=int, default=6)
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--questions", type=int, default=10)
    ap.add_argument("--graph", default="auto", choices=["auto", "db", "fixture"])
    args = ap.parse_args()
    graph, graph_name = load_graph(args.graph)
    common = (graph, graph_name, args.students_per_profile, args.sessions, args.questions)
    result = evaluate(*common, gap=GAP_DAYS)
    spaced = evaluate(*common, gap=7.0)        # the same study with weekly sessions, where forgetting matters more
    result["weekly_sessions"] = {k: spaced[k] for k in ("description", "policies", "estimator_ablation")}
    result["caveats"] = CAVEATS
    curves = result["estimator_rmse_by_session"]
    policies = list(result["true_mastery_by_session"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "simulation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for section in (result, result["weekly_sessions"]):
        print("\n" + section["description"])
        for row in section["policies"] + section["estimator_ablation"]:
            print("  ", row)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for p in policies:
            axes[0].plot(result["true_mastery_by_session"][p], marker="o", label=p)
        axes[0].set(title="True mastery by session", xlabel="session", ylabel="fraction of concepts known")
        axes[0].legend()
        for name, c in curves.items():
            axes[1].plot(c, marker="o", label=name)
        axes[1].set(title="Mastery estimate error (Pramana policy logs)", xlabel="session", ylabel="RMSE vs hidden truth")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(RESULTS / "simulation.png", dpi=140)
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
