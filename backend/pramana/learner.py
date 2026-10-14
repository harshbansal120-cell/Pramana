"""Learner model: prerequisite-aware Bayesian knowledge tracing with forgetting.

Pure Python with no database or LLM dependency, so the same code runs in the product
and in the simulated-student evaluation (eval/simulate.py).

Three ideas, each of which can be switched off for ablation:

1. BKT. Each concept has P(mastered). An answer updates it by Bayes' rule with slip and
   guess probabilities that depend on question type and difficulty.
2. Prerequisite structure.
   a. Attribution (noisy-AND). A question on concept C is answered correctly only if C *and*
      its direct prerequisites are known. A wrong answer is therefore evidence about both: it
      lowers C and also whichever prerequisites are already doubtful. With strong
      prerequisites this reduces exactly to plain BKT.
   b. Consistency. Knowing a concept implies knowing what it is built on, so evidence that
      raises C also lifts every ancestor to at least `consistency` * P(C). One correct answer
      deep in the graph therefore vouches for the whole chain beneath it.
   c. Structural prior. A concept that has never been tested is discounted while its
      prerequisites look unknown.
3. Forgetting. Mastery decays as exp(-dt / stability). Stability grows with each successful
   review (the spacing effect) and shrinks after a lapse, as in spaced-repetition schedulers.

Time `t` is in days throughout.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Params:
    p_init: float = 0.25
    p_transit: float = 0.10                 # chance of learning from one practised question with feedback
    slip: float = 0.10
    guess: dict = field(default_factory=lambda: {"mcq": 0.25, "numeric": 0.03, "short": 0.08})
    difficulty_slip: float = 0.06           # extra slip per difficulty level above 2
    base_stability: float = 10.0            # days; memory time constant after first exposure
    prior_knowledge_boost: float = 3.0      # stability multiplier when the first answer is already correct
    stability_gain: float = 1.8
    stability_loss: float = 0.6
    max_stability: float = 120.0
    target_p: float = 0.70                  # aim questions at ~70% predicted success
    weak: float = 0.5                       # below this a prerequisite counts as a likely root cause
    consistency: float = 0.9                # P(prerequisite known) >= consistency * P(concept known)
    untested_floor: float = 0.5             # prior kept by an untested concept whose prerequisites look unknown


LEVEL_PRIORS = {"beginner": 0.15, "intermediate": 0.40, "advanced": 0.65}


@dataclass
class State:
    p: float
    stability: float
    last_t: float | None = None
    n: int = 0
    n_correct: float = 0.0


@dataclass
class Update:
    concept_id: str
    before: float
    after: float
    direct: bool   # False when the change came through prerequisite attribution


class Graph:
    def __init__(self, concept_ids: list[str], prereqs: dict[str, list[str]], topics: dict[str, str] | None = None):
        self.ids = list(concept_ids)
        known = set(self.ids)
        self.prereqs = {c: [p for p in prereqs.get(c, []) if p in known and p != c] for c in self.ids}
        self.children: dict[str, list[str]] = {c: [] for c in self.ids}
        for c, ps in self.prereqs.items():
            for p in ps:
                self.children[p].append(c)
        self.topics = topics or {}
        self._anc: dict[str, set[str]] = {}
        self._desc: dict[str, set[str]] = {}

    def _closure(self, c: str, edges: dict[str, list[str]], memo: dict[str, set[str]]) -> set[str]:
        if c not in memo:
            memo[c] = set()                 # guards against cycles
            out: set[str] = set()
            for n in edges[c]:
                out.add(n)
                out |= self._closure(n, edges, memo)
            memo[c] = out
        return memo[c]

    def ancestors(self, c: str) -> set[str]:
        return self._closure(c, self.prereqs, self._anc)

    def descendants(self, c: str) -> set[str]:
        return self._closure(c, self.children, self._desc)

    def depth(self, c: str) -> int:
        return 0 if not self.prereqs[c] else 1 + max(self.depth(p) for p in self.prereqs[c])


class LearnerModel:
    def __init__(self, graph: Graph, params: Params | None = None, states: dict[str, State] | None = None,
                 use_prereqs: bool = True, use_forgetting: bool = True, p_init: float | None = None):
        self.g = graph
        self.params = params or Params()
        self.use_prereqs = use_prereqs
        self.use_forgetting = use_forgetting
        p0 = self.params.p_init if p_init is None else p_init
        self.s: dict[str, State] = {c: State(p0, self.params.base_stability) for c in graph.ids}
        if states:
            self.s.update({c: st for c, st in states.items() if c in self.s})

    # ------------------------------------------------------------------ estimates

    def p_now(self, c: str, t: float) -> float:
        st = self.s[c]
        if st.last_t is None:                        # never tested: lean on the prerequisites (2c)
            if not self.use_prereqs or not self.g.prereqs[c]:
                return st.p
            floor = self.params.untested_floor
            return st.p * (floor + (1 - floor) * self.readiness(c, t))
        if not self.use_forgetting:
            return st.p
        return st.p * math.exp(-max(0.0, t - st.last_t) / st.stability)

    def _slip_guess(self, qtype: str, difficulty: int) -> tuple[float, float]:
        P = self.params
        slip = min(0.4, max(0.02, P.slip + P.difficulty_slip * (difficulty - 2)))
        return slip, P.guess.get(qtype, 0.1)

    def readiness(self, c: str, t: float) -> float:
        """Probability that every direct prerequisite is known."""
        r = 1.0
        for p in self.g.prereqs[c]:
            r *= self.p_now(p, t)
        return r

    def predict(self, c: str, qtype: str, difficulty: int, t: float) -> float:
        slip, guess = self._slip_guess(qtype, difficulty)
        able = self.p_now(c, t) * (self.readiness(c, t) if self.use_prereqs else 1.0)
        return able * (1 - slip) + (1 - able) * guess

    # ------------------------------------------------------------------ update

    def observe(self, c: str, score: float, qtype: str = "mcq", difficulty: int = 2, t: float = 0.0,
                weight: float = 1.0, learn: bool = True) -> list[Update]:
        """Fold one piece of evidence into the model.

        score  - 1 correct, 0 wrong, in between for partial credit
        weight - 1 for a quiz answer; lower for softer evidence such as a chat check question
        learn  - apply the learning transition (off for diagnostics, which do not teach)
        """
        slip, guess = self._slip_guess(qtype, difficulty)
        pres = self.g.prereqs[c] if self.use_prereqs else []
        a = self.p_now(c, t)
        q = {p: self.p_now(p, t) for p in pres}
        updates: list[Update] = []

        def posterior(own: float, others: float) -> float:
            """P(this skill known | evidence), where `others` = P(all the other required skills known)."""
            joint = own * others
            p_right = own * (others * (1 - slip) + (1 - others) * guess) / max(1e-9, joint * (1 - slip) + (1 - joint) * guess)
            p_wrong = own * (others * slip + (1 - others) * (1 - guess)) / max(1e-9, joint * slip + (1 - joint) * (1 - guess))
            return score * p_right + (1 - score) * p_wrong

        # Given that C is known, its prerequisites are known with probability >= consistency (2b),
        # so a correct answer is credited to C rather than explained away as a lucky guess.
        b = max(math.prod(q.values()), self.params.consistency) if q else 1.0
        post_c = a + weight * (posterior(a, b) - a)
        for p, qp in q.items():
            rest = a * math.prod(v for k, v in q.items() if k != p)
            post_p = qp + weight * (posterior(qp, rest) - qp)
            st = self.s[p]
            st.p, st.last_t = min(0.999, max(0.001, post_p)), t      # re-base the decayed estimate at t
            updates.append(Update(p, qp, st.p, False))

        st = self.s[c]
        if learn:
            post_c = post_c + (1 - post_c) * self.params.p_transit * weight
        st.p = min(0.999, max(0.001, post_c))
        if st.n == 0 and score >= 0.5 and weight >= 0.5:
            # Right on first sight: this is knowledge the student arrived with, so it is consolidated.
            st.stability = self.params.base_stability * self.params.prior_knowledge_boost
        elif st.last_t is not None or weight >= 0.5:
            if score >= 0.5:
                st.stability = min(self.params.max_stability, st.stability * (1 + (self.params.stability_gain - 1) * weight))
            else:
                st.stability = max(1.0, st.stability * (1 - (1 - self.params.stability_loss) * weight))
        st.last_t = t
        st.n += 1
        st.n_correct += score
        updates.insert(0, Update(c, a, st.p, True))
        if self.use_prereqs and score >= 0.5:
            updates += self._lift_ancestors(c, t)
        return updates

    def _lift_ancestors(self, c: str, t: float) -> list[Update]:
        """Consistency (2b): raise each ancestor to at least `consistency` * its dependant's mastery."""
        out: list[Update] = []
        frontier = [c]
        while frontier:                 # a lifted concept is revisited so the lift carries down the chain
            child = frontier.pop(0)
            floor = self.params.consistency * self.p_now(child, t)
            for p in self.g.prereqs[child]:
                before = self.p_now(p, t)
                if before < floor - 1e-9:
                    st = self.s[p]
                    st.p, st.last_t = floor, t
                    out.append(Update(p, before, floor, False))
                    frontier.append(p)
        return out

    # ------------------------------------------------------------------ decisions

    def pick_difficulty(self, c: str, qtype: str, t: float) -> int:
        return min((1, 2, 3), key=lambda d: abs(self.predict(c, qtype, d, t) - self.params.target_p))

    def root_cause(self, c: str, t: float, allowed: set[str] | None = None) -> str:
        """Follow the weakest doubtful prerequisite down until the prerequisites are sound."""
        seen = {c}
        while True:
            weak = [p for p in self.g.prereqs[c]
                    if p not in seen and self.p_now(p, t) < self.params.weak and (allowed is None or p in allowed)]
            if not weak:
                return c
            c = min(weak, key=lambda p: self.p_now(p, t))
            seen.add(c)

    def select(self, candidates: list[str], t: float, served: dict[str, int] | None = None,
               mode: str = "practice") -> str:
        """Choose the next concept to ask about.

        practice   - weakest concept the student is ready for, descending to root-cause prerequisites
        diagnostic - concept whose answer is most informative about the whole graph
        """
        served = served or {}
        if mode == "diagnostic":
            def info(c: str) -> float:
                p = min(0.999, max(0.001, self.p_now(c, t)))
                entropy = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
                reach = len(self.g.ancestors(c)) + len(self.g.descendants(c))
                return entropy * (1 + 0.15 * reach) - 0.6 * served.get(c, 0)
            return max(candidates, key=info)

        def need(c: str) -> float:
            p = self.p_now(c, t)
            st = self.s[c]
            due = 0.0
            if st.last_t is not None and st.p > 0.6:            # once learned, review as it fades
                due = st.p - p
            return (1 - p) * (0.35 + 0.65 * self.readiness(c, t)) + 0.5 * due - 0.3 * served.get(c, 0)
        best = max(candidates, key=need)
        return self.root_cause(best, t, set(candidates))

    def weak_concepts(self, t: float, threshold: float = 0.6) -> list[tuple[str, float]]:
        out = [(c, self.p_now(c, t)) for c in self.g.ids if self.p_now(c, t) < threshold]
        return sorted(out, key=lambda x: x[1])

    def diagnose(self, t: float) -> list[dict]:
        """Weak concepts explained by a weaker prerequisite: the 'why' behind a low score."""
        out = []
        for c in self.g.ids:
            if self.s[c].n == 0 or self.p_now(c, t) >= self.params.weak:
                continue
            root = self.root_cause(c, t)
            if root != c:
                out.append({"concept": c, "root_cause": root, "p_concept": self.p_now(c, t), "p_root": self.p_now(root, t)})
        return out


def schedule(model: LearnerModel, t: float, days_to_exam: int, minutes_per_day: int = 45,
             minutes_per_block: int = 15) -> list[dict]:
    """Day-by-day study plan up to the exam.

    Simulates forward: each day takes the concepts with the lowest predicted recall *on exam
    day*, respecting prerequisite order, and assumes a completed block restores the concept
    and increases its stability. Concepts therefore come back just before they would fade.
    """
    import copy

    sim = copy.deepcopy(model)
    days_to_exam = max(1, min(days_to_exam, 60))
    blocks = max(1, minutes_per_day // minutes_per_block)
    exam_t = t + days_to_exam
    plan = []
    for d in range(days_to_exam):
        day_t = t + d
        todo: list[dict] = []
        chosen: set[str] = set()
        for _ in range(blocks):
            def exam_recall(c: str) -> float:
                st = sim.s[c]
                base = st.p if st.last_t is not None else sim.p_now(c, day_t)
                since = (exam_t - st.last_t) if st.last_t is not None else 0.0
                return base * math.exp(-since / st.stability)
            pool = [c for c in sim.g.ids if c not in chosen and exam_recall(c) < 0.85]
            if not pool:
                break
            c = sim.root_cause(min(pool, key=exam_recall), day_t)
            if c in chosen:
                c = min(pool, key=exam_recall)
            chosen.add(c)
            st = sim.s[c]
            kind = "learn" if sim.p_now(c, day_t) < 0.5 else "review"
            todo.append({"concept": c, "kind": kind, "minutes": minutes_per_block,
                         "recall_now": round(sim.p_now(c, day_t), 2)})
            st.p = max(st.p, 0.9) if kind == "review" else max(sim.p_now(c, day_t), 0.75)
            st.stability = min(sim.params.max_stability, st.stability * sim.params.stability_gain)
            st.last_t = day_t
        plan.append({"day": d, "blocks": todo})
    return plan
