"""Learner model behaviour: each test pins one property the product relies on."""
import math

from pramana.learner import Graph, LearnerModel, Params, schedule

#  a -> b -> c   (b requires a, c requires b), d stands alone
G = Graph(["a", "b", "c", "d"], {"b": ["a"], "c": ["b"]})


def test_correct_answer_raises_and_wrong_answer_lowers():
    m = LearnerModel(G)
    up = m.observe("d", 1.0, "numeric", 2, t=0)[0]
    assert up.after > up.before
    m2 = LearnerModel(G, p_init=0.6)
    down = m2.observe("d", 0.0, "numeric", 2, t=0, learn=False)[0]
    assert down.after < down.before


def test_guessable_question_is_weaker_evidence():
    mcq, numeric = LearnerModel(G), LearnerModel(G)
    mcq.observe("d", 1.0, "mcq", 2, t=0, learn=False)
    numeric.observe("d", 1.0, "numeric", 2, t=0, learn=False)
    assert numeric.s["d"].p > mcq.s["d"].p


def test_wrong_answer_is_shared_with_a_doubtful_prerequisite():
    m = LearnerModel(G, p_init=0.5)
    before = m.p_now("a", 0)
    updates = m.observe("b", 0.0, "numeric", 2, t=0, learn=False)
    touched = {u.concept_id: u for u in updates}
    assert "a" in touched and not touched["a"].direct
    assert m.p_now("a", 0) < before


def test_with_solid_prerequisites_it_reduces_to_plain_bkt():
    structured = LearnerModel(G, p_init=0.5)
    structured.s["a"].p, structured.s["a"].last_t = 0.999, 0.0
    plain = LearnerModel(G, p_init=0.5, use_prereqs=False)
    structured.s["b"].last_t = plain.s["b"].last_t = 0.0      # both already tested, so no structural prior
    structured.observe("b", 0.0, "numeric", 2, t=0, learn=False)
    plain.observe("b", 0.0, "numeric", 2, t=0, learn=False)
    assert math.isclose(structured.s["b"].p, plain.s["b"].p, abs_tol=0.01)


def test_correct_answer_vouches_for_the_whole_prerequisite_chain():
    m = LearnerModel(G, p_init=0.2)
    m.observe("c", 1.0, "numeric", 2, t=0)
    k = m.params.consistency
    assert m.p_now("b", 0) >= k * m.p_now("c", 0) - 1e-9
    assert m.p_now("a", 0) >= k * k * m.p_now("c", 0) - 1e-9


def test_mastery_fades_and_review_slows_the_fading():
    m = LearnerModel(G)
    m.observe("d", 1.0, "numeric", 2, t=0)
    assert m.p_now("d", 30) < m.p_now("d", 0)
    stability_once = m.s["d"].stability
    m.observe("d", 1.0, "numeric", 2, t=5)
    assert m.s["d"].stability > stability_once
    off = LearnerModel(G, use_forgetting=False)
    off.observe("d", 1.0, "numeric", 2, t=0)
    assert off.p_now("d", 30) == off.p_now("d", 0)


def test_untested_concept_is_discounted_when_prerequisites_look_unknown():
    m = LearnerModel(G, p_init=0.5)
    base = m.p_now("b", 0)
    m.s["a"].p, m.s["a"].last_t = 0.05, 0.0
    assert m.p_now("b", 0) < base


def test_selection_descends_to_the_weak_prerequisite():
    m = LearnerModel(G, p_init=0.6)
    for c, p in (("a", 0.2), ("b", 0.3), ("c", 0.3), ("d", 0.9)):
        m.s[c].p, m.s[c].last_t = p, 0.0
    assert m.select(["a", "b", "c", "d"], t=0) == "a"
    assert m.root_cause("c", 0) == "a"


def test_difficulty_tracks_mastery():
    m = LearnerModel(G)
    m.s["d"].p = 0.2
    easy = m.pick_difficulty("d", "numeric", 0)
    m.s["d"].p = 0.99
    hard = m.pick_difficulty("d", "numeric", 0)
    assert easy <= hard and hard == 3


def test_schedule_covers_weak_concepts_and_respects_time_budget():
    m = LearnerModel(G, Params(), p_init=0.2)
    plan = schedule(m, t=0, days_to_exam=5, minutes_per_day=30, minutes_per_block=15)
    assert len(plan) == 5 and all(len(d["blocks"]) <= 2 for d in plan)
    order = [b["concept"] for d in plan for b in d["blocks"]]
    assert {"a", "b", "c", "d"} <= set(order)
    assert order.index("a") < order.index("b") < order.index("c")      # foundations before dependants
