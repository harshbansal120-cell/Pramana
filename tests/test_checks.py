"""Quote-lock, answer-checking sandbox, grading and parsing helpers."""
from pramana.assess.sandbox import run_check, screen
from pramana.assess.verify import close, grade, parse_number
from pramana.grounding import locate_quote
from pramana.ingest.parse import chunk_transcript, youtube_id

TEXT = ("Total probability theorem\n• Divide and conquer\n• Partition of sample space into A1, A2, A3\n"
        "• Have P(B | Ai), for every i")


def test_quote_must_come_from_the_text():
    assert locate_quote("Partition of sample space into A1, A2, A3", TEXT)
    assert locate_quote("partition of   sample space\ninto A1 A2 A3", TEXT)            # case, spacing, punctuation
    assert locate_quote("Partition of sample spaces into A1, A2, A3", TEXT)             # near-verbatim
    assert not locate_quote("Bayes' rule revises prior beliefs given new evidence", TEXT)
    assert not locate_quote("Divide", TEXT)                                              # too short to count


def test_sandbox_runs_exact_and_simulated_checks():
    code = (
        "from fractions import Fraction\n"
        "def exact():\n    return Fraction(1, 6) * 3\n"
        "def simulate(rng, n):\n    return sum(rng.randint(1, 6) <= 3 for _ in range(n)) / n\n"
    )
    res = run_check(code)
    assert abs(res["exact"] - 0.5) < 1e-12 and abs(res["simulated"] - 0.5) < 0.01


def test_sandbox_refuses_unsafe_code():
    assert "import not allowed" in screen("import os\ndef exact():\n    return 1")
    assert "name not allowed" in screen("def exact():\n    return open('x').read()")
    assert "attribute not allowed" in screen("def exact():\n    return (1).__class__")
    assert "must define exact" in screen("x = 1")
    assert "error" in run_check("import subprocess\ndef exact():\n    return 1")


def test_numeric_parsing_and_tolerance():
    assert parse_number("3/8") == 0.375
    assert parse_number("37.5%") == 0.375
    assert parse_number(" 0.375 ") == 0.375
    assert parse_number("about 0.375") == 0.375
    assert parse_number("no idea") is None
    assert close(0.3751, 0.375, 0.001) and not close(0.39, 0.375, 0.001)


def test_grading_mcq_reports_the_misconception_behind_the_distractor():
    q = {"type": "mcq", "options": [
        {"text": "1/2", "correct": False, "misconception": "Treats dependent events as independent"},
        {"text": "1/3", "correct": True, "misconception": None}]}
    wrong = grade(q, "0", [])
    assert wrong["score"] == 0 and wrong["misconception"] == "Treats dependent events as independent"
    assert grade(q, "1", [])["score"] == 1
    assert grade(q, "", [])["score"] == 0


def test_grading_numeric_accepts_equivalent_forms():
    q = {"type": "numeric", "numeric_answer": 0.375, "tolerance": 0.001}
    assert grade(q, "3/8", [])["score"] == 1
    assert grade(q, "37.5%", [])["score"] == 1
    assert grade(q, "0.5", [])["score"] == 0


def test_transcript_windows_keep_timestamps():
    snippets = [(i * 5.0, 5.0, f"sentence {i}.") for i in range(40)]
    units = chunk_transcript(snippets, window=55)
    assert units[0].loc_start == 0 and units[-1].loc_end == 200
    assert all(u.loc_type == "time" and u.loc_end > u.loc_start for u in units)
    assert all(a.loc_end <= b.loc_start + 1e-6 for a, b in zip(units, units[1:]))


def test_youtube_id_forms():
    for url in ("https://www.youtube.com/watch?v=j9WZyLZCBzs", "https://youtu.be/j9WZyLZCBzs?t=10", "j9WZyLZCBzs"):
        assert youtube_id(url) == "j9WZyLZCBzs"
    assert youtube_id("https://example.com/video") is None
