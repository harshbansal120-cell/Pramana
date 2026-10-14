"""Shared read/write helpers between the API, the tutor and the assessment engine."""
from __future__ import annotations

import json
import time

from .db import DB
from .learner import LEVEL_PRIORS, Graph, LearnerModel, State, Update


def today() -> float:
    """Current time in days, the learner model's unit."""
    return time.time() / 86400.0


def _clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def unit_view(db: DB, unit_id: int, with_links: bool = True) -> dict | None:
    """Everything the UI needs to show a unit and open its exact origin."""
    u = db.one("SELECT u.*, s.title AS stitle, s.kind AS skind, s.local_path, s.meta AS smeta, s.uri "
               "FROM units u JOIN sources s ON s.id=u.source_id WHERE u.id=?", (unit_id,))
    if not u:
        return None
    meta = json.loads(u["meta"])
    if u["loc_type"] == "time":
        label = f"{_clock(u['loc_start'])} in {u['stitle']}"
        yt = json.loads(u["smeta"]).get("youtube")
        open_ = {"type": "youtube", "video": yt, "start": int(u["loc_start"])} if yt else {"type": "text"}
    elif u["loc_type"] == "slide":
        label = f"Slide {int(u['loc_start'])} of {u['stitle']}"
        open_ = {"type": "image", "url": f"/files/{u['image_path']}".replace("\\", "/")}
    else:
        label = f"Page {int(u['loc_start'])} of {u['stitle']}"
        open_ = {"type": "pdf", "url": f"/files/{u['local_path']}".replace("\\", "/"), "page": int(u["loc_start"])}
    view = {
        "id": u["id"], "source_id": u["source_id"], "source_title": u["stitle"], "source_kind": u["skind"],
        "kind": u["kind"], "loc_type": u["loc_type"], "loc_start": u["loc_start"], "loc_end": u["loc_end"],
        "label": label, "text": u["text"], "open": open_, "derived": bool(meta.get("derived")),
        "image_url": f"/files/{u['image_path']}".replace("\\", "/") if u["image_path"] else None,
        "concepts": [r["concept_id"] for r in db.q(
            "SELECT concept_id FROM unit_concepts WHERE unit_id=? ORDER BY score DESC", (unit_id,))],
    }
    if with_links:
        # A figure inherits the cross-modal links of the slide it sits on.
        anchor = meta.get("parent_unit", unit_id)
        linked = db.q("SELECT b_unit AS other, score FROM links WHERE a_unit=? AND kind='shown_at' "
                      "UNION SELECT a_unit AS other, score FROM links WHERE b_unit=? AND kind='shown_at'", (anchor, anchor))
        view["linked"] = [dict(unit_view(db, r["other"], False), link_score=round(r["score"], 3))
                          for r in linked if r["score"] >= 0.45][:3]
    return view


def load_graph(db: DB) -> tuple[Graph, dict[str, dict]]:
    rows = db.q("SELECT id, name, definition, topic, subtopic, order_idx, misconceptions FROM concepts ORDER BY order_idx")
    prereqs: dict[str, list[str]] = {}
    for r in db.q("SELECT * FROM prereqs"):
        prereqs.setdefault(r["concept_id"], []).append(r["prereq_id"])
    for r in rows:
        r["misconceptions"] = json.loads(r["misconceptions"])
    return Graph([r["id"] for r in rows], prereqs, {r["id"]: r["topic"] for r in rows}), {r["id"]: r for r in rows}


def load_model(db: DB, student_id: str, graph: Graph | None = None) -> LearnerModel:
    graph = graph or load_graph(db)[0]
    student = db.one("SELECT * FROM students WHERE id=?", (student_id,)) or {}
    states = {r["concept_id"]: State(r["p"], r["stability"], r["last_t"], r["n"], r["n_correct"])
              for r in db.q("SELECT * FROM mastery WHERE student_id=?", (student_id,))}
    return LearnerModel(graph, states=states, p_init=LEVEL_PRIORS.get(student.get("level"), None))


def save_updates(db: DB, student_id: str, model: LearnerModel, updates: list[Update], kind: str,
                 score: float | None, ref: str = "") -> list[dict]:
    """Persist changed mastery rows and log one event per changed concept."""
    names = {r["id"]: r["name"] for r in db.q("SELECT id, name FROM concepts")}
    ts = time.time()
    out = []
    for u in updates:
        st = model.s[u.concept_id]
        db.x("INSERT OR REPLACE INTO mastery(student_id, concept_id, p, stability, last_t, n, n_correct) VALUES(?,?,?,?,?,?,?)",
             (student_id, u.concept_id, st.p, st.stability, st.last_t, st.n, st.n_correct))
        db.x("INSERT INTO events(student_id, ts, concept_id, kind, correct, p_before, p_after, ref) VALUES(?,?,?,?,?,?,?,?)",
             (student_id, ts, u.concept_id, kind if u.direct else "prerequisite", score if u.direct else None,
              u.before, u.after, ref))
        out.append({"concept_id": u.concept_id, "name": names.get(u.concept_id, u.concept_id),
                    "before": round(u.before, 3), "after": round(u.after, 3), "direct": u.direct})
    return out


def kb_stats(db: DB) -> dict:
    by_kind = {r["kind"]: r["n"] for r in db.q("SELECT kind, COUNT(*) AS n FROM units GROUP BY kind")}
    q = {r["status"]: r["n"] for r in db.q("SELECT status, COUNT(*) AS n FROM questions GROUP BY status")}
    return {
        "sources": db.one("SELECT COUNT(*) AS n FROM sources")["n"],
        "units": by_kind,
        "concepts": db.one("SELECT COUNT(*) AS n FROM concepts")["n"],
        "prerequisite_edges": db.one("SELECT COUNT(*) AS n FROM prereqs")["n"],
        "prerequisite_validation": db.get_kv("prereq_validation"),
        "slide_video_links": db.one("SELECT COUNT(*) AS n FROM links WHERE kind='shown_at'")["n"],
        "questions": q,
    }
