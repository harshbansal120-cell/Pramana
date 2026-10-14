"""Post-parse enrichment: figure understanding, slide<->video alignment, concept graph, tagging."""
from __future__ import annotations

import json
import re

import numpy as np

from .. import config, llm
from ..db import DB, from_blob, to_blob
from ..embed import embed_passages
from ..llm import STR, arr, obj

# --------------------------------------------------------------------------- figures

_VISION_SYSTEM = (
    "You read course material for a study tool. Describe only what is visibly on the image. "
    "Do not add facts from outside the image."
)
_VISION_USER = (
    "This is {what}. List every diagram, plot, table or figure on it. For each, give a short title and a "
    "description a student could learn from without seeing it: what is drawn, the axis and node labels, the "
    "numbers shown, and the idea it illustrates. Write math in plain Unicode. If there is no diagram, plot, "
    "table or figure (text and formulas alone do not count), return an empty list."
)
_VISION_SCHEMA = obj(figures=arr(obj(title=STR, description=STR)))


def describe_figures(db: DB, log=print) -> int:
    """Create a `figure` unit for each diagram found on a slide or textbook page."""
    rows = db.q(
        "SELECT u.*, s.kind AS skind, s.title AS stitle FROM units u JOIN sources s ON s.id=u.source_id "
        "WHERE u.kind IN ('slide','text') AND u.image_path IS NOT NULL ORDER BY u.id"
    )
    todo, textbook_pages = [], 0
    for r in rows:
        meta = json.loads(r["meta"])
        if not meta.get("has_visual") or meta.get("vision_done"):
            continue
        if r["kind"] == "text":
            textbook_pages += 1
            if textbook_pages > config.MAX_TEXTBOOK_FIGURE_PAGES:
                continue
        todo.append((r, meta))
    made = 0
    for i, (r, meta) in enumerate(todo):
        what = "a lecture slide" if r["kind"] == "slide" else "a textbook page"
        out = llm.ask("bulk", _VISION_SYSTEM, _VISION_USER.format(what=what), _VISION_SCHEMA,
                      images=[config.DATA / r["image_path"]], max_tokens=4000, cache=True)
        for fig in out["figures"]:
            text = f"[Figure] {fig['title']}. {fig['description']}"
            db.x(
                "INSERT INTO units(source_id, kind, loc_type, loc_start, loc_end, text, image_path, order_idx, meta) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (r["source_id"], "figure", r["loc_type"], r["loc_start"], r["loc_end"], text, r["image_path"],
                 r["order_idx"] + 0.5, json.dumps({"derived": True, "parent_unit": r["id"]})),
            )
            made += 1
        meta["vision_done"] = True
        db.x("UPDATE units SET meta=? WHERE id=?", (json.dumps(meta), r["id"]))
        if i % 10 == 0:
            log(f"figures: {i + 1}/{len(todo)} images read, {made} figure units")
    return made


# --------------------------------------------------------------------------- alignment


def align_slides_to_video(db: DB, log=print) -> int:
    """Link each slide to the transcript window where it is discussed.

    Slides are shown in order, so we pick the monotone assignment slide_i -> window_a(i),
    a(1) <= a(2) <= ..., that maximises total embedding similarity (dynamic programming).
    """
    made = 0
    groups = db.q("SELECT DISTINCT grp FROM sources WHERE grp IS NOT NULL")
    for g in groups:
        slides = db.q(
            "SELECT u.id, u.emb FROM units u JOIN sources s ON s.id=u.source_id "
            "WHERE s.grp=? AND u.kind='slide' AND u.emb IS NOT NULL ORDER BY u.loc_start", (g["grp"],))
        windows = db.q(
            "SELECT u.id, u.emb FROM units u JOIN sources s ON s.id=u.source_id "
            "WHERE s.grp=? AND u.kind='transcript' AND u.emb IS NOT NULL ORDER BY u.loc_start", (g["grp"],))
        if not slides or not windows:
            continue
        S = np.stack([from_blob(r["emb"]) for r in slides])
        W = np.stack([from_blob(r["emb"]) for r in windows])
        sim = S @ W.T
        # A slide stays up for a while: score it against a short run of windows.
        smooth = sim.copy()
        smooth[:, :-1] += 0.5 * sim[:, 1:]
        n, m = sim.shape
        best = np.full((n, m), -np.inf)
        back = np.zeros((n, m), dtype=int)
        best[0] = smooth[0]
        for i in range(1, n):
            run, arg = -np.inf, 0
            for j in range(m):
                if best[i - 1, j] > run:
                    run, arg = best[i - 1, j], j
                best[i, j] = run + smooth[i, j]
                back[i, j] = arg
        j = int(np.argmax(best[-1]))
        path = [0] * n
        for i in range(n - 1, -1, -1):
            path[i] = j
            j = back[i, j]
        db.x("DELETE FROM links WHERE kind='shown_at' AND a_unit IN (%s)" % ",".join(str(r["id"]) for r in slides))
        db.many("INSERT OR REPLACE INTO links(a_unit, b_unit, kind, score) VALUES(?,?,?,?)",
                [(slides[i]["id"], windows[path[i]]["id"], "shown_at", float(sim[i, path[i]])) for i in range(n)])
        made += n
    log(f"alignment: linked {made} slides to video timestamps")
    return made


# --------------------------------------------------------------------------- concept graph

_GRAPH_SYSTEM = (
    "You organise course material into a concept map for a tutoring system. Use only concepts that the "
    "provided outline actually teaches. Names must be the terms the material itself uses."
)
_GRAPH_USER = """Below is an outline of a course, in teaching order, built from its slides, lecture transcripts and textbook.

Produce a concept map:
- 4 to 10 major topics, each with subtopics, each with key concepts. Aim for 25 to 45 concepts in total.
- List topics, subtopics and concepts in the order the course teaches them.
- A concept is one idea a student can be tested on (for example "Bayes' rule", not "Lecture 2").
- definition: one or two sentences, in the material's own terms.
- prerequisites: names of other concepts in your list that a student must understand first. Only direct
  prerequisites. A prerequisite must be taught earlier in the course.
- misconceptions: one to three specific wrong beliefs students commonly hold about this concept, each a
  short sentence stating the wrong belief.

OUTLINE
{outline}"""
_GRAPH_SCHEMA = obj(topics=arr(obj(name=STR, subtopics=arr(obj(name=STR, concepts=arr(obj(
    name=STR, definition=STR, prerequisites=arr(STR), misconceptions=arr(STR))))))))

_HEADING = re.compile(r"^(\d{1,2}\.\d{1,2})\s+([A-Z][A-Za-z ,'-]{3,60})$", re.M)


def _outline(db: DB) -> str:
    lines: list[str] = []
    for s in db.q("SELECT * FROM sources ORDER BY order_idx"):
        if s["kind"] == "slides":
            lines.append(f"\n## Slides: {s['title']}")
            for u in db.q("SELECT loc_start, text FROM units WHERE source_id=? AND kind='slide' ORDER BY loc_start", (s["id"],)):
                lines.append(f"- slide {int(u['loc_start'])}: " + re.sub(r"\s+", " ", u["text"])[:260])
        elif s["kind"] == "textbook":
            lines.append(f"\n## Textbook: {s['title']}")
            seen = set()
            for u in db.q("SELECT loc_start, text FROM units WHERE source_id=? AND kind='text' ORDER BY order_idx", (s["id"],)):
                for num, title in _HEADING.findall(u["text"]):
                    if num not in seen:
                        seen.add(num)
                        lines.append(f"- section {num} {title.strip()} (page {int(u['loc_start'])})")
                for m in re.finditer(r"^(Definition|Theorem)\s+\d+\.\d+\s+(.{0,160})", u["text"], re.M):
                    lines.append(f"  - {m.group(1)} (page {int(u['loc_start'])}): {m.group(2).strip()}")
        else:
            has_slides = s["grp"] and db.one("SELECT 1 FROM sources WHERE grp=? AND kind='slides'", (s["grp"],))
            lines.append(f"\n## Video: {s['title']}")
            if not has_slides:  # no deck to summarise it, so sample the transcript
                units = db.q("SELECT loc_start, text FROM units WHERE source_id=? ORDER BY loc_start", (s["id"],))
                for u in units[:: max(1, len(units) // 12)]:
                    lines.append(f"- at {int(u['loc_start'])}s: " + u["text"][:260])
    return "\n".join(lines)


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:60]


def build_concept_graph(db: DB, log=print) -> int:
    """Ask the generator for topics, concepts and prerequisites, then store them.

    Existing mastery rows survive a rebuild for any concept whose slug is unchanged.
    """
    out = llm.ask("generator", _GRAPH_SYSTEM, _GRAPH_USER.format(outline=_outline(db)[:120000]),
                  _GRAPH_SCHEMA, max_tokens=32000, cache=True)
    concepts, by_name, edges = [], {}, []
    for topic in out["topics"]:
        for sub in topic["subtopics"]:
            for c in sub["concepts"]:
                cid = slug(c["name"])
                if not cid or cid in {x["id"] for x in concepts}:
                    continue
                concepts.append({"id": cid, "name": c["name"], "definition": c["definition"], "topic": topic["name"],
                                 "subtopic": sub["name"], "misconceptions": c["misconceptions"],
                                 "prereq_names": c["prerequisites"]})
                by_name[c["name"].lower()] = cid
    for c in concepts:
        for p in c["prereq_names"]:
            pid = by_name.get(p.lower()) or (slug(p) if slug(p) in by_name.values() else None)
            if pid and pid != c["id"]:
                edges.append((c["id"], pid))
    vecs = embed_passages([f"{c['name']}: {c['definition']}" for c in concepts])
    db.x("DELETE FROM concepts")
    db.x("DELETE FROM prereqs")
    db.many("INSERT INTO concepts(id, name, definition, topic, subtopic, order_idx, misconceptions, emb) VALUES(?,?,?,?,?,?,?,?)",
            [(c["id"], c["name"], c["definition"], c["topic"], c["subtopic"], i, json.dumps(c["misconceptions"]), to_blob(v))
             for i, (c, v) in enumerate(zip(concepts, vecs))])
    db.many("INSERT OR IGNORE INTO prereqs(concept_id, prereq_id) VALUES(?,?)", edges)
    log(f"concepts: {len(concepts)} concepts, {len(edges)} proposed prerequisite edges")
    return len(concepts)


def tag_units(db: DB, top_k: int = 3, min_sim: float = 0.55, log=print) -> None:
    """Tag every unit with its nearest concepts. Each unit gets at least its best match."""
    concepts = db.q("SELECT id, emb FROM concepts ORDER BY order_idx")
    units = db.q("SELECT id, emb FROM units WHERE emb IS NOT NULL")
    if not concepts or not units:
        return
    C = np.stack([from_blob(c["emb"]) for c in concepts])
    U = np.stack([from_blob(u["emb"]) for u in units])
    sim = U @ C.T
    rows = []
    for i, u in enumerate(units):
        order = np.argsort(-sim[i])[:top_k]
        for rank, j in enumerate(order):
            if rank == 0 or sim[i, j] >= min_sim:
                rows.append((u["id"], concepts[j]["id"], float(sim[i, j])))
    db.x("DELETE FROM unit_concepts")
    db.many("INSERT INTO unit_concepts(unit_id, concept_id, score) VALUES(?,?,?)", rows)
    log(f"tagging: {len(rows)} unit-concept tags over {len(units)} units")


def validate_prerequisites(db: DB, log=print) -> dict:
    """Drop prerequisite edges the material contradicts.

    An edge "B requires A" is kept only if A is taught no later than B, where "taught" is the
    earliest well-matched unit in course order, and the graph must stay acyclic.
    """
    first = {r["concept_id"]: r["o"] for r in db.q(
        "SELECT uc.concept_id, MIN(u.order_idx) AS o FROM unit_concepts uc JOIN units u ON u.id=uc.unit_id "
        "WHERE uc.score >= 0.6 GROUP BY uc.concept_id")}
    listed = {c["id"]: c["order_idx"] for c in db.q("SELECT id, order_idx FROM concepts")}
    edges = [(r["concept_id"], r["prereq_id"]) for r in db.q("SELECT * FROM prereqs")]
    kept, dropped_order = [], []
    for c, p in edges:
        if c in first and p in first:
            ok = first[p] <= first[c]
        else:                       # no strong evidence in the material: trust listing order
            ok = listed.get(p, 0) < listed.get(c, 0)
        (kept if ok else dropped_order).append((c, p))
    # Break any remaining cycle by removing the edge that closes it.
    graph: dict[str, list[str]] = {}
    dropped_cycle = []
    for c, p in sorted(kept, key=lambda e: listed.get(e[0], 0)):
        stack, seen, cyc = [p], set(), False
        while stack:
            node = stack.pop()
            if node == c:
                cyc = True
                break
            if node not in seen:
                seen.add(node)
                stack.extend(graph.get(node, []))
        if cyc:
            dropped_cycle.append((c, p))
        else:
            graph.setdefault(c, []).append(p)
    db.x("DELETE FROM prereqs")
    db.many("INSERT INTO prereqs(concept_id, prereq_id) VALUES(?,?)", [(c, p) for c, ps in graph.items() for p in ps])
    stats = {"proposed": len(edges), "kept": sum(len(v) for v in graph.values()),
             "dropped_taught_later": len(dropped_order), "dropped_cycle": len(dropped_cycle)}
    db.set_kv("prereq_validation", stats)
    log(f"prerequisites: {stats}")
    return stats
