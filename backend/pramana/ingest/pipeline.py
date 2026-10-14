"""Ingestion pipeline: fetch -> parse -> embed -> figures -> align -> concept graph -> tag."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import httpx
import yaml

from .. import config, llm
from ..db import DB
from ..embed import embed_missing_units
from . import enrich, parse


def load_manifest(path: Path | None = None) -> list[dict]:
    data = yaml.safe_load((path or config.MANIFEST).read_text(encoding="utf-8"))
    return data["sources"]


def _fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    with httpx.stream("GET", url, follow_redirects=True, timeout=120,
                      headers={"User-Agent": "Mozilla/5.0 (Pramana study companion)"}) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    return dest


def _youtube_title(video_id: str) -> str:
    try:
        r = httpx.get("https://www.youtube.com/oembed", timeout=20,
                      params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"})
        return r.json()["title"]
    except Exception:
        return f"YouTube video {video_id}"


def ingest_source(db: DB, spec: dict, log=print, force: bool = False) -> str:
    """Fetch and parse one source into units. `spec` is a manifest entry or an upload descriptor."""
    sid = spec["id"]
    if db.one("SELECT 1 FROM sources WHERE id=?", (sid,)):
        if not force:
            log(f"{sid}: already ingested, skipping")
            return sid
        remove_source(db, sid)
    kind = spec["kind"]
    order = (db.one("SELECT COALESCE(MAX(order_idx), -1) + 1 AS n FROM sources") or {"n": 0})["n"]
    meta: dict = {}
    local = None
    if kind == "video":
        vid = spec.get("youtube")
        if vid:
            drafts, meta = parse.parse_youtube(vid)
            meta["youtube"] = vid
            uri = f"https://www.youtube.com/watch?v={vid}"
            title = spec.get("title") or _youtube_title(vid)
        else:
            local = Path(spec["path"])
            drafts, meta = parse.parse_media_file(local)
            uri, title = str(local), spec.get("title") or local.stem
    else:
        if spec.get("url"):
            local = _fetch(spec["url"], config.RAW / f"{sid}.pdf")
        else:
            src = Path(spec["path"])
            local = config.RAW / f"{sid}.pdf"
            if src.resolve() != local.resolve():
                shutil.copyfile(src, local)
        uri = spec.get("url") or str(local)
        title = spec.get("title") or local.stem
        if kind == "slides":
            drafts = parse.parse_slides(local, sid)
        else:
            pages = tuple(spec["pages"]) if spec.get("pages") else None
            drafts = parse.parse_textbook(local, sid, pages)
    db.x("INSERT INTO sources(id, kind, title, uri, local_path, grp, order_idx, meta) VALUES(?,?,?,?,?,?,?,?)",
         (sid, kind, title, uri, str(local.relative_to(config.DATA)) if local and local.is_relative_to(config.DATA) else None,
          spec.get("group"), order, json.dumps(meta)))
    db.many(
        "INSERT INTO units(source_id, kind, loc_type, loc_start, loc_end, text, image_path, order_idx, meta) VALUES(?,?,?,?,?,?,?,?,?)",
        [(sid, d.kind, d.loc_type, d.loc_start, d.loc_end, d.text, d.image_path, order * 100000 + i, json.dumps(d.meta))
         for i, d in enumerate(drafts)])
    log(f"{sid}: {len(drafts)} units from {kind}")
    return sid


def remove_source(db: DB, sid: str) -> None:
    ids = [r["id"] for r in db.q("SELECT id FROM units WHERE source_id=?", (sid,))]
    if ids:
        marks = ",".join("?" * len(ids))
        db.x(f"DELETE FROM unit_concepts WHERE unit_id IN ({marks})", ids)
        db.x(f"DELETE FROM links WHERE a_unit IN ({marks}) OR b_unit IN ({marks})", ids + ids)
    db.x("DELETE FROM units WHERE source_id=?", (sid,))
    db.x("DELETE FROM sources WHERE id=?", (sid,))


def _auto_group(db: DB, sid: str) -> None:
    """Pair an ungrouped slide deck or video with its best-matching counterpart by content."""
    import numpy as np

    from ..db import from_blob

    src = db.one("SELECT * FROM sources WHERE id=?", (sid,))
    if not src or src["grp"] or src["kind"] not in ("slides", "video"):
        return
    other_kind = "video" if src["kind"] == "slides" else "slides"

    def centroid(source_id: str):
        rows = db.q("SELECT emb FROM units WHERE source_id=? AND emb IS NOT NULL", (source_id,))
        if not rows:
            return None
        c = np.mean([from_blob(r["emb"]) for r in rows], axis=0)
        return c / np.linalg.norm(c)

    mine = centroid(sid)
    best, best_sim = None, 0.80
    for o in db.q("SELECT id, grp FROM sources WHERE kind=?", (other_kind,)):
        taken = o["grp"] and db.one("SELECT 1 FROM sources WHERE grp=? AND kind=?", (o["grp"], src["kind"]))
        c = centroid(o["id"])
        if mine is None or c is None or taken:
            continue
        sim = float(mine @ c)
        if sim > best_sim:
            best, best_sim = o, sim
    if best:
        grp = best["grp"] or f"auto-{best['id']}"
        db.x("UPDATE sources SET grp=? WHERE id IN (?, ?)", (grp, sid, best["id"]))


def enrich_all(db: DB, log=print, rebuild_graph: bool = False) -> dict:
    """Run every post-parse stage. LLM stages are skipped, loudly, when no key is configured."""
    skipped = []
    n = embed_missing_units(db)
    log(f"embeddings: {n} new units embedded")
    for s in db.q("SELECT id FROM sources WHERE grp IS NULL"):
        _auto_group(db, s["id"])
    if llm.available("bulk"):
        made = enrich.describe_figures(db, log)
        if made:
            embed_missing_units(db)
    else:
        skipped.append("figure understanding (no key for the bulk model)")
    enrich.align_slides_to_video(db, log)
    has_graph = bool(db.one("SELECT 1 FROM concepts"))
    if rebuild_graph or not has_graph:
        if llm.available("generator"):
            enrich.build_concept_graph(db, log)
            has_graph = True
        else:
            skipped.append("concept graph (no key for the generator model)")
    if has_graph:
        enrich.tag_units(db, log=log)
        if rebuild_graph or not db.get_kv("prereq_validation"):
            enrich.validate_prerequisites(db, log)
    for s in skipped:
        log(f"SKIPPED: {s}")
    return {"skipped": skipped}


def run_manifest(db: DB, log=print, force: bool = False, only: list[str] | None = None) -> dict:
    for spec in load_manifest():
        if only and spec["id"] not in only:
            continue
        try:
            ingest_source(db, spec, log, force)
        except Exception as e:  # one unreachable source must not abort the rest
            log(f"{spec['id']}: FAILED - {type(e).__name__}: {e}")
    return enrich_all(db, log, rebuild_graph=force)
