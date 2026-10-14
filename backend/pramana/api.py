"""HTTP API and static hosting for the web app."""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, extras, grounding, llm
from .assess import generate as G
from .assess import report as R
from .db import get_db
from .embed import HybridIndex
from .ingest import pipeline
from .llm import LLMError, LLMUnavailable
from .service import kb_stats, load_graph, load_model, today, unit_view

app = FastAPI(title="Pramana")

_cors_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
_render_url = os.getenv("RENDER_EXTERNAL_URL")
if _render_url:
    _cors_origins.append(_render_url)
app.add_middleware(CORSMiddleware, allow_origins=_cors_origins,
                   allow_methods=["*"], allow_headers=["*"])
db = get_db()
_index: HybridIndex | None = None


def index() -> HybridIndex:
    global _index
    if _index is None:
        _index = HybridIndex(db)
    return _index


def _llm_guard(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except LLMUnavailable as e:
        raise HTTPException(503, str(e))
    except LLMError as e:
        raise HTTPException(502, str(e))
    except KeyError as e:
        raise HTTPException(404, f"not found: {e}")


# --------------------------------------------------------------------------- health check

@app.get("/healthz")
def healthz():
    """Lightweight liveness probe for Render health checks."""
    return {"status": "ok"}


# --------------------------------------------------------------------------- knowledge base

@app.get("/api/status")
def status():
    return {"models": {r: {"spec": llm.spec_for(r), "ready": llm.available(r)} for r in ("generator", "verifier", "bulk")},
            "kb": kb_stats(db), "coverage_threshold": config.COVERAGE_THRESHOLD}


@app.get("/api/sources")
def sources():
    rows = db.q("SELECT s.*, (SELECT COUNT(*) FROM units u WHERE u.source_id=s.id) AS units, "
                "(SELECT COUNT(*) FROM units u WHERE u.source_id=s.id AND u.kind='figure') AS figures "
                "FROM sources s ORDER BY s.order_idx")
    for r in rows:
        r["meta"] = json.loads(r["meta"])
    return rows


@app.get("/api/sources/{source_id}/units")
def source_units(source_id: str):
    ids = db.q("SELECT id FROM units WHERE source_id=? ORDER BY order_idx", (source_id,))
    return [unit_view(db, r["id"], False) for r in ids]


@app.get("/api/units/{unit_id}")
def unit(unit_id: int):
    v = unit_view(db, unit_id, True)
    if not v:
        raise HTTPException(404, "unit not found")
    return v


def _run_job(job_id: str, work):
    lines: list[str] = []

    def log(msg: str):
        lines.append(msg)
        db.x("UPDATE jobs SET message=?, log=? WHERE id=?", (msg, json.dumps(lines[-200:]), job_id))

    try:
        work(log)
        index().reload()
        db.x("UPDATE jobs SET status='done', progress=1 WHERE id=?", (job_id,))
    except Exception as e:
        log(f"FAILED: {type(e).__name__}: {e}")
        db.x("UPDATE jobs SET status='failed' WHERE id=?", (job_id,))


def _start_job(kind: str, work) -> dict:
    job_id = uuid.uuid4().hex[:10]
    db.x("INSERT INTO jobs(id, kind, status, progress, message, created_at) VALUES(?,?,?,?,?,?)",
         (job_id, kind, "running", 0, "queued", time.time()))
    threading.Thread(target=_run_job, args=(job_id, work), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    j = db.one("SELECT * FROM jobs WHERE id=?", (job_id,))
    if not j:
        raise HTTPException(404, "job not found")
    j["log"] = json.loads(j["log"])
    return j


@app.post("/api/ingest/course")
def ingest_course():
    """Fetch and index the default course listed in course.yaml."""
    return _start_job("course", lambda log: pipeline.run_manifest(db, log))


@app.post("/api/ingest/upload")
async def ingest_upload(kind: str = Form(...), title: str = Form(""), youtube_url: str = Form(""),
                        file: UploadFile | None = File(None)):
    """Add one source: a PDF (slides or textbook), a YouTube lecture, or a local video file."""
    if kind not in ("slides", "textbook", "video"):
        raise HTTPException(400, "kind must be slides, textbook or video")
    sid = re.sub(r"[^a-z0-9]+", "-", (title or (file.filename if file else "") or youtube_url).lower()).strip("-")[:40]
    sid = f"{sid or kind}-{uuid.uuid4().hex[:4]}"
    spec: dict = {"id": sid, "kind": kind, "title": title or None}
    if kind == "video" and youtube_url:
        from .ingest.parse import youtube_id

        vid = youtube_id(youtube_url)
        if not vid:
            raise HTTPException(400, "could not read a YouTube video id from that URL")
        spec["youtube"] = vid
    elif file is not None:
        suffix = Path(file.filename or "").suffix.lower()
        if kind != "video" and suffix != ".pdf":
            raise HTTPException(400, "slides and textbooks must be PDF files")
        dest = config.RAW / f"{sid}{suffix if kind == 'video' else '.pdf'}"
        dest.write_bytes(await file.read())
        spec["path"] = str(dest)
    else:
        raise HTTPException(400, "provide a file or a YouTube URL")

    def work(log):
        pipeline.ingest_source(db, spec, log)
        pipeline.enrich_all(db, log)

    return _start_job("upload", work)


@app.post("/api/ingest/rebuild-graph")
def rebuild_graph():
    return _start_job("graph", lambda log: pipeline.enrich_all(db, log, rebuild_graph=True))


@app.get("/api/search")
def search(q: str, k: int = 8):
    hits = index().search(q, k=k)
    return [dict(unit_view(db, h.unit_id, False), score=round(h.dense, 3)) for h in hits]


# --------------------------------------------------------------------------- students

class StudentIn(BaseModel):
    name: str
    level: str = "beginner"          # beginner | intermediate | advanced (sets the prior)
    language: str = "en"             # en | hi | hinglish
    exam_date: str | None = None


@app.get("/api/students")
def students():
    return db.q("SELECT * FROM students ORDER BY created_at")


@app.post("/api/students")
def create_student(body: StudentIn):
    sid = uuid.uuid4().hex[:8]
    db.x("INSERT INTO students(id, name, level, language, exam_date, created_at) VALUES(?,?,?,?,?,?)",
         (sid, body.name.strip() or "Student", body.level, body.language, body.exam_date, time.time()))
    return db.one("SELECT * FROM students WHERE id=?", (sid,))


@app.patch("/api/students/{student_id}")
def update_student(student_id: str, body: StudentIn):
    db.x("UPDATE students SET name=?, level=?, language=?, exam_date=? WHERE id=?",
         (body.name, body.level, body.language, body.exam_date, student_id))
    return db.one("SELECT * FROM students WHERE id=?", (student_id,))


@app.get("/api/graph")
def graph(student_id: str | None = None):
    """Concept map: topics, concepts, prerequisite edges and, for a student, mastery per concept."""
    g, rows = load_graph(db)
    model = load_model(db, student_id, g) if student_id else None
    t = today()
    counts = {r["concept_id"]: r["n"] for r in db.q("SELECT concept_id, COUNT(*) AS n FROM unit_concepts GROUP BY concept_id")}
    bank = {r["concept_id"]: r["n"] for r in db.q(
        "SELECT concept_id, COUNT(*) AS n FROM questions WHERE status='verified' GROUP BY concept_id")}
    nodes = []
    for cid, r in rows.items():
        node = {"id": cid, "name": r["name"], "definition": r["definition"], "topic": r["topic"], "subtopic": r["subtopic"],
                "depth": g.depth(cid), "units": counts.get(cid, 0), "questions": bank.get(cid, 0),
                "misconceptions": r["misconceptions"]}
        if model:
            st = model.s[cid]
            node.update(mastery=round(model.p_now(cid, t), 3), stored=round(st.p, 3), attempts=st.n,
                        stability_days=round(st.stability, 1), readiness=round(model.readiness(cid, t), 3))
        nodes.append(node)
    return {"nodes": nodes, "edges": [{"from": p, "to": c} for c, ps in g.prereqs.items() for p in ps]}


@app.get("/api/concepts/{concept_id}/units")
def concept_units(concept_id: str, k: int = 6):
    rows = db.q("SELECT unit_id, score FROM unit_concepts WHERE concept_id=? ORDER BY score DESC LIMIT ?", (concept_id, k))
    return [dict(unit_view(db, r["unit_id"], True), tag_score=round(r["score"], 3)) for r in rows]


@app.get("/api/students/{student_id}/dashboard")
def dashboard(student_id: str):
    student = db.one("SELECT * FROM students WHERE id=?", (student_id,))
    if not student:
        raise HTTPException(404, "student not found")
    g, rows = load_graph(db)
    model = load_model(db, student_id, g)
    t = today()
    ps = [model.p_now(c, t) for c in g.ids]
    history = db.q("SELECT ts, concept_id, kind, correct, p_before, p_after FROM events WHERE student_id=? ORDER BY id", (student_id,))
    # Mean mastery after each event, replayed from the log (ignores decay between events).
    cur = {c: model.params.p_init for c in g.ids}
    first_seen = {}
    for e in history:
        first_seen.setdefault(e["concept_id"], e["p_before"])
    cur.update(first_seen)
    trend = []
    for e in history:
        if e["concept_id"] in cur:
            cur[e["concept_id"]] = e["p_after"]
            trend.append({"ts": e["ts"], "mean": round(sum(cur.values()) / len(cur), 4), "kind": e["kind"]})
    mis = db.q("SELECT ai.misconception, COUNT(*) AS n FROM assessment_items ai JOIN assessments a ON a.id=ai.assessment_id "
               "WHERE a.student_id=? AND ai.misconception IS NOT NULL GROUP BY ai.misconception ORDER BY n DESC LIMIT 6", (student_id,))
    chat_mis = db.q("SELECT ref AS misconception, COUNT(*) AS n FROM events WHERE student_id=? AND kind='chat_misconception' "
                    "GROUP BY ref ORDER BY n DESC LIMIT 4", (student_id,))
    assessments = db.q(
        "SELECT a.id, a.mode, a.created_at, a.finished_at, COUNT(ai.idx) AS items, SUM(ai.score) AS score, "
        "SUM(CASE WHEN ai.answered_at IS NOT NULL THEN 1 ELSE 0 END) AS answered "
        "FROM assessments a LEFT JOIN assessment_items ai ON ai.assessment_id=a.id WHERE a.student_id=? "
        "GROUP BY a.id ORDER BY a.created_at DESC LIMIT 10", (student_id,))
    weak = [{"concept_id": c, "name": rows[c]["name"], "mastery": round(p, 3), "attempts": model.s[c].n}
            for c, p in model.weak_concepts(t)][:8]
    return {
        "student": student, "concepts": len(g.ids),
        "mean_mastery": round(sum(ps) / len(ps), 3) if ps else 0,
        "mastered": sum(p >= 0.85 for p in ps), "weak": weak,
        "root_causes": [{"concept": rows[d["concept"]]["name"], "root_cause": rows[d["root_cause"]]["name"],
                         "p_root": round(d["p_root"], 2)} for d in model.diagnose(t)],
        "trend": trend, "misconceptions": mis + chat_mis, "assessments": assessments,
        "events": len(history),
    }


# --------------------------------------------------------------------------- tutor

class ChatIn(BaseModel):
    student_id: str
    message: str
    allow_outside: bool = False


@app.post("/api/chat")
def chat(body: ChatIn):
    return _llm_guard(grounding.answer, db, index(), body.student_id, body.message, body.allow_outside)


@app.get("/api/students/{student_id}/chat")
def chat_history(student_id: str):
    rows = db.q("SELECT role, content, ts FROM chats WHERE student_id=? ORDER BY id DESC LIMIT 40", (student_id,))
    return [{"role": r["role"], "ts": r["ts"], **json.loads(r["content"])} for r in reversed(rows)]


# --------------------------------------------------------------------------- assessment

class AssessmentIn(BaseModel):
    student_id: str
    mode: str = "quiz"                       # quiz | mock | diagnostic
    concept_ids: list[str] = []              # empty means the whole course
    n: int = 8
    types: list[str] = ["mcq", "numeric", "short"]


class AnswerIn(BaseModel):
    idx: int
    response: str


@app.post("/api/assessments")
def create_assessment(body: AssessmentIn):
    if not db.one("SELECT 1 FROM concepts"):
        raise HTTPException(409, "No concept graph yet. Ingest a course first.")
    out = _llm_guard(G.create_assessment, db, body.student_id, body.mode, body.concept_ids,
                     max(1, min(body.n, 30)), body.types)
    if not out["questions"]:
        raise HTTPException(503, "No questions available for this scope. The question bank is empty and no "
                                 "generator key is configured.")
    return out


@app.post("/api/assessments/{assessment_id}/answer")
def answer(assessment_id: str, body: AnswerIn):
    return _llm_guard(R.submit_answer, db, assessment_id, body.idx, body.response)


@app.get("/api/assessments/{assessment_id}/report")
def report(assessment_id: str):
    return _llm_guard(R.build_report, db, assessment_id)


@app.get("/api/bank")
def bank():
    return G.bank_stats(db)


# --------------------------------------------------------------------------- revision

@app.get("/api/students/{student_id}/flashcards")
def flashcards(student_id: str):
    return _llm_guard(extras.flashcards, db, student_id)


@app.get("/api/students/{student_id}/schedule")
def schedule(student_id: str, minutes_per_day: int = 45):
    return extras.study_schedule(db, student_id, minutes_per_day)


# --------------------------------------------------------------------------- evaluation results

@app.get("/api/eval")
def eval_results():
    out = {}
    for p in sorted((config.ROOT / "eval" / "results").glob("*.json")):
        out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    return out


# --------------------------------------------------------------------------- static

# Serve only originals and renders; the database lives in the same folder and must not be exposed.
app.mount("/files/raw", StaticFiles(directory=config.RAW), name="raw")
app.mount("/files/renders", StaticFiles(directory=config.RENDERS), name="renders")
_DIST = config.ROOT / "frontend" / "dist"
if _DIST.exists():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        target = _DIST / path
        return FileResponse(target if path and target.is_file() else _DIST / "index.html")
