"""SQLite persistence. One file holds the knowledge base, question bank and learner state."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources(
  id TEXT PRIMARY KEY, kind TEXT, title TEXT, uri TEXT, local_path TEXT,
  grp TEXT, order_idx INTEGER, meta TEXT DEFAULT '{}');
-- A unit is the smallest citable piece of course material.
-- loc_type: page | slide | time ; loc_start/loc_end are page numbers, slide numbers or seconds.
CREATE TABLE IF NOT EXISTS units(
  id INTEGER PRIMARY KEY AUTOINCREMENT, source_id TEXT, kind TEXT,
  loc_type TEXT, loc_start REAL, loc_end REAL, text TEXT, image_path TEXT,
  order_idx REAL, meta TEXT DEFAULT '{}', emb BLOB);
CREATE INDEX IF NOT EXISTS units_source ON units(source_id);
CREATE TABLE IF NOT EXISTS concepts(
  id TEXT PRIMARY KEY, name TEXT, definition TEXT, topic TEXT, subtopic TEXT,
  order_idx INTEGER, misconceptions TEXT DEFAULT '[]', emb BLOB);
CREATE TABLE IF NOT EXISTS prereqs(concept_id TEXT, prereq_id TEXT, PRIMARY KEY(concept_id, prereq_id));
CREATE TABLE IF NOT EXISTS unit_concepts(unit_id INTEGER, concept_id TEXT, score REAL, PRIMARY KEY(unit_id, concept_id));
-- Cross-modal links, e.g. slide unit -> transcript unit where it is shown.
CREATE TABLE IF NOT EXISTS links(a_unit INTEGER, b_unit INTEGER, kind TEXT, score REAL, PRIMARY KEY(a_unit, b_unit, kind));

CREATE TABLE IF NOT EXISTS students(
  id TEXT PRIMARY KEY, name TEXT, level TEXT, language TEXT, exam_date TEXT,
  created_at REAL, diagnosed INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS mastery(
  student_id TEXT, concept_id TEXT, p REAL, stability REAL, last_t REAL, n INTEGER, n_correct REAL,
  PRIMARY KEY(student_id, concept_id));
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT, ts REAL, concept_id TEXT, kind TEXT,
  correct REAL, p_before REAL, p_after REAL, ref TEXT);
CREATE INDEX IF NOT EXISTS events_student ON events(student_id, ts);

CREATE TABLE IF NOT EXISTS questions(
  id INTEGER PRIMARY KEY AUTOINCREMENT, concept_id TEXT, unit_id INTEGER, qtype TEXT, difficulty INTEGER,
  stem TEXT, payload TEXT, verification TEXT, status TEXT, emb BLOB, created_at REAL);
CREATE INDEX IF NOT EXISTS questions_concept ON questions(concept_id, status);
CREATE TABLE IF NOT EXISTS assessments(
  id TEXT PRIMARY KEY, student_id TEXT, mode TEXT, scope TEXT, created_at REAL, finished_at REAL);
CREATE TABLE IF NOT EXISTS assessment_items(
  assessment_id TEXT, idx INTEGER, question_id INTEGER, response TEXT, score REAL,
  misconception TEXT, feedback TEXT, answered_at REAL, PRIMARY KEY(assessment_id, idx));
CREATE TABLE IF NOT EXISTS chats(
  id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT, ts REAL, role TEXT, content TEXT);
CREATE TABLE IF NOT EXISTS jobs(
  id TEXT PRIMARY KEY, kind TEXT, status TEXT, progress REAL, message TEXT, created_at REAL, log TEXT DEFAULT '[]');
CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT);
"""


class DB:
    """Thin wrapper over one shared connection, serialised by a lock."""

    def __init__(self, path: Path | str | None = None):
        self.path = str(path or config.DB_PATH)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def q(self, sql: str, args: Iterable[Any] = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, tuple(args)).fetchall()]

    def one(self, sql: str, args: Iterable[Any] = ()) -> dict | None:
        rows = self.q(sql, args)
        return rows[0] if rows else None

    def x(self, sql: str, args: Iterable[Any] = ()) -> int:
        with self._lock:
            cur = self._conn.execute(sql, tuple(args))
            self._conn.commit()
            return cur.lastrowid

    def many(self, sql: str, rows: Iterable[Iterable[Any]]) -> None:
        with self._lock:
            self._conn.executemany(sql, [tuple(r) for r in rows])
            self._conn.commit()

    def get_kv(self, k: str, default: Any = None) -> Any:
        row = self.one("SELECT v FROM kv WHERE k=?", (k,))
        return json.loads(row["v"]) if row else default

    def set_kv(self, k: str, v: Any) -> None:
        self.x("INSERT OR REPLACE INTO kv(k, v) VALUES(?, ?)", (k, json.dumps(v)))


def to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def from_blob(b: bytes | None) -> np.ndarray | None:
    return None if b is None else np.frombuffer(b, dtype=np.float32)


def now() -> float:
    return time.time()


_db: DB | None = None


def get_db() -> DB:
    global _db
    if _db is None:
        _db = DB()
    return _db
