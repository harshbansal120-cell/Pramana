"""Local embeddings and the hybrid (dense + BM25) index over units."""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass

import numpy as np

from . import config
from .db import DB, from_blob, to_blob

_STOP = set(
    "a an the of to in on for and or is are was were be been it this that these those with as at by from we you "
    "i so if then what which how why do does did can will would not no yes there here have has had its our your".split()
)
_model = None
_model_lock = threading.Lock()


def _embedder():
    global _model
    with _model_lock:
        if _model is None:
            from fastembed import TextEmbedding

            _model = TextEmbedding(model_name=config.EMBED_MODEL, cache_dir=str(config.CACHE / "models"))
        return _model


def embed_passages(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    vecs = np.array(list(_embedder().passage_embed(texts)), dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def embed_query(text: str) -> np.ndarray:
    v = np.array(next(iter(_embedder().query_embed([text]))), dtype=np.float32)
    return v / np.linalg.norm(v)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP and len(t) > 1]


@dataclass
class Hit:
    unit_id: int
    dense: float   # cosine similarity; drives the coverage gate
    score: float   # fused rank score; drives ordering


class HybridIndex:
    """In-memory index rebuilt from the database. The corpus is a few thousand units."""

    def __init__(self, db: DB):
        self.db = db
        self._lock = threading.Lock()
        self.reload()

    def reload(self) -> None:
        from rank_bm25 import BM25Okapi

        rows = self.db.q("SELECT id, kind, text, emb FROM units WHERE emb IS NOT NULL ORDER BY id")
        with self._lock:
            self.ids = np.array([r["id"] for r in rows], dtype=np.int64)
            self.kinds = [r["kind"] for r in rows]
            self.mat = np.stack([from_blob(r["emb"]) for r in rows]) if rows else np.zeros((0, 384), np.float32)
            docs = [tokenize(r["text"]) or ["_"] for r in rows]
            self.bm25 = BM25Okapi(docs) if rows else None
            self.df: dict[str, int] = {}
            for doc in docs:
                for tok in set(doc):
                    self.df[tok] = self.df.get(tok, 0) + 1

    def rare_terms(self, query: str) -> list[str]:
        """Distinctive query words the corpus uses in only a few units, e.g. a proper noun like "Galton".

        Returns nothing if any distinctive query word is missing from the corpus altogether:
        a question that needs a word the material never uses is not rescued.
        """
        content = [t for t in set(tokenize(query)) if len(t) >= 5]
        if any(self.df.get(t, 0) == 0 for t in content):
            return []
        limit = max(3, int(0.01 * len(self.ids)))
        return [t for t in content if self.df[t] <= limit]

    def search(self, query: str, k: int = 8, allowed: set[int] | None = None, mode: str = "hybrid") -> list[Hit]:
        """mode: hybrid (reciprocal-rank fusion), dense, or bm25. The last two exist for ablations."""
        with self._lock:
            if len(self.ids) == 0:
                return []
            dense = self.mat @ embed_query(query)
            lexical = np.asarray(self.bm25.get_scores(tokenize(query) or ["_"]))
            mask = np.ones(len(self.ids), bool) if allowed is None else np.isin(self.ids, list(allowed))
            if not mask.any():
                return []
            fused = np.zeros(len(self.ids))
            signals = {"hybrid": (dense, lexical), "dense": (dense,), "bm25": (lexical,)}[mode]
            for signal in signals:
                order = np.argsort(-np.where(mask, signal, -np.inf))
                ranks = np.empty(len(order))
                ranks[order] = np.arange(len(order))
                fused += 1.0 / (60 + ranks)
            fused = np.where(mask, fused, -np.inf)
            top = np.argsort(-fused)[:k]
            return [Hit(int(self.ids[i]), float(dense[i]), float(fused[i])) for i in top if mask[i]]


def embed_missing_units(db: DB, batch: int = 64, progress=None) -> int:
    rows = db.q("SELECT id, text FROM units WHERE emb IS NULL AND length(text) > 0")
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        vecs = embed_passages([r["text"][:2000] for r in chunk])
        db.many("UPDATE units SET emb=? WHERE id=?", [(to_blob(v), r["id"]) for v, r in zip(vecs, chunk)])
        if progress:
            progress(min(1.0, (i + batch) / max(1, len(rows))))
    return len(rows)
