"""Paths and environment-driven settings."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA = Path(os.getenv("PRAMANA_DATA", ROOT / "data"))
RAW = DATA / "raw"          # downloaded / uploaded originals
RENDERS = DATA / "renders"  # slide and page images
CACHE = DATA / "cache"      # LLM response cache (keeps ingestion reproducible)
DB_PATH = DATA / "pramana.sqlite"
MANIFEST = ROOT / "course.yaml"

for _d in (RAW, RENDERS, CACHE):
    _d.mkdir(parents=True, exist_ok=True)

GENERATOR = os.getenv("PRAMANA_GENERATOR", "anthropic:claude-opus-5-5")
VERIFIER = os.getenv("PRAMANA_VERIFIER", "anthropic:claude-sonnet-5-5")
BULK = os.getenv("PRAMANA_BULK", GENERATOR)

EMBED_MODEL = os.getenv("PRAMANA_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
COVERAGE_THRESHOLD = float(os.getenv("PRAMANA_COVERAGE_THRESHOLD", "0.72"))

# Vision is the expensive ingestion stage; cap textbook pages sent to the model.
MAX_TEXTBOOK_FIGURE_PAGES = int(os.getenv("PRAMANA_MAX_FIGURE_PAGES", "60"))
