"""Command line entry points.

  python -m pramana.cli ingest [--force] [--only ID ...]   fetch and index the course in course.yaml
  python -m pramana.cli enrich [--rebuild-graph]           re-run figure, alignment and concept stages
  python -m pramana.cli bank --per-concept 2               pre-generate verified questions
  python -m pramana.cli stats                              what is in the knowledge base
  python -m pramana.cli serve [--port 8000]                run the API and web app
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="pramana")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest")
    p.add_argument("--force", action="store_true")
    p.add_argument("--only", nargs="*")
    p = sub.add_parser("enrich")
    p.add_argument("--rebuild-graph", action="store_true")
    p = sub.add_parser("bank")
    p.add_argument("--per-concept", type=int, default=2)
    p.add_argument("--types", nargs="*", default=["mcq", "numeric", "short"])
    sub.add_parser("stats")
    p = sub.add_parser("serve")
    p.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    p.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    args = ap.parse_args(argv)

    from .db import get_db

    db = get_db()
    if args.cmd == "ingest":
        from .ingest.pipeline import run_manifest

        run_manifest(db, force=args.force, only=args.only)
    elif args.cmd == "enrich":
        from .ingest.pipeline import enrich_all

        enrich_all(db, rebuild_graph=args.rebuild_graph)
    elif args.cmd == "bank":
        from .assess.generate import prebuild_bank

        print(json.dumps(prebuild_bank(db, args.per_concept, args.types), indent=2))
    elif args.cmd == "stats":
        from .service import kb_stats

        print(json.dumps(kb_stats(db), indent=2))
    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("pramana.api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
