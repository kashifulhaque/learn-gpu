"""Writes every chapter and lab as one JSON file for the PDF build.

    python3 scripts/dump_book.py book/.build/book.json

The book renderer in `frontend/src/book/` reads this file instead of the API,
so the PDF comes from the same loader the site serves, with no server running.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app import curriculum  # noqa: E402
from backend.app.config import get_settings  # noqa: E402


def commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: dump_book.py OUTPUT.json")
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    body = {
        "commit": commit(),
        "gpu": settings.gpu_type,
        "chapters": curriculum.load_chapters(),
        "labs": curriculum.load_labs(),
    }
    out.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(body['chapters'])} chapters and {len(body['labs'])} labs to {out}")


if __name__ == "__main__":
    main()
