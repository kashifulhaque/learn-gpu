"""Writes every slide deck as one JSON file for the slides PDF build.

    python3 scripts/dump_slides.py book/.slides-build/slides.json

The renderer in `frontend/src/slides/` reads this file instead of the API, so
the PDF comes from the same parser the site serves, with no server running.
Decks with errors are left out, as the site leaves them out.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app import curriculum, decks  # noqa: E402


def commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: dump_slides.py OUTPUT.json")
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    found = decks.load_decks()
    entries = []
    for chapter in curriculum.load_chapters():
        deck = found.get(chapter["slug"])
        if deck is None:
            continue
        entries.append(
            {
                "slug": chapter["slug"],
                "chapter_title": chapter["title"],
                "part": chapter["part"],
                **decks.payload(deck),
            }
        )
    body = {"commit": commit(), "decks": entries}
    out.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    slides = sum(len(d["slides"]) for d in entries)
    print(f"Wrote {len(entries)} decks and {slides} slides to {out}")


if __name__ == "__main__":
    main()
