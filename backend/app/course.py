"""Course-wide facts shared by the REST API and the MCP server."""

from __future__ import annotations

import json
from typing import Any

from . import curriculum, db
from .config import Settings


def book_info(settings: Settings, manifest: str = "manifest.json") -> dict[str, Any] | None:
    """A PDF on disk, as its build's manifest describes it.

    Two PDFs can sit in `book/`: the course book (`manifest.json`) and the
    slide decks (`slides-manifest.json`). The Docker build prints both and
    copies them into the image, so a fresh checkout that hasn't run the build
    scripts has neither. Then this is None and the sidebar shows no download
    link, rather than one that leads to a 404.
    """
    try:
        data = json.loads((settings.book_dir / manifest).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    name = data.get("file")
    if not isinstance(name, str) or "/" in name or not (settings.book_dir / name).is_file():
        return None
    return {
        "title": data.get("title"),
        "pages": data.get("pages"),
        "bytes": data.get("bytes"),
        "built": data.get("built"),
        "commit": data.get("commit"),
        "url": f"/book/{name}",
    }


def progress_summary(user: str) -> dict[str, Any]:
    """How far the user is through the course, for the hub and the MCP server."""
    titles = {c["slug"]: c["title"] for c in curriculum.chapter_list()}
    rows = [r for r in db.get_progress_rows(user) if r["chapter"] in titles]
    last = rows[0] if rows else None  # Rows arrive newest first.
    return {
        "done": sum(1 for r in rows if r["status"] == "done"),
        "total": len(titles),
        "last": {"slug": last["chapter"], "title": titles[last["chapter"]]} if last else None,
    }
