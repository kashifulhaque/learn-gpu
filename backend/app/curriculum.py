"""Loads chapters and labs from the content/ directory.

A chapter is a markdown file with YAML front matter. A lab is a directory under
content/labs/<id>/ holding meta.json, starter.py, tests.py, and solution.py.
Content is cached after the first read; set reload=True to pick up edits during
development.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import frontmatter

from .config import get_settings


def _chapters_dir() -> Path:
    return get_settings().content_dir / "chapters"


def _labs_dir() -> Path:
    return get_settings().content_dir / "labs"


@lru_cache
def load_chapters() -> list[dict[str, Any]]:
    """Every chapter, ordered by the numeric prefix of its filename."""
    chapters: list[dict[str, Any]] = []
    for path in sorted(_chapters_dir().glob("*.md")):
        post = frontmatter.load(path)
        meta = dict(post.metadata)
        slug = meta.get("slug") or path.stem
        chapters.append(
            {
                "slug": slug,
                "title": meta.get("title", slug),
                "part": meta.get("part", "Uncategorised"),
                "summary": meta.get("summary", ""),
                "minutes": meta.get("minutes"),
                "objectives": meta.get("objectives", []),
                "lab": meta.get("lab"),
                "gpu": meta.get("gpu", False),
                "body": post.content,
            }
        )
    return chapters


@lru_cache
def chapter_index() -> dict[str, dict[str, Any]]:
    return {c["slug"]: c for c in load_chapters()}


def chapter_list() -> list[dict[str, Any]]:
    """Chapter metadata without the body, for the sidebar and dashboard."""
    return [{k: v for k, v in c.items() if k != "body"} for c in load_chapters()]


def get_chapter(slug: str) -> dict[str, Any] | None:
    chapters = load_chapters()
    index = {c["slug"]: i for i, c in enumerate(chapters)}
    if slug not in index:
        return None
    i = index[slug]
    chapter = dict(chapters[i])
    chapter["prev"] = chapters[i - 1]["slug"] if i > 0 else None
    chapter["next"] = chapters[i + 1]["slug"] if i < len(chapters) - 1 else None
    return chapter


@lru_cache
def load_labs() -> dict[str, dict[str, Any]]:
    labs: dict[str, dict[str, Any]] = {}
    root = _labs_dir()
    if not root.is_dir():
        return labs
    for path in sorted(root.iterdir()):
        meta_file = path / "meta.json"
        if not meta_file.is_file():
            continue
        meta = json.loads(meta_file.read_text())
        lab_id = meta.get("id", path.name)
        labs[lab_id] = {
            "id": lab_id,
            "title": meta.get("title", lab_id),
            "brief": meta.get("brief", ""),
            "gpu": meta.get("gpu", "A100-80GB"),
            "needs_gpu": meta.get("needs_gpu", True),
            "timeout": meta.get("timeout", 900),
            "hints": meta.get("hints", []),
            "metrics": meta.get("metrics", []),
            "starter": (path / "starter.py").read_text()
            if (path / "starter.py").is_file()
            else "",
            "solution": (path / "solution.py").read_text()
            if (path / "solution.py").is_file()
            else "",
        }
    return labs


def get_lab(lab_id: str) -> dict[str, Any] | None:
    return load_labs().get(lab_id)


def public_lab(lab: dict[str, Any]) -> dict[str, Any]:
    """Lab payload for the client, with the solution withheld."""
    return {k: v for k, v in lab.items() if k != "solution"}


def reload_content() -> None:
    load_chapters.cache_clear()
    chapter_index.cache_clear()
    load_labs.cache_clear()
