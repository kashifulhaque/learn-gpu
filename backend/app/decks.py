"""Slide decks: one authored deck per chapter.

A deck lives at `content/slides/<slug>.md`, next to the chapter it teaches. It
is Markdown with a small amount of structure, described in full in
`docs/SLIDES.md`:

- YAML front matter with the deck's `title`, and optionally a `subtitle` and
  the opening slide's `notes`.
- `# Heading` starts a section. Each section opens with a divider slide, whose
  body is any text between the heading and the first slide.
- `## Heading` starts a subsection. It labels the slides under it and makes no
  slide of its own.
- `### Heading {#id}` starts one slide. The braces are optional: `id`
  defaults to the heading's slug.
- A line that reads `Note:` ends a slide's body. What follows, up to the next
  heading, is the speaker notes.

The parser is pure, so `scripts/check_slides.py` uses it too. It never raises
on a malformed deck: it records what is wrong in `errors` and returns what it
could read, and the site serves a deck only when it has no errors.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_FENCE_OPEN = re.compile(r"^\s*(```|~~~)")
_HEADING = re.compile(r"^(#{1,3}) (.+?)\s*$")
_ATTRS = re.compile(r"\s*\{([^{}]*)\}\s*$")
_SLUG_STRIP = re.compile(r"[`*_$\\]")
_SLUG_JOIN = re.compile(r"[^a-z0-9]+")
_SLUG_TRIM = re.compile(r"^-+|-+$")
_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")

log = logging.getLogger("learn-gpu")


def slugify(text: str) -> str:
    text = _SLUG_STRIP.sub("", text.lower())
    return _SLUG_TRIM.sub("", _SLUG_JOIN.sub("-", text))


@dataclass
class DeckSlide:
    id: str
    kind: str  # "title", "section", or "slide"
    title: str
    body: str = ""
    notes: str = ""
    section: str | None = None
    subsection: str | None = None
    line: int = 0


@dataclass
class Deck:
    title: str
    subtitle: str
    slides: list[DeckSlide] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _attrs(heading: str, line: int, errors: list[str]) -> tuple[str, dict[str, str]]:
    """Splits `Title {#id}` into the title and its attributes."""
    match = _ATTRS.search(heading)
    if not match:
        return heading, {}
    out: dict[str, str] = {}
    for token in match.group(1).split():
        if token.startswith("#"):
            out["id"] = token[1:]
        elif "=" in token:
            key = token.split("=", 1)[0]
            errors.append(f"line {line}: unknown slide attribute {key!r}; only #id is allowed")
        else:
            errors.append(f"line {line}: can't read {token!r} in the heading's braces")
    return heading[: match.start()].rstrip(), out


def parse(text: str) -> Deck:
    errors: list[str] = []
    front: dict[str, Any] = {}
    offset = 0
    match = _FRONT.match(text)
    if match:
        try:
            loaded = yaml.safe_load(match.group(1))
            front = loaded if isinstance(loaded, dict) else {}
        except yaml.YAMLError as exc:
            errors.append(f"the front matter isn't valid YAML: {exc}")
        offset = match.group(0).count("\n")
        text = text[match.end():]
    else:
        errors.append("the deck has no front matter")

    title = str(front.get("title") or "").strip()
    if not title:
        errors.append("the front matter has no title")
    subtitle = str(front.get("subtitle") or "").strip()
    deck = Deck(title=title, subtitle=subtitle, errors=errors)
    deck.slides.append(
        DeckSlide(
            id="title",
            kind="title",
            title=title,
            body=subtitle,
            notes=str(front.get("notes") or "").strip(),
            line=1,
        )
    )

    section: str | None = None
    subsection: str | None = None
    current: DeckSlide | None = None
    body: list[str] = []
    notes: list[str] = []
    in_notes = False
    fence: str | None = None

    def close() -> None:
        if current is not None:
            current.body = "\n".join(body).strip()
            current.notes = "\n".join(notes).strip()

    for number, line in enumerate(text.split("\n"), start=offset + 1):
        opener = _FENCE_OPEN.match(line)
        if fence is not None:
            if opener and opener.group(1) == fence:
                fence = None
            (notes if in_notes else body).append(line)
            continue
        if opener:
            fence = opener.group(1)
            (notes if in_notes else body).append(line)
            continue

        heading = _HEADING.match(line)
        if heading:
            level = len(heading.group(1))
            name, attrs = _attrs(heading.group(2), number, errors)
            close()
            body, notes, in_notes = [], [], False
            if level == 1:
                section, subsection = name, None
                current = DeckSlide(
                    id=f"section-{slugify(name)}",
                    kind="section",
                    title=name,
                    section=name,
                    line=number,
                )
                if attrs:
                    errors.append(f"line {number}: a section heading takes no attributes")
            elif level == 2:
                subsection = name
                current = None
                if attrs:
                    errors.append(f"line {number}: a subsection heading takes no attributes")
            else:
                if section is None:
                    errors.append(f"line {number}: slide {name!r} comes before any # section")
                current = DeckSlide(
                    id=attrs.get("id") or slugify(name),
                    kind="slide",
                    title=name,
                    section=section,
                    subsection=subsection,
                    line=number,
                )
            if current is not None:
                deck.slides.append(current)
            continue

        if line.strip() == "Note:":
            if current is None:
                errors.append(f"line {number}: notes outside a slide")
            elif in_notes:
                errors.append(f"line {number}: a second Note: in one slide")
            in_notes = True
            continue
        if current is None:
            if line.strip():
                errors.append(f"line {number}: text outside a slide (put it under a ### heading)")
            continue
        (notes if in_notes else body).append(line)
    close()
    if fence is not None:
        errors.append("a code fence is never closed")

    seen: set[str] = set()
    for slide in deck.slides:
        if not _ID.match(slide.id):
            errors.append(
                f"line {slide.line}: slide id {slide.id!r} must be lowercase letters, digits, and hyphens"
            )
        if slide.id in seen:
            errors.append(f"line {slide.line}: slide id {slide.id!r} is used twice")
        seen.add(slide.id)
    return deck


# Serving -------------------------------------------------------------------


def payload(deck: Deck) -> dict[str, Any]:
    """The deck as the presenter reads it."""
    return {
        "title": deck.title,
        "subtitle": deck.subtitle,
        "slides": [
            {
                "id": s.id,
                "kind": s.kind,
                "title": s.title,
                "body": s.body,
                "notes": s.notes,
                "section": s.section,
                "subsection": s.subsection,
            }
            for s in deck.slides
        ],
    }


def summary(deck: Deck) -> dict[str, Any]:
    """The deck without its slides, for the index."""
    sections = [s.title for s in deck.slides if s.kind == "section"]
    return {
        "title": deck.title,
        "subtitle": deck.subtitle,
        "slides": len(deck.slides),
        "sections": sections,
    }


# Loading -------------------------------------------------------------------


def _slides_dir() -> Path:
    from .config import get_settings

    return get_settings().content_dir / "slides"


@lru_cache(maxsize=1)
def load_decks() -> dict[str, Deck]:
    """Every deck that parses cleanly, keyed by chapter slug."""
    out: dict[str, Deck] = {}
    root = _slides_dir()
    if not root.is_dir():
        return out
    for path in sorted(root.glob("*.md")):
        deck = parse(path.read_text(encoding="utf-8"))
        if deck.errors:
            log.warning("Skipping deck %s: %s", path.stem, "; ".join(deck.errors[:3]))
            continue
        out[path.stem] = deck
    return out


def reload_decks() -> None:
    load_decks.cache_clear()


def get_deck(slug: str) -> Deck | None:
    return load_decks().get(slug)
