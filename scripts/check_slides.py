#!/usr/bin/env python3
"""Validates slide decks against `docs/SLIDES.md`.

Run it over one chapter while you write, and over everything in CI:

    python3 scripts/check_slides.py 03-memory-and-bandwidth
    python3 scripts/check_slides.py

It parses each deck with the same parser the site uses, then checks the
structure (an Introduction first and a Summary last), the slide count, how much
each slide holds, the speaker notes, the anticipated questions, the diagrams
and the figure names, and the mathematics. It exits non-zero when anything
fails. Mermaid syntax is checked by `npm run check:mermaid` in `frontend/`,
which needs Node.

A chapter without a deck is a warning, since decks are written after chapters.
A deck whose chapter doesn't exist is an error.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.app.decks import parse  # noqa: E402

SLIDES = ROOT / "content" / "slides"
CHAPTERS = ROOT / "content" / "chapters"
VIZ_DIR = ROOT / "frontend" / "src" / "components" / "viz"

SLIDE_RANGE = (22, 40)  # every slide the deck shows, the title and dividers included
MIN_QUESTIONS = 6
MIN_DIAGRAMS = 3
MIN_SECTIONS = 4  # Introduction, Summary, and at least two between

MAX_WORDS = 80  # prose on one slide, with mathematics and fences removed
MAX_WORDS_WITH_FIGURE = 50  # prose on a slide that also holds a diagram or figure
MAX_ITEMS = 6  # list items on one slide
MAX_DISPLAY = 2  # display formulas on one slide
MAX_LINES = 16  # non-blank source lines in one slide's body, fences counted as 4
NOTES_RANGE = (20, 220)  # words of speaker notes on a content slide

MERMAID_KINDS = ("flowchart", "graph", "xychart-beta", "pie", "timeline", "quadrantChart")
CALLOUT_KINDS = {"KEY", "TIP", "WARNING", "QUESTION"}

FENCE = re.compile(r"^[ \t]*(```|~~~)[^\n]*\n.*?^[ \t]*\1[ \t]*$", re.DOTALL | re.MULTILINE)
MERMAID = re.compile(r"^```mermaid[ \t]*\n(.*?)^```", re.DOTALL | re.MULTILINE)
VIZ = re.compile(r"^```viz[ \t]*\n(.*?)^```", re.DOTALL | re.MULTILINE)
DISPLAY_MATH = re.compile(r"\$\$.+?\$\$", re.DOTALL)
INLINE_MATH = re.compile(r"(?<!\$)\$[^$\n]+\$(?!\$)")
INLINE_CODE = re.compile(r"`[^`\n]*`")
CALLOUT = re.compile(r"^>[ \t]*\[!(\w+)\][ \t]*(.*)$", re.MULTILINE)
ITEM = re.compile(r"^[ \t>]*([-*+]|\d+[.)])[ \t]", re.MULTILINE)
TABLE_RULE = re.compile(r"^[ \t]*\|?[\s:|-]+\|[\s:|-]*$", re.MULTILINE)
WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’_-]*")
VIZ_KEY = re.compile(r'^\s*"(\d\d[a-z]?-[a-z0-9-]+)"\s*:', re.MULTILINE)
MATH_FAILURE = re.compile(r"^(?P<where>[^\s:][^:]*):(?P<line>\d+): (?P<message>.*)$")


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, where: str, message: str) -> None:
        self.errors.append(f"{where}: {message}")

    def warn(self, where: str, message: str) -> None:
        self.warnings.append(f"{where}: {message}")


def prose_words(text: str) -> int:
    """Words a reader reads on a slide: fences, mathematics, and code excluded."""
    text = FENCE.sub(" ", text)
    text = DISPLAY_MATH.sub(" ", text)
    text = INLINE_MATH.sub(" ", text)
    text = INLINE_CODE.sub(" ", text)
    text = re.sub(r"^[ \t>]*\[!\w+\]", " ", text, flags=re.MULTILINE)
    text = re.sub(r"https?://\S+", " ", text)
    return len(WORD.findall(text))


def body_lines(body: str) -> int:
    lines = len([line for line in FENCE.sub("", body).split("\n") if line.strip()])
    return lines + 4 * len(FENCE.findall(body))


def figure_names() -> set[str]:
    """The figure names registered in each chapter's viz/<NN>/index.ts."""
    names: set[str] = set()
    for index in VIZ_DIR.glob("[0-9][0-9]*/index.ts"):
        names.update(VIZ_KEY.findall(index.read_text(encoding="utf-8")))
    return names


def check_deck(report: Report, slug: str, path: Path, figures: set[str]) -> None:
    where = slug
    deck = parse(path.read_text(encoding="utf-8"))
    for error in deck.errors:
        report.error(where, error)

    low, high = SLIDE_RANGE
    if not low <= len(deck.slides) <= high:
        report.error(where, f"{len(deck.slides)} slides, and a deck has {low} to {high}")

    sections = [s for s in deck.slides if s.kind == "section"]
    if not sections or sections[0].title != "Introduction":
        report.error(where, "the first # section must be 'Introduction'")
    if not sections or not sections[-1].title.startswith("Summary"):
        report.error(where, "the last # section must start with 'Summary'")
    if len(sections) < MIN_SECTIONS:
        report.error(
            where,
            f"{len(sections)} sections; a deck needs Introduction, Summary, and at least two between",
        )
    if not deck.subtitle:
        report.warn(where, "the front matter has no subtitle")

    questions = 0
    diagrams = 0
    for slide in deck.slides:
        at = f"{where} line {slide.line} ({slide.id})"
        body = slide.body
        blocks = MERMAID.findall(body)
        names = [block.strip() for block in VIZ.findall(body)]
        figures_here = len(blocks) + len(names)
        diagrams += figures_here

        for match in CALLOUT.finditer(body):
            kind, title = match.group(1).upper(), match.group(2).strip()
            if kind not in CALLOUT_KINDS:
                report.error(at, f"unknown callout [!{match.group(1)}]; a slide may use {', '.join(sorted(CALLOUT_KINDS))}")
            if kind == "QUESTION":
                questions += 1
                if not title:
                    report.error(at, "a [!QUESTION] callout needs the question as its title")

        words = prose_words(TABLE_RULE.sub("", body).replace("|", " "))
        limit = MAX_WORDS_WITH_FIGURE if figures_here else MAX_WORDS
        if words > limit:
            report.error(at, f"{words} words on the slide, over {limit}; split it")
        items = len(ITEM.findall(FENCE.sub("", body)))
        if items > MAX_ITEMS:
            report.error(at, f"{items} list items, over {MAX_ITEMS}; split it")
        displays = len(DISPLAY_MATH.findall(body))
        if displays > MAX_DISPLAY:
            report.error(at, f"{displays} display formulas, over {MAX_DISPLAY}; split it")
        if body_lines(body) > MAX_LINES:
            report.error(at, f"the body runs to {body_lines(body)} lines, over {MAX_LINES}; split it")
        if figures_here > 1:
            report.error(at, "more than one diagram or figure on one slide")

        for block in blocks:
            first = block.strip().split()[0] if block.strip() else ""
            if first not in MERMAID_KINDS:
                report.error(at, f"mermaid diagram type {first!r} isn't allowed; use one of {', '.join(MERMAID_KINDS)}")
            if "$" in block:
                report.error(at, "a mermaid diagram can't hold $ mathematics; use Unicode symbols")
        for name in names:
            if name not in figures:
                report.error(at, f"no figure is registered as {name!r} (see frontend/src/components/viz/<NN>/index.ts)")

        if slide.kind == "slide":
            if not body.strip():
                report.error(at, "the slide has no body")
            count = len(slide.notes.split())
            if not NOTES_RANGE[0] <= count <= NOTES_RANGE[1]:
                report.error(at, f"{count} words of notes; a slide has {NOTES_RANGE[0]} to {NOTES_RANGE[1]}")
        elif not slide.notes.strip():
            report.error(at, "the slide has no speaker notes")
        if re.search(r"^#{4,} ", FENCE.sub("", body), re.MULTILINE):
            report.error(at, "a #### heading inside a slide; use bold text")

    if questions < MIN_QUESTIONS:
        report.error(where, f"{questions} [!QUESTION] callouts; a deck needs at least {MIN_QUESTIONS}")
    if diagrams < MIN_DIAGRAMS:
        report.error(where, f"{diagrams} diagrams or figures; a deck needs at least {MIN_DIAGRAMS}")


def check_math(report: Report, paths: list[Path]) -> bool:
    """Renders every formula with the site's KaTeX, through scripts/check_math.mjs."""
    node = shutil.which("node")
    if node is None or not (ROOT / "frontend" / "node_modules" / "katex").is_dir():
        report.warn("math", "skipped: Node, or frontend/node_modules, is missing. Run `npm install` in frontend/.")
        return False
    result = subprocess.run(
        [node, str(ROOT / "scripts" / "check_math.mjs"), *map(str, paths)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    for line in result.stderr.splitlines():
        match = MATH_FAILURE.match(line)
        if match:
            report.error(match.group("where"), f"line {match.group('line')}: {match.group('message')}")
    if result.returncode != 0 and not result.stderr.strip():
        report.error("math", "check_math.mjs failed: " + result.stdout.strip())
    return True


def main() -> int:
    wanted = [Path(arg).stem if arg.endswith(".md") else arg for arg in sys.argv[1:]]
    report = Report()
    chapters = {p.stem for p in CHAPTERS.glob("*.md")} if CHAPTERS.is_dir() else set()
    decks = {p.stem: p for p in sorted(SLIDES.glob("*.md"))} if SLIDES.is_dir() else {}

    for slug in sorted(decks):
        if slug not in chapters:
            report.error(slug, "there is a deck, but no chapter content/chapters/%s.md" % slug)
    for slug in sorted(chapters):
        if slug not in decks and (not wanted or slug in wanted):
            report.warn(slug, "no deck yet")
    for slug in wanted:
        if slug not in decks and slug not in chapters:
            report.error(slug, "no such chapter or deck")

    targets = [slug for slug in sorted(decks) if not wanted or slug in wanted]
    figures = figure_names()
    for slug in targets:
        check_deck(report, slug, decks[slug], figures)
    if targets:
        check_math(report, [decks[slug] for slug in targets])

    for warning in report.warnings:
        print(f"warn  {warning}")
    for error in report.errors:
        print(f"FAIL  {error}")
    if report.errors:
        print(f"\n{len(report.errors)} problems across {len(targets)} decks.")
        return 1
    print(f"\n{len(targets)} decks pass, with {len(report.warnings)} warnings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
