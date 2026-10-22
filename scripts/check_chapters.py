"""Checks each chapter's shape against content/STYLE.md.

    python3 scripts/check_chapters.py [file ...]

Reports callout markers the reader doesn't recognize, ```viz fences that name
no registered figure, front matter
that is missing or points at a lab that doesn't exist, and chapters missing the
parts every chapter has: one summary, one recap, the standard sections, and
questions written as callouts. Exits nonzero if anything fails. A chapter
without a slide deck is a warning, not a failure.
"""

import re
import sys
from pathlib import Path

import frontmatter

ROOT = Path(__file__).resolve().parent.parent
KINDS = {
    "TLDR", "KEY", "INTUITION", "EXAMPLE", "NOTE", "TIP", "WARNING",
    "DEEPDIVE", "QUESTION", "RECAP", "TRY", "IMPORTANT", "CAUTION",
}
REQUIRED_SECTIONS = ["Check your understanding", "Further reading"]
MARKER = re.compile(r"^((?:>[ \t]?)+)\[!([A-Za-z]+)\]")
FENCE = re.compile(r"^(?:>[ \t]?)*[ \t]*```")
VIZ_FENCE = re.compile(r"^(?:>[ \t]?)*[ \t]*```viz\s*$")
VIZ_DIR = ROOT / "frontend" / "src" / "components" / "viz"
LABS_DIR = ROOT / "content" / "labs"
SLIDES_DIR = ROOT / "content" / "slides"
REQUIRED_FRONT = ("title", "slug", "part", "summary", "minutes", "objectives")
# A registry key: a quoted name that starts with a chapter number.
VIZ_KEY = re.compile(r'^\s*"(\d\d[a-z]?-[a-z0-9-]+)"\s*:', re.MULTILINE)


def figure_names() -> set[str]:
    """The figure names registered in each chapter's viz/<NN>/index.ts."""
    names: set[str] = set()
    for index in VIZ_DIR.glob("[0-9][0-9]*/index.ts"):
        names.update(VIZ_KEY.findall(index.read_text()))
    return names


def check_front(path: Path) -> list[str]:
    """The chapter's front matter: required keys, a slug that matches the file, a real lab."""
    problems: list[str] = []
    meta = dict(frontmatter.load(path).metadata)
    for key in REQUIRED_FRONT:
        if not meta.get(key):
            problems.append(f"front matter has no {key}")
    if meta.get("slug") and meta["slug"] != path.stem:
        problems.append(f"slug {meta['slug']!r} must equal the file name {path.stem!r}")
    if not re.match(r"^\d\d[a-z]?-", path.stem):
        problems.append("the file name must start with the chapter number and a hyphen")
    for objective in meta.get("objectives") or []:
        if not isinstance(objective, str):
            problems.append(f"an objective is not a string; quote it: {objective!r}")
    lab = meta.get("lab")
    if lab and not (LABS_DIR / str(lab) / "meta.json").is_file():
        problems.append(f"lab {lab!r} has no content/labs/{lab}/meta.json")
    return problems


def check(path: Path, figures: set[str]) -> list[str]:
    problems: list[str] = check_front(path)
    counts: dict[str, int] = {}
    headings: list[str] = []
    in_fence = False
    in_viz = False
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if FENCE.match(line):
            in_viz = not in_fence and bool(VIZ_FENCE.match(line))
            in_fence = not in_fence
            continue
        if in_fence:
            if in_viz and line.strip():
                name = re.sub(r"^(?:>[ \t]?)*", "", line).strip()
                if name not in figures:
                    problems.append(f"line {line_no}: no figure is registered as {name!r}")
                in_viz = False
            continue
        if line.startswith("## "):
            headings.append(line[3:].strip())
        match = MARKER.match(line)
        if match:
            kind = match.group(2).upper()
            counts[kind] = counts.get(kind, 0) + 1
            if kind not in KINDS:
                problems.append(f"line {line_no}: unknown callout [!{match.group(2)}]")
            if kind == "DEEPDIVE" and not line[match.end():].strip():
                problems.append(f"line {line_no}: a deep dive needs a title")
            if kind == "QUESTION" and not line[match.end():].strip():
                problems.append(f"line {line_no}: a question callout needs the question as its title")
        elif re.match(r"^\s*\[![A-Za-z]+\]", line):
            problems.append(f"line {line_no}: callout marker without a leading '>'")
    if in_fence:
        problems.append("a code fence is never closed")
    for kind in ("TLDR", "RECAP"):
        if counts.get(kind, 0) != 1:
            problems.append(f"expected one [!{kind}], found {counts.get(kind, 0)}")
    for section in REQUIRED_SECTIONS:
        if section not in headings:
            problems.append(f"missing the '## {section}' section")
    if counts.get("QUESTION", 0) == 0:
        problems.append("no [!QUESTION] callouts")
    return problems


def main() -> int:
    files = [Path(arg) for arg in sys.argv[1:]] or sorted(
        (ROOT / "content" / "chapters").glob("*.md")
    )
    failed = 0
    figures = figure_names()
    for path in files:
        problems = check(path, figures)
        if not (SLIDES_DIR / f"{path.stem}.md").is_file():
            print(f"{path.name}: warning: no slide deck yet (content/slides/{path.stem}.md)")
        if problems:
            failed += 1
            for problem in problems:
                print(f"{path.relative_to(ROOT) if path.is_absolute() else path}: {problem}")
    print(f"{len(files)} chapter(s) checked, {failed} with problems")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
