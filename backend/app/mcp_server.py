"""An MCP server that gives AI agents the whole course.

The server speaks the MCP Streamable HTTP transport at `/mcp`, statelessly and
with plain JSON responses, so each request stands alone and needs no session.
It exposes chapters, labs, slide decks, the compute panel's state, the course
documents, the book's details, and the caller's progress, notes, drafts, and
runs as tools, and chapters and decks as resources.

Every request needs `Authorization: Bearer <token>`, where the token comes
from `mcp_tokens`. The token alone decides who the caller is. The SSO headers
that `auth.current_user` trusts are never read here, because Caddy routes
`/mcp` without Authelia and so can't vouch for them.
"""

from __future__ import annotations

import functools
import hmac
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Literal

import anyio
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import Receive, Scope, Send

from . import course, curriculum, db, decks, infra
from .config import REPO_ROOT, get_settings
from .mcp_tokens import user_for_token
from .providers import provider_status
from .runs import start_run

# The ASGI scope key that carries the authenticated reader to the tools.
USER_SCOPE_KEY = "learn_gpu.mcp_user"

COURSE_TITLE = "GPU programming for ML"
COURSE_DESCRIPTION = (
    "A hands-on course on GPU programming for machine learning, from the streaming "
    "multiprocessor to the kernels that train and serve models. Each chapter explains "
    "one topic in long form, links an auto-graded lab in CUDA C++, Triton, or CuTe DSL, "
    "and has a slide deck with speaker notes. Labs run on a real A100 through RunPod."
)

INSTRUCTIONS = (
    f"{COURSE_DESCRIPTION}\n\n"
    "Start with `course_overview` for the syllabus and the reader's progress. Use "
    "`search` to find where a topic is taught, `get_chapter` to read a chapter, and "
    "`get_lab` for a lab's brief, starter code, and checks. `get_deck` returns a "
    "chapter's slides with speaker notes. `run_lab` runs code as the reader and can "
    "start a paid GPU job, so call it only when the reader wants a run. Only call "
    "`get_solution` when the reader asks for the answer."
)

# Documents an agent may read with `get_document`. Nothing else is served.
# Each maps to its path relative to the repository root.
DOCUMENTS = {
    "CURRICULUM.md": ("docs/CURRICULUM.md", "The curriculum: every part and chapter, and why they are in that order."),
    "LABS.md": ("docs/LABS.md", "The contract every lab follows: files, harness, checks, and metrics."),
    "SLIDES.md": ("docs/SLIDES.md", "The slide deck format: headings, speaker notes, and figures."),
    "FACTS.md": ("docs/FACTS.md", "The cited fact sheet that the chapters' numbers come from."),
    "SOURCES.md": ("docs/SOURCES.md", "The sources the fact sheet cites."),
    "STYLE.md": ("content/STYLE.md", "The chapter style guide, including the accuracy rules."),
}

# Bounds that keep a tool result readable.
MAX_CODE_LENGTH = 200_000
MAX_NOTE_LENGTH = 50_000
MAX_LOG_CHARS = 16_000
MAX_CHECK_TEXT = 4_000
MAX_SEARCH_RESULTS = 50
MAX_RUNS = 100
DEFAULT_DOCUMENT_CHARS = 60_000
MAX_DOCUMENT_CHARS = 120_000

mcp = FastMCP(
    name="learn-gpu",
    instructions=INSTRUCTIONS,
    stateless_http=True,
    json_response=True,
    # The bearer token is the protection here. DNS-rebinding checks guard
    # unauthenticated servers on localhost, and would reject `gpu.ifkash.dev`.
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


# Plumbing ---------------------------------------------------------------


def _tool(*, writes: bool = False, open_world: bool = False) -> Callable[[Callable[..., Any]], Any]:
    """Register a tool that returns JSON text only.

    Structured output would send every result twice, as text and as an object,
    which doubles a chapter's size for no gain. `open_world` marks a tool that
    reaches a service outside this app, which here means RunPod.
    """
    return mcp.tool(
        structured_output=False,
        annotations=ToolAnnotations(
            readOnlyHint=not writes, destructiveHint=False, openWorldHint=open_world
        ),
    )


def _threaded(fn: Callable[..., Any]) -> Callable[..., Awaitable[Any]]:
    """Run a blocking tool body (SQLite, file reads) off the event loop."""

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))

    return wrapper


def _user(ctx: Context) -> str:
    request = ctx.request_context.request
    user = request.scope.get(USER_SCOPE_KEY) if request is not None else None
    if not user:
        # The endpoint authenticates before the server runs, so this means a
        # transport that bypassed it.
        raise ToolError("Not authenticated.")
    return str(user)


def _lab_or_error(lab_id: str) -> dict[str, Any]:
    lab = curriculum.get_lab(lab_id)
    if lab is None:
        raise ToolError(f"No lab with ID {lab_id!r}. Chapters name theirs in `lab`.")
    return lab


def _chapter_or_error(slug: str) -> dict[str, Any]:
    chapter = curriculum.get_chapter(slug)
    if chapter is None:
        raise ToolError(f"No chapter with slug {slug!r}. Call `list_chapters` for the slugs.")
    return chapter


def _chapters_for(lab_id: str) -> list[str]:
    return [c["slug"] for c in curriculum.chapter_list() if c.get("lab") == lab_id]


def _clip(text: str, limit: int) -> str:
    """Keep the start and the end of long output, where the useful parts are."""
    if len(text) <= limit:
        return text
    head = limit // 4
    tail = limit - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n... [{omitted} characters omitted] ...\n{text[-tail:]}"


def _chapter_summary(chapter: dict[str, Any], progress: dict[str, str]) -> dict[str, Any]:
    return {
        "slug": chapter["slug"],
        "title": chapter["title"],
        "part": chapter["part"],
        "summary": chapter["summary"],
        "minutes": chapter.get("minutes"),
        "lab": chapter.get("lab"),
        "gpu": chapter.get("gpu", False),
        "has_slides": decks.get_deck(chapter["slug"]) is not None,
        "status": progress.get(chapter["slug"]),
    }


_HEADING_LINE = re.compile(r"^(#{1,3})[ \t]+(.+?)[ \t#]*$", re.MULTILINE)


def _headings(body: str) -> list[str]:
    return [match.group(2).strip() for match in _HEADING_LINE.finditer(body)]


def _lab_summary(lab: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": lab["id"],
        "title": lab["title"],
        "gpu": lab["gpu"],
        "gpus": lab["gpus"],
        "needs_gpu": lab["needs_gpu"],
        "timeout": lab["timeout"],
        "chapters": _chapters_for(lab["id"]),
    }


def _lab_files(lab: dict[str, Any]) -> dict[str, Path]:
    """The extra files a lab's checks import, by name.

    These are the lab's own `fixture_*` files and the shared `lab_common.py`,
    and nothing else, so a name can never reach outside them.
    """
    directory = Path(lab["directory"])
    files = {path.name: path for path in sorted(directory.glob("fixture_*")) if path.is_file()}
    common = directory.parent / "lab_common.py"
    if common.is_file():
        files[common.name] = common
    return files


# Course and chapters ----------------------------------------------------


@_tool()
@_threaded
def course_overview(ctx: Context) -> dict[str, Any]:
    """Get the whole syllabus and the reader's progress in one call.

    Returns the course title and description; every part with its chapters
    (slug, title, summary, minutes, lab ID, whether it has slides, and the
    reader's status: null, "in_progress", or "done"); counts of chapters, labs,
    decks, and slides; the PDF book's and slides PDF's details; the GPU
    providers' availability; a progress summary; and the readable documents.
    Call this first to orient yourself, then `get_chapter` with a slug.
    """
    user = _user(ctx)
    settings = get_settings()
    progress = db.get_progress(user)
    chapters = curriculum.chapter_list()
    # Some chapters sit in a part out of filename order, so group by name
    # rather than by neighbour, as the sidebar does.
    grouped: dict[str, list[dict[str, Any]]] = {}
    for chapter in chapters:
        grouped.setdefault(chapter["part"], []).append(_chapter_summary(chapter, progress))
    parts = [{"part": part, "chapters": items} for part, items in grouped.items()]
    all_decks = decks.load_decks()
    labs = curriculum.load_labs()
    return {
        "title": COURSE_TITLE,
        "description": COURSE_DESCRIPTION,
        "reader": user,
        "gpu": settings.gpu_type,
        "parts": parts,
        "counts": {
            "parts": len(parts),
            "chapters": len(chapters),
            "labs": len(labs),
            "gpu_labs": sum(1 for lab in labs.values() if lab["needs_gpu"]),
            "decks": len(all_decks),
            "slides": sum(len(deck.slides) for deck in all_decks.values()),
        },
        "book": course.book_info(settings),
        "slides_pdf": course.book_info(settings, "slides-manifest.json"),
        "providers": provider_status(),
        "progress": course.progress_summary(user),
        "documents": {name: description for name, (_, description) in DOCUMENTS.items()},
    }


@_tool()
@_threaded
def list_chapters(ctx: Context, part: str | None = None) -> dict[str, Any]:
    """List chapters in course order, with their headings and objectives.

    Args:
        part: Only chapters whose part name contains this text, ignoring case,
            for example "Part 3" or "many GPUs". Omit it for every chapter.

    Each chapter has its slug, title, part, summary, minutes, objectives,
    `##`/`###` headings, lab ID, whether it has slides, and the reader's
    status. Bodies are omitted; call `get_chapter` for the full text.
    """
    progress = db.get_progress(_user(ctx))
    needle = (part or "").strip().lower()
    found = []
    for index, chapter in enumerate(curriculum.load_chapters()):
        if needle and needle not in str(chapter["part"]).lower():
            continue
        entry = _chapter_summary(chapter, progress)
        entry.update(
            index=index,
            objectives=chapter["objectives"],
            headings=_headings(chapter["body"]),
        )
        found.append(entry)
    return {"count": len(found), "chapters": found}


@_tool()
@_threaded
def get_chapter(ctx: Context, slug: str) -> dict[str, Any]:
    """Read one chapter: its full Markdown body and its front matter.

    Args:
        slug: The chapter's slug, such as "01-the-gpu-at-a-glance".

    Returns the title, part, summary, minutes, objectives, the Markdown `body`
    (math is LaTeX between `$` or `$$`; figures are ```viz fences that render
    in the browser only), the previous and next slugs, the lab ID and title,
    whether a deck exists, the reader's status, and the reader's note. Reading
    a chapter here doesn't mark it started.
    """
    chapter = _chapter_or_error(slug)
    user = _user(ctx)
    payload = dict(chapter)
    lab = curriculum.get_lab(str(chapter["lab"])) if chapter.get("lab") else None
    payload["lab_title"] = lab["title"] if lab else None
    payload["has_slides"] = decks.get_deck(slug) is not None
    payload["status"] = db.get_progress(user).get(slug)
    payload["note"] = db.get_note(user, slug)
    return payload


def _snippet(text: str, at: int, length: int) -> str:
    start = max(0, at - 80)
    end = min(len(text), at + length + 160)
    snippet = re.sub(r"\s+", " ", text[start:end]).strip()
    return ("..." if start > 0 else "") + snippet + ("..." if end < len(text) else "")


def _section(body: str, at: int) -> str | None:
    """The heading of the section that position `at` falls in."""
    heading = None
    for match in _HEADING_LINE.finditer(body):
        if match.start() > at:
            break
        heading = match.group(2).strip()
    return heading


def _search_documents() -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for chapter in curriculum.load_chapters():
        documents.append(
            {
                "kind": "chapter",
                "slug": chapter["slug"],
                "id": chapter["slug"],
                "title": str(chapter["title"]),
                "part": chapter["part"],
                "summary": str(chapter.get("summary", "")),
                "headings": " \n".join(_headings(chapter["body"])),
                "body": chapter["body"],
            }
        )
    for lab in curriculum.load_labs().values():
        documents.append(
            {
                "kind": "lab",
                "slug": lab["id"],
                "id": lab["id"],
                "title": lab["title"],
                "part": None,
                "summary": lab["brief"],
                "headings": "",
                "body": "\n".join([lab["brief"], *map(str, lab["hints"])]),
            }
        )
    for slug, deck in decks.load_decks().items():
        for slide in deck.slides:
            documents.append(
                {
                    "kind": "slide",
                    "slug": slug,
                    "id": slide.id,
                    "title": slide.title,
                    "part": slide.section,
                    "summary": deck.title,
                    "headings": "",
                    "body": "\n".join(part for part in (slide.body, slide.notes) if part),
                }
            )
    return documents


@_tool()
@_threaded
def search(query: str, limit: int = 10) -> dict[str, Any]:
    """Search chapters, labs, and slides by text, ignoring case.

    Args:
        query: Words to find, such as "bank conflict" or "online softmax". A
            document matches when it contains every word; an exact phrase match
            ranks higher, as do matches in titles and headings.
        limit: The most results to return, 1 to 50. Default 10.

    Each result has its kind ("chapter", "lab", or "slide"), the slug of the
    chapter or deck (a lab's ID for a lab), the slide's ID for a slide, the
    title, the section heading the hit falls under, and a short snippet. Pass a
    chapter slug to `get_chapter`, a lab ID to `get_lab`, and a deck slug to
    `get_deck`. Slide hits search the speaker notes too.
    """
    phrase = " ".join(query.lower().split())
    terms = phrase.split()
    if not terms:
        raise ToolError("The query is empty.")
    limit = max(1, min(int(limit), MAX_SEARCH_RESULTS))

    results = []
    for doc in _search_documents():
        title = doc["title"].lower()
        headings = doc["headings"].lower()
        body_lower = doc["body"].lower()
        haystack = " ".join((title, doc["summary"].lower(), headings, body_lower))
        if not all(term in haystack for term in terms):
            continue
        score = 0.0
        if phrase in title:
            score += 20
        score += 6 * sum(term in title for term in terms)
        if phrase in headings:
            score += 8
        score += 2 * sum(term in headings for term in terms)
        score += min(body_lower.count(phrase), 10) * 1.5
        score += sum(min(body_lower.count(term), 10) for term in terms) * 0.2

        at = body_lower.find(phrase)
        length = len(phrase)
        if at < 0:
            at = next((body_lower.find(t) for t in terms if t in body_lower), -1)
            length = len(terms[0])
        if at >= 0:
            snippet = _snippet(doc["body"], at, length)
            section = _section(doc["body"], at) if doc["kind"] == "chapter" else doc["part"]
        else:
            snippet = doc["summary"]
            section = None
        result = {
            "kind": doc["kind"],
            "slug": doc["slug"],
            "title": doc["title"],
            "section": section,
            "snippet": snippet,
            "score": round(score, 2),
        }
        if doc["kind"] == "slide":
            result["slide"] = doc["id"]
        results.append(result)
    results.sort(key=lambda result: result["score"], reverse=True)
    return {"query": query, "total": len(results), "results": results[:limit]}


# Labs -------------------------------------------------------------------


@_tool()
@_threaded
def list_labs() -> dict[str, Any]:
    """List every lab with the hardware it needs.

    Each lab has its ID, title, GPU model (`gpu`), GPU count (`gpus`), whether
    it needs a GPU at all, its timeout in seconds, and the chapters that link
    to it. A lab whose GPU endpoint isn't configured on this deployment still
    runs the checks that need no GPU. Call `get_lab` for the brief.
    """
    return {"labs": [_lab_summary(lab) for lab in curriculum.load_labs().values()]}


@_tool()
@_threaded
def get_lab(lab_id: str) -> dict[str, Any]:
    """Get a lab: its brief, hints, starter code, and the checks' source.

    Args:
        lab_id: The lab ID, which a chapter names in its `lab` field, such as
            "10-sgemm".

    Returns the title, brief, hints, the metrics the lab reports, `starter`
    (the file the reader edits), `tests` (the source of tests.py, the checks a
    run applies), the GPU model and count, the timeout in seconds, the chapters
    that link to it, and `files`, the names of the extra files the tests import.
    Read those with `get_lab_file`. The worked solution is withheld; see
    `get_solution`.
    """
    lab = _lab_or_error(lab_id)
    payload = curriculum.public_lab(lab)
    tests = Path(lab["directory"]) / "tests.py"
    payload["tests"] = tests.read_text(encoding="utf-8") if tests.is_file() else ""
    payload["files"] = list(_lab_files(lab))
    payload["chapters"] = _chapters_for(lab_id)
    return payload


@_tool()
@_threaded
def get_lab_file(lab_id: str, name: str) -> dict[str, Any]:
    """Read a file that a lab's checks import.

    Args:
        lab_id: The lab ID.
        name: A name from the `files` list that `get_lab` returns: a lab's own
            `fixture_*` file, such as a recorded profiler report, or
            "lab_common.py", the shared GPU helpers every lab's tests import.
    """
    files = _lab_files(_lab_or_error(lab_id))
    if name not in files:
        raise ToolError(f"Lab {lab_id!r} has no file {name!r}. Choose one of: {', '.join(files) or 'none'}.")
    return {"lab_id": lab_id, "name": name, "content": files[name].read_text(encoding="utf-8")}


@_tool()
@_threaded
def get_solution(lab_id: str) -> dict[str, Any]:
    """Get a lab's worked solution.

    Args:
        lab_id: The lab ID.

    The solution gives the answer away. Only fetch it when the reader asks to
    see it, or to compare after they have passed the lab.
    """
    lab = _lab_or_error(lab_id)
    return {"id": lab_id, "solution": lab["solution"]}


# Slides -----------------------------------------------------------------


@_tool()
@_threaded
def list_decks() -> dict[str, Any]:
    """List every chapter's slide deck, in course order.

    Each entry has the chapter's slug, title, and part, and its deck's title,
    subtitle, slide count, and section titles. `deck` is null for a chapter
    with no deck. Call `get_deck` for the slides.
    """
    have = decks.load_decks()
    entries = []
    for index, chapter in enumerate(curriculum.load_chapters()):
        deck = have.get(chapter["slug"])
        entries.append(
            {
                "slug": chapter["slug"],
                "chapter_title": chapter["title"],
                "part": chapter["part"],
                "order": index,
                "deck": decks.summary(deck) if deck else None,
            }
        )
    return {"decks": entries}


def _deck_or_error(slug: str) -> Any:
    deck = decks.get_deck(slug)
    if deck is None or curriculum.get_chapter(slug) is None:
        raise ToolError(f"Chapter {slug!r} doesn't exist or has no slide deck. Call `list_decks`.")
    return deck


@_tool()
@_threaded
def get_deck(slug: str) -> dict[str, Any]:
    """Get a chapter's slide deck: every slide's Markdown body and speaker notes.

    Args:
        slug: The chapter's slug, from `list_decks`.

    Returns the deck's title and subtitle, the chapter's title and part, the
    previous and next chapter slugs, and the slides in order. Each slide has
    an ID, a kind ("title", "section", or "slide"), a title, a Markdown `body`,
    the speaker `notes`, and its section and subsection.
    """
    deck = _deck_or_error(slug)
    chapter = _chapter_or_error(slug)
    return {
        **decks.payload(deck),
        "slug": slug,
        "chapter_title": chapter["title"],
        "part": chapter["part"],
        "prev": chapter["prev"],
        "next": chapter["next"],
    }


# Compute, documents, and the book ---------------------------------------


@_tool()
def get_providers() -> dict[str, Any]:
    """Get the GPU providers' configuration: which is the default and which can run labs.

    Returns the default provider, the preferred provider, and for each provider
    its name, whether it's available, and why not when it isn't. This reads
    this app's settings and calls no provider.
    """
    return provider_status()


@_tool(open_world=True)
async def get_compute_status() -> dict[str, Any]:
    """Get what is running on the GPU providers right now, as the Compute page shows it.

    Returns each provider's instances (pods and serverless endpoint workers)
    with their status, whether they're active and so billing, the GPU, the
    age, and a note; each provider's notices and facts, such as balance and
    recent spend; the count of active instances; and runs this app started and
    never saw finish, across all readers. Reading it is free and changes
    nothing. The tool offers no way to stop anything; the Compute page does.
    """
    return await infra.snapshot()


@_tool()
@_threaded
def get_document(name: str, offset: int = 0, limit: int = DEFAULT_DOCUMENT_CHARS) -> dict[str, Any]:
    """Read one of the course's reference documents, in Markdown.

    Args:
        name: One of the names that `course_overview` lists under `documents`:
            "CURRICULUM.md", "LABS.md", "SLIDES.md", "FACTS.md", "SOURCES.md",
            or "STYLE.md". Case and the ".md" are optional.
        offset: The character to start from. Default 0.
        limit: The most characters to return, up to 120000. Default 60000.

    Large documents, such as FACTS.md, come in pages. `total_chars` is the
    document's length, and `next_offset` is where the next page starts, or
    null on the last page.
    """
    wanted = name.strip().upper().removesuffix(".MD") + ".md"
    if wanted not in DOCUMENTS:
        raise ToolError(f"No document {name!r}. Choose one of: {', '.join(DOCUMENTS)}.")
    relative, description = DOCUMENTS[wanted]
    path = REPO_ROOT / relative
    if not path.is_file():
        raise ToolError(f"{wanted} isn't on this server.")
    text = path.read_text("utf-8")
    start = max(0, int(offset))
    end = start + max(1, min(int(limit), MAX_DOCUMENT_CHARS))
    return {
        "name": wanted,
        "description": description,
        "total_chars": len(text),
        "offset": start,
        "next_offset": end if end < len(text) else None,
        "markdown": text[start:end],
    }


@_tool()
@_threaded
def get_book_info() -> dict[str, Any]:
    """Get the PDF book's and the slides PDF's details.

    Each of `book` and `slides` has its title, page count, size in bytes, build
    time, commit, and `url` path, or is null when this server has no PDF. The
    PDFs are for readers to download in a browser, and the MCP server can't
    send them.
    """
    settings = get_settings()
    return {
        "book": course.book_info(settings),
        "slides": course.book_info(settings, "slides-manifest.json"),
    }


# The reader's state -----------------------------------------------------


@_tool()
@_threaded
def get_progress(ctx: Context) -> dict[str, Any]:
    """Get the reader's progress: per-chapter status, a summary, and recent runs.

    `chapters` lists every chapter the reader has started or finished, in
    course order, with its status ("in_progress" or "done"). `summary` has the
    done count, the chapter total, and the last chapter touched. `runs` has
    the 10 most recent runs, newest first.
    """
    user = _user(ctx)
    progress = db.get_progress(user)
    chapters = [
        {"slug": c["slug"], "title": c["title"], "status": progress[c["slug"]]}
        for c in curriculum.chapter_list()
        if c["slug"] in progress
    ]
    return {
        "reader": user,
        "summary": course.progress_summary(user),
        "chapters": chapters,
        "runs": db.list_runs(user, limit=10),
    }


@_tool(writes=True)
@_threaded
def set_progress(
    ctx: Context, chapter: str, status: Literal["in_progress", "done"]
) -> dict[str, Any]:
    """Mark a chapter in progress or done for the reader.

    Args:
        chapter: The chapter's slug.
        status: "in_progress" or "done". Unlike the lab pane, this can move a
            done chapter back to in progress. A passing `run_lab` marks the
            lab's chapters done on its own, so you rarely need this.
    """
    _chapter_or_error(chapter)
    user = _user(ctx)
    db.set_progress(user, chapter, status)
    return {"chapter": chapter, "status": db.get_progress(user)[chapter]}


@_tool()
@_threaded
def get_draft(ctx: Context, lab_id: str) -> dict[str, Any]:
    """Get the reader's saved code for a lab.

    Args:
        lab_id: The lab ID.

    `code` is null when the reader has no draft; the editor then shows the
    starter code from `get_lab`.
    """
    _lab_or_error(lab_id)
    return {"lab_id": lab_id, "code": db.get_draft(_user(ctx), lab_id)}


@_tool(writes=True)
@_threaded
def save_draft(ctx: Context, lab_id: str, code: str) -> dict[str, Any]:
    """Save code as the reader's draft for a lab.

    Args:
        lab_id: The lab ID.
        code: The whole file, which replaces the reader's current draft. The
            reader sees it in the editor the next time they open the chapter.
    """
    _lab_or_error(lab_id)
    if len(code) > MAX_CODE_LENGTH:
        raise ToolError(f"The code is longer than {MAX_CODE_LENGTH} characters.")
    db.save_draft(_user(ctx), lab_id, code)
    return {"lab_id": lab_id, "saved": True}


@_tool()
@_threaded
def get_note(ctx: Context, chapter: str) -> dict[str, Any]:
    """Get the reader's note on a chapter.

    Args:
        chapter: The chapter's slug.

    `note` is an empty string when the reader has written none.
    """
    _chapter_or_error(chapter)
    return {"chapter": chapter, "note": db.get_note(_user(ctx), chapter)}


@_tool(writes=True)
@_threaded
def save_note(ctx: Context, chapter: str, body: str) -> dict[str, Any]:
    """Save the reader's note on a chapter.

    Args:
        chapter: The chapter's slug.
        body: The whole note, which replaces the reader's current note. An
            empty string clears it.
    """
    _chapter_or_error(chapter)
    if len(body) > MAX_NOTE_LENGTH:
        raise ToolError(f"The note is longer than {MAX_NOTE_LENGTH} characters.")
    db.save_note(_user(ctx), chapter, body)
    return {"chapter": chapter, "saved": True}


@_tool()
@_threaded
def list_runs(ctx: Context, lab_id: str | None = None, limit: int = 20) -> dict[str, Any]:
    """List the reader's lab runs, newest first.

    Args:
        lab_id: Only this lab's runs. Omit it for every lab.
        limit: The most runs to return, 1 to 100. Default 20.

    Each run has its ID, lab, provider, status ("running", "finished",
    "error", "out_of_credits", or "cancelled"), whether it passed, its
    metrics, and when it started and finished (Unix seconds). Call `get_run`
    for a run's log.
    """
    if lab_id is not None:
        _lab_or_error(lab_id)
    limit = max(1, min(int(limit), MAX_RUNS))
    return {"runs": db.list_runs(_user(ctx), lab_id, limit=limit)}


@_tool()
@_threaded
def get_run(ctx: Context, run_id: str) -> dict[str, Any]:
    """Get one of the reader's runs, with its log and error.

    Args:
        run_id: A run ID from `list_runs` or `run_lab`.

    The log is trimmed in the middle when it is long.
    """
    run = db.get_run(_user(ctx), run_id)
    if run is None:
        raise ToolError(f"No run with ID {run_id!r}.")
    run["log"] = _clip(run.get("log") or "", MAX_LOG_CHARS)
    return run


@_tool(writes=True, open_world=True)
async def run_lab(
    ctx: Context,
    lab_id: str,
    code: str,
    provider: str | None = None,
) -> dict[str, Any]:
    """Run code for a lab on the course's GPU runner, as the reader.

    Args:
        lab_id: The lab ID.
        code: The whole submission file, starting from the starter code.
        provider: A provider name from `get_providers`. Omit it for the default.

    This does what the lab pane's Run button does, so it has the same cost. It
    saves the code as the reader's draft, records the run, and for a lab that
    needs a GPU it starts a serverless job on RunPod, which bills for the
    seconds a worker spends on it. A passing run marks the lab's chapters done.
    The call waits for the run to finish, which can take up to the lab's
    timeout plus two minutes. If the deployment has no endpoint for the lab's
    GPU, the result says so in `notices` and only the checks that need no GPU
    run. Call this only when the reader wants a run, and not in a loop.

    Returns `status` ("passed", "failed", "error", or "out_of_credits"),
    `passed`, `seconds`, the run ID, the provider, each check's name, result,
    and detail, the lab's metrics, the log (trimmed in the middle when long),
    any errors and notices, and the chapters marked done.
    """
    user = _user(ctx)
    if len(code) > MAX_CODE_LENGTH:
        raise ToolError(f"The code is longer than {MAX_CODE_LENGTH} characters.")
    try:
        run = await anyio.to_thread.run_sync(lambda: start_run(user, lab_id, code, provider))
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    if run is None:
        raise ToolError(f"No lab with ID {lab_id!r}.")

    log: list[str] = []
    checks: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    errors: list[str] = []
    notices: list[str] = []
    done: dict[str, Any] | None = None
    status = "error"
    async for event in run.events:
        kind = event.get("type")
        if kind == "log":
            log.append(str(event.get("line", "")))
        elif kind == "result":
            metrics = event.get("metrics") or {}
            checks = [
                {
                    "name": check.get("name"),
                    "passed": bool(check.get("passed")),
                    "detail": _clip(str(check.get("detail", "")), MAX_CHECK_TEXT),
                }
                for check in event.get("checks") or []
            ]
        elif kind == "unavailable":
            notices.append(str(event.get("message", "")))
        elif kind == "error":
            errors.append(str(event.get("message", "")))
        elif kind == "out_of_credits":
            status = "out_of_credits"
            errors.append(str(event.get("hint") or event.get("message", "")))
        elif kind == "done":
            done = event
            status = "passed" if event.get("passed") else "failed"

    marked: list[str] = []
    if done and done.get("passed"):
        marked = await anyio.to_thread.run_sync(_mark_done, user, lab_id)
    return {
        "lab_id": lab_id,
        "run_id": run.run_id,
        "provider": run.provider,
        "status": status,
        "passed": bool(done and done.get("passed")),
        "seconds": done.get("seconds") if done else None,
        "checks": checks,
        "metrics": metrics,
        "log": _clip("\n".join(log), MAX_LOG_CHARS),
        "errors": errors,
        "notices": notices,
        "chapters_done": marked,
    }


def _mark_done(user: str, lab_id: str) -> list[str]:
    slugs = _chapters_for(lab_id)
    for slug in slugs:
        db.set_progress(user, slug, "done")
    return slugs


# Resources --------------------------------------------------------------


@mcp.resource(
    "gpu://chapters/{slug}",
    name="chapter",
    description="A chapter's Markdown body. The slug comes from `list_chapters`.",
    mime_type="text/markdown",
)
@_threaded
def chapter_resource(slug: str) -> str:
    chapter = curriculum.get_chapter(slug)
    if chapter is None:
        raise ValueError(f"No chapter with slug {slug!r}.")
    return chapter["body"]


@mcp.resource(
    "gpu://slides/{slug}",
    name="deck",
    description="A chapter's slide deck as Markdown, speaker notes included. The slug comes from `list_decks`.",
    mime_type="text/markdown",
)
@_threaded
def deck_resource(slug: str) -> str:
    deck = decks.get_deck(slug)
    if deck is None:
        raise ValueError(f"Chapter {slug!r} has no slide deck.")
    lines = [f"# {deck.title}", ""]
    for slide in deck.slides:
        lines += [f"## {slide.title}", "", slide.body, ""]
        if slide.notes:
            lines += ["Notes:", "", slide.notes, ""]
    return "\n".join(lines)


# The HTTP endpoint ------------------------------------------------------

_manager: StreamableHTTPSessionManager | None = None


@asynccontextmanager
async def lifespan() -> AsyncIterator[None]:
    """Run the MCP session manager for the life of the app.

    A session manager runs only once, so each app start makes a new one. That
    lets tests start the app more than once in one process.
    """
    global _manager
    manager = StreamableHTTPSessionManager(
        app=mcp._mcp_server,
        json_response=True,
        stateless=True,
        security_settings=mcp.settings.transport_security,
    )
    async with manager.run():
        _manager = manager
        try:
            yield
        finally:
            _manager = None


def _unauthorized(error: str | None, description: str) -> JSONResponse:
    challenge = 'Bearer realm="learn-gpu-mcp"'
    if error:
        challenge += f', error="{error}"'
    return JSONResponse(
        {"error": error or "unauthorized", "error_description": description},
        status_code=401,
        headers={"WWW-Authenticate": challenge},
    )


class _Endpoint:
    """The ASGI app at `/mcp`: check the bearer token, then hand over to MCP.

    A class rather than a function, because Starlette wraps a plain function
    route as a request handler instead of mounting it as an ASGI app.
    """

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await _serve(scope, receive, send)


def _hub_user(headers: Headers) -> str | None:
    """The reader the MCP hub vouches for, or None without a valid hub secret.

    The hub sends its shared secret and the reader's Authelia username. The
    username maps through SSO_USERS, as a signed-in browser session does.
    Caddy strips both headers from public requests to /mcp.
    """
    settings = get_settings()
    secret = settings.mcp_hub_secret
    sent = headers.get("x-mcp-hub-secret", "")
    name = headers.get("x-mcp-hub-user", "").strip()
    if not secret or not sent or not name:
        return None
    if not hmac.compare_digest(sent.encode(), secret.encode()):
        return None
    return settings.sso_names.get(name.lower(), name)


async def _serve(scope: Scope, receive: Receive, send: Send) -> None:
    headers = Headers(scope=scope)
    hub_user = _hub_user(headers)
    if hub_user is not None:
        await _dispatch(hub_user, scope, receive, send)
        return
    scheme, _, token = headers.get("authorization", "").partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        response = _unauthorized(None, "Send an MCP token as 'Authorization: Bearer <token>'.")
        await response(scope, receive, send)
        return
    user = await anyio.to_thread.run_sync(user_for_token, token)
    if user is None:
        response = _unauthorized("invalid_token", "The token is unknown or revoked.")
        await response(scope, receive, send)
        return
    await _dispatch(user, scope, receive, send)


async def _dispatch(user: str, scope: Scope, receive: Receive, send: Send) -> None:
    if _manager is None:
        response = JSONResponse({"error": "The MCP server isn't running."}, status_code=503)
        await response(scope, receive, send)
        return
    await _manager.handle_request({**scope, USER_SCOPE_KEY: user}, receive, send)


endpoint = _Endpoint()
