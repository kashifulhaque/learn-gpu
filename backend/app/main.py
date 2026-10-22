"""learn-gpu — HTTP API and static file server.

Everything under `/api` needs a session cookie except `/api/health`,
`/api/me`, `/api/login`, and `/api/logout`, and so does the PDF book under
`/book`. Everything else is the single-page app.
"""

import asyncio
import json
import os
import time
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import curriculum, db, decks, infra
from .auth import clear_session, current_user, issue_session, read_session, verify_password
from .config import Settings, get_settings
from .providers import OutOfCredits, ProviderError, get_provider, provider_status

app = FastAPI(title="learn-gpu", docs_url=None, redoc_url=None)


@app.on_event("startup")
def _startup() -> None:
    db.init_db()
    # Read the slide decks now, so a bad one is logged at startup rather than
    # on the first deck someone opens.
    decks.load_decks()


# --- schemas ----------------------------------------------------------------


class LoginBody(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    password: str


class ProgressBody(BaseModel):
    chapter: str
    status: str = Field(pattern="^(in_progress|done)$")


class DraftBody(BaseModel):
    lab: str
    code: str


class NoteBody(BaseModel):
    chapter: str
    body: str


class RunBody(BaseModel):
    lab: str
    code: str
    provider: str | None = None


class ActionBody(BaseModel):
    provider: str
    action: str
    target: str = ""


# --- auth -------------------------------------------------------------------


@app.post("/api/login")
def login(
    body: LoginBody, response: Response, settings: Settings = Depends(get_settings)
) -> dict[str, Any]:
    if not verify_password(body.password, settings):
        raise HTTPException(status_code=401, detail="Wrong password")
    issue_session(response, body.name.strip(), settings)
    return {"name": body.name.strip()}


@app.post("/api/logout")
def logout(response: Response, settings: Settings = Depends(get_settings)) -> dict[str, bool]:
    clear_session(response, settings)
    return {"ok": True}


@app.get("/api/me")
def me(request: Request, settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    name = read_session(request, settings)
    return {
        "name": name,
        "gpu": settings.gpu_type,
    }


# --- content ----------------------------------------------------------------


@app.get("/api/chapters")
def chapters(user: str = Depends(current_user)) -> dict[str, Any]:
    have = decks.load_decks()
    listing = [{**c, "has_slides": c["slug"] in have} for c in curriculum.chapter_list()]
    return {"chapters": listing, "progress": db.get_progress(user)}


@app.get("/api/chapters/{slug}")
def chapter(slug: str, user: str = Depends(current_user)) -> dict[str, Any]:
    found = curriculum.get_chapter(slug)
    if not found:
        raise HTTPException(status_code=404, detail="No such chapter")
    payload = dict(found)
    payload["note"] = db.get_note(user, slug)
    payload["has_slides"] = decks.get_deck(slug) is not None
    if found.get("lab"):
        lab = curriculum.get_lab(found["lab"])
        if lab:
            public = curriculum.public_lab(lab)
            public["draft"] = db.get_draft(user, lab["id"])
            payload["lab_detail"] = public
    return payload


@app.get("/api/slides")
def slides_index(user: str = Depends(current_user)) -> dict[str, Any]:
    """Every chapter, in course order, with its deck's summary or None."""
    have = decks.load_decks()
    out: list[dict[str, Any]] = []
    for index, chapter in enumerate(curriculum.load_chapters()):
        deck = have.get(chapter["slug"])
        out.append(
            {
                "slug": chapter["slug"],
                "chapter_title": chapter["title"],
                "part": chapter["part"],
                "order": index,
                "deck": decks.summary(deck) if deck else None,
            }
        )
    return {"decks": out}


@app.get("/api/slides/{slug}")
def one_deck(slug: str, user: str = Depends(current_user)) -> dict[str, Any]:
    """One chapter's deck: its slides, speaker notes included."""
    chapter = curriculum.get_chapter(slug)
    deck = decks.get_deck(slug)
    if chapter is None or deck is None:
        raise HTTPException(status_code=404, detail="No slides for this chapter")
    return {
        **decks.payload(deck),
        "slug": slug,
        "chapter_title": chapter["title"],
        "part": chapter["part"],
        "prev": chapter["prev"],
        "next": chapter["next"],
    }


@app.get("/api/labs/{lab_id}")
def lab(lab_id: str, user: str = Depends(current_user)) -> dict[str, Any]:
    found = curriculum.get_lab(lab_id)
    if not found:
        raise HTTPException(status_code=404, detail="No such lab")
    payload = curriculum.public_lab(found)
    payload["draft"] = db.get_draft(user, lab_id)
    payload["runs"] = db.list_runs(user, lab_id, limit=20)
    return payload


@app.get("/api/labs/{lab_id}/solution")
def lab_solution(lab_id: str, user: str = Depends(current_user)) -> dict[str, str]:
    """The worked solution. Deliberately behind its own click in the UI."""
    found = curriculum.get_lab(lab_id)
    if not found:
        raise HTTPException(status_code=404, detail="No such lab")
    return {"solution": found["solution"]}


def book_info(settings: Settings, manifest: str = "manifest.json") -> dict[str, Any] | None:
    """A PDF on disk, as its build's manifest describes it.

    Two PDFs can sit in `book/`: the course book (`manifest.json`) and the
    slide decks (`slides-manifest.json`). They are built on the deploying
    machine and copied beside the app, so a fresh checkout has neither. Then
    this is None and the sidebar shows no download link, rather than one that
    leads to a 404.
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


@app.get("/api/book")
def book(
    user: str = Depends(current_user), settings: Settings = Depends(get_settings)
) -> dict[str, Any]:
    """The course book and the slide deck PDF; either is None when this server has no PDF."""
    return {
        "book": book_info(settings),
        "slides": book_info(settings, "slides-manifest.json"),
    }


@app.get("/book/{name}")
def book_file(
    name: str, user: str = Depends(current_user), settings: Settings = Depends(get_settings)
) -> FileResponse:
    """A PDF, sent as a download. Only a file a manifest names is served."""
    for manifest in ("manifest.json", "slides-manifest.json"):
        info = book_info(settings, manifest)
        if info is not None and info["url"] == f"/book/{name}":
            return FileResponse(settings.book_dir / name, media_type="application/pdf", filename=name)
    raise HTTPException(status_code=404, detail="No such book")


# --- user state -------------------------------------------------------------


@app.post("/api/progress")
def set_progress(body: ProgressBody, user: str = Depends(current_user)) -> dict[str, bool]:
    db.set_progress(user, body.chapter, body.status)
    return {"ok": True}


@app.post("/api/drafts")
def save_draft(body: DraftBody, user: str = Depends(current_user)) -> dict[str, bool]:
    db.save_draft(user, body.lab, body.code)
    return {"ok": True}


@app.post("/api/notes")
def save_note(body: NoteBody, user: str = Depends(current_user)) -> dict[str, bool]:
    db.save_note(user, body.chapter, body.body)
    return {"ok": True}


@app.get("/api/runs")
def runs(
    lab: str | None = None, user: str = Depends(current_user)
) -> dict[str, Any]:
    return {"runs": db.list_runs(user, lab)}


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str, user: str = Depends(current_user)) -> dict[str, Any]:
    found = db.get_run(user, run_id)
    if not found:
        raise HTTPException(status_code=404, detail="No such run")
    return found


@app.get("/api/providers")
def providers(user: str = Depends(current_user)) -> dict[str, Any]:
    return provider_status()


# --- compute panel ----------------------------------------------------------


@app.get("/api/infra")
async def infra_snapshot(user: str = Depends(current_user)) -> dict[str, Any]:
    """What is running on every configured provider right now."""
    return await infra.snapshot()


@app.post("/api/infra/action")
async def infra_action(
    body: ActionBody, user: str = Depends(current_user)
) -> dict[str, Any]:
    """Stop a container, pod, or job. Never deletes an endpoint or a volume."""
    try:
        return await infra.act(body.provider, body.action, body.target)
    except ProviderError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - the provider's message is the answer
        raise HTTPException(status_code=502, detail=str(exc)) from exc


# --- lab execution ----------------------------------------------------------


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event)}\n\n"


@app.post("/api/run")
async def run_lab(body: RunBody, user: str = Depends(current_user)) -> StreamingResponse:
    lab = curriculum.get_lab(body.lab)
    if not lab:
        raise HTTPException(status_code=404, detail="No such lab")

    db.save_draft(user, body.lab, body.code)
    provider = get_provider(body.provider)
    run_id = db.create_run(user, body.lab, provider.name)

    async def stream() -> Any:
        log_lines: list[str] = []
        result: dict[str, Any] | None = None
        stream_error: str | None = None
        started = time.time()
        yield _sse({"type": "start", "run_id": run_id, "provider": provider.name})
        try:
            async for event in provider.run_lab(
                lab_id=body.lab,
                code=body.code,
                gpu=lab["gpu"],
                timeout=lab["timeout"],
            ):
                kind = event.get("type")
                if kind == "job":
                    # The provider's own id for this job. Recorded so the
                    # compute panel can cancel a run whose tab has gone away.
                    db.set_run_job(run_id, event.get("job_id", ""))
                    continue
                if kind == "log":
                    log_lines.append(event.get("line", ""))
                elif kind == "result":
                    result = event
                elif kind == "error":
                    # The provider reports failures as events rather than
                    # exceptions, so record them as errors, not as a failed run.
                    stream_error = event.get("message", "")
                yield _sse(event)

            passed = bool(result.get("passed")) if result else False
            metrics = result.get("metrics", {}) if result else {}
            db.finish_run(
                run_id,
                status="error" if stream_error else "finished",
                passed=None if stream_error else passed,
                metrics=metrics,
                log="\n".join(log_lines),
                error=stream_error,
            )
            if stream_error:
                return
            yield _sse(
                {
                    "type": "done",
                    "run_id": run_id,
                    "passed": passed,
                    "seconds": round(time.time() - started, 2),
                }
            )
        except OutOfCredits as exc:
            db.finish_run(
                run_id, status="out_of_credits", log="\n".join(log_lines), error=str(exc)
            )
            hint = f"{provider.name} is out of credit. Add credit, then run again."
            yield _sse(
                {
                    "type": "out_of_credits",
                    "provider": provider.name,
                    "message": str(exc),
                    "hint": hint,
                }
            )
        except (ProviderError, Exception) as exc:  # noqa: BLE001
            db.finish_run(
                run_id, status="error", log="\n".join(log_lines), error=str(exc)
            )
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- health & static --------------------------------------------------------


@app.get("/api/health")
def health(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    return {
        "ok": True,
        "chapters": len(curriculum.chapter_list()),
        # Set at build time from the commit being deployed.
        "commit": os.environ.get("GIT_SHA", "unknown"),
        "book_pages": (book_info(settings) or {}).get("pages", 0),
        "slides": len(decks.load_decks()),
    }


def _mount_frontend() -> None:
    settings = get_settings()
    dist = settings.frontend_dist
    if not dist.is_dir():
        return

    assets = dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str) -> Any:
        if full_path.startswith("api/"):
            return JSONResponse({"detail": "Not found"}, status_code=404)
        candidate = dist / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(dist / "index.html")


_mount_frontend()
