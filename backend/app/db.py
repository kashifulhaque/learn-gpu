"""SQLite storage for users, progress, saved code, and lab runs."""

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

from .config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS progress (
    user       TEXT NOT NULL,
    chapter    TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'in_progress',
    updated_at REAL NOT NULL,
    PRIMARY KEY (user, chapter)
);

CREATE TABLE IF NOT EXISTS drafts (
    user       TEXT NOT NULL,
    lab        TEXT NOT NULL,
    code       TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (user, lab)
);

CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    user        TEXT NOT NULL,
    lab         TEXT NOT NULL,
    provider    TEXT NOT NULL,
    status      TEXT NOT NULL,
    passed      INTEGER,
    metrics     TEXT,
    log         TEXT,
    error       TEXT,
    started_at  REAL NOT NULL,
    finished_at REAL
);

CREATE INDEX IF NOT EXISTS runs_user_lab ON runs (user, lab, started_at DESC);

CREATE TABLE IF NOT EXISTS notes (
    user       TEXT NOT NULL,
    chapter    TEXT NOT NULL,
    body       TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (user, chapter)
);
"""


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    settings = get_settings()
    conn = sqlite3.connect(settings.db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns that arrived after the first release."""
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(runs)")}
    if "job_id" not in columns:
        # The provider's own id for the job, so the compute panel can cancel a
        # run whose browser tab is gone.
        conn.execute("ALTER TABLE runs ADD COLUMN job_id TEXT")


# --- progress ---------------------------------------------------------------


def set_progress(user: str, chapter: str, status: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO progress (user, chapter, status, updated_at) VALUES (?,?,?,?) "
            "ON CONFLICT(user, chapter) DO UPDATE SET status=excluded.status, "
            "updated_at=excluded.updated_at",
            (user, chapter, status, time.time()),
        )


def get_progress(user: str) -> dict[str, str]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT chapter, status FROM progress WHERE user=?", (user,)
        ).fetchall()
    return {r["chapter"]: r["status"] for r in rows}


# --- drafts -----------------------------------------------------------------


def save_draft(user: str, lab: str, code: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO drafts (user, lab, code, updated_at) VALUES (?,?,?,?) "
            "ON CONFLICT(user, lab) DO UPDATE SET code=excluded.code, "
            "updated_at=excluded.updated_at",
            (user, lab, code, time.time()),
        )


def get_draft(user: str, lab: str) -> str | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT code FROM drafts WHERE user=? AND lab=?", (user, lab)
        ).fetchone()
    return row["code"] if row else None


# --- notes ------------------------------------------------------------------


def save_note(user: str, chapter: str, body: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO notes (user, chapter, body, updated_at) VALUES (?,?,?,?) "
            "ON CONFLICT(user, chapter) DO UPDATE SET body=excluded.body, "
            "updated_at=excluded.updated_at",
            (user, chapter, body, time.time()),
        )


def get_note(user: str, chapter: str) -> str:
    with connect() as conn:
        row = conn.execute(
            "SELECT body FROM notes WHERE user=? AND chapter=?", (user, chapter)
        ).fetchone()
    return row["body"] if row else ""


# --- runs -------------------------------------------------------------------


def create_run(user: str, lab: str, provider: str) -> str:
    run_id = uuid.uuid4().hex
    with connect() as conn:
        conn.execute(
            "INSERT INTO runs (id, user, lab, provider, status, started_at) "
            "VALUES (?,?,?,?,'running',?)",
            (run_id, user, lab, provider, time.time()),
        )
    return run_id


def finish_run(
    run_id: str,
    status: str,
    passed: bool | None = None,
    metrics: dict[str, Any] | None = None,
    log: str = "",
    error: str | None = None,
) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE runs SET status=?, passed=?, metrics=?, log=?, error=?, "
            "finished_at=? WHERE id=?",
            (
                status,
                None if passed is None else int(passed),
                json.dumps(metrics or {}),
                log[-200_000:],
                error,
                time.time(),
                run_id,
            ),
        )


def set_run_job(run_id: str, job_id: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE runs SET job_id=? WHERE id=?", (job_id, run_id))


def list_active_runs(max_age: float = 6 * 3600) -> list[dict[str, Any]]:
    """Runs that never reported finishing, newest first, across every user.

    The compute panel shows these next to the provider's own view: a run left
    as 'running' long after its lab timeout usually means the browser tab went
    away mid-stream, and the GPU may still be busy.
    """
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, user, lab, provider, status, job_id, started_at FROM runs "
            "WHERE status='running' AND started_at > ? ORDER BY started_at DESC",
            (time.time() - max_age,),
        ).fetchall()
    return [dict(row) for row in rows]


def _row_to_run(row: sqlite3.Row) -> dict[str, Any]:
    run = dict(row)
    run["metrics"] = json.loads(run["metrics"]) if run["metrics"] else {}
    if run["passed"] is not None:
        run["passed"] = bool(run["passed"])
    return run


def list_runs(user: str, lab: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    query = (
        "SELECT id, lab, provider, status, passed, metrics, started_at, finished_at "
        "FROM runs WHERE user=?"
    )
    params: list[Any] = [user]
    if lab:
        query += " AND lab=?"
        params.append(lab)
    query += " ORDER BY started_at DESC LIMIT ?"
    params.append(limit)
    with connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [_row_to_run(r) for r in rows]


def find_run(run_id: str) -> dict[str, Any] | None:
    """A run by id, whoever started it. The compute panel is not user-scoped."""
    with connect() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    return _row_to_run(row) if row else None


def get_run(user: str, run_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM runs WHERE id=? AND user=?", (run_id, user)
        ).fetchone()
    return _row_to_run(row) if row else None
