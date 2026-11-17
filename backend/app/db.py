"""SQLite storage for progress, saved code, notes, lab runs, and MCP access tokens."""

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

CREATE TABLE IF NOT EXISTS mcp_tokens (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    username     TEXT NOT NULL,
    name         TEXT NOT NULL,
    token_hash   TEXT NOT NULL UNIQUE,
    prefix       TEXT NOT NULL,
    created_at   REAL NOT NULL,
    last_used_at REAL,
    revoked      INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS mcp_tokens_by_user ON mcp_tokens (username, revoked);
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


def get_progress_rows(user: str) -> list[dict[str, Any]]:
    """Every progress row for a user, newest `updated_at` first."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT chapter, status, updated_at FROM progress WHERE user=? "
            "ORDER BY updated_at DESC",
            (user,),
        ).fetchall()
    return [dict(r) for r in rows]


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


# --- MCP tokens -------------------------------------------------------------
#
# Only the SHA-256 of a token is stored. The secret itself is shown once, when
# it is created, by `mcp_tokens.create_token`.

# Writing `last_used_at` on every request would turn each MCP call into a
# write, so it is refreshed at most once per this many seconds.
TOKEN_TOUCH_INTERVAL = 60.0


def insert_token(username: str, name: str, token_hash: str, prefix: str) -> dict[str, Any]:
    created = time.time()
    with connect() as conn:
        cursor = conn.execute(
            "INSERT INTO mcp_tokens (username, name, token_hash, prefix, created_at) "
            "VALUES (?,?,?,?,?)",
            (username, name, token_hash, prefix, created),
        )
        token_id = cursor.lastrowid
    return {"id": token_id, "name": name, "prefix": prefix, "created_at": created}


def list_tokens(username: str | None = None) -> list[dict[str, Any]]:
    """Active tokens, oldest first: one reader's, or everyone's when `username` is None."""
    query = (
        "SELECT id, username, name, prefix, created_at, last_used_at "
        "FROM mcp_tokens WHERE revoked=0"
    )
    params: tuple[Any, ...] = ()
    if username is not None:
        query += " AND username=?"
        params = (username,)
    with connect() as conn:
        rows = conn.execute(query + " ORDER BY id", params).fetchall()
    return [dict(row) for row in rows]


def count_tokens(username: str) -> int:
    with connect() as conn:
        return conn.execute(
            "SELECT COUNT(*) AS n FROM mcp_tokens WHERE username=? AND revoked=0",
            (username,),
        ).fetchone()["n"]


def revoke_token(token_id: int, username: str | None = None) -> bool:
    """Revoke one active token. With `username`, only that reader's own.

    Returns False when no matching active token exists.
    """
    query = "UPDATE mcp_tokens SET revoked=1 WHERE id=? AND revoked=0"
    params: tuple[Any, ...] = (token_id,)
    if username is not None:
        query += " AND username=?"
        params = (token_id, username)
    with connect() as conn:
        return conn.execute(query, params).rowcount > 0


def token_user(token_hash: str) -> str | None:
    """The username an active token belongs to, refreshing its last use."""
    now = time.time()
    with connect() as conn:
        row = conn.execute(
            "SELECT id, username, last_used_at FROM mcp_tokens "
            "WHERE token_hash=? AND revoked=0",
            (token_hash,),
        ).fetchone()
        if row is None:
            return None
        if row["last_used_at"] is None or now - row["last_used_at"] >= TOKEN_TOUCH_INTERVAL:
            conn.execute("UPDATE mcp_tokens SET last_used_at=? WHERE id=?", (now, row["id"]))
    return row["username"]
