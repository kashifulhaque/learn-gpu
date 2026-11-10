"""The progress summary endpoint and its path-scoped CORS headers."""

import time

from fastapi.testclient import TestClient

from backend.app import db
from backend.app.main import app

from .test_api import write_chapter

HUB = "https://courses.ifkash.dev"


def seed(content):
    for slug in ("00-intro", "01-next", "02-last"):
        write_chapter(content, slug)
    from .test_api import reload

    reload()


def test_summary_counts_and_last(client, content):
    seed(content)
    now = time.time()
    with db.connect() as conn:
        conn.executemany(
            "INSERT INTO progress (user, chapter, status, updated_at) VALUES (?,?,?,?)",
            [
                ("tester", "00-intro", "done", now - 30),
                ("tester", "01-next", "in_progress", now - 20),
                # Not in the current list, and newest: must be ignored.
                ("tester", "99-removed", "done", now),
                # Another user's row: must be ignored.
                ("someone", "02-last", "done", now),
            ],
        )
    reply = client.get("/api/progress/summary")
    assert reply.status_code == 200
    assert reply.json() == {
        "done": 1,
        "total": 3,
        "last": {"slug": "01-next", "title": "01-next"},
    }


def test_summary_with_no_progress(client, content):
    seed(content)
    assert client.get("/api/progress/summary").json() == {"done": 0, "total": 3, "last": None}


def test_signed_out_is_401_with_cors_for_the_hub(content):
    with TestClient(app) as anonymous:
        reply = anonymous.get("/api/progress/summary", headers={"Origin": HUB})
    assert reply.status_code == 401
    assert reply.headers["access-control-allow-origin"] == HUB
    assert reply.headers["access-control-allow-credentials"] == "true"
    assert "Origin" in reply.headers["vary"]


def test_signed_in_response_carries_cors_for_the_hub(client, content):
    seed(content)
    reply = client.get("/api/progress/summary", headers={"Origin": HUB})
    assert reply.headers["access-control-allow-origin"] == HUB
    assert reply.headers["access-control-allow-credentials"] == "true"


def test_other_origin_gets_no_allow_origin(client, content):
    seed(content)
    for origin in ("https://evil.example", HUB + ".evil.example", "http://courses.ifkash.dev"):
        reply = client.get("/api/progress/summary", headers={"Origin": origin})
        assert "access-control-allow-origin" not in reply.headers
        assert "access-control-allow-credentials" not in reply.headers
        assert "Origin" in reply.headers["vary"]


def test_other_paths_get_no_cors_headers(client, content):
    reply = client.get("/api/health", headers={"Origin": HUB})
    assert reply.status_code == 200
    assert "access-control-allow-origin" not in reply.headers
    assert "vary" not in reply.headers or "Origin" not in reply.headers["vary"]
