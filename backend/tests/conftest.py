"""Shared fixtures. Each test gets a throwaway data directory and, when it asks
for one, its own content directory, so tests never touch `data/` or each
other."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def content(tmp_path, monkeypatch):
    """A content directory a test can fill in: chapters/, labs/, slides/."""
    root = tmp_path / "content"
    for sub in ("chapters", "labs", "slides"):
        (root / sub).mkdir(parents=True)
    monkeypatch.setenv("CONTENT_DIR", str(root))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("FRONTEND_DIST", str(tmp_path / "no-dist"))
    monkeypatch.setenv("BOOK_DIR", str(tmp_path / "book"))
    _reset()
    yield root
    _reset()


def _reset():
    from backend.app import curriculum, decks
    from backend.app.config import get_settings

    get_settings.cache_clear()
    curriculum.reload_content()
    decks.reload_decks()


@pytest.fixture
def client(content):
    from fastapi.testclient import TestClient

    from backend.app.main import app

    with TestClient(app) as test_client:
        reply = test_client.post("/api/login", json={"name": "tester", "password": "change-me"})
        assert reply.status_code == 200, reply.text
        yield test_client
