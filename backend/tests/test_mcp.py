"""The MCP endpoint, its bearer tokens, and the token management API."""

import io
import json
from contextlib import redirect_stdout

import pytest

from backend.app import curriculum, db, decks, mcp_tokens

ACCEPT = "application/json, text/event-stream"
OTHER = {"Remote-User": "other", "X-SSO-Proxy": "test-proxy-secret"}

CHAPTER = """---
title: {title}
slug: {slug}
part: "Part 1 — The machine"
summary: {summary}
minutes: 10
objectives:
  - Do a thing.
{extra}---

# {title}

Body about coalescing.

## {section}

{text}
"""

DECK = """---
title: Alpha deck
subtitle: Sub
---

# Introduction

### Warp divergence {#divergence}

Threads in a warp share a program counter.

Note:
Say that the penalty is serialization.
"""

TESTS = """def run(submission):
    ok = submission.add(1, 1) == 2
    return {
        "passed": ok,
        "checks": [{"name": "cpu add", "passed": ok, "detail": "1+1"}],
        "metrics": {"calls": 1},
    }
"""


@pytest.fixture
def course_content(content, monkeypatch):
    """Two chapters, one lab with a fixture file, and one deck."""
    for name in (
        "RUNPOD_API_KEY",
        "RUNPOD_ENDPOINT_ID",
        "RUNPOD_ENDPOINT_A100",
        "RUNPOD_ENDPOINT_H100",
        "RUNPOD_ENDPOINT_2XA100",
    ):
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("LG_LABS_DIR", str(content / "labs"))

    def chapter(slug, title, section, text, lab=None):
        (content / "chapters" / f"{slug}.md").write_text(
            CHAPTER.format(
                title=title,
                slug=slug,
                summary=f"About {title}.",
                section=section,
                text=text,
                extra=f"lab: {lab}\n" if lab else "",
            )
        )

    chapter("00-alpha", "Alpha", "The roofline", "Intensity is FLOP per byte.", lab="00-lab")
    chapter("01-beta", "Beta", "Bank conflicts", "Padding removes a bank conflict.")
    lab = content / "labs" / "00-lab"
    lab.mkdir()
    # H100 is not configured here, so a run never leaves this machine.
    (lab / "meta.json").write_text(
        json.dumps({"id": "00-lab", "title": "Add", "brief": "Add two numbers.", "gpu": "H100-80GB"})
    )
    (lab / "starter.py").write_text("def add(a, b):\n    raise NotImplementedError\n")
    (lab / "solution.py").write_text("def add(a, b):\n    return a + b\n")
    (lab / "tests.py").write_text(TESTS)
    (lab / "fixture_report.csv").write_text("kernel,ms\nk,1\n")
    (content / "labs" / "lab_common.py").write_text("A100 = 'a100'\n")
    (content / "slides" / "00-alpha.md").write_text(DECK)
    curriculum.reload_content()
    decks.reload_decks()
    return content


def new_token(client, name="laptop", headers=None):
    reply = client.post("/api/mcp/tokens", json={"name": name}, headers=headers)
    assert reply.status_code == 200, reply.text
    return reply.json()


def rpc(client, token, method, params=None, path="/mcp", extra=None):
    headers = {"Accept": ACCEPT, "Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    headers.update(extra or {})
    body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
    return client.post(path, headers=headers, json=body, follow_redirects=False)


def call(client, token, tool, extra=None, **arguments):
    reply = rpc(client, token, "tools/call", {"name": tool, "arguments": arguments}, extra=extra)
    assert reply.status_code == 200, reply.text
    result = reply.json()["result"]
    text = result["content"][0]["text"]
    return result.get("isError", False), (text if result.get("isError") else json.loads(text))


# Token management -----------------------------------------------------------


def test_token_api_needs_sso(content):
    from fastapi.testclient import TestClient

    from backend.app.main import app

    with TestClient(app) as anonymous:
        assert anonymous.get("/api/mcp/tokens").status_code == 401
        assert anonymous.post("/api/mcp/tokens", json={"name": "x"}).status_code == 401
        assert anonymous.delete("/api/mcp/tokens/1").status_code == 401


def test_create_list_and_revoke(client, course_content):
    assert client.get("/api/mcp/tokens").json() == {"url": "http://testserver/mcp", "tokens": []}

    created = new_token(client)
    assert set(created) == {"id", "name", "prefix", "created_at", "token"}
    assert created["token"].startswith("gpumcp_")
    assert created["prefix"] == created["token"][:12]

    tokens = client.get("/api/mcp/tokens").json()["tokens"]
    assert tokens == [
        {
            "id": created["id"],
            "name": "laptop",
            "prefix": created["prefix"],
            "created_at": created["created_at"],
            "last_used_at": None,
        }
    ]
    # The secret itself is never stored, only its hash.
    with db.connect() as conn:
        row = dict(conn.execute("SELECT * FROM mcp_tokens").fetchone())
    assert created["token"] not in json.dumps(row)
    assert row["username"] == "tester" and len(row["token_hash"]) == 64

    # Another reader neither sees nor revokes it.
    assert client.get("/api/mcp/tokens", headers=OTHER).json()["tokens"] == []
    assert client.delete(f"/api/mcp/tokens/{created['id']}", headers=OTHER).status_code == 404

    assert client.delete(f"/api/mcp/tokens/{created['id']}").json() == {"revoked": True}
    assert client.get("/api/mcp/tokens").json()["tokens"] == []
    assert client.delete(f"/api/mcp/tokens/{created['id']}").status_code == 404
    assert rpc(client, created["token"], "ping").status_code == 401


def test_public_url_setting(client, course_content, monkeypatch):
    from backend.app.config import get_settings

    monkeypatch.setenv("MCP_PUBLIC_URL", "https://gpu.ifkash.dev/mcp")
    get_settings.cache_clear()
    assert client.get("/api/mcp/tokens").json()["url"] == "https://gpu.ifkash.dev/mcp"


@pytest.mark.parametrize("name", ["", "   ", "x" * 65])
def test_name_is_validated(client, course_content, name):
    assert client.post("/api/mcp/tokens", json={"name": name}).status_code == 422


def test_at_most_ten_active_tokens(client, course_content):
    for i in range(10):
        new_token(client, f"t{i}")
    assert client.post("/api/mcp/tokens", json={"name": "t10"}).status_code == 409


def test_cli_creates_lists_and_revokes(client, course_content):
    out = io.StringIO()
    with redirect_stdout(out):
        assert mcp_tokens.main(["create", "reader", "--name", "vm"]) == 0
    token = out.getvalue().strip().splitlines()[-1]
    assert token.startswith("gpumcp_")
    assert rpc(client, token, "ping").status_code == 200

    out = io.StringIO()
    with redirect_stdout(out):
        assert mcp_tokens.main(["list", "reader"]) == 0
    assert token[:12] in out.getvalue() and token not in out.getvalue()

    token_id = db.list_tokens("reader")[0]["id"]
    with redirect_stdout(io.StringIO()):
        assert mcp_tokens.main(["revoke", str(token_id)]) == 0
    assert rpc(client, token, "ping").status_code == 401


# Authentication on /mcp -----------------------------------------------------


def test_mcp_rejects_missing_and_bad_tokens(client, course_content):
    reply = rpc(client, None, "ping")
    assert reply.status_code == 401
    assert reply.headers["www-authenticate"].startswith("Bearer")

    reply = rpc(client, "gpumcp_not-a-real-token", "ping")
    assert reply.status_code == 401
    assert 'error="invalid_token"' in reply.headers["www-authenticate"]

    assert rpc(client, None, "ping", extra={"Authorization": "Basic cmVhZGVyOng="}).status_code == 401


def test_mcp_ignores_sso_headers(client, course_content):
    # The client sends valid SSO headers by default, and they buy nothing here.
    assert rpc(client, None, "ping").status_code == 401

    # With a valid token, SSO headers can't change who the caller is.
    token = new_token(client, headers=OTHER)["token"]
    error, progress = call(client, token, "get_progress")
    assert not error and progress["reader"] == "other"


def test_well_known_and_mcp_subpaths_are_not_found(content, tmp_path, monkeypatch):
    """The SPA catch-all answers 404, not index.html, where an MCP client probes."""
    import importlib

    from fastapi.testclient import TestClient

    from backend.app import main
    from backend.app.config import get_settings

    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html></html>")
    try:
        # The catch-all is registered when main is imported, from the settings then.
        monkeypatch.setenv("FRONTEND_DIST", str(dist))
        get_settings.cache_clear()
        app = importlib.reload(main).app
        with TestClient(app) as spa:
            assert spa.get("/.well-known/oauth-authorization-server").status_code == 404
            assert spa.get("/mcp/extra").status_code == 404
            assert spa.get("/api/nothing").status_code == 404
            assert spa.get("/anything").status_code == 200
    finally:
        monkeypatch.setenv("FRONTEND_DIST", str(tmp_path / "no-dist"))
        get_settings.cache_clear()
        importlib.reload(main)


# The protocol and the tools -------------------------------------------------


def test_initialize_and_list_tools(client, course_content):
    token = new_token(client)["token"]
    for path in ("/mcp", "/mcp/"):
        reply = rpc(
            client,
            token,
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
            path=path,
        )
        assert reply.status_code == 200, reply.text
        result = reply.json()["result"]
        assert result["protocolVersion"] == "2025-06-18"
        assert "tools" in result["capabilities"]
        assert result["serverInfo"]["name"] == "learn-gpu"

    reply = client.post(
        "/mcp",
        headers={"Authorization": f"Bearer {token}", "Accept": ACCEPT},
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
    )
    assert reply.status_code == 202

    tools = {t["name"] for t in rpc(client, token, "tools/list").json()["result"]["tools"]}
    assert tools == {
        "course_overview", "list_chapters", "get_chapter", "search", "list_labs", "get_lab",
        "get_lab_file", "get_solution", "list_decks", "get_deck", "get_providers",
        "get_compute_status", "get_document", "get_book_info", "get_progress", "set_progress",
        "get_draft", "save_draft", "get_note", "save_note", "list_runs", "get_run", "run_lab",
    }
    templates = rpc(client, token, "resources/templates/list").json()["result"]
    assert {t["uriTemplate"] for t in templates["resourceTemplates"]} == {
        "gpu://chapters/{slug}",
        "gpu://slides/{slug}",
    }


def test_last_used_is_recorded(client, course_content):
    token = new_token(client)["token"]
    assert rpc(client, token, "ping").status_code == 200
    assert client.get("/api/mcp/tokens").json()["tokens"][0]["last_used_at"] is not None


def test_get_chapter_does_not_mark_progress(client, course_content):
    token = new_token(client)["token"]
    error, chapter = call(client, token, "get_chapter", slug="00-alpha")
    assert not error
    assert "FLOP per byte" in chapter["body"]
    assert chapter["next"] == "01-beta" and chapter["lab"] == "00-lab"
    assert chapter["lab_title"] == "Add" and chapter["has_slides"] is True
    assert chapter["status"] is None and chapter["note"] == ""
    assert db.get_progress("tester") == {}

    error, message = call(client, token, "get_chapter", slug="nope")
    assert error and "No chapter" in message

    reply = rpc(client, token, "resources/read", {"uri": "gpu://chapters/01-beta"})
    assert "bank conflict" in reply.json()["result"]["contents"][0]["text"]
    reply = rpc(client, token, "resources/read", {"uri": "gpu://slides/00-alpha"})
    assert "serialization" in reply.json()["result"]["contents"][0]["text"]


def test_search_covers_chapters_labs_and_slides(client, course_content):
    token = new_token(client)["token"]
    _, found = call(client, token, "search", query="BANK conflict")
    assert [(r["kind"], r["slug"]) for r in found["results"]] == [("chapter", "01-beta")]
    assert found["results"][0]["section"] == "Bank conflicts"

    _, found = call(client, token, "search", query="serialization")
    hit = found["results"][0]
    assert (hit["kind"], hit["slug"], hit["slide"]) == ("slide", "00-alpha", "divergence")

    _, found = call(client, token, "search", query="add two numbers")
    assert found["results"][0]["kind"] == "lab" and found["results"][0]["slug"] == "00-lab"

    _, found = call(client, token, "search", query="coalescing", limit=1)
    assert found["total"] == 2 and len(found["results"]) == 1
    error, _ = call(client, token, "search", query="  ")
    assert error


def test_overview_labs_and_decks(client, course_content):
    token = new_token(client)["token"]
    _, overview = call(client, token, "course_overview")
    assert overview["reader"] == "tester"
    assert overview["counts"] == {
        "parts": 1, "chapters": 2, "labs": 1, "gpu_labs": 1, "decks": 1, "slides": 3,
    }
    assert [c["slug"] for c in overview["parts"][0]["chapters"]] == ["00-alpha", "01-beta"]
    first = overview["parts"][0]["chapters"][0]
    assert first["slug"] == "00-alpha" and first["has_slides"] is True
    assert overview["providers"]["providers"][0]["name"] == "runpod"
    assert "FACTS.md" in overview["documents"]

    _, listing = call(client, token, "list_chapters", part="part 1")
    assert listing["count"] == 2
    assert "The roofline" in listing["chapters"][0]["headings"]

    _, labs = call(client, token, "list_labs")
    assert labs["labs"][0]["gpu"] == "H100-80GB" and labs["labs"][0]["chapters"] == ["00-alpha"]

    _, lab = call(client, token, "get_lab", lab_id="00-lab")
    assert "NotImplementedError" in lab["starter"] and "def run" in lab["tests"]
    assert "solution" not in lab and "directory" not in lab
    assert lab["files"] == ["fixture_report.csv", "lab_common.py"]
    _, solution = call(client, token, "get_solution", lab_id="00-lab")
    assert "return a + b" in solution["solution"]
    _, extra = call(client, token, "get_lab_file", lab_id="00-lab", name="lab_common.py")
    assert extra["content"] == "A100 = 'a100'\n"
    for bad in ("../meta.json", "tests.py", "solution.py"):
        error, _ = call(client, token, "get_lab_file", lab_id="00-lab", name=bad)
        assert error, bad

    _, index = call(client, token, "list_decks")
    assert index["decks"][0]["deck"]["slides"] == 3 and index["decks"][1]["deck"] is None
    _, deck = call(client, token, "get_deck", slug="00-alpha")
    assert deck["next"] == "01-beta"
    assert deck["slides"][2]["notes"] == "Say that the penalty is serialization."
    error, _ = call(client, token, "get_deck", slug="01-beta")
    assert error


def test_documents_providers_and_book(client, course_content, tmp_path):
    token = new_token(client)["token"]
    _, page = call(client, token, "get_document", name="curriculum", limit=100)
    assert page["name"] == "CURRICULUM.md" and len(page["markdown"]) == 100
    assert page["next_offset"] == 100 and page["total_chars"] > 100
    _, second = call(client, token, "get_document", name="CURRICULUM", offset=100, limit=50)
    assert second["offset"] == 100
    error, _ = call(client, token, "get_document", name="../README")
    assert error

    _, providers = call(client, token, "get_providers")
    assert providers["providers"][0]["available"] is False

    # No RunPod key is set, so the snapshot makes no network call.
    _, compute = call(client, token, "get_compute_status")
    assert compute["active"] == 0 and compute["providers"][0]["available"] is False

    _, info = call(client, token, "get_book_info")
    assert info == {"book": None, "slides": None}
    book = tmp_path / "book"
    book.mkdir()
    (book / "b.pdf").write_bytes(b"%PDF")
    (book / "manifest.json").write_text(json.dumps({"file": "b.pdf", "title": "B", "pages": 3, "bytes": 4}))
    _, info = call(client, token, "get_book_info")
    assert info["book"]["url"] == "/book/b.pdf" and info["slides"] is None


def test_drafts_notes_and_progress_act_as_the_token_owner(client, course_content):
    token = new_token(client)["token"]
    _, saved = call(client, token, "save_draft", lab_id="00-lab", code="x = 1\n")
    assert saved["saved"] and db.get_draft("tester", "00-lab") == "x = 1\n"
    _, draft = call(client, token, "get_draft", lab_id="00-lab")
    assert draft["code"] == "x = 1\n"
    error, _ = call(client, token, "save_draft", lab_id="nope", code="x")
    assert error

    call(client, token, "save_note", chapter="01-beta", body="remember")
    assert db.get_note("tester", "01-beta") == "remember"
    _, note = call(client, token, "get_note", chapter="01-beta")
    assert note["note"] == "remember"

    _, marked = call(client, token, "set_progress", chapter="01-beta", status="in_progress")
    assert marked["status"] == "in_progress"
    error, _ = call(client, token, "set_progress", chapter="01-beta", status="finished")
    assert error
    _, progress = call(client, token, "get_progress")
    assert progress["summary"] == {
        "done": 0, "total": 2, "last": {"slug": "01-beta", "title": "Beta"},
    }
    assert [(c["slug"], c["status"]) for c in progress["chapters"]] == [("01-beta", "in_progress")]


def test_run_lab_records_the_run_and_marks_the_chapter_done(client, course_content):
    token = new_token(client)["token"]
    code = "def add(a, b):\n    return a + b\n"
    error, result = call(client, token, "run_lab", lab_id="00-lab", code=code)
    assert not error, result
    assert result["status"] == "passed" and result["passed"] is True
    assert result["checks"] == [{"name": "cpu add", "passed": True, "detail": "1+1"}]
    assert result["metrics"]["calls"] == 1
    assert result["notices"] == ["needs H100, not configured on this deployment"]
    assert result["chapters_done"] == ["00-alpha"]
    assert db.get_progress("tester")["00-alpha"] == "done"
    assert db.get_draft("tester", "00-lab") == code

    _, runs = call(client, token, "list_runs", lab_id="00-lab")
    assert [r["id"] for r in runs["runs"]] == [result["run_id"]]
    assert runs["runs"][0]["status"] == "finished" and runs["runs"][0]["passed"] is True
    _, run = call(client, token, "get_run", run_id=result["run_id"])
    assert "harness.py" in run["log"]
    error, _ = call(client, token, "get_run", run_id="nope")
    assert error

    error, failed = call(client, token, "run_lab", lab_id="00-lab", code="def add(a, b):\n    return 0\n")
    assert not error and failed["status"] == "failed" and failed["chapters_done"] == []

    error, message = call(client, token, "run_lab", lab_id="00-lab", code=code, provider="nope")
    assert error and "Unknown GPU provider" in message
    error, _ = call(client, token, "run_lab", lab_id="missing", code=code)
    assert error
