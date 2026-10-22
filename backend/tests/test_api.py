"""The content and slides endpoints, against a small throwaway content tree."""

import json

from backend.app import curriculum, decks

CHAPTER = """---
title: {title}
slug: {slug}
part: "Part 1 — The machine"
summary: A summary.
minutes: 10
objectives:
  - Do a thing.
{extra}---

# {title}

Body.
"""

DECK = """---
title: {title} deck
subtitle: Sub
---

# Introduction

### One

Body.

Note:
Notes.

# Summary

### Two

Body.

Note:
Notes.
"""


def reload():
    """The app caches content after the first read; tests write files after it starts."""
    curriculum.reload_content()
    decks.reload_decks()


def write_chapter(content, slug, lab=None):
    extra = f"lab: {lab}\n" if lab else ""
    (content / "chapters" / f"{slug}.md").write_text(CHAPTER.format(title=slug, slug=slug, extra=extra))


def write_lab(content, lab_id):
    lab = content / "labs" / lab_id
    lab.mkdir()
    (lab / "meta.json").write_text(json.dumps({"id": lab_id, "title": "T", "brief": "B"}))
    (lab / "starter.py").write_text("# starter\n")
    (lab / "solution.py").write_text("# solution\n")


def test_chapter_without_a_lab_renders(client, content):
    write_chapter(content, "23-beyond")
    reply = client.get("/api/chapters/23-beyond")
    assert reply.status_code == 200
    body = reply.json()
    assert body["lab"] is None and "lab_detail" not in body and body["has_slides"] is False


def test_lab_is_served_without_its_solution(client, content):
    write_chapter(content, "00-intro", lab="00-lab")
    write_lab(content, "00-lab")
    body = client.get("/api/chapters/00-intro").json()
    assert body["lab_detail"]["id"] == "00-lab" and "solution" not in body["lab_detail"]
    assert "needs_weights" not in body["lab_detail"]
    assert client.get("/api/labs/00-lab/solution").json() == {"solution": "# solution\n"}


def test_slides_index_and_deck(client, content):
    write_chapter(content, "00-intro")
    write_chapter(content, "01-next")
    (content / "slides" / "00-intro.md").write_text(DECK.format(title="Intro"))
    reload()

    listing = client.get("/api/chapters").json()["chapters"]
    assert [c["has_slides"] for c in listing] == [True, False]

    index = client.get("/api/slides").json()["decks"]
    assert [d["slug"] for d in index] == ["00-intro", "01-next"]
    assert index[0]["deck"]["slides"] == 5 and index[0]["deck"]["sections"] == ["Introduction", "Summary"]
    assert index[1]["deck"] is None

    deck = client.get("/api/slides/00-intro").json()
    assert deck["title"] == "Intro deck" and deck["next"] == "01-next" and deck["prev"] is None
    assert [s["kind"] for s in deck["slides"]] == ["title", "section", "slide", "section", "slide"]
    assert deck["slides"][2]["notes"] == "Notes."

    assert client.get("/api/slides/01-next").status_code == 404
    assert client.get("/api/slides/99-none").status_code == 404


def test_a_broken_deck_is_not_served(client, content):
    write_chapter(content, "00-intro")
    (content / "slides" / "00-intro.md").write_text("no front matter\n")
    reload()
    assert client.get("/api/slides/00-intro").status_code == 404
    assert client.get("/api/slides").json()["decks"][0]["deck"] is None


def test_api_needs_a_session(content):
    from fastapi.testclient import TestClient

    from backend.app.main import app

    with TestClient(app) as anonymous:
        assert anonymous.get("/api/slides").status_code == 401
        assert anonymous.get("/api/slides/00-intro").status_code == 401
        assert anonymous.get("/api/health").json()["ok"] is True


def test_the_removed_endpoints_are_gone(client, content):
    for path in ("/api/lectures", "/api/infra/volume?provider=runpod&name=x"):
        assert client.get(path).status_code == 404
    assert "model" not in client.get("/api/me").json()


def test_book_info_reports_both_pdfs(client, content, tmp_path):
    book = tmp_path / "book"
    book.mkdir()
    (book / "b.pdf").write_bytes(b"%PDF")
    (book / "manifest.json").write_text(json.dumps({"file": "b.pdf", "title": "B", "pages": 3, "bytes": 4}))
    reply = client.get("/api/book").json()
    assert reply["book"]["url"] == "/book/b.pdf" and reply["slides"] is None
    assert client.get("/book/b.pdf").status_code == 200
    assert client.get("/book/other.pdf").status_code == 404
