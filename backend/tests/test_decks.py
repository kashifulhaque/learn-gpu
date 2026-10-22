"""The slide deck parser: structure, attributes, and the errors it reports."""

from backend.app.decks import parse, slugify

DECK = """---
title: Demo
subtitle: A subtitle
notes: Opening notes.
---

# Introduction

Divider body.

Note:
Divider notes.

## Basics

### First slide {#first}

- one
- two

```python
# Not a heading
### Not a slide either
```

Note:
Speaker notes for the first slide.

More notes.

### Second slide

$$x^2$$

Note:
Notes.

# Summary

### Wrap up

Done.

Note:
Bye.
"""


def test_parses_structure():
    deck = parse(DECK)
    assert deck.errors == []
    assert [s.kind for s in deck.slides] == ["title", "section", "slide", "slide", "section", "slide"]
    title, intro, first, second, summary, wrap = deck.slides
    assert (title.title, title.body, title.notes) == ("Demo", "A subtitle", "Opening notes.")
    assert intro.body == "Divider body." and intro.notes == "Divider notes."
    assert first.id == "first" and first.section == "Introduction" and first.subsection == "Basics"
    assert "### Not a slide either" in first.body
    assert first.notes == "Speaker notes for the first slide.\n\nMore notes."
    assert second.id == "second-slide" and second.subsection == "Basics"
    assert summary.id == "section-summary" and wrap.section == "Summary"


def test_unknown_attributes_are_errors():
    deck = parse(DECK.replace("{#first}", "{#first video=abc}"))
    assert any("video" in e for e in deck.errors)
    deck = parse(DECK.replace("{#first}", "{#first book=s-x}"))
    assert any("book" in e for e in deck.errors)


def test_reports_malformed_decks_without_raising():
    assert any("front matter" in e for e in parse("# Introduction\n").errors)
    assert any("outside a slide" in e for e in parse("---\ntitle: T\n---\nstray text\n").errors)
    assert any("never closed" in e for e in parse("---\ntitle: T\n---\n# S\n### A\n```\ncode\n").errors)
    assert any("used twice" in e for e in parse("---\ntitle: T\n---\n# S\n### A\n### A\n").errors)
    assert any("second Note" in e for e in parse("---\ntitle: T\n---\n# S\n### A\nNote:\nx\nNote:\ny\n").errors)
    assert any("before any # section" in e for e in parse("---\ntitle: T\n---\n### A\n").errors)


def test_slugify():
    assert slugify("Why `float4` loads? ($x$)") == "why-float4-loads-x"
