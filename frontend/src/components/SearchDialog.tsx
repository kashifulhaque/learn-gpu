import { Fragment, useEffect, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";

import { Icon } from "./shellIcons";

/** One searchable page: a lesson, a chapter, or a project. */
export type SearchEntry = {
  /** Unique key, such as the slug. */
  id: string;
  href: string;
  title: string;
  /** The small line above the title, such as "Lesson 12 · Neural networks". */
  kicker: string;
  /** One line under the title when no heading matches better. */
  summary: string;
  /** More text to match against: objectives, the part name, the slug. */
  keywords?: string[];
  /** Section headings, each with the link that jumps to it. */
  headings?: { text: string; href: string }[];
};

type Props = {
  open: boolean;
  onClose: () => void;
  entries: SearchEntry[];
  /** The singular noun for an entry: "lesson" or "chapter". */
  noun?: string;
  placeholder?: string;
};

type Prepared = {
  entry: SearchEntry;
  title: string;
  rest: string;
  headings: string[];
};

type Hit = { entry: SearchEntry; score: number; detail: string; heading: boolean; href: string };

const LIMIT = 30;

/**
 * The "/" search dialog, styled after the tpu reference site: a native modal
 * <dialog> with a heading, a search field, result rows, and a hint line.
 * Matches titles, headings, summaries, and keywords. Arrow keys move through
 * the results, Return opens one, and Escape closes the dialog.
 *
 * Portable: the course builds `entries` from its own lesson index.
 */
export default function SearchDialog({
  open,
  onClose,
  entries,
  noun = "lesson",
  placeholder = "Try a topic, such as “gradient”, “attention”, or “loss”",
}: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const input = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState(0);

  const prepared = useMemo<Prepared[]>(
    () =>
      entries.map((entry) => ({
        entry,
        title: entry.title.toLowerCase(),
        rest: [entry.summary, entry.kicker, ...(entry.keywords ?? [])].join("\n").toLowerCase(),
        headings: (entry.headings ?? []).map((heading) => heading.text.toLowerCase()),
      })),
    [entries],
  );

  const words = useMemo(() => query.toLowerCase().split(/\s+/).filter((word) => word.length > 0), [query]);

  const hits = useMemo<Hit[]>(() => {
    if (words.length === 0) return [];
    const found: Hit[] = [];
    for (const item of prepared) {
      const inHeading = (word: string) => item.headings.some((heading) => heading.includes(word));
      if (!words.every((word) => item.title.includes(word) || item.rest.includes(word) || inHeading(word))) continue;
      let score = 0;
      for (const word of words) {
        if (item.title.startsWith(word)) score += 6;
        else if (new RegExp(`\\b${escape(word)}`).test(item.title)) score += 4;
        else if (item.title.includes(word)) score += 2;
        else if (item.rest.includes(word)) score += 1;
        else score += 0.5;
      }
      found.push({ entry: item.entry, score, ...detailFor(item, words) });
    }
    return found.sort((a, b) => b.score - a.score).slice(0, LIMIT);
  }, [prepared, words]);

  useEffect(() => setSelected(0), [query]);

  useEffect(() => {
    const box = dialog.current;
    if (!box) return;
    if (open && !box.open) {
      setQuery("");
      box.showModal();
      input.current?.focus();
    } else if (!open && box.open) {
      box.close();
    }
  }, [open]);

  useEffect(() => {
    dialog.current?.querySelector(`[data-index="${selected}"]`)?.scrollIntoView({ block: "nearest" });
  }, [selected]);

  function go(hit: Hit | undefined) {
    if (!hit) return;
    onClose();
    navigate(hit.href);
  }

  function onKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setSelected((i) => Math.min(i + 1, Math.max(hits.length - 1, 0)));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setSelected((i) => Math.max(i - 1, 0));
    } else if (event.key === "Enter") {
      event.preventDefault();
      go(hits[selected]);
    } else if (event.key === "Escape") {
      // A search field clears itself on the first Escape. Close at once instead.
      event.preventDefault();
      onClose();
    }
  }

  const plural = `${noun}s`;

  return (
    <dialog
      ref={dialog}
      className="sh-search"
      aria-labelledby="sh-search-title"
      onClose={onClose}
      onClick={(event) => {
        // A click on the backdrop lands on the dialog element itself.
        if (event.target === dialog.current) onClose();
      }}
    >
      <div className="sh-search-heading">
        <h2 id="sh-search-title">Find your next idea</h2>
        <button type="button" className="sh-icon-button" onClick={onClose} aria-label="Close search">
          <Icon name="close" size={18} />
        </button>
      </div>

      <label className="sh-search-input">
        <Icon name="search" size={18} />
        <span className="sh-sr-only">Search {plural}</span>
        <input
          ref={input}
          type="search"
          enterKeyHint="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={onKeyDown}
          placeholder={placeholder}
          aria-controls="sh-search-results"
          aria-activedescendant={hits[selected] ? `sh-search-hit-${selected}` : undefined}
          autoComplete="off"
          spellCheck={false}
        />
      </label>

      <div className="sh-search-results">
        {words.length === 0 ? (
          <p className="sh-search-empty">
            Type to search {plural} by title, section heading, or summary. Use the arrow keys to move and Return
            to open.
          </p>
        ) : hits.length === 0 ? (
          <p className="sh-search-empty">
            No {noun} matches “{query.trim()}”. Try a shorter or more general word.
          </p>
        ) : (
          <ul id="sh-search-results" role="listbox" aria-label="Results">
            {hits.map((hit, index) => (
              <li key={hit.entry.id}>
                <a
                  id={`sh-search-hit-${index}`}
                  data-index={index}
                  role="option"
                  aria-selected={index === selected}
                  href={hit.href}
                  className="sh-search-result"
                  onMouseMove={() => setSelected(index)}
                  onClick={(event) => {
                    event.preventDefault();
                    go(hit);
                  }}
                >
                  <span className="sh-search-result-text">
                    <small>{hit.entry.kicker}</small>
                    <strong>{mark(hit.entry.title, words)}</strong>
                    {hit.detail && (
                      <span>
                        {hit.heading && <em>Section: </em>}
                        {mark(hit.detail, words)}
                      </span>
                    )}
                  </span>
                  <Icon name="arrow" size={16} />
                </a>
              </li>
            ))}
          </ul>
        )}
      </div>

      <p className="sh-search-hint">
        Search all {entries.length} {plural} · Escape to close
      </p>
    </dialog>
  );
}

/**
 * The line under a result: a heading when one matches at least as many words
 * as the summary does and the title alone does not cover the query.
 */
function detailFor(item: Prepared, words: string[]): Pick<Hit, "detail" | "heading" | "href"> {
  const { entry } = item;
  const summaryCount = count(entry.summary, words);
  if (count(item.title, words) < words.length && entry.headings) {
    let found = -1;
    let foundCount = 0;
    item.headings.forEach((heading, index) => {
      const n = count(heading, words);
      if (n > foundCount) {
        found = index;
        foundCount = n;
      }
    });
    if (found >= 0 && foundCount >= summaryCount) {
      const heading = entry.headings[found];
      return { detail: heading.text, heading: true, href: heading.href };
    }
  }
  return { detail: entry.summary, heading: false, href: entry.href };
}

function count(text: string, words: string[]): number {
  const lower = text.toLowerCase();
  return words.filter((word) => lower.includes(word)).length;
}

function escape(word: string): string {
  return word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** The text with every query word wrapped in <mark>. */
function mark(text: string, words: string[]): ReactNode {
  if (words.length === 0) return text;
  const pattern = new RegExp(`(${words.map(escape).join("|")})`, "gi");
  return text
    .split(pattern)
    .map((piece, index) => (index % 2 === 1 ? <mark key={index}>{piece}</mark> : <Fragment key={index}>{piece}</Fragment>));
}
