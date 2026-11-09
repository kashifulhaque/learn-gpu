import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { api, type ChapterMeta } from "../lib/api";
import { chapterNumbers } from "../lib/chapters";

/**
 * The "/" search dialog, as tpu draws it: chapter titles and summaries match
 * at once, and section headings join them once every chapter body has loaded.
 * A heading result opens its chapter at that section (`/c/<slug>#s-<n>`).
 */

type Entry = {
  key: string;
  slug: string;
  number: string;
  title: string;
  detail: string;
  hash: string;
};

const MAX_RESULTS = 30;

/** Strips the Markdown and the maths a heading can carry, for display. */
function plain(text: string): string {
  return text
    .replace(/\$([^$]+)\$/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\*\*?([^*]+)\*\*?/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/\\[a-zA-Z]+/g, "")
    .replace(/[{}]/g, "")
    .trim();
}

/**
 * The `##` headings of a chapter body, numbered the way rehypeSectionize
 * numbers them, so `#s-<n>` lands on the same section.
 */
function headings(body: string): string[] {
  const found: string[] = [];
  let fence: string | null = null;
  for (const line of body.split("\n")) {
    const opener = /^\s*(`{3,}|~{3,})/.exec(line);
    if (opener) {
      if (fence === null) fence = opener[1][0];
      else if (opener[1][0] === fence) fence = null;
      continue;
    }
    if (fence === null && /^##\s+/.test(line)) found.push(plain(line.replace(/^##\s+/, "")));
  }
  return found;
}

function highlight(text: string, needle: string): ReactNode {
  if (!needle) return text;
  const at = text.toLowerCase().indexOf(needle);
  if (at < 0) return text;
  return (
    <>
      {text.slice(0, at)}
      <mark>{text.slice(at, at + needle.length)}</mark>
      {text.slice(at + needle.length)}
    </>
  );
}

let headingCache: Promise<Map<string, string[]>> | null = null;

function loadHeadings(chapters: ChapterMeta[]): Promise<Map<string, string[]>> {
  headingCache ??= Promise.all(
    chapters.map((chapter) =>
      api
        .chapter(chapter.slug)
        .then((full) => [chapter.slug, headings(full.body)] as const)
        .catch(() => [chapter.slug, [] as string[]] as const),
    ),
  ).then((pairs) => new Map(pairs));
  return headingCache;
}

export default function SearchDialog({
  chapters,
  open,
  onClose,
}: {
  chapters: ChapterMeta[];
  open: boolean;
  onClose: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const input = useRef<HTMLInputElement>(null);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState(0);
  const [sections, setSections] = useState<Map<string, string[]>>(new Map());
  const navigate = useNavigate();

  useEffect(() => {
    const node = dialog.current;
    if (!node) return;
    if (open && !node.open) {
      node.showModal();
      setQuery("");
      setSelected(0);
      input.current?.focus();
      if (chapters.length) loadHeadings(chapters).then(setSections);
    } else if (!open && node.open) {
      node.close();
    }
  }, [open, chapters]);

  const numbers = useMemo(() => chapterNumbers(chapters), [chapters]);

  const entries = useMemo(() => {
    const list: Entry[] = [];
    for (const chapter of chapters) {
      const number = numbers.get(chapter.slug) ?? "";
      list.push({
        key: chapter.slug,
        slug: chapter.slug,
        number,
        title: chapter.title,
        detail: chapter.summary,
        hash: "",
      });
      (sections.get(chapter.slug) ?? []).forEach((heading, index) => {
        list.push({
          key: `${chapter.slug}#${index + 1}`,
          slug: chapter.slug,
          number,
          title: heading,
          detail: chapter.title,
          hash: `#s-${index + 1}`,
        });
      });
    }
    return list;
  }, [chapters, numbers, sections]);

  const needle = query.trim().toLowerCase();
  const results = useMemo(() => {
    if (!needle) return entries.filter((entry) => !entry.hash).slice(0, MAX_RESULTS);
    return entries
      .filter(
        (entry) =>
          entry.title.toLowerCase().includes(needle) ||
          (!entry.hash && entry.detail.toLowerCase().includes(needle)) ||
          entry.number === needle,
      )
      .slice(0, MAX_RESULTS);
  }, [entries, needle]);

  const go = useCallback(
    (entry: Entry) => {
      onClose();
      navigate(`/c/${entry.slug}${entry.hash}`);
    },
    [navigate, onClose],
  );

  return (
    <dialog
      ref={dialog}
      className="search-dialog"
      aria-label="Search the course"
      onClose={onClose}
      onClick={(event) => {
        // A click on the backdrop lands on the dialog itself.
        if (event.target === dialog.current) onClose();
      }}
    >
      <input
        ref={input}
        value={query}
        onChange={(event) => {
          setQuery(event.target.value);
          setSelected(0);
        }}
        onKeyDown={(event) => {
          if (event.key === "ArrowDown") {
            event.preventDefault();
            setSelected((current) => Math.min(current + 1, results.length - 1));
          } else if (event.key === "ArrowUp") {
            event.preventDefault();
            setSelected((current) => Math.max(current - 1, 0));
          } else if (event.key === "Enter" && results[selected]) {
            event.preventDefault();
            go(results[selected]);
          }
        }}
        placeholder="Search chapters and sections"
        aria-label="Search chapters and sections"
        role="combobox"
        aria-expanded="true"
        aria-controls="search-results"
        autoComplete="off"
        spellCheck={false}
      />
      <ul
        className="search-results max-h-[calc(min(70vh,560px)-100px)]"
        id="search-results"
        role="listbox"
      >
        {results.map((entry, index) => (
          <li key={entry.key} role="option" aria-selected={index === selected}>
            <a
              href={`/c/${entry.slug}${entry.hash}`}
              className="search-result"
              aria-selected={index === selected}
              onMouseEnter={() => setSelected(index)}
              onClick={(event) => {
                event.preventDefault();
                go(entry);
              }}
            >
              <span className="search-result-title">
                <span className="mr-2 font-mono text-[11px] font-normal text-fg-subtle">
                  {entry.number}
                </span>
                {highlight(entry.title, needle)}
              </span>
              <span className="search-result-detail line-clamp-2">
                {entry.hash ? `Section in ${entry.detail}` : highlight(entry.detail, needle)}
              </span>
            </a>
          </li>
        ))}
        {results.length === 0 && (
          <li className="px-3 py-6 text-center text-[13px] text-fg-subtle">
            Nothing matches “{query}”.
          </li>
        )}
      </ul>
      <p className="flex items-center gap-3 border-t border-line px-4 py-2 font-mono text-[10px] uppercase tracking-[0.08em] text-fg-subtle">
        <span>
          <kbd className="key">↑</kbd> <kbd className="key">↓</kbd> to move
        </span>
        <span>
          <kbd className="key">↵</kbd> to open
        </span>
        <span>
          <kbd className="key">esc</kbd> to close
        </span>
      </p>
    </dialog>
  );
}

/** Opens the search on "/" unless the reader is typing somewhere. */
export function useSlashKey(onOpen: () => void) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "/" || event.metaKey || event.ctrlKey || event.altKey) return;
      const target = event.target as HTMLElement | null;
      if (target?.closest("input, textarea, select, [contenteditable], .monaco-editor")) return;
      event.preventDefault();
      onOpen();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onOpen]);
}
