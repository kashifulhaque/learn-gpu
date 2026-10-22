import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, type Chapter, type ChapterMeta } from "../lib/api";
import { chapterNumbers, splitPart } from "../lib/chapters";
import Markdown from "../components/Markdown";
import LabPane from "../components/LabPane";
import SplitPane from "../components/SplitPane";
import {
  ArrowLeftIcon,
  ArrowRightIcon,
  CheckIcon,
  ChevronDownIcon,
  ClockIcon,
  FlaskIcon,
  FocusIcon,
  ListIcon,
  PresentIcon,
} from "../components/icons";
import { Button, buttonClass, IconButton, Kbd, Loading } from "../components/ui";
import { useIsWide } from "../lib/useMediaQuery";

type Props = {
  chapters: ChapterMeta[];
  progress: Record<string, string>;
  onProgress: (slug: string, status: "in_progress" | "done") => void;
};

type Section = { index: number; id: string; title: string };
type NoteState = "idle" | "dirty" | "saved" | "failed";

// Scrolling updates this page many times a second. The chapter body is the
// expensive part to render and never changes while you read it.
const ChapterBody = memo(Markdown);

const FOCUS_KEY = "lg.reader.focus";

function readFocus(): boolean {
  try {
    return window.localStorage.getItem(FOCUS_KEY) === "1";
  } catch {
    return false;
  }
}

function pad(value: number): string {
  return String(value).padStart(2, "0");
}

/** A heading's visible text, without the MathML copy KaTeX adds for screen readers. */
function headingText(heading: HTMLElement): string {
  const clone = heading.cloneNode(true) as HTMLElement;
  clone.querySelectorAll(".katex-mathml").forEach((node) => node.remove());
  return (clone.textContent ?? "").replace(/\s+/g, " ").trim();
}

export default function ChapterPage({ chapters, progress, onProgress }: Props) {
  const { slug = "" } = useParams();
  const [chapter, setChapter] = useState<Chapter | null>(null);
  const [note, setNote] = useState("");
  const [noteState, setNoteState] = useState<NoteState>("idle");
  const [error, setError] = useState("");
  const [view, setView] = useState<"read" | "lab">("read");
  const [sections, setSections] = useState<Section[]>([]);
  const [active, setActive] = useState(0);
  const [furthest, setFurthest] = useState(0);
  const [minutesLeft, setMinutesLeft] = useState<number | null>(null);
  const [focus, setFocus] = useState(readFocus);
  const [mapOpen, setMapOpen] = useState(false);
  const noteTimer = useRef<number | null>(null);
  const reading = useRef<HTMLDivElement>(null);
  const article = useRef<HTMLElement>(null);
  const bar = useRef<HTMLDivElement>(null);
  const frame = useRef<number | null>(null);
  const tracker = useRef<HTMLDivElement>(null);
  const wide = useIsWide();

  const numbers = useMemo(() => chapterNumbers(chapters), [chapters]);
  const bySlug = useMemo(() => new Map(chapters.map((c) => [c.slug, c])), [chapters]);

  useEffect(() => {
    setChapter(null);
    setError("");
    setView("read");
    setSections([]);
    setActive(0);
    setFurthest(0);
    setMapOpen(false);
    setNoteState("idle");
    api
      .chapter(slug)
      .then((result) => {
        setChapter(result);
        setNote(result.note);
        if (!progress[slug]) onProgress(slug, "in_progress");
      })
      .catch((e) => setError(String((e as Error).message ?? e)));
    reading.current?.scrollTo({ top: 0 });
    // Progress is intentionally excluded: marking a chapter read must not refetch.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [slug]);

  // The body renders synchronously, so its sections exist by the time this runs.
  useLayoutEffect(() => {
    const root = article.current;
    if (!chapter || !root) return;
    const found: Section[] = [];
    root.querySelectorAll<HTMLElement>("section[data-section]").forEach((section) => {
      const heading = section.querySelector<HTMLElement>(":scope > h2");
      if (!heading) return;
      found.push({
        index: Number(section.dataset.section),
        id: heading.id,
        title: headingText(heading),
      });
    });
    setSections(found);
  }, [chapter]);

  const track = useCallback(() => {
    frame.current = null;
    const box = reading.current;
    const root = article.current;
    if (!box || !root) return;

    const max = box.scrollHeight - box.clientHeight;
    const done = max > 0 ? Math.min(1, box.scrollTop / max) : 1;
    // The bar moves on every frame, so it is styled directly rather than
    // through state.
    if (bar.current) bar.current.style.transform = `scaleX(${done})`;
    if (chapter?.minutes) setMinutesLeft(Math.ceil(chapter.minutes * (1 - done)));

    // The section you are reading is the last one whose heading has passed a
    // line 30% of the way down the pane.
    const line = box.getBoundingClientRect().top + box.clientHeight * 0.3;
    let current = 0;
    root.querySelectorAll<HTMLElement>("section[data-section]").forEach((section) => {
      if (section.getBoundingClientRect().top <= line) current = Number(section.dataset.section);
    });
    // A short last section never reaches the line, so the end of the page
    // counts as reaching it.
    if (max > 0 && box.scrollTop >= max - 4) {
      const all = root.querySelectorAll<HTMLElement>("section[data-section]");
      current = Number(all[all.length - 1]?.dataset.section ?? current);
    }
    setActive(current);
    setFurthest((previous) => Math.max(previous, current));
  }, [chapter]);

  const onScroll = useCallback(() => {
    if (frame.current === null) frame.current = window.requestAnimationFrame(track);
  }, [track]);

  useEffect(() => {
    track();
    return () => {
      if (frame.current !== null) window.cancelAnimationFrame(frame.current);
    };
  }, [track, sections]);

  // Focus mode dims every section except the one you are reading. The
  // sections belong to the memoized body, so their classes are set here
  // rather than re-rendering it.
  useEffect(() => {
    const root = article.current;
    if (!root) return;
    root.classList.toggle("focus-mode", focus);
    root.querySelectorAll<HTMLElement>("section[data-section]").forEach((section) => {
      section.classList.toggle("is-active", Number(section.dataset.section) === active);
    });
  }, [active, focus, sections]);

  useEffect(() => {
    try {
      window.localStorage.setItem(FOCUS_KEY, focus ? "1" : "0");
    } catch {
      // Focus mode still works, it just forgets between visits.
    }
  }, [focus]);

  const goTo = useCallback((id: string) => {
    const box = reading.current;
    const target = document.getElementById(id);
    if (!box || !target) return;
    const top =
      target.getBoundingClientRect().top - box.getBoundingClientRect().top + box.scrollTop - 16;
    box.scrollTo({ top, behavior: "smooth" });
    setMapOpen(false);
  }, []);

  // The section list closes on a click anywhere else, or on Escape.
  useEffect(() => {
    if (!mapOpen) return;
    const onPointer = (event: PointerEvent) => {
      if (!tracker.current?.contains(event.target as Node)) setMapOpen(false);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMapOpen(false);
    };
    window.addEventListener("pointerdown", onPointer);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("pointerdown", onPointer);
      window.removeEventListener("keydown", onKey);
    };
  }, [mapOpen]);

  // `[` and `]` step between sections, unless you are typing somewhere.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (event.key !== "[" && event.key !== "]") return;
      const target = event.target as HTMLElement | null;
      if (target?.closest("input, textarea, select, [contenteditable], .monaco-editor")) return;
      const position = sections.findIndex((section) => section.index === active);
      const next = event.key === "]" ? position + 1 : position - 1;
      if (next >= 0 && next < sections.length) {
        event.preventDefault();
        goTo(sections[next].id);
      } else if (next < 0) {
        event.preventDefault();
        reading.current?.scrollTo({ top: 0, behavior: "smooth" });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [sections, active, goTo]);

  function editNote(value: string) {
    setNote(value);
    setNoteState("dirty");
    if (noteTimer.current) window.clearTimeout(noteTimer.current);
    noteTimer.current = window.setTimeout(() => {
      api
        .saveNote(slug, value)
        .then(() => setNoteState("saved"))
        .catch(() => setNoteState("failed"));
    }, 800);
  }

  if (error) {
    const missing = /no such chapter/i.test(error);
    return (
      <div className="mx-auto max-w-md px-6 py-20 text-center">
        <p className="font-serif text-2xl font-semibold text-fg">
          {missing ? "There's no chapter here" : "This chapter didn't load"}
        </p>
        <p className="mt-2 text-[13.5px] leading-6 text-fg-subtle">
          {missing ? "The link may be out of date. The overview lists every chapter." : error}
        </p>
        <Link to="/" className={buttonClass("secondary", "md", "mt-6")}>
          <ArrowLeftIcon className="size-3.5" />
          Back to the overview
        </Link>
      </div>
    );
  }
  if (!chapter) {
    return <Loading label="Loading the chapter" />;
  }

  const done = progress[slug] === "done";
  const hasLab = Boolean(chapter.lab_detail);
  const position = sections.findIndex((section) => section.index === active);
  const current = position >= 0 ? sections[position] : null;
  const part = splitPart(chapter.part);
  const prev = chapter.prev ? bySlug.get(chapter.prev) : undefined;
  const next = chapter.next ? bySlug.get(chapter.next) : undefined;

  const toggleDone = () => onProgress(slug, done ? "in_progress" : "done");

  const sectionList = (compact: boolean) => (
    <ol className={compact ? "space-y-px" : "grid gap-x-4 gap-y-px sm:grid-cols-2"}>
      {sections.map((section) => {
        const isCurrent = section.index === active;
        const seen = section.index <= furthest && !isCurrent;
        return (
          <li key={section.id}>
            <button
              type="button"
              onClick={() => goTo(section.id)}
              aria-current={isCurrent ? "location" : undefined}
              className={`flex w-full items-baseline gap-2.5 rounded-md px-2 py-1.5 text-left text-[13px] leading-5 transition-colors ${
                isCurrent ? "bg-accent/8 text-fg" : "text-fg-muted hover:bg-tint hover:text-fg"
              }`}
            >
              <span
                className={`flex w-5 shrink-0 justify-end font-mono text-[11.5px] tabular-nums ${
                  isCurrent ? "text-accent" : seen ? "text-ok" : "text-fg-faint"
                }`}
              >
                {seen ? <CheckIcon className="size-3.5 translate-y-0.5" /> : pad(section.index)}
              </span>
              <span className="min-w-0">{section.title}</span>
            </button>
          </li>
        );
      })}
    </ol>
  );

  const readingPane = (
    <div className="@container flex h-full min-h-0 flex-col bg-paper">
      {/* One bar for the whole reader: where you are, a way to jump anywhere,
          and the controls for the chapter. Its height matches the lab's
          header, so the two line up across the split. */}
      <div ref={tracker} className="relative shrink-0">
        <header className="flex h-12 items-center gap-1 border-b border-line px-2 sm:gap-1.5 sm:px-3">
          <button
            type="button"
            onClick={() => setMapOpen((open) => !open)}
            aria-expanded={mapOpen}
            aria-haspopup="true"
            aria-label="Sections of this chapter"
            className="flex h-9 min-w-0 flex-1 items-center gap-2 rounded-md px-2 text-left transition-colors hover:bg-tint"
          >
            <ListIcon className="size-4 text-fg-subtle" />
            <span className="shrink-0 font-mono text-[11.5px] tabular-nums text-fg-faint">
              {current ? `${pad(current.index)}/${pad(sections.length)}` : "00"}
            </span>
            <span className="min-w-0 truncate text-[13px] font-medium text-fg">
              {current ? current.title : chapter.title}
            </span>
            <ChevronDownIcon
              className={`size-3.5 text-fg-faint transition-transform ${mapOpen ? "rotate-180" : ""}`}
            />
          </button>

          <button
            type="button"
            onClick={() => setFocus((on) => !on)}
            aria-pressed={focus}
            title="Dim every section except the one you're reading"
            className={`flex h-8 shrink-0 items-center gap-1.5 rounded-md px-2 text-[12.5px] font-medium transition-colors ${
              focus ? "bg-accent/10 text-accent" : "text-fg-subtle hover:bg-tint hover:text-fg"
            }`}
          >
            <FocusIcon />
            <span className="hidden @xl:inline">Focus</span>
          </button>

          <button
            type="button"
            onClick={toggleDone}
            aria-pressed={done}
            aria-label={done ? "Mark this chapter not done" : "Mark this chapter done"}
            title={done ? "Mark this chapter not done" : "Mark this chapter done"}
            className={`flex h-8 shrink-0 items-center gap-1.5 rounded-md border px-2 text-[12.5px] font-medium transition-colors sm:px-2.5 ${
              done
                ? "border-ok/30 bg-ok/8 text-ok hover:bg-ok/14"
                : "border-line bg-card text-fg-muted hover:border-line-strong hover:text-fg"
            }`}
          >
            <CheckIcon className="size-3.5" />
            {/* In a narrow pane the words would squeeze the section title to
                nothing, so there the check stands in for them. */}
            <span className="hidden @md:inline">{done ? "Done" : "Mark done"}</span>
          </button>

          <span className="ml-0.5 flex items-center">
            {prev ? (
              <Link
                to={`/c/${prev.slug}`}
                title={`Previous: ${prev.title}`}
                aria-label="Previous chapter"
                className="flex size-8 items-center justify-center rounded-md text-fg-subtle transition-colors hover:bg-tint hover:text-fg"
              >
                <ArrowLeftIcon />
              </Link>
            ) : (
              <IconButton size="sm" disabled aria-label="No previous chapter" className="size-8">
                <ArrowLeftIcon />
              </IconButton>
            )}
            {next ? (
              <Link
                to={`/c/${next.slug}`}
                title={`Next: ${next.title}`}
                aria-label="Next chapter"
                className="flex size-8 items-center justify-center rounded-md text-fg-subtle transition-colors hover:bg-tint hover:text-fg"
              >
                <ArrowRightIcon />
              </Link>
            ) : (
              <IconButton size="sm" disabled aria-label="No next chapter" className="size-8">
                <ArrowRightIcon />
              </IconButton>
            )}
          </span>
        </header>

        <div className="absolute inset-x-0 bottom-0 h-[2px]" aria-hidden>
          <div
            ref={bar}
            className="h-full origin-left bg-accent transition-transform duration-150"
            style={{ transform: "scaleX(0)" }}
          />
        </div>

        {mapOpen && sections.length > 0 && (
          <div className="absolute inset-x-2 top-full z-20 mt-1.5 max-h-[65vh] overflow-y-auto rounded-lg border border-line bg-card p-1.5 shadow-pop sm:left-3 sm:right-auto sm:w-[26rem]">
            <div className="flex items-baseline justify-between gap-3 px-2 pb-1.5 pt-1 text-[12px]">
              <span className="font-medium text-fg-muted">
                {sections.length} sections
              </span>
              {minutesLeft !== null && (
                <span className="flex items-center gap-1.5 text-fg-subtle">
                  <ClockIcon className="size-3.5" />
                  {minutesLeft > 0 ? `About ${minutesLeft} min left` : "Finished reading"}
                </span>
              )}
            </div>
            {sectionList(true)}
            <p className="mt-1.5 flex items-center gap-1.5 border-t border-line px-2 pb-0.5 pt-2 text-[11.5px] text-fg-faint">
              <Kbd>[</Kbd>
              <Kbd>]</Kbd>
              step between sections
            </p>
          </div>
        )}
      </div>

      <div ref={reading} onScroll={onScroll} className="min-h-0 flex-1 overflow-y-auto">
        <div className={`px-5 pb-16 pt-8 @lg:px-10 @lg:pt-12 ${hasLab ? "max-w-[46rem]" : "mx-auto max-w-[46rem]"}`}>
          <header className="mb-8">
            <p className="text-[12.5px] text-fg-subtle">
              {part.label && <span>{part.label} · </span>}
              {part.name}
            </p>
            <h1 className="mt-2 font-serif text-[2rem] font-semibold leading-[1.15] tracking-[-0.01em] text-fg sm:text-[2.35rem]">
              <span className="mr-3 font-mono text-[0.55em] font-normal tracking-normal text-accent align-[0.2em]">
                {numbers.get(chapter.slug)}
              </span>
              {chapter.title}
            </h1>
            {chapter.summary && (
              <p className="mt-3 font-serif text-[1.125rem] leading-7 text-fg-muted">{chapter.summary}</p>
            )}
            <p className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-1 text-[12.5px] text-fg-subtle">
              {chapter.minutes && (
                <span className="flex items-center gap-1.5">
                  <ClockIcon className="size-3.5 text-fg-faint" />
                  {chapter.minutes} min
                </span>
              )}
              {sections.length > 0 && (
                <span className="flex items-center gap-1.5">
                  <ListIcon className="size-3.5 text-fg-faint" />
                  {sections.length} sections
                </span>
              )}
              {chapter.has_slides && (
                <Link
                  to={`/slides/${chapter.slug}`}
                  className="flex items-center gap-1.5 text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent"
                >
                  <PresentIcon className="size-3.5" />
                  Slides
                </Link>
              )}
              {hasLab && (
                <span className="flex items-center gap-1.5">
                  <FlaskIcon className="size-3.5 text-fg-faint" />
                  {wide ? "Lab alongside" : (
                    <button
                      type="button"
                      onClick={() => setView("lab")}
                      className="text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent"
                    >
                      Open the lab
                    </button>
                  )}
                </span>
              )}
            </p>
          </header>

          {(chapter.objectives.length > 0 || sections.length > 0) && (
            <section className="mb-10 rounded-lg border border-line bg-well/60">
              {chapter.objectives.length > 0 && (
                <div className="px-5 py-4">
                  <h2 className="text-[13px] font-semibold text-fg">By the end, you can</h2>
                  <ul className="mt-2.5 space-y-1.5 text-[13.5px] leading-6 text-fg-muted">
                    {chapter.objectives.map((objective) => (
                      <li key={objective} className="flex gap-2.5">
                        <span className="mt-[0.7em] h-px w-2.5 shrink-0 bg-fg-faint" aria-hidden />
                        {objective}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {sections.length > 0 && (
                <div className={`px-3 pb-3 pt-3.5 ${chapter.objectives.length ? "border-t border-line" : ""}`}>
                  <h2 className="px-2 pb-1.5 text-[13px] font-semibold text-fg">In this chapter</h2>
                  {sectionList(false)}
                </div>
              )}
            </section>
          )}

          <article ref={article} className="chapter-article">
            <ChapterBody>{chapter.body}</ChapterBody>
          </article>

          <section className="mt-14">
            <div className="mb-2 flex items-baseline justify-between gap-3">
              <label htmlFor="note" className="text-[13px] font-semibold text-fg">
                Your notes
              </label>
              <span
                className={`text-[12px] ${noteState === "failed" ? "text-bad" : "text-fg-faint"}`}
                aria-live="polite"
              >
                {noteState === "dirty"
                  ? "Saving…"
                  : noteState === "saved"
                    ? "Saved"
                    : noteState === "failed"
                      ? "Couldn't save your note"
                      : "Saved as you type"}
              </span>
            </div>
            <textarea
              id="note"
              value={note}
              onChange={(event) => editNote(event.target.value)}
              rows={4}
              placeholder="The idea you want to come back to, a question for later…"
              className="w-full resize-y rounded-lg border border-line bg-card px-3.5 py-3 text-[14px] leading-6 text-fg outline-none transition-colors placeholder:text-fg-faint focus:border-line-strong"
            />
          </section>

          <div className="mt-10 border-t border-line pt-6">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <p className="text-[13.5px] text-fg-muted">
                {done ? "You've marked this chapter done." : "Finished with this chapter?"}
              </p>
              <Button
                variant={done ? "secondary" : "primary"}
                onClick={toggleDone}
                className={done ? "text-ok" : ""}
              >
                <CheckIcon className="size-3.5" />
                {done ? "Done" : "Mark chapter done"}
              </Button>
            </div>

            {(prev || next) && (
              <nav className="mt-6 grid gap-3 sm:grid-cols-2" aria-label="Chapters">
                {prev ? (
                  <Link
                    to={`/c/${prev.slug}`}
                    className="group rounded-lg border border-line px-4 py-3 transition-colors hover:border-line-strong hover:bg-tint/40"
                  >
                    <span className="flex items-center gap-1.5 text-[12px] text-fg-subtle">
                      <ArrowLeftIcon className="size-3.5" />
                      Previous
                    </span>
                    <span className="mt-1 block text-[14px] font-medium text-fg group-hover:text-accent">
                      <span className="mr-2 font-mono text-[12px] text-fg-faint">{numbers.get(prev.slug)}</span>
                      {prev.title}
                    </span>
                  </Link>
                ) : (
                  <span className="hidden sm:block" />
                )}
                {next && (
                  <Link
                    to={`/c/${next.slug}`}
                    className="group rounded-lg border border-line px-4 py-3 text-right transition-colors hover:border-line-strong hover:bg-tint/40"
                  >
                    <span className="flex items-center justify-end gap-1.5 text-[12px] text-fg-subtle">
                      Next
                      <ArrowRightIcon className="size-3.5" />
                    </span>
                    <span className="mt-1 block text-[14px] font-medium text-fg group-hover:text-accent">
                      <span className="mr-2 font-mono text-[12px] text-fg-faint">{numbers.get(next.slug)}</span>
                      {next.title}
                    </span>
                  </Link>
                )}
              </nav>
            )}
          </div>
        </div>
      </div>
    </div>
  );

  if (!hasLab) {
    return <div className="h-full min-h-0">{readingPane}</div>;
  }

  const labPane = (
    <LabPane lab={chapter.lab_detail!} onPassed={() => onProgress(slug, "done")} />
  );

  // Wide screens read on the left and build on the right. Narrow ones get the
  // same two panes as a pair of tabs, because a 380px column cannot hold both.
  if (wide) {
    return (
      <SplitPane
        direction="row"
        storageKey="lg.chapter.split"
        initial={50}
        min={25}
        max={75}
        className="h-full"
        label="Resize the chapter and the lab"
        first={readingPane}
        second={labPane}
      />
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="shrink-0 border-b border-line bg-well px-2 py-1.5" role="tablist">
        <div className="flex rounded-md border border-line bg-paper p-0.5">
          {(["read", "lab"] as const).map((option) => (
            <button
              key={option}
              role="tab"
              aria-selected={view === option}
              onClick={() => setView(option)}
              className={`flex h-8 flex-1 items-center justify-center gap-1.5 rounded text-[13px] font-medium transition-colors ${
                view === option ? "bg-card text-fg shadow-[0_0_0_1px_var(--line)]" : "text-fg-subtle hover:text-fg"
              }`}
            >
              {option === "read" ? <ListIcon className="size-3.5" /> : <FlaskIcon className="size-3.5" />}
              {option === "read" ? "Chapter" : "Lab"}
            </button>
          ))}
        </div>
      </div>
      {/* Both panes stay mounted: switching tabs must not throw away a draft
          or a run in flight. */}
      <div className="min-h-0 flex-1">
        <div className={`h-full ${view === "read" ? "" : "hidden"}`}>{readingPane}</div>
        <div className={`h-full ${view === "lab" ? "" : "hidden"}`}>{labPane}</div>
      </div>
    </div>
  );
}
