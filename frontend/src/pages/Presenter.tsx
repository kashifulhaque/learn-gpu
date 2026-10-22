import { useCallback, useEffect, useRef, useState, type MouseEvent } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import Markdown from "../components/Markdown";
import SlideFrame from "../components/SlideFrame";
import { api, type Deck } from "../lib/api";

/** True when a key press belongs to a field rather than to the deck. */
function typing(target: EventTarget | null): boolean {
  const node = target as HTMLElement | null;
  return !!node && (node.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(node.tagName));
}

/** The slide number in the address, counted from 1, so a reload stays put. */
function slideFromHash(): number {
  const n = Number(window.location.hash.slice(1));
  return Number.isInteger(n) && n > 0 ? n - 1 : 0;
}

function clock(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function Message({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 bg-paper p-8 text-center text-[14px] text-fg-muted">
      {children}
    </div>
  );
}

/**
 * A chapter's deck on a full-window stage, in one of two modes.
 *
 * - `presenter`: the slide, the speaker notes, and a preview of the next slide
 *   side by side. It is the view the speaker keeps on their own screen.
 * - `present`: the slide alone, as large as the window allows, for the
 *   projector. Open it in full screen with **F**.
 *
 * Both move with the keyboard: arrows, space, Page Up and Page Down, Home and
 * End. **B** blacks the screen out, and **P** switches between the two modes
 * at the same slide.
 */
export default function Presenter({ mode }: { mode: "presenter" | "present" }) {
  const { slug = "" } = useParams();
  const [deck, setDeck] = useState<Deck | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    setDeck(null);
    setError("");
    api
      .deck(slug)
      .then(setDeck)
      .catch((e) => setError(String((e as Error).message ?? e)));
  }, [slug]);

  if (error) {
    return (
      <Message>
        <p>{error}</p>
        <Link to={`/c/${slug}`} className="text-accent underline underline-offset-2">
          Back to the chapter
        </Link>
      </Message>
    );
  }
  if (!deck) return <Message>Loading the slides…</Message>;
  return <Stage key={`${deck.slug}-${mode}`} deck={deck} mode={mode} />;
}

function Stage({ deck, mode }: { deck: Deck; mode: "presenter" | "present" }) {
  const navigate = useNavigate();
  const slides = deck.slides;
  const last = slides.length - 1;
  const [index, setIndex] = useState(() => Math.min(slideFromHash(), Math.max(0, last)));
  const [notes, setNotes] = useState(true);
  const [black, setBlack] = useState(false);
  const [full, setFull] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const frame = useRef<HTMLDivElement>(null);
  const wasFull = useRef(!!document.fullscreenElement);

  const go = useCallback((to: number) => setIndex(Math.max(0, Math.min(last, to))), [last]);

  useEffect(() => {
    window.history.replaceState(window.history.state, "", `${window.location.pathname}#${index + 1}`);
    frame.current?.scrollTo({ top: 0 });
  }, [index]);

  useEffect(() => {
    const timer = window.setInterval(() => setElapsed((value) => value + 1), 1000);
    return () => window.clearInterval(timer);
  }, []);

  const base = `/slides/${deck.slug}`;
  const exit = useCallback(() => navigate(mode === "present" ? `${base}#${index + 1}` : `/c/${deck.slug}`), [navigate, mode, base, index, deck.slug]);
  const switchMode = useCallback(() => {
    // The whole document goes full screen, so it survives the change of mode.
    if (mode === "presenter" && !document.fullscreenElement) {
      void document.documentElement.requestFullscreen?.().catch(() => undefined);
    }
    navigate(mode === "present" ? `${base}#${index + 1}` : `${base}/present#${index + 1}`);
  }, [navigate, mode, base, index]);

  useEffect(() => {
    const onChange = () => {
      const now = !!document.fullscreenElement;
      setFull(now);
      // Leaving full screen from present mode, with Escape, returns to the
      // presenter view rather than leaving the reader on a bare stage.
      if (!now && wasFull.current && mode === "present") exit();
      wasFull.current = now;
    };
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, [exit, mode]);

  const toggleFull = useCallback(() => {
    if (document.fullscreenElement) void document.exitFullscreen().catch(() => undefined);
    else void document.documentElement.requestFullscreen?.().catch(() => undefined);
  }, []);

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.metaKey || event.ctrlKey || event.altKey || typing(event.target)) return;
      if (black && !["b", "B", "."].includes(event.key)) {
        // Any key first lifts the blackout, and does nothing else.
        event.preventDefault();
        setBlack(false);
        return;
      }
      switch (event.key) {
        case "ArrowRight":
        case "ArrowDown":
        case "PageDown":
        case " ":
          event.preventDefault();
          go(index + (event.shiftKey && event.key === " " ? -1 : 1));
          break;
        case "ArrowLeft":
        case "ArrowUp":
        case "PageUp":
        case "Backspace":
          event.preventDefault();
          go(index - 1);
          break;
        case "Home":
          event.preventDefault();
          go(0);
          break;
        case "End":
          event.preventDefault();
          go(last);
          break;
        case "n":
        case "N":
          if (mode === "presenter") setNotes((value) => !value);
          break;
        case "f":
        case "F":
          toggleFull();
          break;
        case "p":
        case "P":
          switchMode();
          break;
        case "b":
        case "B":
        case ".":
          setBlack((value) => !value);
          break;
        case "Escape":
          // In full screen the browser takes Escape to leave it.
          if (!document.fullscreenElement) exit();
          break;
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [black, exit, go, index, last, mode, switchMode, toggleFull]);

  // A click on the slide moves on, and a click on its left quarter goes back,
  // unless the click was meant for a link, a control, or a figure, or ended a
  // text selection.
  function onStageClick(event: MouseEvent<HTMLDivElement>) {
    const target = event.target as HTMLElement;
    if (target.closest("a, button, input, select, textarea, label, summary, .viz, .code-block")) return;
    if (window.getSelection()?.toString()) return;
    const box = event.currentTarget.getBoundingClientRect();
    go(event.clientX < box.left + box.width / 4 ? index - 1 : index + 1);
  }

  const slide = slides[index];
  const next = slides[index + 1];
  const progress = last > 0 ? index / last : 1;

  const stage = (
    <div
      className="slide-stage cursor-pointer select-text"
      onClick={onStageClick}
      role="group"
      aria-roledescription="slide"
      aria-label={`Slide ${index + 1} of ${slides.length}`}
    >
      <SlideFrame key={slide.id} slide={slide} frame={frame} />
    </div>
  );

  const blackout = black && (
    <button
      type="button"
      onClick={() => setBlack(false)}
      aria-label="End the blackout"
      className="fixed inset-0 z-50 cursor-default bg-black"
    />
  );

  if (mode === "present") {
    return (
      <div className="presenter fixed inset-0 bg-black">
        <div className="presenter-well absolute inset-0 p-0">{stage}</div>
        <p
          className="pointer-events-none absolute bottom-2 right-3 text-[12px] tabular-nums text-white/40"
          aria-live="polite"
        >
          {index + 1} / {slides.length}
        </p>
        {blackout}
      </div>
    );
  }

  return (
    <div className="presenter flex h-full flex-col bg-paper">
      <header className="flex h-12 shrink-0 items-center gap-1.5 border-b border-line bg-well px-2 lg:h-11">
        <Link to={`/c/${deck.slug}`} className="icon-btn" aria-label="Back to the chapter" title="Back to the chapter (Esc)">
          <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden>
            <path d="M3.5 3.5l7 7M10.5 3.5l-7 7" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
          </svg>
        </Link>
        <p className="ml-1 min-w-0 flex-1 truncate text-[13px] font-medium text-fg-muted">
          {deck.title}
          {deck.chapter_title !== deck.title && (
            <span className="hidden font-normal text-fg-subtle sm:inline"> · {deck.chapter_title}</span>
          )}
        </p>
        <button
          type="button"
          onClick={() => setElapsed(0)}
          title="Elapsed time. Click to reset."
          className="hidden px-2 font-mono text-[12px] tabular-nums text-fg-subtle sm:block"
        >
          {clock(elapsed)}
        </button>
        <button type="button" onClick={() => setNotes((value) => !value)} aria-pressed={notes} title="Speaker notes (N)" className="deck-btn">
          Notes
        </button>
        <Link to={`/slides/${deck.slug}/print`} className="deck-btn" title="Print layout, one slide per page">
          Print
        </Link>
        <button type="button" onClick={switchMode} className="deck-btn deck-btn-primary" title="Present on the whole window (P)">
          Present
        </button>
        <button
          type="button"
          onClick={toggleFull}
          aria-pressed={full}
          aria-label={full ? "Leave full screen" : "Full screen"}
          title="Full screen (F)"
          className="icon-btn"
        >
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden>
            <path
              d="M1.5 4.5v-3h3M9.5 1.5h3v3M12.5 9.5v3h-3M4.5 12.5h-3v-3"
              stroke="currentColor"
              strokeWidth="1.4"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      </header>
      <div className="h-0.5 shrink-0" aria-hidden>
        <div className="h-full origin-left bg-accent transition-transform duration-150 ease-out" style={{ transform: `scaleX(${progress})` }} />
      </div>

      <div className={`flex min-h-0 flex-1 ${notes ? "flex-col lg:flex-row" : "flex-col"}`}>
        <div className="presenter-well min-h-0 flex-1 p-3 sm:p-6">{stage}</div>

        {notes && (
          <aside
            aria-label="Speaker notes"
            className="max-h-[45%] shrink-0 overflow-y-auto overscroll-contain border-t border-line bg-well px-5 py-4 lg:max-h-none lg:w-[26rem] lg:border-l lg:border-t-0"
          >
            {next && (
              <div className="mb-4">
                <p className="mb-1.5 text-[12px] font-semibold text-fg-subtle">Next</p>
                <button
                  type="button"
                  onClick={() => go(index + 1)}
                  className="slide-stage slide-thumb block w-full cursor-pointer text-left"
                  aria-label={`Next slide: ${next.title}`}
                >
                  <SlideFrame key={next.id} slide={next} />
                </button>
              </div>
            )}
            <p className="text-[12px] font-semibold text-fg-subtle">Speaker notes</p>
            <div className="slide-notes mt-2">
              {slide.notes.trim() ? (
                <Markdown variant="slide">{slide.notes}</Markdown>
              ) : (
                <p className="text-[13px] text-fg-faint">No notes for this slide.</p>
              )}
            </div>
          </aside>
        )}
      </div>

      {blackout}

      <footer className="flex h-12 shrink-0 items-center gap-2 border-t border-line bg-well px-2 lg:h-11">
        <button type="button" onClick={() => go(index - 1)} disabled={index === 0} className="deck-btn">
          <span aria-hidden>←</span> Previous
        </button>
        <p className="flex-1 text-center text-[12.5px] tabular-nums text-fg-subtle" aria-live="polite">
          {index + 1} / {slides.length}
        </p>
        <span className="hidden text-[11.5px] text-fg-faint lg:inline">
          <kbd className="deck-kbd">←</kbd> <kbd className="deck-kbd">→</kbd> move · <kbd className="deck-kbd">N</kbd> notes ·{" "}
          <kbd className="deck-kbd">P</kbd> present · <kbd className="deck-kbd">B</kbd> blackout ·{" "}
          <kbd className="deck-kbd">F</kbd> full screen
        </span>
        <button type="button" onClick={() => go(index + 1)} disabled={index >= last} className="deck-btn">
          Next <span aria-hidden>→</span>
        </button>
      </footer>
    </div>
  );
}
