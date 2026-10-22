import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import { DownloadIcon, PresentIcon } from "../components/icons";
import { Loading } from "../components/ui";
import { api, type BookFile, type DeckEntry } from "../lib/api";
import { splitPart } from "../lib/chapters";

function number(slug: string): string {
  return /^\d+[a-z]?/.exec(slug)?.[0] ?? "";
}

/** Every chapter's slide deck, grouped by part, with the way into each view of it. */
export default function SlidesIndex() {
  const [decks, setDecks] = useState<DeckEntry[] | null>(null);
  const [pdf, setPdf] = useState<BookFile | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .slides()
      .then((result) => setDecks(result.decks))
      .catch((e) => setError(String((e as Error).message ?? e)));
    api
      .book()
      .then((result) => setPdf(result.slides))
      .catch(() => undefined);
  }, []);

  const groups = useMemo(() => {
    const out = new Map<string, DeckEntry[]>();
    for (const entry of decks ?? []) {
      const list = out.get(entry.part);
      if (list) list.push(entry);
      else out.set(entry.part, [entry]);
    }
    return [...out.entries()];
  }, [decks]);

  if (error) return <p className="p-8 text-[14px] text-bad">{error}</p>;
  if (!decks) return <Loading label="Loading the slides" />;

  const ready = decks.filter((entry) => entry.deck).length;
  const total = decks.reduce((sum, entry) => sum + (entry.deck?.slides ?? 0), 0);

  return (
    <div className="mx-auto max-w-4xl px-4 pb-16 pt-8 sm:px-8 lg:pt-14">
      <header className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div className="max-w-2xl">
          <h1 className="font-serif text-[2rem] font-semibold leading-[1.15] tracking-[-0.01em] text-fg sm:text-[2.4rem]">
            Slides
          </h1>
          <p className="mt-3 text-[15px] leading-7 text-fg-muted">
            One deck for each chapter, with speaker notes. {ready} of {decks.length} decks are ready, {total} slides in all.
            Open a deck in the presenter view to see the notes and the next slide, or present it on the whole window.
          </p>
        </div>
        {pdf && (
          <a
            href={pdf.url}
            download
            className="deck-btn flex-none"
            title={`All decks as one PDF (${pdf.pages} pages, ${(pdf.bytes / 1e6).toFixed(1)} MB)`}
          >
            <DownloadIcon className="size-3.5" />
            Slides PDF
          </a>
        )}
      </header>

      <div className="mt-10 space-y-8">
        {groups.map(([part, entries]) => {
          const { label, name } = splitPart(part);
          return (
            <section key={part}>
              <div className="flex items-baseline gap-3 border-b border-line pb-2">
                {label && <span className="font-mono text-[12px] text-fg-faint">{label}</span>}
                <h2 className="text-[15px] font-semibold text-fg">{name}</h2>
              </div>
              <ol>
                {entries.map((entry) => (
                  <li key={entry.slug} className="border-b border-line/70 last:border-b-0">
                    <div className="my-2 flex flex-wrap items-center gap-x-4 gap-y-2 px-1 py-2">
                      <span className="w-8 shrink-0 font-mono text-[12.5px] tabular-nums text-fg-faint">{number(entry.slug)}</span>
                      <div className="min-w-0 flex-1">
                        <p className="text-[14px] font-medium text-fg">{entry.chapter_title}</p>
                        <p className="mt-0.5 text-[12.5px] text-fg-subtle">
                          {entry.deck ? `${entry.deck.title} · ${entry.deck.slides} slides` : "No deck yet"}
                        </p>
                      </div>
                      {entry.deck && (
                        <div className="flex flex-wrap items-center gap-2">
                          <Link to={`/slides/${entry.slug}`} className="deck-btn deck-btn-primary">
                            <PresentIcon className="size-3.5" />
                            Open
                          </Link>
                          <Link to={`/slides/${entry.slug}/present`} className="deck-btn">
                            Present
                          </Link>
                          <Link to={`/slides/${entry.slug}/print`} className="deck-btn">
                            Print
                          </Link>
                        </div>
                      )}
                    </div>
                  </li>
                ))}
              </ol>
            </section>
          );
        })}
      </div>
    </div>
  );
}
