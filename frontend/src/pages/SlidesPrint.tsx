import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import DeckPrint from "../components/DeckPrint";
import { api, type Deck } from "../lib/api";

/**
 * A deck laid out for printing: one slide per page, 16:9, answers showing.
 * The toolbar opens the browser's print dialog, where "Save as PDF" makes the
 * file. The page always uses the light theme, so the PDF doesn't spend ink on
 * a dark background.
 */
export default function SlidesPrint() {
  const { slug = "" } = useParams();
  const [deck, setDeck] = useState<Deck | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .deck(slug)
      .then(setDeck)
      .catch((e) => setError(String((e as Error).message ?? e)));
  }, [slug]);

  useEffect(() => {
    const root = document.documentElement;
    const before = root.getAttribute("data-theme");
    root.setAttribute("data-theme", "light");
    return () => {
      if (before) root.setAttribute("data-theme", before);
      else root.removeAttribute("data-theme");
    };
  }, []);

  return (
    <div className="slides-print-page h-full overflow-y-auto bg-well">
      <div className="print-toolbar sticky top-0 z-10 flex flex-wrap items-center gap-3 border-b border-line bg-well px-4 py-2 text-[13px]">
        <Link to={`/slides/${slug}`} className="text-accent underline underline-offset-2">
          Back to the slides
        </Link>
        <span className="text-fg-subtle">
          {deck ? `${deck.slides.length} slides, one per page, 16:9.` : error || "Loading…"}
        </span>
        <button type="button" onClick={() => window.print()} disabled={!deck} className="deck-btn deck-btn-primary ml-auto">
          Print or save as PDF
        </button>
      </div>
      {deck && <DeckPrint slides={deck.slides} label={deck.slug} />}
    </div>
  );
}
