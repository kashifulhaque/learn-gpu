import { createRoot } from "react-dom/client";
import { MemoryRouter } from "react-router-dom";

import DeckPrint from "../components/DeckPrint";
import type { Slide } from "../lib/api";
import "../index.css";

/**
 * The slides PDF's entry point. scripts/build-slides.mjs serves every deck as
 * `slides.json` beside this page, opens it in headless Chrome, waits for
 * `window.__slides.ready`, and prints it. Every slide is one 16:9 page, so the
 * page a deck starts on is the number of slides before it, plus one.
 */
type DeckData = { slug: string; chapter_title: string; part: string; title: string; slides: Slide[] };

declare global {
  interface Window {
    __slides?: {
      ready?: boolean;
      error?: string;
      decks?: { slug: string; title: string; chapter_title: string; slides: number; sections: { title: string; at: number }[] }[];
      katexErrors?: string[];
      diagramErrors?: string[];
    };
  }
}

/** Resolves once fonts are loaded and no diagram is still drawing. */
async function settled(): Promise<void> {
  await document.fonts.ready;
  const started = Date.now();
  let calm = 0;
  while (Date.now() - started < 120_000) {
    const busy = document.querySelectorAll('.mermaid-block[aria-busy="true"]').length;
    calm = busy === 0 ? calm + 1 : 0;
    // Several quiet polls in a row, so a diagram that starts late is caught.
    if (calm >= 6) return;
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
}

async function start() {
  const data = (await (await fetch("slides.json")).json()) as { decks: DeckData[] };
  const root = document.getElementById("slides")!;
  createRoot(root).render(
    <MemoryRouter>
      {data.decks.map((deck) => (
        <section key={deck.slug} data-deck={deck.slug}>
          <DeckPrint slides={deck.slides} label={deck.slug} />
        </section>
      ))}
    </MemoryRouter>,
  );
  await settled();
  let at = 1;
  window.__slides = {
    ready: true,
    decks: data.decks.map((deck) => {
      const sections = deck.slides.flatMap((slide, i) => (slide.kind === "section" ? [{ title: slide.title, at: at + i }] : []));
      const entry = { slug: deck.slug, title: deck.title, chapter_title: deck.chapter_title, slides: deck.slides.length, sections };
      at += deck.slides.length;
      return entry;
    }),
    katexErrors: Array.from(document.querySelectorAll<HTMLElement>(".katex-error")).map(
      (node) => `${node.closest("[data-where]")?.getAttribute("data-where") ?? "?"}: ${node.title || node.textContent}`,
    ),
    diagramErrors: Array.from(document.querySelectorAll<HTMLElement>(".viz-missing")).map(
      (node) => `${node.closest("[data-where]")?.getAttribute("data-where") ?? "?"}: ${(node.textContent ?? "").slice(0, 100)}`,
    ),
  };
}

start().catch((error: unknown) => {
  window.__slides = { error: String(error) };
});
