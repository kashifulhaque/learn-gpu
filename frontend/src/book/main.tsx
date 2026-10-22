import { createRoot } from "react-dom/client";
import { MemoryRouter } from "react-router-dom";

import Book, { type BookData } from "./Book";
import "../index.css";
import "./book.css";

/**
 * The print layout's entry point. scripts/build-book.mjs serves the chapters
 * and labs as `book.json` beside this page, opens it, and prints it.
 *
 * The chapter bodies link to each other with router links, so the book sits
 * inside a router too; Book.tsx then points those links at the book's own
 * pages.
 */
async function start() {
  const data = (await (await fetch("book.json")).json()) as BookData;
  createRoot(document.getElementById("book")!).render(
    <MemoryRouter>
      <Book data={data} />
    </MemoryRouter>,
  );
}

start().catch((error: unknown) => {
  window.__book = { error: String(error) };
});
