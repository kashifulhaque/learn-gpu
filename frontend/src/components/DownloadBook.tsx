import { useEffect, useState } from "react";

import { api, type BookFile } from "../lib/api";
import { DownloadIcon } from "./icons";

/**
 * The whole course as one PDF book: every chapter, every lab with its starter
 * file and hints, and the solutions in an appendix.
 *
 * The PDF is built at deploy time, so a server without one reports none, and
 * then the links below don't appear.
 */
export function useBook(): BookFile | null {
  const [book, setBook] = useState<BookFile | null>(null);
  useEffect(() => {
    api
      .book()
      .then((body) => setBook(body.book))
      .catch(() => undefined);
  }, []);
  return book;
}

/** Megabytes to one decimal place, which is all a download size needs. */
function size(bytes: number): string {
  return `${(bytes / 1e6).toFixed(1)} MB`;
}

function describe(book: BookFile): string {
  return `Download the whole course as a PDF book (${book.pages} pages, ${size(book.bytes)})`;
}

/** The sidebar's download row, styled like its navigation items. */
export function DownloadBookItem({ book }: { book: BookFile }) {
  return (
    <a
      href={book.url}
      download
      title={describe(book)}
      aria-label={describe(book)}
      className="flex h-9 items-center gap-2.5 rounded-md px-2.5 text-[13px] font-medium text-fg-muted transition-colors hover:bg-tint/70 hover:text-fg lg:h-8"
    >
      <span className="text-fg-subtle">
        <DownloadIcon />
      </span>
      <span className="flex-1">Course PDF</span>
      <span className="text-[11.5px] font-normal tabular-nums text-fg-faint">{book.pages} pages</span>
    </a>
  );
}

/** The same download on the collapsed rail, as an icon. */
export function DownloadBookRailItem({ book }: { book: BookFile }) {
  return (
    <a
      href={book.url}
      download
      title={describe(book)}
      aria-label={describe(book)}
      className="relative flex size-9 items-center justify-center rounded-md text-fg-subtle transition-colors hover:bg-tint hover:text-fg"
    >
      <DownloadIcon />
    </a>
  );
}
