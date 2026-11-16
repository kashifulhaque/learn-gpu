/**
 * Builds the shared shell components' props from the learn-gpu chapter
 * index. This file is course-specific: each course writes its own version.
 */

import { useEffect, useMemo, useState } from "react";

import type { SearchEntry } from "../components/SearchDialog";
import { api, type ChapterMeta } from "./api";
import { chapterNumbers, splitPart } from "./chapters";

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

let headingCache: Promise<Map<string, string[]>> | null = null;

/** Every chapter's `##` headings. The chapter index doesn't carry them, so this loads each body once. */
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

/**
 * Search entries for every chapter. Titles and summaries are there at once;
 * section headings join them after the search first opens and every chapter
 * body has loaded. A heading links to its section as `/c/<slug>#s-<n>`.
 */
export function useSearchEntries(chapters: ChapterMeta[], open: boolean): SearchEntry[] {
  const [sections, setSections] = useState<Map<string, string[]>>(new Map());

  useEffect(() => {
    if (!open || chapters.length === 0) return;
    let live = true;
    void loadHeadings(chapters).then((found) => {
      if (live) setSections(found);
    });
    return () => {
      live = false;
    };
  }, [open, chapters]);

  return useMemo(() => {
    const numbers = chapterNumbers(chapters);
    return chapters.map((chapter) => {
      const href = `/c/${chapter.slug}`;
      return {
        id: chapter.slug,
        href,
        title: chapter.title,
        kicker: `Chapter ${numbers.get(chapter.slug) ?? ""} · ${splitPart(chapter.part).name}`,
        summary: chapter.summary,
        keywords: [...chapter.objectives, chapter.part, chapter.slug],
        headings: (sections.get(chapter.slug) ?? []).map((text, index) => ({
          text,
          href: `${href}#s-${index + 1}`,
        })),
      };
    });
  }, [chapters, sections]);
}

/** The chapter to open next: the first one not marked done. */
export function nextChapter(
  chapters: ChapterMeta[],
  progress: Record<string, string>,
): ChapterMeta | undefined {
  return chapters.find((chapter) => progress[chapter.slug] !== "done");
}

export function doneCount(chapters: ChapterMeta[], progress: Record<string, string>): number {
  return chapters.filter((chapter) => progress[chapter.slug] === "done").length;
}

/** "Part 3 — Matrix multiplication" → "03", or the 1-based index when unnumbered. */
export function partNumber(part: string, index: number): string {
  const digits = /\d+/.exec(splitPart(part).label)?.[0];
  return (digits ?? String(index + 1)).padStart(2, "0");
}
