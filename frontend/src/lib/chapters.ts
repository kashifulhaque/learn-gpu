import type { ChapterMeta } from "./api";

/**
 * The number a chapter goes by. The chapters refer to each other by the number
 * in their slug, so every list shows that rather than a position: the two
 * stopped agreeing once a chapter was inserted between 00 and 01.
 */
export function chapterNumber(chapter: ChapterMeta, index: number): string {
  return /^\d+[a-z]?/.exec(chapter.slug)?.[0] ?? String(index).padStart(2, "0");
}

export function chapterNumbers(chapters: ChapterMeta[]): Map<string, string> {
  return new Map(chapters.map((chapter, index) => [chapter.slug, chapterNumber(chapter, index)]));
}

/** Splits "Part 2 — A forward pass" into its label and its name. */
export function splitPart(part: string): { label: string; name: string } {
  const [label, ...rest] = part.split(/\s+—\s+/);
  return rest.length ? { label, name: rest.join(" — ") } : { label: "", name: part };
}

/** Chapters grouped by part, in course order. */
export function groupByPart(chapters: ChapterMeta[]): [string, ChapterMeta[]][] {
  const groups = new Map<string, ChapterMeta[]>();
  for (const chapter of chapters) {
    const group = groups.get(chapter.part);
    if (group) group.push(chapter);
    else groups.set(chapter.part, [chapter]);
  }
  return [...groups.entries()];
}

/** What a chapter's lab marker says when pointed at. */
export function labTitle(chapter: ChapterMeta): string {
  return chapter.gpu ? "Has a lab that runs on a GPU" : "Has a lab";
}
