/**
 * The PDF book: every chapter and lab on one long page, laid out for print.
 *
 * scripts/build-book.mjs opens this page in headless Chrome, waits for
 * `window.__book.ready`, prints it once to find out where everything landed,
 * fills the page numbers in with `window.__setPages`, and prints it again.
 * The chapter bodies go through the same Markdown component the site uses, so
 * the book and the site cannot drift apart.
 *
 * Each chapter opens on a page of its own with what it promises and its
 * sections, then runs through its body. Its lab sits inside the chapter's Lab
 * section: the brief, the starter file, and the hints. The worked solutions
 * sit together in an appendix, so they are never on the page facing the lab
 * they give away.
 */

import hljs from "highlight.js/lib/core";
import python from "highlight.js/lib/languages/python";
import { useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";

import { Mark } from "../components/Brand";
import Markdown from "../components/Markdown";
import type { ChapterMeta } from "../lib/api";
import { chapterNumber, splitPart } from "../lib/chapters";

hljs.registerLanguage("python", python);

export const SITE = "gpu.ifkash.dev";
const TITLE = "GPU programming for ML";

export type BookChapter = ChapterMeta & { body: string };
export type BookLab = {
  id: string;
  title: string;
  brief: string;
  gpu: string;
  needs_gpu: boolean;
  hints: string[];
  metrics: string[];
  starter: string;
  solution: string;
};
export type BookData = {
  commit: string;
  gpu: string;
  chapters: BookChapter[];
  labs: Record<string, BookLab>;
};

/** One bookmark in the PDF's outline. `id` is the anchor it points at. */
export type OutlineEntry = { id: string; title: string; children: OutlineEntry[] };

export type BookReport = {
  error?: string;
  ready?: boolean;
  title?: string;
  subject?: string;
  outline?: OutlineEntry[];
  katexErrors?: string[];
  overflow?: string[];
};

declare global {
  interface Window {
    __book?: BookReport;
    __setPages?: (pages: Record<string, number>) => void;
  }
}

const chapterId = (slug: string) => `ch-${slug}`;
const bodyId = (slug: string) => `ch-${slug}--body`;
const sectionId = (slug: string, index: number) => `${bodyId(slug)}--s-${index}`;
const labId = (slug: string) => `lab-${slug}`;
const solutionId = (slug: string) => `sol-${slug}`;
const pageName = (slug: string) => `pg-${slug}`;

type Numbered = { chapter: BookChapter; n: string; lab?: BookLab };
type Section = { index: number; title: string };

/** Two digits, the way the site numbers a chapter's sections. */
function pad(value: number): string {
  return String(value).padStart(2, "0");
}

/** A CSS string literal, for the running heads. */
function cssString(text: string): string {
  return `"${text.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\n/g, " ")}"`;
}

/** Heading text without its Markdown: backticks, emphasis, and dollar signs. */
function plain(text: string): string {
  return text.replace(/[`*_]/g, "").replace(/==/g, "").replace(/\$([^$]*)\$/g, "$1");
}

/**
 * The chapter's `##` headings, numbered the way rehypeSectionize numbers them.
 * Headings inside a fenced block or a callout are not sections.
 */
function sections(body: string): Section[] {
  const out: Section[] = [];
  let fence: string | null = null;
  for (const line of body.split("\n")) {
    const marker = /^\s*(`{3,}|~{3,})/.exec(line)?.[1];
    if (marker) {
      if (!fence) fence = marker;
      else if (marker.startsWith(fence)) fence = null;
      continue;
    }
    if (fence) continue;
    const heading = /^## +(.+?)\s*#*\s*$/.exec(line);
    if (heading) out.push({ index: out.length + 1, title: heading[1] });
  }
  return out;
}

/** True for the chapter's Lab section, which the lab itself is printed in. */
function isLabSection(section: Section): boolean {
  return plain(section.title).trim().toLowerCase() === "lab";
}

/**
 * Where a section's entry in the contents and the outline points. The Lab
 * section points at the lab, which names it.
 */
function sectionTarget(slug: string, section: Section, lab?: BookLab) {
  if (lab && isLabSection(section)) return { id: labId(slug), title: `Lab: ${lab.title}`, isLab: true };
  return { id: sectionId(slug, section.index), title: plain(section.title), isLab: false };
}

/** Chapters grouped into parts, in the order the parts first appear. */
function groupParts(chapters: Numbered[]) {
  const parts: { part: string; number: string; name: string; chapters: Numbered[] }[] = [];
  for (const entry of chapters) {
    let last = parts[parts.length - 1];
    if (!last || last.part !== entry.chapter.part) {
      const { label, name } = splitPart(entry.chapter.part);
      const number = /\d+/.exec(label)?.[0] ?? String(parts.length + 1);
      last = { part: entry.chapter.part, number, name, chapters: [] };
      parts.push(last);
    }
    last.chapters.push(entry);
  }
  return parts;
}

/** Waits for the fonts and the layout to settle. */
async function settled(): Promise<void> {
  await document.fonts.ready;
  await new Promise((resolve) => window.setTimeout(resolve, 300));
  await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
}

/**
 * Print has no horizontal scroll, so anything wider than the text block is
 * shrunk until it fits, and whatever still does not fit is reported.
 */
function fitToMeasure(): string[] {
  const overflow: string[] = [];
  // `box` takes the smaller type, `inner` is what has to fit, and `width` is
  // the room it has.
  const shrink = (box: HTMLElement, inner: HTMLElement, width: number, what: string, floor: number) => {
    let scale = 1;
    for (let step = 0; step < 8 && inner.scrollWidth > width + 1 && scale > floor; step += 1) {
      scale = Math.max(floor, scale * Math.min(0.97, width / inner.scrollWidth));
      box.style.fontSize = `${scale}em`;
    }
    if (inner.scrollWidth > width + 1) {
      const where = box.closest("[data-where]")?.getAttribute("data-where") ?? "?";
      overflow.push(`${where}: ${what}: ${(box.textContent ?? "").slice(0, 70)}`);
    }
  };
  document.querySelectorAll<HTMLElement>(".book .katex-display").forEach((display) => {
    const katex = display.querySelector<HTMLElement>(".katex");
    if (katex) shrink(display, katex, display.clientWidth, "display maths", 0.55);
  });
  document.querySelectorAll<HTMLElement>(".book .table-wrap").forEach((wrap) => {
    const table = wrap.querySelector<HTMLElement>("table");
    if (table) shrink(table, table, wrap.clientWidth, "table", 0.66);
  });
  return overflow;
}

export default function Book({ data }: { data: BookData }) {
  const numbered: Numbered[] = data.chapters.map((chapter, index) => ({
    chapter,
    n: chapterNumber(chapter, index),
    lab: chapter.lab ? data.labs[chapter.lab] : undefined,
  }));
  const slugs = new Set(numbered.map((entry) => entry.chapter.slug));
  const parts = groupParts(numbered);
  const withSolutions = numbered.filter((entry) => entry.lab?.solution);
  const labs = numbered.filter((entry) => entry.lab).length;

  useEffect(() => {
    document.title = TITLE;

    window.__setPages = (pages) => {
      document.querySelectorAll<HTMLElement>("[data-pg]").forEach((node) => {
        const page = pages[node.dataset.pg ?? ""];
        node.textContent = page ? String(page) : "";
      });
    };

    const outline: OutlineEntry[] = [
      { id: "contents", title: "Contents", children: [] },
      { id: "about", title: "How to use this book", children: [] },
      ...parts.map((part) => ({
        id: `part-${part.number}`,
        title: `Part ${part.number}. ${part.name}`,
        children: part.chapters.map(({ chapter, n, lab }) => ({
          id: chapterId(chapter.slug),
          title: `${n}. ${chapter.title}`,
          children: sections(chapter.body).map((section) => {
            const { id, title } = sectionTarget(chapter.slug, section, lab);
            return { id, title: `${pad(section.index)} ${title}`, children: [] };
          }),
        })),
      })),
      {
        id: "solutions",
        title: "Appendix. Solutions",
        children: withSolutions.map(({ chapter, n, lab }) => ({
          id: solutionId(chapter.slug),
          title: `${n}. ${lab!.title}`,
          children: [],
        })),
      },
    ];

    let live = true;
    settled().then(() => {
      if (!live) return;
      const overflow = fitToMeasure();
      window.__book = {
        ready: true,
        title: TITLE,
        subject: `A course in ${numbered.length} chapters and ${labs} labs, from the streaming multiprocessor to the kernels that train and serve models, with every lab's starter file and solution.`,
        outline,
        katexErrors: Array.from(document.querySelectorAll<HTMLElement>(".book .katex-error")).map(
          (node) => `${node.closest("[data-where]")?.getAttribute("data-where") ?? "?"}: ${node.title || node.textContent}`,
        ),
        overflow,
      };
    });
    return () => {
      live = false;
    };
    // The book renders once per page load.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // The running heads: the book's title on the left-hand page, the chapter on
  // the right. Each chapter is its own named page so its right-hand head can
  // say which chapter it is.
  const pageRules = numbered
    .map(({ chapter, n }) => {
      const name = pageName(chapter.slug);
      return `.${name}{page:${name}}@page ${name}:right{@top-right{content:${cssString(`${n} · ${chapter.title}`)}}}`;
    })
    .join("\n");
  // Chrome lets a later :left rule beat an earlier named page, whatever the
  // specificity, so the pages without a running head say so again here, one
  // rule each, because Chrome ignores a list of page names.
  const heads = [
    `@page :left{@top-left{content:${cssString(TITLE)}}}`,
    `@page solutions:right{@top-right{content:"Appendix · Solutions"}}`,
    ...["cover", "front", "part", "opener"].map((name) => `@page ${name}{@top-left{content:none}@top-right{content:none}}`),
  ].join("\n");

  return (
    <div className="book">
      <style>{`${heads}\n${pageRules}`}</style>

      <Cover parts={parts} chapters={numbered.length} labs={labs} data={data} />
      <Contents parts={parts} solutions={withSolutions.length > 0} />
      <About chapters={numbered.length} labs={labs} gpu={data.gpu} />

      {parts.map((part) => (
        <div key={part.part}>
          <section id={`part-${part.number}`} className="part-page">
            <p className="part-kicker">Part {part.number}</p>
            <h1 className="part-title">{part.name}</h1>
            <ol className="part-chapters">
              {part.chapters.map(({ chapter, n }) => (
                <li key={chapter.slug}>
                  <span className="part-chapter-n">{n}</span>
                  <span className="part-chapter-text">
                    <span className="part-chapter-title">{chapter.title}</span>
                    {chapter.summary && <span className="part-chapter-q">{chapter.summary}</span>}
                  </span>
                  <a href={`#${chapterId(chapter.slug)}`} className="part-chapter-pg">
                    <span className="pg" data-pg={chapterId(chapter.slug)} />
                  </a>
                </li>
              ))}
            </ol>
          </section>

          {part.chapters.map((entry) => (
            <Chapter key={entry.chapter.slug} entry={entry} part={part} slugs={slugs} />
          ))}
        </div>
      ))}

      {withSolutions.length > 0 && <Solutions entries={withSolutions} />}
    </div>
  );
}

function Cover({
  parts,
  chapters,
  labs,
  data,
}: {
  parts: ReturnType<typeof groupParts>;
  chapters: number;
  labs: number;
  data: BookData;
}) {
  const built = new Date().toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" });
  return (
    <section className="cover">
      <div className="cover-top">
        <Mark className="size-11" />
        <span className="cover-site">{SITE}</span>
      </div>
      <div className="cover-main">
        <p className="cover-kicker">
          {chapters} chapters and {labs} labs
        </p>
        <h1 className="cover-title">{TITLE}</h1>
        <p className="cover-subtitle">
          From the streaming multiprocessor to the kernels that train and serve models
        </p>
        <div className="cover-edition">
          <p className="cover-edition-name">The complete course</p>
          <p className="cover-edition-blurb">
            Every chapter, every lab with its starter file and hints, and the worked solutions at the back
            {data.gpu ? `. The labs run on an ${data.gpu}.` : "."}
          </p>
        </div>
      </div>
      <div className="cover-foot">
        <ol className="cover-parts">
          {parts.map((part) => (
            <li key={part.part}>
              <span className="cover-parts-n">{part.number}</span>
              {part.name}
            </li>
          ))}
        </ol>
        <p className="cover-built">
          Built {built}
          {data.commit && data.commit !== "unknown" ? ` from ${data.commit.slice(0, 7)}` : ""}
        </p>
      </div>
    </section>
  );
}

function Contents({ parts, solutions }: { parts: ReturnType<typeof groupParts>; solutions: boolean }) {
  return (
    <nav id="contents" className="front toc" aria-label="Contents">
      <h1 className="front-title">Contents</h1>
      <a href="#about" className="toc-row toc-front">
        <span className="toc-title">How to use this book</span>
        <span className="toc-leader" />
        <span className="pg" data-pg="about" />
      </a>
      {parts.map((part) => (
        <div key={part.part} className="toc-part">
          <a href={`#part-${part.number}`} className="toc-part-title">
            <span className="toc-part-n">Part {part.number}</span>
            <span className="toc-title">{part.name}</span>
            <span className="toc-leader" />
            <span className="pg" data-pg={`part-${part.number}`} />
          </a>
          {part.chapters.map(({ chapter, n }) => (
            <div key={chapter.slug} className="toc-chapter">
              <a href={`#${chapterId(chapter.slug)}`} className="toc-row">
                <span className="toc-n">{n}</span>
                <span className="toc-title">{chapter.title}</span>
                <span className="toc-leader" />
                <span className="pg" data-pg={chapterId(chapter.slug)} />
              </a>
            </div>
          ))}
        </div>
      ))}
      {solutions && (
        <div className="toc-part">
          <a href="#solutions" className="toc-part-title">
            <span className="toc-part-n">Appendix</span>
            <span className="toc-title">Solutions to the labs</span>
            <span className="toc-leader" />
            <span className="pg" data-pg="solutions" />
          </a>
        </div>
      )}
    </nav>
  );
}

function About({ chapters, labs, gpu }: { chapters: number; labs: number; gpu: string }) {
  return (
    <section id="about" className="front about">
      <h1 className="front-title">How to use this book</h1>
      <div className="about-prose">
        <p>
          This course teaches you to program the GPU for machine learning by having you write the kernels: reductions,
          matrix multiplication on tensor cores, attention, and the collectives that join many GPUs. There are{" "}
          {chapters} chapters and {labs} labs. Each chapter teaches one idea well enough that you can build it, and its
          lab is where you do.
        </p>
        <p>
          Every number in a chapter is arithmetic from the hardware&apos;s published specifications, cited to its
          source, or measured on the lab&apos;s A100. Where a chapter cannot measure, it says it is predicting.
        </p>
      </div>

      <dl className="about-list">
        <div>
          <dt>Each chapter</dt>
          <dd>
            Opens on a page of its own with what you will be able to do and its sections, with their page numbers. The
            body starts with a summary you can read in a minute. It ends with what goes wrong, what to remember,
            questions to check yourself with, the lab, and further reading.
          </dd>
        </div>
        <div>
          <dt>Each lab</dt>
          <dd>
            Sits in its chapter&apos;s Lab section, with the brief, the whole starter file, and the hints. The starter
            file is the specification: its docstrings and comments say what each piece takes and returns.
          </dd>
        </div>
        <div>
          <dt>Running a lab</dt>
          <dd>
            Happens online. Open the chapter at {SITE}, write your code in the editor beside it, and run it. The lab
            runs {gpu ? `on an ${gpu}` : "on a GPU"} and streams its checks and measurements back as they finish.
          </dd>
        </div>
        <div>
          <dt>The solutions</dt>
          <dd>
            Are in the appendix, one per lab, and each lab names the page its solution is on. Try the lab first: any
            code that passes the checks is right, and the reference solution is one answer among many.
          </dd>
        </div>
      </dl>

      <div className="about-prose about-notes">
        <p>
          <strong className="runin">Boxes.</strong> A box with a coloured edge marks a key idea, a plain-language
          reading of the maths, a worked example with real hardware numbers, a tip, or a warning. Deep dives
          and self-check questions fold away on the site; here they are printed in full, so cover the answer if you
          want to test yourself.
        </p>
        <p>
          <strong className="runin">Highlights and colours.</strong> A highlighted phrase is the one a skimming reader
          must not miss. In an equation with several named parts, each part keeps the same colour in the prose that
          explains it.
        </p>
        <p>
          <strong className="runin">Code.</strong> In the starter files and solutions, a line too long for the page
          carries on below, indented further than the line it belongs to. Type it as one line.
        </p>
      </div>
    </section>
  );
}

/**
 * Makes a rendered chapter body print-ready: prefixes every id, since every
 * chapter numbers its sections from 1; points links to other chapters at
 * their pages in the book; and opens every collapsed callout.
 */
function usePrintBody(slug: string, slugs: Set<string>) {
  const ref = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const root = ref.current;
    if (!root) return;
    const prefix = bodyId(slug);
    root.querySelectorAll<HTMLElement>("[id]").forEach((node) => {
      if (!node.id.startsWith(`${prefix}--`)) node.id = `${prefix}--${node.id}`;
    });
    root.querySelectorAll<HTMLAnchorElement>('a[href^="/c/"]').forEach((link) => {
      const target = link.getAttribute("href")!.slice(3).split(/[?#]/)[0];
      if (slugs.has(target)) link.setAttribute("href", `#${chapterId(target)}`);
    });
    root.querySelectorAll("details").forEach((details) => {
      details.open = true;
    });
  }, [slug, slugs]);
  return ref;
}

function Chapter({
  entry,
  part,
  slugs,
}: {
  entry: Numbered;
  part: ReturnType<typeof groupParts>[number];
  slugs: Set<string>;
}) {
  const { chapter, n, lab } = entry;
  const outline = sections(chapter.body);
  const bodyRef = usePrintBody(chapter.slug, slugs);
  // The lab goes at the end of the chapter's Lab section, after the callout
  // that says what to build, rather than after the further reading.
  const [labHost, setLabHost] = useState<HTMLElement | null>(null);
  const body = useMemo(() => <Markdown>{chapter.body}</Markdown>, [chapter.body]);

  useLayoutEffect(() => {
    if (!lab || !bodyRef.current) return;
    const heading = Array.from(bodyRef.current.querySelectorAll<HTMLElement>(".chapter-section > h2")).find(
      (node) => node.textContent?.trim().toLowerCase() === "lab",
    );
    if (heading) {
      const host = document.createElement("div");
      heading.parentElement!.appendChild(host);
      setLabHost(host);
    }
  }, [lab, bodyRef]);

  const labBlock = lab ? <Lab slug={chapter.slug} n={n} lab={lab} /> : null;

  return (
    <article className="chapter">
      <header id={chapterId(chapter.slug)} className="opener">
        <p className="opener-kicker">
          <span className="opener-n">Chapter {n}</span>
          <span>{`Part ${part.number}, ${part.name}`}</span>
          {chapter.minutes ? <span className="opener-minutes">About {chapter.minutes} minutes</span> : null}
        </p>
        <h1 className="opener-title">{chapter.title}</h1>
        {chapter.summary && <p className="opener-question">{chapter.summary}</p>}

        {chapter.objectives.length > 0 && (
          <div className="opener-box">
            <p className="mini-toc-label">By the end, you can</p>
            <ul className="opener-objectives">
              {chapter.objectives.map((objective) => (
                <li key={objective}>{objective}</li>
              ))}
            </ul>
          </div>
        )}

        {outline.length > 0 && (
          <div className="opener-box opener-levels">
            <p className="mini-toc-label">In this chapter</p>
            <ol className="mini-toc">
              {outline.map((section) => {
                const { id, title, isLab } = sectionTarget(chapter.slug, section, lab);
                return (
                  <li key={section.index} className={isLab ? "mini-toc-lab" : undefined}>
                    <a href={`#${id}`}>
                      <span className="mini-toc-n">{pad(section.index)}</span>
                      <span className="toc-title">{title}</span>
                      <span className="toc-leader" />
                      <span className="pg" data-pg={id} />
                    </a>
                  </li>
                );
              })}
            </ol>
          </div>
        )}
      </header>

      <section className={`chapter-main ${pageName(chapter.slug)}`}>
        <div ref={bodyRef} className="chapter-body" data-where={chapter.slug}>
          {body}
        </div>
        {labBlock && (labHost ? createPortal(labBlock, labHost) : labBlock)}
      </section>
    </article>
  );
}

function Lab({ slug, n, lab }: { slug: string; n: string; lab: BookLab }) {
  const runs = lab.needs_gpu ? `Runs on an ${lab.gpu}` : "Runs without a GPU";
  return (
    <section id={labId(slug)} className="lab" data-where={`${slug} lab`}>
      <p className="lab-kicker">
        <span>Lab {n}</span>
        <span className="lab-runs">{runs}</span>
      </p>
      <h3 className="lab-title">{lab.title}</h3>
      {/* The site shows the brief and the hints as plain text, and so does the book. */}
      {lab.brief && <p className="lab-brief">{lab.brief}</p>}
      <p className="lab-online">
        Run it at <a href={`https://${SITE}/c/${slug}`}>{`${SITE}/c/${slug}`}</a>, where the editor sits beside the
        chapter.
        {lab.solution && (
          <>
            {" "}
            The solution is on{" "}
            <a href={`#${solutionId(slug)}`}>
              page <span className="pg pg-inline" data-pg={solutionId(slug)} />
            </a>
            .
          </>
        )}
      </p>
      {lab.metrics.length > 0 && (
        <p className="lab-metrics">
          <span className="lab-metrics-label">Reports</span>
          {lab.metrics.map((metric) => (
            <code key={metric}>{metric}</code>
          ))}
        </p>
      )}

      <CodeFile name="starter.py" code={lab.starter} />

      {lab.hints.length > 0 && (
        <div className="lab-hints">
          <h4 className="lab-hints-title">Hints</h4>
          <p className="lab-hints-note">Read one at a time, and only when you are stuck.</p>
          <ol className="lab-hints-list">
            {lab.hints.map((hint, index) => (
              <li key={index}>{hint}</li>
            ))}
          </ol>
        </div>
      )}
    </section>
  );
}

function Solutions({ entries }: { entries: Numbered[] }) {
  return (
    <div className="solutions">
      <section id="solutions" className="part-page">
        <p className="part-kicker">Appendix</p>
        <h1 className="part-title">Solutions to the labs</h1>
        <div className="about-prose solutions-note">
          <p>
            These are the reference solutions the labs&apos; checks were written against. Each one is a correct
            answer, not the only one: any code that passes the checks is right. Try the lab first, and read a solution
            after your own code passes, or when the hints have not been enough.
          </p>
        </div>
      </section>
      <div className="solutions-body">
        {entries.map(({ chapter, n, lab }) => (
          <section key={chapter.slug} id={solutionId(chapter.slug)} className="solution" data-where={`${chapter.slug} solution`}>
            <p className="lab-kicker">
              <span>Solution {n}</span>
              <a href={`#${labId(chapter.slug)}`} className="solution-back">
                Lab on page <span className="pg pg-inline" data-pg={labId(chapter.slug)} />
              </a>
            </p>
            <h2 className="lab-title">{lab!.title}</h2>
            <p className="solution-chapter">
              For chapter {n}, {chapter.title}
            </p>
            <CodeFile name="solution.py" code={lab!.solution} />
          </section>
        ))}
      </div>
    </div>
  );
}

/**
 * Highlighted HTML split into one block per source line. A token that spans
 * lines, such as a docstring, is closed at the end of each line and opened
 * again on the next, so every line is balanced markup on its own.
 *
 * Each line carries its indentation, so a line too long for the page wraps
 * with a hanging indent: the rest of it sits further in than the line it
 * continues, and Python copied from the page keeps its structure.
 */
function splitLines(html: string, source: string): string {
  const sourceLines = source.split("\n");
  const open: string[] = [];
  const lines: string[] = [];
  let current = "";
  for (const token of html.split(/(<span[^>]*>|<\/span>|\n)/)) {
    if (!token) continue;
    if (token === "\n") {
      lines.push(current + "</span>".repeat(open.length));
      current = open.join("");
    } else if (token.startsWith("<span")) {
      open.push(token);
      current += token;
    } else if (token === "</span>") {
      open.pop();
      current += token;
    } else {
      current += token;
    }
  }
  lines.push(current);
  return lines
    .map((line, index) => {
      const text = sourceLines[index] ?? "";
      const indent = text.length - text.trimStart().length + 4;
      // A blank line, even one inside a docstring's markup, still takes a line.
      const body = text.trim() ? line : `${line}&#8203;`;
      return `<span class="ln" style="--indent:${indent}ch">${body}</span>`;
    })
    .join("");
}

/** A whole Python file, highlighted with the site's code colours. */
function CodeFile({ name, code }: { name: string; code: string }): ReactNode {
  const source = code.replace(/\s+$/, "");
  const html = splitLines(hljs.highlight(source, { language: "python", ignoreIllegals: true }).value, source);
  return (
    <div className="prose-chapter codefile">
      <div className="code-block">
        <div className="code-block-bar">
          <span>{name}</span>
          <span className="codefile-lines">{source.split("\n").length} lines</span>
        </div>
        <pre>
          <code className="hljs language-python" dangerouslySetInnerHTML={{ __html: html }} />
        </pre>
      </div>
    </div>
  );
}
