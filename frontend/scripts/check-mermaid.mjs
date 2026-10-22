#!/usr/bin/env node
/**
 * Checks the syntax of every Mermaid diagram in the slide decks.
 *
 * It finds each ```mermaid block in content/slides/*.md, parses it with
 * the same Mermaid the site renders with, and prints the file, the slide, and
 * the line of every block that fails. It exits with 1 when any block fails.
 *
 *   npm run check:mermaid                       # every deck
 *   npm run check:mermaid -- 03-memory-and-bandwidth.md
 *   npm run check:mermaid -- path/to/any.md path/to/a/folder
 *
 * A relative argument is read from content/slides first, then from the
 * current directory. Mermaid needs a DOM to load, so the script gives it one
 * from jsdom; parsing draws nothing.
 */

import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { JSDOM } from "jsdom";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, "..", "..");
const SLIDES = join(REPO, "content", "slides");

// The diagram types docs/SLIDES.md allows. The first word of a block names it.
const ALLOWED = new Set(["flowchart", "graph", "xychart-beta", "xychart", "pie", "timeline", "quadrantChart"]);

const dom = new JSDOM("<!doctype html><html><body></body></html>", { pretendToBeVisual: true });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
const { default: mermaid } = await import("mermaid");
mermaid.initialize({ startOnLoad: false, securityLevel: "strict" });

function markdownFiles(path) {
  const stat = statSync(path);
  if (stat.isFile()) return path.endsWith(".md") ? [path] : [];
  return readdirSync(path)
    .sort()
    .flatMap((name) => markdownFiles(join(path, name)));
}

function locate(arg) {
  for (const candidate of [resolve(SLIDES, arg), resolve(arg)]) {
    if (existsSync(candidate)) return candidate;
  }
  console.error(`Can't find ${arg}, in content/slides or here.`);
  process.exit(2);
}

/** Every ```mermaid block in a deck, with its line and the slide it sits on. */
function blocks(text) {
  const out = [];
  const lines = text.split("\n");
  let slide = { title: "(before the first slide)", line: 0 };
  let fence = null;
  let current = null;
  lines.forEach((line, i) => {
    const number = i + 1;
    if (fence) {
      if (line.trim().startsWith(fence) && line.trim().replace(/[`~]/g, "") === "") {
        if (current) out.push(current);
        fence = null;
        current = null;
      } else if (current) {
        current.source.push(line);
      }
      return;
    }
    const open = /^\s*(`{3,}|~{3,})\s*([\w-]*)/.exec(line);
    if (open) {
      fence = open[1];
      if (open[2] === "mermaid") current = { line: number, slide, source: [] };
      return;
    }
    const heading = /^(#{1,3}) (.+?)\s*(\{[^{}]*\})?\s*$/.exec(line);
    if (heading) slide = { title: heading[2], line: number };
  });
  if (current) out.push({ ...current, unclosed: true });
  return out;
}

function firstWord(source) {
  for (const line of source) {
    const trimmed = line.trim();
    if (trimmed && !trimmed.startsWith("%%")) return trimmed.split(/\s+/)[0];
  }
  return "";
}

const targets = process.argv.slice(2);
if (targets.length === 0 && !existsSync(SLIDES)) {
  console.log("There's no content/slides folder yet, so there's nothing to check.");
  process.exit(0);
}
const files = (targets.length ? targets.map(locate) : [SLIDES]).flatMap(markdownFiles);

let checked = 0;
let failures = 0;
for (const file of files) {
  const name = file.startsWith(REPO) ? relative(REPO, file) : file;
  for (const block of blocks(readFileSync(file, "utf8"))) {
    checked += 1;
    const where = `${name}:${block.line} (slide "${block.slide.title}", line ${block.slide.line})`;
    let problem = null;
    const kind = firstWord(block.source);
    if (block.unclosed) problem = "the fence is never closed";
    else if (!ALLOWED.has(kind)) problem = `"${kind || "(empty)"}" isn't an allowed diagram type (${[...ALLOWED].join(", ")})`;
    else {
      try {
        await mermaid.parse(block.source.join("\n"));
      } catch (error) {
        problem = String(error?.message ?? error).trim();
      }
    }
    if (problem) {
      failures += 1;
      console.log(`\n${where}`);
      console.log(problem.replace(/^/gm, "  "));
    }
  }
}

const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
if (failures) {
  console.log(`\n${failures} of ${plural(checked, "Mermaid diagram")} in ${plural(files.length, "file")} failed.`);
  process.exit(1);
}
console.log(`${plural(checked, "Mermaid diagram")} in ${plural(files.length, "file")} parsed.`);
