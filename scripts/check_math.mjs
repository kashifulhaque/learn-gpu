// Renders every math expression in content/chapters/ and content/slides/ with
// the same KaTeX build the site uses, and reports the ones that fail.
//
//   node scripts/check_math.mjs [file ...]
//
// The parser here mirrors remark-math: `$$...$$` is display math, `$...$` is
// inline math, and neither is recognized inside a fenced or inline code span.
// Math inside a callout carries the blockquote's `>` markers, which are
// stripped first. The colour macros are the ones the site defines.

import { readFileSync, readdirSync } from "node:fs";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const require = createRequire(join(root, "frontend/node_modules/"));
const katex = require("katex");
const macros = JSON.parse(
  readFileSync(join(root, "frontend/src/lib/katexMacros.json"), "utf8"),
);

/** Replaces blockquote markers, fenced blocks, and inline code with spaces, keeping offsets. */
function blankCode(text) {
  const blank = (match) => match.replace(/[^\n]/g, " ");
  return text
    .replace(/^[ \t]*(?:>[ \t]?)+/gm, blank)
    .replace(/^[ \t]*```[\s\S]*?^[ \t]*```/gm, blank)
    .replace(/`[^`\n]*`/g, blank);
}

function lineOf(text, index) {
  return text.slice(0, index).split("\n").length;
}

function collect(text) {
  const scan = blankCode(text);
  const found = [];
  const display = /\$\$([\s\S]+?)\$\$/g;
  const taken = [];
  let match;
  while ((match = display.exec(scan))) {
    found.push({ tex: match[1], display: true, line: lineOf(text, match.index) });
    taken.push([match.index, display.lastIndex]);
  }
  const masked = scan.split("");
  for (const [start, end] of taken) {
    for (let i = start; i < end; i += 1) if (masked[i] !== "\n") masked[i] = " ";
  }
  const inline = /\$([^$\n]+?)\$/g;
  const rest = masked.join("");
  while ((match = inline.exec(rest))) {
    found.push({ tex: match[1], display: false, line: lineOf(text, match.index) });
  }
  return found;
}

const markdownIn = (dir) => {
  try {
    return readdirSync(join(root, dir))
      .filter((name) => name.endsWith(".md"))
      .sort()
      .map((name) => join(root, dir, name));
  } catch {
    return [];
  }
};

const files = process.argv.length > 2
  ? process.argv.slice(2)
  : [...markdownIn("content/chapters"), ...markdownIn("content/slides")];

let expressions = 0;
const failures = [];

for (const file of files) {
  const text = readFileSync(file, "utf8");
  for (const item of collect(text)) {
    expressions += 1;
    try {
      katex.renderToString(item.tex, {
        displayMode: item.display,
        throwOnError: true,
        // The colour macros expand to \htmlClass, so the theme picks the
        // colour. That is the one HTML extension allowed; the site does the same.
        strict: (code) => (code === "htmlExtension" ? "ignore" : "warn"),
        trust: (context) => context.command === "\\htmlClass",
        macros: { ...macros },
      });
    } catch (error) {
      failures.push({ file, ...item, message: error.message.split("\n")[0] });
    }
  }
}

const short = (path) => path.replace(`${root}/`, "");
for (const failure of failures) {
  console.error(`${short(failure.file)}:${failure.line}: ${failure.message}`);
  console.error(`    ${failure.tex.trim().slice(0, 120)}`);
}

console.log(
  `${expressions} expressions in ${files.length} file(s), ${failures.length} failed`,
);
process.exit(failures.length ? 1 : 0);
