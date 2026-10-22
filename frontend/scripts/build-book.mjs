#!/usr/bin/env node
/**
 * Builds the course as one PDF book.
 *
 *   node scripts/build-book.mjs              # into ../book
 *   node scripts/build-book.mjs --force      # rebuild even if nothing changed
 *   node scripts/build-book.mjs --out DIR    # somewhere else
 *
 * It dumps the chapters and labs with the site's own loader
 * (scripts/dump_book.py), builds the print layout in src/book/ with Vite, and
 * prints it with headless Chrome. The book prints more than once: the first
 * pass says which page every anchor landed on, the page numbers go into the
 * contents, and the last pass is the one that ships, with a bookmark outline
 * added by pdf-lib.
 *
 * The build fails on any mathematics KaTeX cannot render, and on anything
 * wider than the page. It skips itself when the manifest records the same
 * input hash, so a deploy that changed no chapter does not wait for Chrome.
 *
 * Chrome is found through CHROME_PATH, or in the usual places on macOS and
 * Linux. Python is PYTHON, or the repository's .venv, or python3.
 */

import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import fs from "node:fs";
import { createServer } from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { PDFDict, PDFDocument, PDFHexString, PDFName, PDFNumber } from "pdf-lib";
import puppeteer from "puppeteer-core";

const FRONTEND = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const ROOT = path.resolve(FRONTEND, "..");
const FILE = "gpu-programming-for-ml.pdf";
// The text block in src/book/book.css: 138 mm at 96 px to the inch.
const TEXT_WIDTH_PX = Math.floor((138 / 25.4) * 96);

const argv = process.argv.slice(2);
const flag = (name) => argv.includes(name);
const option = (name) => {
  const at = argv.indexOf(name);
  return at >= 0 ? argv[at + 1] : undefined;
};

const OUT = path.resolve(option("--out") ?? path.join(ROOT, "book"));
const WORK = path.join(OUT, ".build");

function fail(message) {
  console.error(`build-book: ${message}`);
  process.exit(1);
}

function run(command, args, cwd) {
  const result = spawnSync(command, args, { cwd, stdio: "inherit" });
  if (result.status !== 0) fail(`${command} ${args.join(" ")} exited with ${result.status ?? result.signal}.`);
}

// What the PDF is made from. A change to any of it rebuilds the book.
function inputHash() {
  const hash = createHash("sha256");
  const walk = (target) => {
    const stat = fs.statSync(target, { throwIfNoEntry: false });
    if (!stat) return;
    if (stat.isDirectory()) {
      for (const name of fs.readdirSync(target).sort()) {
        if (name !== "__pycache__" && name !== ".DS_Store") walk(path.join(target, name));
      }
      return;
    }
    hash.update(path.relative(ROOT, target));
    hash.update(fs.readFileSync(target));
  };
  [
    "content",
    "backend/app/curriculum.py",
    "backend/app/config.py",
    "scripts/dump_book.py",
    "frontend/src",
    "frontend/book.html",
    "frontend/vite.book.config.ts",
    "frontend/package-lock.json",
    "frontend/scripts/build-book.mjs",
  ].forEach((entry) => walk(path.join(ROOT, entry)));
  return hash.digest("hex");
}

function findChrome() {
  const candidates = [
    process.env.CHROME_PATH,
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
  ].filter(Boolean);
  const found = candidates.find((candidate) => fs.existsSync(candidate));
  if (!found) fail("Chrome was not found. Set CHROME_PATH to a Chrome or Chromium binary.");
  return found;
}

function findPython() {
  if (process.env.PYTHON) return process.env.PYTHON;
  const venv = path.join(ROOT, ".venv", "bin", "python");
  return fs.existsSync(venv) ? venv : "python3";
}

/** Serves the built print layout, and the chapters as book.json beside it. */
function serve(root, data) {
  const types = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".woff2": "font/woff2", ".woff": "font/woff", ".ttf": "font/ttf", ".svg": "image/svg+xml", ".json": "application/json" };
  const server = createServer((request, response) => {
    const url = new URL(request.url ?? "/", "http://localhost");
    const file = url.pathname === "/book.json" ? data : path.join(root, decodeURIComponent(url.pathname));
    if (!file.startsWith(root) && file !== data) {
      response.writeHead(403).end();
      return;
    }
    fs.readFile(file, (error, body) => {
      if (error) {
        response.writeHead(404).end();
        return;
      }
      response.writeHead(200, { "content-type": types[path.extname(file)] ?? "application/octet-stream" });
      response.end(body);
    });
  });
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve(server)));
}

/**
 * Where every link target landed: anchor id to page number and destination.
 * Chrome writes a named destination for each in-document link target, and
 * only for those, which is why every entry in the contents and the outline
 * is also a link somewhere in the book.
 */
async function destinations(pdf) {
  const doc = await PDFDocument.load(pdf, { updateMetadata: false });
  const pageIndex = new Map(doc.getPages().map((page, index) => [page.ref.toString(), index]));
  const dests = doc.catalog.lookup(PDFName.of("Dests"));
  const out = new Map();
  if (!(dests instanceof PDFDict)) return { pages: doc.getPageCount(), out };
  for (const [key, value] of dests.entries()) {
    let dest = doc.context.lookup(value);
    if (dest instanceof PDFDict) dest = doc.context.lookup(dest.get(PDFName.of("D")));
    const target = dest?.get?.(0);
    if (!target) continue;
    const index = pageIndex.get(target.toString());
    if (index === undefined) continue;
    out.set(key.decodeText().replace(/^\//, ""), index + 1);
  }
  return { pages: doc.getPageCount(), out };
}

/** Adds the bookmark outline and the document information. */
async function finish(pdf, report, pageOf) {
  const doc = await PDFDocument.load(pdf);
  const context = doc.context;
  const pages = doc.getPages();
  const dests = doc.catalog.lookup(PDFName.of("Dests"));

  // A bookmark jumps to the link target's own spot on the page, as a link in
  // the contents does, or to the top of its page when it has no link.
  const destOf = (entry) => {
    const named = dests instanceof PDFDict ? dests.get(PDFName.of(entry.id)) : undefined;
    if (named) return named;
    const page = pageOf.get(entry.id);
    return page ? context.obj([pages[page - 1].ref, PDFName.of("Fit")]) : undefined;
  };

  const build = (entries, parentRef, depth) => {
    const nodes = entries
      .map((entry) => ({ entry, dest: destOf(entry) ?? (entry.children[0] && destOf(entry.children[0])) }))
      .filter((node) => node.dest);
    const refs = nodes.map(() => context.nextRef());
    let visible = 0;
    nodes.forEach((node, index) => {
      const item = context.obj({ Title: PDFHexString.fromText(node.entry.title), Parent: parentRef, Dest: node.dest });
      if (index > 0) item.set(PDFName.of("Prev"), refs[index - 1]);
      if (index < nodes.length - 1) item.set(PDFName.of("Next"), refs[index + 1]);
      const kids = build(node.entry.children, refs[index], depth + 1);
      if (kids.count > 0) {
        item.set(PDFName.of("First"), kids.first);
        item.set(PDFName.of("Last"), kids.last);
        // The parts start open and everything below them starts closed.
        const open = depth === 0;
        item.set(PDFName.of("Count"), PDFNumber.of(open ? kids.visible : -kids.count));
        if (open) visible += kids.visible;
      }
      context.assign(refs[index], item);
      visible += 1;
    });
    return { first: refs[0], last: refs[refs.length - 1], count: nodes.length, visible };
  };

  const rootRef = context.nextRef();
  const top = build(report.outline, rootRef, 0);
  context.assign(
    rootRef,
    context.obj({ Type: "Outlines", First: top.first, Last: top.last, Count: PDFNumber.of(top.visible) }),
  );
  doc.catalog.set(PDFName.of("Outlines"), rootRef);
  doc.catalog.set(PDFName.of("PageMode"), PDFName.of("UseOutlines"));

  doc.setTitle(report.title, { showInWindowTitleBar: true });
  doc.setSubject(report.subject);
  doc.setAuthor("learn-gpu");
  doc.setKeywords(["GPU programming", "CUDA", "Triton", "tensor cores", "GEMM", "FlashAttention", "course", "labs"]);
  doc.setCreator("learn-gpu book build");
  doc.setProducer("Chrome and pdf-lib");
  doc.setLanguage("en-GB");
  return doc.save({ useObjectStreams: true });
}

async function main() {
  const hash = inputHash();
  const manifestPath = path.join(OUT, "manifest.json");
  const previous = fs.existsSync(manifestPath) ? JSON.parse(fs.readFileSync(manifestPath, "utf8")) : null;
  if (!flag("--force") && previous?.hash === hash && fs.existsSync(path.join(OUT, FILE))) {
    console.log("The book is up to date.");
    return;
  }

  fs.mkdirSync(WORK, { recursive: true });
  const data = path.join(WORK, "book.json");
  run(findPython(), [path.join(ROOT, "scripts", "dump_book.py"), data], ROOT);
  run(path.join(FRONTEND, "node_modules", ".bin", "vite"), ["build", "--config", "vite.book.config.ts", "--logLevel", "warn"], FRONTEND);

  const root = path.join(FRONTEND, "dist-book");
  const server = await serve(root, data);
  const port = server.address().port;
  const browser = await puppeteer.launch({
    executablePath: findChrome(),
    headless: true,
    args: ["--no-sandbox", "--font-render-hinting=none"],
  });

  const started = Date.now();
  let entry;
  let failed = false;
  try {
    const page = await browser.newPage();
    page.on("pageerror", (error) => console.warn(`  page error: ${error.message}`));
    // Lay the page out at the width of the text block, so the fitting pass
    // measures what print will actually have. Anything wider makes Chrome
    // shrink the whole book to fit.
    await page.setViewport({ width: TEXT_WIDTH_PX, height: 1100 });
    await page.emulateMediaType("print");
    await page.goto(`http://127.0.0.1:${port}/book.html`, { waitUntil: "load", timeout: 0 });
    await page.waitForFunction(() => window.__book?.ready || window.__book?.error, { timeout: 180_000, polling: 250 });
    const report = await page.evaluate(() => window.__book);
    if (report.error) fail(report.error);
    if (report.katexErrors.length) {
      failed = true;
      console.error(`  ${report.katexErrors.length} formulas KaTeX could not render:`);
      report.katexErrors.slice(0, 20).forEach((line) => console.error(`    ${line}`));
    }
    report.overflow.forEach((line) => console.warn(`  still too wide: ${line}`));
    const wide = await page.evaluate((limit) => {
      const out = [];
      for (const node of document.querySelectorAll(".book *")) {
        const box = node.getBoundingClientRect();
        // SVG clips its own drawing, and KaTeX's MathML copy is hidden.
        if (node.closest("svg, .katex-mathml, .cover")) continue;
        if (box.right > limit + 3) {
          // Report the outermost culprit once, with where it is.
          if (node.parentElement?.getBoundingClientRect().right > limit + 3 && !node.parentElement.matches(".book")) continue;
          const where = node.closest("[data-where]")?.getAttribute("data-where") ?? "?";
          const text = (node.textContent ?? "").replace(/\s+/g, " ").slice(0, 70);
          out.push(`${where}: <${node.tagName.toLowerCase()} class="${String(node.className).slice(0, 40)}"> ${Math.round(box.right - limit)} px over: ${text}`);
          if (out.length > 20) break;
        }
      }
      return out;
    }, TEXT_WIDTH_PX);
    // Chrome shrinks the whole book to fit its widest line, so one formula
    // that runs into the margin sets every page in smaller type.
    if (wide.length) {
      failed = true;
      wide.forEach((line) => console.error(`  wider than the page: ${line}`));
    }

    const print = () =>
      page.pdf({ preferCSSPageSize: true, printBackground: true, tagged: true, outline: false, timeout: 0 });

    // Print, read where the anchors landed, fill the numbers in, and print
    // again, until the page numbers stop moving. Filling in the numbers
    // can reflow the contents onto another page, which moves everything.
    let pdf = await print();
    let found = await destinations(pdf);
    for (let pass = 0; pass < 3; pass += 1) {
      const pageOf = Object.fromEntries(found.out);
      await page.evaluate((pages) => window.__setPages(pages), pageOf);
      pdf = await print();
      const again = await destinations(pdf);
      const same = again.out.size === found.out.size && [...again.out].every(([id, n]) => found.out.get(id) === n);
      found = again;
      if (same) break;
    }

    // Nothing links to the contents, but they always follow the cover.
    if (!found.out.has("contents")) found.out.set("contents", 2);

    const missing = report.outline.flatMap(function walk(item) {
      return [...(found.out.has(item.id) ? [] : [item.title]), ...item.children.flatMap(walk)];
    });
    if (missing.length) console.warn(`  ${missing.length} bookmarks point at the top of a page: ${missing.slice(0, 5).join("; ")}`);

    const bytes = await finish(pdf, report, found.out);
    await page.close();
    if (!failed) fs.writeFileSync(path.join(OUT, FILE), bytes);
    entry = { file: FILE, title: report.title, pages: found.pages, bytes: bytes.length };
    console.log(`  ${FILE}: ${found.pages} pages, ${(bytes.length / 1e6).toFixed(1)} MB, ${((Date.now() - started) / 1000).toFixed(0)} s`);
  } finally {
    await browser.close();
    server.close();
  }

  if (failed) fail("Fix the problems above and build again.");

  const commit = JSON.parse(fs.readFileSync(data, "utf8")).commit;
  fs.writeFileSync(manifestPath, `${JSON.stringify({ hash, commit, built: new Date().toISOString(), ...entry }, null, 2)}\n`);
  console.log(`Wrote ${path.join(OUT, FILE)}`);
}

main().catch((error) => fail(error.stack ?? String(error)));
