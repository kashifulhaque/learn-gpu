#!/usr/bin/env node
/**
 * Builds every chapter's slide deck as one PDF: one 16:9 page per slide, with
 * a bookmark for each deck and each of its sections.
 *
 *   node scripts/build-slides.mjs              # into ../book
 *   node scripts/build-slides.mjs --force      # rebuild even if nothing changed
 *   node scripts/build-slides.mjs --out DIR    # somewhere else
 *
 * It dumps the decks with the site's own parser (scripts/dump_slides.py),
 * builds the print layout in src/slides/ with Vite, and prints it with
 * headless Chrome. Every slide is exactly one page, so a deck starts on the
 * page after the last slide of the one before; the build checks that the page
 * count agrees with the slide count, and fails on mathematics KaTeX cannot
 * render. It skips itself when the manifest records the same input hash.
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

import { PDFDocument, PDFHexString, PDFName, PDFNumber } from "pdf-lib";
import puppeteer from "puppeteer-core";

const FRONTEND = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const ROOT = path.resolve(FRONTEND, "..");
const FILE = "gpu-programming-for-ml-slides.pdf";
const TITLE = "GPU programming for ML: slides";

const argv = process.argv.slice(2);
const flag = (name) => argv.includes(name);
const option = (name) => {
  const at = argv.indexOf(name);
  return at >= 0 ? argv[at + 1] : undefined;
};

const OUT = path.resolve(option("--out") ?? path.join(ROOT, "book"));
const WORK = path.join(OUT, ".slides-build");

function fail(message) {
  console.error(`build-slides: ${message}`);
  process.exit(1);
}

function run(command, args, cwd) {
  const result = spawnSync(command, args, { cwd, stdio: "inherit" });
  if (result.status !== 0) fail(`${command} ${args.join(" ")} exited with ${result.status ?? result.signal}.`);
}

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
    "content/slides",
    "content/chapters",
    "backend/app/decks.py",
    "backend/app/curriculum.py",
    "scripts/dump_slides.py",
    "frontend/src",
    "frontend/slides.html",
    "frontend/vite.book.config.ts",
    "frontend/package-lock.json",
    "frontend/scripts/build-slides.mjs",
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

/** Serves the built print layout, and the decks as slides.json beside it. */
function serve(root, data) {
  const types = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".woff2": "font/woff2", ".woff": "font/woff", ".ttf": "font/ttf", ".svg": "image/svg+xml", ".json": "application/json" };
  const server = createServer((request, response) => {
    const url = new URL(request.url ?? "/", "http://localhost");
    const file = url.pathname === "/slides.json" ? data : path.join(root, decodeURIComponent(url.pathname));
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

/** Adds a bookmark for each deck, with its sections beneath, and the document information. */
async function finish(pdf, report) {
  const doc = await PDFDocument.load(pdf);
  const context = doc.context;
  const pages = doc.getPages();
  const dest = (page) => context.obj([pages[Math.min(page, pages.length) - 1].ref, PDFName.of("Fit")]);

  const rootRef = context.nextRef();
  let at = 1;
  const decks = report.decks.map((deck) => {
    const entry = { deck, page: at, ref: context.nextRef() };
    at += deck.slides;
    return entry;
  });

  let visible = 0;
  decks.forEach((entry, index) => {
    const kids = entry.deck.sections.map(() => context.nextRef());
    entry.deck.sections.forEach((section, i) => {
      const item = context.obj({ Title: PDFHexString.fromText(section.title), Parent: entry.ref, Dest: dest(section.at) });
      if (i > 0) item.set(PDFName.of("Prev"), kids[i - 1]);
      if (i < kids.length - 1) item.set(PDFName.of("Next"), kids[i + 1]);
      context.assign(kids[i], item);
    });
    const item = context.obj({
      Title: PDFHexString.fromText(`${entry.deck.slug.split("-")[0]} · ${entry.deck.chapter_title}`),
      Parent: rootRef,
      Dest: dest(entry.page),
    });
    if (index > 0) item.set(PDFName.of("Prev"), decks[index - 1].ref);
    if (index < decks.length - 1) item.set(PDFName.of("Next"), decks[index + 1].ref);
    if (kids.length) {
      item.set(PDFName.of("First"), kids[0]);
      item.set(PDFName.of("Last"), kids[kids.length - 1]);
      item.set(PDFName.of("Count"), PDFNumber.of(-kids.length));
    }
    context.assign(entry.ref, item);
    visible += 1;
  });
  if (decks.length) {
    context.assign(
      rootRef,
      context.obj({ Type: "Outlines", First: decks[0].ref, Last: decks[decks.length - 1].ref, Count: PDFNumber.of(visible) }),
    );
    doc.catalog.set(PDFName.of("Outlines"), rootRef);
    doc.catalog.set(PDFName.of("PageMode"), PDFName.of("UseOutlines"));
  }

  doc.setTitle(TITLE, { showInWindowTitleBar: true });
  doc.setSubject("The slide decks of the learn-gpu course, one deck per chapter.");
  doc.setAuthor("learn-gpu");
  doc.setKeywords(["GPU programming", "CUDA", "Triton", "slides", "course"]);
  doc.setCreator("learn-gpu slides build");
  doc.setProducer("Chrome and pdf-lib");
  doc.setLanguage("en-GB");
  return doc.save({ useObjectStreams: true });
}

async function main() {
  const hash = inputHash();
  const manifestPath = path.join(OUT, "slides-manifest.json");
  const previous = fs.existsSync(manifestPath) ? JSON.parse(fs.readFileSync(manifestPath, "utf8")) : null;
  if (!flag("--force") && previous?.hash === hash && fs.existsSync(path.join(OUT, FILE))) {
    console.log("The slides PDF is up to date.");
    return;
  }

  fs.mkdirSync(WORK, { recursive: true });
  const data = path.join(WORK, "slides.json");
  run(findPython(), [path.join(ROOT, "scripts", "dump_slides.py"), data], ROOT);
  const dumped = JSON.parse(fs.readFileSync(data, "utf8"));
  if (dumped.decks.length === 0) {
    console.log("There are no decks yet, so there is no slides PDF to build.");
    return;
  }
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
    // The slide page is 13.333 by 7.5 inches, which is 1280 by 720 pixels.
    // Lay the slides out at that size, in print media, so the type-fitting
    // pass measures what the page will have.
    await page.setViewport({ width: 1280, height: 720 });
    await page.emulateMediaType("print");
    await page.goto(`http://127.0.0.1:${port}/slides.html`, { waitUntil: "load", timeout: 0 });
    await page.waitForFunction(() => window.__slides?.ready || window.__slides?.error, { timeout: 300_000, polling: 250 });
    const report = await page.evaluate(() => window.__slides);
    if (report.error) fail(report.error);
    for (const line of report.katexErrors) {
      failed = true;
      console.error(`  KaTeX could not render: ${line}`);
    }
    for (const line of report.diagramErrors) {
      failed = true;
      console.error(`  a figure or diagram failed: ${line}`);
    }

    const pdf = await page.pdf({ preferCSSPageSize: true, printBackground: true, tagged: true, outline: false, timeout: 0 });
    const slides = report.decks.reduce((sum, deck) => sum + deck.slides, 0);
    const counted = (await PDFDocument.load(pdf, { updateMetadata: false })).getPageCount();
    if (counted !== slides) {
      failed = true;
      console.error(`  ${slides} slides printed as ${counted} pages. A slide has spilled onto a second page, or a page is blank.`);
    }

    const bytes = await finish(pdf, report);
    await page.close();
    if (!failed) fs.writeFileSync(path.join(OUT, FILE), bytes);
    entry = { file: FILE, title: TITLE, pages: counted, bytes: bytes.length };
    console.log(`  ${FILE}: ${report.decks.length} decks, ${counted} pages, ${(bytes.length / 1e6).toFixed(1)} MB, ${((Date.now() - started) / 1000).toFixed(0)} s`);
  } finally {
    await browser.close();
    server.close();
  }

  if (failed) fail("Fix the problems above and build again.");
  fs.writeFileSync(manifestPath, `${JSON.stringify({ hash, commit: dumped.commit, built: new Date().toISOString(), ...entry }, null, 2)}\n`);
  console.log(`Wrote ${path.join(OUT, FILE)}`);
}

main().catch((error) => fail(error.stack ?? String(error)));
