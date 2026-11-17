import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The print layouts for the PDF book (book.html) and the slides PDF
// (slides.html), built apart from the site so they never ship in `dist`.
// scripts/build-book.mjs and scripts/build-slides.mjs serve `dist-book` and
// print from it.
//
// Each script sets PRINT_ENTRY to build only its own layout. That lets the
// Docker image's book stage hold only the book's sources and the slides stage
// only the slides', so a change to one doesn't reprint the other. Without
// PRINT_ENTRY, both layouts build.
const ENTRIES: Record<string, string> = { book: "book.html", slides: "slides.html" };
const only = process.env.PRINT_ENTRY;
if (only && !(only in ENTRIES)) {
  throw new Error(`PRINT_ENTRY must be one of ${Object.keys(ENTRIES).join(", ")}, not "${only}".`);
}

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    outDir: "dist-book",
    emptyOutDir: true,
    rollupOptions: { input: only ? { [only]: ENTRIES[only] } : ENTRIES },
    // One page load, printed locally: bundle size doesn't matter here.
    chunkSizeWarningLimit: 4000,
  },
});
