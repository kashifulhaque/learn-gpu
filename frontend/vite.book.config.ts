import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The print layouts for the PDF book (book.html) and the slides PDF
// (slides.html), built apart from the site so they never ship in `dist`.
// scripts/build-book.mjs and scripts/build-slides.mjs serve `dist-book` and
// print from it.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    outDir: "dist-book",
    emptyOutDir: true,
    rollupOptions: { input: { book: "book.html", slides: "slides.html" } },
    // One page load, printed locally: bundle size doesn't matter here.
    chunkSizeWarningLimit: 4000,
  },
});
