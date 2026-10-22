import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.API_TARGET ?? "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      // The PDF book, which the backend serves from book/.
      "/book": {
        target: process.env.API_TARGET ?? "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  // Monaco's editor worker is an ES module, so the worker bundle must be too.
  worker: {
    format: "es",
  },
  build: {
    outDir: "dist",
    // Monaco lands in its own chunk, which is large by nature.
    chunkSizeWarningLimit: 4000,
  },
});
