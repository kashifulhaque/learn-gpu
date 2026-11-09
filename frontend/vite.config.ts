import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Local development only. The backend answers 401 unless the request carries
// the proxy secret and a user, as Caddy adds them after Authelia signs a reader
// in. With VITE_DEV_SSO=1 the dev server adds both, so start the backend with
// SSO_PROXY_SECRET=dev-secret. Production serves the built bundle, never this
// dev server, so these headers never ship.
const devSso =
  process.env.VITE_DEV_SSO === "1"
    ? { headers: { "X-SSO-Proxy": "dev-secret", "Remote-User": "dev" } }
    : {};

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.API_TARGET ?? "http://127.0.0.1:8000",
        changeOrigin: true,
        ...devSso,
      },
      // The PDF book, which the backend serves from book/.
      "/book": {
        target: process.env.API_TARGET ?? "http://127.0.0.1:8000",
        changeOrigin: true,
        ...devSso,
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
