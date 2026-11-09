// The lab pane, fed a recorded run, for checking the editor and the streaming
// console without the backend, a login, or a GPU. Development only: see
// lab-fixture.html, which the production build never includes.
//
// `window.fetch` answers the lab's API calls from the fixture below, and
// /api/run streams the recorded events in the same `data: {json}` frames the
// backend sends, so the real stream parser in lib/api.ts reads them.
//
// Query parameters: `?theme=dark` or `?theme=light` pins a theme, and
// `?tab=checks` (or hints, solution) opens that tab once the run finishes.

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import "../index.css";
import type { Lab, RunEvent } from "../lib/api";
import LabPane from "../components/LabPane";

const params = new URLSearchParams(window.location.search);
const theme = params.get("theme");
if (theme === "light" || theme === "dark") document.documentElement.setAttribute("data-theme", theme);

const STARTER = `import triton
import triton.language as tl


@triton.jit
def softmax_kernel(out_ptr, in_ptr, n_cols, BLOCK: tl.constexpr):
    """One program per row: load, subtract the max, exponentiate, normalise."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(in_ptr + row * n_cols + offs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    tl.store(out_ptr + row * n_cols + offs, num / den, mask=mask)
`;

const LAB: Lab = {
  id: "fixture-softmax",
  title: "Fused softmax",
  brief: "Write a row-wise softmax in one kernel.",
  gpu: "A100-80GB",
  needs_gpu: true,
  timeout: 300,
  hints: ["Subtract the row maximum before exponentiating.", "Mask the tail of the last block."],
  metrics: ["wall_seconds", "gbps"],
  starter: STARTER,
  draft: null,
  runs: [],
};

const EVENTS: RunEvent[] = [
  { type: "start", run_id: "fixture", provider: "runpod" },
  { type: "log", line: "queued on runpod: position 1" },
  { type: "log", line: "worker ready: NVIDIA A100-SXM4-80GB, CUDA 12.8" },
  { type: "log", line: "compiling softmax_kernel (BLOCK=1024)" },
  { type: "log", line: "[PASS] matches torch.softmax on 4096 x 1000 (max err 2.4e-07)" },
  { type: "log", line: "[PASS] rows with -inf entries stay finite" },
  { type: "log", line: "[FAIL] 1.31 TB/s, below the 1.50 TB/s target" },
  { type: "log", line: "  hint: one load per element; check that the max and sum reuse x" },
  {
    type: "result",
    passed: false,
    checks: [
      { name: "Matches torch.softmax", passed: true, detail: "max error 2.4e-07" },
      { name: "Handles -inf", passed: true, detail: "" },
      { name: "Bandwidth", passed: false, detail: "1.31 TB/s, target 1.50 TB/s" },
    ],
    metrics: { wall_seconds: 0.84, gbps: 1310, occupancy: "62%" },
  },
  { type: "done", run_id: "fixture", passed: false, seconds: 14 },
];

function respond(body: unknown): Response {
  return new Response(JSON.stringify(body), { headers: { "Content-Type": "application/json" } });
}

function recordedRun(): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      for (const event of EVENTS) {
        await new Promise((resolve) => setTimeout(resolve, 60));
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
      }
      controller.close();
    },
  });
  return new Response(stream, { headers: { "Content-Type": "text/event-stream" } });
}

window.fetch = async (input: RequestInfo | URL) => {
  const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
  const path = new URL(url, window.location.origin).pathname;
  if (path === "/api/providers") {
    return respond({
      default: "runpod",
      preferred: "runpod",
      providers: [
        { name: "runpod", available: true, reason: "", preferred: true },
        { name: "modal", available: false, reason: "Not configured on this deployment.", preferred: false },
      ],
    });
  }
  if (path === "/api/run") return recordedRun();
  if (path.endsWith("/solution")) return respond({ solution: STARTER });
  return respond({ ok: true });
};

function Fixture() {
  return (
    <div className="h-full">
      <LabPane lab={LAB} onPassed={() => undefined} />
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <Fixture />
    </BrowserRouter>
  </StrictMode>,
);

// Press Run once the providers have loaded, then open the requested tab.
const tab = params.get("tab") ?? "console";
const timer = window.setInterval(() => {
  const run = document.querySelector<HTMLButtonElement>('button[title^="Run on the GPU"]');
  if (!run || run.disabled) return;
  window.clearInterval(timer);
  run.click();
  window.setTimeout(() => {
    const target = [...document.querySelectorAll<HTMLButtonElement>('[role="tab"]')].find(
      (button) => button.textContent?.toLowerCase().startsWith(tab),
    );
    target?.click();
  }, 1500);
}, 100);
