# learn-gpu

A web app that teaches GPU programming for machine learning, from the
streaming multiprocessor to the kernels that train and serve models. You write
CUDA C++ and Triton, and each lab runs on a real A100 through
[RunPod](https://runpod.io) and streams its output back to the browser.

The course assumes you write Python and PyTorch and know what a transformer is.
It doesn't assume C++ beyond reading it. It is the companion to
[learn-inference](https://github.com/kashifulhaque/learn-inference), which builds
an inference engine for one model; this course goes deeper on every kernel topic
that one touches.

## What is in here

| Path | Contents |
|---|---|
| `content/chapters/` | 24 chapters, in Markdown with YAML front matter. |
| `content/labs/` | 24 labs, one per chapter except chapter 23: a starter file, a test harness, and a worked solution. `lab_common.py` holds the shared GPU helpers. |
| `content/slides/` | One slide deck per chapter. |
| `content/STYLE.md` | The chapter style guide, including the accuracy rules. |
| `docs/` | The curriculum, the slide contract (`SLIDES.md`), and the lab contract (`LABS.md`). |
| `gpu/` | The RunPod worker image and the lab runner it uses. |
| `backend/` | FastAPI: auth, content, slides, progress, and the run stream. |
| `frontend/` | React and Vite: reader, editor, slides, dashboard, and compute panel. |
| `scripts/` | Content checkers and the deploy script. |

## The curriculum

Every chapter has a book chapter, a lab, and a slide deck. The lab hardware is
one A100 80GB PCIe (Ampere, sm_80). Hopper and Blackwell features are taught in
prose, figures, and slides, with Ampere labs that exercise the same idea where
one exists.

| # | Chapter | Part | Lab |
|---|---|---|---|
| 00 | [What you are building](content/chapters/00-what-you-are-building.md) | Part 1 — The machine | Query the device, time a copy with CUDA events, report bandwidth. |
| 01 | [The GPU at a glance](content/chapters/01-the-gpu-at-a-glance.md) | Part 1 — The machine | Derive peak FLOP/s and bandwidth from device properties and spec constants; compare A100, H100, B200. |
| 02 | [The execution model](content/chapters/02-the-execution-model.md) | Part 1 — The machine | An occupancy calculator from registers, shared memory, and block size; measure warp divergence cost. |
| 03 | [Memory and bandwidth](content/chapters/03-memory-and-bandwidth.md) | Part 1 — The machine | Copy kernels: scalar, float4 vectorized, grid-stride, strided; reach the copy ceiling. |
| 04 | [The roofline and latency hiding](content/chapters/04-roofline-and-latency.md) | Part 1 — The machine | Predict kernel time from the roofline and Little's law, then measure; wave quantization. |
| 05 | [Profiling and reading the machine code](content/chapters/05-profiling.md) | Part 1 — The machine | Profile three slow kernels with torch.profiler and CUDA events, diagnose, fix; read PTX/SASS. |
| 06 | [Reductions](content/chapters/06-reductions.md) | Part 2 — CUDA patterns | Sum and max reductions: shared-memory tree, warp shuffles, one-pass with atomics; hit bandwidth. |
| 07 | [Scan and histogram](content/chapters/07-scan-and-histogram.md) | Part 2 — CUDA patterns | Block scan (Kogge-Stone, Blelloch), device-wide scan; privatized histogram. |
| 08 | [Shared memory and the transpose](content/chapters/08-shared-memory-and-transpose.md) | Part 2 — CUDA patterns | Matrix transpose: naive, shared-memory tiled, padded and swizzled against bank conflicts. |
| 09 | [Floating point on the GPU](content/chapters/09-floating-point-on-the-gpu.md) | Part 2 — CUDA patterns | fp32/tf32/bf16/fp16/fp8 round-trips, accumulation error, Kahan summation, atomic nondeterminism. |
| 10 | [GEMM from scratch](content/chapters/10-gemm-from-scratch.md) | Part 3 — Matrix multiplication | SGEMM: naive, coalesced, shared-memory tiled, 2D register blocking, vectorized; percent of cuBLAS. |
| 11 | [Tensor cores](content/chapters/11-tensor-cores.md) | Part 3 — Matrix multiplication | fp16/bf16 GEMM on tensor cores with WMMA, then mma.sync with ldmatrix. |
| 12 | [Pipelining and asynchronous copies](content/chapters/12-pipelining-and-async-copies.md) | Part 3 — Matrix multiplication | Multistage cp.async pipeline for the tensor-core GEMM. Hopper TMA/wgmma and Blackwell tcgen05 in prose. |
| 13 | [CUTLASS and CuTe layouts](content/chapters/13-cutlass-and-cute.md) | Part 3 — Matrix multiplication | Implement CuTe layout algebra (shape/stride, coalesce, composition, complement, swizzle) in Python. |
| 14 | [Triton](content/chapters/14-triton.md) | Part 3 — Matrix multiplication | Triton matmul with block pointers, grouped (L2-friendly) ordering, and autotuning vs cuBLAS. |
| 15 | [Softmax and normalization](content/chapters/15-softmax-and-normalization.md) | Part 4 — Kernels for ML | Online softmax and LayerNorm forward and backward (Welford) in Triton. |
| 16 | [FlashAttention](content/chapters/16-flash-attention.md) | Part 4 — Kernels for ML | FlashAttention-2 forward and backward in Triton, with causal masking. |
| 17 | [Fusion, compilers, and launch overhead](content/chapters/17-fusion-and-compilers.md) | Part 4 — Kernels for ML | torch.compile and Inductor output, CUDA graphs, custom ops with torch.library. |
| 18 | [Quantized kernels](content/chapters/18-quantized-kernels.md) | Part 4 — Kernels for ML | Group-quantize weights to int4 and write a fused dequantize-GEMM (W4A16) in Triton. |
| 19 | [Training kernels](content/chapters/19-training-kernels.md) | Part 4 — Kernels for ML | Custom autograd Functions; fused, chunked linear plus cross-entropy that never materializes the logits. |
| 20 | [Mixture-of-experts and sparse kernels](content/chapters/20-moe-and-sparse-kernels.md) | Part 4 — Kernels for ML | Token permutation and a grouped GEMM in Triton; 2:4 structured sparsity. |
| 21 | [Collective communication](content/chapters/21-collective-communication.md) | Part 5 — Many GPUs | Ring all-reduce with torch.distributed point-to-point; the alpha-beta cost model. |
| 22 | [Overlap and parallelism](content/chapters/22-overlap-and-parallelism.md) | Part 5 — Many GPUs | CUDA streams and events: overlap host-device copies with compute; DP/FSDP/TP/PP/EP from the kernel's view. |
| 23 | [Beyond CUDA](content/chapters/23-beyond-cuda.md) | Part 6 — Beyond | AMD CDNA and HIP, TPUs and XLA/Pallas, Apple silicon, and tile DSLs (ThunderKittens, TileLang, CuTe DSL). |

## Run it locally

You need Python 3.11 or later and Node 20 or later.

1. Copy the environment file and fill it in:

   ```bash
   cp .env.example .env
   ```

   Set `SESSION_SECRET` to a random string:

   ```bash
   python3 -c "import secrets; print(secrets.token_urlsafe(48))"
   ```

2. Install and start the backend:

   ```bash
   python3 -m venv .venv && .venv/bin/pip install -r backend/requirements-dev.txt
   .venv/bin/uvicorn backend.app.main:app --reload --port 8000
   ```

3. In another terminal, start the frontend:

   ```bash
   cd frontend && npm install && npm run dev
   ```

Open http://localhost:5173. The Vite dev server proxies `/api` to the backend.
To see every chapter figure on one page, without signing in, open
http://localhost:5173/viz.html.

## Set up the GPU

Labs run on a RunPod serverless endpoint. The worker needs `nvcc`, so its image
is built on a CUDA *devel* base image (`pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`),
with `ninja` and `triton` installed. It needs no network volume and no
Hugging Face token.

1. Build the worker image and push it. Run this from the repository root, so the
   build context includes `content/`:

   ```bash
   docker build -f gpu/runpod_worker/Dockerfile -t REGISTRY/learn-gpu-worker:latest .
   ```

   Replace `REGISTRY` with your container registry. Then push it:

   ```bash
   docker push REGISTRY/learn-gpu-worker:latest
   ```

   Pushing to GitHub Container Registry happens on its own: see
   [the publish workflow](.github/workflows/publish-runpod-worker.yml).

2. In the RunPod console, create a serverless endpoint from that image, on an
   **A100 80GB** worker.

3. Put `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` in `.env`.

For more detail, see [the worker's README](gpu/runpod_worker/README.md). The
endpoint costs nothing while it is idle. `/compute` in the app shows what is
running on RunPod and stops it.

## Slides

Each chapter has a deck at `content/slides/<slug>.md`. Open **Slides** in the
sidebar for the index. A deck has a presenter view (the slide, the notes, and the
next slide), a full-window present mode, and a print layout with one 16:9 slide
per page. Any signed-in user can open them. For the deck format, see
[docs/SLIDES.md](docs/SLIDES.md). To check the decks, run the following commands:

```bash
.venv/bin/python scripts/check_slides.py
cd frontend && npm run check:mermaid
```

## The PDFs

The sidebar's **Course PDF** link downloads the whole course as one book, and the
**Slides** page links a PDF of every deck, one slide per page. Both use headless
Chrome. `frontend/scripts/build-book.mjs` and `frontend/scripts/build-slides.mjs`
find Chrome through `CHROME_PATH` or in the usual install locations, and they
write to `book/`, which is untracked:

```bash
node frontend/scripts/build-book.mjs --force
node frontend/scripts/build-slides.mjs --force
```

Without `--force`, each build skips itself when nothing it reads has changed.
Both builds fail on maths that KaTeX can't render. `scripts/deploy.sh` builds
both on your workstation and copies them to the server; set `LG_SKIP_BOOK=1` to
skip that.

## Deploy with Docker Compose

```bash
docker compose up -d --build
```

The container joins the external `edge` network and answers on port 8000, which
is where a reverse proxy should send traffic. See
[Caddyfile.example](Caddyfile.example) for a Caddy block that keeps server-sent
events unbuffered, which the lab output stream needs. The compose file also
publishes `127.0.0.1:8091`, so the app stays reachable through an SSH tunnel.

The repository isn't deployed yet. When it is, `scripts/deploy.sh` pushes a
commit to the server's clone, rebuilds, and waits for the health check. It reads
`LG_HOST`, `LG_REMOTE_PATH` (default `~/docs/learn-gpu`), `LG_SSH_KEY`,
`LG_BRANCH`, `LG_REPO_URL`, and `LG_SKIP_BOOK`.

## Access

The app is password protected. Set `APP_PASSWORD` in `.env`. Whoever signs in
picks a display name, which scopes their progress, drafts, and run history.
Labs execute code you type. That code runs on the RunPod worker, never on the web
host.

## Check the content

Run these before you commit:

```bash
.venv/bin/python scripts/check_chapters.py   # chapter shape, front matter, figure names
.venv/bin/python scripts/check_slides.py     # decks against docs/SLIDES.md
.venv/bin/python scripts/check_labs.py       # lab files, and solutions where a GPU exists
node scripts/check_math.mjs                  # every formula, with the site's KaTeX
.venv/bin/python -m pytest backend           # backend tests
```

Labs that need CUDA are skipped on a machine without a GPU, after a static check.

## Lectures

This repository has no lecture videos. The lecture pipeline in learn-inference
(narrated slides rendered to video, with a manifest per chapter and a player on
the chapter page) can be ported later.

## License

MIT. See [LICENSE](LICENSE).
