# learn-gpu

A web app that teaches GPU programming for machine learning, from the
streaming multiprocessor to the kernels that train and serve models. You write
CUDA C++ and Triton, and chapter 24 writes a CuTe DSL GEMM. Each lab runs on a real A100 through
[RunPod](https://runpod.io) and streams its output back to the browser.

The course assumes you write Python and PyTorch and know what a transformer is.
It doesn't assume C++ beyond reading it. It is the companion to
[learn-inference](https://github.com/kashifulhaque/learn-inference), which builds
an inference engine for one model; this course goes deeper on every kernel topic
that one touches.

## What is in here

| Path | Contents |
|---|---|
| `content/chapters/` | 35 chapters, in Markdown with YAML front matter. |
| `content/labs/` | 35 labs, one per chapter: a starter file, a test harness, and a worked solution. `lab_common.py` holds the shared GPU helpers. |
| `content/slides/` | One slide deck per chapter. |
| `content/STYLE.md` | The chapter style guide, including the accuracy rules. |
| `docs/` | The curriculum, the slide contract (`SLIDES.md`), and the lab contract (`LABS.md`). |
| `gpu/` | The RunPod worker image and the lab runner it uses. |
| `backend/` | FastAPI: auth, content, slides, progress, and the run stream. |
| `frontend/` | React and Vite: reader, editor, slides, dashboard, and compute panel. |
| `scripts/` | Content checkers and the deploy script. |

## The curriculum

Every chapter has a book chapter, a lab, and a slide deck. The lab hardware is
one A100 80GB PCIe (Ampere, sm_80). Chapters 26 and 34 are Hopper labs, gated on an
H100: the CPU checks run anywhere, and the TMA, wgmma, and cluster kernels run only when
that endpoint is configured. Blackwell stays in prose, figures, and slides.

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
| 14 | [Triton](content/chapters/14-triton.md) | Part 3 — Matrix multiplication | Triton matmul with plain pointer arithmetic, grouped (L2-friendly) ordering, and autotuning vs cuBLAS. |
| 15 | [Softmax and normalization](content/chapters/15-softmax-and-normalization.md) | Part 4 — Kernels for ML | Online softmax and LayerNorm forward and backward (Welford) in Triton. |
| 16 | [FlashAttention](content/chapters/16-flash-attention.md) | Part 4 — Kernels for ML | FlashAttention-2 forward and backward in Triton, with causal masking. |
| 17 | [Fusion, compilers, and launch overhead](content/chapters/17-fusion-and-compilers.md) | Part 4 — Kernels for ML | torch.compile and Inductor output, CUDA graphs, custom ops with torch.library. |
| 18 | [Quantized kernels](content/chapters/18-quantized-kernels.md) | Part 4 — Kernels for ML | Group-quantize weights to int4 and write a fused dequantize-GEMM (W4A16) in Triton. |
| 19 | [Training kernels](content/chapters/19-training-kernels.md) | Part 4 — Kernels for ML | Custom autograd Functions; fused, chunked linear plus cross-entropy that never materializes the logits. |
| 20 | [Mixture-of-experts and sparse kernels](content/chapters/20-moe-and-sparse-kernels.md) | Part 4 — Kernels for ML | Token permutation and a grouped GEMM in Triton; 2:4 structured sparsity. |
| 21 | [Collective communication](content/chapters/21-collective-communication.md) | Part 5 — Many GPUs | Ring all-reduce with torch.distributed point-to-point; the alpha-beta cost model. |
| 22 | [Overlap and parallelism](content/chapters/22-overlap-and-parallelism.md) | Part 5 — Many GPUs | CUDA streams and events: overlap host-device copies with compute; DP/FSDP/TP/PP/EP from the kernel's view. Bucketed DDP and ZeRO-1 are the lab in chapter 25. |
| 23 | [Beyond CUDA](content/chapters/23-beyond-cuda.md) | Part 6 — Beyond | A pure-Python wavefront reduction, a 64-bit ballot, the ridge point, and a scratchpad budget. The checks run without a GPU. |
| 24 | [A CuTe DSL GEMM](content/chapters/24-cute-dsl-gemm.md) | Part 3 — Matrix multiplication | CuTe DSL Ampere GEMM: layouts, a TiledCopy, a TiledMMA, and a fused bias-plus-GELU epilogue. |
| 25 | [Data parallel and ZeRO-1](content/chapters/25-ddp-and-zero.md) | Part 5 — Many GPUs | Bucketed overlap DDP on gloo, ZeRO-1 optimizer-state sharding, and a gated two-GPU NCCL sweep. |
| 26 | [Hopper GEMM](content/chapters/26-hopper-gemm.md) | Part 3 — Matrix multiplication | TMA box address, the 128-byte swizzle, a wgmma descriptor, and sm_90a kernels gated on an H100. |
| 27 | [Debugging GPU kernels](content/chapters/27-debugging-gpu-kernels.md) | Part 1 — The machine | Fix five buggy CUDA kernels under compute-sanitizer, parse recorded sanitizer reports, and find racing reads and out-of-bounds threads in plain Python. |
| 28 | [Reading profiler reports](content/chapters/28-reading-profiler-reports.md) | Part 1 — The machine | Parse a recorded ncu CSV and nsys stats export, place kernels on the roofline from raw counters, and classify them: memory-bound, compute-bound, latency-bound, uncoalesced, bank-conflicted, tail-limited, launch-bound. The checks run without a GPU. |
| 29 | [GEMM scheduling](content/chapters/29-gemm-scheduling.md) | Part 3 — Matrix multiplication | Wave counts, split-K, stream-K ranges, persistent tile order, fused epilogues, a schedule chooser, and Triton kernels on tall-skinny shapes. |
| 30 | [Radix sort, top-k, and sampling](content/chapters/30-sort-top-k-and-sampling.md) | Part 2 — CUDA patterns | Order-preserving float keys, a stable LSD radix sort, merge path, top-k by radix select, top-p without a sort, and Philox seeded sampling, with a CUDA digit histogram. |
| 31 | [Convolution and stencils](content/chapters/31-convolution-and-stencils.md) | Part 2 — CUDA patterns | Tiled conv2d with halo cells and a constant-memory filter, a z-coarsened 3D stencil, im2col, implicit GEMM, and a fused conv, ReLU, and maxpool. The CPU checks run anywhere. |
| 32 | [Sparse and irregular work](content/chapters/32-sparse-and-irregular-work.md) | Part 2 — CUDA patterns | COO, CSR, ELL, and hybrid byte costs, three CSR SpMV kernels and merge path, frontier and direction-optimizing BFS, and an atomic work queue. The CPU checks run anywhere. |
| 33 | [FP8, MXFP4, and NVFP4](content/chapters/33-fp8-and-fp4.md) | Part 4 — Kernels for ML | Bit-exact E4M3, E5M2, and E2M1 encoders against torch, MXFP4 and NVFP4 quantizers, stochastic rounding, and an emulated block-scaled GEMM, plus a Triton check on the A100. The CPU checks run anywhere. |
| 34 | [Clusters and distributed shared memory](content/chapters/34-clusters-and-distributed-shared-memory.md) | Part 3 — Matrix multiplication | Cluster launch and `map_shared_rank` addressing, a DSMEM histogram and reduction, cluster-fit and TMA-multicast byte models, and sm_90 kernels gated on an H100. |

Chapter 24 belongs to Part 3. It is numbered 24, after chapter 23, because chapters 00 through 23 stay where they are.

Chapter 25 belongs to Part 5, with chapters 21 and 22. It is numbered 25, after chapter 24, because chapters 00 through 24 stay where they are. Filename order places 25 after 24.

Chapter 26 belongs to Part 3, with chapters 10 to 14 and chapter 24. It is numbered 26, after chapter 25, because chapters 00 through 25 stay where they are. Filename order places 26 after 25. Prerequisites are chapter 12 and chapter 24.

Chapter 27 belongs to Part 1, with chapters 0 to 5. It is numbered 27, after chapter 26, because chapters 00 through 26 stay where they are. Filename order places 27 after 26. Prerequisites are chapters 2, 5, 6, 8, and 14.
Chapter 29 belongs to Part 3, with chapters 10 to 14, 24, and 26. Its file number is 29 because chapters 00 through 26 stay where they are. Prerequisites are chapters 04, 10, and 14.
Chapter 28 belongs to Part 1, with chapters 0 to 5. It is numbered 28, after chapter 26, because chapters 00 through 26 stay where they are. Filename order places 28 after 26. Prerequisites are chapter 4 and chapter 5. Its lab runs without a GPU.
Chapter 30 belongs to Part 2, with chapters 6 to 9. It is numbered 30 because chapters 00 through 26 stay where they are. Prerequisites are chapter 6 and chapter 7.

Chapter 31 belongs to Part 2, with chapters 6 to 9. It is numbered 31, after chapter 27, because chapters 00 through 27 stay where they are. Prerequisites are chapters 3, 8, 10, and 14.

Chapter 32 belongs to Part 2, with chapters 6 to 9. It is numbered 32 because other chapters fill the numbers 28 to 31. Prerequisites are chapters 3, 6, and 7.

Chapter 33 belongs to Part 4, with chapters 15 to 20. It is numbered 33, after chapter 27, because chapters 00 through 27 stay where they are. Filename order places 33 after 27. Prerequisites are chapters 9 and 18.

Chapter 34 belongs to Part 3, with chapters 10 to 14, 24, and 26. It is numbered 34 because chapters 00 through 27 stay where they are. Filename order places 34 after them. Prerequisites are chapters 7, 12, and 26.

## Run it locally

You need Python 3.11 or later and Node 20 or later.

1. Copy the environment file and fill it in:

   ```bash
   cp .env.example .env
   ```

   Set `SSO_PROXY_SECRET` to the same secret configured on the trusted Caddy
   reverse proxy. Add `SSO_USERS` mappings if an Authelia username differs from
   the name used by existing course progress, for example `alice:alice,bob:robert`.

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
is built on a CUDA *devel* base image (`pytorch/pytorch:2.14.1-cuda13.0-cudnn9-devel`),
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
Both builds fail on maths that KaTeX can't render.

You don't need to run these builds to deploy. The Docker build prints both PDFs
inside the image, so each image carries the PDFs that match its own chapters.

## Deploy

A deploy is a merge to `main`. The GitHub Actions workflow
[build-and-deploy](.github/workflows/deploy.yml) builds the image, including both
PDFs, pushes it to GitHub Container Registry as
`ghcr.io/kashifulhaque/learn-gpu`, and runs `deploy learn-gpu` on the server. That
command pulls the new image and restarts the container. The server never builds
from source.

To run the image yourself, use the example [docker-compose.yml](docker-compose.yml):

```bash
docker compose pull
docker compose up -d
```

The container joins the external `edge` network and answers on port 8000, which
is where a reverse proxy should send traffic. See
[Caddyfile.example](Caddyfile.example) for a Caddy block that keeps server-sent
events unbuffered, which the lab output stream needs. The compose file also
publishes `127.0.0.1:8091`, so the app stays reachable through an SSH tunnel.

To build the image locally, run `docker build -t learn-gpu:local .`. The build
prints the two PDFs in parallel, so it takes several minutes the first time.
After that, Docker reuses each PDF until a file that PDF reads changes. The
`commit` in each PDF manifest is `unknown`, because the commit hash isn't passed
into the PDF stages, where a value that changes on every commit would force a
rebuild.

## Access

The course relies on the trusted Caddy reverse proxy to enforce Authelia sign-in
and forward `Remote-User` with the configured `SSO_PROXY_SECRET` in `X-SSO-Proxy`.
The login screen redirects to the central portal and returns to the exact course
URL. `SSO_USERS` can map portal usernames to existing progress names.
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
