# Curriculum

`learn-gpu` teaches GPU programming for machine learning, from the streaming
multiprocessor up to the kernels that train and serve models. Its companion,
`learn-inference` (qwen.ifkash.dev), builds an inference engine for one model;
this course is about the machine and the kernels themselves, and goes deeper
on every kernel topic that course touches.

Reader: a capable engineer who writes Python and PyTorch and knows what a
transformer is, but has never written a kernel. No C++ background assumed
beyond reading it.

Lab hardware: one **A100 80GB PCIe** (Ampere, sm_80) on a RunPod serverless
endpoint, the same as learn-inference. Chapter 26 is a Hopper lab, gated on an
H100: the CPU checks run on any machine, and the TMA and `wgmma` kernels run
only when an H100 endpoint is configured. Blackwell (B200, sm_100) stays in
prose, figures, and slides.

Labs write CUDA C++ compiled from Python with
`torch.utils.cpp_extension.load_inline`, or Triton. Chapter 24 writes the
kernel in the CuTe DSL (`nvidia-cutlass-dsl` 4.8.1), which the worker image
already installs. Every lab has `meta.json`, `starter.py`, `tests.py`, and
`solution.py`.

Each chapter has a book chapter (`content/chapters/`), a lab (`content/labs/`),
and a slide deck (`content/slides/<slug>.md`). Lectures (videos) come later.

| # | Slug | Title | Part | Lab id | Lab: what the reader builds |
|---|---|---|---|---|---|
| 00 | 00-what-you-are-building | What you are building | Part 1 — The machine | 00-hello-gpu | Query the device, time a copy with CUDA events, report bandwidth. |
| 01 | 01-the-gpu-at-a-glance | The GPU at a glance | Part 1 — The machine | 01-device-peaks | Derive peak FLOP/s and bandwidth from device properties and spec constants; compare A100, H100, B200. |
| 02 | 02-the-execution-model | The execution model | Part 1 — The machine | 02-occupancy | An occupancy calculator from registers, shared memory, and block size; measure warp divergence cost. |
| 03 | 03-memory-and-bandwidth | Memory and bandwidth | Part 1 — The machine | 03-bandwidth-probe | Copy kernels: scalar, float4 vectorized, grid-stride, strided; reach the copy ceiling. |
| 04 | 04-roofline-and-latency | The roofline and latency hiding | Part 1 — The machine | 04-roofline | Predict kernel time from the roofline and Little's law, then measure; wave quantization. |
| 05 | 05-profiling | Profiling and reading the machine code | Part 1 — The machine | 05-diagnose | Profile three slow kernels with torch.profiler and CUDA events, diagnose, fix; read PTX/SASS. |
| 06 | 06-reductions | Reductions | Part 2 — CUDA patterns | 06-reduce | Sum and max reductions: shared-memory tree, warp shuffles, one-pass with atomics; hit bandwidth. |
| 07 | 07-scan-and-histogram | Scan and histogram | Part 2 — CUDA patterns | 07-scan-histogram | Block scan (Kogge-Stone, Blelloch), device-wide scan; privatized histogram. |
| 08 | 08-shared-memory-and-transpose | Shared memory and the transpose | Part 2 — CUDA patterns | 08-transpose | Matrix transpose: naive, shared-memory tiled, padded and swizzled against bank conflicts. |
| 09 | 09-floating-point-on-the-gpu | Floating point on the GPU | Part 2 — CUDA patterns | 09-numerics | fp32/tf32/bf16/fp16/fp8 round-trips, accumulation error, Kahan summation, atomic nondeterminism. |
| 10 | 10-gemm-from-scratch | GEMM from scratch | Part 3 — Matrix multiplication | 10-sgemm | SGEMM: naive, coalesced, shared-memory tiled, 2D register blocking, vectorized; percent of cuBLAS. |
| 11 | 11-tensor-cores | Tensor cores | Part 3 — Matrix multiplication | 11-wmma-gemm | fp16/bf16 GEMM on tensor cores with WMMA, then mma.sync with ldmatrix. |
| 12 | 12-pipelining-and-async-copies | Pipelining and asynchronous copies | Part 3 — Matrix multiplication | 12-cp-async | Multistage cp.async pipeline for the tensor-core GEMM. Hopper TMA/wgmma and Blackwell tcgen05 in prose. |
| 13 | 13-cutlass-and-cute | CUTLASS and CuTe layouts | Part 3 — Matrix multiplication | 13-layouts | Implement CuTe layout algebra (shape/stride, coalesce, composition, complement, swizzle) in Python. |
| 14 | 14-triton | Triton | Part 3 — Matrix multiplication | 14-triton-matmul | Triton matmul with plain pointer arithmetic, grouped (L2-friendly) ordering, and autotuning vs cuBLAS. |
| 15 | 15-softmax-and-normalization | Softmax and normalization | Part 4 — Kernels for ML | 15-softmax-layernorm | Online softmax and LayerNorm forward and backward (Welford) in Triton. |
| 16 | 16-flash-attention | FlashAttention | Part 4 — Kernels for ML | 16-flash-attention | FlashAttention-2 forward and backward in Triton, with causal masking. |
| 17 | 17-fusion-and-compilers | Fusion, compilers, and launch overhead | Part 4 — Kernels for ML | 17-compile-graphs | torch.compile and Inductor output, CUDA graphs, custom ops with torch.library. |
| 18 | 18-quantized-kernels | Quantized kernels | Part 4 — Kernels for ML | 18-w4a16 | Group-quantize weights to int4 and write a fused dequantize-GEMM (W4A16) in Triton. |
| 19 | 19-training-kernels | Training kernels | Part 4 — Kernels for ML | 19-fused-xent | Custom autograd Functions; fused, chunked linear plus cross-entropy that never materializes the logits. |
| 20 | 20-moe-and-sparse-kernels | Mixture-of-experts and sparse kernels | Part 4 — Kernels for ML | 20-grouped-gemm | Token permutation and a grouped GEMM in Triton; 2:4 structured sparsity. |
| 21 | 21-collective-communication | Collective communication | Part 5 — Many GPUs | 21-ring-allreduce | Ring all-reduce with torch.distributed point-to-point; the alpha-beta cost model. |
| 22 | 22-overlap-and-parallelism | Overlap and parallelism | Part 5 — Many GPUs | 22-streams-overlap | CUDA streams and events: overlap host-device copies with compute; DP/FSDP/TP/PP/EP from the kernel's view. Bucketed DDP and ZeRO-1 are the lab in chapter 25. |
| 23 | 23-beyond-cuda | Beyond CUDA | Part 6 — Beyond | 23-wavefront-reduction | A pure-Python wavefront reduction, a 64-bit ballot, the ridge point, and a scratchpad budget. The checks run without a GPU. |
| 24 | 24-cute-dsl-gemm | A CuTe DSL GEMM | Part 3 — Matrix multiplication | 24-cute-dsl-gemm | CuTe DSL Ampere GEMM: layouts, a TiledCopy, a TiledMMA, and a fused bias-plus-GELU epilogue. |
| 25 | 25-ddp-and-zero | Data parallel and ZeRO-1 | Part 5 — Many GPUs | 25-ddp-and-zero | Bucketed overlap DDP on gloo, ZeRO-1 optimizer-state sharding, and a gated two-GPU NCCL sweep. |
| 26 | 26-hopper-gemm | Hopper GEMM | Part 3 — Matrix multiplication | 26-hopper-gemm | TMA box address, the 128-byte swizzle, a wgmma descriptor, and sm_90a kernels gated on an H100. |
| 27 | 27-debugging-gpu-kernels | Debugging GPU kernels | Part 1 — The machine | 27-sanitize | Five buggy CUDA kernels fixed under compute-sanitizer, a parser for recorded sanitizer reports, and pure-Python race and bounds reasoning. The CPU checks run anywhere. |
| 28 | 28-reading-profiler-reports | Reading profiler reports | Part 1 — The machine | 28-profile-reports | Parse recorded ncu CSV and nsys stats exports, place kernels on the roofline from raw counters, and classify them (memory-bound, compute-bound, latency-bound, uncoalesced, bank conflicts, tail effect, launch-bound). No GPU needed. |
| 29 | 29-gemm-scheduling | GEMM scheduling | Part 3 — Matrix multiplication | 29-gemm-scheduling | Wave counts, split-K, stream-K ranges, persistent tile order, fused epilogues, a schedule chooser on a cost model, and Triton kernels on tall-skinny shapes. |
| 30 | 30-sort-top-k-and-sampling | Radix sort, top-k, and sampling | Part 2 — CUDA patterns | 30-radix-sort-top-k | Order-preserving float keys, a stable LSD radix sort, merge path, top-k by radix select, top-p without a sort, and Philox seeded sampling, with a CUDA digit histogram. |
| 31 | 31-convolution-and-stencils | Convolution and stencils | Part 2 — CUDA patterns | 31-conv-stencil | Output sizes, halo overhead, im2col and implicit-GEMM index maps, a 7-point stencil, and a fused conv, ReLU, and maxpool, all checked on the CPU. A tiled CUDA conv2d with a constant-memory filter and a z-coarsened 3D stencil run on the A100. |

Chapter 24 is part of Part 3, with the GEMM chapters. Its file number is 24, after chapter 23, because chapters 00 through 23 are not renumbered.

Chapter 25 is part of Part 5, with chapters 21 and 22. Its file number is 25, after chapter 24, because chapters 00 through 24 are not renumbered. Filename order places 25 after 24.

Chapter 26 is part of Part 3, with chapters 10 to 14 and chapter 24. Its file number is 26, after chapter 25, because chapters 00 through 25 are not renumbered. Filename order places 26 after 25. Prerequisites are chapter 12 and chapter 24.

Chapter 27 is part of Part 1, with chapters 0 to 5. Its file number is 27, after chapter 26, because chapters 00 through 26 are not renumbered. Filename order places 27 after 26. Prerequisites are chapters 2, 5, 6, 8, and 14.
Chapter 29 is part of Part 3, with chapters 10 to 14, 24, and 26. Its file number is 29 because chapters 00 through 26 are not renumbered, and chapters 27 and 28 are numbered by other work. Prerequisites are chapters 04, 10, and 14. It builds on the split-K and stream-K counts that chapter 4's lab grades and extends chapter 14's Triton kernel. The skinny decode GEMM and the W4A16 GEMV stay in learn-inference.
Chapter 28 is part of Part 1, with chapters 0 to 5. Its file number is 28, after chapter 26, because chapters 00 through 26 are not renumbered. Filename order places 28 after 26. Prerequisites are chapter 4 and chapter 5. It goes beyond chapter 5's recorded-report lab: the roofline placement from raw counters, stall reasons, occupancy, sector and bank-conflict counters, wave quantization, and an `nsys stats` export. Its lab is `device_agnostic` with `needs_gpu: false`.
Chapter 30 is part of Part 2, with chapters 6 to 9. Its file number is 30, because chapters 00 through 26 are not renumbered and chapters 27 to 29 belong to other topics. Filename order places 30 after 26. Prerequisites are chapter 6 and chapter 7. Sampling inside an inference engine belongs to learn-inference. This chapter owns the kernels.

Chapter 31 is part of Part 2, with chapters 6 to 9. Its file number is 31, after chapter 27, because chapters 00 through 27 are not renumbered. Prerequisites are chapters 3, 8, 10, and 14.
