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
endpoint, the same as learn-inference. Hopper (H100, sm_90) and Blackwell
(B200, sm_100) features are taught in prose, figures, and slides, with Ampere
labs that exercise the same idea where one exists.

Labs write CUDA C++ compiled from Python with
`torch.utils.cpp_extension.load_inline`, or Triton. Every lab has `meta.json`,
`starter.py`, `tests.py`, and `solution.py`.

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
| 14 | 14-triton | Triton | Part 3 — Matrix multiplication | 14-triton-matmul | Triton matmul with block pointers, grouped (L2-friendly) ordering, and autotuning vs cuBLAS. |
| 15 | 15-softmax-and-normalization | Softmax and normalization | Part 4 — Kernels for ML | 15-softmax-layernorm | Online softmax and LayerNorm forward and backward (Welford) in Triton. |
| 16 | 16-flash-attention | FlashAttention | Part 4 — Kernels for ML | 16-flash-attention | FlashAttention-2 forward and backward in Triton, with causal masking. |
| 17 | 17-fusion-and-compilers | Fusion, compilers, and launch overhead | Part 4 — Kernels for ML | 17-compile-graphs | torch.compile and Inductor output, CUDA graphs, custom ops with torch.library. |
| 18 | 18-quantized-kernels | Quantized kernels | Part 4 — Kernels for ML | 18-w4a16 | Group-quantize weights to int4 and write a fused dequantize-GEMM (W4A16) in Triton. |
| 19 | 19-training-kernels | Training kernels | Part 4 — Kernels for ML | 19-fused-xent | Custom autograd Functions; fused, chunked linear plus cross-entropy that never materializes the logits. |
| 20 | 20-moe-and-sparse-kernels | Mixture-of-experts and sparse kernels | Part 4 — Kernels for ML | 20-grouped-gemm | Token permutation and a grouped GEMM in Triton; 2:4 structured sparsity. |
| 21 | 21-collective-communication | Collective communication | Part 5 — Many GPUs | 21-ring-allreduce | Ring all-reduce with torch.distributed point-to-point; the alpha-beta cost model. |
| 22 | 22-overlap-and-parallelism | Overlap and parallelism | Part 5 — Many GPUs | 22-streams-overlap | CUDA streams and events: overlap host-device copies with compute; DP/FSDP/TP/PP/EP from the kernel's view. |
| 23 | 23-beyond-cuda | Beyond CUDA | Part 6 — Beyond | (none) | AMD CDNA and HIP, TPUs and XLA/Pallas, Apple silicon, and tile DSLs (ThunderKittens, TileLang, CuTe DSL). |
