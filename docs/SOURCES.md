# Sources by chapter

Each chapter lists 6 to 12 references. For each one, the entry gives the title, authors or organization, year, URL, and what the chapter should take from it. Numbers in a chapter must still come from `FACTS.md` or from the lab's own measurements; use these sources for explanation and citation, and check any number against `FACTS.md` before reusing it.

**URL check.** Every URL was requested on 2026-10-22 with a browser-style user agent. All returned HTTP 200 except the following, which are reachable but answer automated requests differently:

- `https://epubs.siam.org/doi/10.1137/0914050` returns 403 to automated clients (publisher bot protection). The page is the SIAM landing page for Higham (1993), confirmed by search results.
- `https://doi.org/10.1109/TC.1973.5009159` returns 202 and redirects to `https://ieeexplore.ieee.org/document/5009159/`.
- `https://zenodo.org/record/1236156` returned 200 on one request and 403 on a later one; its page metadata names the Roofline paper (Zenodo record of the CACM article).

Authors are as printed on the source's first page or abstract page. arXiv titles, authors, and years were checked against the arXiv API.

Descriptions of NVIDIA, AMD, Google, and Apple documentation refer to the pages as read on the check date; those pages change, so prefer the version-pinned URLs where given.

## 00-what-you-are-building: What you are building

Part 1 — The machine.

1. **CUDA Programming Guide.** NVIDIA, 2026. <https://docs.nvidia.com/cuda/cuda-programming-guide/>  
   Take: The grid, block, thread, and kernel vocabulary and the asynchronous launch model.
2. **Making Deep Learning Go Brrrr From First Principles.** Horace He, 2022. <https://horace.io/brrr_intro.html>  
   Take: The compute, bandwidth, and overhead framing the whole course reuses.
3. **GPU Glossary.** Modal, 2025. <https://modal.com/gpu-glossary>  
   Take: Short, linked definitions of device, memory, and software terms; point readers here when they meet a new word.
4. **An Even Easier Introduction to CUDA.** Mark Harris, NVIDIA, 2017. <https://developer.nvidia.com/blog/even-easier-introduction-cuda/>  
   Take: The smallest complete CUDA program (add two arrays) and how to launch it.
5. **CUDA semantics.** PyTorch, 2026. <https://docs.pytorch.org/docs/stable/notes/cuda.html>  
   Take: Asynchronous execution, streams, and why timing needs events or a synchronize.
6. **torch.utils.cpp_extension (load_inline).** PyTorch, 2026. <https://docs.pytorch.org/docs/2.14/cpp_extension.html>  
   Take: Exact signature and arguments for the lab build path.
7. **torch.cuda.Event.** PyTorch, 2026. <https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html>  
   Take: Event-based timing, used in the first lab.
8. **GPU MODE lectures.** GPU MODE community, 2024. <https://github.com/gpu-mode/lectures>  
   Take: Lecture 1 on profiling CUDA kernels in PyTorch; the community's reference lecture list.
9. **CUDA C++ Best Practices Guide.** NVIDIA, 2026. <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>  
   Take: The APOD cycle and the list of what to measure first.

## 01-the-gpu-at-a-glance: The GPU at a glance

Part 1 — The machine.

1. **NVIDIA A100 Tensor Core GPU datasheet.** NVIDIA, 2021. <https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf>  
   Take: Rated TFLOP/s, bandwidth, TDP, and NVLink figures for the lab GPU (PCIe and SXM).
2. **NVIDIA A100 Tensor Core GPU Architecture (GA100 whitepaper).** NVIDIA, 2020. <https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf>  
   Take: SM count, per-SM tensor-core rates, 40 MB L2, 164 KB shared memory, cp.async, 2:4 sparsity.
3. **NVIDIA H100 Tensor Core GPU Architecture whitepaper (preliminary figures).** NVIDIA, 2022. <https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf>  
   Take: 132 or 114 SMs, 50 MB L2, TMA, clusters; throughput figures are preliminary, so cross-check with the product page.
4. **NVIDIA Hopper Architecture In-Depth.** Michael Andersch and others, NVIDIA, 2022. <https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/>  
   Take: SM design, thread block clusters, distributed shared memory, TMA, FP8.
5. **NVIDIA H100 Tensor Core GPU product page.** NVIDIA, 2026. <https://www.nvidia.com/en-us/data-center/h100/>  
   Take: Final H100 SXM and NVL throughput table (values are with sparsity; halve for dense).
6. **Inside NVIDIA Blackwell Ultra: The Chip Powering the AI Factory Era.** NVIDIA, 2025. <https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/>  
   Take: Dual-die structure, 256 KB tensor memory per SM, NVFP4 rates, HBM3e, NVLink 5.
7. **NVIDIA HGX platform page.** NVIDIA, 2026. <https://www.nvidia.com/en-us/data-center/hgx/>  
   Take: HGX B200 eight-GPU totals; divide by 8 for per-GPU figures.
8. **Compute capabilities (CUDA Programming Guide appendix).** NVIDIA, 2026. <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html>  
   Take: Per-architecture limits table: warps, blocks, registers, shared memory.
9. **Nvidia's B200: Keeping the CUDA Juggernaut Rolling.** Chips and Cheese, 2025. <https://chipsandcheese.com/p/nvidias-b200-keeping-the-cuda-juggernaut>  
   Take: Measured B200 structure: 148 SMs, 126 MB L2, tensor memory geometry.
10. **Benchmarking and Dissecting the Nvidia Hopper GPU Architecture.** Weile Luo and others, 2024. <https://arxiv.org/abs/2402.13499>  
   Take: Measured latencies and mma throughput on a PCIe A100, an RTX 4090, and an H800 PCIe.

## 02-the-execution-model: The execution model

Part 1 — The machine.

1. **Programming model (CUDA Programming Guide).** NVIDIA, 2026. <https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html>  
   Take: Thread hierarchy, warps, blocks, clusters, and how branch divergence is described.
2. **Advanced kernel programming: independent thread scheduling.** NVIDIA, 2026. <https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html>  
   Take: Per-thread program counter on Volta and later; why __syncwarp is required.
3. **Compute capabilities (CUDA Programming Guide appendix).** NVIDIA, 2026. <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html>  
   Take: Source for the per-architecture occupancy limits table.
4. **Better Performance at Lower Occupancy.** Vasily Volkov, 2010. <https://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf>  
   Take: Why instruction-level parallelism can replace occupancy; the case against chasing 100%.
5. **NVIDIA Tesla V100 GPU Architecture whitepaper.** NVIDIA, 2017. <https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf>  
   Take: Independent thread scheduling and the Volta SM, which Ampere and Hopper inherit.
6. **Inside Volta: The World's Most Advanced Data Center GPU.** Luke Durant and others, NVIDIA, 2017. <https://developer.nvidia.com/blog/inside-volta/>  
   Take: Readable account of independent thread scheduling and warp convergence.
7. **Using CUDA Warp-Level Primitives.** Yuan Lin and Vinod Grover, NVIDIA, 2018. <https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/>  
   Take: Shuffle, vote, and __syncwarp with masks; porting pre-Volta warp-synchronous code.
8. **CUDA C++ Best Practices Guide: Execution Configuration Optimizations.** NVIDIA, 2026. <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>  
   Take: Block-size guidance and the occupancy API.
9. **Nsight Compute Profiling Guide.** NVIDIA, 2026. <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>  
   Take: How achieved and theoretical occupancy are reported and which limiter is named.
10. **Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking.** Zhe Jia and others, 2018. <https://arxiv.org/abs/1804.06826>  
   Take: Method for measuring instruction and memory latency on a real SM.

## 03-memory-and-bandwidth: Memory and bandwidth

Part 1 — The machine.

1. **CUDA C++ Best Practices Guide: Memory Optimizations.** NVIDIA, 2026. <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>  
   Take: 32-byte sector coalescing rule, aligned and misaligned examples, strided access.
2. **How to Access Global Memory Efficiently in CUDA C/C++ Kernels.** Mark Harris, NVIDIA, 2013. <https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/>  
   Take: Offset and stride experiments that map bandwidth against access pattern.
3. **CUDA Pro Tip: Increase Performance with Vectorized Memory Access.** Justin Luitjens, NVIDIA, 2013. <https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/>  
   Take: float4 loads and the instruction-count argument for the vectorized copy lab.
4. **CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops.** Mark Harris, NVIDIA, 2013. <https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/>  
   Take: The grid-stride loop pattern.
5. **Benchmarking and Dissecting the Nvidia Hopper GPU Architecture.** Weile Luo and others, 2024. <https://arxiv.org/abs/2402.13499>  
   Take: A100 PCIe cache and global memory latencies and throughput in cycles.
6. **Nvidia's H100: Funny L2, and Tons of Bandwidth.** Chips and Cheese, 2023. <https://chipsandcheese.com/p/nvidias-h100-funny-l2-and-tons-of-bandwidth>  
   Take: Split L2 with near and far partitions; measured L2 and DRAM bandwidth.
7. **NVIDIA A100 Tensor Core GPU Architecture (GA100 whitepaper).** NVIDIA, 2020. <https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf>  
   Take: L2 partitioning, 5120-byte per clock L2 read bandwidth, L2 residency controls.
8. **How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog.** Simon Boehm, 2022. <https://siboehm.com/articles/22/CUDA-MMM>  
   Take: The coalescing step (309 to 1986 GFLOP/s on an RTX A6000) as a worked example.
9. **Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking.** Zhe Jia and others, 2018. <https://arxiv.org/abs/1804.06826>  
   Take: Pointer-chasing method for measuring cache and memory latency.

## 04-roofline-and-latency: The roofline and latency hiding

Part 1 — The machine.

1. **Roofline: An Insightful Visual Performance Model for Multicore Architectures.** Samuel Williams, Andrew Waterman, David Patterson, 2009. <https://zenodo.org/record/1236156>  
   Take: The original roofline model and the arithmetic-intensity definition.
2. **Making Deep Learning Go Brrrr From First Principles.** Horace He, 2022. <https://horace.io/brrr_intro.html>  
   Take: Compute-bound, memory-bound, and overhead-bound regimes for ML ops.
3. **Better Performance at Lower Occupancy.** Vasily Volkov, 2010. <https://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf>  
   Take: Bytes in flight needed to saturate bandwidth.
4. **Matrix Multiplication Background User's Guide.** NVIDIA, 2026. <https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html>  
   Take: Arithmetic intensity, tile quantization, and wave quantization with A100 108-SM examples.
5. **GPU Performance Background User's Guide.** NVIDIA, 2026. <https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html>  
   Take: Math-bound versus memory-bound, ops:byte ratio.
6. **How to Scale Your Model: All About Rooflines.** Jacob Austin and others, Google DeepMind, 2025. <https://jax-ml.github.io/scaling-book/roofline/>  
   Take: Roofline applied to transformer layers; a good second explanation.
7. **Transformer Inference Arithmetic.** Carol Chen (kipply), 2022. <https://kipp.ly/transformer-inference-arithmetic/>  
   Take: Worked FLOP and byte counts for decoding.
8. **Benchmarking and Dissecting the Nvidia Hopper GPU Architecture.** Weile Luo and others, 2024. <https://arxiv.org/abs/2402.13499>  
   Take: Measured A100 PCIe global-memory latency (466 cycles) used for Little's law.

## 05-profiling: Profiling and reading the machine code

Part 1 — The machine.

1. **Nsight Compute Profiling Guide.** NVIDIA, 2026. <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>  
   Take: Metrics, sections, speed-of-light analysis, and the roofline chart.
2. **Nsight Systems User Guide.** NVIDIA, 2026. <https://docs.nvidia.com/nsight-systems/UserGuide/index.html>  
   Take: Timeline view for launch gaps, streams, and host-device copies.
3. **PyTorch Profiler recipe.** PyTorch, 2026. <https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html>  
   Take: torch.profiler usage, tables, and Chrome trace export.
4. **PTX ISA.** NVIDIA, 2026. <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>  
   Take: Reference for reading PTX instructions.
5. **CUDA Binary Utilities (cuobjdump, nvdisasm).** NVIDIA, 2026. <https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html>  
   Take: Dumping SASS from a compiled kernel and the SASS instruction listing per architecture.
6. **CUDA Runtime API: event management.** NVIDIA, 2026. <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html>  
   Take: cudaEventRecord and cudaEventElapsedTime semantics.
7. **Compiler Explorer.** Matt Godbolt and contributors, 2026. <https://godbolt.org>  
   Take: Interactive PTX and SASS output from nvcc for tiny kernels.
8. **GPU MODE lectures.** GPU MODE community, 2024. <https://github.com/gpu-mode/lectures>  
   Take: Profiling lectures with worked Nsight Compute sessions.
9. **Triton Proton profiler.** Triton project, 2026. <https://github.com/triton-lang/triton/tree/main/third_party/proton>  
   Take: Profiler built for Triton kernels.

## 06-reductions: Reductions

Part 2 — CUDA patterns.

1. **Optimizing Parallel Reduction in CUDA.** Mark Harris, NVIDIA, 2007. <https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf>  
   Take: Seven-step kernel progression from 8.054 ms to 0.268 ms (30x) on a G80.
2. **Faster Parallel Reductions on Kepler.** Justin Luitjens, NVIDIA, 2014. <https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/>  
   Take: Warp shuffle reductions replacing shared-memory trees.
3. **Using CUDA Warp-Level Primitives.** Yuan Lin and Vinod Grover, NVIDIA, 2018. <https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/>  
   Take: __shfl_down_sync and the full-mask rule.
4. **Cooperative Groups: Flexible CUDA Thread Programming.** Mark Harris and Kyrylo Perelygin, NVIDIA, 2017. <https://developer.nvidia.com/blog/cooperative-groups/>  
   Take: Group-level reduce abstractions.
5. **CUB: cub::DeviceReduce.** NVIDIA CCCL team, 2026. <https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceReduce.html>  
   Take: The library reference to compare the lab kernels against.
6. **The Accuracy of Floating Point Summation.** Nicholas J. Higham, 1993. <https://epubs.siam.org/doi/10.1137/0914050>  
   Take: Error bounds for recursive, pairwise, and compensated summation.
7. **Floating Point and IEEE 754 Compliance for NVIDIA GPUs.** NVIDIA, 2026. <https://docs.nvidia.com/cuda/floating-point/index.html>  
   Take: Why a parallel reduction order changes the result.
8. **CUDA C++ Best Practices Guide.** NVIDIA, 2026. <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>  
   Take: Atomics and shared-memory guidance.

## 07-scan-and-histogram: Scan and histogram

Part 2 — CUDA patterns.

1. **Single-pass Parallel Prefix Scan with Decoupled Look-back.** Duane Merrill and Michael Garland, NVIDIA, 2016. <https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf>  
   Take: Single-pass scan that moves about 2n data and matches memcpy throughput.
2. **Parallel Prefix Sum (Scan) with CUDA.** Mark Harris, Shubhabrata Sengupta, John Owens (GPU Gems 3, ch. 39), 2007. <https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda>  
   Take: Naive (Hillis-Steele style), work-efficient Blelloch scan, and bank-conflict avoidance.
3. **Prefix Sums and Their Applications.** Guy E. Blelloch, 1990. <https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf>  
   Take: The up-sweep and down-sweep algorithm and applications such as compaction.
4. **A Parallel Algorithm for the Efficient Solution of a General Class of Recurrence Equations.** Peter M. Kogge and Harold S. Stone, 1973. <https://doi.org/10.1109/TC.1973.5009159>  
   Take: The Kogge-Stone network.
5. **CUB: cub::DeviceScan.** NVIDIA CCCL team, 2026. <https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceScan.html>  
   Take: Production decoupled-look-back scan to benchmark against.
6. **CUB: cub::DeviceHistogram.** NVIDIA CCCL team, 2026. <https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceHistogram.html>  
   Take: Privatized shared-memory histogram design.
7. **Using Shared Memory in CUDA C/C++.** Mark Harris, NVIDIA, 2013. <https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/>  
   Take: Shared memory and __syncthreads semantics for the block scan.
8. **CUB: cub::BlockScan.** NVIDIA CCCL team, 2026. <https://nvidia.github.io/cccl/unstable/cub/api/classcub_1_1BlockScan.html>  
   Take: Block-level scan algorithms (raking, warp-scans) to compare with the lab Kogge-Stone and Blelloch versions.

## 08-shared-memory-and-transpose: Shared memory and the transpose

Part 2 — CUDA patterns.

1. **An Efficient Matrix Transpose in CUDA C/C++.** Mark Harris, NVIDIA, 2013. <https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/>  
   Take: The naive, coalesced, shared-memory, and padded transpose kernels with bandwidth figures.
2. **Using Shared Memory in CUDA C/C++.** Mark Harris, NVIDIA, 2013. <https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/>  
   Take: Static and dynamic shared memory, __syncthreads, bank conflicts.
3. **CUDA C++ Best Practices Guide: Shared Memory and Bank Conflicts.** NVIDIA, 2026. <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>  
   Take: 32 banks of 4 bytes; broadcast rules; strided access conflicts.
4. **CUDA Shared Memory Bank.** Lei Mao, 2022. <https://leimao.github.io/blog/CUDA-Shared-Memory-Bank/>  
   Take: Worked bank-conflict examples with measurements.
5. **CUTLASS Tutorial: Fast Matrix-Transpose in CUDA with Cutlass.** Colfax Research, 2024. <https://research.colfax-intl.com/tutorial-matrix-transpose-in-cutlass/>  
   Take: Swizzled shared-memory layouts as the alternative to padding.
6. **CuTe Swizzle.** Lei Mao, 2024. <https://leimao.github.io/blog/CuTe-Swizzle/>  
   Take: How the Swizzle<B,M,S> function permutes addresses to avoid conflicts.
7. **Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking.** Zhe Jia and others, 2018. <https://arxiv.org/abs/1804.06826>  
   Take: Measured shared-memory latency and bank behavior.
8. **Benchmarking and Dissecting the Nvidia Hopper GPU Architecture.** Weile Luo and others, 2024. <https://arxiv.org/abs/2402.13499>  
   Take: A100 PCIe shared-memory latency of 29 cycles.

## 09-floating-point-on-the-gpu: Floating point on the GPU

Part 2 — CUDA patterns.

1. **OCP 8-bit Floating Point Specification (OFP8) Revision 1.0.** Open Compute Project, 2023. <https://opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf>  
   Take: Exact E4M3 and E5M2 encodings, maxima, and special values.
2. **OCP Microscaling Formats (MX) Specification v1.0.** Open Compute Project, 2023. <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>  
   Take: MXFP8, MXFP6, MXFP4, and MXINT8: block size 32 with an E8M0 scale.
3. **Microscaling Data Formats for Deep Learning.** Bita Darvish Rouhani and others, 2023. <https://arxiv.org/abs/2310.10537>  
   Take: Evidence for MX formats in training and inference.
4. **Introducing NVFP4 for Efficient and Accurate Low-Precision Inference.** NVIDIA, 2025. <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>  
   Take: NVFP4: block size 16, E4M3 block scale, per-tensor FP32 scale.
5. **What Every Computer Scientist Should Know About Floating-Point Arithmetic.** David Goldberg, 1991. <https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html>  
   Take: Rounding, ulp, and cancellation.
6. **Floating Point and IEEE 754 Compliance for NVIDIA GPUs.** NVIDIA, 2026. <https://docs.nvidia.com/cuda/floating-point/index.html>  
   Take: FMA, order of operations, and why GPU results differ from CPU results.
7. **Mixed Precision Training.** Paulius Micikevicius and others, 2017. <https://arxiv.org/abs/1710.03740>  
   Take: fp16 loss scaling and fp32 master weights.
8. **FP8 Formats for Deep Learning.** Paulius Micikevicius and others, 2022. <https://arxiv.org/abs/2209.05433>  
   Take: Why E4M3 for forward values and E5M2 for gradients.
9. **A Study of BFLOAT16 for Deep Learning Training.** Dhiraj Kalamkar and others, 2019. <https://arxiv.org/abs/1905.12322>  
   Take: Why bf16 needs no loss scaling.
10. **Accelerating AI Training with NVIDIA TF32 Tensor Cores.** NVIDIA, 2021. <https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/>  
   Take: TF32 layout: 8 exponent bits and 10 mantissa bits.
11. **Reproducibility.** PyTorch, 2026. <https://docs.pytorch.org/docs/stable/notes/randomness.html>  
   Take: Deterministic algorithms flag and atomics nondeterminism.

## 10-gemm-from-scratch: GEMM from scratch

Part 3 — Matrix multiplication.

1. **How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog.** Simon Boehm, 2022. <https://siboehm.com/articles/22/CUDA-MMM>  
   Take: The progression from 309 to 21,779 GFLOP/s (93.7% of cuBLAS) on an RTX A6000.
2. **Advanced Matrix Multiplication Optimization on NVIDIA GPUs.** Amanzhol Salykov, 2025. <https://salykova.github.io/sgemm-gpu>  
   Take: A second independent worklog (RTX 3090) with a different sequence of optimizations.
3. **CUTLASS: Fast Linear Algebra in CUDA C++.** Andrew Kerr and others, NVIDIA, 2017. <https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/>  
   Take: The tile hierarchy (thread block, warp, thread) that the lab reproduces.
4. **Matrix Multiplication Background User's Guide.** NVIDIA, 2026. <https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html>  
   Take: Tile and wave quantization; arithmetic intensity of GEMM.
5. **Anatomy of High-Performance Matrix Multiplication.** Kazushige Goto and Robert van de Geijn, 2008. <https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf>  
   Take: Cache-blocking theory that tiling copies.
6. **cuBLAS documentation.** NVIDIA, 2026. <https://docs.nvidia.com/cuda/cublas/index.html>  
   Take: The baseline library and its tensor-core math modes.
7. **Optimizing Matrix Multiplication on a GPU.** Lei Mao, 2022. <https://leimao.github.io/blog/CUDA-Matrix-Multiplication/>  
   Take: Concise tiled GEMM derivation.

## 11-tensor-cores: Tensor cores

Part 3 — Matrix multiplication.

1. **Programming Tensor Cores in CUDA 9.** Jeremy Appleyard and Scott Yokim, NVIDIA, 2017. <https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/>  
   Take: The WMMA API and fragment model.
2. **Warp matrix functions (CUDA Programming Guide).** NVIDIA, 2026. <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html>  
   Take: Supported WMMA shapes and types; alignment rules.
3. **PTX ISA: warp-level matrix instructions (mma, ldmatrix).** NVIDIA, 2026. <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>  
   Take: mma.sync shapes and fragment layouts per type; ldmatrix.
4. **Dissecting Tensor Cores via Microbenchmarks: Latency, Throughput and Numeric Behaviors.** Wei Sun and others, 2022. <https://arxiv.org/abs/2206.02874>  
   Take: mma instruction latency and throughput; rounding inside the tensor core.
5. **Benchmarking and Dissecting the Nvidia Hopper GPU Architecture.** Weile Luo and others, 2024. <https://arxiv.org/abs/2402.13499>  
   Take: A100 PCIe mma.sync throughput up to 310.6 TFLOP/s against a 312 peak.
6. **NVIDIA Tensor Core Programmability, Performance and Precision.** Stefano Markidis and others, 2018. <https://arxiv.org/abs/1803.04014>  
   Take: Early evidence on tensor-core precision behavior.
7. **How to Write a Fast Matrix Multiplication from Scratch with Tensor Cores.** Alex Armbruster, 2024. <https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html>  
   Take: A readable sequence from WMMA to a cuBLAS-class kernel on an A100.
8. **NVIDIA A100 Tensor Core GPU Architecture (GA100 whitepaper).** NVIDIA, 2020. <https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf>  
   Take: Per-tensor-core 256 FP16 FMA per clock; TF32, BF16, and FP64 modes.
9. **GPU Glossary: Tensor Core.** Modal, 2025. <https://modal.com/gpu-glossary/device-hardware/tensor-core>  
   Take: Short definition with links to instructions.

## 12-pipelining-and-async-copies: Pipelining and asynchronous copies

Part 3 — Matrix multiplication.

1. **Controlling Data Movement to Boost Performance on the NVIDIA Ampere Architecture.** NVIDIA, 2020. <https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/>  
   Take: cp.async, pipelines, and the multistage mainloop.
2. **PTX ISA: cp.async, cp.async.bulk, tcgen05.** NVIDIA, 2026. <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>  
   Take: Exact semantics of commit_group and wait_group; bulk and tensor copies; tensor memory instructions.
3. **CUTLASS Tutorial: Mastering the NVIDIA Tensor Memory Accelerator (TMA).** Colfax Research, 2024. <https://research.colfax-intl.com/tutorial-hopper-tma/>  
   Take: TMA descriptors, mbarrier transaction counts, multicast.
4. **CUTLASS Tutorial: Fast Matrix-Multiplication with WGMMA on NVIDIA Hopper GPUs.** Colfax Research, 2024. <https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/>  
   Take: wgmma issue model, warpgroups, fences.
5. **FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision.** Jay Shah and others, 2024. <https://arxiv.org/abs/2407.08608>  
   Take: Warp specialization and ping-pong scheduling on Hopper; 740 TFLOP/s FP16.
6. **tcgen05 MMA Programming Guide (CuTe DSL).** NVIDIA, 2026. <https://docs.nvidia.com/cutlass/4.6.0/media/docs/pythonDSL/mma_docs/tcgen05_programming.html>  
   Take: Tensor memory, single-thread MMA issue, CTA pairs on Blackwell.
7. **CUDA Hopper Tuning Guide.** NVIDIA, 2026. <https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html>  
   Take: 228 KB shared memory, clusters, TMA facts.
8. **CUDA Blackwell Tuning Guide.** NVIDIA, 2026. <https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html>  
   Take: Blackwell SM resources and cluster limits.
9. **How to Write a Fast Matrix Multiplication from Scratch with Tensor Cores.** Alex Armbruster, 2024. <https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html>  
   Take: Software pipelining on Ampere in practice.
10. **GPU Glossary: Tensor Memory.** Modal, 2025. <https://modal.com/gpu-glossary/device-hardware/tensor-memory>  
   Take: Concise description of tensor memory.

## 13-cutlass-and-cute: CUTLASS and CuTe layouts

Part 3 — Matrix multiplication.

1. **CuTe Layout Representation and Algebra.** Cris Cecka, 2026. <https://arxiv.org/abs/2603.02298>  
   Take: Formal definitions of coalesce, composition, complement, and inverse.
2. **CuTe Layout Algebra.** NVIDIA, 2026. <https://docs.nvidia.com/cutlass/4.6.0/media/docs/cpp/cute/02_layout_algebra.html>  
   Take: The reference definitions the lab implements in Python.
3. **CuTe quickstart.** NVIDIA, 2026. <https://docs.nvidia.com/cutlass/4.6.0/media/docs/cpp/cute/00_quickstart.html>  
   Take: Shape, stride, Layout, and Tensor basics.
4. **A Note on the Algebra of CuTe Layouts.** Jay Shah, Colfax Research, 2024. <https://research.colfax-intl.com/a-note-on-the-algebra-of-cute-layouts/>  
   Take: A careful derivation of layout composition and complement.
5. **Categorical Foundations for CuTe Layouts.** Jack Carlisle, Jay Shah, Reuben Stern, Paul VanKoughnett, 2026. <https://arxiv.org/abs/2601.05972>  
   Take: Category-theoretic view of the algebra, for readers who want rigor.
6. **CUTLASS 3.x: Orthogonal, Reusable, and Composable Abstractions for GEMM Kernel Design.** NVIDIA, 2023. <https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/>  
   Take: The atom, tiled MMA, collective, kernel, device layering.
7. **CUTLASS repository.** NVIDIA, 2026. <https://github.com/NVIDIA/cutlass>  
   Take: Source, examples, and release notes (4.8.1 in October 2026).
8. **CuTe DSL overview.** NVIDIA, 2026. <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html>  
   Take: The Python front end for CuTe.
9. **CuTe Swizzle.** Lei Mao, 2024. <https://leimao.github.io/blog/CuTe-Swizzle/>  
   Take: Swizzle as a layout function.

## 14-triton: Triton

Part 3 — Matrix multiplication.

1. **Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations.** Philippe Tillet, H. T. Kung, David Cox, 2019. <https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf>  
   Take: The original design: block-level programming model.
2. **Triton matrix multiplication tutorial.** Triton project, 2026. <https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html>  
   Take: Grouped (L2-friendly) program ordering: 90 loads reduced to 54 in the 9x9 example.
3. **Triton persistent matmul tutorial.** Triton project, 2026. <https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html>  
   Take: Tensor descriptors, TMA gating (compute capability 9 or higher), persistent grids.
4. **triton.language.make_tensor_descriptor.** Triton project, 2026. <https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html>  
   Take: Signature and requirements of the descriptor API that replaces block pointers.
5. **Triton release notes.** Triton project, 2026. <https://github.com/triton-lang/triton/releases>  
   Take: 3.7.0 added a deprecation warning for tl.make_block_ptr.
6. **Triton repository.** Triton project (created by Philippe Tillet at OpenAI), 2026. <https://github.com/triton-lang/triton>  
   Take: Source, README, and the Gluon tutorials folder.
7. **Triton-Puzzles.** Sasha Rush, 2024. <https://github.com/srush/Triton-Puzzles>  
   Take: Exercises for building intuition.
8. **Deep Dive on the Hopper TMA Unit for FP8 GEMMs.** PyTorch, 2024. <https://pytorch.org/blog/hopper-tma-unit/>  
   Take: How Triton uses TMA descriptors.
9. **triton.autotune.** Triton project, 2026. <https://triton-lang.org/main/python-api/generated/triton.autotune.html>  
   Take: Autotune configs, keys, and pruning.

## 15-softmax-and-normalization: Softmax and normalization

Part 4 — Kernels for ML.

1. **Online Normalizer Calculation for Softmax.** Maxim Milakov and Natalia Gimelshein, NVIDIA, 2018. <https://arxiv.org/abs/1805.02867>  
   Take: The one-pass running max and sum that FlashAttention builds on.
2. **Fused softmax tutorial.** Triton project, 2026. <https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html>  
   Take: A fused row-wise softmax in Triton and its bandwidth comparison.
3. **Layer normalization tutorial.** Triton project, 2026. <https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html>  
   Take: Forward and backward LayerNorm kernels, including the partial-sum reduction for dW.
4. **Layer Normalization.** Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey Hinton, 2016. <https://arxiv.org/abs/1607.06450>  
   Take: Definition of the operation.
5. **Root Mean Square Layer Normalization.** Biao Zhang and Rico Sennrich, 2019. <https://arxiv.org/abs/1910.07467>  
   Take: RMSNorm.
6. **Algorithms for calculating variance.** Wikipedia, 2026. <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance>  
   Take: Welford's online algorithm and Chan's parallel merge, with citations.
7. **Liger Kernel: Efficient Triton Kernels for LLM Training.** Pin-Lun Hsu and others, 2024. <https://arxiv.org/abs/2410.10989>  
   Take: Production Triton norm kernels; about 20% throughput gain and 60% memory reduction.
8. **Making Deep Learning Go Brrrr From First Principles.** Horace He, 2022. <https://horace.io/brrr_intro.html>  
   Take: Why normalization and softmax are bandwidth-bound.

## 16-flash-attention: FlashAttention

Part 4 — Kernels for ML.

1. **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.** Tri Dao and others, 2022. <https://arxiv.org/abs/2205.14135>  
   Take: Tiling plus online softmax; HBM access analysis; 3x on GPT-2.
2. **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.** Tri Dao, 2023. <https://arxiv.org/abs/2307.08691>  
   Take: Fewer non-matmul FLOPs, sequence-length parallelism; up to 73% of A100 peak.
3. **FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision.** Jay Shah and others, 2024. <https://arxiv.org/abs/2407.08608>  
   Take: Hopper asynchrony; about 740 TFLOP/s FP16 and 1.2 PFLOP/s FP8.
4. **FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling.** Ted Zadouri and others, 2026. <https://arxiv.org/abs/2603.05451>  
   Take: Blackwell: up to 1613 TFLOP/s BF16, software-emulated exp, 2-CTA MMA.
5. **FlashAttention-4 (blog).** Tri Dao, 2026. <https://tridao.me/blog/2026/flash4/>  
   Take: Per-SM resource arithmetic: 8192 tensor ops, 16 exp ops, 128 shared-memory bytes per cycle.
6. **Online Normalizer Calculation for Softmax.** Maxim Milakov and Natalia Gimelshein, 2018. <https://arxiv.org/abs/1805.02867>  
   Take: The recurrence.
7. **Self-attention Does Not Need O(n^2) Memory.** Markus Rabe and Charles Staats, 2021. <https://arxiv.org/abs/2112.05682>  
   Take: An independent derivation of chunked attention.
8. **Fused attention tutorial.** Triton project, 2026. <https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html>  
   Take: Forward and backward FlashAttention-2 in Triton.
9. **FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention.** PyTorch, 2024. <https://pytorch.org/blog/flexattention/>  
   Take: Masking and score modifications compiled into the kernel.
10. **flash-attention repository.** Dao-AILab, 2026. <https://github.com/Dao-AILab/flash-attention>  
   Take: Reference implementations for FA2, FA3, and FA4.

## 17-fusion-and-compilers: Fusion, compilers, and launch overhead

Part 4 — Kernels for ML.

1. **PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation.** Jason Ansel and others, 2024. <https://docs.pytorch.org/assets/pytorch2-2.pdf>  
   Take: TorchDynamo and TorchInductor design; measured speedups across benchmarks.
2. **torch.compile documentation.** PyTorch, 2026. <https://docs.pytorch.org/docs/stable/torch.compiler.html>  
   Take: Modes, graph breaks, recompilation, and debugging.
3. **Making Deep Learning Go Brrrr From First Principles.** Horace He, 2022. <https://horace.io/brrr_intro.html>  
   Take: Why fusion removes memory traffic and overhead.
4. **TorchInductor: a PyTorch-native Compiler with Define-by-Run IR and Symbolic Shapes.** Jason Ansel and others, 2022. <https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747>  
   Take: Inductor's lowering and Triton codegen.
5. **Accelerating PyTorch with CUDA Graphs.** PyTorch, 2021. <https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/>  
   Take: Capture, replay, static input buffers.
6. **Getting Started with CUDA Graphs.** NVIDIA, 2019. <https://developer.nvidia.com/blog/cuda-graphs/>  
   Take: Measured launch overhead on a V100: 9.6 us per kernel with sync, 3.4 us per kernel in a graph.
7. **Python custom operators tutorial.** PyTorch, 2026. <https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html>  
   Take: torch.library.custom_op, register_fake, register_autograd.
8. **torch.library.** PyTorch, 2026. <https://docs.pytorch.org/docs/2.14/library.html>  
   Take: custom_op, triton_op, wrap_triton, and opcheck signatures.
9. **PyTorch 2.14 release notes.** PyTorch, 2026. <https://github.com/pytorch/pytorch/releases/tag/v2.14.0>  
   Take: NVGEMM (CuTeDSL kernels in Inductor), Triton 3.8.0, CUDA graph changes.
10. **CUDA graphs (CUDA Programming Guide).** NVIDIA, 2026. <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html>  
   Take: Graph model, capture rules.

## 18-quantized-kernels: Quantized kernels

Part 4 — Kernels for ML.

1. **MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models.** Elias Frantar, Roberto Castro, Jiale Chen, Torsten Hoefler, Dan Alistarh, 2024. <https://arxiv.org/abs/2408.11743>  
   Take: W4A16 kernel design; close to 4x speedup at batch 16-32; up to 2.8x end to end in vLLM.
2. **Marlin repository.** IST-DASLab, 2024. <https://github.com/IST-DASLab/marlin>  
   Take: Kernel source for reading the layout and pipeline.
3. **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.** Elias Frantar and others, 2022. <https://arxiv.org/abs/2210.17323>  
   Take: Where group-wise int4 weights come from.
4. **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.** Ji Lin and others, 2023. <https://arxiv.org/abs/2306.00978>  
   Take: Salient-channel scaling.
5. **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.** Tim Dettmers and others, 2022. <https://arxiv.org/abs/2208.07339>  
   Take: Outlier features in activations.
6. **SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.** Guangxuan Xiao and others, 2022. <https://arxiv.org/abs/2211.10438>  
   Take: Moving activation outliers into weights for W8A8.
7. **The Case for 4-bit Precision: k-bit Inference Scaling Laws.** Tim Dettmers and Luke Zettlemoyer, 2022. <https://arxiv.org/abs/2212.09720>  
   Take: Why 4-bit weights are the usual operating point.
8. **Microscaling Data Formats for Deep Learning.** Bita Darvish Rouhani and others, 2023. <https://arxiv.org/abs/2310.10537>  
   Take: Block-scaled formats.
9. **Introducing NVFP4 for Efficient and Accurate Low-Precision Inference.** NVIDIA, 2025. <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>  
   Take: NVFP4 block layout and hardware.
10. **torchao.** PyTorch, 2026. <https://github.com/pytorch/ao>  
   Take: Reference quantization APIs and Triton kernels.

## 19-training-kernels: Training kernels

Part 4 — Kernels for ML.

1. **Cut Your Losses in Large-Vocabulary Language Models.** Erik Wijmans and others, Apple, 2024. <https://arxiv.org/abs/2411.09009>  
   Take: Fused linear cross-entropy: 24 GB of loss memory cut to 1 MB on Gemma 2 (2B).
2. **Liger Kernel: Efficient Triton Kernels for LLM Training.** Pin-Lun Hsu and others, 2024. <https://arxiv.org/abs/2410.10989>  
   Take: Fused linear cross-entropy, RMSNorm, RoPE kernels; 20% throughput and 60% memory.
3. **Liger-Kernel repository.** LinkedIn, 2026. <https://github.com/linkedin/Liger-Kernel>  
   Take: Reference Triton kernels.
4. **torch.nn.functional.linear_cross_entropy.** PyTorch, 2026. <https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.linear_cross_entropy.html>  
   Take: The built-in chunked fused loss and its limits.
5. **torch.nn.LinearCrossEntropyOptions.** PyTorch, 2026. <https://docs.pytorch.org/docs/2.14/generated/torch.nn.LinearCrossEntropyOptions.html>  
   Take: Chunking and accumulation options.
6. **Extending PyTorch: custom autograd Functions.** PyTorch, 2026. <https://docs.pytorch.org/docs/stable/notes/extending.html>  
   Take: Writing forward and backward, saving tensors, and gradient checks.
7. **Training Deep Nets with Sublinear Memory Cost.** Tianqi Chen and others, 2016. <https://arxiv.org/abs/1604.06174>  
   Take: Recomputation as a memory-for-compute trade, the idea chunked loss reuses.
8. **Mixed Precision Training.** Paulius Micikevicius and others, 2017. <https://arxiv.org/abs/1710.03740>  
   Take: fp32 accumulation and master weights.
9. **Layer normalization tutorial.** Triton project, 2026. <https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html>  
   Take: Backward kernels with partial reductions.
10. **TransformerEngine.** NVIDIA, 2026. <https://github.com/NVIDIA/TransformerEngine>  
   Take: FP8 training layers and fused kernels.

## 20-moe-and-sparse-kernels: Mixture-of-experts and sparse kernels

Part 4 — Kernels for ML.

1. **MegaBlocks: Efficient Sparse Training with Mixture-of-Experts.** Trevor Gale and others, 2022. <https://arxiv.org/abs/2211.15841>  
   Take: Block-sparse formulation of MoE and why token dropping is unnecessary.
2. **Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.** Noam Shazeer and others, 2017. <https://arxiv.org/abs/1701.06538>  
   Take: Top-k gating.
3. **Switch Transformers.** William Fedus, Barret Zoph, Noam Shazeer, 2021. <https://arxiv.org/abs/2101.03961>  
   Take: Capacity factor and load-balancing loss.
4. **Mixtral of Experts.** Albert Jiang and others, 2024. <https://arxiv.org/abs/2401.04088>  
   Take: A widely used open MoE architecture.
5. **DeepSeek-V3 Technical Report.** DeepSeek-AI, 2024. <https://arxiv.org/abs/2412.19437>  
   Take: Fine-grained experts and expert-parallel communication.
6. **Accelerating Sparse Deep Neural Networks.** Asit Mishra and others, NVIDIA, 2021. <https://arxiv.org/abs/2104.08378>  
   Take: The 2:4 format, metadata, and measured sparse GEMM speedups.
7. **Accelerating Neural Network Training with Semi-Structured (2:4) Sparsity.** PyTorch, 2024. <https://pytorch.org/blog/accelerating-neural-network-training/>  
   Take: PyTorch's 2:4 API; 1.3x on ViT-L MLP layers on A100.
8. **Grouped GEMM tutorial.** Triton project, 2026. <https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html>  
   Take: A Triton grouped GEMM with a persistent tile scheduler.
9. **cuSPARSELt.** NVIDIA, 2026. <https://docs.nvidia.com/cuda/cusparselt/>  
   Take: The library behind sparse tensor-core GEMM.
10. **vLLM fused MoE kernel.** vLLM project, 2026. <https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/fused_moe>  
   Take: Production Triton MoE kernels with token permutation.

## 21-collective-communication: Collective communication

Part 5 — Many GPUs.

1. **Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations.** Pitch Patarasuk and Xin Yuan, 2009. <https://www.cs.fsu.edu/~xyuan/paper/09jpdc.pdf>  
   Take: Proof that ring all-reduce is bandwidth-optimal: 2(p-1)/p of the data per process.
2. **Optimization of Collective Communication Operations in MPICH.** Rajeev Thakur, Rolf Rabenseifner, William Gropp, 2005. <https://www.cs.umd.edu/class/fall2019/cmsc714/readings/Thakur-MPICH.pdf>  
   Take: Alpha-beta cost model for each collective algorithm.
3. **Bringing HPC Techniques to Deep Learning.** Andrew Gibiansky, Baidu, 2017. <https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/>  
   Take: Step-by-step ring all-reduce explanation.
4. **NCCL tests: performance notes.** NVIDIA, 2026. <https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md>  
   Take: Definitions of algbw and busbw and the correction factor for each collective.
5. **NCCL documentation.** NVIDIA, 2026. <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html>  
   Take: API, algorithms, and environment variables.
6. **Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms.** Zhiyi Hu and others, 2025. <https://arxiv.org/abs/2507.04786>  
   Take: Ring, tree, and protocol internals (Simple, LL, LL128).
7. **Massively Scale Your Deep Learning Training with NCCL 2.4.** NVIDIA, 2019. <https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/>  
   Take: Tree algorithm and why latency matters at scale.
8. **Distributed communication package (torch.distributed).** PyTorch, 2026. <https://docs.pytorch.org/docs/stable/distributed.html>  
   Take: send, recv, isend, irecv, and the collectives used in the lab.
9. **NVIDIA NVLink and NVLink Switch.** NVIDIA, 2026. <https://www.nvidia.com/en-us/data-center/nvlink/>  
   Take: NVLink 4 and 5 totals per GPU; NVSwitch aggregate figures.
10. **How to Scale Your Model: Collectives.** Jacob Austin and others, 2025. <https://jax-ml.github.io/scaling-book/>  
   Take: All-reduce, all-gather, and reduce-scatter costs on rings and tori.

## 22-overlap-and-parallelism: Overlap and parallelism

Part 5 — Many GPUs.

1. **How to Overlap Data Transfers in CUDA C/C++.** Mark Harris, NVIDIA, 2012. <https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/>  
   Take: Streams, pinned memory, and copy-compute overlap.
2. **CUDA semantics.** PyTorch, 2026. <https://docs.pytorch.org/docs/stable/notes/cuda.html>  
   Take: Streams, events, and pinned memory in PyTorch.
3. **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.** Mohammad Shoeybi and others, 2019. <https://arxiv.org/abs/1909.08053>  
   Take: Tensor parallelism: column and row splits and where the all-reduces go.
4. **Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM.** Deepak Narayanan and others, 2021. <https://arxiv.org/abs/2104.04473>  
   Take: Combining tensor, pipeline, and data parallelism.
5. **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.** Samyam Rajbhandari and others, 2019. <https://arxiv.org/abs/1910.02054>  
   Take: Sharding optimizer state, gradients, and parameters.
6. **PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel.** Yanli Zhao and others, 2023. <https://arxiv.org/abs/2304.11277>  
   Take: FSDP overlap of all-gather with compute.
7. **GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism.** Yanping Huang and others, 2018. <https://arxiv.org/abs/1811.06965>  
   Take: Micro-batching and the pipeline bubble.
8. **PyTorch Distributed: Experiences on Accelerating Data Parallel Training.** Shen Li and others, 2020. <https://arxiv.org/abs/2006.15704>  
   Take: Gradient bucketing and overlap with backward.
9. **GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.** Dmitry Lepikhin and others, 2020. <https://arxiv.org/abs/2006.16668>  
   Take: Expert parallelism and all-to-all.
10. **The Ultra-Scale Playbook: Training LLMs on GPU Clusters.** Hugging Face, 2025. <https://huggingface.co/spaces/nanotron/ultrascale-playbook>  
   Take: A readable survey of DP, TP, PP, CP, and EP with measured throughput.

## 23-beyond-cuda: Beyond CUDA

Part 6 — Beyond.

1. **AMD CDNA 3 Architecture whitepaper.** AMD, 2023. <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf>  
   Take: MI300X structure: 304 CUs, 64-wide wavefronts, 64 KB LDS, 5.3 TB/s HBM3.
2. **AMD CDNA 4 Architecture whitepaper.** AMD, 2025. <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf>  
   Take: MI355X: 256 CUs, 160 KB LDS, MXFP4 and MXFP6 support.
3. **HIP documentation: hardware features.** AMD, 2026. <https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hardware_features.html>  
   Take: Per-CU limits; how HIP differs from CUDA.
4. **TPU architecture (Cloud TPU docs).** Google Cloud, 2026. <https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm>  
   Take: MXU size (256x256 on v6e and later), TensorCores, SparseCores.
5. **Pallas: a JAX kernel language.** JAX project, 2026. <https://docs.jax.dev/en/latest/pallas/index.html>  
   Take: Writing custom kernels for TPU and GPU from JAX.
6. **XLA architecture.** OpenXLA, 2026. <https://openxla.org/xla/architecture>  
   Take: Compiler-driven fusion as the TPU analogue of Triton plus Inductor.
7. **In-Datacenter Performance Analysis of a Tensor Processing Unit.** Norman Jouppi and others, 2017. <https://arxiv.org/abs/1704.04760>  
   Take: The systolic-array design and roofline analysis.
8. **ThunderKittens: Simple, Fast, and Adorable AI Kernels.** Benjamin Spector and others, 2024. <https://arxiv.org/abs/2410.20399>  
   Take: Tile-based CUDA DSL matching cuBLAS GEMM and FlashAttention-3.
9. **TileLang.** tile-ai, 2026. <https://github.com/tile-ai/tilelang>  
   Take: Tile DSL with CUDA, HIP, and Metal backends.
10. **HipKittens: Fast and Furious AMD Kernels.** HazyResearch, 2025. <https://arxiv.org/abs/2511.08083>  
   Take: Tile primitives for CDNA3 and CDNA4.
11. **MLX.** Apple, 2026. <https://github.com/ml-explore/mlx>  
   Take: Array framework for Apple-silicon GPUs with unified memory.
12. **Metal Shading Language Specification.** Apple, 2026. <https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf>  
   Take: simdgroup_matrix and threadgroup memory.

## 24-cute-dsl-gemm: A CuTe DSL GEMM

Part 3 — Matrix multiplication. Numbered 24, after chapter 23, because chapters 00 through 23 are not renumbered.

1. **CuTe DSL overview.** NVIDIA, 2026. <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html>  
   Take: The Python programming model: tensors, layouts, atoms, and tiled operations.
2. **Warp-level MMA instructions.** NVIDIA CUTLASS 4.5.2. <https://docs.nvidia.com/cutlass/4.5.2/media/docs/pythonDSL/mma_docs/wmma_programming.html>  
   Take: `MmaF16BF16Op`, `make_tiled_mma`, `cute.copy`, and `cute.gemm` for `mma.sync` on Ampere.
3. **CUTLASS repository.** NVIDIA. <https://github.com/NVIDIA/cutlass>  
   Take: The Ampere CuTe DSL examples under `examples/python/CuTeDSL`.
4. **GPU MODE, lecture 15: CUTLASS.** Eric Auld. <https://github.com/gpu-mode/lectures#lecture-15-cutlass>  
   Take: The lecture list's CUTLASS session.
5. **GPU MODE, lecture 57: CuTE.** Cris Cecka. <https://github.com/gpu-mode/lectures#lecture-57-cute>  
   Take: The layout algebra, from the published lecture titles.
6. **GPU MODE, lecture 86: Introduction to CuTeDSL (for NVIDIA competition).** Vicki Wang. <https://github.com/gpu-mode/lectures/tree/main/lecture_086>  
   Take: The DSL session. The slide PDF is `lecture_086/cute_dsl_introduce.pdf`.
7. **GPU MODE, lecture 103: Fundamentals of CuTe Layout Algebra and Category-theoretic Interpretation.** Jack Carlisle and Jay Shah. <https://github.com/gpu-mode/lectures#lecture-103-fundamentals-of-cute-layout-algebra-and-category-theoretic-interpretation>  
   Take: The algebra chapter 13 implements, as a lecture.
8. **GPU MODE, lecture 114: PyCuTe.** Cris Cecka. <https://github.com/gpu-mode/lectures#lecture-114-pycute>  
   Take: PyCuTe is Cecka's library at <https://github.com/NVlabs/CuTe>. The kernel in this chapter uses NVIDIA's CuTe DSL, a different package.

## 25-ddp-and-zero: Data parallel and ZeRO-1

Part 5 — Many GPUs. Numbered 25, after chapter 24, because chapters 00 through 24 are not renumbered.

1. **Distributed data parallel.** PyTorch 2.14. <https://docs.pytorch.org/docs/2.14/notes/ddp.html>  
   Take: Reverse-order buckets, the asynchronous all-reduce, and launching in bucket-index order.
2. **`DistributedDataParallel`.** PyTorch 2.14. <https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html>  
   Take: `bucket_cap_mb` defaults to 25 MiB.
3. **`register_post_accumulate_grad_hook`.** PyTorch 2.14. <https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.register_post_accumulate_grad_hook.html>  
   Take: The hook runs after `.grad` is written, and only on a leaf tensor.
4. **CS336 assignment 2, systems and parallelism.** Stanford, 2026. <https://github.com/stanford-cs336/assignment2-systems/blob/main/cs336_assignment2_systems.pdf>  
   Take: Overlapped DDP from that hook, with `async_op=True` and `finish_gradient_synchronization`, and a sharded optimizer.
5. **ZeRO: Memory optimizations toward training trillion parameter models.** Samyam Rajbhandari and others, 2019. <https://arxiv.org/abs/1910.02054>  
   Take: $K = 12$, the $P_{os}$ memory $4\Psi + K\Psi/N$, and the $2\Psi$ volume of reduce-scatter plus all-gather.
6. **nccl-tests performance notes.** NVIDIA. <https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md>  
   Take: algbw, busbw, and the $2(n-1)/n$ factor for all-reduce.
7. **GPU MODE, lecture 17: GPU collective communication (NCCL).** Dan Johnson. <https://github.com/gpu-mode/lectures/tree/main/lecture_017>  
   Take: The lecture index's NCCL session. Chapter 21 is the cost model this chapter applies.
8. **GPU MODE, lecture 67: NCCL and NVSHMEM.** Jeff Hammond. <https://github.com/gpu-mode/lectures#lecture-67-nccl--nvshmem>  
   Take: The lecture index lists the session. This chapter stops at the collectives NCCL runs.
9. **GPU MODE, lecture 39: TorchTitan.** Mark Saroufim and Tianyu Liu. <https://github.com/gpu-mode/lectures#lecture-39-torchtitan>  
   Take: The lecture index lists TorchTitan. The folder `lecture_039` was not present when this chapter was written.

## 26-hopper-gemm: Hopper GEMM

Part 3 — Matrix multiplication. Numbered 26, after chapter 25, because chapters 00 through 25 are not renumbered. Prerequisites are chapter 12 and chapter 24.

1. **Tensor map object management.** NVIDIA, CUDA Driver API. <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html>  
   Take: `cuTensorMapEncodeTiled`. Strides are rank minus 1, in bytes, and a multiple of 16. Each box dimension is at most 256. Out-of-bounds fill none writes zeros. The 128-byte swizzle is the fourth driver enumerator.
2. **PTX ISA.** NVIDIA. <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>  
   Take: Section 5.5.7, the 128-byte swizzle table with 16-byte atomicity. The mbarrier phase completes when the arrival count and the transaction-byte count are both zero. `cp.async.bulk.tensor` completes through that barrier.
3. **`mma_sm90_desc.hpp`.** NVIDIA, CUTLASS 3.9.2. <https://github.com/NVIDIA/cutlass/blob/v3.9.2/include/cute/arch/mma_sm90_desc.hpp>  
   Take: The 64-bit descriptor. Layout type 0, 1, 2, 3 is none, 128B, 64B, 32B. Address fields drop the low 4 bits.
4. **CuTe DSL primitives.** NVIDIA. <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/primitives.html>  
   Take: The same descriptor bit layout, and the rule that a start address advances in multiples of 16 bytes.
5. **CUTLASS tutorial: wgmma on Hopper.** Colfax Research. <https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/>  
   Take: B is a shared-memory descriptor, the accumulator is registers, and the K-major no-swizzle atom uses a leading offset of 128 bytes and a stride of 256 bytes.
6. **GPU MODE, lecture 36: CUTLASS and FlashAttention-3.** Jay Shah. <https://github.com/gpu-mode/lectures/tree/main/lecture_036>  
   Take: The lecture index titles this session CUTLASS and FlashAttention-3. The slides cover WGMMA, TMA, and warp specialization.
7. **GPU MODE, lecture 23: Tensor Cores.** Vijay Thakkar and Pradeep Ramani. <https://github.com/gpu-mode/lectures#lecture-23-tensor-cores>  
   Take: The lecture index titles this session Tensor Cores. It is the series' tensor-core lecture. This chapter does not use it as a TMA source.
8. **`mma_traits_sm90_gmma.hpp`.** NVIDIA, CUTLASS 3.9.2. <https://github.com/NVIDIA/cutlass/blob/v3.9.2/include/cute/atom/mma_traits_sm90_gmma.hpp>  
   Take: `CLayout_64xN` strides `(128,1,16)` and `(64,8,512)`. The 64 by 8 fragment map in the lab is the column-major reading of those strides. The HTML fetch of this file dropped template arguments, so the thread shape `(4,8,4)` is the shape that makes the strides a bijection, not a line copied verbatim.

## 27-debugging-gpu-kernels: Debugging GPU kernels

Part 1 — The machine. Numbered 27, after chapter 26, because chapters 00 through 26 are not renumbered. Prerequisites are chapters 2, 5, 6, 8, and 14.

1. **Compute Sanitizer documentation.** NVIDIA. <https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html> (key CS-DOCS)  
   Take: The four tools and what each detects, the report headers, the ERROR SUMMARY line, and `--tool`, `--error-exitcode`, `--racecheck-report`, `--initcheck-address-space`, and `--show-backtrace`. The page did not state the CUDA version that added shared-memory initcheck.
2. **CUDA-GDB documentation.** NVIDIA. <https://docs.nvidia.com/cuda/cuda-gdb/index.html> (key CUDA-GDB-DOCS)  
   Take: `-g -G`, device breakpoints, the `cuda` focus commands, `set cuda break_on_launch`, `CUDA_DEVICE_WAITS_ON_EXCEPTION`, no device watchpoints, and the removal of cuda-memcheck in CUDA 12.0. Only the first 100,000 characters of the page were read.
3. **CUDA programming guide, C++ language extensions.** NVIDIA. <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html> (key CUDA-PG-EXT)  
   Take: `__syncthreads` in conditional code, `__syncwarp`, and `assert`. The page's printf section was not read, so the chapter gives no printf buffer size.
4. **CUDA programming guide, environment variables.** NVIDIA. <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html> (key CUDA-PG-ENV)  
   Take: `CUDA_LAUNCH_BLOCKING`.
5. **Triton debugging guide.** Triton project. <https://triton-lang.org/main/programming-guide/chapter-3/debugging.html> (key TRITON-DEBUG)  
   Take: `TRITON_INTERPRET`, `device_print`, `static_print`, `device_assert` with `TRITON_DEBUG`, and the limits on `bfloat16` and indirect loads.
6. **Using CUDA warp-level primitives.** Yuan Lin and Vinod Grover, NVIDIA, 2018. <https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/>  
   Take: Chapter 2 already cites it for independent thread scheduling. Chapter 27 links it for why lockstep code needs `__syncwarp`. Not re-fetched for this chapter.
7. **GPU MODE lecture index.** <https://github.com/gpu-mode/lectures> (key GPUMODE-IDX)  
   Take: Lecture 1 is Profiling and Integrating CUDA kernels in PyTorch, and lecture 16 is On Hands profiling. The index lists no debugging session, so the chapter does not cite a lecture for the sanitizer. The research report also named CS149 assignment 3 for this topic, and that page was not fetched.

## 29-gemm-scheduling: GEMM scheduling

Part 3 — Matrix multiplication. Numbered 29 because chapters 00 through 26 are not renumbered, and chapters 27 and 28 are numbered by other work. Prerequisites are chapters 04, 10, and 14.

1. **Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU.** Osama, Merrill, Cecka, Garland, and Owens. <https://arxiv.org/abs/2301.03598>  
   Take: The even division of inner-loop iterations among processing elements, the `ceil(total / g)` range, the owner of a tile as the CTA with its k = 0 iteration, the flag-and-workspace fix-up, the two hybrids, and a grid size of at most the SM count. The abstract's speedups of up to 14x and 6.7x name no GPU model in the abstract, so the chapter does not quote them.
2. **Triton persistent matmul tutorial.** <https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html>  
   Take: A grid of about one program per SM, a tile loop in steps of the SM count, grouped ordering, epilogue subtiling, TMA, and warp specialization.
3. **Triton matrix multiplication tutorial.** <https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html>  
   Take: The grouped-ordering lines, `pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)`.
4. **CUTLASS, Efficient GEMM in CUDA.** NVIDIA. <https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md>  
   Take: Split-K as a partitioned-K GEMM plus a batched reduction with a caller-allocated workspace, sliced-K, and the persistent cooperative and ping-pong kernels. The page has no stream-K or wave-quantization text.
5. **CUTLASS, GEMM API 3.x.** NVIDIA. <https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/gemm_api_3x.md>  
   Take: The tile scheduler parameter of a kernel, the work-tile loop of a persistent kernel, and the collective epilogue as the element-wise operations on the output. The page has no stream-K or split-K decomposition text.
6. **Matrix multiplication background.** NVIDIA Deep Learning Performance guide. <https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html>  
   Take: The 108-tile and 117-tile example, tile quantization, and wave quantization. Chapter 4 already cites it.
## 28-reading-profiler-reports: Reading profiler reports

Part 1 — The machine. Numbered 28, after chapter 26, because chapters 00 through 26 are not renumbered. Prerequisites are chapter 4 and chapter 5. The lab's recorded reports are synthesized.

1. **Nsight Compute Profiling Guide.** NVIDIA. <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>  
   Take: The stall-reason metric family `smsp__pcsamp_warps_issue_stalled_<reason>` and its reasons, theoretical occupancy `sm__maximum_warps_per_active_cycle_pct`, `launch__occupancy_limit_<resource>`, `launch__waves_per_multiprocessor`, and the memory-table quantities for sectors, wavefronts, and bank conflicts. The exact sector, FLOP, and bank-conflict counter strings in the lab were not confirmed on the page.
2. **Nsight Compute CLI.** NVIDIA. <https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html>  
   Take: `--csv` implies `--print-units base`, `--metrics`, `-k`, `-s`, `-c`, and the auto-scaling factors of 1000.
3. **Nsight Systems User Guide.** NVIDIA. <https://docs.nvidia.com/nsight-systems/UserGuide/index.html>  
   Take: `nsys profile`, `nsys stats`, the default reports `cuda_api_sum` and `cuda_gpu_kern_sum`, and the `--format` list. The `cuda_gpu_trace` column headers were not on the pages read.
4. **GPU MODE, lecture 1: Profiling and Integrating CUDA kernels in PyTorch.** Mark Saroufim. <https://github.com/gpu-mode/lectures>  
   Take: The lecture index lists the title. This chapter did not read the lecture itself.
5. **GPU MODE, lecture 8: CUDA Performance Checklist.** Mark Saroufim. <https://github.com/gpu-mode/lectures>  
   Take: The lecture index lists the title. This chapter did not read the lecture itself.
6. **GPU MODE, lecture 16: On Hands profiling.** Taylor Robbie. <https://github.com/gpu-mode/lectures>  
   Take: The lecture index lists the title. This chapter did not read the lecture itself.
7. **Chapter 4 and chapter 5 of this course.** Roofline, wave efficiency, bank model, and the profiling funnel that this chapter reads from counters.
## 30-sort-top-k-and-sampling: Radix sort, top-k, and sampling

Part 2 — CUDA patterns. Numbered 30, because chapters 00 through 26 are not renumbered. Prerequisites are chapter 6 and chapter 7. Sampling inside an inference engine belongs to learn-inference.

1. **CUB `DeviceRadixSort`.** NVIDIA. <https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceRadixSort.html>  
   Take: Stable. Sign-bit flip for positive floats and full inversion for negative floats. `begin_bit` and `end_bit`. Zeros are equivalent, and NaNs sort by bit pattern.
2. **CUB `DeviceTopK`.** NVIDIA. <https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceTopK.html>  
   Take: Based on AIR TopK. The tie contract depends on the requested mode. Only the first 100,000 characters of the page were read.
3. **Onesweep: A Faster Least Significant Digit Radix Sort for GPUs.** Adinets and Merrill. <https://arxiv.org/abs/2206.01784>  
   Take: Single-pass prefix sum lowers memory traffic per digit. 29.4 giga-keys per second on an A100 for 256M 32-bit keys.
4. **Sorting-free GPU kernels for LLM sampling.** FlashInfer, 10 March 2025. <https://flashinfer.ai/2025/03/10/sampling.html>  
   Take: Rejection sampling with a pivot, a dual-pivot variant, and the monotonicity caveat for float prefix sums. More than 50% less sampling time on one H100 under vLLM.
5. **cuRAND device API overview.** NVIDIA. <https://docs.nvidia.com/cuda/curand/device-api-overview.html>  
   Take: `Philox_4x32_10`, seed, subsequence, offset, and the period.
6. **`triton.language.rand`.** Triton. <https://triton-lang.org/main/python-api/generated/triton.language.rand.html>  
   Take: A scalar seed and a block of offsets give uniform float32 values.
7. **`torch.sort` and `torch.topk`.** PyTorch 2.14. <https://docs.pytorch.org/docs/2.14/generated/torch.sort.html>, <https://docs.pytorch.org/docs/2.14/generated/torch.topk.html>  
   Take: `stable=True` preserves the order of equal elements. `topk` tie indices are not guaranteed.
8. **Reproducibility.** PyTorch 2.14. <https://docs.pytorch.org/docs/2.14/notes/randomness.html>  
   Take: A seed does not guarantee identical CPU and GPU results.
9. **Parallel Random Numbers: As Easy as 1, 2, 3.** Salmon, Moraes, Dror, and Shaw, SC'11. <https://www.thesalmons.org/john/random123/papers/random123sc11.pdf>  
   Take: Cited as the origin of Philox and Threefry. The PDF could not be read, so the chapter takes no claim from it.
10. **GPU MODE lecture index.** <https://github.com/gpu-mode/lectures>  
    Take: Lecture 19 is Data Processing on GPUs (Devavret Makkar). Lectures 20 and 21 are Scan Algorithm and Scan Algorithm Part 2 (Izzat El Haj). Lecture 24 is Scan at the Speed of Light (Jake Hemstad and Georgii Evtushenko). Titles only; the slides were not read.
11. **Programming Massively Parallel Processors**, fourth edition. Hwu, Kirk, and Hajj. Take: The merge and sorting chapters. The table of contents could not be fetched, so no chapter numbers are cited.

## 31-convolution-and-stencils: Convolution and stencils

Part 2 — CUDA patterns. Numbered 31, after chapter 27. Prerequisites are chapters 3, 8, 10, and 14.

1. **CUDA C++ Best Practices Guide.** NVIDIA. <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html> (key CUDA-BPG)  
   Take: The constant memory section: 64 KB, cached, as fast as a register when a warp reads one address, serialized when the addresses differ. The section's position in the page was read from characters 100,000 to 200,000.
2. **Convolutional layers user's guide.** NVIDIA Deep Learning Performance. <https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html> (key CUDNN-CONV)  
   Take: Implicit GEMM, the forward-pass GEMM dimensions, NHWC against NCHW, and the channel alignment.
3. **Implicit GEMM convolution.** NVIDIA, CUTLASS 3.9.2. <https://github.com/NVIDIA/cutlass/blob/v3.9.2/media/docs/cpp/implicit_gemm_convolution.md> (key CUTLASS-IGEMM)  
   Take: The mapping of M, N, K to the convolution, the index decomposition, the activation tile iterators, and the precomputed pointer deltas. The fetch returned a summary, so the reference loop's R - r - 1 form is stated with that caveat.
4. **`torch.nn.Conv2d`.** PyTorch 2.15. <https://docs.pytorch.org/docs/2.15/generated/torch.nn.Conv2d.html> (key PT-CONV2D)  
   Take: The output-size formula with dilation, and the cross-correlation statement. The stable URL redirects to this one.
5. **cuDNN: Efficient Primitives for Deep Learning.** Chetlur, Woolley, Vandermersch, Cohen, Tran, Catanzaro, Shelhamer, 2014. <https://arxiv.org/abs/1410.0759>  
   Take: The library behind PyTorch's convolutions. The abstract does not mention matrix-multiplication lowering, so the chapter does not attribute that to it.
6. **Stanford CS149, Fall 2024.** <https://gfxcourses.stanford.edu/cs149/fall24> (key CS149-F24)  
   Take: The lecture titled "Efficiently Evaluating DNNs (Software Solutions)". The page does not number lectures, so no number is cited. Its assignment 4 title does not mention convolution.
7. **GPU MODE lecture index.** <https://github.com/gpu-mode/lectures> (key GPUMODE-IDX)  
   Take: No lecture title mentions convolution, stencils, or implicit GEMM, so the chapter cites no GPU MODE lecture.
8. **Programming Massively Parallel Processors, 4th edition.** Hwu, Kirk, El Hajj, 2022.  
   Take: Chapters 7, 8, and 16, from the research plan. Not re-fetched or re-checked against the book. ECE 408 was named in the plan, and its site could not be reached.
9. **The ml course.** <https://ml.ifkash.dev/lessons/038-convolution-and-receptive-field> and <https://ml.ifkash.dev/lessons/039-im2col-and-convolution-backward>  
   Take: The layer arithmetic, im2col in NumPy, and the backward pass, which this chapter links to and does not repeat. The lesson slugs come from the ml repository's content directory.

## 32-sparse-and-irregular-work: Sparse and irregular work

Part 2 — CUDA patterns. Numbered 32 because other chapters fill 28 to 31. Prerequisites are chapters 3, 6, and 7.

1. **Efficient Sparse Matrix-Vector Multiplication on CUDA.** Nathan Bell and Michael Garland, NVIDIA technical report NVR-2008-004. <https://www.nvidia.com/docs/IO/66889/nvr-2008-004.pdf> (key BG08)  
   Take: The COO, CSR, ELL, and hybrid formats, the scalar and vector CSR kernels, and the hybrid's one-third rule. The PDF was read as extracted text. The report's measurements are on a GeForce GTX 280, and the chapter quotes none of them.
2. **Searching for a Parent Instead of Fighting Over Children: A Fast Breadth-First Search Implementation for Graph500.** Scott Beamer, Krste Asanovic, and David Patterson, UC Berkeley EECS-2011-117. <https://www2.eecs.berkeley.edu/Pubs/TechRpts/2011/EECS-2011-117.html> (key BEAMER11)  
   Take: The bottom-up step, no atomics in it, the queue-to-bitmap conversion, and cutoffs on the current and predicted frontier size. The report was read as extracted text and its figure with the cutoff constants was not recovered, so the chapter's alpha and beta are the lab's own forms and values. The SC12 paper on the same method was not fetched.
3. **Merge-based parallel sparse matrix-vector multiplication.** Duane Merrill and Michael Garland, SC16. Not fetched: the NVIDIA research page and the other candidate URLs returned errors. The chapter derives the merge-path split itself and cites the paper by title only.
4. **GPU MODE lecture index.** <https://github.com/gpu-mode/lectures> (key GPUMODE-IDX)  
   Take: Lecture 11 is Sparsity by Jesse Cai. The index lists no SpMV or graph session.
5. **Programming Massively Parallel Processors, 4th edition, chapters 14 and 15.** Not fetched. Cited by chapter title.
6. **CMU 15-418 and Stanford CS149.** The research report named them for work distribution. The syllabus pages returned errors, so the chapter cites no lecture number from either.


## 33-fp8-and-fp4: FP8, MXFP4, and NVFP4

Part 4 — Kernels for ML. Numbered 33, after chapter 27, because chapters 00 through 27 are not renumbered. Prerequisites are chapters 9 and 18.

1. **Microscaling Data Formats for Deep Learning.** Rouhani and others, 2023. <https://arxiv.org/abs/2310.10537> (key OCP-MX-PAPER)  
   Take: Algorithm 1, the shared-exponent conversion, block size 32, clamping of overflow, and the rounding used in the experiments. Read through the arXiv HTML version. The page doesn't give the E8M0 bias or the `emax` values.
2. **Introducing NVFP4 for Efficient and Accurate Low-Precision Inference.** NVIDIA. <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/> (key NVFP4-BLOG)  
   Take: 16-element blocks, E4M3 scales, the fp32 per-tensor scale, the E2M1 value set, 4.5 bits per value, and the E8M0 against E4M3 MSE example. The post prints no scale formulas.
3. **DeepSeek-V3 Technical Report.** DeepSeek-AI, 2024. <https://arxiv.org/abs/2412.19437> (key DSV3)  
   Take: section 3.3, 1x128 and 128x128 scaling, E4M3 on all tensors, online amax, the 14-bit accumulation limit, and N_C = 128. Only the first 100,000 characters of the HTML version were read, which cover section 3.3.
4. **FP8 Formats for Deep Learning.** Micikevicius and others, 2022. <https://arxiv.org/abs/2209.05433>  
   Take: the recommendation of E4M3 for forward values and E5M2 for gradients. Chapter 9 already cites it, and it was not re-fetched for chapter 33.
5. **OCP 8-bit Floating Point Specification and OCP Microscaling Formats specification.** Open Compute Project. Cited in the further reading and not fetched for this chapter. The format facts were checked against torch 2.11 instead.
6. **learn-inference, quantization chapter.** <https://qwen.ifkash.dev/c/18-quantization>  
   Take: owns weight quantization for serving (GPTQ, AWQ, KV-cache quantization). Linked, not read again.
7. **GPU MODE lecture index.** <https://github.com/gpu-mode/lectures>  
   Take: the research report named lectures 84, 30, and 69 for numerics and quantization. The index was not fetched for this chapter, so the chapter cites no lecture number.

## 34-clusters-and-distributed-shared-memory: Clusters and distributed shared memory

Part 3 — Matrix multiplication. Numbered 34 because chapters 00 through 27 are not renumbered. Prerequisites are chapters 7, 12, and 26. Hardware: H100 only, gated like lab 26. Every URL below was fetched while the chapter was written.

1. **CUDA Programming Guide, programming model.** NVIDIA. <https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html> (key CUDA-PG-MODEL)  
   Take: All blocks of a cluster execute in a single GPC and are scheduled simultaneously. The page says the maximum cluster size is hardware dependent and does not give 8 or 16.
2. **CUDA Programming Guide, C++ language extensions.** NVIDIA. <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html> (key CUDA-PG-EXT)  
   Take: `__cluster_dims__`, the `cudaLaunchKernelEx` example with `cudaLaunchAttributeClusterDimension`, and the multiple-of-cluster-size guidance. Only the first 100,000 characters were read.
3. **CUDA Programming Guide, Cooperative Groups.** NVIDIA. <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html> (key CUDA-PG-CG)  
   Take: `this_cluster()` and the `barrier_arrive` and `barrier_wait` pair. The page does not cover `map_shared_rank`.
4. **CUDA C++ Programming Guide 12.4, archived.** NVIDIA. <https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html> (key CUDA-CPG-124)  
   Take: The portable cluster size of 8, `map_shared_rank`, and the rule that all distributed shared memory operations finish before a block exits. The non-portable 16 and `cudaFuncAttributeNonPortableClusterSizeAllowed` were not in the text range read, so they come from the tuning guide.
5. **CUDA Hopper tuning guide.** NVIDIA. <https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html> (key HOPPER-TUNING)  
   Take: The portable size of 8, the opt-in to 16, the occupancy warning, `cudaOccupancyMaxActiveClusters`, and the DSMEM access advice.
6. **NVIDIA Hopper architecture in depth.** NVIDIA. <https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/> (key HOPPER-BLOG)  
   Take: 8 GPCs and 132 SMs on the H100 SXM5, clusters run within a GPC, and the 7x DSMEM claim. The blog's cluster section does not mention multicast.
7. **Benchmarking and dissecting the Nvidia Hopper GPU architecture.** Luo and others. <https://arxiv.org/abs/2402.13499> (key LUO-HOPPER)  
   Take: The 180-cycle SM-to-SM latency and the throughput at cluster sizes 2 and 4, on an H800 PCIe. Not re-fetched for this chapter; the figures come from docs/FACTS.md section 3.
8. **CUTLASS tutorial: mastering the Tensor Memory Accelerator.** Colfax Research. <https://research.colfax-intl.com/tutorial-hopper-tma/> (key COLFAX-TMA)  
   Take: TMA multicast, the 16-bit mask, the cluster-shape constraint on the grid, and the CuTe API names. The page does not quantify the traffic reduction, so the chapter derives it.
9. **FlashAttention-3.** Shah and others. <https://arxiv.org/abs/2407.08608> (key FA3)  
   Take: The hierarchy table with the 12 TB/s L2 figure and the statement that CTAs of a cluster are co-scheduled on one GPC. A text search of the paper finds no multicast or distributed shared memory.
10. **GPU MODE, lecture 36: CUTLASS and FlashAttention-3.** Jay Shah. <https://github.com/gpu-mode/lectures/tree/main/lecture_036>  
    Take: The lecture index titles this session CUTLASS and FlashAttention-3. Not re-fetched for this chapter, and the slides were not read, so the chapter does not cite them for clusters.
11. **CUTLASS pipeline documentation.** NVIDIA. <https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/pipeline.md>  
    Take: It lists thread block cluster synchronization and query APIs among the CUTLASS abstractions. It does not mention multicast.
