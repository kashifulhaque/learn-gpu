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
