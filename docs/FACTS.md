# learn-gpu fact sheet

This file is the shared ground truth for the 24 `learn-gpu` chapters, labs, and slides. Research date: 2026-10-22.

**Rule for writers.** Every number in a chapter must come from one of three places: arithmetic on a row of this file (show the arithmetic), a cited source in this file, or a measurement your own lab code produces. If a number is not here, do not invent it. Ask the research lead or leave it out.

**Status tags.** Each row carries a tag after its value.

| Tag | Meaning |
|---|---|
| **C** | Confirmed. Read in the cited primary source (vendor document, paper, or official documentation). |
| **D** | Derived. Computed from confirmed rows; the arithmetic is shown. |
| **S** | Secondary. From a reputable but non-primary source (microbenchmark article, vendor reseller page, Wikipedia). Cite it as such. |
| **U** | Unconfirmed. Could not be confirmed from a primary source. Do not state it as fact without a hedge. |

Source keys in square brackets, such as [A100-DS], resolve to URLs in the last section. Some rows (PTX ISA, CUDA Programming Guide, OCP specs, PyTorch source, AMD, TPU, and Apple pages) were read by a delegated research agent that worked from the fetched pages; those rows are tagged C with that provenance in the source key table.

---

## 1. Software versions to target

A 2026 lab image should pin the following. Versions were read on 2026-10-22.

| Component | Pin | Notes | Tag | Source |
|---|---|---|---|---|
| PyTorch | 2.14.1 (released 30 Sep 2026; 2.14.0 on 02 Sep 2026) | Previous minors: 2.13.0 (08 Jul), 2.12.0 (13 May), 2.11.0 (23 Mar). | C | [PT-RELEASES], [PT-PYPI] |
| CUDA runtime in the PyTorch wheel | 13.0.3 is the default pin in the 2.14.1 PyPI metadata; Linux CUDA 13.2 binaries use 13.2.2 | The wheel also pulls cuDNN 9.24.0.43, NCCL 2.30.7, NVSHMEM 3.4.5. | C | [PT-PYPI], [PT-2140] |
| CUDA Toolkit (nvcc for `load_inline`) | 13.x. The newest toolkit release is 13.4 Update 2. Install an `nvcc` whose major version matches the wheel (13). | A100 is sm_80, which CUDA 13 still builds. CUDA 13.x minor-version compatibility needs driver 580 or later. | C (release notes); U (that matching minors is required) | [CUDA-RN] |
| Triton | 3.8.0 (28 Aug 2026), bundled with PyTorch 2.14 | Releases: 3.7.1 (18 Jun), 3.7.0 (07 May), 3.6.0 (21 Jan), 3.5.0 (21 Oct 2025). Requires Python 3.10 to 3.14. | C | [TRITON-REL], [TRITON-PYPI], [PT-2140] |
| Python | 3.12 or 3.13 | PyTorch 2.14 publishes 3.15 wheels, but `torch.compile` does not support Python 3.15 in this release. | C | [PT-2140] |
| CUTLASS and CuTe DSL | CUTLASS 4.8.1 (README dated Oct. 2026) | The CuTe DSL is the Python front end; PyTorch 2.14 Inductor uses it ("NVGEMM"). | C | [CUTLASS-GH], [PT-2140] |
| NCCL | 2.30.7 inside the PyTorch wheel; the NCCL documentation is for 2.32.3 | Do not quote 2.32 features unless the lab image installs it. | C | [PT-PYPI], [NCCL-DOCS] |
| Driver | 580 or later for CUDA 13.x minor-version compatibility | CUDA 13.4 corresponds to the R615 branch. | C | [CUDA-RN] |

### API differences writers must know

**Triton block pointers versus tensor descriptors.**

- `tl.make_block_ptr` still exists in Triton 3.8.0 but is deprecated: Triton 3.7.0 (May 2026) emits a deprecation warning when it is used, and 3.8.0 release notes still mention a block-pointer padding fix. The current Triton language reference lists no `make_block_ptr` or `advance` entry, and its documentation page returns 404. [TRITON-REL], [TRITON-LANG]
- The replacement is `tl.make_tensor_descriptor(base, shape, strides, block_shape, padding_option='zero')`, with `desc.load([m, n])` and `desc.store([m, n], value)`. The base pointer must be 16-byte aligned, leading strides must be multiples of 16 bytes, the last dimension must be contiguous, and only 2 to 5 dimensions are supported. It needs a global scratch allocator: call `triton.set_allocator(fn)` before launch. [TRITON-MKDESC]
- The documentation says the descriptor is backed by TMA hardware "on NVIDIA GPUs with TMA support". TMA exists on compute capability 9.0 and later; the official persistent-matmul tutorial gates its TMA kernels on `major >= 9`. **What `make_tensor_descriptor` does on an A100 (sm_80) is not documented (U). Test it on the lab image before a chapter depends on it.** [TRITON-MKDESC], [TRITON-PERSIST]
- The official matmul tutorial (03) uses plain pointer arithmetic, not block pointers or descriptors. Chapter 14's lab (block pointers, per the curriculum) should therefore print a note that the API is deprecated, and show the plain-pointer and descriptor forms in prose. [TRITON-MM]
- `tl.dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc=None, fast_math=False, ...)` takes e8m0 scales with group size 32 and formats e2m1, e4m3, e5m2, bf16, fp16. It is experimental. [TRITON-DOTSCALED]

**`torch.library.custom_op` (PyTorch 2.14).** Signature: `custom_op(name, fn=None, /, *, mutates_args, device_types=None, schema=None, tags=None)`. `name` has the form `"namespace::name"`. `mutates_args` is required and must be accurate, or behavior is undefined. If `schema` is `None` (recommended), PyTorch infers it from type annotations. Related entry points: `register_fake`, `register_autograd`, `triton_op`, `wrap_triton`, `opcheck`. [PT-LIBRARY]

**`torch.utils.cpp_extension.load_inline` (PyTorch 2.14).** Signature: `load_inline(name, cpp_sources, cuda_sources=None, sycl_sources=None, functions=None, extra_cflags=None, extra_cuda_cflags=None, extra_sycl_cflags=None, extra_ldflags=None, extra_include_paths=None, build_directory=None, verbose=False, with_cuda=None, with_sycl=None, is_python_module=True, with_pytorch_error_handling=True, keep_intermediates=True, use_pch=False, no_implicit_headers=False)`. If `functions` is given, bindings are generated for each name. `with_cuda=None` is inferred from `cuda_sources`. `no_implicit_headers=True` stops the automatic `#include <torch/extension.h>`. Set `TORCH_CUDA_ARCH_LIST` to override the default compute capabilities, for example `"8.0"` for the lab A100. `name` must be unique per distinct source or the build cache returns stale code (U for the cache behavior; established practice, not read in the docs). [PT-CPPEXT]

**Fused linear cross-entropy is now in PyTorch (chapter 19).** `torch.nn.functional.linear_cross_entropy(input, linear_weight, target, *, linear_bias=None, weight=None, reduction='mean', ignore_index=None, label_smoothing=0.0, options=None)` and `torch.nn.LinearCrossEntropyLoss` exist in the 2.14 documentation (2.13 documentation pages also appear in search; the first version that shipped them is U). Passing `options=None` uses the reference path (full logits). Passing `torch.nn.LinearCrossEntropyOptions(allow_retain_graph=False, batch_chunk_size=None, chunking_method='auto', acc_policy='auto', acc_dtype=None)` opts into chunking over batch rows, so the full logits tensor is never built. The chunked path precomputes gradients inside forward, does not support higher-order or forward-mode autodiff or `torch.func.grad`, and `torch.compile` falls back to eager at that op. `LinearCrossEntropyOptions` no longer accepts `acc_policy="balanced"` as of 2.14. The chapter must present the lab's hand-written version as the mechanism and this API as the production equivalent. [PT-LCE], [PT-LCEOPT], [PT-2140]

**Other 2.14 changes that touch chapters.**

- `torch.profiler` removed the `use_cuda` argument. Use `activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]`. (Chapter 05.) [PT-2140]
- Inductor gained "NVGEMM": CuTeDSL-generated CUTLASS kernels with epilogue fusion, `addmm` autotuning, and scaled and NVFP4 GEMM, autotuned alongside Triton and ATen. (Chapter 17.) [PT-2140]
- `torch.while_loop` can be captured in a CUDA graph; `CUDAGraph.register_generator_state()` is a no-op with a warning; one `CUDAGraph` can use multiple memory pools. (Chapter 17.) [PT-2140]
- `torch.nn.functional.scaled_mm` is the public scaled-GEMM entry point; `torch._scaled_mm` requires compute capability 8.9 or 9.0 and later (or ROCm MI300+), so it fails on the lab A100. MXFP4 checks `major == 10`. (Chapter 09 and 18, prose only.) [PT-DTYPES-RPT]
- `torch.backends.cuda.matmul.allow_tf32` defaults to `False`, so an FP32 `torch.matmul` on the A100 uses CUDA cores at the 19.5 TFLOP/s FP32 peak, not the 156 TFLOP/s TF32 peak. `torch.backends.cudnn.allow_tf32` defaults to `True`. BF16 GEMM allows reduced-precision reduction by default. Chapter 10 compares SGEMM against cuBLAS with TF32 off. [PT-CUDA-NOTES]

---

## 2. Lab GPU: A100 80GB PCIe (sm_80), with comparison parts

### 2.1 Headline specification table

Compute figures are **dense** (no sparsity) unless marked. NVIDIA prints sparse figures in its tables; the dense value is half the sparse value. Bandwidth figures are decimal (1 GB = 10^9 bytes).

| Property | A100 80GB PCIe | A100 80GB SXM4 | H100 SXM5 | H100 PCIe | B200 (HGX, per GPU) |
|---|---|---|---|---|---|
| Architecture, compute capability | Ampere, 8.0 | Ampere, 8.0 | Hopper, 9.0 | Hopper, 9.0 | Blackwell, 10.0 |
| SMs | 108 | 108 | 132 | 114 | 148 (2 dies, 74 enabled per die) |
| FP32 CUDA cores per SM | 64 | 64 | 128 | 128 | 128 (U; derived from FP32 rate below) |
| FP32 CUDA cores per GPU | 6,912 | 6,912 | 16,896 | 14,592 | U |
| Tensor cores per SM | 4 (3rd gen) | 4 | 4 (4th gen) | 4 | 4 (5th gen) |
| Tensor cores per GPU | 432 | 432 | 528 | 456 | 592 (D: 148 x 4) |
| Boost clock | 1410 MHz | 1410 MHz | 1830 MHz tensor; 1980 MHz vector (D, see 2.3) | about 1620 MHz tensor (D) | about 1856 MHz (D, U) |
| FP64 (CUDA cores) | 9.7 TFLOP/s | 9.7 | 34 (33.5) | 26 (S) | 37 (D from HGX 296 / 8) |
| FP64 tensor core | 19.5 | 19.5 | 67 (66.9) | 51 (S, with FP32 51) | 37 |
| FP32 (CUDA cores) | 19.5 | 19.5 | 67 (66.9) | 51 (S) | 75 (D from HGX 600 / 8) |
| TF32 tensor core, dense | 156 | 156 | 494.7 | 378 (peak used by [LUO-HOPPER] for an H800 PCIe, same 114 SMs) | 1,125 (D: 18 PFLOP/s sparse / 8 / 2) |
| BF16 and FP16 tensor core, dense | 312 | 312 | 989.4 | 756.5 | 2,250 (HGX) |
| FP8 tensor core, dense | none | none | 1,978.9 | 1,513 | 4,500 (HGX) |
| INT8 tensor core, dense | 624 TOPS | 624 TOPS | 1,978.9 TOPS | 1,513 TOPS | 4,500 TOPS (HGX B200) |
| FP4 tensor core, dense | none | none | none | none | 9,000 (HGX; NVFP4, MXFP4) |
| Memory | 80 GB HBM2e | 80 GB HBM2e | 80 GB HBM3 | 80 GB HBM2e | 180 GB HBM3e in HGX B200 (192 GB per the architecture blog) |
| Memory bandwidth | 1,935 GB/s | 2,039 GB/s | 3,350 GB/s | about 2,000 GB/s | 8,000 GB/s |
| Memory interface | 5,120-bit (5 stacks) | 5,120-bit | 5,120-bit (5 stacks) | 5,120-bit | 8,192-bit (8 stacks) |
| L2 cache | 40 MB | 40 MB | 50 MB | 50 MB | 126 MB |
| Shared memory plus L1 per SM | 192 KB | 192 KB | 256 KB | 256 KB | 256 KB |
| Max shared memory per SM | 164 KB | 164 KB | 228 KB | 228 KB | 228 KB |
| Max shared memory per block | 163 KB | 163 KB | 227 KB | 227 KB | 227 KB |
| Register file per SM | 256 KB (64K 32-bit registers) | 256 KB | 256 KB | 256 KB | 256 KB |
| Tensor memory (TMEM) per SM | none | none | none | none | 256 KB |
| Max registers per thread | 255 | 255 | 255 | 255 | 255 |
| Max resident threads, warps, blocks per SM | 2048, 64, 32 | 2048, 64, 32 | 2048, 64, 32 | 2048, 64, 32 | 2048, 64, 32 |
| Warp schedulers (sub-partitions) per SM | 4 | 4 | 4 | 4 | 4 (U) |
| NVLink | bridge for 2 GPUs: 600 GB/s | 600 GB/s | NVLink 4: 900 GB/s | bridge for 2 GPUs: 600 GB/s | NVLink 5: 1,800 GB/s |
| Host link | PCIe Gen4 x16: "64 GB/s" (both directions) | PCIe Gen4 x16 | PCIe Gen5: "128 GB/s" (both directions) | PCIe Gen5 | PCIe Gen5 x16 (U; Blackwell Ultra uses Gen6) |
| TDP | 300 W | 400 W | 700 W | 350 W | 1,000 W in HGX B200 (S); 1,200 W per the architecture blog |

Sources by column. A100 PCIe and SXM: [A100-DS] for rated compute, bandwidth, TDP, NVLink, and PCIe; [GA100-WP] for SM count, core counts, 40 MB L2, 164 KB shared memory, 192 KB L1 plus shared, 256 KB register file, 255 registers per thread, 5 stacks and 5,120-bit interface, 1410 MHz; [CPG-CC] for 2048 threads, 64 warps, 32 blocks, 163 KB per block; [SUN-TC] for four warp schedulers. H100 SXM: [H100-PAGE] (final throughput table, with sparsity; halved here), [HOPPER-WP] (132 SMs, 128 FP32 cores per SM, 50 MB L2, 228 KB, 256 KB register file, 5,120-bit HBM3, 700 W, PCIe Gen5 128 GB/s total), [HOPPER-TUNING] (227 KB per block). H100 PCIe: [HOPPER-WP] (114 SMs, 14,592 cores, 456 tensor cores, 50 MB L2, 350 W, HBM2e "over 2 TB/s", bridge 600 GB/s); 756.5 TFLOP/s dense is the figure [LUO-HOPPER] uses for an H800 PCIe with the same 114 SMs, and 1,513 sparse FP8 appears in [FLOPPER]; the exact final PCIe FP64 and FP32 figures come from a reseller summary (S). B200: [HGX-PAGE] (8-GPU totals), [BW-ULTRA-BLOG] (dual die, 8 TB/s, 192 GB, 1.8 TB/s NVLink, 1,200 W, 5th-gen tensor cores, 256 KB TMEM), [CNC-B200] (148 SMs, 126 MB L2, 256 KB L1 plus shared), [BW-TUNING] (228 KB, 227 KB per block, 64 warps, 32 blocks, 64K registers, 255 per thread).

### 2.2 Arithmetic for every derived peak

All A100 figures use 108 SMs at 1.41 GHz. Peak rate = SMs x (FLOP per SM per clock) x clock.

| Quantity | FLOP per SM per clock | Arithmetic | Result | Tag |
|---|---|---|---|---|
| FP16 or BF16 tensor | 2,048 (1,024 FMA = 4 tensor cores x 256 FMA) | 108 x 2,048 x 1.41e9 | 311.9 TFLOP/s (rated 312) | D; FMA rate in [GA100-WP] |
| TF32 tensor | 1,024 | 108 x 1,024 x 1.41e9 | 155.9 TFLOP/s (rated 156) | D |
| INT8 tensor | 4,096 | 108 x 4,096 x 1.41e9 | 623.7 TOPS (rated 624) | D |
| FP64 tensor | 128 (64 FMA per SM) | 108 x 128 x 1.41e9 | 19.49 TFLOP/s (rated 19.5) | D; 64 FMA in [GA100-WP] |
| FP32 CUDA cores | 128 (64 cores x 2) | 108 x 128 x 1.41e9 | 19.49 TFLOP/s (rated 19.5) | D |
| FP64 CUDA cores | 64 (32 cores x 2) | 108 x 64 x 1.41e9 | 9.75 TFLOP/s (rated 9.7) | D; 32 FP64 cores per SM from [HOPPER-WP] table |

The per-SM FP16 rate in the first row comes from the whitepaper: "Each of the A100 Tensor Cores can execute 256 FP16 FMA operations per clock" and each SM has four. The TF32 and INT8 rows are the datasheet ratios (156 / 312 and 624 / 312) applied to that rate; the whitepaper does not print the per-clock TF32 or INT8 rates in the text I read (U for the per-clock values, but the products match the rated peaks).

H100 SXM (132 SMs):

| Quantity | Arithmetic | Result | Tag |
|---|---|---|---|
| BF16 or FP16 tensor, dense | 132 x 4,096 x 1.83e9 (2x the A100 per-SM, per-clock rate: 4,096 FLOP per SM per clock) | 989.4 TFLOP/s (product page lists 1,979 with sparsity) | D |
| FP8 tensor, dense | 132 x 8,192 x 1.83e9 | 1,978.9 TFLOP/s (product page 3,958 with sparsity) | D |
| TF32 tensor, dense | 132 x 2,048 x 1.83e9 | 494.7 TFLOP/s (product page 989 with sparsity) | D |
| FP32 CUDA cores | 132 x 256 x 1.98e9 | 66.9 TFLOP/s | D |
| FP64 tensor | 132 x 256 x 1.98e9 | 66.9 TFLOP/s (rated 67) | D |

Clocks: the whitepaper says the per-SM tensor-core rate is "double ... clock-for-clock" versus A100, and prints "Not Finalized" for the H100 boost clock. Working backwards from the final rated 989.4 TFLOP/s gives 1.830 GHz for the tensor figures, while the final FP32 and FP64 figures (66.9) imply 1.980 GHz. Glenn Klockwood's table labels 1.83 GHz "low precision" and 1.98 GHz "high precision" (S). Treat both clocks as derived (D, not stated by NVIDIA in a document I read). [HOPPER-WP], [H100-PAGE], [KLOCKWOOD]

H100 PCIe (114 SMs): 756.5e12 / (114 x 4,096) = 1.620 GHz. NVIDIA's whitepaper lists a larger boost clock (the paper by [LUO-HOPPER] lists 1,755 MHz for the H800 PCIe), and 114 x 4,096 x 1.755e9 = 819.5 TFLOP/s, which is not the rated figure. **The rated PCIe peaks therefore correspond to about 1.62 GHz, not the maximum clock (D).**

B200 (148 SMs): the architecture blog and Tri Dao give 8,192 tensor-core FLOP per SM per clock for BF16 (1,024 16-bit MAC per tensor-core partition per cycle times 4 partitions, times 2 FLOP per MAC). 2.25e15 / (148 x 8,192) = 1.856 GHz. NVIDIA does not publish this clock in a document I read (U). [FA4-BLOG], [CNC-B200], [HGX-PAGE]

HGX B200 per-GPU figures divide the 8-GPU totals by 8: FP4 72 PFLOP/s dense (144 sparse) gives 9.0; FP8 72 sparse gives 36 dense, 4.5 per GPU; BF16 36 sparse gives 18 dense, 2.25 per GPU; TF32 18 sparse gives 9 dense, 1.125 per GPU; FP64 296 TFLOP/s gives 37 per GPU; FP32 600 TFLOP/s gives 75 per GPU. [HGX-PAGE]

### 2.3 Ridge points

The ridge point is peak compute divided by peak bandwidth, in FLOP per byte: I* = C / beta. Use dense peaks.

| GPU and precision | C (TFLOP/s) | beta (TB/s) | Ridge point (FLOP per byte) |
|---|---|---|---|
| A100 PCIe, BF16 or FP16 | 312 | 1.935 | 161.2 |
| A100 PCIe, BF16, against measured 1.275 TB/s copy | 312 | 1.275 | 244.7 |
| A100 SXM, BF16 or FP16 | 312 | 2.039 | 153.0 |
| A100 PCIe, TF32 | 156 | 1.935 | 80.6 |
| A100 PCIe, INT8 | 624 | 1.935 | 322.5 |
| A100 PCIe, FP32 CUDA cores | 19.5 | 1.935 | 10.1 |
| A100 PCIe, FP64 tensor core | 19.5 | 1.935 | 10.1 |
| A100 PCIe, FP64 CUDA cores | 9.7 | 1.935 | 5.0 |
| H100 SXM, BF16 | 989.4 | 3.35 | 295.3 |
| H100 SXM, TF32 | 494.7 | 3.35 | 147.7 |
| H100 SXM, FP8 | 1,978.9 | 3.35 | 590.7 |
| H100 SXM, FP32 CUDA cores | 66.9 | 3.35 | 20.0 |
| H100 PCIe, BF16 | 756.5 | 2.0 | 378.3 |
| B200, BF16 | 2,250 | 8.0 | 281.3 |
| B200, FP8 | 4,500 | 8.0 | 562.5 |
| B200, FP4 | 9,000 | 8.0 | 1,125 |

All values are D. The 8.0 TB/s B200 figure and 2,250 TFLOP/s BF16 figure are the HGX B200 (1,000 W) numbers; NVIDIA's architecture blog lists 5 PFLOP/s dense FP8 and 10 PFLOP/s dense NVFP4 for the same GPU at a higher power limit, which would give 625 and 1,250 FLOP per byte. See section 11. The 1.275 TB/s row uses the measured value from chapter 10 of learn-inference (section 10).

### 2.4 Other lab-GPU facts

| Fact | Value | Tag | Source |
|---|---|---|---|
| HBM2e data rate for the 80GB PCIe card | 1,935e9 x 8 / 5,120 = 3.02 Gb/s per pin, which is 1.512 GHz double data rate | D | [A100-DS], [GA100-WP] |
| 40GB A100 HBM2 data rate | 1,555e9 x 8 / 5,120 = 2.43 Gb/s per pin; the whitepaper states 1215 MHz DDR | C | [GA100-WP] |
| A100 shared-memory plus L1 carveouts | 0, 8, 16, 32, 64, 100, 132, 164 KB of shared memory; the rest of the 192 KB is L1 | C | [CPG-CC] |
| A100 register file, whole GPU | 108 x 256 KB = 27,648 KB | C | [GA100-WP] |
| A100 L2 structure | 2 partitions; each has 40 slices of 512 KB (40 x 512 KB = 20 MB per partition); 8 slices per memory controller; 10 memory controllers | C | [GA100-WP] |
| A100 L2 read bandwidth, theoretical | 5,120 B per clock x 1.41 GHz = 7.2 TB/s | D | [GA100-WP] |
| A100 shared-memory bandwidth, theoretical | 32 banks x 4 B = 128 B per clock per SM; 128 x 108 x 1.41e9 = 19.5 TB/s across the GPU | D | [CPG-BANKS] |
| A100 board memory ECC | HBM2e has SECDED ECC; the datasheet claims 95% DRAM utilization efficiency | C | [A100-DS], [GA100-WP] |
| MIG | up to 7 instances of 10 GB each on the 80GB parts | C | [A100-DS] |
| A100 SXM and PCIe peak figures are identical in the datasheet | The PCIe card has a 300 W limit against 400 W, so sustained clocks may fall below 1410 MHz under heavy tensor load. The datasheet does not quantify this. | C (identical figures); U (throttling magnitude) | [A100-DS] |
| A100 transistor count, process | 54.2 billion, TSMC 7 nm | C | [GA100-WP] |
| H100 transistor count, process | 80 billion, TSMC 4N custom | C | [HOPPER-WP] |
| B200 | 208 billion transistors, two reticle-sized dies joined by NV-HBI at 10 TB/s, TSMC 4NP, presented as one CUDA device | C | [BW-ULTRA-BLOG] |

---

## 3. Memory system facts

### 3.1 Granularity and coalescing

| Fact | Value | Tag | Source |
|---|---|---|---|
| Global-memory access unit (sector) | 32 bytes, "regardless of whether global loads are cached in L1 or not" | C | [BPG] |
| Cache line | 128 bytes (a 128-byte aligned segment; L1 and L2 hits are tracked in 128-byte lines made of four 32-byte sectors) | C for the 128-byte line (archived CUDA 11.8 guide); the current guide counts only 32-byte transactions | [CPG-118] |
| Coalescing rule (compute capability 6.0 and later) | The accesses of a warp "coalesce into a number of transactions equal to the number of 32-byte transactions necessary" to serve all threads | C | [BPG] |
| Aligned warp float load | 32 threads x 4 B = 128 B = four 32-byte sectors (one 128-byte line) | D, consistent with [BPG] "four coalesced 32-byte transactions" | [BPG] |
| Misaligned sequential float load | "five 32-byte segments" are requested when the base is not 32-byte aligned; two cache lines when the base is not 128-byte aligned | C for five segments; D for lines | [BPG] |
| Widest per-thread access | 16 bytes (`float4`, `LDG.128`), so one warp instruction can touch 512 B, or four 128-byte lines | D | [BPG] |
| Natural alignment | Global accesses of 1, 2, 4, 8, or 16 B compile to a single instruction only if naturally aligned. Allocations from `cudaMalloc` are at least 256-byte aligned. | C | [CPG-118] |
| Strided 32-float read | each of the 32 threads needs its own 32-byte sector: 32 sectors fetched (1,024 B) for 128 useful bytes, an 8x read amplification | D | learn-inference chapter 12 |
| Strided copy amplification, read plus write | read 32 B plus coalesced write 4 B = 36 B moved per 8 B counted, a 4.5x amplification | D | learn-inference chapter 12 |

### 3.2 Shared memory

| Fact | Value | Tag | Source |
|---|---|---|---|
| Banks | 32 banks; successive 32-bit words map to successive banks; each bank serves 32 bits per clock | C | [CPG-BANKS] |
| Bank index | bank = (byte address / 4) mod 32. The formula follows from the two statements above; no NVIDIA page prints it verbatim. | D | [CPG-BANKS] |
| Same-word access | Reads of the same 32-bit word by several threads do not conflict (broadcast); for writes one thread wins | C | [CPG-BANKS], [BPG] |
| N-way conflict | Threads that hit N different words in the same bank serialize into N transactions | C | [BPG] |
| 32x32 float tile, column read | Address = row x 32 + col; every thread of a warp reads the same column, so all 32 words land in one bank: a 32-way conflict. Padding the row to 33 words puts consecutive rows in consecutive banks and removes the conflict. | D | [HARRIS-TRANSPOSE] |
| 64-bit bank mode | Documented only for compute capability 3.x. No current page describes it for Ampere and later. Do not teach it. | C | [CPG-118] |
| Per-block shared-memory limit | 48 KB static; larger dynamic allocations need an explicit opt-in (`cudaFuncSetAttribute` with `cudaFuncAttributeMaxDynamicSharedMemorySize`) | C | [HOPPER-TUNING] |
| Reserved shared memory | CUDA reserves 1 KB per thread block | C | [HOPPER-TUNING] |

### 3.3 Measured latencies (cycles)

A100 PCIe 40GB (1410 MHz), H800 PCIe, and RTX 4090, from the microbenchmark paper [LUO-HOPPER], Table IV. Cycle-to-nanosecond conversion: ns = cycles / 1.41 for the A100.

| Level | A100 PCIe (cycles) | A100 (ns, D) | H800 PCIe (cycles) | RTX 4090 (cycles) |
|---|---|---|---|---|
| L1 cache | 37.9 | 26.9 | 40.7 | 43.4 |
| Shared memory | 29.0 | 20.6 | 29.0 | 30.1 |
| L2 cache | 261.5 | 185.5 | 263.0 | 273.0 |
| Global memory (HBM) | 466.3 | 330.7 | 478.8 | 541.5 |

- The paper's A100 is the 40 GB PCIe card (1,555 GB/s); no 80 GB measurement exists in the sources read (U for the 80 GB card, but the latencies are expected to be close). C for the table.
- Distributed shared memory on the H800: SM-to-SM latency 180 cycles, "32% reduction compared to L2". Peak SM-to-SM throughput about 3.27 TB/s at cluster size 2 and 2.65 TB/s at size 4. C. [LUO-HOPPER]
- B200 (Chips and Cheese): L1 latency 39 cycles (19.6 ns); L2 local partition about 150 ns; thread-pair latency 90 to 100 ns same partition, 190 to 220 ns cross-partition; no DRAM figure. C (as measured by the article). [CNC-B200]
- Older source: the Volta microbenchmark paper [JIA-VOLTA] gives the method (pointer chasing) and V100 numbers; use it for methodology, not for A100 values.
- The learn-inference table (~30 cycles shared, ~200 L2, ~400 HBM) is a rounded teaching table. When a chapter needs a number, use the measured row above or say "roughly 30, 260, and 470 cycles".

### 3.4 Measured throughputs

| Fact | Value | Tag | Source |
|---|---|---|---|
| A100 L1 throughput, measured | 99.5 (FP32), 106.8 (FP32.v4) bytes per clock per SM; theoretical 128 | C | [LUO-HOPPER], Table V |
| A100 L2 throughput, measured | 1,853.7 (FP32) to 2,007.9 (FP32.v4) bytes per clock; this is 2.8 TB/s at 1.41 GHz | C; D for TB/s | [LUO-HOPPER], Table V |
| A100 L2 throughput, vendor claim | 5,120 B per clock read (theoretical) | C | [GA100-WP] |
| A100 L2 read bandwidth, article measurement | 4.5 TB/s (Chips and Cheese, which tested an SXM part; H100 PCIe: 3.8 TB/s for the full 50 MB) | S | [CNC-H100] |
| L2 versus DRAM bandwidth on A100 | L2 throughput is 2.01 times the global-memory throughput in the paper's tests | C | [LUO-HOPPER] |
| Share of theoretical memory throughput reached | 90% (A100), 91% (H800), 92% (RTX 4090) in the paper's tests | C | [LUO-HOPPER] |
| Bytes in flight needed to saturate A100 HBM (Little's law) | 1,935e9 B/s x 466.3 cycles / 1.41e9 Hz = 640 KB across the GPU, or 5.9 KB per SM | D (uses the 40GB card latency) | [LUO-HOPPER], [A100-DS] |

The 5,120 versus about 2,000 bytes per clock gap on L2 is a source disagreement (section 11).

### 3.5 Asynchronous copy and TMA

| Fact | Value | Tag | Source |
|---|---|---|---|
| `cp.async` (sm_80 and later) | copies 4, 8, or 16 bytes from global to shared memory; `.ca` caches at all levels, `.cg` caches in L2 only and is allowed only with 16 bytes; an optional `src-size` zero-fills the rest | C | [PTX] |
| `cp.async` completion | `cp.async.commit_group` closes a per-thread group; `cp.async.wait_group N` waits until at most N groups are pending; `cp.async.wait_all` equals commit plus wait 0 | C | [PTX] |
| What `cp.async` removes | The copy bypasses the register file ("eliminating the need for intermediate register file usage") and can bypass L1 | C | [GA100-WP] |
| `cp.async.bulk` (sm_90 and later) | size must be a multiple of 16 bytes; addresses 16-byte aligned; completes through an `mbarrier` | C | [PTX] |
| `cp.async.bulk.tensor` (TMA, sm_90 and later) | 1D to 5D tensors; modes tile and im2col; gather4 and scatter4 added later; issued by one thread | C | [PTX], [HOPPER-TUNING] |
| TMA from shared to global | can apply add, min, max, and, or reductions on write | C | [HOPPER-TUNING] |
| `mbarrier.expect_tx` | `txCount` is a 32-bit unsigned byte count; scope `.cta` or `.cluster` | C | [PTX] |
| TMA descriptor needs | a global-memory allocation (hence `triton.set_allocator`) | C | [TRITON-MKDESC] |
| Hopper TMA and tuning guidance | TMA avoids using registers and SM instructions for data movement, which enables warp specialization | C | [HOPPER-TUNING] |
| Thread block clusters | Portable maximum 8 blocks; 16 on H100 and B200 with `cudaFuncAttributeNonPortableClusterSizeAllowed`; DSMEM accesses should be coalesced and aligned to 32-byte segments | C | [HOPPER-TUNING], [BW-TUNING] |
| L2 residency control | Ampere adds L2 cache residency controls for the programmer | C | [GA100-WP] |

---

## 4. Execution facts

### 4.1 Warps and scheduling

| Fact | Value | Tag | Source |
|---|---|---|---|
| Warp size | 32 threads on every NVIDIA architecture in this course | C | [CPG-CC] |
| Max threads per block | 1,024; block dimensions up to 1,024 (x, y) and 64 (z); grid x up to 2^31 - 1, y and z up to 65,535 | C | [CPG-CC] |
| Warp schedulers per SM | four warp schedulers ("sub-cores") per SM on Volta, Turing, and Ampere; each sub-core has its own warp scheduler, register file, CUDA cores, and tensor core | C | [SUN-TC] |
| FP32 cores per sub-partition (A100) | 64 / 4 = 16 | D | [GA100-WP] |
| Independent thread scheduling (compute capability 7.0 and later) | The GPU keeps a program counter and call stack per thread and "can yield execution at a per-thread granularity"; a schedule optimizer regroups active threads into SIMT units. Pre-Volta code used one program counter per warp plus an active mask. | C | [CPG-ITS] |
| Consequence | Warp-synchronous code that assumed lockstep must call `__syncwarp()`; threads can diverge and reconverge at sub-warp granularity | C | [CPG-ITS] |
| Divergence cost | The warp executes each taken branch path in turn and masks off the other threads; "Branch divergence occurs only within a warp" | C | [CPG-ITS], [CPG-PROG] |
| `*_sync` intrinsic masks | Every calling thread must have its own bit set; all non-exited threads named in the mask must execute the intrinsic with the same mask; violations are undefined behavior | C | [CPG-LANG] |
| Execution scope of divergence | Differences between blocks do not diverge; only threads in the same warp | C | [CPG-ITS] |

### 4.2 Per-architecture occupancy limits

| Limit | CC 7.5 | CC 8.0 (A100) | CC 8.6 | CC 8.9 | CC 9.0 (H100) | CC 10.0 (B200) | CC 12.x |
|---|---|---|---|---|---|---|---|
| Max resident blocks per SM | 16 | 32 | 16 | 24 | 32 | 32 | 24 |
| Max resident warps per SM | 32 | 64 | 48 | 48 | 64 | 64 | 48 |
| Max resident threads per SM | 1,024 | 2,048 | 1,536 | 1,536 | 2,048 | 2,048 | 1,536 |
| 32-bit registers per SM | 64K | 64K | 64K | 64K | 64K | 64K | 64K |
| Max registers per thread | 255 | 255 | 255 | 255 | 255 | 255 | 255 |
| Max shared memory per SM | 64 KB | 164 KB | 100 KB | 100 KB | 228 KB | 228 KB | 100 KB |
| Max shared memory per block | 64 KB | 163 KB | 99 KB | 99 KB | 227 KB | 227 KB | 99 KB |
| Shared-memory banks | 32 | 32 | 32 | 32 | 32 | 32 | 32 |
| Local memory per thread | 512 KB | 512 KB | 512 KB | 512 KB | 512 KB | 512 KB | 512 KB |

Tag C. Source: [CPG-CC] for all columns except 10.0, which is [BW-TUNING] (the Programming Guide's flattened 10.x cell was ambiguous). Compute capability 7.0 (V100) is not in the retrieved table (U).

### 4.3 Occupancy arithmetic for A100 (all D)

Resident blocks per SM = min over four limits: threads, warps, registers, shared memory, and the 32-block cap.

Assumptions (U, not read in a primary document: standard occupancy-calculator granularity): registers are allocated per warp in units of 256 registers (8 per thread, so a thread using 33 registers is charged 40); the occupancy calculator is deprecated in favor of Nsight Compute, so prefer `cudaOccupancyMaxActiveBlocksPerMultiprocessor` in labs rather than hard-coding granularity. [FORUM-OCC], [NCU-GUIDE]

Worked examples, 256-thread blocks (8 warps), 64K registers per SM, 2,048 threads per SM:

| Registers per thread | Registers per block | Blocks by registers | Blocks by threads (2048 / 256) | Resident blocks | Resident warps | Occupancy |
|---|---|---|---|---|---|---|
| 32 | 8,192 | 8 | 8 | 8 | 64 | 100% |
| 64 | 16,384 | 4 | 8 | 4 | 32 | 50% |
| 128 | 32,768 | 2 | 8 | 2 | 16 | 25% |
| 255 (charged as 256) | 65,536 | 1 | 8 | 1 | 8 | 12.5% |

Shared-memory example: a block that uses 48 KB static shared memory is charged 49 KB with the 1 KB reservation; 164 KB / 49 KB = 3.35, so 3 blocks per SM (768 threads at 256 threads per block = 37.5% occupancy). A block that uses 163 KB (the maximum) is the only block on the SM.

Volkov's argument that occupancy is a means, not a goal: a kernel with enough independent memory requests or instruction-level parallelism per thread can saturate bandwidth at 25% occupancy. [VOLKOV-GTC]

### 4.4 Waves and quantization (A100)

| Fact | Value | Source |
|---|---|---|
| Concurrent blocks across the GPU | 108 SMs x resident blocks per SM; a "wave" is one full set of resident blocks | D |
| NVIDIA's worked example | M = 2304, K = 4096 with 256x128 tiles: N = 1536 gives 9 x 12 = 108 tiles, one full wave; N in (1536, 1664] gives 9 x 13 = 117 tiles, so a second wave of 9 tiles uses 9 / 108 = 8.33% of the SMs; throughput roughly halves from N = 1536 to N = 1544 | C, [NV-MATMUL] |
| Tile quantization | With 128x128 tiles, matrix dimensions that are not multiples of the tile size run partly empty tiles; the worst case does 1.5x the arithmetic of the best case | C, [NV-MATMUL] |
| Tensor-core-friendly dimensions (cuBLAS 11 and later, most efficient on A100) | FP16: multiples of 64; TF32: multiples of 32; INT8: multiples of 128; FP64: multiples of 16 | C, [NV-MATMUL] |
| Arithmetic intensity example | 8192 x 128 x 8192 GEMM: 124.1 FLOP per byte (V100 ridge 138.9, so memory limited); 8192^3: 2,730 | C, [NV-MATMUL] |

### 4.5 Launch overhead and CUDA graphs

| Fact | Value | Tag | Source |
|---|---|---|---|
| Host launch cost, lab A100 | about 5 microseconds to queue a kernel (learn-inference chapter 12) | measured by learn-inference, see section 10 | learn-inference |
| Course rule of thumb | each launch costs 5 to 10 microseconds | learn-inference chapter 00a | learn-inference |
| NVIDIA measurement on V100 (500,000-element kernel, 2.9 microseconds of execution) | 9.6 microseconds per kernel with a sync after every launch (about 6.7 microseconds overhead); 3.8 microseconds when launches overlap with one sync per step; 3.4 microseconds per kernel inside a graph of 20 kernels (about 0.5 microsecond overhead); graph creation and instantiation about 400 microseconds, once; first graph launch about 33% slower | C (overhead values are subtraction by the reader: article gives times, not overheads) | [CUDA-GRAPHS-BLOG] |
| Why graphs help | One `cudaGraphLaunch` submits many kernels; capture and instantiation are one-time costs amortized over replays | C | [CUDA-GRAPHS-BLOG] |
| PyTorch CUDA graphs | capture, static input buffers, replay; see the PyTorch blog | C | [PT-CUDAGRAPH-BLOG] |
| An A100 elementwise kernel on 2^24 floats | moves 3 x 64 MiB = 192 MiB (D); at 1,304 GB/s that is 154 microseconds (D), 30 times the launch cost | D | learn-inference chapter 12 |

---

## 5. Tensor core and PTX facts per architecture

### 5.1 Per-architecture summary

| Architecture | Tensor-core generation | New instruction path | Operand and accumulator location | Data types (tensor core) | Tag | Source |
|---|---|---|---|---|---|---|
| Volta (sm_70) | 1st | `wmma`, `mma.sync` m8n8k4 | registers | FP16 in, FP16 or FP32 accumulate | C | [PTX] |
| Turing (sm_75) | 2nd | `ldmatrix`, `mma.sync` m16n8k8 | registers | adds INT8, INT4, INT1 | C | [PTX] |
| Ampere (sm_80) | 3rd | `mma.sync` m16n8k16, `cp.async`, `mma.sp` | registers; A, B from registers fed by `ldmatrix` from shared memory | adds BF16, TF32, FP64 | C | [PTX], [GA100-WP] |
| Hopper (sm_90) | 4th | `wgmma.mma_async` (warpgroup), TMA, clusters | A from registers or shared memory; B from shared memory; accumulator in registers | adds FP8 (E4M3, E5M2) | C | [PTX], [HOPPER-WP] |
| Blackwell (sm_100) | 5th | `tcgen05.mma` (one thread issues), tensor memory | A from shared memory or tensor memory; B from shared memory; accumulator in tensor memory | adds FP6, FP4, block-scaled MXFP8, MXFP4, NVFP4 | C | [PTX], [TCGEN05-GUIDE] |

Hopper in the Luo microbenchmarks: there is no `mma.sync` for FP8; INT4 `mma` compiles to CUDA-core instructions; `wgmma` is exclusive to Hopper (not Ada). [LUO-HOPPER]

### 5.2 WMMA (CUDA C++ `nvcuda::wmma`)

| Input type to accumulator | Shapes (m x n x k) | Min arch | Tag |
|---|---|---|---|
| f16 to f16 or f32 | 16x16x16, 32x8x16, 8x32x16 | sm_70 | C |
| s8 or u8 to s32 | 16x16x16, 32x8x16, 8x32x16 | sm_72 | C |
| bf16 to f32 | 16x16x16, 32x8x16, 8x32x16 | sm_80 | C |
| tf32 to f32 | 16x16x8 | sm_80 | C |
| f64 | 8x8x4 | sm_80 | C |
| u4 or s4 to s32 | 8x8x32 (experimental namespace, "preview") | sm_75 | C |
| b1 to s32 | 8x8x128 (experimental namespace) | sm_75 | C |

Alignment rules: `mptr` must be 256-bit aligned; `ldm` must be a multiple of 8 for half-precision and 4 for float (16 bytes). TF32 inputs to WMMA must be converted with `__float_to_tf32`. Source: [PTX], [CPG-LANG].

### 5.3 `mma.sync` shapes (dense)

| Type | Shapes | Min arch | Tag |
|---|---|---|---|
| f16 | m8n8k4, m16n8k8, m16n8k16 | m8n8k4: sm_70 (slow elsewhere); m16n8k8: sm_75; m16n8k16: sm_80 | C |
| bf16 | m16n8k8, m16n8k16 | sm_80 | C |
| tf32 | m16n8k4, m16n8k8 | sm_80 | C |
| f64 | m8n8k4 | sm_80 | C |
| f64 | m16n8k4, m16n8k8, m16n8k16 | sm_90 | C |
| s8 or u8 | m8n8k16; m16n8k16, m16n8k32 | sm_75; sm_80 | C |
| s4 or u4 | m8n8k32; m16n8k32, m16n8k64 | sm_75; sm_80 | C |
| b1 | m8n8k128; m16n8k128, m16n8k256 | sm_75; sm_80 | C |
| e4m3, e5m2 | m16n8k32 | sm_89 | C |
| e3m2, e2m3, e2m1, block-scaled | m16n8k32, m16n8k64 | sm_120a (consumer Blackwell) | C |

Sparse `mma.sp` (sm_80): f16 and bf16 m16n8k16 and m16n8k32; tf32 m16n8k8 and m16n8k16 (**1:2** sparsity, not 2:4); s8 and u8 m16n8k32 and m16n8k64; s4 and u4 m16n8k64 and m16n8k128 (**4:8**, pair-wise); fp8 m16n8k64 (sm_89). Metadata is two 2-bit indices per 4-wide chunk. [PTX]

For the lab's `mma.sync.m16n8k16` with fp16 inputs on sm_80: A is a 16x16 tile (row-major), B is a 16x8 tile (column-major), C and D are 16x8; one warp executes it. The per-instruction work is 16 x 8 x 16 = 2,048 FMA = 4,096 FLOP. D.

### 5.4 `ldmatrix`

| Fact | Value | Tag | Source |
|---|---|---|---|
| Syntax | `ldmatrix.sync.aligned.m8n8.x1/.x2/.x4{.trans}.shared.b16 r, [addr]` | C | [PTX] |
| Matrix size | each `.num` step loads one 8x8 matrix of 16-bit elements; the destination is 1, 2, or 4 32-bit registers per thread | C | [PTX] |
| Addresses | threads 0 to 7 supply the eight row addresses of matrix 0; threads 8 to 15 matrix 1 (x2); 16 to 23 and 24 to 31 matrices 2 and 3 (x4) | C | [PTX] |
| Row width | each row is 8 x 16 bits = 16 bytes and must be naturally aligned; "a group of four consecutive threads loads 16 bytes" | C | [PTX] |
| `.trans` | loads the matrix in column-major format | C | [PTX] |
| Min arch | sm_75 (not sm_80 specific) | C | [PTX] |

### 5.5 `wgmma` (sm_90a)

| Fact | Value | Tag | Source |
|---|---|---|---|
| Warpgroup | four contiguous warps, the first with rank a multiple of 4: 128 threads | C | [PTX] |
| M | always 64 | C | [PTX] |
| N | 8 to 256 in steps of 8 for f16, bf16, tf32, fp8 (steps of 16 after 32 for int8 and b1) | C | [PTX] |
| K per instruction | 16 (f16, bf16), 8 (tf32), 32 (fp8, int8); sparse doubles K | C | [PTX] |
| Operands | A from registers or a shared-memory descriptor; B from a shared-memory descriptor; accumulator in registers | C | [PTX] |
| Sequence | `wgmma.fence` then `wgmma.mma_async` then `wgmma.commit_group` then `wgmma.wait_group N`; touching the accumulator before the matching wait is undefined | C | [PTX] |
| Measured on H800 PCIe | FP16 m64n256k16: 729.3 TFLOP/s of 756.5 peak (96%) with zero-filled inputs; 704.5 with random inputs (93%); FP8 m64n256k32: 1,448 of 1,513 (96%); latency 128 cycles per instruction | C | [LUO-HOPPER], Table VIII |
| Measured on A100 PCIe, `mma.sync` | m16n8k16 FP16 to FP16: 310.6 TFLOP/s; FP16 to FP32: 303.4 (97% of 312); TF32 m16n8k8: 151.5 (97% of 156); INT8 m16n8k32: 607.6 TOPS (97% of 624); latency 24.6 to 26.3 cycles | C | [LUO-HOPPER], Table VII |
| Dense versus sparse measured on A100 | FP16 m16n8k16 sparse: 622.8 TFLOP/s | C | [LUO-HOPPER], Table VII |

### 5.6 Blackwell: `tcgen05` and tensor memory

| Fact | Value | Tag | Source |
|---|---|---|---|
| Tensor memory (TMEM) | 128 rows (lanes) x 512 columns of 32-bit cells = 128 x 512 x 4 B = 262,144 B = 256 KiB per SM; 64 KB per sub-partition | C; D for the product | [PTX], [CNC-B200], [BW-ULTRA-BLOG] |
| TMEM address | 32 bits: lane index in the upper 16 bits, column index in the lower 16 | C | [PTX] |
| TMEM allocation | one warp allocates; columns are a power of two in [32, 512] | C | [PTX] |
| Issue model | "Only one thread issues the MMA instruction"; asynchronous; completion tracked with `tcgen05.commit` plus an `mbarrier` | C | [TCGEN05-GUIDE], [PTX] |
| Operands | accumulator always in TMEM; A from shared memory or TMEM (K-major in TMEM); B from shared memory | C | [TCGEN05-GUIDE] |
| `tcgen05.mma` shapes, dense | `cta_group::1`: M = 64 or 128; `cta_group::2` (a CTA pair): M = 128 or 256; N = 8 to 256 in steps of 8 (group 1) or 16 (group 2); K = 16 (f16, bf16), 8 (tf32), 32 (fp8, int8), 64 for mxf4 and nvf4 (96 on sm_103a) | C | [PTX] |
| Largest BF16 single-CTA tile | 128x256x16, about 2x the largest Hopper `wgmma` atom; 2-CTA mode gives 256x256x16 | C | [FA4-BLOG] |
| Kinds | `kind::f16`, `tf32`, `f8f6f4`, `i8`, `mxf8f6f4`, `mxf4`, `mxf4nvf4`; block-scale types ue8m0, ue4m3, ue5m3 | C | [PTX] |
| Throughput versus Hopper | "2x to 4x the throughput of Hopper's WGMMA instructions, depending on data type" | C | [TCGEN05-GUIDE] |
| Data movement | `tcgen05.ld`, `tcgen05.st`, `tcgen05.cp` (shared memory to TMEM), `tcgen05.alloc`, `tcgen05.dealloc`, `tcgen05.commit` | C | [PTX] |
| Per-SM resource ratio on B200 (M = N = D = 128) | tensor cores 8,192 ops per cycle; exponential unit 16 ops per cycle; shared memory 128 bytes per cycle; so the tensor-to-exp ratio is 512:1 | C (stated in the blog); D for the ratio | [FA4-BLOG] |
| Per-tile cycles (forward): tensor 1,024, exp 1,024, shared memory 768; backward 1-CTA: tensor 2,560, exp 1,024, shared memory 3,328 | | C | [FA4-BLOG] |
| Exponential unit throughput | 10.7 tera-exponentials per second on Blackwell Ultra, 5 on Blackwell, 4.5 on Hopper | C | [BW-ULTRA-BLOG] |
| sm_100 versus sm_120 | CC 12.x consumer parts have 128 KB L1 plus shared, 99 KB per block, 48 warps; block-scaled `mma.sync` (not `tcgen05`) is their path | C | [BW-TUNING], [PTX] |

### 5.7 Thread block clusters and distributed shared memory

| Fact | Value | Tag | Source |
|---|---|---|---|
| Cluster | a group of thread blocks "guaranteed to be concurrently scheduled"; all blocks of a cluster run in one GPC | C | [HOPPER-BLOG], [PTX] |
| Max cluster size | 8 portable; 16 non-portable on H100 and B200 (opt in with `cudaFuncAttributeNonPortableClusterSizeAllowed`); larger clusters may reduce active blocks | C | [HOPPER-TUNING], [BW-TUNING] |
| Launch | `__cluster_dims__(x, y, z)` (compute capability 9.0 and later) or `cudaLaunchKernelEx` with the cluster-dimension attribute; the grid must be a multiple of the cluster size | C | [CPG-LANG] |
| Distributed shared memory | blocks in a cluster can load, store, and do atomics on each other's shared memory; PTX `mapa` maps an address to a rank; Cooperative Groups `cluster.map_shared_rank(ptr, rank)` | C | [HOPPER-TUNING], [PTX], [CPG-LANG] |
| NVIDIA's claim | DSMEM "accelerates data exchange between thread blocks by about 7x" compared with global memory | C | [HOPPER-BLOG] |

### 5.8 2:4 structured sparsity

| Fact | Value | Tag | Source |
|---|---|---|---|
| Pattern | in every group of 4 contiguous values at most 2 are non-zero (50% sparsity); only the 2 kept values are stored | C | [MISHRA] |
| Metadata | 2 bits per kept value (the position within the group of 4); overhead 12.5% for 16-bit values and 25% for 8-bit | C | [MISHRA] |
| Storage | A 16-bit group: dense 4 x 16 = 64 bits; compressed 2 x 16 + 2 x 2 = 36 bits, a ratio of 0.5625 | D | [MISHRA] |
| Hardware | Sparse tensor cores (Ampere and later) support FP16, BF16, INT8; TF32 uses 1:2 | C | [MISHRA], [PTX] |
| NVIDIA claim | "twice the math throughput of dense matrix units"; large INT8 GEMMs on A100 (M = N = 10,240) reach "nearly a 2x speedup" (cuSPARSELt versus cuBLAS) | C | [MISHRA] |
| A100 measured | sparse FP16 `mma.sp` m16n8k16 622.8 TFLOP/s versus dense 310.6 | C | [LUO-HOPPER] |
| PyTorch API | `torch.sparse.to_sparse_semi_structured(tensor)`; fp16 and bf16 (2:4), int8 (2:4); float32 only 1:2; CUDA tensors; 2D; sizes multiples of 64 (16-bit) or 128 (int8); backends cuSPARSELt and CUTLASS; prototype feature | C | [PT-SPARSE] |
| PyTorch reported speedups | BERT-base SQuAD on A100: batch 4 0.87x, batch 16 1.28x, batch 64 1.25x, batch 256 1.18x (tutorial headline "1.3x"); ViT-L MLP linear layers 1.3x forward plus backward on A100; matmul up to 1.6x with cuSPARSELt; DINOv2 ViT-L end to end 6%; segment-anything inference 10% | C (the GPU is named in the setup note, not per table row, for the BERT numbers) | [PT-SPARSE-TUT], [PT-SPARSE-BLOG] |
| Sparse GEMM timing | ViT-L shape 13,008 x 4,096 x 1,024: dense 538 microseconds, sparse 387 microseconds on A100; the compression step costs 380 microseconds with cuSPARSELt versus 36 microseconds with PyTorch's own kernel | C | [PT-SPARSE-BLOG] |

---

## 6. Floating-point formats

Value rule: normal value = (-1)^S x 2^(E - bias) x (1 + M / 2^m); subnormal (E = 0) = (-1)^S x 2^(1 - bias) x (M / 2^m). **Machine epsilon here is 2^-m, the gap between 1.0 and the next value.** Unit roundoff (round to nearest) is half of that.

| Format | Bits (S/E/M) | Bias | Max finite | Min positive normal | Min positive subnormal | Epsilon | Notes |
|---|---|---|---|---|---|---|---|
| fp32 | 32 (1/8/23) | 127 | (2 - 2^-23) x 2^127 = 3.4028e38 | 2^-126 = 1.1755e-38 | 2^-149 = 1.4013e-45 | 2^-23 = 1.1921e-7 | IEEE 754 binary32 (S: Wikipedia) |
| tf32 | 19 meaningful bits (1/8/10) in a 32-bit container | 127 | (2 - 2^-10) x 2^127 = 3.4012e38 | 2^-126 | 2^-136 = 1.148e-41 (D) | 2^-10 = 9.766e-4 | Layout "implementation defined" per PTX; NVIDIA states 8 exponent and 10 mantissa bits |
| bf16 | 16 (1/8/7) | 127 | (2 - 2^-7) x 2^127 = 3.3895e38 | 2^-126 = 1.1755e-38 | 2^-133 = 9.18e-41 | 2^-7 = 7.8125e-3 | same exponent range as fp32 |
| fp16 | 16 (1/5/10) | 15 | (2 - 2^-10) x 2^15 = 65,504 | 2^-14 = 6.1035e-5 | 2^-24 = 5.9605e-8 | 2^-10 = 9.766e-4 | IEEE 754 binary16 (S: Wikipedia) |
| fp8 E4M3 ("fn", OCP) | 8 (1/4/3) | 7 | S.1111.110 = 2^8 x 1.75 = **448**; no infinity; NaN only at S.1111.111 | 2^-6 = 0.015625 | 2^-9 = 0.001953125 | 2^-3 = 0.125 | The top exponent is reused for finite values. PyTorch: `torch.float8_e4m3fn` |
| fp8 E5M2 | 8 (1/5/2) | 15 | S.11110.11 = 2^15 x 1.75 = **57,344**; has infinity and NaN | 2^-14 = 6.1035e-5 | 2^-16 = 1.5259e-5 | 2^-2 = 0.25 | PyTorch: `torch.float8_e5m2` |
| fp6 E3M2 (MX) | 6 (1/3/2) | 3 | 2^4 x 1.75 = 28 | 2^-2 = 0.25 | 2^-4 = 0.0625 | 2^-2 = 0.25 | no infinity or NaN |
| fp6 E2M3 (MX) | 6 (1/2/3) | 1 | 2^2 x 1.875 = 7.5 | 2^0 = 1 | 2^-3 = 0.125 | 2^-3 = 0.125 | no infinity or NaN |
| fp4 E2M1 (MX, NVFP4) | 4 (1/2/1) | 1 | 2^2 x 1.5 = 6 | 2^0 = 1 | 2^-1 = 0.5 | 2^-1 = 0.5 | positive values: 0, 0.5, 1, 1.5, 2, 3, 4, 6 |
| E8M0 (MX scale) | 8 (0/8/0), unsigned | 127 | 2^127 = 1.70e38 | 2^-127 = 5.88e-39 | none | n/a | power-of-two only; NaN = 0xFF; PyTorch `torch.float8_e8m0fnu` |
| ue4m3 (NVFP4 block scale) | 7 (0/4/3), unsigned | 7 | 448 (D) | 2^-6 | 2^-9 | 2^-3 | NVFP4 uses an FP8 E4M3 scale per 16 values |
| int8 | 8-bit two's complement | n/a | +127 (min -128) | n/a | 1 | n/a | derived from two's complement; MXINT8 differs (implicit scale 2^-6) |
| int4 | 4-bit two's complement | n/a | +7 (min -8); uint4 0 to 15 | n/a | 1 | n/a | derived |

Tags: fp8, fp6, fp4, E8M0, MX rows C [OFP8], [MX-SPEC], [NVFP4-BLOG], [PTX]; fp32 and fp16 maxima C by arithmetic and S from Wikipedia (IEEE 754 itself was not fetched, U); bf16 and tf32 C [OFP8], [TF32-BLOG], [PTX]; int rows D.

E2M1 positive values by exponent field: E = 0 gives 0 and 0.5 (subnormal: 2^0 x 0.5 x M); E = 1 gives 2^0 x {1, 1.5} = 1, 1.5; E = 2 gives 2^1 x {1, 1.5} = 2, 3; E = 3 gives 2^2 x {1, 1.5} = 4, 6. D, matches [NVFP4-BLOG].

Bits per element including the scale (D from [MX-SPEC] and [NVFP4-BLOG]):

| Format | Block size | Scale | Bits per element |
|---|---|---|---|
| MXFP8 | 32 | E8M0 | 8 + 8/32 = 8.25 |
| MXFP6 | 32 | E8M0 | 6 + 8/32 = 6.25 |
| MXFP4 | 32 | E8M0 | 4 + 8/32 = 4.25 |
| MXINT8 | 32 | E8M0 | 8 + 8/32 = 8.25 |
| NVFP4 | 16 | FP8 E4M3 per block, plus one FP32 per tensor | 4 + 8/16 = 4.5 (per-tensor scale not counted) |
| int4, group 128, FP16 scale (learn-inference) | 128 | fp16 | 4 + 16/128 = 4.125 |

**Hardware that runs the low-precision formats.**

- FP8 tensor-core math starts at compute capability 8.9 (`mma.sync`) and 9.0 (`wgmma`). The A100 (8.0) has none. [CPG-CC], [PTX]
- FP6, FP4, and block-scaled MX types run on compute capability 10.0, 10.3, and later through `tcgen05.mma` (kinds `mxf8f6f4`, `mxf4`, `mxf4nvf4`), and on 12.x through warp-level `mma.sync` (needs `sm_120a`). Hopper has none. [PTX], [CPG-CC]
- cuBLASLt block scaling: `VEC32_UE8M0` for FP8 tensors and `VEC16_UE4M3` for FP4 tensors; mixing precisions is not supported. [CUBLAS-DOC]
- PyTorch 2.14: `torch.float8_e4m3fn`, `torch.float8_e5m2`, `torch.float8_e8m0fnu`, and `torch.float4_e2m1fn_x2` (two 4-bit values per byte) are "shell dtypes" with limited operator support. [PT-DTYPES]

**TF32 rounding.** `cvt.rna.tf32.f32` rounds to nearest with ties away from zero; `wgmma` on raw FP32 inputs truncates the lower 13 bits; NVIDIA's TF32 blog says inputs are rounded. Do not state one rule for all paths. [PTX], [TF32-BLOG]

**Accumulation.** Tensor-core accumulation order and rounding are unspecified in PTX ("at least single precision"); the Dissecting Tensor Cores paper studies the numeric behavior. [PTX], [SUN-TC]

---

## 7. Interconnect

### 7.1 Link bandwidths

NVIDIA quotes NVLink figures as the **sum of both directions**. Divide by 2 for one direction.

| Link | Total (both directions) | Per direction (D) | Tag | Source |
|---|---|---|---|---|
| PCIe Gen3 x16 | 31.5 GB/s | 15.75 | S | [WIKI-PCIE] |
| PCIe Gen4 x16 (A100 host link) | 63.0 GB/s (datasheet prints 64) | 31.5 (16 GT/s x 128/130 x 16 lanes / 8) | C datasheet; S per-direction | [A100-DS], [WIKI-PCIE] |
| PCIe Gen5 x16 (H100 host link) | 126.0 GB/s (NVIDIA prints 128, 64 each direction) | 63.0 | C; S | [HOPPER-WP], [WIKI-PCIE] |
| PCIe Gen6 x16 | about 242 GB/s (Wikipedia: 121 per direction) | 121 | S | [WIKI-PCIE] |
| NVLink 3 (A100): 12 links | 600 GB/s | 300 GB/s | C total; D per direction (12 x 50 GB/s per link per the 600 / 12) | [A100-DS] |
| A100 PCIe NVLink Bridge | 600 GB/s between exactly 2 GPUs | 300 | C | [A100-DS] |
| NVLink 4 (H100): 18 links | 900 GB/s | 450 | C | [HOPPER-WP], [KLOCKWOOD] |
| NVLink 5 (B200): 18 links | 1,800 GB/s ("18 links x 100 GB/s") | 900 | C | [BW-ULTRA-BLOG], [NVLINK-PAGE] |
| NVLink 6 (Vera Rubin) | 3,000 GB/s (36 links) | 1,500 | C | [NVLINK-PAGE] |
| NVLink Switch 4 (8-GPU) | 900 GB/s per GPU, 7.2 TB/s aggregate | | C | [NVLINK-PAGE] |
| NVLink Switch 5 | 1,800 GB/s per GPU; 130 TB/s across an NVL72 rack | | C | [NVLINK-PAGE] |
| HGX B200, 8 GPUs | 14.4 TB/s total NVLink; networking 0.8 TB/s | | C | [HGX-PAGE] |
| InfiniBand NDR (Quantum-2) | 400 Gb/s per port = 50 GB/s per direction (D: 400 / 8); 64 ports per switch; 51.2 Tb/s aggregate switch throughput | D; C for 400 Gb/s | [QUANTUM2] |
| ConnectX-7 | one or two ports of 400 Gb/s; PCIe Gen4 and Gen5 | C | [QUANTUM2] |
| Hopper DSMEM | measured SM-to-SM 3.27 TB/s (cluster size 2) | C | [LUO-HOPPER] |

The A100 80GB PCIe in this course has two interconnects of practical interest: PCIe Gen4 x16 (31.5 GB/s per direction) and, when two cards are bridged, NVLink 600 GB/s total. A single RunPod endpoint A100 has neither multi-GPU link available to the lab; chapter 21 and 22 labs use `torch.distributed` on one GPU or on CPU processes, or analytic models. (U for whether any lab GPU shares a bridge.)

### 7.2 The alpha-beta model and ring all-reduce

Hockney's model: the time to send an n-byte message over one link is

T = alpha + n / B

where alpha is the per-message latency (seconds) and B is the link bandwidth in one direction (bytes per second). Collective cost models add up messages. Source for the cost-model approach: [THAKUR].

**Ring all-reduce.** With p ranks and an array of n bytes, the ring runs a reduce-scatter (p - 1 steps) then an all-gather (p - 1 steps). Each step sends n / p bytes per rank. So:

T_ring = 2 (p - 1) alpha + 2 (p - 1) / p x n / B

- Each rank sends (and receives) 2 (p - 1) / p x n bytes in total. This is a lower bound for any all-reduce algorithm in a cluster of workstations, which the ring meets. [PATARASUK]
- Factor 2(p - 1)/p for p = 2, 4, 8, 16: 1.000, 1.500, 1.750, 1.875. D.
- As p grows the bandwidth term tends to 2n / B; the latency term grows linearly in p. That is why NCCL also has a tree algorithm for small messages and many ranks.
- Worked example (all D, assumptions stated): 1 GiB of gradients (1,073,741,824 B) all-reduced across 8 A100 SXM GPUs on NVSwitch at 300 GB/s per direction: bandwidth term = 1.75 x 1.0737e9 / 300e9 = 6.26 ms. The latency term 2 x 7 x alpha is 14 alpha; with alpha = 5 microseconds (U: the learn-inference chapter uses 5 to 10 microseconds per small all-reduce) it adds 70 microseconds.
- The same all-reduce over PCIe Gen4 at 31.5 GB/s per direction: 1.75 x 1.0737e9 / 31.5e9 = 59.6 ms. D.

**Cost of the other collectives (bandwidth term only, n = the full array size).** All-gather and reduce-scatter: (p - 1) / p x n / B each. Broadcast and reduce: about n / B at the root. All-to-all: (p - 1) / p x n / B per rank. D, consistent with the nccl-tests correction factors below. [NCCL-TESTS-PERF]

**NCCL bandwidth conventions (nccl-tests).** `algbw = S / t` where S is the data size. `busbw = algbw x factor`, so the number can be compared with the hardware link bandwidth regardless of rank count. Factors for n ranks: all-reduce 2(n - 1)/n; reduce-scatter (n - 1)/n; all-gather (n - 1)/n; broadcast 1; reduce 1; all-to-all (n - 1)/n. C. [NCCL-TESTS-PERF]

**NCCL algorithms and protocols.** Ring (bandwidth-optimal, linear latency), tree (lower latency at scale), plus NVLS (NVLink SHARP), CollNet, and PAT variants; protocols Simple, LL, and LL128. The Demystifying NCCL paper analyzes ring and tree and the three protocols (C); the NCCL index names NVLS and CollNet through environment variables, and says nothing about PAT (so PAT, NVLS, and CollNet behaviors are U in this fact sheet, stated only as names). NCCL selects an algorithm automatically; `NCCL_ALGO` and `NCCL_PROTO` force one. [NCCL-DEMYSTIFY], [NCCL-DOCS]

**Published NCCL bus bandwidth figures: none confirmed.** Reports of about 230 to 240 GB/s busbw for 8x A100 on NVSwitch and 450 to 480 GB/s for 8x H100 appear in forum threads and vendor blogs (S, unverified) but not in a primary document I could read. For the ceiling, use the link bandwidth per direction (300 GB/s A100, 450 GB/s H100); a measured busbw below that is normal. Writers should compute the ceiling and tell the reader to measure it. (U)

---

## 8. Other accelerators (for chapter 23 only)

Values below were read by a delegated research agent from the vendor pages named in the source keys. Dense unless marked.

### 8.1 AMD Instinct (CDNA)

| Property | MI300X | MI325X | MI350X | MI355X | Tag |
|---|---|---|---|---|---|
| Architecture | CDNA3 | CDNA3 | CDNA4 | CDNA4 | C |
| Compute units, XCDs | 304, 8 | 304, 8 | 256, 8 | 256, 8 | C |
| Matrix cores | 1,216 | 1,216 | 1,024 | 1,024 | C |
| Wavefront size | 64 | 64 | 64 | 64 | C |
| Peak engine clock | 2,100 MHz | 2,100 | 2,200 | 2,400 | C |
| HBM | 192 GB HBM3, 8 stacks | 256 GB HBM3E | 288 GB HBM3E | 288 GB HBM3E | C |
| HBM bandwidth | 5.3 TB/s | 6.0 TB/s | 8.0 TB/s | 8.0 TB/s | C |
| Infinity Cache | 256 MB | 256 MB | 256 MB | 256 MB | C |
| L2 | 4 MB per XCD (32 MB) | 32 MB | 32 MB | 32 MB | C |
| LDS per CU | 64 KB | 64 KB | 160 KB | 160 KB | C |
| VGPR file per CU | 512 KiB | 512 KiB | 512 KiB | 512 KiB | C (CDNA4 not re-checked) |
| TF32 matrix | 653.7 | 653.7 | not in hardware | not in hardware | C |
| FP16 and BF16 matrix | 1,307.4 | 1,307.4 | 2,300 | 2,500 | C |
| FP8 matrix | 2,614.9 | 2,614.9 | 4,600 | 5,000 | C |
| INT8 matrix (TOPS) | 2,614.9 | 2,614.9 | 4,600 | 5,000 | C |
| MXFP6 and MXFP4 matrix | none | none | 9,200 | 10,000 | C |
| FP64 matrix | 163.4 | 163.4 | 72.1 | 78.6 | C |
| TBP | 750 W | 1,000 W | 1,000 W | 1,400 W | C |
| Infinity Fabric | 7 x16 links, 896 GB/s peer ring | same | 1,075.2 GB/s peer ring | same | C |

Sparse figures (2x, 4:2 structured) are in the CDNA whitepapers. Per-CU per-clock matrix rates (CDNA3 / CDNA4 MI355X): FP16 and BF16 2,048 / 4,096; FP8 and INT8 4,096 / 8,192; FP64 256 / 128. MFMA example instructions: `V_MFMA_F32_32X32X8_F16` and `V_MFMA_F32_16X16X16_F16` (CDNA3); `V_MFMA_F32_16X16X32_F16` and `V_MFMA_F32_16X16X128_F8F6F4` (CDNA4). HIP: resident limits per CU on CDNA3 are 32 blocks, 32 wavefronts, 2,048 threads. Libraries: rocWMMA, Composable Kernel, hipBLASLt; HIPIFY translates CUDA to HIP. ROCm: the release page lists both 7.x (7.14.1, 2026-09-02) and 10.x (10.1.0, 2026-10-05) lines; the reason is U, so do not write a ROCm version number in prose. [AMD-CDNA3], [AMD-CDNA4], [AMD-DS355], [HIP-HW], [ROCM-VERSIONS]

### 8.2 Google TPU

| Property | v5e | v5p | v6e (Trillium) | TPU7x (Ironwood) | Tag |
|---|---|---|---|---|---|
| Peak BF16 per chip | 197 TFLOP/s | 459 | 918 | 2,307 | C |
| Peak FP8 per chip | n/a | 459 (listed; not independently confirmed as separate hardware) | 918 (same) | 4,614 | C / U |
| Peak INT8 per chip | 393 TOPS | n/a | 1,836 TOPS | n/a | C |
| HBM | 16 GB | 95 GiB | 32 GB | 192 GiB | C |
| HBM bandwidth | 800 GiB/s | 2,765 GB/s | 1,638 GB/s | 7,380 GB/s | C |
| ICI bandwidth per chip | 400 GB/s | 1,200 | 800 | 1,200 | C |
| Pod size | 256 chips | 8,960 | 256 | 9,216 | C |
| TensorCores per chip | 1 | 2 | 1 | 2 | C |
| MXU array size | 128x128 | 128x128 | 256x256 | 256x256 | C |

The v5e figure is in GiB per second and the others in GB per second; do not compare without converting. Pallas is "an extension to JAX for writing custom kernels for GPU and TPU", aimed at cases where XLA's automatic optimization falls short. XLA fuses pipelined operations to reduce memory overhead and eliminates intermediate buffers. [TPU-DOCS], [PALLAS], [XLA-ARCH]

### 8.3 Apple M-series GPU

| Chip | GPU cores | Memory bandwidth | Max memory | Source |
|---|---|---|---|---|
| M4 | 10 | 120 GB/s | 32 GB | [APPLE-SUPPORT] |
| M4 Pro | 16 or 20 | 273 GB/s | 64 GB | [APPLE-SUPPORT] |
| M4 Max | 32 or 40 | 410 or 546 GB/s | 128 GB | [APPLE-SUPPORT] |
| M5 | 10 (a Neural Accelerator in each GPU core) | 153 GB/s | 32 GB | [APPLE-M5] |
| M5 Max | 32 or 40 | 460 or 614 GB/s | 128 GB | [APPLE-M5] |
| M5 Ultra | 64 or 80 | 1.2 TB/s | 512 GB | [APPLE-M5] |

All C except FP32 TFLOP/s, which Apple does not state (U). SIMD-group width is 32 on all Apple GPUs (WWDC22 session; Metal Shading Language only says to query `threads_per_simdgroup`). Maximum threadgroup memory is 32 KB on Apple4 and later GPU families; which M-series chip sits in which family is U. `simdgroup_matrix` types exist in Metal Shading Language (8x8 half, bfloat, float). MLX is the NumPy-like array framework for Apple silicon. [APPLE-METAL], [APPLE-MSL], [MLX]

### 8.4 Tile DSLs and kernel languages (state as of 2026-10-22)

| Project | State | Source |
|---|---|---|
| CUTLASS and CuTe DSL | CUTLASS 4.8.1 (Oct 2026); CuTe DSL is the Python front end mirroring CuTe C++ layouts, tensors, and atoms | [CUTLASS-GH] |
| ThunderKittens | no tagged release; "ThunderKittens 2.0" news item dated 11 Jan 2026; NVIDIA Hopper and Blackwell tested; Ampere no longer actively supported | [TK-GH] |
| TileLang | v0.1.15 (30 Sep); CUDA, AMD HIP, Apple Metal, and Ascend backends | [TILELANG-GH] |
| Triton Gluon | lower-level layout and memory control front end inside Triton; tutorials for TMA, wgmma, tcgen05, warp specialization (described from file names, U as a definition) | [TRITON-GH] |
| HipKittens | paper arXiv 2511.08083 (Nov 2025); `main` targets MI350X and MI355X, branch `cdna3` targets MI300X and MI325X | [HIPKITTENS] |

---

## 9. Published benchmark numbers writers may cite

Always name the hardware. None of these are lab numbers.

| Result | Value | Hardware | Source |
|---|---|---|---|
| Boehm SGEMM worklog, kernel 1 naive | 309.0 GFLOP/s (1.3% of cuBLAS) | **RTX A6000** (84 SMs, 768 GB/s), 4092 x 4092 FP32 | [BOEHM] |
| kernel 2 global-memory coalescing | 1,986.5 GFLOP/s (8.5%) | same | [BOEHM] |
| kernel 3 shared-memory caching | 2,980.3 (12.8%) | same | [BOEHM] |
| kernel 4 1D blocktiling | 8,474.7 (36.5%) | same | [BOEHM] |
| kernel 5 2D blocktiling | 15,971.7 (68.7%) | same | [BOEHM] |
| kernel 6 vectorized memory access | 18,237.3 (78.4%) | same | [BOEHM] |
| kernel 9 autotuning | 19,721.0 (84.8%) | same | [BOEHM] |
| kernel 10 warptiling | 21,779.3 (93.7%) | same | [BOEHM] |
| cuBLAS | 23,249.6 (100%) | same | [BOEHM] |
| Boehm memory figures | minimum traffic 201 MB read plus 67 MB write = about 268 MB; naive worst case about 548 GB; naive 15 GB/s, coalesced 110 GB/s; cuBLAS about 500 MB moved; about 80% L2 hit rate in warptiling | same | [BOEHM] |
| Salykov SGEMM worklog | with unlocked clocks the 128x128x8 kernel averages 3 to 4% faster than cuBLAS at about 12% more power | RTX 3090 | [SALYKOV] |
| Harris parallel reduction, 4M elements | kernel 1: 8.054 ms (2.083 GB/s); kernel 2: 3.456 ms; kernel 3: 1.722 ms; kernel 4: 0.965 ms (17.4 GB/s); kernel 5: 0.536 ms (31.3 GB/s); kernel 6: 0.381 ms (44.0 GB/s); kernel 7: 0.268 ms (62.7 GB/s); 73 GB/s at 32M elements; G80 peak 86.4 GB/s | NVIDIA G80 (2007) | [HARRIS-REDUCE] |
| Harris speedup, kernel 1 to 7 | 8.054 / 0.268 = 30.05x | D | [HARRIS-REDUCE] |
| Merrill and Garland single-pass scan | moves about 2n data (n reads, n writes); "approaches that of copy operations"; matches memcpy ceiling on Kepler (K40) and Maxwell (M40) for large inputs; harmonic-mean speedups of 1.1x, 1.4x, and 2.3x over StreamScan, MGPU (reduce-then-scan, about 3n traffic), and Thrust (about 4n traffic) | Tesla M40, K40, C2050 | [MERRILL-GARLAND] |
| FlashAttention (v1) | about 15% end-to-end speedup on BERT-large (sequence 512) over the MLPerf 1.1 record; about 3x on GPT-2 (sequence 1K); about 2.4x on long-range arena (1K to 4K) | A100 (per the paper) | [FA1] |
| FlashAttention-2 | about 2x over FlashAttention; 50 to 73% of theoretical peak FLOP/s; up to 225 TFLOP/s per GPU end to end (72% model FLOPs utilization) | A100 | [FA2] |
| FlashAttention-3 | 1.5 to 2.0x over FlashAttention-2; up to about 740 TFLOP/s in FP16 (about 75% utilization); about 1.2 PFLOP/s in FP8; 2.6x lower FP8 numerical error than a baseline | H100 | [FA3] |
| FlashAttention-4 | up to 1.3x over cuDNN 9.13 and 2.7x over Triton; up to 1,613 TFLOP/s BF16 (71% utilization; the author's blog says 1,605) | B200 | [FA4], [FA4-BLOG] |
| MARLIN (W4A16) | close to the maximum 4x quantization speedup up to batch size 16 to 32; "gradually decreases" for batch 64 to 128; up to 2.8x end to end in vLLM | GPU not named in the abstract (U; paper evaluates Ampere GPUs, not verified) | [MARLIN] |
| Liger Kernel | about 20% higher training throughput and about 60% lower GPU memory versus Hugging Face implementations | hardware not stated in the abstract | [LIGER] |
| Cut Cross-Entropy | loss-computation memory from 24 GB to 1 MB; classifier-head training memory from 28 GB to 1 GB (Gemma 2 2B) | not stated in the abstract | [CCE] |
| ThunderKittens | matches cuBLAS on GEMM and FlashAttention-3 on attention inference; 10 to 40% faster than the strongest baselines on attention backward; 8x on state-space models; 14x on linear attention | H100 (not stated in the abstract, U) | [TK] |
| Triton matmul tutorial | grouped ordering: 9x9 blocks, 90 loads reduced to 54 for the first 9 output blocks; "220 to 245 TFLOPS on A100" with grouping; 4096^3 FP16: cuBLAS 218.95 TFLOP/s versus Triton 219.67 | A100 named in the text; the table does not name its GPU (U) | [TRITON-MM] |
| Triton persistent matmul tutorial | M = N = 8192, K = 512: cuBLAS 176.9, naive Triton 175.7, persistent Triton 171.2 TFLOP/s | GPU not named (U) | [TRITON-PERSIST] |
| CUTLASS README sample | tensor-core GEMM FP16, 3456 x 4096 x 4096: 74,378.8 GFLOP/s; CUDA-core SGEMM: 17,218.4 GFLOP/s | GPU not named (U) | [CUTLASS-GH] |
| A100 `mma.sync` measured | up to 310.6 TFLOP/s FP16 (99.6% of 312) | A100 PCIe 40GB | [LUO-HOPPER] |
| H800 PCIe `wgmma` measured | 729 TFLOP/s FP16 (96% of 756.5); 1,448 FP8 (96% of 1,513) | H800 PCIe | [LUO-HOPPER] |
| NVIDIA CUDA graphs example | 9.6 us per kernel (sync each launch), 3.8 us (overlapped), 3.4 us in graph of 20 | V100 | [CUDA-GRAPHS-BLOG] |
| PyTorch 2:4 sparsity | up to 1.6x matmul; 1.3x ViT-L MLP forward plus backward | A100 | [PT-SPARSE-BLOG] |
| cuBLAS or CUTLASS percent of peak | **No primary percent-of-peak figure for cuBLAS or CUTLASS GEMM on A100 was found.** The CUTLASS README makes only the qualitative claim "nearly optimal utilization". For an A100 anchor use the `mma.sync` measurement above (97 to 99.6% of peak for the instruction alone) and the Triton tutorial range (220 to 245 TFLOP/s, 71 to 79% of 312). | | U |
| NCCL bus bandwidth | none confirmed from a primary source (see section 7.2) | | U |

Boehm's worklog ran on a **consumer RTX A6000 (GA102, sm_86)**, not an A100. FP32 SGEMM on the lab A100 peaks at 19.5 TFLOP/s, below the A6000's quoted 30 to 38.7 TFLOP/s (the page gives both), so absolute GFLOP/s do not transfer. The percentages of cuBLAS are the transferable part. Say so in chapter 10.

---

## 10. Measured on the lab A100 (from learn-inference)

Every row was measured on an A100 80GB PCIe by the learn-inference course (the provider may supply the SXM4 variant on some runs; treat differences under 5% as noise, per chapter 10). Writers may reuse these as "measured on the course's lab GPU". Paths are under `/Users/kashifulhaque/Documents/test/vm/learn-inference/content/chapters/`.

| Measurement | Value | Chapter file | Notes |
|---|---|---|---|
| Plain device-to-device copy bandwidth | **1,275 GB/s** (66% of the 1,935 GB/s rating) | `10-roofline.md`, "Why measured bandwidth is 66% of rated" | Counts both read and write: 2N / t. 42 MB in and 42 MB out in 0.066 ms (`13-fusion-in-triton.md`). |
| Ridge point against the measured copy | 312e12 / 1.275e12 = **244.7** FLOP/byte | `10-roofline.md` | D from the above |
| Hand-written CUDA vector add | **1,304 GB/s** (counts 3n x 4 bytes: two reads, one write) | `12-your-first-cuda-kernel.md`, `10-roofline.md` | n = 2^24 float32 |
| Strided copy, stride 32 floats | **189 GB/s** (counts 2n x 4 bytes) | `12-your-first-cuda-kernel.md` | n = 2^24; read footprint 16.8 MB, which fits in the 40 MB L2 |
| Coalesced versus strided | 1,304 / 189 = **6.9x** | `12-your-first-cuda-kernel.md`, `00a-notation-and-prerequisites.md` | decomposed: 4.5 (bytes) x 1.53 (extra request cost) = 6.9 |
| Expected strided bandwidth from bytes alone | 1,304 / 4.5 = 290 GB/s | `12-your-first-cuda-kernel.md` | D |
| Kernel queue time, host side | about **5 microseconds** per launch | `12-your-first-cuda-kernel.md` | learn-inference rule of thumb: 5 to 10 microseconds |
| Fused RMSNorm (single block, `num_warps=8`), 4096 rows x 5120, bf16 | **0.089 ms, 945 GB/s** (74% of the copy, 49% of the rating) | `13-fusion-in-triton.md`, `10-roofline.md` | 41.9 MB tensor |
| Tiled RMSNorm, `BLOCK=2048` | **0.090 ms, 929 GB/s** | `13-fusion-in-triton.md` | within 1% despite re-reading each row, because the row stays in cache |
| `num_warps` sweep, 5120-wide rows | 8: 0.0888 ms (945 GB/s); 16: 0.0908 ms (924); 32: 0.0955 ms (878) | `13-fusion-in-triton.md` | |
| Fusion speedup, 4096 rows (41.9 MB) | **0.92x** (fused slower) | `13-fusion-in-triton.md` | the byte count predicts 1.25x |
| Fusion speedup, 16384 rows (167.8 MB) | **1.10x** | `13-fusion-in-triton.md` | |
| Fusion sweep (rows, size, fused, unfused, speedup) | 256: 2.6 MB, 0.072 / 0.074 ms, 1.02x; 1024: 10.5 MB, 0.081 / 0.074, 0.91x; 2048: 21.0 MB, 0.107 / 0.090, 0.84x; 4096: 41.9 MB, 0.155 / 0.148, 0.95x; 8192: 83.9 MB, 0.252 / 0.263, 1.05x; 16384: 167.8 MB, 0.444 / 0.495, 1.12x; 32768: 335.5 MB, 0.823 / 0.958, 1.16x | `13-fusion-in-triton.md` | The crossover is near the 40 MB L2. Run-to-run difference of a few percent (0.95x versus 0.92x). |
| FlashAttention in Triton, causal, L = 8192, 8 heads, d = 128, tile 128 x 64 | **1.64 ms**; with tile 64 x 64: **3.35 ms** | `14-flash-attention.md` | from the engine docstring |
| Same kernel as a fraction of the vendor kernel | **0.63x** at 8k tokens (vendor about 1.03 ms, D) | `14-flash-attention.md` | |
| FlashAttention FLOP count and floor, same geometry | 4 x 8192^2 x 128 = 34.4 GFLOP per head, halved by causality to 17.2, x 8 heads = 137 GFLOP; floor 137e9 / 312e12 = **0.44 ms**; the Triton kernel reaches 137e9 / 1.64e-3 = 83.5 TFLOP/s = 27% of peak; vendor 133 TFLOP/s = 43% (D) | `14-flash-attention.md` | |
| FlashAttention peak memory versus naive, L = 4096 | **27.5x less** | `14-flash-attention.md` | |
| Lab thresholds | RMSNorm 60% of the copy ceiling (chapter 13); flash attention at least 0.33x of vendor (chapter 14); vector add at least 60% of 1,275 GB/s (chapter 12) | respective lab docs | |
| Host-to-device PCIe copy | about **21 GB/s** (a planning number) | `15-paged-attention.md` | 21 / 31.5 = 67% of the Gen4 x16 per-direction ceiling (D); the chapter calls it "roughly 60 times slower than HBM" |
| NVLink and PCIe numbers used in chapter 19 | 600 GB/s NVLink and 64 GB/s PCIe | `19-tensor-parallelism.md` | These are **total (both-direction)** figures and apply to a two-GPU bridged pair. See section 11. |
| Per-all-reduce latency assumed in chapter 19 | about 8 microseconds (5 to 10 range) | `19-tensor-parallelism.md` | an assumption, not a measurement |

The learn-inference decode tables (tokens per second at each batch size, chapter 16; the 42.2 ms per-token bf16 floor, chapters 18 and 19) are arithmetic on the 1,275 GB/s figure, not measurements.

---

## 11. Contradictions, ambiguities, and unconfirmed values

Read this before writing. Each item says what to do.

1. **Measured copy bandwidth versus the datasheet's "95% DRAM utilization".** The A100 datasheet text claims a "DRAM utilization efficiency of 95%" for HBM2e [A100-DS]. The lab's copy reaches 66% of the rating (1,275 of 1,935 GB/s) and the vector add 67% (1,304). The Luo paper reached about 90% of theoretical on a 40GB PCIe A100 with its throughput test. **No source explains the gap.** Plausible causes (U): the read-plus-write mix and bus turnaround, ECC overhead, a power-limited PCIe card, or too-small test sizes. Do not claim any one cause. In chapter 03, have the lab sweep size from 64 MB to 4 GiB, report the plateau, and call 1,275 GB/s "the lab number measured in learn-inference at 42 MB per direction".

2. **Cache-hierarchy latencies.** The learn-inference table gives about 30 (shared), 200 (L2), 400 (HBM) cycles; Luo et al. measured 29.0, 261.5, 466.3 on a 40GB A100 PCIe. Use the measured values.

3. **A100 L2 bandwidth.** Whitepaper theoretical read bandwidth is 5,120 B/clk (7.2 TB/s at 1.41 GHz). Luo's microbenchmark measured about 1,850 to 2,010 B/clk (2.6 to 2.8 TB/s) on an A100 PCIe. Chips and Cheese measured 4.5 TB/s on an A100 (SXM). Different parts, different methods. If a chapter quotes L2 bandwidth, cite the paper and the method.

4. **NVLink and PCIe "bandwidth" are bidirectional totals.** The datasheet's "NVLink 600 GB/s" and "PCIe Gen4 64 GB/s" are totals for both directions; per direction they are 300 and 31.5 GB/s (Gen4 x16: 16 GT/s x 128/130 x 16 / 8 = 31.5). Learn-inference chapter 19 divides data by 600 and 64 as if they were one-directional; the ring all-reduce formula needs the per-direction value B. Chapter 21 must state which one it uses. The A100 PCIe NVLink bridge connects exactly two GPUs.

5. **H100 and B200 peak figures depend on the document and power SKU.** The Hopper whitepaper prints preliminary numbers (FP16 dense 1,000 TFLOP/s, 3 TB/s HBM); the final product page gives 989.4 dense (1,979 sparse) and 3.35 TB/s. Use the final numbers. For B200, HGX B200 (about 1,000 W) divides to 9 PFLOP/s dense FP4, 4.5 FP8, 2.25 BF16; the Blackwell Ultra architecture blog lists Blackwell at 10 dense NVFP4 and 5 dense FP8 PFLOP/s at up to 1,200 W. Use the HGX values in tables and mention the second as the maximum-power SKU. B200 memory is 180 GB (HGX, cloud consoles) or 192 GB (architecture blog); both are right for different SKUs.

6. **H100 clocks.** The final rated H100 SXM FP32 and FP64 figures imply 1.98 GHz; the tensor-core figures imply 1.83 GHz (D). NVIDIA does not publish either in a document I read. Quote only the TFLOP/s values; if a chapter must show a clock, say "implied by NVIDIA's rated throughput".

7. **A100 PCIe versus SXM.** The datasheet lists identical compute peaks for both but 300 W versus 400 W. 1,935 versus 2,039 GB/s is a 5.4% difference. The provider may hand either 80GB form; the lab prints `torch.cuda.get_device_name()` to say which.

8. **H100 NVL.** The product page lists an NVL part (94 GB, 3.9 TB/s, 835 TFLOP/s TF32 with sparsity). Not used in this course except as a footnote.

9. **tf32 rounding path** (section 6) differs by instruction. State that explicitly.

10. **Triton block pointers** are deprecated (3.7.0) but present (3.8.0), while the official reference no longer documents them. The chapter 14 lab, which the curriculum defines around block pointers, will emit a deprecation warning. **Open for the research lead and the chapter 14 writer to decide**: keep block pointers with a warning, or switch the lab to plain pointers (the official tutorial) with descriptors in prose. Whether `make_tensor_descriptor` runs on sm_80 is U.

11. **Boehm numbers are from an RTX A6000.** Do not present them as A100 results.

12. **Marlin, Liger, ThunderKittens, and Cut Cross-Entropy abstracts do not name the GPU.** Cite only the number and say "reported by the authors". The Marlin paper's hardware is U here.

13. **Occupancy granularity** (256 registers per warp, allocation in groups of 4 warps, 128-byte shared memory units) is common knowledge from the retired occupancy-calculator spreadsheet, but the only supporting text found was a developer-forum post (S). Labs should use `cudaOccupancyMaxActiveBlocksPerMultiprocessor` or Nsight Compute for ground truth and teach the formula as an approximation.

14. **CC 7.0 limits** are not in the retrieved table (U). Chapter 02's per-architecture table starts at 7.5.

15. **Measured versus rated numbers for the A100 `mma.sync`.** 310.6 TFLOP/s is 99.6% of 312, but this is a single-instruction loop on a 40GB PCIe card at 1410 MHz. Real GEMM kernels reach less (the Triton tutorial: 220 to 245 TFLOP/s).

16. **NCCL**: published busbw numbers, NVLS, CollNet, and PAT details are U. Compute the ceiling; do not cite a measured busbw.

17. **ROCm version naming** (7.x and 10.x both listed in Oct 2026) is U.

18. **Hopper whitepaper mirror.** The original `resources.nvidia.com` whitepaper link redirects to the NVIDIA home page; the document is read from a third-party mirror (same file). Confirm numbers against the H100 product page.

19. **learn-inference wording.** Chapter 13 calls SMs "processor cores" ("108 processor cores"); chapter 00a says "L2: hundreds of cycles". Do not copy either phrasing.

20. **A100 80GB PCIe boost clock.** The datasheet does not print clocks; 1,410 MHz is from the whitepaper (SXM 40GB) and the Luo paper (PCIe 40GB). The 80GB PCIe card is U for sustained clock; its rated peaks match 1,410 MHz.

---

## 12. Source keys

Primary and secondary sources used above. Each URL was fetched on 2026-10-22 unless noted.

| Key | Source | URL |
|---|---|---|
| A100-DS | NVIDIA A100 Tensor Core GPU datasheet (Jun 2021) | https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf |
| GA100-WP | NVIDIA A100 Tensor Core GPU Architecture whitepaper (2020) | https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf |
| HOPPER-WP | NVIDIA H100 Tensor Core GPU Architecture whitepaper v1.x (preliminary figures; third-party mirror of the NVIDIA PDF) | https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf |
| HOPPER-BLOG | NVIDIA Hopper Architecture In-Depth | https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/ |
| H100-PAGE | NVIDIA H100 product page, specification table (values with sparsity) | https://www.nvidia.com/en-us/data-center/h100/ |
| HGX-PAGE | NVIDIA HGX platform page (HGX B200 and B300, 8-GPU totals) | https://www.nvidia.com/en-us/data-center/hgx/ |
| BW-ULTRA-BLOG | Inside NVIDIA Blackwell Ultra: The Chip Powering the AI Factory Era | https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/ |
| NVLINK-PAGE | NVIDIA NVLink and NVLink Switch | https://www.nvidia.com/en-us/data-center/nvlink/ |
| QUANTUM2 | NVIDIA Quantum-2 InfiniBand | https://www.nvidia.com/en-us/networking/quantum2/ |
| CPG-CC | CUDA Programming Guide, compute capabilities appendix (read via the .md rendering) | https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html |
| CPG-ITS | CUDA Programming Guide, advanced kernel programming (independent thread scheduling) | https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html |
| CPG-PROG | CUDA Programming Guide, programming model | https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html |
| CPG-LANG | CUDA Programming Guide, C++ language extensions (warp matrix functions, `__syncwarp`, clusters) | https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html |
| CPG-BANKS | CUDA Programming Guide (archive 12.9), shared memory banks | https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html |
| CPG-118 | CUDA C Programming Guide archive 11.8 (cache line 128 bytes, 64-bit bank mode for CC 3.x) | https://docs.nvidia.com/cuda/archive/11.8.0/cuda-c-programming-guide/index.html |
| BPG | CUDA C++ Best Practices Guide | https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html |
| HOPPER-TUNING | CUDA Hopper Tuning Guide | https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html |
| BW-TUNING | CUDA Blackwell Tuning Guide | https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html |
| PTX | PTX ISA 9.4 (read by the research agent) | https://docs.nvidia.com/cuda/parallel-thread-execution/index.html |
| CUDA-RN | CUDA Toolkit release notes | https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html |
| NCU-GUIDE | Nsight Compute Profiling Guide | https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html |
| NV-MATMUL | NVIDIA Matrix Multiplication Background User's Guide | https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html |
| CUDA-GRAPHS-BLOG | Getting Started with CUDA Graphs (NVIDIA) | https://developer.nvidia.com/blog/cuda-graphs/ |
| HARRIS-TRANSPOSE | An Efficient Matrix Transpose in CUDA C/C++ | https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/ |
| TCGEN05-GUIDE | CUTLASS CuTe DSL, tcgen05 MMA Programming Guide (4.6.0) | https://docs.nvidia.com/cutlass/4.6.0/media/docs/pythonDSL/mma_docs/tcgen05_programming.html |
| CUBLAS-DOC | cuBLAS documentation (block scaling) | https://docs.nvidia.com/cuda/cublas/index.html |
| CUTLASS-GH | NVIDIA CUTLASS repository | https://github.com/NVIDIA/cutlass |
| NCCL-DOCS | NCCL documentation (2.32.3) | https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html |
| NCCL-TESTS-PERF | nccl-tests PERFORMANCE.md | https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md |
| NCCL-DEMYSTIFY | Demystifying NCCL (arXiv 2507.04786) | https://arxiv.org/abs/2507.04786 |
| TF32-BLOG | Accelerating AI Training with NVIDIA TF32 Tensor Cores | https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/ |
| NVFP4-BLOG | Introducing NVFP4 for Efficient and Accurate Low-Precision Inference | https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/ |
| OFP8 | OCP 8-bit Floating Point Specification (OFP8) Rev 1.0 | https://opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf |
| MX-SPEC | OCP Microscaling Formats (MX) v1.0 | https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf |
| LUO-HOPPER | Luo et al., Benchmarking and Dissecting the Nvidia Hopper GPU Architecture (arXiv 2402.13499) | https://arxiv.org/abs/2402.13499 |
| SUN-TC | Sun et al., Dissecting Tensor Cores via Microbenchmarks (arXiv 2206.02874) | https://arxiv.org/abs/2206.02874 |
| JIA-VOLTA | Jia et al., Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking (arXiv 1804.06826) | https://arxiv.org/abs/1804.06826 |
| MISHRA | Mishra et al., Accelerating Sparse Deep Neural Networks (arXiv 2104.08378) | https://arxiv.org/abs/2104.08378 |
| CNC-B200 | Chips and Cheese, Nvidia's B200 | https://chipsandcheese.com/p/nvidias-b200-keeping-the-cuda-juggernaut |
| CNC-H100 | Chips and Cheese, Nvidia's H100: Funny L2 | https://chipsandcheese.com/p/nvidias-h100-funny-l2-and-tons-of-bandwidth |
| KLOCKWOOD | Glenn Klockwood, H100 page (secondary) | https://glennklockwood.com/garden/processors/H100 |
| FLOPPER | flopper.io H100 PCIe versus SXM5 spec sheet (secondary) | https://flopper.io/compare/nvidia-h100-sxm5-80gb-vs-nvidia-h100-pcie-80gb/spec-sheet |
| FORUM-OCC | NVIDIA developer forum, occupancy summary table (secondary) | https://forums.developer.nvidia.com/t/optimisation-of-occupancy-summary-table/265684 |
| VOLKOV-GTC | Volkov, Better Performance at Lower Occupancy (GTC 2010) | https://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf |
| FA4-BLOG | Tri Dao, FlashAttention-4 blog | https://tridao.me/blog/2026/flash4/ |
| FA1 | Dao et al., FlashAttention (arXiv 2205.14135) | https://arxiv.org/abs/2205.14135 |
| FA2 | Dao, FlashAttention-2 (arXiv 2307.08691) | https://arxiv.org/abs/2307.08691 |
| FA3 | Shah et al., FlashAttention-3 (arXiv 2407.08608) | https://arxiv.org/abs/2407.08608 |
| FA4 | Zadouri et al., FlashAttention-4 (arXiv 2603.05451) | https://arxiv.org/abs/2603.05451 |
| MARLIN | Frantar et al., MARLIN (arXiv 2408.11743) | https://arxiv.org/abs/2408.11743 |
| LIGER | Hsu et al., Liger Kernel (arXiv 2410.10989) | https://arxiv.org/abs/2410.10989 |
| CCE | Wijmans et al., Cut Your Losses in Large-Vocabulary Language Models (arXiv 2411.09009) | https://arxiv.org/abs/2411.09009 |
| TK | Spector et al., ThunderKittens (arXiv 2410.20399) | https://arxiv.org/abs/2410.20399 |
| MERRILL-GARLAND | Merrill and Garland, Single-pass Parallel Prefix Scan with Decoupled Look-back (NVR-2016-002) | https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf |
| HARRIS-REDUCE | Harris, Optimizing Parallel Reduction in CUDA | https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf |
| BOEHM | Boehm, How to Optimize a CUDA Matmul Kernel (worklog) | https://siboehm.com/articles/22/CUDA-MMM |
| SALYKOV | Salykov, Advanced Matrix Multiplication Optimization on NVIDIA GPUs | https://salykova.github.io/sgemm-gpu |
| PATARASUK | Patarasuk and Yuan, Bandwidth Optimal All-reduce Algorithms (JPDC 2009) | https://www.cs.fsu.edu/~xyuan/paper/09jpdc.pdf |
| THAKUR | Thakur, Rabenseifner, Gropp, Optimization of Collective Communication Operations in MPICH (2005) | https://www.cs.umd.edu/class/fall2019/cmsc714/readings/Thakur-MPICH.pdf |
| WIKI-PCIE | Wikipedia, PCI Express comparison table | https://en.wikipedia.org/wiki/PCI_Express |
| PT-RELEASES | PyTorch GitHub releases | https://github.com/pytorch/pytorch/releases |
| PT-2140 | PyTorch 2.14.0 release notes | https://github.com/pytorch/pytorch/releases/tag/v2.14.0 |
| PT-PYPI | PyPI metadata for torch 2.14.1 | https://pypi.org/pypi/torch/json |
| PT-CPPEXT | PyTorch 2.14, torch.utils.cpp_extension | https://docs.pytorch.org/docs/2.14/cpp_extension.html |
| PT-LIBRARY | PyTorch 2.14, torch.library | https://docs.pytorch.org/docs/2.14/library.html |
| PT-LCE | PyTorch 2.14, torch.nn.functional.linear_cross_entropy | https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.linear_cross_entropy.html |
| PT-LCEOPT | PyTorch 2.14, LinearCrossEntropyOptions | https://docs.pytorch.org/docs/2.14/generated/torch.nn.LinearCrossEntropyOptions.html |
| PT-CUDA-NOTES | PyTorch 2.14, CUDA semantics (TF32 flags) | https://docs.pytorch.org/docs/2.14/notes/cuda.html |
| PT-DTYPES | PyTorch 2.14, tensor attributes (dtypes) | https://docs.pytorch.org/docs/2.14/tensor_attributes.html |
| PT-DTYPES-RPT | PyTorch v2.14.0 source, `ScaledBlas.cpp` and `Blas.cpp` (read by the research agent) | https://github.com/pytorch/pytorch/tree/v2.14.0/aten/src/ATen/native/cuda |
| PT-SPARSE | PyTorch 2.14, torch.sparse (semi-structured) | https://docs.pytorch.org/docs/2.14/sparse.html |
| PT-SPARSE-TUT | PyTorch tutorial, semi-structured sparsity | https://docs.pytorch.org/tutorials/advanced/semi_structured_sparse.html |
| PT-SPARSE-BLOG | PyTorch blog, Accelerating Neural Network Training with 2:4 sparsity | https://pytorch.org/blog/accelerating-neural-network-training/ |
| PT-CUDAGRAPH-BLOG | PyTorch blog, Accelerating PyTorch with CUDA Graphs | https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/ |
| TRITON-REL | Triton GitHub releases | https://github.com/triton-lang/triton/releases |
| TRITON-PYPI | PyPI metadata for triton | https://pypi.org/pypi/triton/json |
| TRITON-LANG | Triton language API index | https://triton-lang.org/main/python-api/triton.language.html |
| TRITON-MKDESC | triton.language.make_tensor_descriptor | https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html |
| TRITON-DOTSCALED | triton.language.dot_scaled | https://triton-lang.org/main/python-api/generated/triton.language.dot_scaled.html |
| TRITON-MM | Triton matmul tutorial 03 | https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html |
| TRITON-PERSIST | Triton persistent matmul tutorial 09 | https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html |
| TRITON-GH | Triton repository | https://github.com/triton-lang/triton |
| TK-GH | ThunderKittens repository | https://github.com/HazyResearch/ThunderKittens |
| TILELANG-GH | TileLang repository | https://github.com/tile-ai/tilelang |
| HIPKITTENS | HipKittens (arXiv 2511.08083) | https://arxiv.org/abs/2511.08083 |
| AMD-CDNA3 | AMD CDNA 3 whitepaper | https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf |
| AMD-CDNA4 | AMD CDNA 4 whitepaper | https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf |
| AMD-DS355 | AMD Instinct MI355X brochure | https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi355x-gpu-brochure.pdf |
| HIP-HW | ROCm HIP hardware features | https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hardware_features.html |
| ROCM-VERSIONS | ROCm release versions | https://rocm.docs.amd.com/en/latest/release/versions.html |
| TPU-DOCS | Google Cloud TPU documentation (TPU7x, v6e, v5p, v5e pages; architecture page) | https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm |
| PALLAS | Pallas documentation | https://docs.jax.dev/en/latest/pallas/index.html |
| XLA-ARCH | OpenXLA architecture | https://openxla.org/xla/architecture |
| APPLE-SUPPORT | Apple support spec pages for M4, M4 Pro, M4 Max | https://support.apple.com/en-us/121553 |
| APPLE-M5 | Apple newsroom, M5 | https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/ |
| APPLE-METAL | Metal Feature Set Tables | https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf |
| APPLE-MSL | Metal Shading Language Specification | https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf |
| MLX | MLX repository | https://github.com/ml-explore/mlx |
