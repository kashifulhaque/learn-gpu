---
title: Beyond CUDA
slug: 23-beyond-cuda
part: "Part 6 — Beyond"
summary: What the course taught, mapped onto AMD Instinct GPUs, Google TPUs, Apple silicon, and the tile-level languages that try to span them, ending with the three questions to ask of any kernel.
minutes: 75
gpu: false
objectives:
  - "Translate the course's vocabulary (SM, warp, shared memory, tensor core) into AMD, TPU, and Apple terms, and say which ideas carry over unchanged."
  - "Compute a ridge point and a decode-time floor for any accelerator from its peak matrix rate and memory bandwidth, and explain why the ridge points of current parts cluster between about 160 and 560 FLOP per byte."
  - "Find the code that breaks when a kernel moves from 32-wide warps to 64-wide wavefronts, and fix a warp reduction so it works on both."
  - "Describe how a TPU's systolic array and the XLA compiler change what a kernel author controls, and read a small Pallas kernel."
  - "Explain what unified memory removes and what it doesn't on Apple silicon, and say what Metal and MLX are for."
  - "Say what a tile-level language (Triton, CuTe DSL, ThunderKittens, TileLang, HipKittens) abstracts, and what it leaves to you."
  - "Choose what to keep hardware-agnostic in a kernel project, and apply the three questions (what bounds it, how far from the bound, what moves it) to a kernel on any machine."
---

# Beyond CUDA

> [!TLDR]
> - Almost everything in this course is a statement about memory, arithmetic, and parallelism, not about NVIDIA. The roofline, coalescing, tiling, pipelining, fusion, and quantization all carry over. What changes is the constants: warp width, scratchpad size, matrix instruction shapes, and which number formats exist.
> - AMD's Instinct GPUs (CDNA 3 and 4) are the closest cousin: compute units for SMs, 64-wide wavefronts for 32-wide warps, the LDS for shared memory, MFMA for tensor core instructions. The classic trap is a hard-coded 32.
> - A TPU is a different shape of machine: large systolic arrays fed by a compiler (XLA) that sees the whole program. You write kernels with Pallas only where the compiler falls short.
> - Apple silicon trades bandwidth for unified memory: no copy across PCIe, up to 128 GB visible to the GPU on a laptop chip, and 120 to 614 GB/s of bandwidth on the parts listed here.
> - Tile-level languages are converging on one idea: you name the tile and the compiler or library chooses the instructions. Ask three questions of any kernel on any machine: what bounds it, how far is it from the bound, and what moves it.

You have spent twenty-two chapters on one machine, an A100, and one vendor's vocabulary. That was deliberate: you learn a machine's reasoning by working through one in detail. But the systems you meet at work may not be NVIDIA's. Training clusters run on AMD Instinct GPUs and on Google TPUs, and a large share of local model development happens on a laptop with an Apple chip. A reader who can only recite CUDA terms is stuck the first time the warp size changes. A reader who can reason from bytes and FLOP is not.

This chapter is a survey, and it is honest about its limits. The lab A100 is the only machine in this course that you have measured, so every figure here is a published rating or arithmetic on one, and the chapter says so. You will learn what to look up, what to expect, and what to distrust when you move a kernel between machines.

The chapter has two parts. The first four sections take each family of hardware in turn and say what transfers. The last three step back: the pull toward tile-level programming, a strategy for portability, and a review of the whole course through three questions.

## Before you start

This chapter leans on the vocabulary of the earlier chapters. Here is a quick refresher of the terms it uses most.

**Warps.** A GPU runs threads in lockstep groups. NVIDIA calls a group of 32 a *warp* ([the execution model](/c/02-the-execution-model)).

**Ridge point.** The ridge point is the peak arithmetic rate divided by the memory bandwidth, in FLOP per byte. A kernel with fewer FLOP per byte than the ridge point is memory bound ([the roofline chapter](/c/04-roofline-and-latency)). On the lab A100 it is 312 TFLOP/s over 1,935 GB/s, which is 161 FLOP per byte.

**Tensor core.** A tensor core is a hardware unit that multiplies small matrices in one instruction ([tensor cores](/c/11-tensor-cores)). Every vendor has one under a different name.

**Tile.** A tile is a rectangular block of a matrix that one thread block, or one Triton program, owns and reuses ([GEMM from scratch](/c/10-gemm-from-scratch) and [Triton](/c/14-triton)).

**Dense and sparse peaks.** Vendors print both. This course quotes dense figures ([MoE and sparse kernels](/c/20-moe-and-sparse-kernels)).

## What carries over, and what doesn't

Most of the course carries over unchanged, and the exceptions are specific. This section answers one question: what do you have to relearn on a new machine?

Start with what is true of every machine in this chapter. Each one has a large, slow, high-bandwidth memory, a small, fast, software-managed or cache-managed memory near the arithmetic, and a matrix unit that does far more FLOP per cycle than the general-purpose vector units. The ratio of the matrix unit's rate to the memory's rate, the ridge point, is high on all of them. So the central lesson of the course holds everywhere: ==most of a model's kernels are limited by bytes, and the ones that aren't need data reuse in the fast memory to stay fed==.

The following table maps the vocabulary. It is a guide to the idea, not a claim that the units are interchangeable.

| Idea | NVIDIA (this course) | AMD Instinct (CDNA) | Google TPU | Apple silicon |
|---|---|---|---|---|
| Parallel core | Streaming multiprocessor (SM) | Compute unit (CU) | TensorCore | GPU core |
| Lockstep thread group | Warp, 32 lanes | Wavefront, 64 lanes | None: a compiler schedules vector ops | SIMD-group, 32 lanes |
| Programmer-managed scratchpad | Shared memory | Local data share (LDS) | On-chip memory the compiler or Pallas manages | Threadgroup memory |
| Matrix unit | Tensor core, `mma` and `wgmma` | Matrix core, MFMA instructions | Matrix multiply unit (MXU), a systolic array | `simdgroup_matrix` (8 by 8) |
| Kernel language | CUDA C++, PTX | HIP, which mirrors CUDA | XLA through JAX, plus Pallas | Metal Shading Language |
| Main framework entry | PyTorch, Triton | PyTorch on ROCm, Triton | JAX | MLX, PyTorch's Metal backend |

The sources for each cell are in the sections below. Notice two things. First, the AMD column is a near-direct rename, which is why porting is plausible. Second, the TPU column has no entry for the warp, because a TPU doesn't expose a thread-level programming model at all.

What does change are four kinds of constant. They are worth listing now, because each one is a way a kernel silently goes wrong.

1. **Lockstep width.** 32 on NVIDIA and Apple, 64 on the AMD Instinct parts below.
2. **Scratchpad size.** The A100 allows up to 163 KB of shared memory per block ([the GPU at a glance](/c/01-the-gpu-at-a-glance)). The MI300X has 64 KB of LDS per CU, and the MI355X has 160 KB.
3. **Matrix instruction shapes and formats.** The tile an instruction multiplies, and the formats it accepts, differ. CDNA 4 drops TF32 matrix support from the hardware, for instance, and adds MXFP6 and MXFP4.
4. **The memory-to-compute ratio.** It ranges from 160 to 560 FLOP per byte across the parts below, which moves the break-even point of every optimization.

## The same roofline on ten machines

Every machine in this chapter fits the same two-number model, and the numbers differ by a small factor. This section answers one question: how do the ridge points compare?

Divide the peak dense matrix rate by the memory bandwidth:

$$
I^{\ast} = \frac{\hla{P}}{\hlb{\beta}}
$$

Here $\hla{P}$ is the peak dense BF16 or FP16 matrix rate and $\hlb{\beta}$ is the HBM bandwidth. The result $I^{\ast}$ is the FLOP per byte a kernel needs before arithmetic, not memory, is the limit. The figure lets you switch between the ridge point, the time to read 14 GB of weights once, and the raw peak.

```viz
23-accelerator-ridge
```

The following table shows the arithmetic. Every rate and bandwidth is a vendor rating from the source listed. The A100 row is the lab card, and the 14 GB column is the decode-time floor for a 7-billion-parameter bfloat16 model, the same arithmetic as in [the first chapter](/c/00-what-you-are-building).

| Part | Peak BF16 or FP16 (TFLOP/s) | HBM bandwidth (GB/s) | Ridge point (FLOP/B) | Read 14 GB once (ms) |
|---|---|---|---|---|
| A100 80GB PCIe | 312 | 1,935 | 161 | 7.24 |
| H100 SXM5 | 989.4 | 3,350 | 295 | 4.18 |
| B200 (HGX, per GPU) | 2,250 | 8,000 | 281 | 1.75 |
| MI300X | 1,307.4 | 5,300 | 247 | 2.64 |
| MI325X | 1,307.4 | 6,000 | 218 | 2.33 |
| MI355X | 2,500 | 8,000 | 313 | 1.75 |
| TPU v5e | 197 | 859 (800 GiB/s) | 229 | 16.3 |
| TPU v5p | 459 | 2,765 | 166 | 5.06 |
| TPU v6e (Trillium) | 918 | 1,638 | 560 | 8.55 |
| TPU7x (Ironwood) | 2,307 | 7,380 | 313 | 1.90 |

Check one row by hand. For the MI300X, 1,307.4 TFLOP/s over 5.3 TB/s is $1{,}307.4 \times 10^{12} / 5.3 \times 10^{12} = 246.7$ FLOP per byte, and 14 GB over 5.3 TB/s is 2.64 ms. The TPU v5e lists its bandwidth in GiB per second, so convert before comparing: $800 \times 2^{30}$ bytes is 858.99 GB/s.

Sources are [AMD-CDNA3 and CDNA4 whitepapers](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf), the [Cloud TPU documentation](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm), and the NVIDIA datasheets that [chapter 1](/c/01-the-gpu-at-a-glance) uses.

> [!KEY] The ridge point is a design choice, and it keeps rising
> The A100 needs 161 FLOP per byte. Every newer part in the table, apart from the TPU v5p, asks for 218 or more, and the TPU v6e asks for 560. Newer parts add arithmetic faster than bandwidth, so a kernel that was balanced on an A100 is more memory bound on its successor. The lesson of chapter 4 gets stronger with each generation.

Read the table with two cautions. First, a peak is a vendor's rating under the vendor's conditions, so 2,500 against 2,307 is not a measured ranking. Second, the decode column is a floor from bytes alone: it ignores the KV cache and the fact that [the lab A100 reached 1,275 of its 1,935 GB/s](/c/03-memory-and-bandwidth).

## AMD Instinct: the closest cousin

AMD's data-center GPUs are built on the same plan as NVIDIA's, with different names and different constants. This section answers one question: what is the CDNA machine, and what breaks when you port CUDA to it?

### The machine

An MI300X has 304 *compute units* (CUs), spread across 8 chiplets that AMD calls XCDs. Each CU is the counterpart of an SM. It schedules *wavefronts* of 64 lanes, holds a 512 KiB vector register file and 64 KB of *local data share* (LDS), and contains matrix cores that run *MFMA* instructions. The numbers below come from AMD's [CDNA 3](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf) and [CDNA 4](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf) whitepapers and the [MI355X brochure](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi355x-gpu-brochure.pdf).

| Property | MI300X | MI325X | MI355X |
|---|---|---|---|
| Architecture | CDNA 3 | CDNA 3 | CDNA 4 |
| Compute units | 304 | 304 | 256 |
| Wavefront size | 64 | 64 | 64 |
| LDS per CU | 64 KB | 64 KB | 160 KB |
| HBM | 192 GB HBM3 | 256 GB HBM3E | 288 GB HBM3E |
| HBM bandwidth | 5.3 TB/s | 6.0 TB/s | 8.0 TB/s |
| Infinity Cache | 256 MB | 256 MB | 256 MB |
| Dense FP16 and BF16 matrix | 1,307.4 TFLOP/s | 1,307.4 | 2,500 |
| Dense FP8 matrix | 2,614.9 TFLOP/s | 2,614.9 | 5,000 |
| Board power | 750 W | 1,000 W | 1,400 W |

The memory system differs from NVIDIA's in a way that matters to a kernel author. Between the CUs and HBM sit a 4 MB L2 per XCD (32 MB in total on the MI300X) and a 256 MB *Infinity Cache*, a last-level cache shared by the whole package. The Infinity Cache is 6.4 times the A100's 40 MB L2 by size. Between GPUs, AMD's *Infinity Fabric* links connect a node's peers: 7 links of x16 each, 896 GB/s for the peer ring on the MI300X and 1,075.2 GB/s on the CDNA 4 parts ([chapter 21](/c/21-collective-communication) explains why the per-direction figure, not the total, goes in the all-reduce formula; check which one the vendor quotes).

The matrix cores explain the peak rates. AMD lists the per-CU, per-clock matrix rate for FP16 and BF16 on CDNA 3 as 2,048 FLOP, and the MI300X peak engine clock as 2,100 MHz. Multiply them:

$$
304 \times 2{,}048 \times 2.1 \times 10^{9} = 1.307 \times 10^{15}
$$

That is the 1,307.4 TFLOP/s on the data sheet, so the peak is a clean product of CU count, per-CU rate, and clock, the same derivation as [chapter 1's](/c/01-the-gpu-at-a-glance) for the A100. On the MI355X, 256 CUs at 4,096 FLOP per CU per clock at 2.4 GHz give 2,517 TFLOP/s, which matches AMD's 2,500 to within rounding.

### Wavefronts of 64

The one change that bites hardest is the lockstep width. A wavefront on these parts is 64 lanes, twice a warp. The HIP documentation lists the resident limits per CU on CDNA 3 as 32 blocks, 32 wavefronts, and 2,048 threads. That is the same 2,048 threads per SM as the A100's, but arranged as 32 wavefronts of 64 instead of 64 warps of 32 ([the execution model](/c/02-the-execution-model)). The scheduler has half as many independent instruction streams to switch among for the same thread count, so latency hiding leans more on instruction-level parallelism within a wavefront.

A kernel that assumes 32 fails quietly. The most common form is a warp reduction with a hard-coded start offset. The following CUDA fragment is correct on an NVIDIA GPU:

```cpp
// Sums 32 lanes. Lane 0 ends with the total.
__device__ float warp_sum_32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}
```

On a 64-lane wavefront, lane 0 adds lanes 0 to 31 and never sees lanes 32 to 63. The result is wrong without an error, and a test that happens to fill the input with a constant might not notice. The figure shows which lanes land in lane 0's sum.

```viz
23-wavefront-width
```

The fix is to take the width from the machine. Both CUDA and HIP expose `warpSize`, and HIP code that has to run on both starts the loop from it:

```cpp
// Sums warpSize lanes on either vendor. In HIP, use __shfl_down without the mask.
__device__ float warp_sum(float v) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        v += __shfl_down(v, offset);
    return v;
}
```

Treat this as a pattern to verify on your toolchain, because the exact shuffle signature differs between CUDA and HIP and between versions. The point to carry is the rule: ==every literal 32 in a kernel, a launch configuration, or a mask is an assumption about the machine==. Ballot results are another case: on a 64-lane wavefront a ballot has 64 bits, so storing it in a 32-bit integer truncates it.

The same reasoning applies to block shapes. A block of 256 threads is 8 warps on an A100 and 4 wavefronts on a CDNA 3 CU. A kernel that gives each warp a row, as the softmax in [chapter 15](/c/15-softmax-and-normalization) does, now has half as many rows in flight per block, and the right block size may change.

### The LDS and tile sizes

Shared memory sizing is the second place a port breaks. The A100 lets one block use up to 163 KB. A tile configuration tuned to fill that, such as the three-stage pipeline of 128 by 256 by 64 fp16 tiles in [chapter 12](/c/12-pipelining-and-async-copies), needs far more than an MI300X CU's 64 KB. Budget the tile from the LDS size, then use the arithmetic from [chapter 14](/c/14-triton) to see how much reuse the smaller tile gives.

> [!EXAMPLE] Does the A100's pipelined tile fit an MI300X CU?
> A 128 by 256 by 64 fp16 tile needs $(128 \times 64 + 64 \times 256) \times 2 = 49{,}152$ bytes per stage, which is 48 KiB. Three stages need 144 KiB, which fits the A100's 163 KiB but not the MI300X's 64 KB. A single stage of the same tile (48 KiB) fits the 64 KB LDS but leaves no room for a second stage, so the tile has to shrink or the pipeline has to get shallower. On the MI355X's 160 KB the three-stage tile at 144 KiB fits again.

The arithmetic: $(128 \cdot 64 + 64 \cdot 256) = 8{,}192 + 16{,}384 = 24{,}576$ elements, times 2 bytes is 49,152 bytes, which is 48 KiB. Three stages is 144 KiB. The LDS size is a hardware figure and the tile bytes are arithmetic, so this is a prediction about whether the allocation succeeds, not a measurement of any kernel.

### MFMA, HIP, and the porting tools

AMD's matrix instructions are *MFMA* instructions, the analogue of `mma.sync`. Example mnemonics from the CDNA documentation are `V_MFMA_F32_32X32X8_F16` and `V_MFMA_F32_16X16X16_F16` on CDNA 3, and `V_MFMA_F32_16X16X32_F16` and `V_MFMA_F32_16X16X128_F8F6F4` on CDNA 4. Read the name the way you read an NVIDIA `mma` shape ([chapter 11](/c/11-tensor-cores)): output type, then tile $M \times N \times K$, then input type. The instruction shapes differ from NVIDIA's, and the register layout of the operands differs, which is why a hand-written tensor core kernel does not port by renaming.

The software side is easier. *HIP* is AMD's C++ GPU programming language. It mirrors the CUDA runtime API closely enough that the *HIPIFY* tools can translate most CUDA source mechanically, and AMD ships ports of the libraries you know: rocWMMA for WMMA, Composable Kernel for CUTLASS-style templates, and hipBLASLt for cuBLASLt ([HIP hardware features](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hardware_features.html)). *ROCm* is the umbrella software stack. This chapter gives no ROCm version number, because the vendor's release page lists two numbering lines at the time of writing and the reason isn't clear. Check the version your cluster runs.

Runtime calls and library calls translate. The *performance tuning* doesn't: HIPIFY gives you a program that runs, with CUDA-shaped block sizes, tile sizes, and 32-wide assumptions, and making it fast is the work of chapters 6 to 12 again.

### Triton and the tile libraries on AMD

If you wrote the labs in [Triton](/c/14-triton), most of your code is already portable. Triton's block-level programs say what a tile does, not how a warp does it, and the compiler chooses the instructions for the target, including AMD's. Expect to re-tune: `num_warps`, tile shapes, and `num_stages` are the knobs whose best values move with the LDS size and wavefront width. Treat a Triton kernel's AMD performance as unmeasured until you measure it, because the best configuration on one vendor is rarely the best on another.

For the last stretch of performance, AMD's best kernels have often been written in assembly. The HipKittens paper (arXiv 2511.08083) argues that tile-based abstractions can replace that: it reports GEMM and attention kernels competitive with AMD's hand-optimized assembly kernels on CDNA 3 and CDNA 4, and, in some settings, 1.2 to 2.4 times faster than every baseline the authors tried, as reported by the authors. The paper also says the algorithms behind the tile abstractions had to be rethought for AMD, which is the same warning as above from the other side: the interface can be shared, but the schedule underneath is hardware-specific.

## Google TPUs: a compiler-first machine

A TPU is built around a different bet from a GPU's: fewer, much larger matrix units, fed by a compiler that sees the whole program. This section answers one question: how does that change what you write?

### The systolic array

The core of a TPU is the *matrix multiply unit* (MXU), a *systolic array*: a grid of multiply-accumulate cells that pass operands to their neighbours each cycle, so each value read from memory is used by a whole row or column of cells. The original TPU, described by Jouppi and colleagues in 2017 ([arXiv 1704.04760](https://arxiv.org/abs/1704.04760)), had a matrix unit of 65,536 8-bit multiply-accumulates (a 256 by 256 grid), a 92 TOPS peak, and a 28 MiB software-managed on-chip memory. Those figures describe that first, inference-only chip, not a current part.

Current parts keep the idea at larger scale. From the Cloud TPU documentation, the v5e and v5p MXUs are 128 by 128 and the v6e (Trillium) and TPU7x (Ironwood) MXUs are 256 by 256. A 256 by 256 array holds 65,536 multiply-accumulates, and each one is two FLOP, so one MXU does 131,072 FLOP per cycle at full occupancy.

Compare this with the GPU's design. A tensor core multiplies a small tile, 16 by 8 by 16 for the Ampere `mma.sync` in [chapter 11](/c/11-tensor-cores), and the programmer assembles larger products from many such instructions with their own registers. An MXU multiplies an operand that fills the whole array, so a matrix whose dimensions don't fill it wastes cells. A tile of 128 on a 256-wide array uses a quarter of the cells. The practical rule, which follows from the array size and which you should confirm in the TPU documentation for your generation: make matrix dimensions multiples of the array size.

### The XLA model

On a GPU you launch kernels one at a time and fuse by hand or with Triton. On a TPU, you normally hand the compiler the whole computation. *XLA*, the compiler behind JAX, takes a graph of operations, fuses them to cut memory traffic and removes intermediate buffers, then schedules the result across the chip ([OpenXLA architecture](https://openxla.org/xla/architecture)). It is the same job that Inductor and Triton do in [chapter 17](/c/17-fusion-and-compilers), but compiler-driven fusion is the default path rather than an optimization you opt into.

```python
import jax
import jax.numpy as jnp

@jax.jit
def mlp(x, w1, w2):
    h = jax.nn.gelu(x @ w1)   # XLA sees the matmul and the activation together
    return h @ w2             # and chooses how to fuse and tile them
```

Nothing in this function names a tile, a thread, or a memory space. That is the trade. You give up the control the course taught, and in return the compiler plans the whole graph, including the data layout and the communication across chips. When the compiler's plan is good, you write less code. When it isn't, you have fewer levers.

### Pallas: kernels for the gap

*Pallas* is the escape hatch. It is, in its documentation's words, an extension to JAX for writing custom kernels for GPU and TPU, aimed at cases where XLA's automatic optimization falls short: custom fusions, explicit control of memory movement and pipelining, and direct use of hardware features ([Pallas](https://docs.jax.dev/en/latest/pallas/index.html)). On GPU, its documentation lists kernels for Hopper and Blackwell. On TPU, it covers the TensorCore and the SparseCore, the latter being the unit for embedding-style lookups.

A Pallas kernel looks like a Triton program in a different notation. You describe one block with *refs* (handles to a tile in fast memory), and a `BlockSpec` says which tile of each array a grid index maps to:

```python
import jax
from jax.experimental import pallas as pl

def add_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]   # one block, already in fast memory

@jax.jit
def add(x, y):
    bm = 256                                 # rows per block
    spec = pl.BlockSpec((bm, x.shape[1]), lambda i: (i, 0))
    return pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(x.shape[0] // bm,),
        in_specs=[spec, spec],
        out_specs=spec,
    )(x, y)
```

Compare it with the vector add of [chapter 14](/c/14-triton). The grid index plays the role of `tl.program_id`, and the block spec replaces the pointer arithmetic and the mask. The pipelining that you wrote as multistage `cp.async` in [chapter 12](/c/12-pipelining-and-async-copies) is the framework's job. Pallas is under active development, so check the current documentation for the exact API and for layout constraints on block shapes.

### The TPU generations

The headline specifications, from the Cloud TPU documentation, show a family built around scale and interconnect.

| Property | v5e | v5p | v6e (Trillium) | TPU7x (Ironwood) |
|---|---|---|---|---|
| Peak BF16 per chip | 197 TFLOP/s | 459 | 918 | 2,307 |
| HBM | 16 GB | 95 GiB | 32 GB | 192 GiB |
| HBM bandwidth | 800 GiB/s | 2,765 GB/s | 1,638 GB/s | 7,380 GB/s |
| Interconnect (ICI) per chip | 400 GB/s | 1,200 GB/s | 800 GB/s | 1,200 GB/s |
| Pod size | 256 chips | 8,960 | 256 | 9,216 |
| MXU | 128 by 128 | 128 by 128 | 256 by 256 | 256 by 256 |

The pod size is the number that has no GPU counterpart in this course. A full Ironwood pod of 9,216 chips at 2,307 TFLOP/s each is $9{,}216 \times 2{,}307 = 21{,}261{,}312$ TFLOP/s, about 21.3 EFLOP/s of peak BF16, wired as one interconnected system. A GPU cluster reaches that scale with NVLink inside a node and InfiniBand or Ethernet between nodes ([chapter 21](/c/21-collective-communication)). On a TPU, XLA compiles the collective operations along with the matrix multiplications, so the overlap in [chapter 22](/c/22-overlap-and-parallelism) is the compiler's concern more than yours.

> [!NOTE] FP8 on the TPU parts
> The documentation lists Ironwood at 4,614 FP8 TFLOP/s. It also lists FP8 figures for the v5p and v6e equal to their BF16 rates, which this chapter doesn't treat as separate FP8 hardware, because the source doesn't confirm it.

## Apple silicon: unified memory

An Apple chip puts the CPU and GPU on one package sharing one pool of memory, which changes both the cost model and the audience. This section answers one question: what does unified memory remove, and what does it leave?

### What the numbers say

Apple's published memory bandwidths are far below a data-center GPU's, and its capacities are far above a consumer card's. The figures below come from Apple's [support pages](https://support.apple.com/en-us/121553) and [M5 announcement](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/).

| Chip | GPU cores | Memory bandwidth | Maximum memory | Read 14 GB once (ms) |
|---|---|---|---|---|
| M4 | 10 | 120 GB/s | 32 GB | 117 |
| M4 Pro | 16 or 20 | 273 GB/s | 64 GB | 51.3 |
| M4 Max | 32 or 40 | 410 or 546 GB/s | 128 GB | 25.6 at 546 GB/s |
| M5 | 10 | 153 GB/s | 32 GB | 91.5 |
| M5 Max | 32 or 40 | 460 or 614 GB/s | 128 GB | 22.8 at 614 GB/s |

The last column is the same decode-floor arithmetic as in the ridge-point table: 14 GB over 614 GB/s is 22.8 ms, about 3.2 times the A100's 7.24 ms, because the A100's rated bandwidth is about 3.2 times the M5 Max's. What the Mac offers instead is room: 128 GB visible to the GPU, against the A100's 80 GB. A model whose weights are 100 GB doesn't fit the lab card at all, and it fits one of these laptops with room for a cache, at a slower rate per token.

Apple does not state an FP32 TFLOP/s figure for these GPUs, so this chapter can't give you a ridge point for them. If you need one, measure the matrix rate yourself with a GEMM. The M5 line adds a Neural Accelerator to each GPU core, per Apple's announcement, so the matrix hardware changed between the M4 and M5 generations.

### What unified memory removes

On a discrete GPU, data crosses PCIe. The learn-inference course measured about 21 GB/s host to device on the lab card, which is 67% of the Gen4 x16 per-direction ceiling of 31.5 GB/s, and roughly 60 times slower than HBM ([chapter 22](/c/22-overlap-and-parallelism) overlaps those copies with compute). On Apple silicon, the CPU and GPU address the same memory, so there is no host-to-device copy to hide. MLX, Apple's NumPy-like array framework for these chips, is built on that model: arrays live in shared memory and an operation can run on the CPU or the GPU without moving the data ([MLX](https://github.com/ml-explore/mlx)).

Unified memory doesn't remove bandwidth limits. A decode step still reads every weight, and 614 GB/s is the cap. It also doesn't remove the need to be careful about cache behaviour and memory layout. What it removes is the copy, one of the stalls that [the streams chapter](/c/22-overlap-and-parallelism) teaches you to hide.

### Metal and the programming model

Apple's GPU language is the *Metal Shading Language* (MSL), a C++ dialect. The vocabulary maps cleanly onto what you know. A *threadgroup* is a block, *threadgroup memory* is shared memory, and a *SIMD-group* is a warp. The SIMD-group width is 32 on all Apple GPUs, according to Apple's WWDC22 material, although the Metal specification only tells you to query `threads_per_simdgroup`, and you should. Matrix hardware shows up as `simdgroup_matrix` types, 8 by 8 matrices of half, bfloat, or float ([MSL specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)).

Two limits are worth knowing. Maximum threadgroup memory is 32 KB on Apple GPU families 4 and later, which is a fifth of the A100's 163 KB per block, so tile budgets shrink again. And which M-series chip belongs to which family isn't confirmed in the sources this course uses, so look it up in Apple's [feature set tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf) for your chip.

For most readers of this course, the entry point isn't Metal. It is MLX, which supplies array operations, automatic differentiation, and fused kernels, and PyTorch's Metal backend, which runs ordinary PyTorch code. You write Metal kernels the way you write CUDA kernels in this course: only when the framework's operations leave bytes on the table.

## Tile-level languages

The kernels in chapters 14 to 16 were written in terms of tiles, and a growing family of languages makes that the unit of programming on every platform. This section answers one question: what is converging, and what isn't?

A *tile language* has you declare a block of a matrix, load it, multiply it, and store it. The compiler or library decides which threads hold which elements, which instruction multiplies them, and how the copies overlap with the arithmetic. Compare the inner loop of a GEMM in two such languages. The Triton version from [chapter 14](/c/14-triton) is on the left in spirit, and the Pallas version from the section above is on the right:

```python
# Triton: pointer arithmetic, a mask, and tl.dot on tiles.
acc += tl.dot(a_tile, b_tile)

# Pallas: refs and a block spec say which tile arrives; jnp.dot multiplies it.
acc += jnp.dot(a_ref[...], b_ref[...], preferred_element_type=jnp.float32)
```

The algorithm is the same loop. What changes is how much the language reveals. The following table orders the tools in this chapter by how low they let you go.

| Tool | Hardware it targets (per the sources) | What it abstracts | What you still own |
|---|---|---|---|
| Triton | NVIDIA, and AMD through a separate backend | Threads and instruction choice within a block program | Tile shapes, `num_warps`, `num_stages`, masks |
| Triton Gluon | Inside Triton | Less: explicit control of layouts and memory, with tutorials for TMA, `wgmma`, `tcgen05`, and warp specialization | Layouts, pipelines, barriers |
| CuTe DSL (CUTLASS 4.8.1) | NVIDIA | Python front end that mirrors CuTe C++ layouts, tensors, and atoms ([chapter 13](/c/13-cutlass-and-cute)) | Layouts, tile shapes, the MMA atom |
| ThunderKittens | NVIDIA Hopper and Blackwell tested; Ampere no longer actively supported | Tile primitives for registers and shared memory | The schedule across a block |
| TileLang (v0.1.15) | NVIDIA CUDA, AMD HIP, Apple Metal, and Ascend backends | Tile-level programs lowered to several backends | Tile shapes and pipelining directives |
| HipKittens | AMD CDNA 3 and 4 | The ThunderKittens tile idea on AMD | The schedule, which the paper says differs from NVIDIA's |
| Pallas | TPU and NVIDIA GPU (Hopper, Blackwell) | Memory movement and pipelining across both | Block shapes, grid, memory spaces |

The table reflects the projects' state on 2026-10-22, and each is under active development. The Gluon row is described from its tutorial file names, so treat the details as unconfirmed. Mojo is outside the sources this course uses, and this chapter says nothing about it.

Every tool abstracts the work between "I want this tile product" and "these instructions run on these threads", and every one stops short of the algorithm. FlashAttention's online softmax ([chapter 16](/c/16-flash-attention)) is still yours to write. That makes tile languages a good bet for portability: the part that transfers is the part you design.

One caution belongs here because of the lab card. ThunderKittens is described by its repository as tested on Hopper and Blackwell, with Ampere no longer actively supported. An A100 is Ampere. Before adopting any tile language, read which hardware its maintainers test and for how long they plan to keep supporting it.

## Choosing and keeping things portable

You can plan a kernel project so that most of it survives a change of hardware. This section answers one question: what do you keep agnostic, and how do you choose a target?

### What to keep agnostic

Keep these hardware-independent, because they are the course's durable content:

- **The algorithm.** The online softmax, the chunked cross-entropy, the grouped GEMM, and the ring all-reduce are all statements about data movement and numerics. Write them once, in terms of tiles and bytes.
- **The reference and the tests.** A pure PyTorch reference, a set of awkward shapes (non-multiples of the block, size 1), and tolerances derived from the number format ([chapter 9](/c/09-floating-point-on-the-gpu)) transfer to any backend. This is what lets you find the wrong lanes in the reduction above.
- **The performance model.** Keep the FLOP count, the byte count, and the ridge point as inputs, not constants. A function that takes peak rate and bandwidth as arguments tells you on any machine how far from the bound a kernel is.
- **The tile shapes as parameters.** Don't bake 128 into the code. Derive it from the scratchpad size and the register budget, as [the tile budget in chapter 14](/c/14-triton) does.

### What to isolate

Put these behind a small interface, one file per target:

- **Lockstep width.** Read `warpSize` or the equivalent and never write 32 or 64 in a kernel body.
- **Matrix instruction and layout.** The MMA shape and the operand layouts are per-vendor and per-generation.
- **Number formats.** Which formats the matrix unit supports changes between parts: CDNA 4 has no TF32 matrix mode and adds MXFP6 and MXFP4, and Hopper and Blackwell differ from Ampere in their FP8 and FP4 support ([chapter 9](/c/09-floating-point-on-the-gpu) and [chapter 18](/c/18-quantized-kernels)).
- **Scratchpad budget.** Keep the byte budget as a named constant per target.

### How to choose

The following table gives a default for each situation. It is advice, and your constraints may overrule it.

| Situation | Default | Why |
|---|---|---|
| One NVIDIA generation, a hot kernel that dominates your profile | CUDA C++ or CuTe DSL, tuned to that part | You control the schedule, so you can reach the bound |
| NVIDIA and AMD both, kernels that change often | Triton, with per-vendor tuning tables | One source, and the tile abstraction transfers |
| You need the last 10% on a new part | A tile library for that vendor (ThunderKittens, HipKittens), or the vendor's templates | Fewer lines than assembly, close to the hardware |
| TPU fleet | JAX and XLA first, Pallas for the gap | The compiler plans the whole graph |
| Local development on a laptop | MLX or PyTorch's Metal backend | Large memory and no copies, at lower bandwidth |
| Model must run anywhere | Framework operations, with fused library kernels where they exist | Someone else keeps the ports current |

Before you write a custom kernel on any of these, check whether the framework already has an operator for it. The [fused linear cross-entropy in PyTorch 2.14](/c/19-training-kernels) is the course's example: the mechanism is yours to understand, and the shipped operator is what you should use in production.

## What goes wrong

The failures below are the ones a port provokes. Each is silent.

- **A literal 32.** A mask, a loop bound, a block size, or a `% 32` was written for a warp. On a 64-wide wavefront the kernel returns a partial result. Search for 32 and 0xffffffff in every kernel you port.
- **A tile that doesn't fit.** A shared-memory allocation tuned for 163 KB fails or forces a lower occupancy on a part with 64 KB of LDS. Recompute the byte budget per target.
- **Comparing peaks across vendors.** A sparse figure against a dense one, GiB against GB (the TPU v5e's 800 is GiB per second), or a total against a per-direction link rate gives a ratio that is wrong by a factor of 2 or more. Convert before you divide.
- **Trusting a translation.** A kernel that HIPIFY converted runs on AMD hardware, and that's all you know. It isn't tuned, and its speed says nothing about the hardware.

## The course in three questions

You now have the toolkit, and you can compress it into three questions you ask of any kernel on any machine. This section answers one question: what is the habit that outlasts the details?

**1. What bounds it?** Count the FLOP and the bytes. Divide the arithmetic by the bytes, and compare to the ridge point. Below it, the kernel is memory bound, and the floor is bytes over bandwidth. Above it, it is compute bound, and the floor is FLOP over the matrix or vector rate. If neither fits, look for the third bound: launch overhead, a serial dependency, or the interconnect ([chapter 4](/c/04-roofline-and-latency), [chapter 17](/c/17-fusion-and-compilers), [chapter 21](/c/21-collective-communication)).

**2. How far is it from the bound?** Measure, and divide the floor by the measurement. The lab A100's copy reaches 1,275 of 1,935 GB/s, and the Triton FlashAttention in learn-inference reached 27% of the matrix peak against a vendor kernel at 43%. A kernel at 90% of its bound is finished. A kernel at 30% has a reason, and the reason is the next question ([chapter 5](/c/05-profiling)).

**3. What moves it?** Match the gap to a mechanism. Uncoalesced accesses ([chapter 3](/c/03-memory-and-bandwidth)), bank conflicts ([chapter 8](/c/08-shared-memory-and-transpose)), too little occupancy or latency hiding ([chapter 2](/c/02-the-execution-model)), a tile with too little reuse ([chapter 10](/c/10-gemm-from-scratch) and [chapter 14](/c/14-triton)), no overlap of copies and compute ([chapter 12](/c/12-pipelining-and-async-copies)), a type that wastes bytes ([chapter 18](/c/18-quantized-kernels)), or a kernel boundary that forces a round trip through memory ([chapter 17](/c/17-fusion-and-compilers)).

Each of these questions is a statement about the machine's constants and the kernel's counts, and none names a vendor. This is why the course transfers. On an MI355X the ridge point is 313, the wavefront is 64, and the LDS is 160 KB. On Ironwood the matrix unit is 256 by 256 and the compiler plans the schedule. On an M5 Max the bandwidth is 614 GB/s and there is no copy to hide. The three questions are the same on all of them, and only the answers change.

### Where to go next

The best way to keep the skill is to apply it to a kernel you didn't write against a machine you can reach. Four projects, in rising order of difficulty:

1. **Port a lab to another backend.** Take your Triton softmax from [chapter 15](/c/15-softmax-and-normalization) to an AMD, TPU (as a Pallas kernel), or Apple target, write down the ridge point of the new machine first, and report how far you are from the bound.
2. **Make a warp-width test.** Write a kernel test that fails when a reduction assumes 32 lanes, and keep it in your suite.
3. **Read a production kernel against the three questions.** Pick a FlashAttention forward kernel in an open-source repository, count its FLOP and bytes, and name its bound.
4. **Write one kernel in a second language.** Re-express your tiled GEMM in a tile library, and compare the code size and the performance.

## Recap

> [!RECAP]
> - The reasoning of the course carries over to every machine here: a large slow memory, a small fast one, and a matrix unit with a high ridge point. Those ridge points range from 161 FLOP per byte on the lab A100 to 560 on the TPU v6e, and the trend is upward.
> - AMD's CDNA parts rename the NVIDIA concepts: CU, wavefront of 64, LDS (64 KB on the MI300X, 160 KB on the MI355X), MFMA. HIPIFY translates the code, and the tuning, especially anything that assumes 32 lanes or a 163 KB tile, is yours.
> - A TPU is a systolic-array machine compiled by XLA. You write kernels in Pallas only where the compiler's plan falls short, and you keep matrix dimensions matched to the array.
> - Apple silicon removes the host-to-device copy and offers up to 128 GB of GPU-visible memory, at 120 to 614 GB/s. MLX and Metal are the entry points.
> - Tile-level languages abstract the schedule between a tile and its instructions and leave the algorithm to you. Check the hardware each one supports before you adopt it.
> - Ask of any kernel: what bounds it, how far is it from the bound, and what moves it.

## Check your understanding

> [!QUESTION] A warp reduction that works on an A100 returns half the sum on an MI300X. What went wrong, and how do you fix it?
> The loop starts at offset 16 and a shuffle-down reduction then combines only the 32 lanes from lane 0 up. A CDNA wavefront has 64 lanes, so lanes 32 to 63 are never added to lane 0's result. Start the loop at `warpSize / 2`, or at 32 for a 64-lane target, and use a lane mask or ballot type of the right width.

> [!QUESTION] A 128 by 256 by 64 fp16 tile with three pipeline stages needs how much shared memory, and does it fit an MI300X CU's LDS?
> One stage holds $(128 \cdot 64 + 64 \cdot 256) \cdot 2 = 49{,}152$ bytes, which is 48 KiB. Three stages need 144 KiB. The MI300X has 64 KB of LDS per CU, so it doesn't fit. The tile has to shrink or the pipeline has to get shallower. On the MI355X's 160 KB, the 144 KiB fits.

> [!QUESTION] Why does the TPU v6e have a ridge point of 560 FLOP per byte when the TPU v5p has 166?
> The ridge point is the peak rate over the bandwidth. The v6e's 918 TFLOP/s over 1,638 GB/s is 560. The v5p's 459 TFLOP/s over 2,765 GB/s is 166. The v6e roughly doubles the arithmetic of the v5p but has less bandwidth per chip, so a kernel needs far more reuse to be compute bound on it.

> [!QUESTION] Does unified memory on an Apple chip make decode faster than on a discrete GPU?
> No. It removes the copy between host and device, and it lets the GPU address up to 128 GB. A decode step still reads every weight once, so the floor is bytes over bandwidth: 14 GB at 614 GB/s is 22.8 ms for the M5 Max, against 7.24 ms at the A100's rated 1,935 GB/s. The Mac's advantage is capacity, not speed per token.

> [!QUESTION] You have a Triton kernel tuned on an A100. What do you expect when you run it on an AMD GPU, and what do you do first?
> It probably compiles and may run, but its performance is unmeasured, because the best `num_warps`, tile shape, and `num_stages` depend on the wavefront width and the LDS size. First check correctness against your reference over awkward shapes, then re-tune, then compute the ridge point of the new part so you know the bound you are chasing.

> [!QUESTION] Why does a tile-level language not make an algorithm like FlashAttention portable by itself?
> It abstracts the mapping from a tile product to threads and instructions, but the algorithm, here the online softmax and the order in which tiles stream through memory, is your design. The HipKittens paper also says the algorithms behind the tile abstractions had to be rethought for AMD, so the interface can be shared while the schedule is hardware-specific.

> [!QUESTION] A blog compares an MI300X's FP8 rate against an H100's, and the ratio looks too good. Name two things to check.
> Check whether both figures are dense, because sparse figures are twice the dense ones, and whether both are rated at the same precision and clock. Also check the hardware of any measured result. A peak is a rating, so the ratio of two ratings isn't a measured speedup.

## Further reading

- AMD, [CDNA 3](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf) and [CDNA 4](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf) architecture whitepapers: the structure of the CU, the LDS, and the matrix cores.
- ROCm, [HIP hardware features](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hardware_features.html): per-CU limits, and how HIP differs from CUDA.
- Google Cloud, [TPU system architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm): MXU sizes, TensorCores, and SparseCores for each generation.
- JAX, [Pallas](https://docs.jax.dev/en/latest/pallas/index.html), and OpenXLA, [XLA architecture](https://openxla.org/xla/architecture).
- Jouppi and others, [In-Datacenter Performance Analysis of a Tensor Processing Unit](https://arxiv.org/abs/1704.04760) (2017): the first TPU, its systolic array, and why it was fast.
- Spector and others, [ThunderKittens](https://arxiv.org/abs/2410.20399), and the [HipKittens paper](https://arxiv.org/abs/2511.08083): tile primitives on NVIDIA and AMD.
- [TileLang](https://github.com/tile-ai/tilelang) and [CUTLASS with the CuTe DSL](https://github.com/NVIDIA/cutlass): read the examples directory of each.
- Apple, [MLX](https://github.com/ml-explore/mlx) and the [Metal Shading Language specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf).
- [GPU MODE](https://www.gpumode.com): a community for GPU programming, with a lecture series and kernel competitions. Its lectures are on [GitHub](https://github.com/gpu-mode/lectures).
- The companion course, [learn-inference](https://qwen.ifkash.dev/c/12-your-first-cuda-kernel), builds the inference engine that these kernels serve.
- Open-source kernels worth reading line by line: the Triton tutorials (matmul, fused softmax, fused attention), the FlashAttention repository, and the CUTLASS examples. Read each with the three questions above next to you.
