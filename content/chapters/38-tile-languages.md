---
title: Tile languages in practice
slug: 38-tile-languages
part: "Part 6 — Beyond"
summary: What a tile language takes off your hands (the tile, the layout, the async copy, the pipeline), what Triton, Gluon, TileLang, ThunderKittens, cuTile, and Pallas each leave to you, the same softmax and GEMM loop in several of them, when to leave Triton for Gluon, and how to read the PTX they emit.
minutes: 100
gpu: false
objectives:
  - "Name the four things a tile language abstracts (the tile, the layout, the async copy, the pipeline) and say, for Triton, Gluon, TileLang, ThunderKittens, cuTile, and Pallas, which of the four it hides and which it exposes."
  - "Write a row softmax and a GEMM tile loop in at least two tile languages, line against line, and point to the tile, the mask, the copy, and the pipeline in each."
  - "Compute a Pallas BlockSpec index map as block indices, not element offsets, and say how the grid covers an array whose size is not a multiple of the block."
  - "Compute which warp, lane, and register hold an element of a tile under a Gluon-style blocked layout, and count the registers per thread."
  - "Schedule an S-stage software pipeline, count its steps and its resident tiles, and say what the schedule costs in shared memory."
  - "Decide when to leave Triton for Gluon, and read the PTX a tile language emits to confirm which copy, which matrix instruction, and which synchronization it chose."
lab: 38-tile-languages
---

# Tile languages in practice

> [!TLDR]
> - Every tile language removes the same four chores from CUDA: choosing the tile, assigning elements to threads (the layout), moving tiles into fast memory (the async copy), and overlapping the copy with the math (the pipeline). They differ in how many of the four they let you override.
> - Triton hides all four and gives you `num_warps` and `num_stages`. Gluon, a lower level inside Triton, exposes layouts, shared memory, and warp specialization. TileLang and Pallas expose the pipeline as a loop construct and a block spec. ThunderKittens exposes register and shared tiles in C++. cuTile is NVIDIA's own tile model.
> - A row softmax is the same eight lines in Triton, Gluon, Pallas, and cuTile, because the tile is a row. A GEMM tile loop is the same loop in Triton, TileLang, Pallas, and ThunderKittens: load an A tile and a B tile, multiply into an accumulator, advance K.
> - Leave Triton for Gluon when the IR shows the compiler converting layouts through shared memory, or when you need a producer warp and a consumer warp. Stay when a roofline fraction you can accept is within reach with `num_stages` and the autotuner.
> - You verify what a tile language did by reading its PTX: `cp.async` or `cp.async.bulk.tensor` for the copy, `mma.sync` or `wgmma` for the math, `bar.sync` or `mbarrier` for the wait.

[Chapter 23](/c/23-beyond-cuda) gave tile languages one section and one table. That was enough to say they exist and what they target. It was not enough to write in them, to choose among them, or to tell what one did to your kernel. This chapter goes one level down.

The approach is to treat a tile language as four decisions that the language either makes for you or hands back. You will see those four in a Triton kernel you already know, then watch each language make a different choice about which to hand back. You will write the same softmax and the same GEMM loop in several of them, so the differences show up line against line.

The lab runs on a CPU. None of these languages is in the lab image, and adding them is not the point: the point is the abstractions. You build a block index map, a masked tile loop and an online softmax on a tiny tile API, a register layout, and a pipeline schedule. An optional Triton softmax runs on the A100.

## Before you start

**A Triton program.** In [Triton](/c/14-triton) you write one program per output tile, with `tl.program_id`, `tl.arange`, masked `tl.load` and `tl.store`, and `tl.dot`. The compiler picks the layout and the pipeline.

**Online softmax.** [Chapter 15](/c/15-softmax-and-normalization) derives the running maximum $m$ and running sum $d$ that let one pass over a row replace two. The tile program in this chapter's lab is that recurrence.

**Layouts and fragments.** [Chapter 11](/c/11-tensor-cores) states which lane of a warp holds which element of an `mma.sync.m16n8k16` operand, and [chapter 13](/c/13-cutlass-and-cute) gives the layout algebra. This chapter reuses the idea: a layout is a function from an element to a thread and a register.

**Pipelines and async copy.** [Chapter 12](/c/12-pipelining-and-async-copies) overlaps the copy of the next tile with the multiply of the current one, with `cp.async` groups on Ampere and TMA on Hopper. [Chapter 24](/c/24-cute-dsl-gemm) and [chapter 26](/c/26-hopper-gemm) built a CuTe DSL GEMM and a Hopper box. Chapter 23's tile section is the survey that this chapter extends, and it isn't repeated.

## Four things a tile language takes off your hands

The section answers one question: what, precisely, does "the compiler handles it" mean? Four decisions sit between "I want this tile product" and the instructions that run, and you can place every tile language by which of the four it leaves to you.

Picture one output tile of a GEMM, 128 by 128, with a 128 by 32 slab of A and a 32 by 128 slab of B feeding it. In CUDA C++ you made four decisions about that tile by hand in chapters 10 to 12.

- **The tile.** The shape of the block of data one group of threads owns and reuses: 128 by 128 for C, 32 for the K step.
- **The layout.** Which thread holds which element, in registers and in shared memory. A layout decides whether a warp's load is coalesced, whether the tensor-core operands are where the instruction wants them, and whether a shared-memory read hits one bank.
- **The async copy.** How a tile gets from DRAM into shared memory without the threads stalling: `cp.async` on Ampere, TMA on Hopper.
- **The pipeline.** How many tiles are in flight, so that the copy of tile $s+1$ runs while the tensor cores multiply tile $s$.

Now put numbers on them. A 128 by 128 fp16 tile is 16,384 elements, or 32 KiB. With four warps, which is 128 threads, each thread holds $16384/128 = 128$ elements, 256 bytes, which is sixteen 16-byte vector loads. That division is the layout's job: someone must choose which sixteen runs of eight elements belong to which thread.

The pipeline has a price you can compute. One stage holds the A slab and the B slab: $(128 \times 32 + 32 \times 128) \times 2 = 16{,}384$ bytes. Three stages hold 49,152 bytes of shared memory, 48 KiB, which fits in the 163 KiB per block that the A100 allows ([chapter 1](/c/01-the-gpu-at-a-glance)). Every added stage costs another 16 KiB.

> [!KEY] A tile language is a division of labor
> The tile is always yours: you choose its shape, because the compiler can't know your problem. What varies is who owns the other three. A language that hides the layout, the copy, and the pipeline is faster to write and harder to tune. A language that exposes them is the reverse.

The following table shows who decides each of the four in the tools below. "You" means the language exposes a way to override the choice, not that you must.

| Tool | The tile | The layout | The async copy | The pipeline |
|---|---|---|---|---|
| Triton | You: `tl.constexpr` block sizes | Compiler | Compiler | `num_stages` launch parameter |
| Gluon | You | You: `gl.BlockedLayout` and shared layouts | You: `cp.async` and TMA calls | You: buffers, groups, barriers, warp specialization |
| TileLang | You: `T.Kernel` block sizes | Compiler infers from `T.alloc_fragment` | `T.copy` | `T.Pipelined(num_stages=...)` |
| ThunderKittens | You: `rt` and `st` tile types | Row or column layout, chosen by type | `tma::load_async` or plain loads | You: producer and consumer, stage count |
| cuTile | You: tile shapes, powers of two | Compiler | `ct.load` | Not read in the sources (U) |
| Pallas | You: `BlockSpec` shape | Compiler | `BlockSpec` and the grid | Framework, with `emit_pipeline` on GPU |

Each cell of the table is traced to a project's own documentation in the sections that follow. A cell that says "not read" means this chapter did not find it on a page the author could fetch.

## The map of the languages

The section answers one question: what does each tool call a tile, and what hardware will run it? The answer decides which tool you can use on the lab card, which is an A100 (sm_80), and which you can only read about.

### Triton: blocks, implicit layouts

Triton is the baseline, because chapters 14 to 16 teach it. You write a program per tile and every value inside it is a tile. The compiler assigns layouts, stages shared memory, selects `mma.sync` or `wgmma`, and pipelines loops with `num_stages`. The lab image pins Triton 3.8.0 ([FACTS](../../docs/FACTS.md) section 1).

Triton leaves you three levers on those hidden decisions: `num_warps`, `num_stages`, and the tile shape. When a kernel is slow and those three don't fix it, you have no handle on the layout. That gap is why Gluon exists.

### Gluon: Triton's lower level

*Gluon* is a language inside the Triton repository, imported as `from triton.experimental import gluon` and `from triton.experimental.gluon import language as gl`. Its intro tutorial describes it as a lower-level language that gives you more control and more responsibility. Triton defers tile layouts, memory allocation, data movement, and asynchrony to the compiler, and Gluon exposes those details to you.

You decorate a kernel with `@gluon.jit` and launch it with `kernel[grid](..., num_warps=...)`. The tutorials in the repository are numbered by topic: layouts, async copy, TMA, `wgmma`, `tcgen05`, persistence, warp specialization, and more. Each builds on the one before.

A Gluon tensor has an explicit layout. The most common one is `gl.BlockedLayout(size_per_thread, threads_per_warp, warps_per_cta, order)`, and the tutorial defines a layout as the rule that distributes a tensor's elements among the threads of a thread block. You will compute one in the section on layouts.

Warp specialization in Gluon is `gl.warp_specialize`, and its tutorial says it supports Hopper and newer. A Hopper-class GPU is not the lab's A100, so this chapter describes it and the lab does not run it.

### TileLang: tiles on TVM

*TileLang* describes itself as a concise DSL for GPU, CPU, and NPU kernels, built on top of TVM. A kernel is a Python function whose body declares memory and loops. `T.Kernel` sets the grid and thread count, `T.alloc_shared` and `T.alloc_fragment` allocate tiles in shared memory and registers, `T.copy` moves a tile, `T.gemm` multiplies, and `T.Pipelined(n, num_stages=3)` is the pipelined loop.

Its README lists NVIDIA CUDA (SM70 to SM120) as the primary backend, with AMD ROCm and HIP, Apple Metal, and Huawei Ascend as supported, and LLVM CPU, a CuTe DSL backend, and WebGPU as experimental. A CUDA backend that reaches SM70 includes the A100. One news item in the README mentions automatic warp-specialized pipelines.

### ThunderKittens: register and shared tiles in C++

*ThunderKittens* is a header-only C++ library. You include `kittens.cuh`, and you declare register tiles (`rt_bf`, `rt_fl`) and shared tiles (`st_bf`, `st_hf`) whose template arguments are a size, with row or column layout as part of the type. Its README says tiles are no smaller than 16 by 16 values. Operations take the destination first: `kittens::mul(c, a, b)`. Most operations are warp-level, and `mma_AB` on a warpgroup issues `wgmma`.

The library is aimed at Hopper and Blackwell. Its README states that it is mainly built and tested for those, and that it no longer actively supports Ampere, so the A100 in this course is outside its tested range. The README also states a CUDA 12.8 and C++20 requirement.

### cuTile: NVIDIA's tile model

The name needs care, because NVIDIA's pages use three. *CUDA Tile* is the programming model. *CUDA Tile IR* is the virtual instruction set it is built on. *cuTile* is the user-facing language, in Python (*cuTile Python*), and NVIDIA's page lists a C++ expression of the model too. The Python package is `cuda-tile` on PyPI, and you `import cuda.tile as ct`.

A cuTile kernel is a Python function decorated with `@ct.kernel`. It takes a block ID with `ct.bid(0)`, loads a tile with `ct.load(array, index=..., shape=...)`, and stores one with `ct.store(array, index=..., tile=...)`. You launch it from the host with `ct.launch(stream, grid, kernel, args)`. Tile dimensions must be compile-time constants that are powers of two.

The quickstart lists the requirements: a GPU of compute capability 8.x through 12.x, driver r580 or later, and either a system CUDA Toolkit 13.1 or later or the optional `[tileiras]` extra that installs the CUDA TileIR compiler. NVIDIA's overview page does not say whether the model is released, in beta, or experimental, and it does not list architectures, so the chapter makes no claim about its status (U). The lab image carries CUDA 13.0 and doesn't include `cuda-tile`, so the lab doesn't use it.

### Pallas: BlockSpec and grid, on TPU and GPU

*Pallas* is a JAX extension for custom kernels. Its documentation lists two backends, GPU and TPU, that share the ideas of `Ref`s, `BlockSpec`s, and pipelining and provide distinct, hardware-specific APIs. The GPU quickstart covers NVIDIA Hopper (H100) and Blackwell (B200) and uses `from jax.experimental.pallas import mosaic_gpu as plgpu`. The TPU quickstart covers the TensorCore and the SparseCore.

The unit of programming is a grid of *programs*, `pl.pallas_call(kernel, grid=(n,))`, and `pl.program_id(axis)` returns a program's index. A `BlockSpec` pairs a block shape with an index map and says which block of each array a program sees. The kernel receives references to those blocks and reads them with `x_ref[...]`.

## Writing the same row softmax in four of them

The section answers one question: what is a row softmax in each language, and where is the tile? Use the same problem, a matrix $x$ with $M$ rows and $N$ columns, and one program per row.

In every language the tile is the row. The program loads it, subtracts the maximum, exponentiates, divides by the sum, and stores. The steps come from [chapter 15](/c/15-softmax-and-normalization). The differences are in how you address the row, how you handle a row whose length isn't the tile size, and where the layout lives.

Start with Triton, from the project's fused-softmax tutorial, reduced to one row per program:

```python
@triton.jit
def softmax_kernel(y_ptr, x_ptr, n_cols, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)                       # the tile: one row, BLOCK wide
    mask = offs < n_cols                             # the tail
    x = tl.load(x_ptr + row * stride + offs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)                        # stable: subtract the max
    num = tl.exp(x)
    y = num / tl.sum(num, axis=0)
    tl.store(y_ptr + row * stride + offs, y, mask=mask)
```

The lines that matter are the two that name the tail. `BLOCK` is a power of two, `triton.next_power_of_2(n_cols)`, so a row of 1,000 columns uses a tile of 1,024 and the mask turns off the last 24 lanes. The load fills those lanes with $-\infty$, which neither wins the maximum nor adds to the sum. The tutorial's kernel uses the same `other=-float('inf')` and `tl.max`, `tl.exp`, `tl.sum`.

The Gluon version has the same eight steps with one addition: you pick the layout. The names `gluon.jit`, `gl.BlockedLayout`, `gl.arange(..., layout=...)`, `gl.load(..., mask=...)`, `gl.store`, `gl.program_id`, and `gl.constexpr` come from the Gluon tutorials. The reductions are written below as `gl.max`, `gl.exp`, and `gl.sum`, by analogy with Triton; the tutorials the author read don't use them, so treat those three names as unconfirmed (U):

```python
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

@gluon.jit
def softmax_kernel(y_ptr, x_ptr, n_cols, stride,
                   BLOCK: gl.constexpr, layout: gl.constexpr):
    row = gl.program_id(0)
    offs = gl.arange(0, BLOCK, layout=layout)        # a layout is part of the tensor
    mask = offs < n_cols
    x = gl.load(x_ptr + row * stride + offs, mask=mask, other=-float("inf"))
    x = x - gl.max(x, axis=0)                        # reduction names: U
    num = gl.exp(x)
    y = num / gl.sum(num, axis=0)
    gl.store(y_ptr + row * stride + offs, y, mask=mask)

layout = gl.BlockedLayout([4], [32], [4], [0])       # 4 elements per thread, 32 lanes, 4 warps
```

The only new idea is the last line. Four contiguous elements per thread, 32 lanes, and four warps cover 512 elements in one pass, and a row of 1,024 repeats the layout twice. In Triton the compiler writes that line. In Gluon you write it, and you can change it.

The Pallas version works on a block of rows, because a Pallas kernel sees a block of the array and not a program's index arithmetic. The `BlockSpec` replaces both the pointer arithmetic and the mask:

```python
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl

def softmax_kernel(x_ref, o_ref):
    x = x_ref[...]                                   # a (bm, n) block, already in fast memory
    e = jnp.exp(x - x.max(axis=-1, keepdims=True))
    o_ref[...] = e / e.sum(axis=-1, keepdims=True)

def softmax(x, bm=8):
    m, n = x.shape
    spec = pl.BlockSpec((bm, n), lambda i: (i, 0))   # block i of rows, all columns
    return pl.pallas_call(
        softmax_kernel,
        grid=(pl.cdiv(m, bm),),
        in_specs=[spec],
        out_specs=spec,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    )(x)
```

The index map `lambda i: (i, 0)` returns block indices. Block 1 is rows 8 to 15, not row 1. This is the most common Pallas mistake, and the lab makes you compute it. If $M$ isn't a multiple of 8, the documentation states that out-of-bounds elements are padded on input and discarded on output, and that the padding values are unspecified. A row-wise softmax is safe, because each row is independent and the padded rows are discarded. A column reduction over a padded block isn't.

cuTile's version follows the same shape. The names `ct.kernel`, `ct.bid`, `ct.load`, `ct.store`, and `ct.Constant` come from NVIDIA's quickstart, and the operations page lists `max`, `sum`, and `exp` as tile operations that reduce along an axis or apply elementwise. The keyword names and the padding of an out-of-bounds load are not in the pages the author read (U), so the sketch below marks them:

```python
import cuda.tile as ct

@ct.kernel
def softmax_kernel(x, y, tile_n: ct.Constant[int]):
    pid = ct.bid(0)
    t = ct.load(x, index=(pid, 0), shape=(1, tile_n))   # one row; tail padding: U
    t = t - ct.max(t, axis=1)                           # keyword and shape rule: U
    e = ct.exp(t)
    ct.store(y, index=(pid, 0), tile=e / ct.sum(e, axis=1))
```

The cuTile quickstart documents `ct.PaddingMode` with members `UNDETERMINED`, `ZERO`, `NEG_ZERO`, `NAN`, `POS_INF`, and `NEG_INF`, which select the padding value of a load. A `NEG_INF` padding would be the cuTile counterpart of `other=-float("inf")`. How a load selects one is not on the pages read, so don't copy the sketch; look up `ct.load` in NVIDIA's reference.

> [!NOTE] Unconfirmed details
> This chapter marks a detail U when it could not read it on a project's own page. The `gl.max`, `gl.exp`, and `gl.sum` names, the cuTile reduction keywords, and the cuTile load padding are U. The FACTS file lists each one, so that you and a reviewer know which lines to check before you rely on them.

### The cost, and the one bound

A row softmax on fp32 reads each element once and writes it once. For $M = N = 4096$ that is $2 \times 4096 \times 4096 \times 4 = 134{,}217{,}728$ bytes, so the bytes-only floor is $134{,}217{,}728 / 1935\ \text{GB/s} = 69.4\ \mu\text{s}$ at the rated bandwidth and $105.3\ \mu\text{s}$ at the 1,275 GB/s copy ceiling that [learn-inference measured](/c/03-memory-and-bandwidth) on the lab card. These are predictions from specs. The lab's optional check prints the bandwidth that your Triton kernel reaches and applies no threshold.

The point of the floor is that softmax is memory bound. No language here can do better than the bytes allow, so the four versions above should land close together. A difference larger than that, on the same card, is a sign that one kernel is doing something extra, such as a second pass over $x$ or a layout conversion. The last section of this chapter shows how to find out.

## Writing the same GEMM tile loop in four of them

The section answers one question: what is the K loop of a tiled GEMM in each language, and which line is the copy and which is the pipeline? Every version below is one output tile, a loop over K, a multiply-accumulate, and a store.

Start with Triton, the inner loop of the chapter 14 kernel, with the tail masks on K:

```python
for k in range(0, tl.cdiv(K, BLOCK_K)):
    a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
    acc = tl.dot(a, b, acc)                      # one tile product, accumulated in fp32
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk
```

The loop has no copy or pipeline call. `tl.load` is the copy, and the compiler turns it into `cp.async` and multibuffers it when you pass `num_stages`. The mask on K is yours: a K that isn't a multiple of `BLOCK_K` would read garbage in the last tile otherwise.

TileLang's version is the README's example, a GEMM with a ReLU epilogue. The program declares its memory, then loops:

```python
@tilelang.jit
def matmul_relu(A, B, block_M: int = 128, block_N: int = 128, block_K: int = 32):
    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), T.float16)
    B: T.Tensor((K, N), T.float16)
    C = T.empty((M, N), T.float16)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), T.float16)
        B_shared = T.alloc_shared((block_K, block_N), T.float16)
        C_local = T.alloc_fragment((block_M, block_N), T.float32)

        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        for i, j in T.Parallel(block_M, block_N):
            C_local[i, j] = T.max(C_local[i, j], 0)

        T.copy(C_local, C[by * block_M, bx * block_N])
    return C
```

Here the three memory spaces are explicit: `A_shared` and `B_shared` are the shared tiles, and `C_local` is a register fragment. The copy is `T.copy`, the multiply is `T.gemm`, and the pipeline is the loop type itself, `T.Pipelined(..., num_stages=3)`. A tile of 128 by 32 for A and 32 by 128 for B at three stages is the 48 KiB computed earlier. The README's example doesn't show a mask on K, and the chapter doesn't claim that TileLang masks a ragged K for you (U).

The Pallas version is the TPU matmul from the JAX documentation. The `BlockSpec` index maps carry the K loop. The grid has three axes, and the third axis is K:

```python
def matmul_kernel(x_ref, y_ref, z_ref, acc_ref, *, nsteps):
    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += jnp.dot(x_ref[...], y_ref[...], preferred_element_type=jnp.float32)

    @pl.when(pl.program_id(2) == nsteps - 1)
    def _():
        z_ref[...] = acc_ref[...].astype(z_ref.dtype)

# in the pallas_call:
#   in_specs=[pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
#             pl.BlockSpec((bk, bn), lambda i, j, k: (k, j))],
#   out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
#   scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
#   grid=(m // bm, n // bn, k // bk),
```

Compare it with the Triton loop. There is no loop in the kernel body at all: the grid is the loop. The two index maps are the whole addressing story: A's block at grid point $(i, j, k)$ is block $(i, k)$, and B's is $(k, j)$. The accumulator is an explicit scratch buffer, zeroed at $k = 0$ and written out at the last step. The documentation marks the K axis `"arbitrary"` and the other two `"parallel"` in `dimension_semantics`. Because the grid is the loop, the framework can pipeline the block copies for you, and on GPU `plgpu.emit_pipeline` makes that loop explicit.

ThunderKittens is the odd one out because it is C++ and because its README example, a Hopper matmul of under 100 lines, is structured around specialization. The kernel is a template with a `producer` and a `consumer`:

```cuda
struct producer {
    __device__ static void load(producer_load_args<layout> args) {
        if (warpgroup::laneid() == 0) {
            tma::expect(args.inputs_arrived, args.input);
            for (int i = 0; i < M_BLOCK; i++)
                tma::load_async(args.input.a[i], args.globals.A,
                                {args.common.coord.x + i, args.iter}, args.inputs_arrived);
            // ... the B tiles the same way
        }
    }
};
struct consumer {
    __device__ static void compute(consumer_compute_args<layout> args) {
        warpgroup::mma_AB(args.state.accum,
                          args.input.a[warpgroup::groupid()],
                          reinterpret_cast<wide_tile&>(args.input.b));
        warpgroup::mma_async_wait();
        if (warp::laneid() == 0) arrive(args.inputs_finished);
    }
};
```

The K loop is gone from view. The template calls `load` once per iteration on producer warps and `compute` on consumer warps. The copy is `tma::load_async` with an arrival barrier. The multiply is `warpgroup::mma_AB`, which issues `wgmma`. The pipeline depth is a constant in the template, `INPUT_PIPE_STAGES=4`. This is the pipeline you wrote in chapter 12, with the roles split between warps.

The README says the kernel reaches about 855 TFLOP/s on an H100, 86% of the theoretical maximum. That is a result on one H100, and it says nothing about the A100.

### Where each line went

The following table puts the same four lines of a GEMM in each language side by side. A dash means the language doesn't expose the step.

| Step | Triton | TileLang | Pallas | ThunderKittens |
|---|---|---|---|---|
| Tile shape | `BLOCK_M`, `BLOCK_N`, `BLOCK_K` | `block_M`, `block_N`, `block_K` | `bm`, `bn`, `bk` in `BlockSpec` | `st_bf<64, 64>`, `M_BLOCK`, `N_BLOCK` |
| Which tile this program owns | `tl.program_id` and pointers | `T.Kernel(...) as (bx, by)` | `index_map(i, j, k)` | `common_setup` coordinates |
| Copy to fast memory | `tl.load` | `T.copy` | `BlockSpec`, automatic | `tma::load_async` |
| Multiply | `tl.dot` | `T.gemm` | `jnp.dot` | `warpgroup::mma_AB` |
| Pipeline depth | `num_stages` launch argument | `num_stages=3` in `T.Pipelined` | The framework, or `plgpu.emit_pipeline` | `INPUT_PIPE_STAGES` |
| Ragged K | Mask with `other=0.0` | Not shown (U) | Padded on input, unspecified value | Not shown (U) |

The comparison explains why the four feel so different at the keyboard and so alike on the chip. They all issue a tile copy, a tile multiply, and a barrier in the same rhythm. The language decides which of those you type.

## The block index map, and why it is not an offset

The section answers one question: what is a Pallas `BlockSpec`, exactly? It is the cleanest example in this chapter of an abstraction that hides arithmetic you used to write, and the lab's first piece.

Picture a 10 by 20 array cut into blocks of 4 by 8. There are $\lceil 10/4 \rceil = 3$ block rows and $\lceil 20/8 \rceil = 3$ block columns. The grid is 3 by 3, and the last block row has only 2 real rows, the last block column only 4 real columns.

An index map takes a grid point and returns one *block index* per array dimension. The documentation is explicit: the index map returns block indices, not element offsets, and each block index is multiplied by the matching `block_shape` entry to get the element at which the block starts. Say the index map sends a grid point to $(1, 2)$. The block starts at element $(1 \times 4,\ 2 \times 8) = (4, 16)$ and its slice is rows 4 to 8 and columns 16 to 20, where the 24 that a full block would reach is clipped to the array's 20.

Write the rule as a formula. For dimension $d$, with block index $\hla{b_d}$, block size $\hlb{B_d}$, and array size $\hlc{N_d}$:

$$
\text{slice}_d = \big[\, \hla{b_d}\,\hlb{B_d},\ \min(\hla{b_d}\,\hlb{B_d} + \hlb{B_d},\ \hlc{N_d}) \,\big)
$$

The block index $\hla{b_d}$ is what your index map returns. Mistake it for an element offset and block $(1, 2)$ of this array becomes rows 1 to 5 and columns 2 to 10, overlapping its neighbors and missing the end of the array. That is the wrong implementation the lab's first checks catch.

The default mode is `pl.Blocked`, and an integer in `block_shape` means the same. The documentation describes `pl.Element(block_size, padding)`, where the values the index map returns are used directly as array indices, and it says that mode is currently supported only on TPUs. The restrictions on shapes differ by backend: on a TPU the last two block dimensions must equal the array's dimensions or be divisible by 8 and 128, and with the Mosaic GPU backend the minor dimension must be a multiple of 16 bytes.

> [!EXAMPLE] The GEMM index maps
> In the GEMM above, at grid point $(i, j, k) = (1, 2, 3)$, A's block is $(1, 3)$, B's block is $(3, 2)$, and C's block is $(1, 2)$. With $bm = bn = 128$ and $bk = 32$, A's slice starts at element $(128, 96)$ and B's at $(96, 256)$. The 3 on the K axis walks one tile along A's columns and down B's rows, which is the pointer increment `a_ptrs += BLOCK_K * stride_ak` of the Triton loop, written as an index map.

## The layout: which thread holds which element

The section answers one question: what does Gluon's `BlockedLayout` compute? The answer is a function from an element of a tile to a warp, a lane, and a register, and it is the thing Triton decides for you and Gluon hands back.

The picture is a 16 by 16 tile laid over one warp. A warp has 32 lanes, and 256 elements divided among 32 lanes is 8 elements per lane. The layout says which 8. The four arguments of `gl.BlockedLayout` tell you how:

- `size_per_thread`: how many contiguous elements, per dimension, one thread holds in a run.
- `threads_per_warp`: how the 32 lanes are arranged over the tile. The product is 32.
- `warps_per_cta`: how the warps are arranged. The product is `num_warps`.
- `order`: which dimension varies fastest when you number the lanes, the warps, and a thread's registers. `order=[1, 0]` means the last dimension is fastest, which is row-major.

Take the layout `size_per_thread=[1, 4]`, `threads_per_warp=[8, 4]`, `warps_per_cta=[1, 1]`, `order=[1, 0]` on a 16 by 16 tile. One pass of the layout covers $1 \times 8 = 8$ rows by $4 \times 4 = 16$ columns, so the 16 rows are covered twice, and each thread holds a run of 4 contiguous columns in each of 2 repeats, 8 elements.

For one element, split each coordinate into digits, from the lowest. Take element $(1, 5)$. Column 5 is position 1 inside a run of 4, and then thread-column $\lfloor 5/4 \rfloor \bmod 4 = 1$. Row 1 is thread-row $1 \bmod 8 = 1$. With the last dimension fastest, the lane is $1 + 4 \times 1 = 5$. The element sits in lane 5, register 1. Element $(9, 5)$ has thread-row $9 \bmod 8 = 1$ and one repeat, so it is also in lane 5, at register $1 + 4 \times 1 = 5$.

Write it as a formula. For dimension $d$ with run length $s_d$, lane extent $t_d$, and warp extent $w_d$, the coordinate $e_d$ splits into:

$$
\text{reg}_d = e_d \bmod s_d, \quad
\text{lane}_d = \left\lfloor \frac{e_d}{s_d} \right\rfloor \bmod t_d, \quad
\text{warp}_d = \left\lfloor \frac{e_d}{s_d t_d} \right\rfloor \bmod w_d
$$

Linearize the per-dimension lanes with `order[0]` fastest to get the lane, and the same for warps. The registers are the in-run position plus one run's worth per repeat. The lab names this the *lab's convention* for the register numbering: the Gluon tutorial describes which thread owns an element, and the order in which a thread's registers are numbered is not something the chapter confirmed (U). The warp and lane assignment is the part to trust.

```viz
38-blocked-layout
```

Why does the layout matter? Because a warp's 32 lanes issue one memory request together. In the layout above, one row of the tile is covered by 4 lanes, each holding 4 contiguous fp16 elements, which is 8 bytes. A different layout with `size_per_thread=[1, 8]` would hold 16 bytes per lane, one vector load. That is the coalescing decision of chapter 3, written as an argument. And the layout of a tile on its way into `mma.sync` is the fragment layout of chapter 11: the instruction fixes it, and a mismatch between the layout the data arrives in and the layout the instruction wants is a conversion that goes through shared memory.

> [!INTUITION]
> A layout is a function with no side effects. It maps an element to a place. A tile language either chooses that function for you, so you can't see it, or makes it an argument, so you can't forget it. Gluon makes it an argument.

## The async copy and the pipeline

The section answers one question: how does a tile language let you overlap the copy with the math, and what does it cost? The answer is the same schedule in every language, and the languages differ in who writes it.

Picture a loop over $n$ K tiles on a GPU where a copy and a multiply take about the same time. A serial loop copies tile 0, waits, multiplies tile 0, copies tile 1, waits, and so on. That is $2n$ steps of one tile each.

Now let $S$ buffers hold tiles at once. Issue the copies of the first $S - 1$ tiles before any multiply, and from then on, each step issues the copy of one tile and multiplies a tile that arrived earlier. Step $s$ copies tile $s$ and multiplies tile $s - (S - 1)$. The following table shows $n = 5$ tiles and $S = 3$ stages:

| Step | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| Copy tile | 0 | 1 | 2 | 3 | 4 | none | none |
| Multiply tile | none | none | 0 | 1 | 2 | 3 | 4 |

The loop takes $n + S - 1$ steps, 7 here, against 10 serial. A tile occupies a buffer from the step that copies it through the step that multiplies it, so at most $S$ tiles are resident, and you need $S$ buffers. The first $S - 1$ steps are the *prologue* and the last $S - 1$ are the *drain*.

For the time, with one step as the unit and equal copy and multiply times, the pipelined loop takes:

$$
\boxed{T_{\text{pipe}} = \hla{n} + \hlb{S} - 1 \quad\text{against}\quad T_{\text{serial}} = 2\hla{n}}
$$

For $n = 8$ and $S = 3$, that is 10 steps against 16, a ratio of $16/10 = 1.6$. As $n$ grows the ratio approaches 2, which is the most overlap can give when the copy and the multiply are equal. If the copy takes longer than the multiply, the longer one sets the step and the gain is smaller. The model is an idealization, not a measurement, and the lab doesn't time it.

```viz
38-pipeline-schedule
```

Now see who writes this schedule. In Triton you write nothing: `num_stages=3` on the launch tells the compiler to build it. The Triton tutorial's softmax kernel exposes the same knob on a row loop, `tl.range(..., num_stages=num_stages)`. In TileLang you write `T.Pipelined(..., num_stages=3)`, and the loop is the schedule. In Pallas the grid and the index maps are the loop, and on GPU `plgpu.emit_pipeline` runs the pipelined loop over K and `plgpu.BlockSpec(..., delay_release=1)` keeps an extra buffer alive for an in-flight `wgmma`.

In Gluon you write the prologue, the steady state, and the drain. The async-copy tutorial does exactly that, with calls from `triton.experimental.gluon.language.nvidia.ampere`:

```python
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp

# Prologue: issue num_buffers - 1 loads, each followed by a commit.
for _ in gl.static_range(num_buffers - 1):
    issue_loads(...)               # cp.async_load(smem, ptr + offsets, mask=mask)
    cp.commit_group()

# Steady state: overlap the next load with compute on the oldest buffer.
for _ in range(steady_trips):
    issue_loads(...)
    cp.commit_group()
    cp.wait_group(num_buffers - 1)
    compute(...)

# Drain: wait on the remaining groups.
for i in gl.static_range(num_buffers - 1):
    cp.wait_group(num_buffers - 2 - i)
    compute(...)
```

That is the three-phase schedule from the table, with `cp.wait_group(n)` waiting until $n$ or fewer commit groups are pending. It is the same pattern as the `cp.async` pipeline of chapter 12, with Python names. The shared buffers come from `gl.allocate_shared_memory(dtype, [num_buffers, XBLOCK, YBLOCK], layout=...)`, with a layout such as `gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])`.

On Hopper and newer the copy becomes TMA and the wait becomes an `mbarrier`. The Gluon warp-specialization tutorial shows the calls: `mbarrier.init`, `mbarrier.expect` with the transaction bytes, `mbarrier.wait` with a phase, and `mbarrier.arrive`. It explains that warp specialization gives different warps completely different tasks, and lists the costs: extra synchronization, more shared memory, and higher register pressure. ThunderKittens' `producer` and `consumer` are the same idea in C++.

Warp specialization and the mbarrier phases are the tail of [chapter 26](/c/26-hopper-gemm). Its 49,152-byte transaction count and the phase flip are the arithmetic behind `mbarrier.expect`. This chapter doesn't repeat them.

## When to leave Triton for Gluon

The section answers one question: what are the signs that you have outgrown Triton's hidden decisions, and what do you buy when you leave? Leaving has a cost. A Gluon kernel is longer, tied to one architecture's features, and not portable to AMD the way a Triton kernel is, so you leave only for a reason you can point to in the compiler's output.

Use the following checks, in order.

1. **The roofline fraction is acceptable.** If the kernel reaches a fraction of its bound that your schedule can live with, stop. A softmax or a LayerNorm is memory bound and sits close to the bytes floor in Triton. Gluon has nothing to add.
2. **`num_warps`, `num_stages`, and the tile shape have been swept.** The autotuner of [chapter 14](/c/14-triton) covers these. If a sweep doesn't move the result, the limit isn't one of the levers.
3. **The IR shows a layout conversion in the hot loop.** A tensor produced in one layout and consumed in another goes through shared memory, with a barrier. In Triton you can see this in the TTGIR dump but can't remove it. In Gluon you pick the two layouts so that none is needed.
4. **You need two roles.** A producer warp that issues copies and a consumer warp group that multiplies need register budgets that differ: the tutorial's `gl.warp_specialize` takes a list of per-partition register counts, and the default partition gets what is left over. Triton has no handle on that split.
5. **You need persistence or a scheduling order the compiler won't emit.** The Gluon tutorials include persistence, TMA, `wgmma`, and `tcgen05` as separate topics, because each one is a decision Triton makes for you.

The reverse test is as important. If the answer to the first question is yes, don't open Gluon. A kernel written in Gluon to save 3% of a memory-bound softmax is a kernel you now have to maintain per architecture.

> [!WARNING] The A100 is not the target for the specialized parts
> Gluon's warp specialization is stated to support Hopper and newer. Its Ampere async-copy tutorial uses `cp.async`, which runs on the lab A100, so the layout and pipeline parts are within reach, and the warp-specialization part isn't. ThunderKittens no longer actively supports Ampere. Check the target in the project's README before you start.

## Reading the PTX a tile language emits

The section answers one question: how do you confirm what a tile language did with your kernel? The answer is the same as in [chapter 5](/c/05-profiling): read the machine code. The tile languages differ in how you get at it, and in what you look for.

### Getting the PTX

For Triton, the README documents environment variables that dump the compiler's work. `TRITON_KERNEL_DUMP=1` dumps the IR from each compilation stage and the final PTX or AMDGCN, and `TRITON_DUMP_DIR` names the directory. `MLIR_ENABLE_DUMP=1` dumps the IR before every MLIR pass, and `MLIR_ENABLE_DUMP=kernelName` restricts it to one kernel. `TRITON_ALWAYS_COMPILE=1` forces a compile past the cache, which you need when the dump is empty because the kernel was cached. The PyTorch blog on Triton's compilation stages reads the stages from the compiled kernel object, as in `compiled.asm["ptx"]`, and a Triton version's key names should be checked against its own `compiler.py` (U).

The stages, in order, are Triton IR, Triton GPU IR (the layouts are attached here), LLVM IR, and PTX. 

For ThunderKittens, which is C++, use `nvcc -ptx` or `--keep`. For Pallas on GPU, the JAX lowering and the Mosaic GPU compiler have their own dump options that this chapter didn't verify (U). For cuTile, the pipeline is Python to CUDA Tile IR to a binary built by the `tileiras` compiler, and you can disassemble the resulting kernel with `cuobjdump -sass` like any CUDA binary. For TileLang, the project has a way to print the generated CUDA source that this chapter couldn't confirm in its README (U).

### What to look for

You don't read the whole file. You search for a short list of instructions, and each one answers a question from the four decisions.

| Question | Instruction to find | What it tells you |
|---|---|---|
| Did the copy go through `cp.async`? | `cp.async.cg.shared.global`, `cp.async.commit_group`, `cp.async.wait_group` | An Ampere-style pipelined copy. The number of `wait_group` calls and their argument is the stage count. |
| Did it use TMA? | `cp.async.bulk.tensor` | A Hopper-style copy, with an `mbarrier` for completion. |
| Which matrix instruction? | `mma.sync.aligned.m16n8k16` or `wgmma.mma_async` | Ampere `mma.sync` or Hopper `wgmma`. |
| Are the loads vectorized? | `ld.global.v4`, `ld.shared.v4` | A 16-byte access per lane. A run of `ld.global.b32` is a scalar load: the layout isn't vectorizing. |
| Is there a layout conversion? | `st.shared` followed by `bar.sync` and `ld.shared` inside the loop | The compiler moved a tensor between layouts through shared memory. |
| Did anything spill? | `ld.local`, `st.local` | Register spills to local memory. |

The first two rows read the pipeline. The `wait_group` argument for an $S$-stage pipeline depends on the order of the wait and the commit. [Chapter 12](/c/12-pipelining-and-async-copies) waits and then issues the next copy, so the argument is $S - 2$. The Gluon tutorial issues and commits first, so its argument is `num_buffers - 1`. Either way, a kernel written with `num_stages=3` that shows `wait_group 0` in its loop didn't get the overlap you asked for, and you see that before you profile.

> [!TIP] Compare the loops, not the files
> Dump the PTX of the same problem in two languages and diff only the inner loop: the instructions between the loop label and its back-edge branch. The two should have the same count of matrix instructions per iteration. If one has extra `bar.sync` or `st.shared`/`ld.shared` pairs, you have found the layout conversion.

## What goes wrong

- **A block index mistaken for an offset.** In Pallas, `lambda i: (i * bm, 0)` is wrong: it returns an element offset, and the framework multiplies by `bm` again. The lab's block-slice check uses block $(1, 2)$ of shape $(4, 8)$, which must be rows 4 to 8 and columns 16 to 20.
- **No mask on the last partial tile.** A 37 by 50 by 29 GEMM with 16-wide tiles has 6 programs of 4 K steps each, 48 tile loads in all, and 27 of the 48 hang over an edge of an array. The lab's tile API raises an error there. A real kernel reads whatever is next in memory or faults. The useful work is $2 \times 37 \times 50 \times 29 = 107{,}300$ FLOP of the $2 \times 48 \times 64 \times 32 = 196{,}608$ the padded tiles issue, 54.6%.
- **A running sum that isn't rescaled.** The online softmax multiplies the old sum by $e^{m_{\text{old}} - m_{\text{new}}}$ whenever a tile raises the maximum. Skip it and the result is right whenever the maximum is in the first tile and wrong otherwise, which is a bug that passes a unit test on random data and fails on a ramp.
- **A padded tail that wins the maximum.** Loading the masked tail with 0 instead of $-\infty$ makes a row of large negative values report a maximum of 0. Every real element then exponentiates to nearly nothing.
- **A layout that ignores `order`.** Numbering lanes with the last dimension fastest when the layout says `order=[0, 1]` puts neighboring elements in the wrong lanes, and the coalescing you computed isn't the coalescing you get.
- **One more resident tile than buffers.** A schedule that copies tile $s + 1$ at step $s$ holds $S + 1$ tiles at once. In shared memory that is one buffer you didn't allocate, and at 16 KiB per stage in the 128 by 128 by 32 example, it is the difference between 48 KiB and 64 KiB.
- **Choosing a language by the benchmark chart.** A project's chart names its hardware: ThunderKittens' 855 TFLOP/s is an H100 figure. Don't carry it to the A100, where the library isn't actively supported.

> [!RECAP]
> - Every tile language abstracts the tile, the layout, the async copy, and the pipeline. You always choose the tile shape. Triton hides the other three, Gluon exposes all of them, and TileLang, Pallas, ThunderKittens, and cuTile sit between.
> - A row softmax and a GEMM tile loop have the same shape in every language. The tile is a row, or an output tile plus a K loop. The mask for the ragged edge is yours in Triton and Gluon, and the framework's in Pallas.
> - A Pallas index map returns block indices. Block $(1, 2)$ of shape $(4, 8)$ is rows 4 to 8 and columns 16 to 20, clipped at the array edge.
> - A blocked layout maps an element to a warp, a lane, and a register. The registers per thread are the tile's elements divided by $32 \times$ warps.
> - An $S$-stage pipeline takes $n + S - 1$ steps against $2n$ serial, keeps at most $S$ tiles resident, and costs $S$ buffers of shared memory.
> - Leave Triton for Gluon for a reason visible in the IR: a layout conversion in the loop, or a need for producer and consumer warps. Confirm the result in the PTX.

## Check your understanding

> [!QUESTION] Which of the four decisions does Triton leave to you, and which does it hide?
> It leaves the tile, through `tl.constexpr` block sizes, and it gives you `num_stages` as a lever on the pipeline. It hides the layout and the async copy: the compiler assigns the thread-to-element map, stages the data through shared memory, and chooses `cp.async` or TMA.

> [!QUESTION] A Pallas array is 10 by 20 and the block is 4 by 8. How big is the grid, and what does block $(2, 2)$ cover?
> The grid is $\lceil 10/4 \rceil \times \lceil 20/8 \rceil = 3 \times 3$. Block $(2, 2)$ starts at element $(8, 16)$ and is clipped at the edge: rows 8 to 10 and columns 16 to 20, a 2 by 4 piece of a 4 by 8 block. The framework pads the missing part on input and discards it on output.

> [!QUESTION] In the layout `size_per_thread=[1, 4]`, `threads_per_warp=[8, 4]`, `warps_per_cta=[1, 1]`, `order=[1, 0]`, which lane and register hold element $(9, 5)$ of a 16 by 16 tile?
> Lane 5, register 5. Column 5 is position 1 in a run of 4 and thread-column 1. Row 9 is thread-row $9 \bmod 8 = 1$ and repeat 1. The lane is $1 + 4 \times 1 = 5$, and the register is $1 + 4 \times 1 = 5$ under the lab's numbering.

> [!QUESTION] A 128 by 128 fp16 tile is held by 4 warps. How many elements and bytes does each thread hold, and how many 16-byte vector loads is that?
> $16384 / 128 = 128$ elements, 256 bytes, and $256 / 16 = 16$ vector loads. A layout with `size_per_thread` of 8 contiguous fp16 along the fast dimension makes each of those loads one 16-byte access.

> [!QUESTION] An 8-tile loop runs with $S = 4$ stages. How many steps does it take, how many steps overlap a copy with a multiply, and how many tiles are resident at most?
> $n + S - 1 = 11$ steps. A step overlaps both when it is at least $S - 1 = 3$ and at most $n - 1 = 7$, so steps 3 to 7: 5 steps. At most 4 tiles are resident.

> [!QUESTION] A kernel written with `num_stages=3` shows `cp.async.wait_group 0` inside its loop. What does that tell you?
> The loop waits for every outstanding copy group on each iteration, so the next tile's copy can't be in flight while the current one multiplies. The pipeline you asked for isn't there. Check the PTX for how the compiler built the loop, and whether a dependency forced the wait.

> [!QUESTION] When should you not leave Triton for Gluon?
> When the kernel is memory bound and already near the bytes floor, such as a softmax, and when a sweep of `num_warps`, `num_stages`, and tile shapes still moves the result. A Gluon rewrite is longer, tied to one architecture's features, and for warp specialization needs Hopper or newer, so you leave only for a reason you can see in the IR.

## Lab

> [!TRY] Model the abstractions on a CPU
> Open `38-tile-languages`. `grid_shape` is the one function the first check needs. Then write `block_slices` and `gemm_block_indices`, and a small `pallas_call` that runs a kernel over the grid with zero-padded input blocks. Write `tile_gemm` against the provided tile API (`arange`, `load_tile`, `store_tile`, `mma`), with a mask on every load and the store, and `online_softmax` with a running maximum and sum. Write `blocked_owner` for a blocked layout, and `pipeline_schedule` and `max_resident` for an S-stage pipeline. The last function, `softmax_triton`, is optional and runs on the A100. The checks run on a CPU, and the Triton check is skipped there with a message.

The tile API raises `OutOfBounds` when a load or a store touches an element outside the array that the mask doesn't turn off. A tile loop with no mask on the last partial tile fails on a 37 by 50 by 29 problem with that message.

The lab reports three metrics:

- `tile_gemm_loads`: the number of tile loads in a 64 by 64 by 64 GEMM with 32-wide tiles, which is 16.
- `pipeline_steps`: the length of a 4-tile, 2-stage schedule, which is 5.
- `triton_softmax_gbs`: on a GPU, the bandwidth of your Triton softmax on a 4096 by 4096 fp32 matrix, counting one read and one write of each element. The spec floor is 69.4 microseconds at the rated 1,935 GB/s. There is no threshold. A machine without a GPU or without Triton records `skipped`.

## Further reading

- [Pallas: grids and BlockSpecs](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html), JAX documentation. `pl.pallas_call`, `pl.program_id`, `pl.BlockSpec`, the statement that an index map returns block indices, `pl.Blocked` and `pl.Element`, and how a block that doesn't divide the array is padded.
- [Pallas TPU matmul](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html) and [Pallas GPU quickstart](https://docs.jax.dev/en/latest/pallas/gpu/quickstart.html), JAX documentation. The matmul kernel with a three-axis grid, and `plgpu.emit_pipeline` and `plgpu.wgmma` on Hopper.
- [Gluon tutorials](https://github.com/triton-lang/triton/tree/main/python/tutorials/gluon), Triton repository. Start with `01-intro.py`, then `02-layouts.py`, `03-async-copy.py`, and `08-warp-specialization.py`. The files are numbered by topic, through TMA, `wgmma`, `tcgen05`, and persistence.
- [Triton fused softmax tutorial](https://github.com/triton-lang/triton/blob/main/python/tutorials/02-fused-softmax.py), Triton repository. The softmax kernel this chapter reduces to one row per program.
- [TileLang](https://github.com/tile-ai/tilelang), the tile-ai project. The README's GEMM example, the list of backends, and the install command.
- [ThunderKittens](https://github.com/HazyResearch/ThunderKittens), HazyResearch. The README's matmul kernel, its note on supported hardware, and its description of tile types and layouts.
- [cuTile Python](https://docs.nvidia.com/cuda/cutile-python/), NVIDIA, and [NVIDIA CUDA Tile](https://developer.nvidia.com/cuda/tile). The quickstart's vector-add kernel and requirements, the operations reference, and the page that names CUDA Tile, CUDA Tile IR, and cuTile.
- [Triton README](https://github.com/triton-lang/triton), the section on debugging environment variables. `TRITON_KERNEL_DUMP`, `TRITON_DUMP_DIR`, `MLIR_ENABLE_DUMP`, `TRITON_ALWAYS_COMPILE`, and `TRITON_INTERPRET`.
- [Chapter 14](/c/14-triton), [chapter 24](/c/24-cute-dsl-gemm), [chapter 26](/c/26-hopper-gemm), and [chapter 23](/c/23-beyond-cuda), for the Triton model, the CuTe DSL GEMM, the Hopper TMA and `wgmma` details, and the survey of tile languages across hardware.
