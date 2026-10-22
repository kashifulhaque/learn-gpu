---
title: Triton
slug: 14-triton
part: "Part 3 — Matrix multiplication"
summary: How Triton turns a tile-level Python function into a tensor-core GEMM, what the compiler does for you and what it doesn't, why program order changes L2 reuse, and how to autotune against cuBLAS.
minutes: 150
gpu: true
objectives:
  - "Explain Triton's programming model: one program per tile, no threads, and the work the compiler takes over from you."
  - "Trace a kernel through Triton IR, TritonGPU IR, LLVM IR, PTX, and the cubin, and inspect each stage."
  - "Write a matmul kernel with plain pointer arithmetic, masks for ragged M, N, and K, and an fp32 accumulator."
  - "Derive a tile's shared-memory use, register use, and arithmetic intensity from BLOCK_M, BLOCK_N, BLOCK_K, num_stages, and num_warps."
  - "Derive the 90-to-54 load count that grouped program ordering saves, and predict how it behaves on an A100."
  - "Choose autotune configs and keys, and name the pitfalls: benchmark noise, tuning cost, and config explosion."
  - "Describe tensor descriptors and TMA, and why they replace block pointers."
  - "Decide when Triton is the right tool and when CUDA C++ is."
lab: 14-triton-matmul
---

# Triton

> [!TLDR]
> - In Triton you write what one *program* does with one *tile* of the output. The compiler picks the threads, the
>   memory layouts, the shared-memory staging, the tensor-core instruction, and the software pipeline.
> - A matmul kernel is a loop over K that loads a tile of A and a tile of B with masks, calls `tl.dot` into an fp32
>   accumulator, and stores a masked tile of C. The tile shape and `num_stages` decide shared memory use, registers,
>   and reuse.
> - Program order is yours to choose. Grouped ordering visits tiles in vertical groups so that the programs running
>   together share rows of A and columns of B: 90 tile loads become 54 in the tutorial's 9 x 9 example.
> - `triton.autotune` benchmarks a list of configs per key and keeps the fastest. It costs compile time, it retunes for
>   every new key, and its timings are noisy.
> - Triton gives up the last stretch of control: you can't place warps or choose layouts by hand. The
>   companion course measured a Triton attention kernel at 0.63x of the vendor kernel at 8k tokens on this A100.

[Chapter 10](/c/10-gemm-from-scratch) built an SGEMM by hand, and chapters
[11](/c/11-tensor-cores) and [12](/c/12-pipelining-and-async-copies) moved it onto tensor cores with `cp.async`
pipelines. Each step took hundreds of lines of CUDA C++ and an exact understanding of fragments, banks, and barriers.
Production code can't afford that for every shape and every fusion, so most new ML kernels start in Triton instead.

Triton lets you describe a GEMM in about forty lines of Python, and the compiler writes the shared-memory staging, the
`ldmatrix` and `mma.sync` instructions, and the pipeline that chapters 11 and 12 built by hand. This chapter shows how
that works, where it falls short, and how to get from a first working kernel to something near cuBLAS. The companion
course's [chapter 13](https://qwen.ifkash.dev/c/13-fusion-in-triton) introduced Triton for fusing memory-bound
operations. This chapter goes deeper on the compute-bound case, where the tile shape and the program order decide the
result.

## Before you start

**Tile.** A rectangular block of a matrix that one thread block computes or loads at once. In a GEMM, a program owns a
`BLOCK_M x BLOCK_N` tile of C and walks `BLOCK_K` columns of A and rows of B at a time. Chapter 10 introduced the idea.

**Tensor core instruction.** On the A100, a warp executes `mma.sync.m16n8k16`, which multiplies a 16 x 16 fp16 tile by
a 16 x 8 tile and accumulates into a 16 x 8 fp32 tile: $16 \times 8 \times 16 = 2048$ fused multiply-adds, or 4096 FLOP
([chapter 11](/c/11-tensor-cores)).

**`cp.async` and pipelining.** A copy from global to shared memory that doesn't block the thread, so the load for step
$k+1$ overlaps the math of step $k$ ([chapter 12](/c/12-pipelining-and-async-copies)).

**Wave quantization.** A grid of $T$ equal tiles on $S$ SMs runs in $\lceil T / S \rceil$ waves, and the last wave can
be mostly empty ([chapter 4](/c/04-roofline-and-latency)).

**The A100's L2.** 40 MB shared by all 108 SMs. A tile of A that one program loaded stays there for the next program
that wants it, if the cache hasn't evicted it.

## The programming model: programs over tiles

This section answers one question: what do you write in Triton, and what does the compiler write for you?

CUDA asks you to describe one *thread*. You compute `threadIdx` and `blockIdx`, build the address each thread touches,
and decide which threads cooperate through shared memory and where the barriers go. Triton asks you to describe one
*program*, which is what CUDA calls a thread block, and every value inside the program is a whole tile: a scalar, a
vector, or a matrix of a compile-time shape. There is no `threadIdx`.

The following table shows where each decision goes:

| Decision | CUDA C++ | Triton |
|---|---|---|
| Unit you write | One thread | One program, a block of threads seen as one |
| Grid index | `blockIdx`, `threadIdx` | `tl.program_id(axis)` only |
| Which thread loads which element | You, by index arithmetic | The compiler (layout assignment) |
| Coalescing and vector width | You, by index arithmetic and `float4` | The compiler, from the pointer expression |
| Shared memory, barriers | You: `__shared__`, `__syncthreads()` | The compiler |
| Bank-conflict avoidance (swizzle) | You, with padding or XOR | The compiler |
| Tensor core instruction | You: WMMA or `mma.sync` | The compiler, for `tl.dot` |
| Software pipeline depth | You, by hand | A launch parameter, `num_stages` |
| Threads per block | `blockDim` | A launch parameter, `num_warps` |
| Tile shape | You | You, as `tl.constexpr` values |
| Order of programs, tails, masks | You | You |

The last two rows are the point. Triton takes over everything between the tile and the hardware, and leaves you in
charge of the decisions that set performance at the tile level: the shape of the tile, the order the tiles run in, and
what happens at the ragged edge. ==You choose the tile and the order, and the compiler maps them onto threads.==

### What the compiler decides

Take the load of an A tile as an example. You write a `(BLOCK_M, BLOCK_K)` array of pointers and call `tl.load`. The
compiler then decides several things you would have written by hand in chapter 12:

- **Coalescing.** It assigns each thread a run of consecutive elements along the contiguous axis, so a warp's request
  covers whole 128-byte lines, and it widens each thread's access to 16 bytes when the pointer arithmetic is contiguous and
  the integer arguments, such as the strides, are divisible by 16, which Triton specializes on.
- **Shared-memory staging.** Because the A tile feeds `tl.dot`, it allocates shared memory for the tile and moves the
  data through it, so the tensor core operands can come from `ldmatrix`.
- **Swizzling.** It stores the tile in a swizzled layout so that the `ldmatrix` reads don't hit the same bank
  ([chapter 8](/c/08-shared-memory-and-transpose), and the layout algebra in [chapter 13](/c/13-cutlass-and-cute)).
- **Instruction selection.** On sm_80 it lowers `tl.dot` on 16-bit inputs to `mma.sync`, and on sm_90 it lowers the same
  source to `wgmma`.
- **Pipelining.** For a loop that loads and then calls `tl.dot`, it prefetches `num_stages - 1` iterations ahead with
  `cp.async` and multi-buffers the shared memory.

### What it doesn't decide

The compiler doesn't know your problem's shape, so it can't pick the tile for you; autotuning does that by measurement.
It doesn't reorder programs, so the L2 behavior of the grid is yours. It doesn't fuse across kernels. And it gives you no
handle on warp specialization, the register allocation, or the exact shared-memory layout, so a kernel that needs them
hits a ceiling. The last section of this chapter returns to that ceiling.

## What the compiler does with your tile

This section answers one question: how does a Python function become a cubin, and how do you look at each stage?

Triton compiles in stages, and each one is inspectable. Python source is parsed into an AST and lowered to *Triton IR*
(`ttir`), which still describes tiles in a hardware-independent way. A set of passes then assigns each tile a *layout*,
a map from tile elements to threads and registers, and produces *TritonGPU IR* (`ttgir`). That IR lowers to LLVM IR,
then to PTX, and NVIDIA's assembler turns the PTX into a cubin.

| Stage | Key in `kernel.asm` | What you see |
|---|---|---|
| Triton IR | `ttir` | Tile-level ops (`tt.load`, `tt.dot`), no layouts |
| TritonGPU IR | `ttgir` | The same ops with a layout on every tile, plus shared-memory allocations and async copies |
| LLVM IR | `llir` | Per-thread code |
| PTX | `ptx` | The instructions: `cp.async`, `ldmatrix`, `mma.sync` |
| cubin | `cubin` | The binary. Dump the SASS from it with `cuobjdump`. |

A launch returns the compiled kernel, so you can read every stage from Python:

```python
k = matmul_kernel[grid](a, b, c, c, M, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1),
                        BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, GROUP_M=8,
                        HAS_BIAS=False, ACTIVATION="none",
                        num_warps=4, num_stages=4)

print(list(k.asm.keys()))        # ttir, ttgir, llir, ptx, cubin
print(k.asm["ttgir"][:2000])     # layouts, shared memory, async copies
print(k.n_regs, k.n_spills)      # registers per thread, and spills to local memory
print(k.metadata.shared)         # bytes of shared memory the kernel allocates
```

To dump the IR after every pass, set `MLIR_ENABLE_DUMP=1`. To force a recompile past the on-disk cache, set
`TRITON_ALWAYS_COMPILE=1`, and to find the cache, set `TRITON_CACHE_DIR`. The attribute names in the IR change between
releases, so search for stable words instead of exact spellings.

### What to look for

Four searches tell you most of what the compiler did:

1. **`mma` in the PTX.** For fp16 or bf16 inputs on the A100, expect `mma.sync.aligned.m16n8k16` instructions. If you
   see none, `tl.dot` fell back to CUDA-core math, which is the first thing to check when a matmul is 10x too slow.
2. **`cp.async` in the PTX, and `async_copy` in the TTGIR.** Their presence means the loop was pipelined. A kernel with
   `num_stages=1`, or a loop the compiler couldn't pipeline, has plain `ld.global` instead.
3. **`ldmatrix`.** It means the operands travel through shared memory and the swizzle did its job.
4. **`n_spills` above zero.** A big tile with few warps overflows the 255-register limit, and the spilled values go to
   local memory. The accumulator alone is the culprit, as the next sections show.

> [!TIP] Read the TTGIR once for your own kernel
> The layout attributes in the TTGIR (a *blocked* layout for global loads, a *shared* layout with a swizzle, an *MMA*
> layout for the accumulator, and *dot operand* layouts for A and B) are exactly the objects that chapter 13 builds with
> CuTe's algebra. Dumping them for a 20-line kernel is the fastest way to see that Triton is automating the layout
> work from the earlier chapters.

## Masks, tails, and `tl.dot`

This section answers one question: how does a kernel handle sizes that aren't multiples of the tile?

A Triton array's shape must be a compile-time power of two, so `tl.arange(0, BLOCK_M)` always yields a full tile, even
when only 5 rows of the matrix are left. The *mask* is a boolean tile that marks which lanes are real. A masked-off
lane of a `tl.load` doesn't touch memory and takes the value `other`, and a masked-off lane of a `tl.store` doesn't
write. The companion chapter covers the mechanics; a matmul has three tails, and each needs its own treatment.

| Tail | Where it appears | Mask | `other` |
|---|---|---|---|
| Rows of A past M | A load, and the C store | `offs_m < M` | `0.0`, and the C store skips those rows |
| Columns of B past N | B load, and the C store | `offs_n < N` | `0.0`, and the C store skips those columns |
| K in the last loop iteration | A load and B load | `offs_k < K - k * BLOCK_K` | `0.0`, the identity for a dot product |

The K tail is the dangerous one, because it feeds the accumulator. Rows past M and columns past N only produce garbage
in lanes that the final store discards, but a K position past the end would add a product of unrelated memory into
every output element. Masking the K loads to zero makes the extra positions contribute nothing.

### `tl.dot`

`tl.dot(a, b, acc)` multiplies a `(BLOCK_M, BLOCK_K)` tile by a `(BLOCK_K, BLOCK_N)` tile and adds the result to `acc`.
Three rules matter:

- **Every dimension is at least 16.** The tensor core instruction works on 16 x 8 x 16 pieces, so a smaller tile can't
  map onto it. This is why the lab requires at least 32 for `BLOCK_M` and `BLOCK_N` and at least 16 for `BLOCK_K`.
- **The accumulator's dtype is yours to choose.** Pass an fp32 accumulator, `tl.zeros(..., dtype=tl.float32)`, and the
  16-bit products accumulate in fp32, exactly as the tensor core does. Cast only when you store. Chapter 9's results on
  accumulation error apply here: an fp16 accumulator loses the tail of a long K.
- **fp32 inputs don't mean fp32 math.** By default `tl.dot` on fp32 inputs uses TF32 on the tensor cores, which is
  rated at 156 TFLOP/s on the A100 against 19.5 TFLOP/s for the CUDA cores, but rounds the inputs to a 10-bit mantissa.
  The `input_precision="ieee"` argument asks for full fp32 and runs on the CUDA cores.

## The matmul kernel

This section answers one question: what is the whole kernel, and what does each part cost?

The lab's kernel follows the structure of the official [Triton matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html),
which uses plain pointer arithmetic. It has five steps: find the tile, build the pointer tiles, loop over K, apply an
optional epilogue, and store.

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, HAS_BIAS: tl.constexpr, ACTIVATION: tl.constexpr,
):
    # 1. Which tile? (grouped order, explained in the next section)
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 2. Pointer tiles: one pointer per element of the A and B tiles.
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # 3. The K loop, with an fp32 accumulator.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_left = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_left), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_left) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # 4. Optional epilogue, on the fp32 accumulator, before the downcast.
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc += bias[None, :]
    if ACTIVATION == "relu":
        acc = tl.maximum(acc, 0.0)
    elif ACTIVATION == "silu":
        acc = acc * tl.sigmoid(acc)

    # 5. Store, masked on both M and N.
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

The `[:, None]` and `[None, :]` indexing turns two vectors into a 2D array of pointers by broadcasting, so one line
addresses a whole tile. Passing the strides in makes the kernel work for transposed and sliced inputs: a transposed B
just has `stride_bk = 1`. A launch uses a one-dimensional grid with one program per output tile:

```python
grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
```

The epilogue is the fusion from the companion course in its cheapest form. The bias and the activation run on the
accumulator while it is still in registers, so they add no memory traffic: the unfused version would write C, read it
back, and write it again.

### What one tile costs

A tile is a bundle of three budgets: shared memory, registers, and reuse. Work them out for the largest config in the
lab, a $128 \times 256 \times 64$ tile with 3 stages and 8 warps, using fp16 (2 bytes per element).

**Shared memory.** Each pipeline stage holds one A tile and one B tile. The A tile is $128 \times 64 \times 2 = 16384$
bytes (16 KiB), and the B tile is $64 \times 256 \times 2 = 32768$ bytes (32 KiB). Three stages take
$3 \times 48 = 144$ KiB, which fits under the A100's 163 KiB per-block limit. A fourth stage would take 192 KiB and
fail to launch.

**Registers.** The fp32 accumulator holds $128 \times 256 = 32768$ values. Spread over 8 warps of 32 threads, that is
$32768 / 256 = 128$ registers per thread, before the addressing and the operand fragments. The limit is 255 per thread,
so 8 warps is the smallest count that fits this tile comfortably, and 4 warps would need 256 for the accumulator alone.

**Reuse.** Each K step loads $(128 + 256) \times 64 \times 2$ bytes and performs $2 \times 128 \times 256 \times 64$
FLOP. Divide FLOP by bytes:

$$
\boxed{I_{\text{tile}} = \frac{2 \, \hla{B_M} \hlb{B_N} B_K}{2 \, (\hla{B_M} + \hlb{B_N}) B_K} = \frac{\hla{B_M} \hlb{B_N}}{\hla{B_M} + \hlb{B_N}} = \frac{128 \times 256}{384} = 85.3\ \text{FLOP/byte}}
$$

Here $\hla{B_M}$ and $\hlb{B_N}$ are `BLOCK_M` and `BLOCK_N`, and the result holds for fp16 or bf16 whatever
$B_K$ is. A bigger tile reuses each loaded byte more, which is why GEMM kernels want big tiles and why chapter 10's
register blocking was the same idea at a smaller scale.

The grid adds one more constraint. A $4096 \times 4096$ output cut into $128 \times 256$ tiles has $32 \times 16 = 512$
tiles. With one program per SM, 512 tiles run in $\lceil 512 / 108 \rceil = 5$ waves, and the fill is
$512 / (5 \times 108) = 94.8\%$, because the last wave has only $512 - 4 \times 108 = 80$ tiles.

```viz
14-tile-budget
```

Compare 85.3 with the ridge point, 161 FLOP per byte against the 1935 GB/s rating. The tile's intensity is *below* the
ridge, so if every loaded byte came from HBM the kernel would be memory bound, even though the whole matmul's
intensity is far above it. Counting the minimum traffic of a $4096^3$ fp16 matmul shows the difference:

$$
\frac{2 \times 4096^3\ \text{FLOP}}{3 \times 4096^2 \times 2\ \text{bytes}} = \frac{137.4 \times 10^{9}}{100.7 \times 10^{6}} = 1365\ \text{FLOP/byte}
$$

The matmul's minimum traffic is 100.7 MB, and its arithmetic floor at 312 TFLOP/s is $137.4 \times 10^{9} / 312 \times
10^{12} = 0.44$ ms. A tile that loads 85.3 FLOP per byte from the L2 and gets most of its data there stays compute
bound, and a tile that misses the L2 too often doesn't. The next section is about making the L2 hit.

## Grouped program ordering

This section answers one question: in what order should the programs run, so that the L2 holds what they share?

Programs with consecutive ids run at about the same time. The simplest mapping gives program `pid` the tile
`(pid // num_pid_n, pid % num_pid_n)`, which is row-major order. The programs that run together then cover a few whole
rows of C tiles: they share the rows of A they read, but each one needs a different column of B, so nothing is shared
on the B side.

### The 9 x 9 example

The tutorial's example counts loads for a 9 x 9 grid of C tiles where K is also 9 tiles long. A program reads a row of
9 A tiles and a column of 9 B tiles. Run the first 9 programs and count the distinct tiles that they load:

- **Row-major.** The 9 programs fill one row of C. They share 1 row of A (9 tiles) and need 9 columns of B
  ($9 \times 9 = 81$ tiles): 90 loads.
- **Grouped, `GROUP_M = 3`.** The 9 programs form a $3 \times 3$ block. They need 3 rows of A ($3 \times 9 = 27$ tiles)
  and 3 columns of B (27 tiles): 54 loads.

In general, a block of $r$ rows and $c$ columns of C tiles, with $k$ K-tiles, loads $(r + c)\,k$ tiles for $rc$
programs. For a fixed $rc$, the sum $r + c$ is smallest when the block is as square as it can be.

```viz
14-grouped-ordering
```

### The remapping

*Grouped ordering* walks the C tiles in vertical strips of `GROUP_M` rows. Within a strip, the row index changes
fastest, so consecutive program ids go down the strip before they move one column right:

- `num_pid_in_group = GROUP_M * num_pid_n` is the number of programs in one strip.
- `group_id = pid // num_pid_in_group` is which strip the program belongs to, and `first_pid_m = group_id * GROUP_M`
  is the strip's first row.
- `group_size_m = min(num_pid_m - first_pid_m, GROUP_M)` is the strip's real height, which is shorter for the last
  strip when `num_pid_m` isn't a multiple of `GROUP_M`.
- `pid_m = first_pid_m + (pid % group_size_m)` and `pid_n = (pid % num_pid_in_group) // group_size_m` give the row and
  the column inside the strip.

With `GROUP_M = 1` this reduces to row-major. The lab's `grouped_tile` is this arithmetic in plain Python, so you can
check it without a GPU, and `unique_tile_loads` reproduces the 90 and the 54.

> [!EXAMPLE] The grouping on an A100, for a 4096 x 4096 x 4096 matmul
> Cut C into $128 \times 256$ tiles: 32 tile rows by 16 tile columns. A K step reads a 16 KiB A tile and a 32 KiB B
> tile per program. At 144 KiB of shared memory per program, one program fits on each of the 108 SMs, so 108 programs
> run at once.
>
> - **Requested per K step:** $108 \times 48 = 5184$ KiB.
> - **Row-major:** the 108 programs cover $108 / 16 = 6.75$, so 7, tile rows and all 16 columns. Distinct tiles:
>   $7 \times 16 + 16 \times 32 = 624$ KiB, which is 12.0% of the requests.
> - **`GROUP_M = 8`:** the programs cover 8 rows and $108 / 8 = 13.5$, so 14, columns:
>   $8 \times 16 + 14 \times 32 = 576$ KiB, 11.1% of the requests and 7.7% fewer than row-major.
> - **`GROUP_M = 16`:** 16 rows and 7 columns: $16 \times 16 + 7 \times 32 = 480$ KiB, 23% fewer than row-major.
>
> The distinct tiles are what the L2 must fetch from HBM, if the programs run in lockstep and the L2 keeps what they
> share. Move the sliders in the figure to see the minimum.

The example says the grouping wins, but by less than the 9 x 9 case suggests, and it raises an honest question: does
HBM matter here at all? Over the whole kernel, the programs request $512 \times 64 \times 48$ KiB, which is 1.61 GB (512 tiles, 64 K steps).
If 12% of that reaches HBM, the traffic is 193 MB, which takes $193 \times 10^{6} / (1.275 \times 10^{12}) = 0.15$ ms
at the measured copy ceiling. That is well under the 0.44 ms arithmetic floor, so by
this model the row-major order isn't HBM bound at 4096.

The L2-to-SM traffic is the other side of the budget, and it is large. At the peak tensor rate, a tile of intensity 85.3
FLOP per byte needs $312 \times 10^{12} / 85.3 = 3.66$ TB/s from the L2. Luo and colleagues' microbenchmarks measured
about 2.6 to 2.8 TB/s of L2 throughput on an A100 40GB PCIe, against a 7.2 TB/s whitepaper figure
(the two differ, and the course's fact sheet records the disagreement). At 2.6 to 2.8 TB/s and 85.3 FLOP per byte the limit is
$2.6 \times 10^{12} \times 85.3 \approx 222$ to $2.8 \times 10^{12} \times 85.3 \approx 239$ TFLOP/s. That range is
close to the 220 to 245 TFLOP/s the Triton tutorial reports on an A100, so a bound from L2 bandwidth, not from the
tensor cores, is a plausible reason that real kernels stop near 70 to 79% of the 312 TFLOP/s peak. Treat that as an
estimate across two different cards and two different methods, not a measurement.

The tutorial states that grouping improves performance by more than 10% on some hardware, for example from about 220 to
245 TFLOP/s on an A100, and the arithmetic above predicts a smaller, shape-dependent effect. Both can be true: the
model assumes lockstep programs and a perfect cache. ==Your lab reports `group_speedup`, the measured ratio of
row-major to grouped time, and the model predicts it near 1 at 4096.== A value well above 1 would tell you that the L2
behavior is worse than the model, and that is the kind of discrepancy the course exists to find.

> [!KEY] Order is a tuning parameter, not a correctness detail
> Every ordering computes the same C. The order changes only which tiles share the cache at the same time. When the
> grid is much larger than the number of resident programs, or the L2 is small against the matrices, the effect grows.

## Autotuning

This section answers one question: how do you choose `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `num_warps`, and `num_stages`
without guessing?

You don't choose; you measure. `triton.autotune` takes a list of `triton.Config` objects and a list of argument names
called the *key*. On the first call for each distinct key value, it compiles every config, times each one, and caches the
fastest for that key.

```python
configs = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M":  32, "BLOCK_N":  64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=5, num_warps=2),
    # ... more configs
]
tuned = triton.autotune(configs=configs, key=["M", "N", "K"])(matmul_kernel)
```

The lab's list is adapted from the tutorial's and spans three regimes: big tiles with many warps for large matrices,
mid-size tiles for medium ones, and small tiles with few warps and deep pipelines for small ones, where there aren't
enough tiles to fill 108 SMs. The following table shows each config's cost, from the arithmetic in the last section:

| Tile (M x N x K) | Stages | Warps | Shared memory per block | Accumulator registers per thread |
|---|---|---|---|---|
| 128 x 256 x 64 | 3 | 8 | 144 KiB | 128 |
| 64 x 256 x 32 | 4 | 4 | 80 KiB | 128 |
| 128 x 128 x 32 | 4 | 4 | 64 KiB | 128 |
| 128 x 64 x 32 | 4 | 4 | 48 KiB | 64 |
| 64 x 32 x 32 | 5 | 2 | 30 KiB | 32 |

### Choosing the key

The key lists the arguments whose values change the best config. For a matmul, `M`, `N`, and `K` are the obvious
choice, and the key decides how much the tuner works: every new combination retunes. A serving system whose M is the
number of tokens in a batch would retune for every batch size it sees, so a wrapper typically rounds M up to a power of
two, or to a coarse bucket, before it reaches the tuner.

### Pitfalls

- **Autotuning cost.** Each config compiles once, and each key then runs a benchmark of every config. Eight configs at
  one key can take seconds to tens of seconds, which is a cost you pay at the first call. Warm the tuner before you
  time anything, as the lab does.
- **Config explosion.** The number of compiles is configs $\times$ dtypes $\times$ constexpr variants (here, bias or
  not, and each activation). Ten configs and three epilogues is thirty compiles, and each key then benchmarks ten.
  Keep the list short, and prune configs that can't fit, such as a tile whose shared memory exceeds 163 KiB.
- **Benchmark noise.** A timing includes clock throttling, a cold or warm L2, and the other work on the device. Two
  configs within a few percent of each other can swap places between runs, so the tuner's choice isn't stable at that
  resolution. `triton.testing.do_bench` helps by repeating and by flushing the L2 between runs by default.
- **Tuning on the wrong shape.** The best config for $4096^3$ isn't the best for $M = 16$. The tutorial's own table
  shows the difference: at 2048 it reports cuBLAS at about 220.8 TFLOP/s against Triton at about 184.4, and at 1920 about
  200.3 against 162.6 (the tutorial doesn't name the GPU for that table). Triton wins at other sizes, such as 896 and
  1792. Benchmark across a sweep of shapes, never at one.
- **Timing the tuner.** Tuning happens inside the first call, so a first-call timing includes it. Warm up, then time.

> [!NOTE] Pruning and persistence
> `triton.autotune` accepts `prune_configs_by` to drop configs with a function of your own before benchmarking, and
> recent releases can persist the winners to disk so a second process doesn't retune. Check the signature in your
> installed version before you rely on either.

## Tensor descriptors and TMA

This section answers one question: what replaced `tl.make_block_ptr`, and what does it do on the lab's card?

`tl.make_block_ptr` described a tile once, as a base pointer, a shape, strides, and a block shape, and let you advance
it with `tl.advance`. Triton 3.7 deprecated it, and the lab doesn't use it. Its replacement is
`tl.make_tensor_descriptor`, which has the same ingredients and loads and stores by tile coordinate:

```python
a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, 1],
                                   block_shape=[BLOCK_M, BLOCK_K])
b_desc = tl.make_tensor_descriptor(b_ptr, shape=[K, N], strides=[N, 1],
                                   block_shape=[BLOCK_K, BLOCK_N])
for k in range(0, tl.cdiv(K, BLOCK_K)):
    a = a_desc.load([pid_m * BLOCK_M, k * BLOCK_K])   # a whole tile, from its corner
    b = b_desc.load([k * BLOCK_K, pid_n * BLOCK_N])
    acc = tl.dot(a, b, acc)
```

The descriptor knows the tensor's shape, so a load that crosses the edge returns zeros (the default is
`padding_option='zero'`), and the masks from the earlier section disappear. The documentation sets four requirements:
the base pointer is 16-byte aligned, the leading strides are multiples of 16 bytes, the last dimension is contiguous,
and the tensor has two to five dimensions. The descriptor also lives in global memory, so you register an allocator with
`triton.set_allocator(fn)` before the launch.

The reason for the redesign is hardware. On NVIDIA GPUs with the *Tensor Memory Accelerator* (TMA), which starts at
compute capability 9.0, the documentation says the descriptor is backed by TMA hardware. TMA is a copy engine that
moves a whole tile between global and shared memory from a single instruction issued by one thread, with the
bounds checking and the address generation done in hardware. That frees the threads' registers and issue slots, and it
is the Hopper counterpart of the `cp.async` pipeline from [chapter 12](/c/12-pipelining-and-async-copies). Triton's
official persistent-matmul tutorial gates its TMA kernels on a compute capability of 9 or higher.

> [!WARNING] The descriptor API on the A100 is untested here
> The documentation doesn't say what `make_tensor_descriptor` does on an sm_80 card, which has no TMA. It might fall
> back to ordinary loads, or it might refuse to compile. This course didn't confirm either, so the lab uses plain
> pointer arithmetic, which works on every architecture and is what the official matmul tutorial uses. Test a
> descriptor kernel on your own card before you depend on it.

The same Triton source thus reaches different machinery on different hardware: `cp.async` and `mma.sync` on Ampere,
and TMA with `wgmma` on Hopper. That portability is the case for Triton, and the next section weighs it.

## Debugging with `TRITON_INTERPRET`

This section answers one question: how do you debug a kernel whose indices are wrong?

Set `TRITON_INTERPRET=1` before you run the script, and Triton executes each kernel on the CPU, in Python and NumPy,
one program at a time. There is no compile step, so you can use `print`, a debugger, and ordinary Python exceptions.
Use a tiny shape, such as $33 \times 17 \times 20$ with the lab's smallest block sizes, because the interpreter is
slow.

```bash
TRITON_INTERPRET=1 python my_test.py
```

The interpreter checks the logic, and it can't check the machine. It has no layouts, no shared memory, and no
parallelism, so it can't reproduce a performance problem, a register spill, or a race. A kernel that is correct in the
interpreter and wrong on the GPU points to something device-specific, such as a dtype that the hardware rounds
differently or a value that overflows a 32-bit offset.

For a wrong answer on the GPU, compare against a float32 PyTorch reference over several shapes first. The failing
shapes tell you where to look: wrong only when $K$ isn't a multiple of `BLOCK_K` means the K mask, and wrong only when
$M$ is small means an M mask or a tile below 16.

## Triton versus CUDA C++

This section answers one question: when do you write Triton, and when do you reach for the CUDA C++ of chapters 10 to 12?

What you gain is large. A GEMM that took hundreds of lines in chapter 12 takes about forty. The same source runs on
Ampere, Hopper, and Blackwell, and on AMD. Fusion is trivial: the epilogue above is four lines. Autotuning replaces
the guesswork about tile shapes, and `num_stages` is a single integer in place of a hand-written pipeline.

What you give up is control. You can't choose which warp loads and which computes (warp specialization), you can't pin
a shared-memory layout or a register allocation, and you can't write the instruction schedule that closes the last
gap. Three data points show the size of that gap:

- On the lab A100, the companion course's Triton FlashAttention kernel reached 0.63x of the vendor kernel at 8k
  tokens, with the vendor at about 1.03 ms ([chapter 14 of that course](https://qwen.ifkash.dev/c/14-flash-attention),
  measured on this card).
- For a plain GEMM, the Triton tutorial reports parity with cuBLAS at $4096^3$ (about 219.7 against 219.0 TFLOP/s) and
  a deficit at some other sizes, as the autotuning section showed.
- On a B200, the FlashAttention-4 paper reports a kernel up to 2.7x faster than Triton, and up to 1.3x faster than cuDNN
  9.13, which suggests that the newest hardware features reach hand-written kernels before they reach the compiler.

Triton's repository also has a lower-level front end, Gluon, that exposes layouts, TMA, and warp specialization to the
programmer. This course doesn't cover it, but it shows the direction: when the compiler's defaults stop being enough,
the answer is more control, not a different compiler.

| You need | Use |
|---|---|
| A new fused operation, a custom attention variant, or a quick experiment | Triton |
| A GEMM near the vendor's, for the shapes you actually run | Triton, autotuned, then benchmark against cuBLAS |
| The last 10 to 20% on one architecture | CUTLASS or hand-written CUDA ([chapter 13](/c/13-cutlass-and-cute)) |
| A standard GEMM with no fusion | `torch.matmul`, which calls cuBLAS |

## What goes wrong

- **A K tail without a mask.** The kernel passes at $K = 4096$ and fails at $K = 4097$, because the last iteration adds
  products of memory that belongs to something else. Mask the K loads to zero, and test a K that `BLOCK_K` doesn't
  divide.
- **An M or N mask that uses `other=0.0` but forgets the store mask.** The kernel computes a full tile and writes
  past the edge of C, corrupting the next row. No error is raised. Mask the store.
- **32-bit offsets overflow.** `offs_m * stride_am` is an int32 product. For $M \times K$ above $2^{31}$ elements it
  wraps, and the kernel reads or writes the wrong place. Cast the offsets with `.to(tl.int64)` for very large tensors.
- **fp32 inputs with the default precision.** `tl.dot` on fp32 uses TF32, so your "fp32" matmul is accurate to about
  three decimal digits. Pass `input_precision="ieee"` if you need more, and expect it to be much slower.
- **A tolerance too tight for the dtype.** bf16 keeps 8 significant bits (7 stored), so a correctly rounded bf16 result is off by up
  to $2^{-8}$ of its value. Compare against a float32 reference computed from the same rounded inputs, with a tolerance
  tied to the dtype, not to $10^{-5}$.
- **Timing the first call.** The first call compiles and autotunes. A benchmark that includes it reports a time that is
  orders of magnitude off. Warm up, then time with CUDA events.
- **Benchmarking one shape.** A kernel that matches cuBLAS at $4096^3$ can lose at 1920 or 2048. Sweep the shapes your
  model runs.
- **A config that doesn't fit.** A tile with too many stages asks for more shared memory than the block limit, and the
  autotuner skips it with a failed launch. Compute the shared memory before you add a config, as the lab's tests do.
- **Using `tl.make_block_ptr`.** It works in Triton 3.8, but it's deprecated, prints a warning from 3.7 on, and the
  current language reference no longer documents it.

> [!RECAP]
> - A Triton program owns one tile. You choose the tile shape, the program order, and the masks; the compiler chooses
>   the threads, layouts, shared memory, `mma` instructions, and the `cp.async` pipeline.
> - A matmul is a K loop of masked loads and `tl.dot` into an fp32 accumulator. Mask K to zero, and mask M and N on the
>   store.
> - A 128 x 256 x 64 tile with 3 stages uses 144 KiB of shared memory and 128 accumulator registers per thread, and
>   gives 85.3 FLOP per byte, below the ridge point of 161.
> - Grouped ordering makes the programs that run together share rows of A and columns of B: 90 loads become 54 in the
>   9 x 9 case. On a 4096 matmul the model predicts a smaller gain, and the lab measures it.
> - Autotuning measures configs per key. Keep the list short, warm up before timing, and sweep shapes, not one size.
> - `tl.make_block_ptr` is deprecated. `tl.make_tensor_descriptor` replaces it, is TMA-backed on Hopper, and is
>   unconfirmed on the A100.

## Check your understanding

> [!QUESTION] Why must the K loads be masked to zero, when M and N tails only need a masked store?
> The K index is summed over: every position of the last tile contributes a product to every output element. A
> position past K that isn't zeroed adds unrelated memory into the result. A row past M only affects an output row that
> the masked store discards, and the same holds for a column past N. Zero is the identity for the sum, so it makes the
> extra K positions harmless.

> [!QUESTION] A 128 x 128 fp16 tile has an intensity of 64 FLOP per byte, below the ridge of 161. Why isn't the GEMM memory bound?
> The tile's intensity counts every byte the program loads, but most of those bytes come from the L2 and the caches,
> because the programs running together share rows and columns. Only the share that reaches HBM counts against the
> 1935 GB/s. The matmul's own intensity at $4096^3$, 1365 FLOP per byte of minimum traffic, is far above the ridge. The
> tile's intensity does set how much L2-to-SM bandwidth you need: $312 \times 10^{12} / 64 = 4.9$ TB/s at the peak rate.

> [!QUESTION] You launch a kernel with a $128 \times 256 \times 64$ tile and 4 stages and it fails. Why?
> Each stage holds a 16 KiB A tile and a 32 KiB B tile, which is 48 KiB. Four stages need 192 KiB, which exceeds the
> A100's 163 KiB per-block limit. Three stages, 144 KiB, fit.

> [!QUESTION] With 9 x 9 output tiles and `GROUP_M = 4`, which tiles do programs 0 to 3 compute, and how tall is the last group?
> The grid has 9 tile rows, so the groups cover rows 0 to 3, 4 to 7, and 8 alone. Program ids 0 to 3 run down the first
> column of group 0: (0,0), (1,0), (2,0), (3,0). The last group has height $\min(9 - 8, 4) = 1$, so its programs
> walk along row 8 as in row-major order.

> [!QUESTION] Your kernel reaches 85% of `torch.matmul` at 4096 and 60% at 1920. What do you do?
> Don't conclude that the kernel is wrong; the tutorial's own table shows Triton trailing cuBLAS at some sizes. Check
> that the config list contains tiles suited to the smaller grid, since 1920 gives fewer tiles per wave, then look at the
> wave count: if the last wave is mostly empty, a smaller tile can fill the SMs better. Add configs, retune, and
> re-sweep.

> [!QUESTION] Why does `tl.dot` on fp32 tensors disagree with a float32 PyTorch reference in the third decimal place?
> The default input precision for fp32 operands is TF32, which keeps a 10-bit mantissa. The tensor core rounds the
> inputs before it multiplies. Pass `input_precision="ieee"` to get full fp32 on the CUDA cores, or keep the fast path
> and loosen the tolerance.

> [!QUESTION] The tutorial reports grouping gains above 10% on an A100, but the arithmetic predicts little at 4096. Which do you trust?
> Neither on its own, and you measure. The model assumes programs run in lockstep and that the L2 keeps everything
> they share, and neither is exact. Your lab's `group_speedup` is the measurement on your card. A value near 1.0 agrees
> with the model, and a larger one says the real L2 behavior is worse than the model assumes.

## Lab

> [!TRY] Triton matmul
> In `starter.py`, fill in `grouped_tile` and `unique_tile_loads` (plain Python, no GPU), the `matmul_kernel` body, and
> `autotune_configs`. Use pointer arithmetic and masks, and don't call `tl.make_block_ptr`. Optionally, set
> `SUPPORTS_EPILOGUE = True` after you add a fused bias and a ReLU or SiLU.
>
> You pass when the program order covers every tile once, `unique_tile_loads` returns 90 and 54 for the tutorial's
> example, every config fits the A100's shared memory and registers, and fp16 and bf16 results match a float32
> reference over awkward shapes, including $1 \times 1 \times 1$, a transposed B, and 1000 x 1000 x 1000. The lab then
> reports `tflops_4096`, `pct_of_torch_4096`, `pct_of_peak_4096`, `group_speedup`, and the same figures for the ragged
> size $4095 \times 4097 \times 4093$. It fails only if your kernel runs below 50% of `torch.matmul` at $4096^3$. The
> roofline floor is 0.44 ms, which is 312 TFLOP/s, and the tutorial reports 220 to 245 TFLOP/s on an A100; the lab measures
> what your kernel does. The first run compiles and autotunes, which takes a minute or more.

## Further reading

- [Triton matrix multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html):
  the structure the lab follows, including the 9 x 9 grouping example.
- [Triton persistent matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html):
  tensor descriptors, TMA gating on compute capability 9, and persistent grids.
- [`triton.language.make_tensor_descriptor`](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html):
  the signature and requirements of the API that replaces block pointers.
- [`triton.autotune`](https://triton-lang.org/main/python-api/generated/triton.autotune.html): configs, keys, and
  pruning.
- [Triton release notes](https://github.com/triton-lang/triton/releases): where the block-pointer deprecation (3.7.0)
  appears.
- Tillet, Kung, and Cox, [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)
  (2019): the original design of the block-level programming model.
- [Deep Dive on the Hopper TMA Unit for FP8 GEMMs](https://pytorch.org/blog/hopper-tma-unit/): how Triton uses TMA
  descriptors.
- [Triton-Puzzles](https://github.com/srush/Triton-Puzzles): exercises for building intuition.
- The companion course's [chapter 13, fusion in Triton](https://qwen.ifkash.dev/c/13-fusion-in-triton): Triton for
  memory-bound fusion.
