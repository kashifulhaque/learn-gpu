---
title: GEMM from scratch
slug: 10-gemm-from-scratch
part: "Part 3 — Matrix multiplication"
summary: Build an fp32 matrix multiplication up from one thread per output to a register-blocked kernel, and learn which resource, global traffic, shared-memory reads, or arithmetic, limits each step.
minutes: 180
gpu: true
objectives:
  - "Compute the FLOPs (2MNK), the least memory traffic, and the arithmetic intensity of a GEMM, and explain why a large GEMM is bound by arithmetic."
  - "Derive how the arithmetic intensity of a block tile grows with its size, and pick a tile large enough to clear the A100's fp32 ridge point."
  - "Explain why the naive kernel is slow, and fix it with global-memory coalescing and shared-memory tiling, including the tail guards for sizes that don't divide the tile."
  - "Derive the shared-memory bytes per FMA of a thread tile, and explain why 2D register blocking turns an outer product into the inner loop."
  - "Describe the hierarchy block tile, warp tile, thread tile, registers, and what each level reuses."
  - "Explain what vectorized loads and a transposed shared tile buy, and what alignment they require."
  - "Read a kernel ladder in percentages of cuBLAS, name the bottleneck at every rung, and say why 19.5 TFLOP/s is the ceiling until tensor cores."
lab: 10-sgemm
---

# GEMM from scratch

> [!TLDR]
> - A GEMM, $C = AB$ with $A$ of $M \times K$ and $B$ of $K \times N$, does $2MNK$ FLOPs and needs at least $4(MK + KN + MN)$ bytes in fp32. A $4096^3$ GEMM does 137.4 GFLOP on 201.3 MB, which is 682.7 FLOP per byte, 68 times the A100's fp32 ridge point of 10.1. A big GEMM is bound by arithmetic, and the fp32 floor is 7.05 ms at 19.5 TFLOP/s.
> - A naive kernel doesn't get there because every output re-reads a row of $A$ and a column of $B$. Each rung of the ladder removes one limit: coalescing fixes the access pattern, shared-memory tiling cuts global traffic, and register tiling cuts shared-memory traffic.
> - A thread that owns a $T_M \times T_N$ tile reads $T_M + T_N$ values and does $T_M T_N$ FMAs per step of $K$, an outer product. An 8 by 8 thread tile needs 1 byte of shared memory per FMA, and the SM has a budget of 2.
> - The hierarchy is block tile, warp tile, thread tile, registers. Boehm's worklog climbs from 1.3% to 93.7% of cuBLAS this way on an RTX A6000. Treat the percentages as the shape of the climb; the GFLOP/s belong to that card.
> - The CUDA-core ceiling is 19.5 TFLOP/s. The TF32 tensor cores are 8 times higher, which is chapter 11.

Matrix multiplication is where a model spends its time. Every linear layer, every attention score, and every MLP is a GEMM, and in a training step the forward and backward passes are mostly GEMMs. If you understand why one kernel can run at 1% of the hardware's peak and another at 90%, you understand most of what the rest of this course builds on.

This chapter writes that kernel the way people find it: start with the obvious version, measure what limits it, and remove the limit. Each step is explained by a number you can compute from the A100's specification. The kernels are fp32 on the CUDA cores, which keeps the ideas visible. [Chapter 11](/c/11-tensor-cores) swaps the arithmetic for tensor cores, and [chapter 12](/c/12-pipelining-and-async-copies) adds the pipelining that hides the loads. Both keep the tiling you learn here.

The lab has you write the naive, shared-memory tiled, and 2D register-blocked kernels, then report your GFLOP/s as a percentage of cuBLAS on the lab A100. The chapter predicts what the roofline allows. The lab measures what you get.

## Before you start

This chapter uses four ideas from earlier chapters. Skim them if they are not fresh.

- **Coalescing.** The 32 threads of a warp issue a load together, and the hardware fetches 32-byte sectors. Consecutive threads reading consecutive 4-byte words need 4 sectors for 128 useful bytes, and threads reading words far apart need one sector each. [Chapter 3](/c/03-memory-and-bandwidth) measures this.
- **Shared memory and banks.** Shared memory is a small, fast, per-block scratchpad organized as 32 banks of 4 bytes. A warp's access is fast when the 32 threads hit different banks or read one address, and it serializes when they hit different addresses in one bank. [Chapter 8](/c/08-shared-memory-and-transpose) covers the conflicts.
- **The roofline.** A kernel's attainable rate is the smaller of the arithmetic peak and bandwidth times arithmetic intensity, with the ridge point $I^{\ast} = C / \beta$ between them. [Chapter 4](/c/04-roofline-and-latency) draws it.
- **Registers and occupancy.** Each SM has 65,536 32-bit registers and at most 2048 resident threads, and a thread can use at most 255 registers. A kernel that uses many registers per thread runs fewer threads at a time. [Chapter 2](/c/02-the-execution-model) computes the limits.

Matrices here are row-major. Element $(i, j)$ of an $R \times C$ matrix sits at offset `i * C + j`, so walking along a row is walking through consecutive addresses, and walking down a column strides by the row length.

## What a GEMM costs

A GEMM multiplies an $M \times K$ matrix by a $K \times N$ matrix into an $M \times N$ one. The section answers one question: how much arithmetic does that take, how little memory traffic could it take, and what does the ratio say about the best possible kernel?

Each of the $MN$ outputs is a dot product of $K$ pairs, and each pair is one multiply and one add. That gives the FLOP count:

$$
\boxed{\text{FLOPs} = 2MNK}
$$

The least traffic reads $A$ and $B$ once and writes $C$ once. In fp32 that is 4 bytes per element:

$$
\text{bytes}_{\min} = 4\,(\hla{MK} + \hlb{KN} + \hlc{MN})
$$

Here $\hla{MK}$ is the elements of $A$, $\hlb{KN}$ is the elements of $B$, and $\hlc{MN}$ is the elements of $C$. Divide the first by the second and you get the arithmetic intensity at minimum traffic. For a square matrix of side $N$:

$$
I = \frac{2N^3}{4 \cdot 3N^2} = \boxed{\frac{N}{6}\ \text{FLOP/byte}}
$$

Intensity grows with the size of the matrix, because the arithmetic is cubic and the data is quadratic. That is the property that makes GEMM special among kernels: you can reuse every loaded value many times, if you arrange the loops to do it.

> [!EXAMPLE] A 4096 by 4096 by 4096 fp32 GEMM on the A100
> FLOPs: $2 \times 4096^3 = 137.44 \times 10^9$, which is 137.4 GFLOP.
> Minimum traffic: $3 \times 4096^2 \times 4 = 201.3 \times 10^6$ bytes, which is 201.3 MB.
> Intensity: $137.44 \times 10^9 / 201.33 \times 10^6 = 682.7$ FLOP/byte, which is $4096 / 6$.
> The fp32 ridge point is $19.5 \times 10^{12} / 1.935 \times 10^{12} = 10.08$ FLOP/byte, so the GEMM sits 67.7 times to the right of it.
> Time floors: arithmetic $137.44 \times 10^9 / 19.5 \times 10^{12} = 7.05$ ms. Memory $201.3\ \text{MB} / 1935\ \text{GB/s} = 0.104$ ms, or 0.158 ms at the measured 1275 GB/s copy ceiling.

The arithmetic floor is 68 times the memory floor at the rated bandwidth, and 45 times at the measured copy ceiling, so a perfect kernel takes 7.05 ms and spends almost all of it computing. The memory system has a wide margin. A good kernel's job is to *use* that margin, by moving each byte from DRAM once and then reusing it from faster memory. The lab prints both floors for $4096^3$ and measures how close you get to the first.

The 19.5 TFLOP/s is the fp32 peak on the CUDA cores: 108 SMs, 64 FP32 cores each, 2 FLOP per FMA, and 1.41 GHz give $108 \times 64 \times 2 \times 1.41 \times 10^9 = 19.49$ TFLOP/s. PyTorch's `torch.matmul` on fp32 uses exactly these cores by default, because `torch.backends.cuda.matmul.allow_tf32` defaults to `False`. The lab keeps it that way, so your kernels and cuBLAS compete on the same hardware.

### Why a model cares about this shape

A linear layer $y = xW$ over $B \cdot T$ tokens, with $W$ of $d_{in} \times d_{out}$, is a GEMM with $M = B \cdot T$, $K = d_{in}$, and $N = d_{out}$. Attention scores $QK^{\top}$ and the product with $V$ are batches of GEMMs. In training, each linear layer runs three: one forward, and two in the backward pass, $dX = dY\,W^{\top}$ and $dW = X^{\top} dY$. That is the usual reason training costs about three times the forward FLOPs.

A concrete case: 8192 tokens through a $4096 \times 4096$ layer is $2 \times 8192 \times 4096 \times 4096 = 274.9$ GFLOP. In bf16 on the tensor cores, at the 312 TFLOP/s peak, that is at least 0.88 ms. A model has dozens of these per layer, so shaving a few percent off a GEMM kernel shows up directly in the step time.

> [!WARNING] A skinny GEMM is not compute bound
> The $N/6$ result assumes a square problem. With $M = 1$, one token against a $4096 \times 4096$ weight, the GEMM is a matrix-vector product: $2 \times 4096^2 = 33.6$ MFLOP on $4 \times (4096 + 4096^2 + 4096) = 67.1$ MB, an intensity of 0.5 FLOP/byte. That is far below the ridge point, and the kernel is bound by reading the weights. This is why decoding a language model is a bandwidth problem, and why the rest of this chapter assumes large $M$, $N$, and $K$.

## The naive kernel

The simplest kernel gives one thread one element of $C$. The section answers one question: how much memory traffic does that cost, and why does it make the kernel slow?

Thread $(i, j)$ walks $k$ from 0 to $K - 1$, multiplies `A[i*K + k]` by `B[k*N + j]`, and accumulates. The kernel is short:

```cuda
__global__ void naive_kernel(int M, int N, int K,
                             const float* A, const float* B, float* C) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k)
            acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}
```

The launch uses 32 by 32 blocks, `dim3 grid(CEIL_DIV(M, 32), CEIL_DIV(N, 32))`, and the guard handles sizes that aren't multiples of 32. It is correct for every shape and it works. It is also the version a reader writes first, and it has two separate problems.

### Problem one: no reuse

Every thread loads $2K$ floats and does $K$ FMAs, so each FMA costs two 4-byte loads, 8 bytes for 2 FLOP. The kernel's intensity, counted at the load instructions, is 0.25 FLOP/byte. Across the grid that is $MN \times 2K \times 4$ bytes of loads:

$$
16{,}777{,}216 \times 8192 \times 4 = 549.8\ \text{GB for}\ 4096^3
$$

That is 2,731 times the 201.3 MB minimum. If every one of those loads went to DRAM, the kernel would take at least $549.8\ \text{GB} / 1935\ \text{GB/s} = 284$ ms, which caps it at 484 GFLOP/s, 2.5% of the fp32 peak. The caches soften this, because neighboring threads read the same rows and columns, but the loads still all pass through the load/store units and the L1 cache. The kernel has no *planned* reuse, only whatever the caches happen to catch.

### Problem two: the access pattern

Look at which thread is which. The mapping above takes `row` from `threadIdx.x`, so the 32 threads of a warp are 32 different rows with the same column. In one iteration of the loop they read:

- `A[row*K + k]`: 32 addresses $K$ floats apart, in 32 different sectors.
- `B[k*N + col]`: one address, the same for all 32 threads, which is one sector.

That is 33 sectors to deliver 33 useful floats, and each sector of `A` brings 32 bytes of which a thread uses 4. Chapter 3 calls this an 8 times read amplification. Boehm measures this kernel, the worklog's kernel 1, at 1.3% of cuBLAS on an RTX A6000.

> [!NOTE] Boehm's numbers are from a different card
> Simon Boehm's [SGEMM worklog](https://siboehm.com/articles/22/CUDA-MMM) ran on an RTX A6000, a GeForce-class Ampere GPU with 84 SMs and 768 GB/s of memory bandwidth. This chapter quotes only his percentages of cuBLAS, which describe how far each kernel is from that card's best, and which transfer to the A100 as a shape: the order of the rungs, and roughly how much each one buys. His GFLOP/s don't transfer, and the A100 has its own cuBLAS baseline, which your lab measures.

## Coalescing: swap the roles

Fixing the access pattern costs one line. The section answers one question: which thread should own which output so that a warp's loads are contiguous?

Let `threadIdx.x`, the fast index within a warp, walk along a row of $C$ instead of down a column:

```cuda
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // was blockIdx.y, threadIdx.y
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // was blockIdx.x, threadIdx.x
```

The grid dimensions swap with them: `dim3 grid(CEIL_DIV(N, 32), CEIL_DIV(M, 32))`. Now the 32 threads of a warp share a row and cover 32 consecutive columns. In one iteration of the loop:

- `A[row*K + k]` is one address for the whole warp, a broadcast: one sector.
- `B[k*N + col]` is 32 consecutive floats, 128 bytes: four sectors.
- The final store `C[row*N + col]` is also 128 contiguous bytes.

A warp now touches 5 sectors per iteration for 33 floats, against 33 sectors before, so it moves $33 / 5 = 6.6$ times fewer sectors for the same work. Boehm's coalesced kernel, his kernel 2, reaches 8.5% of cuBLAS: about 6.5 times his naive kernel, which agrees with the sector count.

The kernel is still at 8.5%, and the remaining problem is the one from before: no planned reuse. Every thread still loads $2K$ floats. The next step is to give the block a place to keep what it loads.

## Shared-memory tiling

Shared memory lets the threads of a block load a piece of $A$ and $B$ once and use it many times. The section answers one question: how much does a tile cut the global traffic, and how do you write the loop so it handles any size?

Take a block of 32 by 32 threads that owns a $32 \times 32$ tile of $C$. To compute it, the block needs 32 rows of $A$ and 32 columns of $B$, all $K$ long. Instead of each thread streaming its own row and column, the block walks along $K$ in steps of 32. At each step it loads a $32 \times 32$ piece of $A$ and a $32 \times 32$ piece of $B$ into shared memory, then every thread uses a row of the first and a column of the second.

Each loaded value is now used by 32 threads, so the global loads per output drop from $2K$ to $2K / 32$. For $4096^3$ the total is $549.8 / 32 = 17.2$ GB, a 32-fold cut.

> [!KEY] Tiling trades global loads for shared-memory loads
> A block tile of $B_M \times B_N$ loads $B_M + B_N$ values per step of $K$ and does $B_M B_N$ FMAs with them. Reuse is the ratio, and it grows with the tile.

Here is the kernel. It keeps the coalesced mapping, so `threadIdx.x` is the column:

```cuda
#define TILE 32

__global__ void tiled_kernel(int M, int N, int K,
                             const float* A, const float* B, float* C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.f;
    for (int t = 0; t < K; t += TILE) {
        As[ty][tx] = (row < M && t + tx < K) ? A[row * K + t + tx] : 0.f;
        Bs[ty][tx] = (t + ty < K && col < N) ? B[(t + ty) * N + col] : 0.f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
```

Three details carry the correctness, and the lab tests each of them with awkward sizes.

1. **Zero-fill the tails.** When $K$ isn't a multiple of 32, the last step has a partial tile, and when $M$ or $N$ isn't a multiple of 32, the edge blocks have threads outside the matrix. Loading `0.f` for an element out of range makes the extra terms add nothing, so the loop body is unchanged and no thread needs to skip it.
2. **Guard the loads and the store, never the loop.** `__syncthreads()` waits for every thread of the block, so a thread that skipped it would hang the block, or worse, let it continue on a half-loaded tile. Every thread runs every iteration, including the ones whose elements are out of range.
3. **Two barriers per step.** The first makes the freshly written tiles visible to all threads before anyone reads them. The second keeps a fast thread from overwriting the tiles for the next step while a slow thread is still reading this one.

The reads are free of bank conflicts. In `As[ty][k]` all 32 threads of a warp share `ty` and `k`, so they read one address, which is a broadcast. In `Bs[k][tx]` the 32 threads read 32 consecutive words, one per bank.

### What limits it now

Boehm's kernel 3 reaches 12.8% of cuBLAS, a 1.5 times gain over coalescing, and far short of the 32-fold traffic cut. The profiler explains why: the dominant stall is "MIO throttle", the queue in front of the shared-memory and special-function pipeline backing up. The bottleneck moved from global memory to shared memory.

The reason is the inner loop's ratio. For each FMA a thread reads two values from shared memory, `As[ty][k]` and `Bs[k][tx]`. The SM can retire 64 FMAs per clock (64 FP32 cores) but shared memory delivers 128 bytes per clock (32 banks of 4 bytes). That budget is 2 bytes of shared-memory reads per FMA, and this kernel asks for 8: two 4-byte loads per FMA. It can reach at most a quarter of the FMA peak, and in practice it reaches less.

Global traffic isn't the binding limit here, by the way. If all 17.2 GB came from the L2 cache, which delivers about 2.8 TB/s on the A100 in Luo and others' microbenchmarks, the 32 by 32 tile's intensity of 8 FLOP/byte would allow $8 \times 2.8 = 22.4$ TFLOP/s, above the 19.5 peak. The next rung attacks the shared-memory reads.

## How big a tile has to be

Tiling has a size knob, and the roofline says how far to turn it. The section answers one question: how large must a block tile be for global traffic to stop being the limit?

A block tile of $B_M \times B_N$ does $2 B_M B_N B_K$ FLOPs per step and loads $4 (B_M + B_N) B_K$ bytes. Divide:

$$
I_{\text{tile}} = \frac{2\,\hla{B_M B_N B_K}}{4\,(\hlb{B_M + B_N})\,\hla{B_K}} = \boxed{\frac{B_M B_N}{2\,(B_M + B_N)}\ \text{FLOP/byte}}
$$

The $B_K$ cancels, and for a square tile of side $B$ the intensity is $B / 4$. The following table prices each tile at the two bandwidths in this course. The attainable rate is $I \times \beta$, capped at the 19.5 TFLOP/s peak:

| Square tile | $I$ (FLOP/byte) | Attainable at 1275 GB/s | Attainable at 1935 GB/s |
|---|---|---|---|
| 16 by 16 | 4 | 5.1 TFLOP/s | 7.7 TFLOP/s |
| 32 by 32 | 8 | 10.2 TFLOP/s | 15.5 TFLOP/s |
| 64 by 64 | 16 | 19.5 TFLOP/s (capped from 20.4) | 19.5 TFLOP/s (capped from 31.0) |
| 128 by 128 | 32 | 19.5 TFLOP/s (capped from 40.8) | 19.5 TFLOP/s (capped from 61.9) |

A tile must have $B / 4 \ge I^{\ast}$ to be arithmetic bound: $B \ge 40.3$ at the rated 1935 GB/s, and $B \ge 61.2$ at the measured 1275 GB/s. So 64 by 64 is the smallest power-of-two tile that clears the DRAM ridge point, and 128 by 128 clears it with room to spare. Drag the sliders in the figure to see the bars move.

```viz
10-tile-intensity
```

This is a DRAM-level argument, and it is pessimistic, because the L2 cache catches reuse between blocks that load the same rows of $A$ and columns of $B$ at about the same time. Boehm reports roughly an 80% L2 hit rate for his final kernel. Small tiles still ask the memory system for more bytes per FLOP, whichever level serves them.

The cost of a large tile is on-chip. A 128 by 128 block tile with $B_K = 8$ stages $(128 + 128) \times 8 \times 4 = 8{,}192$ bytes, 8 KiB, in shared memory. That is small against the 164 KiB per SM. The tile's accumulators are the larger cost: $128 \times 128 = 16{,}384$ floats, which are 64 KiB if they live in registers across the block, a quarter of the SM's 256 KiB register file. The next section is about where those accumulators live.

## Register blocking: the outer product

Shared memory is the bottleneck of the 32 by 32 kernel, and registers are the way out. The section answers one question: how do you make each shared-memory read feed more than one FMA?

Let each thread compute not one output but a $T_M \times T_N$ sub-tile of the block's tile, keeping all $T_M T_N$ accumulators in registers. At each step $k$ the thread reads $T_M$ values of $A$ (one column of its sub-tile of rows) and $T_N$ values of $B$ (one row of its sub-tile of columns) from shared memory into registers, and then does *every* pairing: $T_M T_N$ FMAs. That is an outer product:

$$
\hld{\text{acc}}_{ij} \mathrel{+}= \hla{a_i}\,\hlb{b_j}, \qquad i < T_M,\ j < T_N
$$

Here $\hla{a_i}$ is the thread's $T_M$ values of $A$ at this $k$, $\hlb{b_j}$ is its $T_N$ values of $B$, and $\hld{\text{acc}}_{ij}$ is the register tile. The thread reads $T_M + T_N$ values and does $T_M T_N$ FMAs, so the shared-memory bytes per FMA are:

$$
\boxed{\frac{4\,(T_M + T_N)}{T_M T_N}\ \text{bytes per FMA}}
$$

Compare it with the budget of 2 bytes per FMA from the last section:

| Thread tile | Reads per $k$ | FMAs per $k$ | Bytes per FMA | FMA peak the budget allows |
|---|---|---|---|---|
| 1 by 1 (the tiled kernel) | 2 | 1 | 8 | 25% |
| 8 by 1 (1D blocktiling) | 9 | 8 | 4.5 | 44% |
| 4 by 4 | 8 | 16 | 2 | 100% |
| 8 by 8 | 16 | 64 | 1 | 100%, with half the bandwidth to spare |

The last column is the budget divided by the bytes per FMA, capped at 100%. The 4 by 4 tile just meets the budget, and 8 by 8 leaves room for the other loads and for bank conflicts.

```viz
10-outer-product
```

> [!NOTE] The count is conservative
> The table counts every thread's read separately. In the 1 by 1 kernel all 32 threads of a warp read the same `As` element, and the hardware serves that as one request, so the true ceiling of that kernel is higher than 25%. The conclusion holds: a thread tile of one is limited by shared-memory reads, and a larger tile moves the limit away.

The same picture explains Boehm's two rungs. His kernel 4, 1D blocktiling, computes a column of 8 outputs per thread on a 64 by 64 block tile and reaches 36.5% of cuBLAS. His kernel 5 makes the thread tile 8 by 8 and reaches 68.7%. The profile of kernel 5 still shows stalls waiting on the memory pipeline, which the next two sections reduce.

### What the registers cost

An 8 by 8 thread tile keeps 64 accumulators, plus 8 values of $A$ and 8 of $B$ in flight, so 80 registers hold data before you count addresses and loop counters. Expect on the order of 100 to 130 registers per thread, and read the real number from `nvcc -Xptxas -v` or `cuobjdump --dump-resource-usage`. Assume 128 for the arithmetic: at 128 registers and 256 threads per block, one block uses 32,768 registers, so two blocks fit in an SM's 65,536: 16 resident warps, 25% occupancy.

That sounds low, and it is fine. [Chapter 4](/c/04-roofline-and-latency) showed that latency hiding can come from *instruction-level* parallelism instead of more threads, and here each thread has 64 independent FMAs between loads. Low occupancy is a cost only when there is nothing else to overlap, and a 64-FMA block of independent work is plenty. If the compiler spills the accumulators to local memory, though, the speedup disappears, so check that the register count stays at or below 255 with no spills.

## The complete kernel

This section assembles the pieces. It answers one question: how do the block tile, the thread tile, and the loads fit together in one kernel that works for any size?

The configuration is the one in Boehm's later kernels: a block tile of $B_M \times B_N = 128 \times 128$ with $B_K = 8$, and a thread tile of $T_M \times T_N = 8 \times 8$. The block has $(128 / 8) \times (128 / 8) = 256$ threads. Thread `tid` owns the sub-tile at:

```cuda
constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
constexpr int NT = (BM / TM) * (BN / TN);          // 256 threads

const int threadRow = tid / (BN / TN);             // 0..15
const int threadCol = tid % (BN / TN);             // 0..15
// rows rowBase + threadRow*TM + [0, TM), cols colBase + threadCol*TN + [0, TN)
```

The block's work for one step of $K$ has three parts.

**Load.** The block stages the $128 \times 8$ tile of $A$ and the $8 \times 128$ tile of $B$ in shared memory, 1,024 floats each, four per thread. It writes `As` transposed, `As[k*BM + m]`, so the $T_M$ values a thread needs for one $k$ are contiguous. Out-of-range elements load as zero:

```cuda
for (int i = tid; i < BM * BK; i += NT) {
    const int m = i / BK, k = i % BK;
    const int gr = rowBase + m, gk = k0 + k;
    As[k * BM + m] = (gr < M && gk < K) ? A[gr * K + gk] : 0.f;
}
for (int i = tid; i < BK * BN; i += NT) {
    const int k = i / BN, n = i % BN;
    const int gk = k0 + k, gc = colBase + n;
    Bs[k * BN + n] = (gk < K && gc < N) ? B[gk * N + gc] : 0.f;
}
__syncthreads();
```

**Compute.** For each $k$ in the step, read the thread's values into registers and do the outer product:

```cuda
#pragma unroll
for (int kk = 0; kk < BK; ++kk) {
    for (int i = 0; i < TM; ++i) regA[i] = As[kk * BM + threadRow * TM + i];
    for (int j = 0; j < TN; ++j) regB[j] = Bs[kk * BN + threadCol * TN + j];
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) acc[i][j] += regA[i] * regB[j];
}
__syncthreads();
```

**Store.** After the last step, each thread writes its $T_M \times T_N$ results, skipping any row or column outside $C$.

The loops over `i` and `j` have compile-time bounds, so `nvcc` unrolls them and keeps `acc`, `regA`, and `regB` in registers. If the loop bounds depended on a runtime value, the arrays would go to local memory and the kernel would collapse. This is the one place where being explicit about constants decides the performance.

### Sizes that don't divide the tile

The kernel is correct for any $M$, $N$, and $K$ because every tail has a rule.

- A partial tile in $K$: loads past $K$ give `0.f`, so the extra products are 0.
- A partial block in $M$ or $N$: rows or columns past the edge load `0.f` and compute into registers that are never stored.
- The store: `if (gr < M && gc < N)`.
- Every thread, including the idle ones, runs both `__syncthreads()` in every iteration.

The lab tests shapes such as $1 \times 1 \times 1$, $5 \times 1 \times 9$, $129 \times 127 \times 130$, and $1023 \times 1025 \times 1021$ for exactly this reason: each of them leaves a different tail.

### The grid and the waves

For $4096^3$ the grid has $(4096 / 128)^2 = 1024$ blocks. If two blocks fit per SM, 216 run at once, so 1024 blocks take $1024 / 216 = 4.74$ waves, which is 5 waves with the last one 74% full (160 of 216 slots). The tail costs about 5% of the work in the best case, $1024 / (5 \times 216) = 94.8\%$ efficiency. Chapter 4 calls this wave quantization, and it is one reason the sizes the labs report are not exactly proportional to the work.

## Vectorized loads and the transposed tile

The global loads in the last kernel are 4 bytes each. The section answers one question: what do 16-byte loads change, and what do they require?

A thread can load 16 bytes at once, a `float4`, in one instruction. That is a quarter of the load instructions for the same bytes, and chapter 3's copy kernel showed the same effect on bandwidth. Boehm's kernel 6 vectorizes the global loads and the shared-memory accesses and reaches 78.4% of cuBLAS, a gain of 1.14 times over the 2D blocktiling kernel.

For the $128 \times 8$ tile of $A$, each row has 8 floats, which is two `float4`. The 256 threads load 256 `float4`, one each, and scatter the four lanes into the transposed layout:

```cuda
const int m = tid / 2, k4 = (tid % 2) * 4;
const int gr = rowBase + m, gk = k0 + k4;
float4 va = make_float4(0.f, 0.f, 0.f, 0.f);
if (gr < M && gk < K) va = *reinterpret_cast<const float4*>(&A[gr * K + gk]);
As[(k4 + 0) * BM + m] = va.x;
As[(k4 + 1) * BM + m] = va.y;
As[(k4 + 2) * BM + m] = va.z;
As[(k4 + 3) * BM + m] = va.w;
```

The transpose is what makes the compute loop fast: for a fixed `kk`, the thread's $T_M = 8$ values sit at 8 consecutive words, so two 128-bit shared loads read them. The cost is that the four scalar stores into `As` land in different rows of the transposed tile, with some bank conflicts. Boehm's attempts to remove the conflicts made the kernel slower on his card, so his final kernels keep them. [Chapter 8](/c/08-shared-memory-and-transpose)'s padding and swizzles are the tools if you want to try.

### What vectorization requires

A `float4` load needs a 16-byte aligned address, and a thread must not read a `float4` that straddles the end of the matrix. For row-major $A$ with $K$ columns that means $K$ must be a multiple of 4, so each row starts at a 16-byte boundary and a group of four never crosses the edge. The same holds for $B$ and $C$ with $N$. A fresh `cudaMalloc` allocation is aligned to at least 256 bytes, which covers the base pointer.

A real kernel handles both cases: take the vector path when $K \bmod 4 = 0$, $N \bmod 4 = 0$, and the pointers are aligned, and fall back to the scalar-load path otherwise. The lab's reference solution compiles one kernel template for both and chooses on the host.

> [!WARNING] Misaligned float4 fails loudly or silently
> A `float4` load at an address that isn't a multiple of 16 raises a "misaligned address" error on the GPU, or on some paths reads the wrong bytes. A matrix that is a view into a larger tensor can start at any offset, so check the pointer, not just the shape.

## Warptiling and the hierarchy

The last published rungs reorganize the work between the block tile and the thread tile. The section answers one question: what is the full hierarchy, and what does each level reuse?

A GEMM kernel is a four-level tiling:

| Level | Tile it owns | What it reuses | Where the operands live |
|---|---|---|---|
| Grid | The whole $M \times N$ output | Nothing | Global memory |
| Thread block | A $B_M \times B_N$ tile of $C$, for example 128 by 128 | Each loaded $A$ and $B$ value is used by many threads | Shared memory |
| Warp | A $W_M \times W_N$ part of the block tile, for example 64 by 32 | Shared-memory values reused across the warp's thread tiles | Shared memory, read into registers |
| Thread | A $T_M \times T_N$ tile, for example 8 by 8 | Each register value is used $T_N$ or $T_M$ times | Registers |

At each level a tile of the output is the unit of ownership, and the operands that tile needs are the unit of data movement. Moving data to the next level up costs one load, and the next level down reuses it many times. CUTLASS uses the same levels, a thread block tile, a warp tile, and a thread or instruction tile, and [chapter 13](/c/13-cutlass-and-cute) formalizes them.

The warp level is where our kernel is implicit. Warp 0 of the 256-thread block holds `tid` 0 to 31, so `threadRow` is 0 or 1 and `threadCol` runs over 0 to 15. It covers a $16 \times 128$ slab of the block tile, a long thin shape. Per $k$ it reads 2 distinct 8-value groups of `As` and 16 of `Bs`: 18 groups for 32 thread tiles.

Compare an explicit warp tile of 32 by 64, four thread tiles tall and eight wide. It needs 4 groups of `As` and 8 of `Bs`, 12 in all, for the same 32 thread tiles. A squarer warp tile reads fewer distinct shared-memory addresses per FMA, which is the same argument as the block tile's: the intensity of a tile grows with $B_M B_N / (B_M + B_N)$, and a square tile maximizes it for a given area.

Boehm's kernel 10 makes this explicit and reaches 93.7% of cuBLAS on his A6000. The warp tile also gives the compiler a regular pattern of loads to overlap with FMAs, and it keeps each warp's accesses within a smaller window of the shared tile. His kernel 9 autotunes the tile parameters and reaches 84.8% before warptiling, and the best configuration depends on the GPU: Boehm reports that on an A100 his autotuner preferred a 64 by 64 block tile with $B_K = 16$ and a 4 by 4 thread tile, which are the 4 by 4 tile of our table that just meets the shared-memory budget. A 128 by 128 tile with an 8 by 8 thread tile is a good default, and not a universal one.

### The whole ladder

The following table lists the kernels in order, the resource that bounds each one, and Boehm's result as a percentage of cuBLAS on his RTX A6000. Treat the percentages as the shape of the climb, not as A100 predictions:

| Step | What changed | Bound by | Boehm, % of cuBLAS |
|---|---|---|---|
| 1. Naive | One thread per output | Uncoalesced global loads | 1.3% |
| 2. Coalescing | Warp walks along a row of $C$ | Global traffic, no reuse | 8.5% |
| 3. Shared-memory tiling | 32 by 32 tiles staged in shared memory | Shared-memory reads, 8 B per FMA | 12.8% |
| 4. 1D blocktiling | Each thread computes a column of 8 outputs | Shared-memory reads, 4.5 B per FMA | 36.5% |
| 5. 2D blocktiling | 8 by 8 thread tile, outer product | Memory pipeline stalls | 68.7% |
| 6. Vectorized loads | `float4`, transposed `As` | Bank conflicts, no double buffering | 78.4% |
| 9. Autotuning | Best tile parameters for the card | Tuning | 84.8% |
| 10. Warptiling | Explicit warp tile between block and thread | Remaining latency and issue slots | 93.7% |

Boehm's kernels 7 and 8 removed bank conflicts and were slower, so his post doesn't report them. The table keeps his numbering for that reason.

## The ceiling and what comes next

The best kernel in this chapter is capped. The section answers one question: where does the CUDA-core road end?

Cap one is the fp32 peak, 19.5 TFLOP/s: the floor of 7.05 ms for $4096^3$. cuBLAS's fp32 kernel and the hand-written kernels all compete for the same 128 FLOP per clock per SM. The published worklogs show that a hand-written kernel can get within a few percent of cuBLAS. Salykov's independent worklog on an RTX 3090 reports a 128 by 128 by 8 kernel averaging 3 to 4% faster than cuBLAS with unlocked clocks, at about 12% more power. The practical reading: past roughly 90% of cuBLAS the remaining gains are small and card-specific.

Cap two is the arithmetic itself. The A100's TF32 tensor cores peak at 156 TFLOP/s, $156 / 19.5 = 8$ times the fp32 CUDA-core peak, and the bf16 and fp16 tensor cores at 312 are 16 times. A kernel that doesn't use them gives up that factor, which is why a framework's production GEMMs use them wherever the precision allows. The tiling you built stays, and the inner loop changes: instead of 64 scalar FMAs per thread, a warp issues a matrix instruction over a fragment. [Chapter 11](/c/11-tensor-cores) builds that, and [chapter 12](/c/12-pipelining-and-async-copies) overlaps the global loads with it using asynchronous copies. [Chapter 14](/c/14-triton) shows the same structure in Triton, where the compiler picks the thread tile for you.

## What goes wrong

- **A `__syncthreads()` inside a divergent branch.** Wrapping the loop body in `if (row < M && col < N)` makes some threads skip the barrier. The block hangs or reads half-loaded tiles. Guard the loads with the ternary, and the store with the `if`.
- **Skipping the zero-fill.** Uninitialized shared memory in the K tail produces wrong numbers that depend on what ran before, and sometimes NaNs. Every out-of-range element must be written, as `0.f`.
- **Comparing against TF32 cuBLAS.** If TF32 is on, cuBLAS runs on the tensor cores at up to 156 TFLOP/s and your kernel looks like 10% of it. Set `torch.backends.cuda.matmul.allow_tf32 = False`, as the lab does, and check that the baseline stays under the 19.5 TFLOP/s fp32 peak.
- **`float4` on an unaligned shape.** The vector path needs $K$ and $N$ divisible by 4. A test size such as 1021 or 513 must take the scalar path, and a view with an odd offset needs the pointer check.
- **Index overflow.** `row * K + k` in `int` overflows at $2^{31}$ elements. The lab's wrapper rejects matrices that large. For bigger ones, use 64-bit indices.
- **Trusting a number above the peak.** A kernel reporting more than 19.5 TFLOP/s on the CUDA cores has a measurement bug or a wrong answer, and the lab fails it. The naive kernel is slow, so the lab times it only a few times.
- **Comparing percentages from different cards.** An 80% of cuBLAS on an RTX A6000 and 80% on an A100 are both honest, but they are not the same number of FLOP/s, and the A100 cuBLAS baseline can sit higher or lower against its own peak.

> [!RECAP]
> - A GEMM does $2MNK$ FLOPs and needs at least $4(MK + KN + MN)$ bytes. At $4096^3$ in fp32, that is 137.4 GFLOP on 201.3 MB, an intensity of 682.7, and a floor of 7.05 ms at 19.5 TFLOP/s.
> - A block tile of $B_M \times B_N$ has intensity $B_M B_N / (2 (B_M + B_N))$. A 64 by 64 tile clears the fp32 ridge point, and 128 by 128 does with room to spare.
> - Each rung of the ladder removes one limit: coalescing fixes the access pattern, shared-memory tiling cuts global loads, and register tiling cuts shared-memory reads to $4 (T_M + T_N) / (T_M T_N)$ bytes per FMA.
> - The inner loop of a fast kernel is an outer product on registers. Zero-filled tails, guards on loads and stores, and `__syncthreads()` outside every guard make it correct for any size.
> - The hierarchy is block tile, warp tile, thread tile, registers, and Boehm's worklog reaches 93.7% of cuBLAS on an RTX A6000 with it. The CUDA-core ceiling is 19.5 TFLOP/s.

## Check your understanding

> [!QUESTION] A GEMM has $M = 2048$, $N = 2048$, and $K = 64$. What is its intensity at minimum traffic in fp32, and is it above the A100's fp32 ridge point?
> FLOPs: $2 \times 2048 \times 2048 \times 64 = 536.9$ MFLOP. Bytes: $4 \times (2048 \times 64 + 64 \times 2048 + 2048 \times 2048) = 4 \times (131{,}072 + 131{,}072 + 4{,}194{,}304) = 17.8$ MB. The intensity is $536.9 / 17.8 = 30.1$ FLOP/byte, about three times the 10.08 ridge point, so the GEMM is still arithmetic bound, but with a much smaller margin than the cube. A small $K$ makes the output write dominate the traffic.

> [!QUESTION] Why does the naive kernel with `row` taken from `threadIdx.x` read memory badly, and what does swapping the roles change?
> The 32 threads of a warp are 32 different rows, so their loads of `A[row*K + k]` are $K$ floats apart, one sector each: 32 sectors for 32 useful floats, an 8 times read amplification. After the swap they share a row and cover 32 consecutive columns, so the `A` load is a one-sector broadcast and the `B` and `C` accesses are 128 contiguous bytes. A warp moves 5 sectors instead of 33 per iteration.

> [!QUESTION] The tiled kernel uses two `__syncthreads()` per step. What goes wrong if you delete the second?
> A fast warp finishes its dot products for the step and moves on to load the next tile into `As` and `Bs` while a slower warp is still reading the old one. The slow warp then reads a mixture of two steps and the result is wrong, nondeterministically. The second barrier guarantees every thread has finished reading before anyone overwrites.

> [!QUESTION] How many bytes of shared memory does a thread read per FMA with a 4 by 8 thread tile, and does it fit the A100's budget?
> A thread reads $4 + 8 = 12$ values, 48 bytes, for $4 \times 8 = 32$ FMAs: 1.5 bytes per FMA. The budget is 128 bytes per clock over 64 FMAs per clock, 2 bytes per FMA, so it fits, with a quarter of the budget to spare.

> [!QUESTION] A kernel with a 128 by 128 block tile and 8 by 8 thread tiles runs 256 threads per block at 128 registers per thread. How many blocks fit per SM by registers, and what is the occupancy?
> One block needs $256 \times 128 = 32{,}768$ registers, so $65{,}536 / 32{,}768 = 2$ blocks per SM. That is 512 threads, 16 warps of the 64 an SM can hold: 25% occupancy. The kernel hides latency with the 64 independent FMAs per thread instead of with many warps.

> [!QUESTION] Your SGEMM measures 17 TFLOP/s against a torch.matmul that measures 130 TFLOP/s. Is your kernel at 13% of cuBLAS?
> Not on the same footing. 130 TFLOP/s is far above the 19.5 TFLOP/s fp32 peak, so the baseline ran on tensor cores: TF32 is enabled. Turn it off with `torch.backends.cuda.matmul.allow_tf32 = False` and compare again. Against the fp32 cuBLAS kernel the figure is probably much closer.

> [!QUESTION] Why can't you use `float4` loads when $K = 1021$?
> A row of $A$ starts at `row * 1021 * 4` bytes, which isn't a multiple of 16 for most rows, so the address of a `float4` is misaligned, and a group of four can straddle the end of the row. The kernel must take the scalar path, or pad the matrix to a multiple of 4.

## Lab

> [!TRY] SGEMM from scratch
> Write fp32 GEMM kernels for $C = A B$ on row-major matrices, correct for any $M$, $N$, and $K$, and compare them with `torch.matmul` at TF32 off. The lab compiles your CUDA with `load_inline`, so the first run takes 30 to 60 seconds.
>
> - Fill in the four pure-Python helpers: FLOPs, minimum bytes, the intensity of a block tile, and the shared-memory bytes per FMA.
> - Write `tiled_kernel`, a 32 by 32 shared-memory tiling with zero-filled tails.
> - Write `regblock_kernel`, a 128 by 128 block tile with an 8 by 8 thread tile and a transposed `As`.
> - Optional: add `sgemm_coalesced` and a `sgemm_vectorized` that uses `float4` loads and falls back to scalar loads when $K$ or $N$ isn't a multiple of 4.
>
> The tests check every kernel against a float64 reference on shapes from $1 \times 1 \times 1$ to $1023 \times 1025 \times 1021$, then time each at $4096^3$. They report your GFLOP/s and your percentage of cuBLAS, and require at least 5% for the tiled kernel, 30% for the register-blocked one, and a register-blocked kernel at least twice as fast as the tiled one. Those thresholds are loose by design: they are less than half of Boehm's percentages for the same steps, because nobody has run this lab on an A100 yet. Your lab prints the actual figures.

## Further reading

- Simon Boehm, [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM) (2022). The progression this chapter follows, with profiler output at every step. His numbers are from an RTX A6000.
- Amanzhol Salykov, [Advanced Matrix Multiplication Optimization on NVIDIA GPUs](https://salykova.github.io/sgemm-gpu) (2025). An independent worklog on an RTX 3090 with a different order of optimizations.
- Andrew Kerr and others, [CUTLASS: Fast Linear Algebra in CUDA C++](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/) (NVIDIA, 2017). The block, warp, and thread tile hierarchy.
- NVIDIA, [Matrix Multiplication Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html). Tile and wave quantization, and the arithmetic intensity of GEMM.
- Kazushige Goto and Robert van de Geijn, [Anatomy of High-Performance Matrix Multiplication](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf) (2008). The cache-blocking theory that tiling copies.
- Lei Mao, [Optimizing Matrix Multiplication on a GPU](https://leimao.github.io/blog/CUDA-Matrix-Multiplication/) (2022). A concise tiled GEMM derivation.
- The companion course, [Your first CUDA kernel](https://qwen.ifkash.dev/c/12-your-first-cuda-kernel), for the launch and indexing basics this chapter assumes.
