---
title: Shared memory and the transpose
slug: 08-shared-memory-and-transpose
part: "Part 2 — CUDA patterns"
summary: What shared memory is, why its 32 banks can serialize a warp, and how a matrix transpose goes from a strided write to a copy-speed kernel with a tile, a padded row, and an XOR swizzle.
minutes: 150
gpu: true
objectives:
  - "Say where shared memory physically lives on an A100, how its carveout trades against L1, and how a block asks for more than 48 KB."
  - "Compute the bank of any address and the conflict degree of any warp access, including broadcasts and the odd-stride rule."
  - "Place __syncthreads() correctly, and name the race it prevents and the deadlock it can cause."
  - "Explain why a naive transpose reads well and writes badly, and why a copy is its speed limit."
  - "Write a shared-memory tiled transpose, then remove its bank conflicts with padding and with an XOR swizzle."
  - "Choose a tile and a padding amount for 2-byte elements, and say why one element of padding is not enough for bfloat16."
  - "Read a CuTe Swizzle<B,M,S> and say which address bits it mixes."
lab: 08-transpose
---

# Shared memory and the transpose

> [!TLDR]
> - Shared memory is on-chip SRAM inside each SM, carved out of the same 192 KB array as the L1 cache. It is a few
>   times faster to reach than L2 and, at 128 bytes per clock per SM, about 10x the bandwidth of HBM across the A100.
> - It has 32 banks, each 4 bytes wide. A warp's access takes one transaction when its lanes touch different banks,
>   or the same word, and N transactions when N different words share a bank.
> - A transpose moves the same bytes as a copy, so a copy sets its speed limit. The naive kernel coalesces its reads
>   and scatters its writes. Staging a tile in shared memory coalesces both.
> - The staged tile reads a column, which puts all 32 lanes in one bank. Padding each row by one word, or XORing the
>   column with the row, removes the conflict.
> - Your lab times four variants for float32 and bfloat16 on awkward shapes and reports each against the 1275 GB/s
>   copy ceiling.

A transpose looks too small to deserve a chapter. It does no arithmetic, and the whole job is to move each element
from `in[m][n]` to `out[n][m]`. But every model you train runs it constantly: the `K` in `Q @ K.T`, a weight stored
in the layout one kernel wants and read in the layout another wants, a switch from channels-first to channels-last,
the permutation in front of a mixture-of-experts layer. Each of those is a layout change, and a layout change is
a transpose in disguise.

It also happens to be the smallest kernel that needs shared memory. [Chapter 3](/c/03-memory-and-bandwidth) showed
that a warp must read and write contiguous addresses to use DRAM well. A transpose can't do both with registers
alone, because reading a row and writing a column are opposite access patterns. Shared memory resolves the
conflict, and then it introduces its own conflict, between the lanes of a warp, that you have to resolve too. The
tiled GEMM in [chapter 10](/c/10-gemm-from-scratch) and the FlashAttention kernel in
[chapter 16](/c/16-flash-attention) stage tiles the same way, so what you learn on a transpose carries straight over.

## Before you start

**Coalescing.** A warp's load or store reaches DRAM in 32-byte sectors, and it uses them well only when its 32
lanes touch contiguous addresses. [Chapter 3](/c/03-memory-and-bandwidth) derives this, and the companion course
measured the cost: a strided read ran 6.9x slower than a coalesced one on this card.

**Blocks and warps.** A block runs on one SM, its threads are scheduled in warps of 32, and the warp's lanes issue
the same instruction together ([chapter 2](/c/02-the-execution-model)). Threads in different warps of one block
can run at different times, which is why this chapter needs a barrier.

**The copy ceiling.** A plain device-to-device copy measured 1275 GB/s on the lab A100, against a rating of 1935
GB/s. The measurement is from [learn-inference, chapter 10](https://qwen.ifkash.dev/c/10-roofline). A memory-bound
kernel is judged against 1275, and this chapter's kernel is memory bound.

## What shared memory is

This section answers one question: where does shared memory live, and how do you ask for it?

Shared memory is a block of on-chip SRAM inside each streaming multiprocessor, and the threads of one block read and
write it directly. Nothing about it is cached or automatic. You decide what goes in, you decide when, and its
contents belong to the block and vanish when the block finishes. That makes it the GPU's scratchpad, in contrast to
the caches, which decide for themselves.

### One array, two jobs

On the A100 each SM has a single 192 KB SRAM array that serves as both the L1 data cache and shared memory. The
split is a *carveout*: the driver sets aside some of the array for shared memory, and what remains acts as L1.
The CUDA programming guide lists the supported shared-memory sizes for compute capability 8.0 as 0, 8, 16, 32, 64,
100, 132, and 164 KB. At the largest carveout, 164 KB is shared memory and the remaining 28 KB is L1.

A single block may use at most 163 KB, because CUDA reserves 1 KB of every block's allocation for its own use.
That reservation counts against the SM too. A block that asks for 48 KB is charged 49 KB, so 164 KB holds 3 of
them, not 3.4.

You rarely set the carveout, because the driver chooses one that fits the blocks it makes resident. You can
hint at a preference per kernel with the `cudaFuncAttributePreferredSharedMemoryCarveout` attribute, as a percentage
of the maximum, but treat it as a hint.

### How fast it is

Shared memory is close, and it is wide. Three numbers carry the argument.

| Quantity | A100 | How it is known |
|---|---|---|
| Latency | 29.0 cycles, about 20.6 ns at 1.41 GHz | Measured on an A100 PCIe 40GB by Luo and others; no 80GB figure exists |
| L2 latency, for comparison | 261.5 cycles | Same measurement |
| HBM latency, for comparison | 466.3 cycles | Same measurement |
| Peak bandwidth | 128 B per clock per SM, 19.5 TB/s for 108 SMs | Arithmetic: 32 banks x 4 B, below |

The bandwidth comes from the bank structure, and you can recompute it:

$$
\boxed{128\ \text{B/clock} \times 108\ \text{SMs} \times 1.41 \times 10^{9}\ \text{clock/s} = 19.5\ \text{TB/s}}
$$

That is about ten times the 1.935 TB/s of HBM ($19.5 / 1.935 = 10.1$), and it is a ceiling the whole GPU can hold
at once because each SM has its own array. Shared memory is fast enough that it usually isn't the limit, except
when a warp's access pattern serializes it. That serialization is the subject of the next section.

### Static and dynamic allocation

You get shared memory in two ways. A *static* allocation has a size known at compile time, and the compiler lays
it out:

```cuda
__global__ void k_static(const float* in, float* out) {
    __shared__ float tile[32][33];      // 32 * 33 * 4 = 4,224 bytes
    // ...
}
```

A *dynamic* allocation has a size chosen at launch. You declare an unsized `extern` array and pass the byte count as
the third launch parameter:

```cuda
__global__ void k_dynamic(const float* in, float* out) {
    extern __shared__ float buf[];      // size comes from the launch
    // ...
}

k_dynamic<<<grid, block, 4224>>>(in, out);   // third argument: bytes
```

Static allocations are limited to 48 KB per block. To use more, which a tensor-core GEMM does, you must use a
dynamic allocation and opt in before the launch:

```cuda
cudaFuncSetAttribute(k_dynamic,
                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                     100 * 1024);            // allow 100 KB for this kernel
k_dynamic<<<grid, block, 100 * 1024>>>(in, out);
```

Without the opt-in, a launch that asks for more than 48 KB fails with an invalid-argument error, which is easy to
mistake for a bug in your indexing. The opt-in can go up to the per-block limit, 163 KB on the A100.

> [!EXAMPLE] What the transpose tile costs the SM
> The lab's float32 tile is 32 rows of 33 floats: $32 \times 33 \times 4 = 4224$ bytes, plus the 1 KB reservation, so
> 5248 bytes per block. A 256-thread block means the 2048-thread limit allows 8 blocks per SM, and
> $8 \times 5248 = 41{,}984$ bytes, about a quarter of the 164 KB. Shared memory doesn't limit occupancy here.
> The bfloat16 tile is 64 rows of 66 elements, 8448 bytes, and 8 blocks use 75,776 bytes, still under half.

## Banks: 32 lanes, 32 ports

This section answers: when does a warp's access to shared memory take one step, and when does it take many?

Shared memory is divided into 32 *banks*. Successive 32-bit words go to successive banks, and each bank can
serve one 32-bit word per clock. A warp has 32 lanes, so if each lane reads a word from a different bank, the whole
warp is served in one clock, which is the 128 bytes per clock above. The bank of a byte address follows from those
two facts:

$$
\boxed{\text{bank}(a) = \left\lfloor \frac{a}{4} \right\rfloor \bmod 32}
$$

NVIDIA's documents state the two facts, and the formula is their direct consequence. Address 0 is in bank 0, address
4 in bank 1, address 124 in bank 31, and address 128 wraps back to bank 0.

### The conflict rule

Lanes that need different words in the same bank can't be served together. The bank handles one word per clock, so
the hardware splits the warp's request into as many transactions as the busiest bank has distinct words. That count
is the *conflict degree*, and an access with degree $N$ is an $N$-way bank conflict.

Two exceptions keep the rule from being too harsh:

- **Broadcast.** Lanes that read the *same* word don't conflict. All 32 lanes reading one word is one transaction,
  not 32.
- **Different bytes of one word.** Two lanes that touch different halves of one 32-bit word are touching the same
  word, so they share a transaction. This is why 32 consecutive bfloat16 values, which occupy 16 words, are conflict
  free.

For a write, lanes that hit the same word don't conflict either, but only one of them wins, so you get one value
and not a combination. You avoid it by never writing one word from two lanes.

### A stride is a gcd

The commonest pattern is a strided one: lane $l$ reads word $s \cdot l$ for a stride of $s$ words. The conflict
degree of that pattern depends only on $\gcd(s, 32)$:

| Pattern | Word stride $s$ | Banks touched | Degree |
|---|---|---|---|
| 32 consecutive floats | 1 | 32 | 1 |
| Every other float | 2 | 16 | 2 |
| Every fourth float | 4 | 8 | 4 |
| One float per 32-float row (a column of a 32-wide tile) | 32 | 1 | 32 |
| One float per 33-float row (a column of a padded tile) | 33 | 32 | 1 |
| All lanes read one float | 0 | 1 | 1, a broadcast |

The rule it encodes is that ==an odd stride is conflict free, and a power-of-two stride is the worst case==. An
odd number shares no factor with 32, so the 32 lanes land in 32 different banks. That single observation explains
padding, and it is half the reason to avoid tile widths that are powers of two without a fix.

> [!DEEPDIVE] Why the degree is gcd(s, 32)
> Lane $l$ reads word $s l$, which is in bank $(s l) \bmod 32$. Two lanes $l$ and $l'$ share a bank when
> $s(l - l') \equiv 0 \pmod{32}$. Write $g = \gcd(s, 32)$. That congruence holds exactly when $l - l'$ is a
> multiple of $32 / g$. The lanes $0, \dots, 31$ therefore split into sets of lanes that differ by multiples of
> $32/g$, and each set has $32 / (32/g) = g$ lanes, all in one bank.
>
> Those $g$ lanes read $g$ different words, because $s l \ne s l'$ when $l \ne l'$ and $s \ne 0$. So the busiest bank
> holds $g$ distinct words, and the degree is $g$. For $s = 32$, $g = 32$ and every lane is in one bank. For $s = 33$,
> $g = 1$. For $s = 2$, $g = 2$.

### Wider accesses

A thread can load 8 bytes (`float2`) or 16 bytes (`float4`) from shared memory in one instruction. A warp-wide 16-byte
access moves $32 \times 16 = 512$ bytes, and the banks serve 128 bytes per clock, so the floor is four
transactions. Nsight Compute calls each one a *wavefront*, and it reports the wavefronts it counted and the minimum
the access needed. A conflict-free 128-bit access takes four wavefronts, not one, and a conflict is anything above
four. For an 8-byte access the floor is two.

The practical consequence is that the conflict-free condition for a wide access is about groups of lanes, not the
whole warp. Which lanes share a wavefront, and how the profiler counts broadcasts across them, depends on the
architecture, so check the wavefront counts in Nsight Compute rather than trusting a hand count for 128-bit
patterns. The 4-byte case, which this chapter's lab uses for float32, and the 2-byte case for bfloat16, follow the
simple rule above.

The next figure applies the rule to a column read of the transpose's tile, the access that causes all the trouble
below. Pick a layout and an element size, and drag the column.

```viz
08-bank-map
```

With the initial state, a plain float32 tile, all 32 lanes land in the one bank that holds the chosen column. Switch
to padded, and the 32 words spread across 32 banks.

## Synchronizing a block

This section answers: when can one thread safely read what another thread wrote to shared memory?

Threads in a block are scheduled as warps, and different warps run at different times. If thread 5 writes a
shared-memory word and thread 70 reads it, nothing orders the two unless you do. The read could run first and see
stale data, which is a race.

`__syncthreads()` is the block-wide barrier that orders them. No thread continues past it until every thread of the
block has reached it, and every shared-memory write that came before it is visible to the whole block after it. The
pattern for a staged tile has three beats:

```cuda
tile[...] = in[...];     // 1. every thread writes its piece
__syncthreads();         // 2. wait until all pieces are written
out[...] = tile[...];    // 3. now any thread may read any piece
```

Two rules keep the barrier safe.

**Every thread of the block must reach the same barrier.** A `__syncthreads()` inside a branch that only some
threads take is undefined and usually deadlocks, because the threads that skipped the branch never arrive, and the
ones that took it wait forever. The same happens when a thread returns early. This matters for edge tiles: a thread
whose element is out of bounds must skip its load and store, but it must not skip the barrier.

```cuda
// Wrong: threads past the edge return before the barrier.
if (m >= M || n >= N) return;
tile[r][c] = in[m * N + n];
__syncthreads();

// Right: guard the work, not the barrier.
if (m < M && n < N) tile[r][c] = in[m * N + n];
__syncthreads();
```

**A second barrier protects reuse.** If a block loops, loading a new tile into the same shared array each
iteration, then a thread that finishes reading the old tile early could write the new one over data a slower thread
hasn't read. You need a barrier after the reads and before the next write. The lab's kernel handles one tile per
block, so it has one barrier, but the chapters on GEMM have two per iteration.

> [!NOTE] A barrier inside a warp
> Lanes of one warp run in lockstep on modern hardware only as far as the programming model promises, and
> independent thread scheduling since Volta means you shouldn't rely on more. `__syncwarp()` is the cheap
> barrier among the lanes of one warp. The chapters on reductions use it. This chapter doesn't need it.

## The transpose and its speed limit

This section answers: how fast can a transpose possibly be?

Define the operation. The input is a row-major matrix with $M$ rows and $N$ columns, so element $(m, n)$ lives at
`in[m * N + n]`. The output has $N$ rows and $M$ columns, and element $(n, m)$ lives at `out[n * M + m]` and holds
the value of `in[m * N + n]`.

There is no arithmetic, so the kernel is memory bound ([chapter 4](/c/04-roofline-and-latency)), and the time is
set by the bytes moved. Every element is read once and written once. For elements of $s$ bytes:

$$
\boxed{\text{bytes} = 2 \, M N s}
$$

That is exactly the traffic of a copy of the same matrix. So ==a transpose can't beat a copy, and the best a
transpose kernel can do is match one==. The 1275 GB/s copy ceiling is its target, and this is how you know when to
stop optimizing.

> [!EXAMPLE] The time floor for an 8192 x 8192 transpose
> A float32 matrix of $8192 \times 8192$ is $8192^2 \times 4 = 268{,}435{,}456$ bytes, 268.4 MB. The transpose moves
> that twice, 536.9 MB. At the measured 1275 GB/s that takes $536.9 \times 10^{6} / 1275 \times 10^{9} = 421\ \mu\text{s}$.
> At the rated 1935 GB/s it would take 277 $\mu$s, a floor nothing reaches. The same matrix in bfloat16 is half the
> bytes, so 210.5 $\mu$s at 1275 GB/s. Your lab times these shapes, so the 421 and 210.5 are the predictions to check.

The matrix is also far larger than the 40 MiB L2, so the lab doesn't need to flush the cache: every byte comes from
HBM.

## The naive kernel

This section answers: what goes wrong when each thread writes straight to its transposed position?

The obvious kernel gives each thread one element. Thread $(x, y)$ of the block reads its input element and writes the
output element in the transposed place. The lab's version handles a $32 \times 32$ tile per block with a $32 \times 8$
thread block, so each thread loops over four rows:

```cuda
template <typename T, int TILE>
__global__ void transpose_naive(const T* __restrict__ in, T* __restrict__ out, int M, int N) {
    const int n0 = blockIdx.x * TILE + threadIdx.x;   // input column
    const int m0 = blockIdx.y * TILE + threadIdx.y;   // input row
    for (int j = 0; j < TILE; j += 8) {
        for (int i = 0; i < TILE; i += 32) {
            const int m = m0 + j, n = n0 + i;
            if (m < M && n < N)
                out[(size_t)n * M + m] = in[(size_t)m * N + n];
        }
    }
}
```

Look at the two accesses of one warp. The warp's lanes have consecutive `threadIdx.x`, so consecutive $n$. The read
`in[m * N + n]` is 32 consecutive floats: 128 contiguous bytes, four full sectors, coalesced. The write
`out[n * M + m]` puts lane $l$ at row $n + l$ of the output, so lane $l$'s address differs from lane $l+1$'s by
$M \times 4$ bytes. For $M = 8192$ that is 32 KiB apart. The write is 32 scattered 4-byte stores.

A warp's store of one float per lane, to 32 different sectors, touches 32 sectors for 128 useful bytes. The sector
traffic at the cache interface is $32 \times 32 = 1024$ bytes against 128 bytes of data, 8x. The L2 can merge
partial sectors from different warps before they go to DRAM, so the amplification that reaches HBM is smaller than
8x and depends on how soon neighbors arrive. The roofline can't predict it tightly, so measure it: your lab reports
the naive kernel's GB/s.

Published measurements show the same shape on other cards. The table lists the naive transpose as a fraction of a
copy on the same card and shape. Different cards, so compare the fractions, not the GB/s.

| Source and card | Copy, GB/s | Naive transpose, GB/s | Naive as a fraction of copy |
|---|---|---|---|
| [Harris, NVIDIA](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/), Tesla M2050 | 105.2 | 18.8 | 18% |
| Harris, Tesla K20c | 136.0 | 55.3 | 41% |
| [Colfax](https://research.colfax-intl.com/tutorial-matrix-transpose-in-cutlass/), H100 PCIe, 32768 x 32768 | 1783 | 424.1 | 24% |

The naive transpose reaches roughly a fifth to two-fifths of a copy on each of these cards. The A100 will land
somewhere of its own, and it won't be near the ceiling.

## Staging through shared memory

This section answers: how does a tile in shared memory make both global accesses coalesced?

The problem is that a row of the input must become a column of the output. Nothing in registers can fix that,
because each thread holds only its own elements. A tile in shared memory can. Let a block own a $T \times T$
tile of the input, read it into shared memory *by rows*, then write it out *by rows* of the output, which are
columns of the tile. Both global accesses are contiguous. The reshuffle happens on-chip, where a strided access
costs clocks instead of DRAM sectors.

The kernel has two phases around a barrier:

```cuda
template <typename T, int TILE, int LAYOUT>
__global__ void transpose_tiled(const T* __restrict__ in, T* __restrict__ out, int M, int N) {
    __shared__ T tile[tile_elems<T, TILE, LAYOUT>()];
    const int n_base = blockIdx.x * TILE;     // first input column of this tile
    const int m_base = blockIdx.y * TILE;     // first input row of this tile

    // Phase 1: lane x reads consecutive input columns of one row.
    for (int j = 0; j < TILE; j += 8)
        for (int i = 0; i < TILE; i += 32) {
            const int r = threadIdx.y + j, c = threadIdx.x + i;
            const int m = m_base + r,      n = n_base + c;
            if (m < M && n < N)
                tile[smem_index<T, TILE, LAYOUT>(r, c)] = in[(size_t)m * N + n];
        }
    __syncthreads();

    // Phase 2: lane x writes consecutive output columns of one output row.
    for (int j = 0; j < TILE; j += 8)
        for (int i = 0; i < TILE; i += 32) {
            const int o_row = threadIdx.y + j;   // input column inside the tile
            const int o_col = threadIdx.x + i;   // input row inside the tile
            const int n = n_base + o_row, m = m_base + o_col;
            if (n < N && m < M)
                out[(size_t)n * M + m] = tile[smem_index<T, TILE, LAYOUT>(o_col, o_row)];
        }
}
```

Three details are worth reading twice. The output tile sits at the swapped block coordinates, so the output row
of an element is `n_base + o_row` and its column is `m_base + o_col`. The roles of $x$ and $y$ swap in phase 2: lane
$x$ now varies the tile's *row* index `o_col`. And the barrier is outside every `if`, so threads whose elements are
out of bounds still arrive.

The next figure steps one row of a small tile through the five moments. Notice that the global read and the global
write each move a row, and the turn happens in the shared tile.

```viz
08-tile-path
```

The shared-memory writes of phase 1 go along a row of the tile: fixed `r`, consecutive `c`, so 32 consecutive words
and one transaction. The shared-memory reads of phase 2 go along a column: lane $x$ reads `tile[x][o_row]`, so the
lanes' rows differ and the column is fixed. For the plain layout, `smem_index(r, c) = r * 32 + c`, so lane $x$ reads
word $32 x + c$. That is a word stride of 32, which is the worst entry in the stride table.

> [!KEY] The tile turns the problem, and the turn is a bank conflict
> Shared memory fixes the global access pattern by moving the strided access on-chip. The plain tile has a row
> pitch of 32 words, so the strided access lands every lane in one bank: a 32-way conflict on every shared read.
> The kernel is correct, and the lab's tiled variant will pass its tests, but it isn't fast.

### What a 32-way conflict costs

A conflicted read takes 32 transactions instead of one. Does that matter for a kernel that waits on HBM? Do the
bookkeeping per SM. The HBM-bound copy rate of 1275 GB/s spread over 108 SMs at 1.41 GHz is

$$
\frac{1275 \times 10^{9}\ \text{B/s}}{108 \times 1.41 \times 10^{9}\ \text{clock/s}} = 8.37\ \text{B per clock per SM}.
$$

Each float32 element moved costs 8 HBM bytes, 4 in and 4 out, so an SM must pass about $8.37 / 8 = 1.05$ elements per
clock. The shared array handles one transaction per clock. In the tiled kernel, 32 elements need one write
transaction and, with a 32-way conflict, 32 read transactions, so 33 transactions per 32 elements. That caps the
SM at $32 / 33 = 0.97$ elements per clock.

Take the ratio of what the shared array can pass to what HBM can feed:

$$
\frac{0.97}{1.05} = 0.93
$$

Under the best overlap, the conflicted float32 tile can run at 93% of the copy ceiling. Real kernels don't overlap
perfectly, because a block that waits at a barrier isn't loading, so expect less. The point of the model is that the
shared array is now *on the critical path*, with almost no slack, where before it had a margin of about 15x. Published
results agree in direction. Harris's coalesced-but-conflicted kernel reached 51.3 GB/s against a 105.2 GB/s copy on
a Tesla M2050 (49%), and Colfax's conflicted shared-memory transpose reached 1255 GB/s against a 1783 GB/s copy on
an H100 PCIe (70%).

For bfloat16 the model is harsher. An element is 4 HBM bytes, so the SM must pass $8.37/4 = 2.09$ elements per clock,
and the conflicted shared array still passes at most 0.97. That is $0.97 / 2.09 = 0.46$: the unpadded bfloat16 tile
can't exceed about 46% of the copy ceiling, however well it overlaps. Your lab reports `bf16_tiled_gbs`, and the
prediction is that it lands below half the ceiling.

## Padding the tile

This section answers: how do you make the column read conflict free without changing the algorithm?

Make the rows 33 words apart instead of 32. A tile declared `[32][33]` has a row pitch of 33 words, and the element
at row $r$, column $c$ is at word $33 r + c$. Its bank is

$$
(33 r + c) \bmod 32 = (r + c) \bmod 32.
$$

For a fixed column $c$, the 32 rows $r = 0, \dots, 31$ give 32 different banks, because adding a constant to
a complete set of residues leaves it complete. The column read is now one transaction. A row write is still
conflict free, because consecutive columns are still consecutive words.

The general rule is: ==pad until the row pitch, counted in 32-bit words, is odd==. An odd pitch shares no factor
with 32, so the rows of any column land in different banks. The cost is the padding itself. The float32 tile grows
from 4096 to 4224 bytes, 3.1% more, and the extra words are never used.

Harris measured the effect on a Tesla M2050: the padded transpose reached 99.5 GB/s against a 105.2 GB/s copy, 95%,
up from 49% for the conflicted tile. That is the reference result for this technique, on hardware several generations
older. The lab's padded variant is the one to compare with.

Padding has a second cost that doesn't show up in a bandwidth table. A row pitch of 33 words breaks 16-byte
alignment of the rows: row 1 starts at byte 132, which isn't a multiple of 16. That rules out `float4` shared
accesses along the rows. For a transpose of scalars it doesn't matter, and for a GEMM kernel feeding tensor cores
with 128-bit shared loads, it does. That is the reason the field moved to swizzling.

### Padding two-byte elements

The same rule applies to bfloat16, but the unit is the word, not the element. The lab's bfloat16 tile is 64 columns
wide, so a row is 128 bytes, 32 words, and a column read of the plain tile hits one bank for all 32 lanes: 32-way.

The tempting fix is to pad by one element, a 65-wide row. Count words: a row is then 65 elements, 130 bytes,
32.5 words. Rows alternate between starting on a word boundary and in the middle of one. Even columns come out
conflict free, but odd columns don't: two rows land in the same bank at different words, a 2-way conflict. Your
Python model, `column_read_degree("padded", ...)`, shows it if you try pad = 1 for 2-byte elements. The fix is to pad
by a whole word, two elements, so the pitch is 66 elements, 33 words, odd. The lab's layout does that with
`4 / sizeof(T)` elements of padding.

## Swizzling the tile

This section answers: how do you remove the conflict without padding and without losing the rows' alignment?

Instead of moving rows apart, rearrange the columns within each row. Store the element at row $r$, column $c$ at column
$c \oplus (r \bmod 32)$ of the same row, where $\oplus$ is bitwise XOR. This is a *swizzle*. A row keeps its 32-word
size, so the tile is exactly 4096 bytes, and every row still starts on a 128-byte boundary.

Check the two accesses. A row write has a fixed $r$ and $c$ running over $0, \dots, 31$. XOR with a constant is
a permutation of $0..31$, so the 32 lanes still hit 32 different columns, which are 32 different banks. A column
read has a fixed $c$ and $r$ running over $0, \dots, 31$. The stored column $c \oplus r$ is different for every $r$,
because XOR with a fixed $c$ is also a permutation. So the 32 lanes hit 32 different banks too.

The following table shows it on an 8-bank stand-in. Each cell is the bank that holds the element at row $r$, column
$c$, for 8 rows and 8 banks. The plain layout has the same bank down every column. The swizzled layout has every
bank once in every row and once in every column.

| Row | Plain, banks of columns 0 to 7 | Swizzled ($c \oplus r$), banks of columns 0 to 7 |
|---|---|---|
| 0 | 0 1 2 3 4 5 6 7 | 0 1 2 3 4 5 6 7 |
| 1 | 0 1 2 3 4 5 6 7 | 1 0 3 2 5 4 7 6 |
| 2 | 0 1 2 3 4 5 6 7 | 2 3 0 1 6 7 4 5 |
| 3 | 0 1 2 3 4 5 6 7 | 3 2 1 0 7 6 5 4 |
| 4 | 0 1 2 3 4 5 6 7 | 4 5 6 7 0 1 2 3 |
| 5 | 0 1 2 3 4 5 6 7 | 5 4 7 6 1 0 3 2 |
| 6 | 0 1 2 3 4 5 6 7 | 6 7 4 5 2 3 0 1 |
| 7 | 0 1 2 3 4 5 6 7 | 7 6 5 4 3 2 1 0 |

Read the swizzled column 0 downward, 0 1 2 3 4 5 6 7, and column 3: 3 2 1 0 7 6 5 4. Each is a permutation, so no
bank repeats. XOR is the right operation because it is invertible, costs one instruction, and keeps every result
inside the row.

### The bfloat16 swizzle

Two bfloat16 values share a word, so the swizzle must keep each pair together. The lab multiplies the row by the
number of elements per word: the column becomes $c \oplus \big((r \bmod 32) \cdot 2\big)$. The multiplication by 2
leaves bit 0 of the column alone, so the two halves of a word stay together. The word within the row is then
$(c \gg 1) \oplus (r \bmod 32)$, which is the float32 formula applied to words. A column read of 32 rows gives 32 distinct words in 32 distinct banks, and a row write stays conflict free.

### The same idea in CuTe and TMA

A swizzle is a small piece of address arithmetic, and the libraries have a notation for it. CuTe, the layout library
under CUTLASS ([chapter 13](/c/13-cutlass-and-cute)), writes `Swizzle<B, M, S>`: it takes bits $[M + S, M + S + B)$ of
an offset and XORs them into bits $[M, M + B)$, leaving the lowest $M$ bits alone
([Lei Mao's explanation](https://leimao.github.io/blog/CuTe-Swizzle/)). Apply it to an element offset
`r * 32 + c` of the float32 tile. The column is bits 0 to 4 and the row is bits 5 to 9, so XORing the row into the
column is `Swizzle<5, 0, 5>`. The bfloat16 tile has 64 columns, so the column is bits 0 to 5 and the row starts at
bit 6. Keeping the lowest bit alone, and XORing 5 row bits into bits 1 to 5, is `Swizzle<5, 1, 5>`.

Colfax's CUTLASS transpose tutorial uses `Swizzle<5, 0, 5>` and reports 1779 GB/s against a 1783 GB/s copy on an H100
PCIe at $32768 \times 32768$. That is 99.8% of the copy, and a conflict-free tile with no padding. They also note that
padding the tile by one element works but wastes shared memory, which is the trade you can quantify above: 128 bytes
on a 4096-byte tile.

The scalar swizzle above isn't the one a GEMM uses. A tensor-core kernel loads 16-byte chunks, so the swizzle moves
whole 16-byte chunks and leaves the bits inside a chunk alone. In byte offsets that is `Swizzle<3, 4, 3>`: XOR the
three row bits at byte bits 7 to 9 into the chunk index at bits 4 to 6. This is the same kind of pattern that the
Hopper Tensor Memory Accelerator can apply in hardware while it writes a tile into shared memory, from a small fixed
set of patterns. [Chapter 12](/c/12-pipelining-and-async-copies) covers TMA, and chapter 13 builds CuTe's layout
algebra, so the lab here stays with the scalar swizzle, which is the clearest instance of the idea.

## Tiles for 4-byte and 2-byte elements

This section answers: why does the lab use a 32 x 32 tile for float32 and a 64 x 64 tile for bfloat16?

For float32, a $32 \times 32$ tile makes a tile row exactly 128 bytes, one cache line, and a warp's global access
covers it in one request. For bfloat16, a 32-wide tile row is 64 bytes. A warp's access would then touch two full
sectors, which is not wasteful at the sector level, so the global read and write still coalesce. But each request
moves half as much, and each thread has half the bytes in flight.

The lab uses 64 columns for bfloat16, so a row is again 128 bytes, and the block stays $32 \times 8$ threads. Each
thread then handles $64/8 \times 64/32 = 16$ elements instead of 4, which gives each thread more independent loads
to keep in flight. That is a design choice, not a law, and you can change `TILE` in the lab to compare. The 46%
prediction above is the reason bfloat16 needs the conflict removed even more than float32.

| | float32 | bfloat16 |
|---|---|---|
| Tile | 32 x 32 elements, 128 B per row | 64 x 64 elements, 128 B per row |
| Plain tile bytes | 4096 | 8192 |
| Column read, plain | 32-way | 32-way |
| Padding that works | 1 element (1 word) | 2 elements (1 word) |
| Padded tile bytes | 4224 | 8448 |
| Swizzle on the column | $c \oplus (r \bmod 32)$ | $c \oplus (2 (r \bmod 32))$ |

## Shapes that aren't multiples of the tile

This section answers: what changes when $M$ and $N$ aren't multiples of 32 or 64?

The grid is `ceil(N / TILE)` by `ceil(M / TILE)`, so the edge blocks cover tiles that hang over the matrix. Every
global access carries a guard, `m < M && n < N`. The barrier doesn't, as the section on synchronization explained,
and the guard in phase 2 tests the same condition as the guard in phase 1 so each element read from the tile was
written into it. An uninitialized tile word is never read.

Index arithmetic needs care. `n * M + m` for a matrix with more than $2^{31}$ elements overflows `int`, so the
kernels cast to `size_t` before the multiply. The lab's matrices are well under that limit, but the habit costs
nothing.

Odd sizes affect speed too. When $N$ isn't a multiple of 8 for float32, a row of the input doesn't start on a
32-byte boundary, so a warp's coalesced read touches one more sector than the aligned case. The extra sector is
about a quarter more traffic for the rows that straddle. Your lab times a $8191 \times 8193$ float32 transpose and
reports it as `fp32_ragged_padded_gbs`, next to the aligned `fp32_padded_gbs`. The check allows the ragged shape to
run at half the aligned speed, and expects less of a loss.

## Measuring against the copy

This section answers: how do you know a transpose is finished?

Time it with CUDA events ([chapter 0](/c/00-what-you-are-building)) and count every byte, read plus write, as
$2 M N s$. Time a plain copy of the same shape and dtype on the same device in the same run. That copy is the ceiling
for this card and this run, and it protects you from the lab landing on the SXM4 card, which is rated at 2039 GB/s
and so has a higher ceiling than the 1275 GB/s measured on the PCIe card.

The table sets out what each variant's limiter is, and what to expect. The "expect" column is a prediction from
the arithmetic above or from a published result on the named hardware, and your lab measures the real figure.

| Variant | What limits it | What to expect |
|---|---|---|
| Copy | HBM | The ceiling: 1275 GB/s measured in learn-inference |
| Naive | Scattered writes, up to 8x sector traffic | Far below the ceiling; 18% to 41% on the cards in the table above |
| Tiled, plain | A 32-way conflict on the shared read | float32: at most 93% by the model, less in practice; bfloat16: at most 46% |
| Padded | Almost nothing | Near the ceiling: 95% on a Tesla M2050 (Harris) |
| Swizzled | Almost nothing | Near the ceiling: 99.8% on an H100 PCIe (Colfax) |

The lab's pass thresholds are loose fractions of the 1275 GB/s ceiling: 50% for float32 and 40% for bfloat16, which
are 637.5 GB/s and 510 GB/s. A padded or swizzled kernel that reaches anything close to the table's last two rows
passes with room to spare.

Two practical notes from outside the lab. First, you often don't need a transpose at all. `x.t()` in PyTorch is a
view that swaps strides and moves no data. A kernel that can read with strides, or a matrix multiply that reads its
operand transposed (tensor cores can load a tile transposed from shared memory, as [chapter 11](/c/11-tensor-cores)
shows), skips the copy. Second, when a transpose is needed, such as `x.t().contiguous()`, the kernel PyTorch runs
is itself a tiled transpose. Colfax measured it at 860 GB/s on an H100 PCIe for the large shape, and 1694 GB/s with
`torch.compile`, against a 1783 GB/s copy. That is the gap you can close with a better kernel. Look at which kernel
your model actually launches ([chapter 5](/c/05-profiling)) before you write one.

## What goes wrong

- **The barrier inside a guard.** A `__syncthreads()` that sits inside `if (m < M && n < N)` deadlocks on edge
  tiles, where some threads fail the guard. The kernel passes on sizes that are multiples of the tile and hangs on
  the first ragged shape. Put the barrier outside every conditional.
- **Padding bfloat16 by one element.** The tile looks padded and the kernel is correct. The odd columns still conflict
  2-way. Pad a whole word, which is two elements for 2-byte types, and let your Python model confirm it before you
  look at timings.
- **The Python and CUDA layouts drift apart.** The bank model and the `smem_index` in CUDA describe the same layouts.
  If you change one and not the other, the model reports conflict-free while the kernel conflicts. Change both.
- **Writing the output tile at the input tile's position.** The output block for input tile $(b_y, b_x)$ goes at
  output tile $(b_x, b_y)$. Using the input block coordinates for the output writes each tile to the wrong place and
  fails every shape larger than one tile.
- **A second tile without a second barrier.** If you extend the kernel so a block loops over several tiles, a fast
  warp can overwrite the shared tile while a slow warp is still reading it. Add a barrier after the reads.
- **Timing a matrix that fits in L2.** A $1024 \times 1024$ float32 matrix is 4 MiB, and after the first iteration
  it lives in the 40 MiB L2, so the GB/s looks impossibly good. Use a shape much larger than L2, as the lab does,
  or flush the cache with `cuda_time(..., flush_l2=True)`.
- **Asking for more than 48 KB statically.** A static array over 48 KB fails to compile, and a dynamic one without the
  `cudaFuncSetAttribute` opt-in fails at launch. The lab's tiles are far smaller, but a GEMM tile isn't.
- **Judging against 1935.** The rating isn't reachable. A padded transpose at 90% of the 1275 GB/s copy is done, and
  hunting for the remaining 10% is the wrong job.

> [!RECAP]
> - Shared memory is per-SM on-chip SRAM sharing a 192 KB array with L1, up to 164 KB per SM and 163 KB per block,
>   with a 29-cycle latency and 128 bytes per clock per SM. Static allocations stop at 48 KB, and more needs a
>   dynamic allocation and `cudaFuncSetAttribute`.
> - There are 32 banks of 4 bytes. A warp access takes as many transactions as the busiest bank has distinct words;
>   the same word is a broadcast, and a word stride $s$ conflicts $\gcd(s, 32)$-way.
> - `__syncthreads()` separates a write phase from a read phase, and every thread in the block must reach it.
> - A transpose moves $2MNs$ bytes, the same as a copy, so the copy ceiling is its limit. The naive kernel writes
>   scattered; staging a tile makes both global accesses coalesced.
> - The staged tile conflicts 32-way on its column read. Pad to an odd pitch in words, which is 1 element for float32
>   and 2 for bfloat16, or XOR the column with the row.
> - Compare against the measured copy, not the rating, and stop when you're close.

## Check your understanding

> [!QUESTION] A warp reads `s[2 * lane]` from a float array in shared memory. How many transactions does it take, and why?
> It takes 2. Lane $l$ reads word $2l$, so the stride is 2 words and $\gcd(2, 32) = 2$. The even-numbered banks hold
> all 32 words, 16 banks with 2 distinct words each, so each bank serves 2 words and the access needs 2 transactions.

> [!QUESTION] All 32 lanes read `s[7]`. Is that a conflict?
> No. All the lanes request the same 32-bit word, so the hardware broadcasts it in one transaction. A conflict needs
> different words in the same bank.

> [!QUESTION] Why does the naive transpose read well and write badly, and does swapping which side is strided help?
> Consecutive lanes have consecutive columns $n$. The read `in[m*N + n]` is contiguous across lanes, and the write
> `out[n*M + m]` jumps by $M \times 4$ bytes between lanes, touching 32 sectors for 128 useful bytes. Swapping the
> indices makes the read strided and the write contiguous, which just moves the problem, because the global access
> that has to turn rows into columns is strided on one side either way. The tile fixes that by turning the pattern
> inside shared memory instead.

> [!QUESTION] A 32 x 32 float tile has a 32-way conflict on its column read. Why does padding to 33 columns fix it, and what would 34 do?
> With 33 columns, the word index is $33r + c$, whose bank is $(r + c) \bmod 32$, a different bank for each of the
> 32 rows. With 34, the stride is $\gcd(34, 32) = 2$, so the column read is 2-way. The pitch in words must be odd.

> [!QUESTION] You pad a bfloat16 tile of 64 columns by one element and the kernel is still slower than the padded float32 one. What went wrong?
> One bfloat16 element is half a word, so the pitch is 32.5 words. Rows alternate between word-aligned and
> half-word-aligned starts, and odd columns end up with two different words in the same bank: a 2-way conflict. Pad
> by a whole word, two elements, for a pitch of 33 words.

> [!QUESTION] Prove that $c \oplus r$ for $r = 0, \dots, 31$ and a fixed $c < 32$ gives 32 different columns.
> Suppose $c \oplus r = c \oplus r'$. XOR both sides with $c$ and the equality becomes $r = r'$. So different rows
> give different columns, and since all 32 results are in $0..31$ they are exactly the 32 banks.

> [!QUESTION] What is the best time for the float32 transpose of an 8192 x 8192 matrix on the lab A100, and what does a kernel at 800 GB/s tell you?
> The transpose moves $2 \times 8192^2 \times 4 = 536.9$ MB. At the 1275 GB/s copy ceiling that is 421 $\mu$s; a
> kernel at 800 GB/s takes 671 $\mu$s, 63% of the ceiling. It passes the lab's 50% bar but has about 37% left, which
> usually points at a remaining conflict, an unhidden barrier stall, or the ragged-shape penalty.

## Lab

> [!TRY] Model the banks, then write four transposes
> Open `08-transpose`. In Part 1, implement the bank model in Python: `bank`, `conflict_degree`, `smem_index` for the
> plain, padded, and swizzled layouts, `tile_bytes`, and `column_read_degree`. These run without a GPU, and the tests
> check that a column read is 32-way for the plain tile and 1-way for the padded and swizzled tiles, for both 4-byte
> and 2-byte elements.
>
> In Part 2, write the CUDA: the device `smem_index`, `transpose_naive`, and `transpose_tiled` with its two phases.
> The launcher, the dispatch on float32 and bfloat16, and the PyTorch binding are given. The tests check every variant
> is bit-exact on 13 shapes, including a single element, single rows and columns, and sizes one off a multiple of the
> tile. Then they time all four variants at $8192 \times 8192$ against a plain copy and report GB/s, and they require the
> padded and swizzled kernels to reach 50% (float32) and 40% (bfloat16) of the 1275 GB/s copy ceiling.
>
> The first compile takes 30 to 60 seconds. Compare your `fp32_tiled_gbs` to `fp32_padded_gbs`, and your
> `bf16_tiled_gbs` to the 46% prediction.

## Further reading

- Mark Harris, [An Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/),
  NVIDIA, 2013. The naive, coalesced, and padded kernels with their bandwidths on a Tesla M2050 and K20c.
- Mark Harris, [Using Shared Memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/),
  NVIDIA, 2013. Static and dynamic allocation, `__syncthreads`, and the bank rules.
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html), NVIDIA. The
  shared-memory and bank-conflict section.
- Lei Mao, [CUDA Shared Memory Bank](https://leimao.github.io/blog/CUDA-Shared-Memory-Bank/), 2022. Worked conflict
  examples with measurements.
- Colfax Research,
  [CUTLASS Tutorial: Fast Matrix-Transpose in CUDA with Cutlass](https://research.colfax-intl.com/tutorial-matrix-transpose-in-cutlass/),
  2024. The swizzled tile against padding, and the TMA store, on an H100 PCIe.
- Lei Mao, [CuTe Swizzle](https://leimao.github.io/blog/CuTe-Swizzle/), 2024. How `Swizzle<B, M, S>` permutes
  offsets, with worked bit examples.
- Weile Luo and others, [Benchmarking and Dissecting the Nvidia Hopper GPU Architecture](https://arxiv.org/abs/2402.13499),
  2024. The A100 PCIe shared-memory latency of 29 cycles used above.
- The companion course, [Your first CUDA kernel](https://qwen.ifkash.dev/c/12-your-first-cuda-kernel), for the
  coalescing measurement this chapter builds on.
