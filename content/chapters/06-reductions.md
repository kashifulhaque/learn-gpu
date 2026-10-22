---
title: Reductions
slug: 06-reductions
part: "Part 2 — CUDA patterns"
summary: How to turn millions of values into one on a GPU, from Harris's shared-memory tree to warp shuffles, atomics, and one-warp-per-row kernels, and why the best version is limited only by memory bandwidth.
minutes: 150
gpu: true
objectives:
  - "Explain why a reduction does almost no arithmetic per byte, and compute its time floor from the measured copy ceiling."
  - "Write a shared-memory tree reduction with sequential addressing and say why interleaved addressing diverges and bank-conflicts."
  - "Follow Harris's seven steps and say which ones still matter on an A100 and which the hardware made obsolete."
  - "Write a warp-level reduction with __shfl_down_sync and build a block reduction from it with one barrier."
  - "Combine per-block results three ways, with two passes, with atomicAdd, and with the last-block pattern, and choose between them."
  - "Choose one thread, one warp, or one block per row for a row-wise reduction, and explain how softmax and normalization use it."
  - "Explain why atomic floating-point sums can change from run to run while a max cannot."
lab: 06-reduce
---

# Reductions

> [!TLDR]
> - A reduction turns $n$ values into one with an associative operation such as
>   sum or max. It does about one FLOP per four bytes read, so on an A100 it is
>   limited by memory bandwidth: 256 MiB of float32 has a floor of 0.2105 ms at
>   the measured 1275 GB/s.
> - Every fast reduction has three levels: each thread accumulates many elements
>   in registers, each warp combines its 32 threads with shuffles, and the
>   blocks combine their results with a second pass or an atomic.
> - Harris's classic progression went from 2.083 GB/s to 62.671 GB/s on a G80,
>   a factor of 30. Its lessons transfer: avoid divergence and bank conflicts,
>   give each thread more work, and stop synchronizing where you don't need to.
> - Modern CUDA replaces the unrolled shared-memory tree with `__shfl_down_sync`,
>   and replaces the second kernel with one `atomicAdd` per block.
> - Floating-point addition isn't associative, so an atomic sum can differ in the
>   last bits from run to run. A max can't, because max is exact.

Almost every normalization and loss in a model ends in a reduction. LayerNorm
and RMSNorm reduce each row to a mean and a variance. Softmax reduces each row
twice, to a max and then to a sum. A loss reduces a batch to one number, a
gradient clip reduces every gradient to a norm, and sampling reduces a vocabulary
to an argmax. When these kernels are slow, a training step or a decoded token
waits on them.

A reduction is also the cleanest memory-bound kernel there is. It reads every
byte once, does one add per element, and writes almost nothing. That makes it a
direct test of whether you can drive the memory system at full speed, which is
the subject of [chapter 3](/c/03-memory-and-bandwidth). The difficulty is that the
threads that read the data must agree on one answer, and a GPU gives you very
little help with agreement.

This chapter follows the path Mark Harris laid out in 2007 in *Optimizing
Parallel Reduction in CUDA*, then modernizes it with warp shuffles, atomics, and
one-warp-per-row kernels. The lab has you write four of them and measure how close
each gets to the card's ceiling.

## Before you start

**Warp.** The 32 threads that execute one instruction together. A warp is the
unit that issues loads and the unit that diverges, as [chapter 2](/c/02-the-execution-model) explains.

**Shared memory and `__syncthreads()`.** Shared memory is a small, fast scratchpad
that the threads of one block share. `__syncthreads()` is a barrier: no thread of
the block continues until every thread has reached it, which is how a thread
safely reads what another thread wrote.

**Banks.** Shared memory has 32 banks, each 4 bytes wide, and the word at index
$i$ lives in bank $i \bmod 32$. A warp's access is one transaction when its
threads touch different banks, or the same word, and $N$ transactions when $N$
different words fall in one bank. [Chapter 8](/c/08-shared-memory-and-transpose) goes deeper.

**Coalescing.** A warp that reads 32 consecutive floats touches 128 contiguous
bytes, which memory serves efficiently. Every kernel in this chapter loads that way.

**Atomics.** `atomicAdd(p, v)` reads `*p`, adds `v`, and writes it back as one
indivisible step, so concurrent threads don't lose updates. Contended atomics
queue up at the L2 cache.

## What does a reduction cost at best?

The question for this section: how fast could the best possible reduction run? Count the bytes, then divide by a bandwidth.

A sum of $n$ float32 values reads $4n$ bytes and writes four. The
arithmetic is $n - 1$ adds, about one FLOP per four bytes, an intensity of 0.25
FLOP/byte. The A100's ridge point is 161 FLOP/byte (chapter 0), so the
reduction sits more than 600 times to the memory side of it. Only bandwidth
matters.

Divide the bytes by the bandwidth to get the time floor:

$$
\boxed{t_{\min} = \frac{\hla{B}}{\hlb{\beta}} = \frac{4 \times 2^{26}\ \text{byte}}{1.275 \times 10^{12}\ \text{byte/s}} = 0.2105\ \text{ms}}
$$

Here $\hla{B}$ is the bytes read and $\hlb{\beta}$ is the 1275 GB/s that a plain
copy measured on this card ([learn-inference chapter 10](https://qwen.ifkash.dev/c/10-roofline)).
The lab's input is $2^{26}$ floats, which is 268,435,456 bytes or 256 MiB, and
that is six times the 40 MiB L2, so every byte comes from HBM. Against the rated 1935 GB/s the floor
is 0.1387 ms, but the rating isn't a target.

> [!NOTE] A read-only kernel can beat the copy figure
> The 1275 GB/s ceiling counts a read and a write for every byte copied, and the
> hand-written vector add in learn-inference reached 1304 GB/s. A kernel that
> only reads can land near, or somewhat above, that number. Treat 1275 as a
> yardstick: 50% of it means something is wrong, and 95% means you're done.

This floor is the thing to measure against. A reduction that takes ten times the
floor isn't doing ten times the work. It's wasting the memory system's time on
something else: idle threads, serialized shared-memory accesses, or too few bytes
in flight.

### Why you can reorder the work at all

A GPU can only split a reduction across thousands of threads if the combining
operation lets you regroup and reorder it. Sum, product, max, min, and, and or
are all *associative* ($(a \circ b) \circ c = a \circ (b \circ c)$) and
*commutative* ($a \circ b = b \circ a$). Each also has an *identity element*, a value that changes nothing: 0 for a sum, 1 for a
product, $-\infty$ for a max. You pad an incomplete tail with the identity.

Associativity turns a chain of $n - 1$ dependent adds into a tree of depth
$\lceil \log_2 n \rceil$. Eight values sum in three rounds: four pairs, then two
pairs, then one. The tree does the same $n - 1$ adds, but each round's adds are
independent, so all of them can run at once.

Floating-point addition is commutative but not associative, because every add
rounds. A tree and a serial loop give different answers in the last bits. That
is correct behavior, not a bug, and the end of this chapter returns to it.

## Three levels: thread, block, grid

A GPU can't reduce $2^{26}$ values in one step, because threads in different blocks can't wait for each other
inside a kernel. The section answers one question: how do the pieces of the reduction map onto the hardware?

Every fast reduction uses the same three levels:

1. **Thread.** Each thread loops over many elements in a grid-stride loop and
   keeps a running sum in a register. No communication, no synchronization.
2. **Block.** The 256 threads of a block combine their 256 partial sums, through
   shared memory or warp shuffles, into one value.
3. **Grid.** The per-block values, one per block, combine into the final answer:
   with a second kernel, with atomics, or with the last block to finish.

The lab launches 864 blocks of 256 threads. That is $108 \times 8$: 108 SMs
times eight blocks of 256 threads, which fills the 2048 resident-thread limit of
each SM. The grid is 221,184 threads, so each one accumulates about
$2^{26} / 221184 = 303$ elements before the tree begins. The tree itself is then a small fixed
cost per block: it reduces 256 values no matter how large the input is.

That shape is the end of a journey. Harris's 2007 kernels started from one
element per thread and a poorly built tree, and the next four sections retrace
his steps, because each step names a hazard you'll meet again in other kernels.

> [!KEY] Reduce in registers first, then in shared memory, then across blocks
> The cheapest reduction step is the one a thread does alone in a register. The
> most expensive is the one that crosses blocks. Do as much as possible at the
> lowest level.

## Interleaved addressing: the divergent tree

The first tree most people write pairs up neighbors, then pairs of pairs. This section answers: what is wrong with the obvious shared-memory tree?

Harris's kernel 1 loads one element per thread into shared memory, then repeats
for strides $s = 1, 2, 4, \dots$: thread $t$ adds `s[t + s]` into `s[t]` if $t$ is
a multiple of $2s$.

```cuda
__global__ void reduce1(const float* in, float* out, int n) {
    extern __shared__ float s_data[];
    unsigned t = threadIdx.x;
    unsigned i = blockIdx.x * blockDim.x + t;
    s_data[t] = (i < n) ? in[i] : 0.0f;       // identity pads the tail
    __syncthreads();

    for (unsigned s = 1; s < blockDim.x; s *= 2) {
        if (t % (2 * s) == 0)                  // only every (2s)-th thread works
            s_data[t] += s_data[t + s];
        __syncthreads();
    }
    if (t == 0) out[blockIdx.x] = s_data[0];   // one partial sum per block
}
```

It is correct, and it is slow for a reason that has nothing to do with the
memory. In round one half of the threads are active, and they're every other
thread, so every warp contains both active and idle threads. A warp executes
whatever its active threads need, so it runs the add for the whole warp while
half the lanes sit masked off.

Count the cost in warp issues, the number of warp-instruction slots the block
consumes. A 256-thread block has 8 warps and 8 rounds. For $s = 1, 2, 4, 8, 16$
the active threads are spread across all 8 warps, which is $5 \times 8 = 40$ issues. For
$s = 32, 64, 128$ the active threads are 4, 2, then 1 in number, spaced a multiple of
64 apart, which leaves 4, 2, and 1 warps with work. The total is
$40 + 4 + 2 + 1 = 47$ warp issues. ==A tree that only needs 12 is paying 47==, a factor
of 3.9.

Use the figure to see which threads work in each round. Set the addressing to
the first option and move the round slider, then compare the two other choices
in the next section.

```viz
06-tree-rounds
```

On Harris's G80, this kernel reached 8.054 ms for 4M elements, 2.083 GB/s. The
full table is below, but the lesson here is general: ==divergence
inside a warp costs whole-warp issue slots==, so an `if` whose condition depends
on `threadIdx.x % k` for small $k$ is almost always a mistake.

## Bank conflicts and sequential addressing

The question for this section: if you remove the divergence, what is the next problem, and what fixes both?

Harris's kernel 2 keeps the interleaved pairing but compacts the threads: thread $t$
works on index $2st$, so the first $n/2s$ threads are the active ones and the
`%` test disappears. The warps now retire in order, so the issue count drops from
47 to 12, the same as the best case. But it trades one problem for another.

Thread $t$ now reads word $2st$ of shared memory. For $s = 4$ and a 256-thread
block, the 32 threads of warp 0 read words $0, 8, 16, \dots, 248$. Word $i$ lives
in bank $i \bmod 32$, so those words fall in banks 0, 8, 16, and 24, and eight
different words share each of the four banks. The hardware serves them in eight
transactions instead of one, an *8-way bank conflict*. Over the eight rounds, 8-way is the
worst case, and the figure computes each round's conflict degree when you pick the strided option.

Kernel 3 fixes the access pattern by changing the pairing. In each round, thread
$t < s$ adds `s_data[t + s]` into `s_data[t]`, and $s$ *starts at half the block and
halves* each round:

```cuda
for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (t < s) s_data[t] += s_data[t + s];
    __syncthreads();            // outside the if: every thread must reach it
}
```

This is *sequential addressing*. In every round, consecutive threads read
consecutive words, which are 32 different banks, so there are no conflicts. The
active threads are always the lowest-numbered ones, so whole warps go idle
together and the warp count is 12 again. Both problems are gone at once.

> [!WARNING] Keep the barrier outside the `if`
> `__syncthreads()` must be reached by every thread of the block. Put it inside
> `if (t < s)` and the threads that skip it never arrive, so the block waits
> forever or reads stale data. The loop above does it right: the `if` guards the
> add, and the barrier sits after it.

The cost that sequential addressing leaves behind is that half the threads idle in the first
round, a quarter more in the second, and so on. Harris's next steps attack that waste.

## From kernel 4 to kernel 7: more work per thread

This section answers: how does Harris get from 9.7 GB/s to 62.7 GB/s once the tree is clean? By making each thread do more before the tree starts, and by dropping barriers the last warp doesn't need.

Here are all seven kernels, as Harris measured them on a G80 with $2^{22}$ integers:

| Kernel | Change | Time (ms) | GB/s | Step speedup | % of 86.4 GB/s peak |
|---|---|---|---|---|---|
| 1 | Interleaved addressing, divergent | 8.054 | 2.083 | n/a | 2.4% |
| 2 | Interleaved, strided index (bank conflicts) | 3.456 | 4.854 | 2.33x | 5.6% |
| 3 | Sequential addressing | 1.722 | 9.741 | 2.01x | 11.3% |
| 4 | First add during the global load | 0.965 | 17.377 | 1.78x | 20.1% |
| 5 | Unroll the last warp | 0.536 | 31.289 | 1.80x | 36.2% |
| 6 | Completely unrolled | 0.381 | 43.996 | 1.41x | 50.9% |
| 7 | Multiple elements per thread | 0.268 | 62.671 | 1.42x | 72.5% |

The G80 is a 2006 GPU with a 384-bit bus at 1800 MHz, a peak of
$384 \times 1800 / 8 = 86.4$ GB/s, and it is the card of Harris's slides. ==Only
the ratios and the lessons transfer to the A100==. An A100's rated bandwidth is 22 times
larger, its threads have their own program counters where the G80's warp had one, and the exact
speedup of each step differs. The last column divides his bandwidth by the G80's peak.

### Kernel 4: add during the load

In kernel 3, half the threads of a block do nothing in round one. Kernel 4 halves
the number of blocks and has each thread load two elements, add them as it loads,
and then store the sum. The first tree round is now free: the block has
already reduced $2 \times 256 = 512$ elements by the time the tree starts.

### Kernels 5 and 6: unroll the last warp

When the stride drops to 32 or less, only one warp is active, and a warp's threads
execute together. Harris drops the `__syncthreads()` and the `if (t < s)` for those
last six rounds, and writes them out by hand. Kernel 6 unrolls the whole loop
with C++ templates on the block size so the compiler removes every dead branch.

The warp-synchronous trick has a catch on modern hardware. Since Volta, each
thread has its own program counter, and the threads of a warp aren't guaranteed
to run in lockstep ([chapter 2](/c/02-the-execution-model)). Code that assumes
lockstep and uses `volatile` shared memory must now call `__syncwarp()` between
rounds. A cleaner fix, which the next section builds, is to not use shared memory
for the last warp at all.

### Kernel 7: many elements per thread

Kernel 7 is the step that matters most for the A100. Each thread loops over
the input with a stride of the whole grid, adding as it goes, so a block's tree
amortizes over about 77,700 elements ($2^{26} / 864$) instead of 512. Harris explains the choice
with a cost argument.

> [!DEEPDIVE] Why Harris gives each thread about $\log n$ elements
> A tree over $n$ elements does $O(n)$ work in $O(\log n)$ steps. If you
> launch $n$ threads, you use $n$ processors for $\log n$ steps, which is
> $O(n \log n)$ processor-time, more than the $O(n)$ a serial loop costs. The
> work-efficient version uses $p = n / \log n$ threads. Each thread first sums
> $\log n$ elements serially, which takes $O(\log n)$ time, then the $p$ partial
> sums go through a tree of depth $\log p \le \log n$. The total time is still
> $O(\log n)$, but the cost is $p \times O(\log n) = O(n)$. Harris calls this
> *algorithm cascading*, and it's Brent's theorem applied to a reduction. On a GPU the
> same shape appears for a second reason: a fixed grid that fills the machine
> and loops is how you keep enough loads in flight, as the next part shows.

### Why a grid-stride loop saturates memory

A second argument, specific to a GPU, comes from [chapter 4](/c/04-roofline-and-latency).
Little's law says to saturate the A100's HBM you need about 640 KB of loads in
flight across the card, which is 5.9 KB per SM (derived in that chapter from a 466-cycle
memory latency measured on a 40 GB A100). Each SM runs at most 2048 threads. If every thread has
one 4-byte load outstanding, that is 8 KB per SM, a margin of only 1.4. If every
thread issues a `float4`, a 16-byte load, the same threads have 32 KB in flight, 5.5 times the need.

That is why the lab's loading loop reads a `float4` per iteration when the
pointer is 16-byte aligned. It is the same vectorization that [chapter 3](/c/03-memory-and-bandwidth)
used to reach the copy ceiling, and it is why a reduction kernel's loads matter more than its tree.

```cuda
__device__ float load_sum(const float* x, int n) {
    float acc = 0.0f;
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int n4 = (((uintptr_t)x & 15) == 0) ? (n >> 2) : 0;   // float4 only if aligned
    const float4* x4 = reinterpret_cast<const float4*>(x);
    for (int i = tid; i < n4; i += stride) {
        float4 v = x4[i];
        acc += (v.x + v.y) + (v.z + v.w);
    }
    for (int i = n4 * 4 + tid; i < n; i += stride) acc += x[i];   // the tail
    return acc;
}
```

Two details matter. The scalar loop after the vector loop handles the one to
three leftover elements, and the whole input when the pointer is misaligned, so
the kernel is right for any $n$. And the accumulation is in float32 even when the
input is bfloat16 or float16, a rule that [chapter 9](/c/09-floating-point-on-the-gpu) explains.

## Warp shuffles: the tree without shared memory

The modern replacement for kernels 5 and 6 is a warp-level instruction. This section answers: how do the threads of a warp reduce without touching shared memory?

`__shfl_down_sync(mask, v, offset)` lets each lane read the register `v` of the
lane `offset` places above it, in one instruction, with no shared memory and no
barrier. To sum a warp, halve the offset from 16 to 1:

```cuda
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;                      // lane 0 holds the sum of all 32 lanes
}
```

Five rounds for 32 lanes. Lane $i$ adds the value from lane $i + 16$, then $i + 8$,
and so on, so lane 0 ends with the total. When the source lane is past the end of
the warp, the shuffle hands a lane its own value back. Those lanes end with
meaningless sums, and nothing reads them.

```viz
06-warp-shuffle
```

The first argument, `mask`, names the lanes that take part, and `0xffffffff` means
all 32. ==Every lane named in the mask must execute the shuffle with the same mask==,
or the result is undefined. That rule is why the lab's row kernel makes a
whole warp return together when it has no row, instead of letting some lanes exit.

### From a warp to a block

A block of 256 threads has eight warps. To reduce the block, each warp reduces
itself, lane 0 of each warp writes its result to an eight-word shared array, one
barrier makes the writes visible, and the first warp reduces those eight values:

```cuda
__device__ float block_sum(float v) {       // all 256 threads call this once
    __shared__ float warp_part[8];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) warp_part[warp] = v;
    __syncthreads();
    v = (lane < 8) ? warp_part[lane] : 0.0f;   // pad with the identity
    if (warp == 0) v = warp_sum(v);
    return v;                                   // thread 0 holds the block sum
}
```

Compare the cost with the shared-memory tree: eight rounds, each with a shared
read, an add, a shared write, and a barrier, against one 32-byte shared array, one barrier,
and two short chains of shuffles. The shuffle version also uses about 1 KB less
shared memory per block than a 256-float tree, which matters when other kernels fight for the
164 KB per SM.

### Hardware reduce and cooperative groups

Newer APIs hide the shuffle loop. On compute capability 8.0 and later,
`__reduce_add_sync(mask, x)` is a single warp-wide instruction for 32-bit
integers, with `__reduce_min_sync` and `__reduce_max_sync` beside it. It doesn't
take floats, so a float sum still uses shuffles. Cooperative groups wraps both:

```cuda
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

auto warp = cg::tiled_partition<32>(cg::this_thread_block());
float total = cg::reduce(warp, v, cg::plus<float>());   // every thread of the tile gets the total
```

`cg::reduce` picks the best implementation for the type and the architecture: the
hardware instruction for integers on sm_80, shuffles otherwise. The lab writes the shuffle loop by
hand so you see what the library does, and a production kernel often uses CUB's
`BlockReduce` or Triton instead.

## Combining the blocks

Each block now holds one number, and 864 numbers must become one. The question here: what is the cheapest way to cross blocks?

There are three options, with different costs and different guarantees.

| Method | Launches | Needs a zeroed output | Same bits every run | Cost beyond the main kernel |
|---|---|---|---|---|
| Two passes | 2 | No | Yes, for a fixed grid | A second launch over 864 values |
| One pass with `atomicAdd` | 1 | Yes | No | One atomic per block |
| Last block does the final sum | 1 | A counter | Yes, for a fixed grid | A fence and an atomic counter |

**Two passes.** The first kernel writes `partial[blockIdx.x]`. A second kernel with
a single block reduces the 864 partials with the same code. The lab's `sum_tree`
does this, and the second pass reads only $864 \times 4 = 3456$ bytes. The price is a second kernel launch,
which costs about 5 µs to queue on this card (learn-inference chapter 12) against a 210 µs main
kernel, though consecutive launches in a stream queue back to back.

**One pass with atomics.** Each block's thread 0 calls `atomicAdd(out, block_total)`.
There are 864 atomics on one address, one per block, and they finish quickly. What
to avoid is the per-element version, where every thread atomically adds into one
address: that is $2^{26} = 67{,}108{,}864$ updates to one location, which all
queue at one place in the L2. The rule is *one atomic per block*, never per thread.

**The last block.** A third pattern gives atomics' single launch and a fixed order. Each
block writes its partial result, issues a memory fence, and increments a counter. The
block that sees the counter reach `gridDim.x - 1` knows every other partial is written, so it sums them:

```cuda
__device__ unsigned int blocks_done = 0;

__global__ void sum_last_block(const float* x, float* partial, float* out, int n) {
    __shared__ bool is_last;
    float total = block_sum(load_sum(x, n));
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = total;
        __threadfence();                                  // publish partial[] first
        is_last = (atomicAdd(&blocks_done, 1u) == gridDim.x - 1);
    }
    __syncthreads();                                      // also separates the two block_sum calls
    if (is_last) {
        float s = 0.0f;
        for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x)
            s += __ldcg(&partial[i]);                     // bypass L1: other blocks wrote these
        s = block_sum(s);
        if (threadIdx.x == 0) { *out = s; blocks_done = 0; }   // reset for the next launch
    }
}
```

This is the pattern in NVIDIA's `threadFenceReduction` sample. The fence before the
counter is the part that's easy to forget, and without it the last block can read a
stale partial sum. Two-pass is simpler and as fast in practice for a single large
reduction, so reach for the last-block pattern when a launch is too costly,
such as inside a CUDA graph with many tiny reductions.

### The max reduction, and an atomic for it

Max follows the same three levels with `fmaxf` and an identity of $-\infty$. The
identity matters: a max initialized to 0 returns 0 for an array of negative numbers,
and the lab's tests include one.

CUDA has no `atomicMax` for floats, but a bit trick makes one. For a non-negative
float, the bit pattern read as a signed integer orders the same way as the float.
For a negative float, the bit pattern read as an unsigned integer orders in reverse:

```cuda
__device__ void atomic_max_float(float* addr, float v) {
    int bits = __float_as_int(v);
    if (bits >= 0) atomicMax((int*)addr, bits);                  // v >= 0
    else           atomicMin((unsigned int*)addr, (unsigned)bits);   // v < 0
}
```

The target must start at $-\infty$, which the host sets before the launch. Mixed
signs work. If the target is positive, a negative value's bits are at least
0x80000000 as an unsigned integer, above any positive pattern, so `atomicMin`
leaves the target alone. If the target is negative, a positive value's bits are
a larger `int`, so `atomicMax` replaces it. The lab's tests check negatives,
positives, and a mix.

### Why the atomic sum isn't repeatable and the max is

A float32 add rounds to 24 bits of significand. Add 1 to $2^{24} = 16{,}777{,}216$
and the exact answer, 16,777,217, isn't representable, because above $2^{24}$ the
spacing between floats is 2. The sum rounds back to 16,777,216. So

$$
(2^{24} + 1) + 1 = 2^{24} \qquad\text{but}\qquad 2^{24} + (1 + 1) = 16{,}777{,}218.
$$

The two groupings differ by 2. In an atomic sum the grouping is decided by which
block's `atomicAdd` reaches the L2 first, and that changes from run to run. Two
launches on the same data can return values that differ in the last bits. The
two-pass version has a fixed order for a given grid, so it repeats exactly, but it changes if the grid size
changes, for example on a card with a different SM count.

A max has no such problem. The maximum of a set is one of its elements, so every
order gives the same bit pattern, and `max_atomic` repeats exactly. [Chapter 9](/c/09-floating-point-on-the-gpu)
measures this, adds compensated summation, and shows how PyTorch's deterministic mode picks the fixed-order path.

> [!DEEPDIVE] How big is the error of a reordered sum?
> Higham (1993) shows that adding $m$ float32 numbers one after another has an
> error of at most $(m - 1)\,u \sum |x_i|$, where $u = 2^{-24}$ is the unit
> roundoff. A pairwise tree of depth $\log_2 m$ has a bound of about $\log_2 m \cdot u \sum |x_i|$.
> The lab's kernels do a mix: each thread adds about 303 terms serially, then a
> tree of depth 8, then a second stage. The test's tolerance is $2(\lceil n/221184 \rceil + 40)\,u \sum|x_i|$,
> which covers every order the hardware can pick, so it doesn't fail on atomic
> ordering. The actual error is usually far below the bound, because rounding
> errors of both signs partly cancel.

## Row-wise reductions

Softmax, LayerNorm, and RMSNorm reduce each row of a matrix independently. The question here: how do you map many small reductions onto threads?

Take a matrix of `rows` by `cols`, and a result of one value per row. The three choices
differ in how many threads cooperate on one row:

| Columns per row | Mapping | Why |
|---|---|---|
| A few dozen or fewer | A fraction of a warp per row (the `width` argument of `__shfl_down_sync`) | Several rows share one warp's 32 lanes |
| Hundreds to a few thousand | One warp per row | No shared memory, no barrier, one shuffle chain |
| Many thousands | One block per row, with `block_sum` | One warp can't keep enough loads in flight |
| Few rows, very long rows | Several blocks per row, then combine | Otherwise most of the GPU is idle |

These boundaries are rules of thumb, not constants. Tune them on your shapes.

The lab's `row_sum` uses one warp per row, with eight rows per 256-thread block:

```cuda
__global__ void row_sum_kernel(const float* x, float* out, int rows, int cols) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row  = blockIdx.x * 8 + warp;
    if (row >= rows) return;                  // the whole warp leaves together
    const float* p = x + (long long)row * cols;
    float acc = 0.0f;
    for (int i = lane; i < cols; i += 32) acc += p[i];   // a coalesced 128-byte row slice
    acc = warp_sum(acc);
    if (lane == 0) out[row] = acc;
}
```

The `return` is safe because every lane of a warp has the same `row`, so all 32
return together and no shuffle ever sees half a warp. The `(long long)` cast
guards against 32-bit overflow in `row * cols` for a large matrix.

The case that breaks this mapping is a small batch. The A100 holds
$108 \times 64 = 6912$ resident warps, so a one-warp-per-row kernel needs 6,912 rows to fill every warp slot. A softmax over 32 rows of a
long vocabulary, as in decoding, launches 32 warps and leaves the rest of the
card idle. For that shape, split each row across blocks, which is the multi-block
design from the previous section applied once per row.

The reduction in softmax and normalization isn't the whole kernel. Softmax needs
the max, then the sum of exponentials, and normalization needs the mean and then
the variance. [Chapter 15](/c/15-softmax-and-normalization) shows how to compute
those in one pass over the row, with an online update, so the row is read once instead of three times.

## Reductions in Triton

Triton expresses a reduction in one line, and the compiler writes the shuffles and shared memory for you.
The question: what does the Triton version look like, and what does it hide?

```python
@triton.jit
def row_sum_kernel(x_ptr, out_ptr, cols, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        v = tl.load(x_ptr + row * cols + offs, mask=offs < cols, other=0.0)
        acc += v                       # per-lane accumulation, like the thread level
    tl.store(out_ptr + row, tl.sum(acc, axis=0))   # the warp and block levels
```

The loop is the thread level, `tl.sum` is the warp and block levels, and one
program per row is one block per row. The `other=0.0` is the identity of the sum,
and for a max you'd pass `-float("inf")`. For a 1-D sum across a whole grid, each program ends with
`tl.atomic_add(out_ptr, tl.sum(v, axis=0))`, with the same one-atomic-per-block rule and the
same nondeterminism. [Chapter 14](/c/14-triton) introduces the language.

## Measuring a reduction honestly

The section answers: how do you time a reduction so the number means something? Four rules apply.

- **Count the bytes read.** A sum reads $4n$ bytes. Counting the 4-byte output
  changes nothing, and counting a write that didn't happen inflates the result.
- **Flush the L2 when the data fits in it.** An input below 40 MiB, which is
  10,485,760 floats, is served from cache after the first call, so the
  bandwidth looks impossibly high. The lab's tests flush before every timed call.
- **Beware of small sizes.** At $n = 2^{16}$ floats the data is 256 KiB, and at 1275 GB/s that moves in 0.2 µs. A 5 µs
  launch is 24 times that, so a small reduction measures the launch, not the kernel.
- **Compare against a library.** The lab times `torch.sum` on the same tensor and reports
  it as `torch_sum_gbs`, which tells you what a well-tuned implementation achieves on the card you were given.

> [!TIP] Debug the tree with a small input first
> Set `n` to 257 and print the result against `x.double().sum()`. A missing
> `__syncthreads()` or a wrong identity shows up at small sizes, long before a
> bandwidth number does.

## What goes wrong

- **A barrier inside a branch.** `__syncthreads()` under `if (t < s)` deadlocks
  or races. Guard the work, not the barrier.
- **A missing barrier between rounds.** Without it, a thread reads a word
  another thread hasn't finished writing, and the sum is wrong only sometimes.
- **The wrong identity.** A max that starts at 0, or a block that pads its tail with 0, returns the
  wrong answer on all-negative data. Pad with the identity of the operation.
- **Assuming the size is a multiple of the block.** The tail guard and the
  scalar cleanup loop exist for $n = 1, 31, 257$ and for a tensor that starts 4 bytes past an aligned address.
- **A partial mask.** A `__shfl_down_sync(0xffffffff, ...)` in a branch that only some lanes take
  is undefined. Make the whole warp take the branch, or build the mask with `__ballot_sync`.
- **Forgetting to initialize the output.** `atomicAdd` into uninitialized memory adds to garbage, and
  an atomic max into zeros can never go below zero. The host wrapper in the lab sets both.
- **Atomics in a hot loop.** One atomic per element serializes at the L2. One per block is fine.
- **32-bit index overflow.** `row * cols` and `blockIdx.x * blockDim.x` overflow `int` past 2 billion. Use
  `long long` for offsets of large tensors.
- **Comparing a sum with a tolerance of 1e-6.** A float32 sum of a million terms differs from the
  float64 sum by more than that. Scale the tolerance by $\sum |x_i|$, as the lab does.

> [!RECAP]
> - A reduction does about 0.25 FLOP per byte, so its floor is the bytes read
>   divided by the bandwidth: 0.2105 ms for 256 MiB at 1275 GB/s.
> - The fast structure is thread, then block, then grid: accumulate in
>   registers, combine with warp shuffles, then combine blocks.
> - Harris's lessons hold: avoid divergence (47 versus 12 warp issues for a
>   256-thread block), avoid bank conflicts, and give each thread many elements.
> - `__shfl_down_sync` sums a warp in five rounds with no shared memory, and a
>   block sum needs one barrier.
> - Combine blocks with a second pass, one atomic per block, or the last block.
> - Atomic float sums vary in the last bits, because rounding makes addition order-dependent. Max doesn't vary.
> - Row-wise reductions use one warp or one block per row, and need enough rows to fill the card.

## Check your understanding

> [!QUESTION] Sum $2^{26}$ floats on the A100. What is the fastest time you can expect, and what does it assume?
> The kernel reads $4 \times 2^{26} = 268{,}435{,}456$ bytes. At the measured 1275 GB/s copy ceiling that
> is $268{,}435{,}456 / 1.275 \times 10^{12} = 0.2105$ ms. It assumes the kernel keeps enough loads in flight to
> reach the copy ceiling, and that the arithmetic hides under the memory time, which it does at 0.25 FLOP/byte.

> [!QUESTION] In Harris's kernel 1, a 256-thread block issues 47 warp-instructions of adds. What would the same tree cost with sequential addressing, and why?
> It costs 12. With sequential addressing the active threads in each round are the lowest-numbered ones: 128, 64, 32, 16, 8, 4, 2, 1
> threads. Those occupy 4, 2, 1, 1, 1, 1, 1, 1 warps, which sums to 12. Interleaved addressing keeps one active thread in every warp for the first five rounds, so all 8 warps issue.

> [!QUESTION] Why does `if (t % (2 * s) == 0)` hurt, but `if (t < s)` doesn't, when both activate the same number of threads?
> Divergence is per warp. `t % (2 * s) == 0` spreads the active threads across every warp, so each warp
> executes the add with most lanes masked off. `t < s` activates a contiguous group, so entire warps are either fully active or fully idle, and the idle ones issue nothing.

> [!QUESTION] A 256-thread block reads shared memory at word $2st$ with $s = 4$. How many bank conflicts occur, and what is the fix?
> Warp 0's 32 threads read words $0, 8, 16, \dots, 248$, which land in banks 0, 8, 16, and 24 with eight distinct words each, an 8-way conflict. Switch to sequential addressing, where thread $t$ reads word $t$, and the 32 threads hit 32 different banks.

> [!QUESTION] Why does `max_atomic` return the same value every run, while `sum_atomic` may not?
> A maximum is one of the input elements, so any order of comparing them produces the same bits. A float sum rounds at every add, and the order in which blocks reach `atomicAdd` varies from run to run, so the rounding errors and the result vary.

> [!QUESTION] A softmax runs over 32 rows of 128,000 columns, one warp per row. Why is it slow, and what do you change?
> One warp per row launches 32 warps, out of the 6,912 the card can hold, so most SMs sit idle and the loads in flight are far
> below the 5.9 KB per SM that Little's law needs. Split each row across many blocks, and combine the per-block max and sum with a second pass or atomics.

> [!QUESTION] What breaks if `block_sum` pads lanes 8 to 31 of warp 0 with 0 and you reuse it for a max?
> Zero isn't the identity of max. On all-negative data the padded lanes contribute 0, which beats every real value, so the block returns 0.
> Pad with $-\infty$, as the lab's `block_max` does.

## Lab

> [!TRY] Four reductions on the A100
> Open `starter.py` and fill in the nine TODOs: `warp_sum`, `warp_max`, `block_sum`,
> `block_max`, `atomic_max_float`, and the four kernels `sum_tree_kernel`,
> `sum_atomic_kernel`, `max_atomic_kernel`, and `row_sum_kernel`. The loading loops and
> host wrappers are provided. The tests check every function against PyTorch on sizes 1, 31, 33, 257,
> 1,000,003, and a misaligned input, then time each on $2^{26}$ floats and report
> `sum_tree_gbs`, `sum_atomic_gbs`, `max_atomic_gbs`, `row_sum_gbs`, and `torch_sum_gbs`.
> You pass when everything is correct and each kernel reaches at least 30 to 40% of the 1275 GB/s ceiling. The roofline floor is
> 0.2105 ms, so a good kernel should land well above the threshold. Compare your numbers with `torch_sum_gbs`.

## Further reading

- [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf),
  Mark Harris, 2007: the seven-step progression, measured on a G80.
- [Faster Parallel Reductions on Kepler](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/),
  Justin Luitjens, 2014: warp shuffles in place of shared-memory trees.
- [Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/),
  Yuan Lin and Vinod Grover, 2018: the mask rule and `__syncwarp`.
- [Cooperative Groups: Flexible CUDA Thread Programming](https://developer.nvidia.com/blog/cooperative-groups/),
  Mark Harris and Kyrylo Perelygin, 2017: `cg::reduce` and tiled partitions.
- [cub::DeviceReduce](https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceReduce.html): the library version to compare your kernels against.
- [The Accuracy of Floating Point Summation](https://epubs.siam.org/doi/10.1137/0914050),
  Nicholas Higham, 1993: error bounds for recursive, pairwise, and compensated sums.
- [Floating Point and IEEE 754 Compliance for NVIDIA GPUs](https://docs.nvidia.com/cuda/floating-point/index.html):
  why a parallel reduction order changes the result.
- [Chapter 7](/c/07-scan-and-histogram) takes the same tree one step further, to prefix sums.
