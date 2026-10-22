---
title: Scan and histogram
slug: 07-scan-and-histogram
part: "Part 2 — CUDA patterns"
summary: How to compute every prefix sum of an array and a count per bin on a GPU, from a warp-shuffle scan to a device-wide scan, and from global atomics to privatized and aggregated histograms.
minutes: 150
gpu: true
objectives:
  - "Define inclusive and exclusive scan, and compute both by hand for a small array."
  - "Compare Kogge-Stone and Blelloch by work and depth, and say what each costs inside one thread block."
  - "Write a warp-level scan with shuffles and build a block-level scan from it."
  - "Build a device-wide scan from three kernels and explain how decoupled look-back reduces its memory traffic from 3n to 2n."
  - "Explain why a histogram with global atomics collides, and how shared-memory privatization removes most of the collisions."
  - "Choose between privatization, per-warp copies, and warp aggregation for a given bin count and data distribution."
  - "Use a histogram and a scan together to build a counting sort and the expert offsets of a mixture-of-experts layer."
lab: 07-scan-histogram
---

# Scan and histogram

> [!TLDR]
> - A scan (prefix sum) turns an array into its running totals. The exclusive
>   scan of `[3,1,7,0]` is `[0,3,4,11]`. A loop does it in n-1 additions but
>   can't use a GPU's thousands of threads, because each step needs the
>   previous one.
> - Kogge-Stone finishes in log2 n steps but does n log2 n - n + 1 additions.
>   Blelloch does 2(n - 1) additions in 2 log2 n steps. Inside one warp,
>   shuffles make Kogge-Stone cheap, so real kernels scan a warp with
>   shuffles and combine warps through shared memory.
> - Blocks can't wait for each other, so a device-wide scan is built from
>   several kernels. Reduce-then-scan moves 3n elements of memory traffic.
>   Decoupled look-back moves 2n, the same as a copy.
> - A histogram is a reduction with many outputs. Global atomics collide on
>   hot bins. Giving each block a private copy in shared memory cuts the global
>   atomics by a factor of about 300 in the lab's configuration.
> - A histogram followed by a scan is a counting sort, and it's also how a
>   mixture-of-experts layer finds where each expert's tokens start.

You've met the reduction, which turns n numbers into one. This chapter covers its
two close relatives. A *scan* keeps every running total instead of only the last.
A *histogram* counts how many inputs fall into each of several bins instead of
adding them all into one.

Both show up in model code more often than the names suggest. Top-p sampling
takes a cumulative sum of sorted probabilities. Dropping padding tokens from a
batch, or the tokens an expert doesn't own, is a stream compaction, which is a
scan of a 0-or-1 flag array. A mixture-of-experts layer counts tokens per expert
and scans the counts to get each expert's offset. Variable-length attention
carries its sequence boundaries as the exclusive scan of the sequence lengths.
Radix sort is a histogram, a scan, and a scatter, repeated per digit.

Neither pattern is hard to write badly. A naive scan is a loop on one thread, and
a naive histogram is one atomic add per element. Both are correct, and both leave
the GPU almost idle. The chapter shows why, and builds the versions that reach
the memory system's limit. The lab asks you to write them and measures each
against the 1275 GB/s copy ceiling.

## Before you start

**Reductions.** [Chapter 6](/c/06-reductions) built the block-wide sum from warp
shuffles and a shared-memory handoff between warps. A scan reuses that skeleton,
so read it first if `__shfl_down_sync` or `__syncthreads()` is unfamiliar.

**Shared memory and bank conflicts.** Shared memory has 32 banks, and word $w$
lives in bank $w \bmod 32$. Two lanes of a warp that touch different words in the
same bank take two passes. [Chapter 8](/c/08-shared-memory-and-transpose) covers
this in depth. This chapter uses it for the Blelloch scan and the histogram.

**Atomics.** `atomicAdd(&x, v)` reads `x`, adds `v`, and writes the result back as
one indivisible step, so concurrent updates never lose a count. Updates to the
same address take turns. Integer atomics give the same answer in any order, so
they're deterministic. Floating-point atomics aren't, because float addition
isn't associative ([chapter 9](/c/09-floating-point-on-the-gpu)).

**The copy ceiling.** A plain device-to-device copy measured 1275 GB/s on the lab
A100, against a rating of 1935 GB/s (learn-inference, chapter 10). A copy moves
$2N$ bytes per $N$ copied, and the chapter judges every kernel's memory traffic
against that figure.

## What a scan computes

A scan replaces each element of an array with the combined value of the elements
up to it. The section defines the two versions and the one property an operator
needs.

Take the array $x = [3, 1, 7, 0, 4, 1, 6, 3]$. The *inclusive scan* at position
$i$ includes $x_i$ itself. The *exclusive scan* stops just before it:

$$
\text{inclusive}_i = \hla{\sum_{j \le i} x_j}, \qquad \text{exclusive}_i = \hlb{\sum_{j < i} x_j}
$$

Here $\hla{\text{inclusive}_i}$ counts the element at $i$, and $\hlb{\text{exclusive}_i}$
doesn't. For the example array, the two results are:

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| Input $x$ | 3 | 1 | 7 | 0 | 4 | 1 | 6 | 3 |
| Inclusive scan | 3 | 4 | 11 | 11 | 15 | 16 | 22 | 25 |
| Exclusive scan | 0 | 3 | 4 | 11 | 11 | 15 | 16 | 22 |

==The exclusive scan is the inclusive scan shifted right by one==, with the identity
(0 for addition) in the first slot. Most GPU uses want the exclusive version,
because $\text{exclusive}_i$ is the number of items that come before item $i$:
the position where item $i$ should go. The lab computes the exclusive scan.

Nothing here requires addition. A scan works for any *associative* operator
with an identity: `max`, `min`, multiplication, or a custom operator. Associative
means $(a \oplus b) \oplus c = a \oplus (b \oplus c)$, and that property is what
lets a parallel algorithm regroup the work. Floating-point addition is only
approximately associative, so a parallel float scan gives a slightly different
answer from a serial loop. The lab checks float scans two ways: exactly on small
integers stored as floats, and within a tolerance on random data.

## Work and depth

A scan looks sequential, because each output depends on the one before it. The
section defines the two quantities that decide how well a parallel version can do.

The serial loop is `y[i] = y[i-1] + x[i]`. It does $n - 1$ additions, which is
the least possible work, but it takes $n - 1$ steps one after another. Two
numbers describe a parallel algorithm:

- **Work** is the total number of operations, summed over all threads.
- **Depth** (also called *span*) is the length of the longest chain of operations
  that must run one after another.

A machine with enough processors takes time proportional to the depth, and a
machine with $p$ processors takes at least the work divided by $p$. A good parallel
scan has low depth and work close to the serial loop's $n - 1$.

You can't have both extremes. Any scan network that finishes in $\log_2 n$ steps
must do more than $n - 1$ additions, because each step can combine only two
partial results into one. The two classic networks sit at opposite ends of that
trade-off, and the next sections build each.

## Where ML uses scan

The section answers why the kernel is worth writing. Four uses cover most of what
you'll meet.

**Top-p sampling.** Sort the next-token probabilities in descending order, take
the running sum, and keep the tokens whose running sum before them is below $p$.
The companion course derives the rule in
[its sampling chapter](https://qwen.ifkash.dev/c/11-sampling). With $p = 0.85$ and
sorted probabilities $[0.50, 0.20, 0.12, 0.08, 0.05, 0.03, 0.02]$, the exclusive
scan is $[0, 0.50, 0.70, 0.82, 0.90, 0.95, 0.98]$. The first four entries are
below 0.85, so the sampler keeps four tokens. The exclusive scan is the right
variant: a token is kept when the mass *before* it is under $p$, so the token
that crosses the threshold stays in.

**Stream compaction.** To drop elements that fail a test, mark survivors with 1
and the rest with 0, take the exclusive scan of the flags, and write each
survivor to the slot its scan value names. Here the survivors are the nonzero
entries:

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| Values | 5 | 0 | 3 | 0 | 0 | 8 | 2 | 0 |
| Flag | 1 | 0 | 1 | 0 | 0 | 1 | 1 | 0 |
| Exclusive scan of flags | 0 | 1 | 1 | 2 | 2 | 2 | 3 | 4 |

The survivors at indices 0, 2, 5, and 6 go to slots 0, 1, 2, and 3, and the last
scan value plus the last flag, $4 + 0 = 4$, is the count. Every thread computes
its own destination, so the writes are collision-free and in order.

**Sequence offsets.** A batch of variable-length sequences packs into one tensor,
and attention kernels find each sequence's start from the exclusive scan of the
lengths. Lengths $[5, 3, 8]$ give start offsets $[0, 5, 8]$ and a total of 16.

**Linear recurrences.** The scan isn't limited to addition. A recurrence
$h_t = a_t h_{t-1} + b_t$, which underlies state-space and gated linear
recurrent layers, is a scan over pairs $(a, b)$ with the associative operator
$(a_1, b_1) \circ (a_2, b_2) = (a_1 a_2,\ a_2 b_1 + b_2)$. Kernels for those
layers run the same network as the one below, with a different operator.

> [!NOTE] The operator is a parameter
> Every algorithm in this chapter uses `+` in the code, but none of them depends
> on it. Replace `+` and the identity `0` with an associative operator and its
> identity and the scan still works. Only the order of the operands matters, so
> the operator needn't be commutative.

## Kogge-Stone: shallow and wasteful

The Kogge-Stone network ([Kogge and Stone, 1973](https://doi.org/10.1109/TC.1973.5009159))
reaches the minimum depth. The section shows how it works, and counts its additions.

Every element works at every step. At step $k$, each element adds the value that
sits $d = 2^{k}$ places to its left, if there is one. After the step with
distance $d$, element $i$ holds the sum of the $2d$ elements ending at $i$. After
$\log_2 n$ steps, it holds the sum of everything up to $i$.

For the eight-element example, the three steps look as follows:

| Step | Distance | After the step |
|---|---|---|
| Input | | 3, 1, 7, 0, 4, 1, 6, 3 |
| 1 | 1 | 3, 4, 8, 7, 4, 5, 7, 9 |
| 2 | 2 | 3, 4, 11, 11, 12, 12, 11, 14 |
| 3 | 4 | 3, 4, 11, 11, 15, 16, 22, 25 |

The last row is the inclusive scan. Element 3 after step 1 is $0 + 7 = 7$, the sum
of two inputs. After step 2 it's $7 + 4 = 11$, where the 4 is element 1's pair sum
$3 + 1$, so element 3 now holds the sum of the first four inputs.

At distance $d$, the $d$ leftmost elements have nothing to add, so $n - d$ elements
do an addition. The total is:

$$
\boxed{W_{\text{KS}} = \sum_{k=0}^{\log_2 n - 1} (n - 2^{k}) = n \log_2 n - n + 1}
$$

For $n = 8$ that's $24 - 8 + 1 = 17$ additions against the serial loop's 7. For a
256-element block it's $256 \times 8 - 255 = 1793$ against 255: seven times the
work in exchange for 8 steps instead of 255.

The shared-memory version has one hazard that the table hides. In the table, every
element reads the *old* row and writes the *new* row. In place, a thread that adds
and stores can overwrite a value that its right-hand neighbor hasn't read yet.
The fix is a read, a barrier, a write, and a barrier, at every step:

```cuda
for (int d = 1; d < BLOCK; d <<= 1) {
  int add = (tid >= d) ? buf[tid - d] : 0;   // read the old value
  __syncthreads();                           // everyone has read
  buf[tid] += add;                           // now overwrite
  __syncthreads();                           // everyone has written
}
```

The `add` variable holds the read in a register across the barrier. That's the
whole trick: registers are private, so a value read before the barrier survives a
write by another thread after it.

## Blelloch: work-efficient and deep

Blelloch's scan ([Blelloch, 1990](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf))
trades depth for work. It builds a binary tree over the array, sums up the tree,
and then pushes prefixes back down. The section walks both sweeps on the example.

**The up-sweep** is a reduction. At stride 1, add each odd-position element to the
one before it. At stride 2, add pairs of pairs, and so on, until the last element
holds the total. **The down-sweep** then clears the root and works back down. At
each node, the left child receives the parent's value, and the right child
receives the parent's value plus the old left child:

| Stage | Array |
|---|---|
| Input | 3, 1, 7, 0, 4, 1, 6, 3 |
| Up-sweep, stride 1 | 3, **4**, 7, **7**, 4, **5**, 6, **9** |
| Up-sweep, stride 2 | 3, 4, 7, **11**, 4, 5, 6, **14** |
| Up-sweep, stride 4 | 3, 4, 7, 11, 4, 5, 6, **25** |
| Clear the root | 3, 4, 7, 11, 4, 5, 6, **0** |
| Down-sweep, stride 4 | 3, 4, 7, **0**, 4, 5, 6, **11** |
| Down-sweep, stride 2 | 3, **0**, 7, **4**, 4, **11**, 6, **16** |
| Down-sweep, stride 1 | **0**, **3**, **4**, **11**, **11**, **15**, **16**, **22** |

The final row is the exclusive scan from the earlier table. The up-sweep does
$n/2 + n/4 + \dots + 1 = n - 1$ additions, the down-sweep does $n - 1$ more, and
the swaps are free:

$$
\boxed{W_{\text{B}} = 2(n - 1), \qquad D_{\text{B}} = 2 \log_2 n}
$$

For a 256-element block that's 510 additions in 16 steps, two times the serial
loop's work and 3.5 times fewer additions than Kogge-Stone's 1793. This network is
the Brent-Kung prefix network from adder design, with the bookkeeping that makes
the output exclusive. In the lab's index arithmetic, node $t$ at stride
`offset` combines `a = offset*(2*t+1)-1` with `b = offset*(2*t+2)-1`. Check that
against the table: at stride 2, thread 0 has $a = 2 \times 1 - 1 = 1$ and $b = 2 \times 2 - 1 = 3$,
so it adds element 1 (value 4) into element 3 (value 7), which is the 11.

```viz
07-scan-networks
```

Step through both networks in the figure and watch the arrows. Kogge-Stone has
long rows of arrows and few rows. Blelloch has short rows that shrink toward the
root and widen again on the way down.

## Which network wins inside a block

Counting additions says Blelloch wins. The GPU doesn't charge by the addition, so
this section recounts what a block of 256 threads actually pays for.

A step costs a warp-wide instruction for each warp that has an active thread, not
for each active thread. An `if (tid < d)` with $d = 8$ still occupies one whole
warp. So count warp-steps from the lab's code:

| Network | Steps | Active warps per step | Warp-steps | Barriers |
|---|---|---|---|---|
| Kogge-Stone | 8 | 8, every step | $8 \times 8 = 64$ | 17 |
| Blelloch | 16 | 4, 2, 1, 1, 1, 1, 1, 1 up, and the reverse down | $12 + 12 = 24$ | 18 |

The active-warp counts follow from `d` doubling or halving: with $d = 128$,
threads 0 to 127 are active, which is $128 / 32 = 4$ warps. Kogge-Stone has more
warp-steps, but its 17 barriers equal Blelloch's 18, and a barrier is where each
warp waits for the slowest. When a block is this small, the cost is dominated by
barriers and shared-memory round trips, not by the additions. The work
advantage of Blelloch doesn't turn into a time advantage by itself, and the lab
measures both.

Blelloch also has a memory problem. Consecutive threads at stride `offset` touch
addresses $2 \times \text{offset}$ words apart, and a power-of-two stride collides in the
32 banks: a stride of $s$ words puts $\gcd(s, 32)$ lanes of a warp on one bank, so at
$s = 32$ every active lane hits the same bank. The classic cure is to pad the index
with `i + (i >> 5)`, which adds one word of padding per 32 words
([chapter 8](/c/08-shared-memory-and-transpose) derives it). The lab's block scans
leave the padding out, so you can find the conflicts with the tools of
[chapter 5](/c/05-profiling) and add the padding as an exercise.

> [!KEY] Work-efficiency matters when work is the limit
> Blelloch's advantage is fewer operations. Inside one block, the limit is
> barriers and shared-memory latency, and there the two networks cost about the
> same. The scan that production kernels use inside a warp is neither: it is
> Kogge-Stone with shuffles, which has no barriers at all.

## Scan inside a warp, with shuffles

A warp already moves data between its lanes without shared memory. This section
builds a 32-element scan from five shuffles, and explains the one rule that keeps it correct.

`__shfl_up_sync(mask, v, d)` hands lane $i$ the value that lane $i - d$ holds.
A lane with $i < d$ gets back its own value unchanged. Kogge-Stone maps onto it
directly:

```cuda
template <typename T>
__device__ __forceinline__ T warp_inclusive_scan(T v) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    T up = __shfl_up_sync(0xffffffffu, v, d);   // every lane takes part
    if (lane >= d) v += up;                     // only lanes with a left neighbor add
  }
  return v;
}
```

Five steps scan 32 lanes. Each step is one shuffle and one add, with no
shared-memory traffic and no barrier, because the lanes of a warp execute each
instruction together. The 129 additions that this does ($31 + 30 + 28 + 24 + 16$)
are free in practice: all 32 lanes issue in the same cycle either way.

> [!WARNING] The shuffle must run on every lane
> The mask `0xffffffff` says that all 32 lanes take part. If you write
> `if (lane >= d) v += __shfl_up_sync(0xffffffffu, v, d);`, the lanes below $d$
> skip the shuffle, the call names lanes that never arrive, and the result is
> undefined or the kernel stops responding. Keep the shuffle outside the `if` and
> guard only the add, as the code above does. NVIDIA's
> [warp-level primitives post](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)
> explains the rule.

To get the *exclusive* scan from the inclusive one, don't subtract the lane's own
value: for floating point that cancels digits. Shuffle the inclusive result up by
one more lane, and put the identity in lane 0:

```cuda
T excl = __shfl_up_sync(0xffffffffu, incl, 1);
if (lane == 0) excl = T(0);
```

## From a warp to a block, and to four items per thread

A block of 256 threads is eight warps, and a warp scan only covers its own 32
lanes. This section combines them, then gives each thread four elements to
reduce the work further.

The block scan has three phases:

1. Each warp scans its 32 values with the shuffle scan. Lane 31 holds the warp's
   total.
2. Each warp's lane 31 writes that total to shared memory, one slot per warp.
   After a barrier, warp 0 scans those 8 totals with the same shuffle scan, and
   another barrier follows.
3. Each thread adds the scanned total of the warps *before* its own to its
   in-warp prefix.

That costs two barriers and a 32-byte shared array for 8 integers, against the 17
or 18 barriers and kilobytes of traffic of a shared-memory network. This is how
CUB's block scan works too, and it's what the lab uses for the device scan.

The lab adds *thread coarsening*: each thread handles four consecutive elements,
so a block covers a tile of $256 \times 4 = 1024$ elements. The kernel has three
phases:

```cuda
// 1. Each thread scans its own 4 items serially, in registers.
T run = 0;
for (int k = 0; k < 4; ++k) { T x = items[k]; items[k] = run; run += x; }
// 2. Scan the 256 per-thread totals across the block.
T prefix = block_exclusive_scan<T>(run, &total);
// 3. Add the thread's prefix (and the tile's offset) to its items.
for (int k = 0; k < 4; ++k) items[k] += prefix + tile_offset;
```

Coarsening helps three ways. The parallel scan runs on 256 values instead of 1024,
saving two of the ten Kogge-Stone steps. The serial phase uses registers and no
barriers. And four consecutive 32-bit values are one 128-bit load, the widest
access a thread can issue ([chapter 3](/c/03-memory-and-bandwidth)): a block's
load puts $256 \times 16 = 4096$ bytes in flight. [Chapter 4](/c/04-roofline-and-latency)'s
Little's-law estimate needs about 5.9 KB in flight per SM to saturate the A100's
memory, so two resident blocks are already enough.

## A scan across blocks

A block can scan only its own tile, and blocks can't wait for one another. The
section sets up the problem a device-wide scan has to solve, then builds the
three-kernel answer.

Element 5,000 needs the sum of everything before it, including tiles that other
blocks are processing right now. Within a kernel launch, the programming model
guarantees no order between blocks and gives no safe way for one block to wait for
another's result. The classic answer is ==to make the dependency cross a kernel
boundary==, because consecutive kernel launches in a stream *are* ordered.

**Reduce-then-scan** uses three steps:

1. **Reduce.** Each block sums its tile and writes one value to `sums[block]`.
   This reads $n$ elements.
2. **Scan the sums.** An exclusive scan of `sums` gives each tile's offset. For a
   million elements this array is only about a thousand entries, so the step
   recurses: if the sums need more than one tile, scan them with the same
   algorithm.
3. **Scan with offsets.** Each block rescans its tile and adds its tile's offset
   to every output. This reads $n$ elements and writes $n$.

A variant, *scan-then-propagate*, scans each tile first, scans the sums, and
then makes a third pass to add the offsets. That pass reads $n$ and writes $n$
again, for 4n in total against 3n here. The lab builds reduce-then-scan.

### The traffic and the floor

For an array of $n = 2^{26}$ int32 values, one copy of the array is
$2^{26} \times 4 = 268{,}435{,}456$ bytes, 256 MiB. That's far larger than the
40 MiB L2, so each pass really reads DRAM. Count the traffic of each scheme and
divide by the 1275 GB/s copy ceiling:

| Scheme | Traffic | Bytes | Time floor at 1275 GB/s |
|---|---|---|---|
| Decoupled look-back, single pass | $2n$ elements (read $n$, write $n$) | 536,870,912 | 421.1 µs |
| Reduce-then-scan | $3n$ (read $n$, then read $n$ and write $n$) | 805,306,368 | 631.6 µs |
| Scan-then-propagate | $4n$ | 1,073,741,824 | 842.2 µs |

Each time is bytes divided by $1.275 \times 10^{12}$ bytes per second. The single
pass takes exactly the time of a copy, which is its selling point. The tile sums
add little: $2^{26} / 1024 = 65{,}536$ sums, 262,144 bytes, 0.1% of the array.

The recursion's depth for this size is three levels. The first level has 65,536
tiles, so its sums form a 65,536-element array. That array has 64 tiles, and its
sums form a 64-element array, which one block scans directly. The whole scan
launches five kernels (two reduces and three scans), and at roughly 5 µs to queue
each (learn-inference, chapter 12) that's about 25 µs, or 4% of the 631.6 µs floor. The
small kernels are latency-bound, not bandwidth-bound, but they touch almost no data.

> [!EXAMPLE] What the lab's scan floor means
> The lab reports `scan_gbs` as $3n \times 4$ bytes divided by the measured time,
> and requires at least 40% of the copy ceiling: $0.40 \times 1275 = 510$ GB/s.
> At $n = 2^{26}$, that allows $805{,}306{,}368 / 510 \times 10^9 = 1.58$ ms,
> against the 0.632 ms floor. The test is loose on purpose, and your lab prints the
> real figure.

## Decoupled look-back: one pass over the data

Reduce-then-scan reads the input twice because the offsets aren't known until
every tile is summed. This section explains how Merrill and Garland's
[single-pass scan](https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf)
gets around that, and it's an optional stretch goal for the lab.

The idea is to let a tile's block wait for the earlier tiles' results *inside*
one kernel, using flags in global memory. Each tile has a status slot with one of
three states:

- **Invalid.** The tile hasn't published anything yet.
- **Aggregate.** The tile's own sum is available, but its inclusive prefix isn't.
- **Prefix.** The tile's inclusive prefix, the sum of everything up to and
  including it, is available.

A block runs the following steps:

1. Take a tile number from a global counter with `atomicAdd`, so tiles are
   numbered in the order blocks start.
2. Load the tile, compute its sum, and publish it with the flag *aggregate*
   (tile 0 publishes *prefix* straight away).
3. **Look back.** Walk from tile $k - 1$ toward tile 0. Wait while a slot is
   invalid. If it's *aggregate*, add its value and continue. If it's *prefix*,
   add its value and stop, because that value already covers every earlier tile.
4. The sum of what the walk collected is this tile's offset. Scan the tile with
   it, write the output, and publish the inclusive prefix with the flag *prefix*.

Step 3 is the "decoupled" part: a tile that can't see its neighbor's final prefix
yet still makes progress by adding the aggregate, and only the rare long walk
waits. The following sketch shows the protocol with one thread doing the walk.
Production code uses a warp for it and handles memory ordering with care:

```cuda
// One 64-bit word per tile: the value in the low 32 bits, the status in the high 32.
// Packing both makes them visible together, so a reader never sees a new flag with an old value.
enum { INVALID = 0, AGGREGATE = 1, PREFIX = 2 };

int tile = atomicAdd(tile_counter, 1);               // dynamic numbering, once per block
int agg  = tile_sum(...);
publish(&slots[tile], agg, tile == 0 ? PREFIX : AGGREGATE);

int offset = 0;
for (int j = tile - 1; j >= 0; --j) {
  unsigned long long w;
  do { w = *(volatile unsigned long long*)&slots[j]; } while ((w >> 32) == INVALID);
  offset += (int)w;                                  // an aggregate or a prefix: either way, add it
  if ((w >> 32) == PREFIX) break;                    // a prefix covers everything before j
}
publish(&slots[tile], offset + agg, PREFIX);
```

The dynamic tile number in the first line prevents a deadlock. If blocks took
their tile number from `blockIdx.x`, a block could wait for tile $k-1$ while every
SM slot is held by tiles that came after it and never started. With the counter,
a block's tile number is assigned when it starts running, so every tile it waits
on has already started. That makes progress guaranteed in practice, and it's why
the algorithm needs the hardware to schedule blocks in launch order.

The result is one read of the input, one write of the output, and a few small
flag reads: the $2n$ traffic in the table. Merrill and Garland report that it
matches memcpy throughput on the hardware they tested. CUB's `DeviceScan` is built on this
algorithm, so for production scans, call the library and use this chapter to
understand what it does.

## The histogram: an update that collides

A histogram counts how many inputs fall into each bin. This section explains why
the obvious kernel is slow, and where the collisions come from.

The task is `bins[x[i]] += 1` for every element, with $B$ bins. The lab uses
$B = 256$ and byte inputs, as in an image histogram, a byte-frequency table for
compression, or the calibration step of a quantizer. Each thread handles an
element, but the *address* it updates depends on the data. Two threads can
update the same bin at the same moment, so the update must be an atomic.

The baseline is one global atomic per element:

```cuda
for (int64_t i = tid; i < n; i += stride) atomicAdd(&bins[in[i]], 1);
```

Atomics on global memory are carried out in the L2 cache. Updates to different
addresses proceed in parallel, but ==updates to one address are applied one after
another==. That makes the cost depend on the data:

- **Uniform bytes.** With 256 bins, each bin receives about one update in 256, so
  collisions are rare among the lanes of a warp, but the whole GPU is still
  sending $n$ atomic operations to 256 addresses, which is only 1 KiB of L2 lines.
- **Skewed bytes.** A text file, an image with a black border, or a token stream
  where one token dominates sends a large share of updates to one bin. Those
  serialize, and the kernel can run far below the memory bandwidth.
- **All bytes equal.** Every update in the program targets one address. This is the
  worst case, and the lab measures it.

```viz
07-histogram-contention
```

The figure draws 32 random bytes as one warp and counts collisions. At 0% hot
bytes, a warp's 32 lanes touch about 30 distinct bins, and the deepest queue on
one bin is small. At 100% the figure shows 1 distinct bin, a queue of 32, and the
global-atomic queue reaches its maximum of 32. The figure's numbers are statistics of a
random draw, not GPU measurements. The lab measures the real cost.

## Privatize: one histogram per block

Privatization gives each block its own copy of the bins in shared memory, so the
hot updates never leave the SM. The section describes the kernel and counts how
many global atomics it saves.

Each block zeroes a 256-bin array in shared memory, updates it with shared-memory
atomics as it streams through its share of the input, and adds its array to the
global bins when it's done:

```cuda
__shared__ int local[256];
for (int b = threadIdx.x; b < 256; b += blockDim.x) local[b] = 0;
__syncthreads();
// ... stream the input: atomicAdd(&local[byte], 1) for every byte ...
__syncthreads();
for (int b = threadIdx.x; b < 256; b += blockDim.x)
  if (local[b]) atomicAdd(&bins[b], local[b]);      // 256 global atomics per block, at most
```

The lab launches $108 \times 8 = 864$ blocks of 256 threads, so 8 blocks fill an
SM's 2048 threads, and each block loops over the input with a grid stride. That
gives 864 merges of at most 256 global atomics each, which is 221,184 in the
worst case. The baseline sends $2^{26} = 67{,}108{,}864$ atomics to global memory
for the same input, so privatization removes a factor of
$67{,}108{,}864 / 221{,}184 = 303$ of the global atomics.

Three costs remain, and each is worth understanding.

**Shared-memory atomics still collide.** Within a warp, two lanes that update the
same bin serialize, and so do two lanes whose bins fall in the same bank: bin $b$
sits in bank $b \bmod 32$. With random bytes, a few lanes per warp share a bank
(the figure's third readout estimates it), and with all bytes equal all 32 do.
A shared atomic is much cheaper than a global one, but it isn't free.

**The read side needs vector loads.** Reading bytes one per thread wastes the memory
pipeline: 32 lanes request 32 bytes, one sector. The lab's kernel loads a
`uint4`, 16 bytes per thread, and extracts the 16 bytes with shifts. With 864
blocks of 256 threads that's 221,184 threads and
$2^{26} / 16 = 4{,}194{,}304$ loads, about 19 per thread. The kernel handles any
tail bytes and an input that doesn't start on a 16-byte boundary: the host passes
`nvec = 0` for a misaligned pointer and the byte loop covers everything.

**The bins must fit.** A 256-bin, 32-bit array is 1 KiB, so 8 blocks use 8 KiB of an
SM's 164 KiB of shared memory. The ceiling is the 163 KiB one block may hold:
$163 \times 1024 / 4 = 41{,}728$ bins. A histogram over a 128,000-token vocabulary
doesn't fit, and one with more than a few thousand bins already cuts the number of
resident blocks per SM. Such a histogram privatizes a slice of the bins per block, or sorts first, or
falls back to global atomics.

How fast should a good kernel go? The histogram reads $n$ bytes and writes 256
counters, so it's a read-only stream. At the 1275 GB/s copy ceiling, 67,108,864
bytes take $67{,}108{,}864 / 1.275 \times 10^{12} = 52.6$ µs. The atomics need
$2^{26} / 108 / 32 = 19{,}418$ warp-wide updates per SM. If an SM completed one
conflict-free warp update per clock, that's $19{,}418 / 1.41 \times 10^{9} = 13.8$ µs,
well under the memory time, so the kernel is memory-bound when the data
doesn't collide, and the data decides whether that holds. The 1-per-clock rate is
an assumption, not a spec: the lab reports `hist_gbs` for uniform bytes, and
requires 25% of 1275 GB/s, about 319 GB/s, which allows four times the floor.

## Aggregate: when the data is skewed

A skewed input defeats privatization, because the hot bin still collides inside
the block. The section covers two ways to merge duplicates before they reach an atomic.

**Per-warp copies.** Give each warp its own array of 256 bins instead of one per
block: $8 \times 1\ \text{KiB} = 8$ KiB per block, 64 KiB for the 8 blocks of an
SM. Lanes of different warps no longer collide on a bin, so the number of
contenders drops from 256 threads to 32 lanes. The block adds the eight arrays
at the end. It's more shared memory for less contention, so it's worth it when
the bins are few. Lanes inside a warp still collide, which the next option fixes.

**Warp aggregation.** Let the lanes that hold the same byte agree among
themselves first, then do one atomic per group. `__match_any_sync(mask, key)`
returns, for each lane, a bitmask of the lanes whose key equals its own:

```cuda
unsigned peers = __match_any_sync(0xffffffffu, key);
if (lane == __ffs(peers) - 1)                  // the lowest lane in each group is the leader
  atomicAdd(&local[key], __popc(peers));       // one atomic adds the whole group's count
```

When all 32 bytes are equal, 32 atomics become 1. When all 32 differ, nothing is
saved and the match costs extra. So aggregation wins on skewed data and loses on
uniform data, and which kernel to use depends on the data.
The lab's stretch goal has you write it and compare the two on both inputs.

**Run aggregation.** When the same value often repeats along an array, such as a
run of black pixels, a thread can scan its own consecutive items in registers,
count a run of equal values, and flush one atomic when the value changes. It
costs one compare per item and is cheaper than a match, but it only helps for
runs within a thread's own contiguous items.

> [!KEY] Move the contention to the cheapest place
> Global atomics collide at L2, shared atomics collide at the bank, and
> warp-aggregated updates collide at nothing. Each level of privatization or
> aggregation trades extra work, shared memory, or registers for fewer atomics at
> the level above.

## Counting sort: scan and histogram together

The two halves of the chapter combine, and the combination is behind two kernels
you'll meet again. The section builds a counting sort and connects it to mixture-of-experts routing.

A counting sort of small integer keys has three steps: histogram the keys, take
the exclusive scan of the counts to get each key's first output slot, and scatter
each element to its key's slot. Take the 2-bit digits $[3, 1, 2, 1, 0, 3, 2, 1]$:

| Digit | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| Histogram | 1 | 3 | 2 | 2 |
| Exclusive scan | 0 | 1 | 4 | 6 |

Digit 0 starts at slot 0, digit 1 at slot 1, digit 2 at slot 4, and digit 3 at
slot 6. A radix sort repeats this for each digit of the key, and in a GPU radix
sort each block computes its own histogram and a scan combines them. GPU
libraries' radix sorts, such as CUB's, are built from these parts.

A mixture-of-experts layer does the same with experts as the keys. With 8
token-to-expert assignments and 4 experts, suppose the expert counts are
$[3, 0, 2, 3]$. The exclusive scan $[0, 3, 3, 5]$ is where each expert's block of
tokens starts in the permuted buffer, and the total, 8, is the buffer's length.
[Chapter 20](/c/20-moe-and-sparse-kernels) builds the permutation and the grouped
GEMM that consumes it.

## What goes wrong

- **Mixing up inclusive and exclusive.** An off-by-one in a compaction, or a
  token kept in top-p that shouldn't be, comes from using the inclusive scan
  where a position or a "mass before" is needed. Test both on `[1,1,1,1]`.
- **One barrier in Kogge-Stone.** Without the barrier between the read and the
  write, a thread can overwrite a value its neighbor hasn't read, and the result
  is wrong only on some runs. Read into a register, barrier, write, barrier.
- **A `__syncthreads()` inside an `if (tid < d)`.** Threads that skip the branch
  never reach the barrier, and the block stops responding. Keep every barrier at
  the top level of the loop body.
- **A shuffle under a divergent branch.** A full-mask `__shfl_up_sync` needs all 32
  lanes. Put the shuffle outside the condition and guard only the add.
- **Not padding the tail with the identity.** For the last, partial tile, any
  element past $n$ must count as 0, or the garbage that was in memory corrupts the
  tile's total and every offset after it. The lab tests sizes like 1, 255, 257,
  1025, and $2^{20} + 1$ for this reason.
- **Forgetting that the tile sums need scanning too.** For more than 1024 elements
  the sums array has more than one tile, and one block can't scan it. The lab's
  host code recurses, and a size of $2^{20} + 1$ reaches three levels.
- **Comparing float scans exactly.** A parallel scan adds in a different order
  from `torch.cumsum`, so random floats differ in the last bits. Compare small
  integers stored as floats exactly, and random data with a relative tolerance.
- **Not zeroing the bins, or not synchronizing around them.** The shared bins
  need a zero pass and a barrier before the first atomic, and another barrier
  before the merge. The global bins need `torch.zeros`, not `torch.empty`.
- **Vector loads on a misaligned pointer.** A `uint4` load needs a 16-byte aligned
  address. A tensor view such as `x[1:]` starts one byte in, and the lab tests it.
- **Timing a tiny input.** A launch costs about 5 µs, so a kernel that moves a few
  kilobytes measures the launch, not the memory. Both timed kernels use
  $2^{26}$ elements.

> [!RECAP]
> - A scan gives every running total. Use the exclusive version when you need
>   "how many before me" or a destination slot.
> - Kogge-Stone has the least depth and $n \log_2 n - n + 1$ additions. Blelloch
>   has $2(n - 1)$ additions and twice the depth. In a warp, use shuffles
>   (Kogge-Stone, five steps, no barrier). In a block, scan warps with shuffles and
>   scan the warp totals with one warp.
> - A device-wide scan crosses kernel boundaries. Reduce-then-scan moves 3n
>   elements, and decoupled look-back moves 2n, the traffic of a copy.
> - A histogram's collisions come from the data. Privatize in shared memory to
>   cut the global atomics by about 300 times, and aggregate inside the warp
>   when the data is skewed.
> - A histogram plus a scan is a counting sort, and the offsets of a
>   mixture-of-experts layer come from the same pair.

## Check your understanding

> [!QUESTION] What are the inclusive and exclusive scans of `[2, 5, 1, 4]`?
> The inclusive scan is `[2, 7, 8, 12]` and the exclusive scan is `[0, 2, 7, 8]`.
> The exclusive scan shifts the inclusive scan right by one and puts the identity,
> 0, in front.

> [!QUESTION] For a 1024-element tile, how many additions do Kogge-Stone and Blelloch do, and what does the serial loop do?
> Kogge-Stone does $1024 \times 10 - 1024 + 1 = 9217$. Blelloch does
> $2 \times 1023 = 2046$. The serial loop does 1023. Blelloch is within a factor
> of 2 of the serial loop's work, and Kogge-Stone is about 9 times above it.

> [!QUESTION] A block-level Kogge-Stone scan has the lines `buf[tid] += buf[tid - d];` and one `__syncthreads()` per step. What goes wrong?
> A thread can add into `buf[tid]` before a thread to its right has read the old
> `buf[tid]`, so that thread then adds the wrong value. The fix is to read into a
> register, barrier, write, and barrier again, so every read of the old row
> finishes before any write of the new one.

> [!QUESTION] The lab's device scan of $2^{26}$ int32 values moves 3n elements. What is the time floor at the copy ceiling, and what would a single-pass scan's floor be?
> $3 \times 2^{26} \times 4 = 805{,}306{,}368$ bytes at $1.275 \times 10^{12}$ bytes
> per second is 631.6 µs. A single-pass scan moves 2n, which is 536,870,912 bytes
> and 421.1 µs, the time of a copy.

> [!QUESTION] Why does decoupled look-back take its tile numbers from an atomic counter and not from `blockIdx.x`?
> A block waits for earlier tiles. If tile numbers followed `blockIdx.x`, the GPU
> could run later tiles on every SM slot, and a block waiting on an earlier
> tile that hasn't started would wait forever. The counter numbers tiles in the
> order blocks start, so every tile a block waits on is already running.

> [!QUESTION] A histogram has 256 bins and $2^{26}$ byte inputs, and the lab launches 864 blocks. How many global atomics does the baseline send, and how many does the privatized kernel send at most?
> The baseline sends one per element: $2^{26} = 67{,}108{,}864$. The privatized
> kernel sends at most 256 per block: $864 \times 256 = 221{,}184$, about 303 times fewer.

> [!QUESTION] On which input does warp aggregation help, and on which does it hurt?
> It helps on skewed input, where many lanes of a warp hold the same byte, so 32
> atomics collapse to a few. It hurts on uniform input, where almost every lane
> differs: the aggregation saves nothing and the match instruction adds work.

> [!QUESTION] Why is a histogram with a 128,000-entry vocabulary a poor fit for per-block shared-memory privatization?
> A 32-bit array of 128,000 bins is 500 KiB, more than the 163 KiB a block may
> use on the A100, and even a quarter of that would leave one block per SM.
> Privatize a slice of the bins per block, sort first, or use global atomics.

## Lab

> [!TRY] Scan and histogram
> Fill in the `TODO` markers in the CUDA source in `starter.py`. You write:
> a Kogge-Stone block scan and a Blelloch block scan (`block_scan_kogge_stone`,
> `block_scan_blelloch`), a warp shuffle scan and a block scan that the device
> scan builds on, the tile-reduce and tile-scan kernels of a reduce-then-scan
> (`exclusive_scan`, for int32 and float32, any length), and a shared-memory
> privatized histogram (`histogram256`). The global-atomic baseline is given.
>
> The tests check every function against PyTorch on sizes that include 1, 255,
> 257, 1025, and $2^{20} + 1$, and on an unaligned view. You pass when the
> results are exact (or within the stated float tolerance), the device scan
> reaches 40% of the 1275 GB/s copy ceiling counting 3n elements, and the
> privatized histogram reaches 25% counting n bytes.
>
> The lab also reports the metrics this chapter predicts: `kogge_stone_gbs` and
> `blelloch_gbs` for the two block scans, `scan_ms` against `torch_cumsum_ms`,
> and `hist_gbs`, `hist_global_gbs`, `hist_const_gbs`, and `hist_global_const_gbs`
> for the histograms on uniform and all-equal bytes. The first run compiles with
> `nvcc` and takes 30 to 90 seconds. Two stretch goals are the warp-aggregated
> histogram and a decoupled look-back scan.

## Further reading

- [Single-pass parallel prefix scan with decoupled look-back](https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf),
  Merrill and Garland, 2016: the protocol in this chapter's look-back section,
  with its memory-traffic analysis.
- [Parallel prefix sum (scan) with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda),
  GPU Gems 3, chapter 39: the shared-memory Kogge-Stone-style and Blelloch scans,
  with the bank-conflict padding.
- [Prefix sums and their applications](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf),
  Blelloch, 1990: the up-sweep and down-sweep, and applications such as compaction.
- [A parallel algorithm for the efficient solution of a general class of recurrence equations](https://doi.org/10.1109/TC.1973.5009159),
  Kogge and Stone, 1973: the original network.
- [`cub::DeviceScan`](https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceScan.html),
  [`cub::BlockScan`](https://nvidia.github.io/cccl/unstable/cub/api/classcub_1_1BlockScan.html),
  and [`cub::DeviceHistogram`](https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceHistogram.html):
  the production versions to benchmark your kernels against.
- [Using CUDA warp-level primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/),
  Lin and Grover, 2018: the full-mask rule for `__shfl_up_sync` and
  `__match_any_sync`.
- [Using shared memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/),
  Harris, 2013: shared memory and `__syncthreads()` semantics.
