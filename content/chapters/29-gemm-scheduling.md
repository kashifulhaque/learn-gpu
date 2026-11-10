---
title: GEMM scheduling
slug: 29-gemm-scheduling
part: "Part 3 — Matrix multiplication"
summary: Decide which SM computes which slice of a matrix multiplication, with wave counts, split-K, stream-K, persistent tiles, and a fused epilogue, and pick the schedule from the shape.
minutes: 120
gpu: true
objectives:
  - "Compute the tiles, waves, and wave efficiency of a GEMM on 108 SMs, and say which shapes leave most of the GPU idle."
  - "Partition K for split-K, reduce the fp32 partials in a workspace, and explain why the atomic variant is not bit-reproducible."
  - "Deal MAC-iterations to CTAs for stream-K as fixed-size ranges, and find the tiles that need a partial-tile fix-up."
  - "Write a persistent kernel that loops over work items with a static or an atomic tile counter and visits tiles in grouped order."
  - "Count the bytes that fusing a bias, an activation, a residual, or a scale into the epilogue saves."
  - "Pick a schedule for a shape from a cost model, and say what the model leaves out."
lab: 29-gemm-scheduling
---

# GEMM scheduling

> [!TLDR]
> - A tiled GEMM is a pile of identical work units, one MAC-iteration (one K-step of one output tile). The schedule decides which SM runs which units. The kernel ends when the busiest SM ends, so an uneven deal wastes the machine.
> - Data-parallel scheduling gives each CTA one output tile. On the A100's 108 SMs, 117 tiles take two waves at 54.2% efficiency, and a decode-shaped GEMM with 32 tiles fills 29.6% of the GPU.
> - Split-K cuts K so that each tile becomes several CTAs. It needs a workspace and a reduction: reduced in a fixed order the result is reproducible, and added with fp32 atomics it is not.
> - Stream-K cuts the total MAC-iterations into fixed-size ranges, one per CTA, ignoring tile edges. Every SM does within one iteration of the same work, and the tiles that straddle a range boundary need a partial-tile fix-up.
> - A persistent kernel launches one CTA per SM and loops over work items. A fused epilogue applies bias, activation, residual, and scale to the fp32 accumulator before the only store, and saves 134 MB of traffic on a 4096 by 4096 bf16 output. The lab's cost model picks stream-K for 64 by 4096 by 4096 and data-parallel for 4096 cubed.

[Chapter 10](/c/10-gemm-from-scratch) builds a fast tile, [chapter 14](/c/14-triton) writes it in Triton, and [chapter 4](/c/04-roofline-and-latency) counts the waves of a grid and grades the counts for split-K and stream-K. None of them says what to do about the counts. This chapter does. It treats the tile kernel as a given and asks only which SM runs which part of the matrix.

The question matters most at the edges of the shape space. A training GEMM with thousands of tiles hides any schedule, because the last wave is a sliver of the work. A decode GEMM with a batch of 64 tokens has 32 tiles and 108 SMs, and a GEMM that grows by eight columns can run at half speed. In both, the arithmetic is the same and the time differs by a factor of two or three, because of how the work is dealt.

You write the CPU side of every schedule in the lab as plain Python: wave counts, the K partition, the stream-K plan, the tile order of a persistent CTA, and the bytes of an epilogue. Then you write the Triton kernels and compare them with `torch.matmul` on the A100. The chapter's times come from a cost model, and the lab measures how far reality is from it.

## Before you start

You need four things from earlier chapters, and the ownership of two topics that this chapter doesn't teach:

- **A wave.** One full set of resident CTAs. With one resident CTA per SM, the A100 holds 108, and a grid of $G$ equal CTAs takes $\lceil G/108 \rceil$ block-times. [Chapter 4](/c/04-roofline-and-latency) derives it, and its lab grades `partial_wave`, `split_k_blocks`, and `stream_k_bounds`. This chapter reuses those counts and doesn't rederive them.
- **A tile and a K-step.** One CTA computes a $B_M \times B_N$ tile of C by looping over K in steps of $B_K$, accumulating in fp32. [Chapter 14](/c/14-triton) writes that loop in Triton, and this chapter extends that kernel instead of replacing it.
- **Grouped program ordering.** The remap from a linear id to a tile that keeps neighbouring tiles' operands in the L2, from chapter 14's `grouped_tile`.
- **fp32 accumulation.** The accumulator of a long dot product stays in fp32, and chapter 9 shows what fp16 and bf16 accumulators lose.

Two topics sit in the companion course, [learn-inference](https://qwen.ifkash.dev), and this chapter links to them instead of teaching them. The skinny decode GEMM and the W4A16 GEMV are inference kernels with their own roofline. Here a decode-shaped GEMM is only a shape that stresses the schedule.

## Waves on a GEMM

This section answers the first question: how much of the GPU does a given shape use, and what is the unit of work you can move around?

Picture the output matrix C as a grid of tiles. A data-parallel kernel assigns one tile to one CTA, and the CTAs run in waves of 108. The tail wave is whatever is left over. If the tail has 9 tiles, 99 SMs wait for 9.

Take NVIDIA's worked example, which chapter 4 uses: $M = 2304$, $N = 1544$, $K = 4096$, and $256 \times 128$ tiles. The grid is $9 \times 13 = 117$ tiles. One wave runs 108, and a second wave runs the other 9. The efficiency is $117 / 216 = 54.2\%$.

Now take a decode-shaped GEMM, $M = 64$, $N = 4096$, $K = 4096$, with $64 \times 128$ tiles. The grid is $1 \times 32 = 32$ tiles. There is one wave, and it uses 32 of 108 SMs, which is $29.6\%$. No amount of tail-trimming helps, because there is no tail. There are too few tiles.

The two shapes have different problems and the same cure, which is to stop tying a CTA to a whole tile. To say how, name the unit of work. A *MAC-iteration* is one K-step of one tile: a $B_M \times B_N \times B_K$ multiply-accumulate, the body of chapter 14's `for k in range(...)` loop. A tile of K = 4096 with $B_K = 64$ has $4096 / 64 = 64$ of them. The 117-tile example is $117 \times 64 = 7488$ MAC-iterations. The decode example is $32 \times 64 = 2048$.

> [!KEY] The unit of work is the MAC-iteration, not the tile
> Every MAC-iteration costs the same, so the kernel's time is the number of iterations on the busiest SM. A schedule is a way of dealing the iterations to SMs. Data-parallel deals them in blocks of $I$ at a time, where $I = \lceil K / B_K \rceil$ is the iterations per tile. The other schedules in this chapter deal them in smaller pieces.

With $T$ tiles, $I$ iterations per tile, and $S = 108$ SMs, the best any schedule can do is the even share, and data-parallel is a multiple of the tile size:

$$
\text{even share} = \frac{\hla{T} \cdot \hlb{I}}{S}, \qquad
\text{data-parallel} = \left\lceil \frac{\hla{T}}{S} \right\rceil \hlb{I}
$$

For the 117-tile example the even share is $7488 / 108 = 69.3$ iterations and data-parallel takes $2 \times 64 = 128$. The ratio, $69.3 / 128 = 54.2\%$, is the wave efficiency from chapter 4 again. For the decode example the even share is $2048/108 = 19.0$ and data-parallel takes 64, which is $29.6\%$.

> [!EXAMPLE] What 128 against 69.3 costs, on the 256 x 128 x 64 tile
> One iteration is $2 \times 256 \times 128 \times 64 = 4{,}194{,}304$ FLOP. One SM's share of the A100's 312 TFLOP/s is $312 \times 10^{12} / 108 = 2.889$ TFLOP/s, so an iteration takes $4.194 \times 10^{6} / 2.889 \times 10^{12} = 1.452\ \mu\text{s}$ at the rated peak. Data-parallel's 128 iterations take $185.8\ \mu\text{s}$. The even share takes $69.3 \times 1.452 = 100.6\ \mu\text{s}$. Those are predictions from the ratings, and the lab does not measure them.

Three ways out follow from the unit. Cut each tile's K range so that one tile becomes several CTAs, which is split-K. Cut the whole iteration space into equal ranges, which is stream-K. Or keep one tile per CTA and make a fixed number of CTAs loop over tiles, which is persistent scheduling and helps only if tiles differ in cost or you need something else a persistent CTA gives you. The next section gives each of them a price.

## A cost model in iterations and bytes

This section answers the question you need before comparing schedules: what should a prediction include?

A kernel's time has a compute side and a memory side, and chapter 4 says which wins. Two things differ for a scheduling decision. The compute side is the critical path in iterations, because the busiest SM sets the time. The memory side depends on how many SMs are pulling data, because a grid that fills a third of the GPU pulls about a third of the bandwidth.

The lab's model has four lines. Write $c$ for the critical-path iterations, $t_{\text{it}}$ for one iteration at the per-SM peak, $a$ for the number of SMs that have work, and $\beta$ for the 1275 GB/s copy ceiling that learn-inference measured on this A100:

$$
T_{\text{kernel}} = \max\!\Big( \hla{c}\cdot \hlb{t_{\text{it}}},\ \frac{\text{bytes}}{\beta} \cdot \frac{108}{\hlc{a}} \Big) + T_{\text{extra}}
$$

$\hla{c}$ is the busiest SM's iterations, and $\hlb{t_{\text{it}}} = 2 B_M B_N B_K \cdot 108 / 312\text{ TFLOP/s}$. The factor $108/\hlc{a}$ stretches the memory time when only $a$ SMs run. $T_{\text{extra}}$ is what a schedule adds, which later sections fill in. The bytes are the compulsory ones for a bf16 GEMM, $2(MK + KN + MN)$.

Two caveats belong with the model before any number does.

- **The linear bandwidth scale is an assumption.** It says that 32 busy SMs pull $32/108$ of the bandwidth. Chapter 4's warning about grids smaller than the machine says the penalty is real, because Little's law needs bytes in flight across the whole GPU. A single SM can have more in flight than its fair share, so the real penalty is smaller than the model's.
- **One CTA per SM.** The model gives every SM a single resident CTA, as chapter 4's lab does. A small tile can have two or more resident, which changes the wave counts.

Apply it to the decode shape. The compulsory bytes are $2 \times (64 \cdot 4096 + 4096 \cdot 4096 + 64 \cdot 4096) = 34.6$ MB, which is $27.1\ \mu\text{s}$ at 1275 GB/s. The arithmetic is $2 \times 64 \times 4096^2 = 2.15$ GFLOP, which is $6.9\ \mu\text{s}$ at 312 TFLOP/s. The intensity is $62$ FLOP per byte, well below the 244.7 ridge that chapter 4 computes against the measured bandwidth, so the GEMM is memory bound. A data-parallel launch of 32 CTAs stretches the $27.1\ \mu\text{s}$ by $108/32$ to $91.6\ \mu\text{s}$, where the compute side is $23.2\ \mu\text{s}$. The prediction is that the kernel is 3.4 times slower than its memory floor because two thirds of the SMs have no tile.

## Split-K

This section answers how to give a small grid more CTAs without making the tile smaller.

Picture a tile of C as a column of K-steps. Cut the column into $s$ pieces and give each piece to its own CTA. Each CTA produces a partial sum of the same $B_M \times B_N$ tile over its slice of K, and the true tile is the sum of the partials. Instead of $T$ CTAs, you launch $T \cdot s$. The decode shape's 32 tiles with $s = 3$ become 96 CTAs, which is 88.9% of the SMs.

### The partition

Cut the $I$ iterations of a tile into $s$ ranges whose sizes differ by at most one. Let $q = \lfloor I/s \rfloor$ and $r = I \bmod s$. The first $r$ splits take $q+1$ iterations and the rest take $q$:

```python
def split_k_ranges(k_iters, splits):
    base, rem = divmod(k_iters, splits)
    out, start = [], 0
    for s in range(splits):
        size = base + (1 if s < rem else 0)
        out.append((start, start + size))
        start += size
    return out
```

For 16 iterations in 5 splits the ranges are $[0,4)$, $[4,7)$, $[7,10)$, $[10,13)$, $[13,16)$: one split of 4 and four of 3. A plausible wrong version uses `k_iters // splits` for every split, and it covers 15 of the 16 iterations and silently drops the last. The lab's test catches it.

K rarely divides evenly. With $K = 1000$ and $B_K = 64$ there are $\lceil 1000/64 \rceil = 16$ iterations, and the last covers columns 960 to 999, 40 columns of a 64-wide step. The masks from chapter 14 handle that, and the partition counts iterations, not columns, so a short last step goes to whichever split owns it.

### The workspace and the reduction

The CTAs of one tile finish at different times and can't add their results in place without a protocol. The standard answer has two parts. Each CTA writes its fp32 partial to a workspace of shape $(s, M, N)$, and a second kernel adds the $s$ slices in a fixed order and casts to the output dtype. CUTLASS describes the same design, a partitioned-K GEMM followed by a batched reduction, with workspace the caller allocates.

The partial kernel is chapter 14's kernel with two changes. The grid gains a second axis for the split, and the K loop runs only over the split's range:

```python
tile_id = tl.program_id(0)
split = tl.program_id(1)
k_iters = tl.cdiv(K, BLOCK_K)
base = k_iters // SPLITS
rem = k_iters % SPLITS
k_begin = split * base + tl.minimum(split, rem)
k_end = k_begin + base + (split < rem).to(tl.int32)
for kk in range(k_begin, k_end):
    ...                                   # the chapter 14 body, with ks = kk * BLOCK_K + offs_k
tl.store(ws + split * M * N + offs_m[:, None] * N + offs_n[None, :], acc, mask=mask)
```

The reduction kernel is a loop over splits on a flat range of $M \times N$ elements. The loop order is fixed, so the same inputs give the same bits on every run:

```python
acc = tl.zeros((BLOCK,), dtype=tl.float32)
for s in tl.static_range(SPLITS):
    acc += tl.load(ws_ptr + s * MN + offs, mask=mask, other=0.0)
tl.store(c_ptr + offs, acc.to(c_ptr.dtype.element_ty), mask=mask)
```

### What the reduction costs

The partials are fp32 whatever the output dtype, so the workspace is $4 s M N$ bytes. The partial kernels write it and the reduction kernel reads it, which is $2 \cdot s \cdot M N \cdot 4$ bytes of traffic. The reduction is a second launch, which chapter 4's launch floor prices at about 5 microseconds on this card:

$$
\boxed{T_{\text{extra}}^{\text{split-K}} = 5\ \mu\text{s} + \frac{2\,\hla{s}\,M N \cdot 4}{\beta}}
$$

The first term is the launch of the reduction kernel and the second is the workspace traffic at the measured copy ceiling $\beta$, with $\hla{s}$ the split count. Neither depends on K. That is why split-K wins where K is long and the output is small, and loses where the output is large.

> [!EXAMPLE] Split-K on the decode shape and on the 117-tile shape
> **Decode, $64 \times 4096 \times 4096$, $64 \times 128$ tiles.** With $s = 3$ the grid is 96 CTAs, the critical path is 22 iterations, and utilisation is $2048/(108 \times 22) = 86.2\%$. The workspace traffic is $2 \times 3 \times 64 \times 4096 \times 4 = 6.29$ MB, which is $4.9\ \mu\text{s}$, plus the $5\ \mu\text{s}$ launch. The model gives $\max(8.0, 30.5) + 9.9 = 40.5\ \mu\text{s}$ against $91.6\ \mu\text{s}$ for data-parallel, a 2.3 times speedup. With $s = 4$ it gives $38.7\ \mu\text{s}$, because the extra SMs lift the memory side from 30.5 to $27.1\ \mu\text{s}$ even though 128 CTAs run in two waves.
>
> **The 117-tile shape, $2304 \times 1544$.** With $s = 2$ the grid is 234 CTAs on 3 waves of 32 iterations, which is 96 iterations against 128. But the workspace is $2 \times 2 \times 2304 \times 1544 \times 4 = 56.9$ MB of traffic, which costs $44.6 + 5 = 49.6\ \mu\text{s}$. The model gives $139.4 + 49.6 = 189.0\ \mu\text{s}$, slightly worse than the $185.8\ \mu\text{s}$ of data-parallel. The workspace for $s = 2$ is 28.5 MB, which fits in the 40 MiB L2, so the real reads may hit the cache and the model is pessimistic here. The lesson survives: a large output makes the reduction expensive.

### The atomic variant

The second way to combine partials skips the workspace. Zero the output in fp32 once. Each CTA then adds its partial into the output with `tl.atomic_add`, and a final cast produces the output dtype. There is no second reduction kernel, so the launch and the read of the workspace go away.

The price is determinism. fp32 addition is not associative, and the hardware adds the partials in whatever order the CTAs reach the L2. Chapter 9 shows rounding. Here is a case where it is large. The fp32 spacing at $10^8$ is 8, so adding 1 to $10^8$ returns $10^8$. Take the four partials $10^8$, $1$, $-10^8$, $1$ and add them one at a time in fp32:

| Order of the partials | Running sum | Result |
|---|---|---|
| 0, 1, 2, 3 | $10^8 \to 10^8 \to 0 \to 1$ | 1.0 |
| 0, 2, 1, 3 | $10^8 \to 0 \to 1 \to 2$ | 2.0 |
| 1, 3, 0, 2 | $1 \to 2 \to 10^8 \to 0$ | 0.0 |

The true sum is 2. Three orders of the same set give three answers, and one of them is exactly right only by luck. Real partial sums are not so adversarial, and the difference is usually in the last bits, but you can't reproduce a run bit for bit. The lab's `reduce_partials` takes an order and adds in fp32, so you can see the table yourself. It also catches a version that sums in float64, which returns 2.0 for every order and hides the problem.

> [!WARNING] A library's split-K choice changes the bits
> Chapter 9 notes that a library picks split-K by shape and reduces in an order of its choice. The same GEMM called with $M = 64$ and $M = 65$ can take different schedules and return results that differ in the last bits. If a test compares two runs bit for bit, fix the schedule, or use the workspace reduction, whose order is fixed.

### When split-K is the right tool

Split-K helps when three things hold together: the grid has fewer tiles than SMs, K is long enough that each split still has several iterations, and the output is small enough that the reduction is cheap. It hurts when the output is large, as the 117-tile example shows, and when the splits leave too few iterations to hide the pipeline's fill. The lab chooser caps splits at 16 and at the number of K-iterations.

## Stream-K

This section answers whether you can balance the iterations exactly, whatever the shape.

Split-K gives you a multiple of the tile count in CTAs, and $T \cdot s$ is rarely a multiple of 108. The 117-tile example with $s = 4$ gives 468 CTAs, which is 4.33 waves, so the last wave is a third full. The deal still follows tile boundaries.

*Stream-K*, from Osama, Merrill, Cecka, Garland, and Owens, drops the tile boundary from the deal. It divides the aggregate inner-loop iterations evenly among the processing elements, and the paper reports that this keeps utilisation high whatever the shape of the output tiles.

### The ranges

Number the $T \cdot I$ MAC-iterations in order, tile 0's first, then tile 1's, and so on. Give CTA $c$ one contiguous range of that order. Let $b = \lfloor TI/g \rfloor$ and $r = TI \bmod g$, for $g$ CTAs. The first $r$ CTAs take $b+1$ iterations and the others take $b$:

```python
def stream_k_plan(tiles, iters_per_tile, ctas):
    total = tiles * iters_per_tile
    base, rem = divmod(total, ctas)
    plan = []
    for c in range(ctas):
        start = c * base + min(c, rem)
        end = start + base + (1 if c < rem else 0)
        segments, i = [], start
        while i < end:
            tile = i // iters_per_tile
            stop = min(end, (tile + 1) * iters_per_tile)
            segments.append((tile, i - tile * iters_per_tile, stop - tile * iters_per_tile))
            i = stop
        plan.append(segments)
    return plan
```

Each CTA's range is cut at tile boundaries into *segments*, each one a tile and a K range local to the tile. The paper writes the range size as $\lceil TI/g \rceil$, which leaves the last CTA short. The lab uses the balanced version, so that the sizes differ by at most one and agree with chapter 4's `stream_k_bounds`. The two have the same critical path.

For the 117-tile example, $g = 108$ and $TI = 7488$. The base is $\lfloor 7488/108 \rfloor = 69$ and the remainder is $7488 - 108 \times 69 = 36$. So 36 CTAs run 70 iterations and 72 run 69. The critical path is 70 iterations against data-parallel's 128, and the utilisation is $7488/(108 \times 70) = 99.0\%$.

The decode shape gives $2048 = 108 \times 18 + 104$: 104 CTAs run 19 iterations and 4 run 18. The critical path is 19 against data-parallel's 64.

### The partial-tile fix-up

A CTA's range starts and ends anywhere. A range that ends inside a tile leaves that tile's remaining iterations to another CTA, and the tile's accumulator is split between the two. This is the *partial-tile fix-up*. In the paper's design, the CTA that holds the tile's first iteration, $k = 0$, is the tile's owner. The other CTAs write their fp32 partials to a workspace and signal a flag. The owner waits on the flags, adds the partials to its own accumulator, and stores the tile once.

The lab's `fixup_tiles` finds the tiles with more than one contributor. The lowest CTA index in each list is the owner, because the iteration order increases with the CTA index. In the 117-tile example each CTA spans 69 or 70 iterations, more than a tile's 64, so almost every tile straddles a boundary: 105 of the 117 tiles need a fix-up. In the decode example, all 32 do.

That count says the plain stream-K fix-up is cheap in bytes and expensive in coordination. The bytes are one fp32 tile written and read per straddling tile. For the decode shape that is $32 \times 2 \times 64 \times 128 \times 4 = 2.1$ MB, which is $1.6\ \mu\text{s}$. The coordination is a flag handoff between SMs, and the owner may wait while a peer finishes. Chapter 4's latency of 330.7 nanoseconds per round trip puts a handoff at a few microseconds at worst, and the model doesn't price it separately.

> [!WARNING] Stream-K needs the contributing CTAs to run at the same time
> An owner that waits on a flag from a CTA that hasn't started can wait forever if the GPU can't fit both. Stream-K therefore launches at most as many CTAs as can be resident at once: $g \le 108$ for one resident CTA per SM. Split-K has no such constraint, because the reduction runs in a separate kernel after every partial finishes.

### Hybrids

The 105 straddling tiles are the cost of balancing everything. The paper's hybrids balance less. *Data-parallel plus one-tile stream-K* runs full waves of whole tiles data-parallel and stream-K balances only the leftover partial wave. For the 117-tile example, 108 tiles run data-parallel in one wave, and the last 9 tiles, $9 \times 64 = 576$ iterations, go to 108 CTAs at 5 or 6 iterations each. The critical path is $64 + 6 = 70$ iterations, the same as pure stream-K, but only 9 tiles need a fix-up instead of 105.

The paper reports the reason for the hybrid is partly cache behaviour. With plain stream-K, CTAs that work on neighbouring tiles start at different K offsets, which scatters their reads of A and B across the L2. The two-tile variant in the paper runs one fewer data-parallel wave and gives each stream-K CTA between one and two tiles of work, so an owner waits for at most one peer. The lab teaches the plain version because its ranges and fix-ups are the part you can check on a CPU.

### The grid size

The paper doesn't fix $g$ at the SM count. It picks $g$ per problem with an analytical model that balances the iterations saved against the fix-up cost, with $g \le 108$ on the A100. The lab's model takes $g = \min(108, TI)$, which is the right choice for the examples here and is a simplification in general. A small problem with few iterations per CTA would pay more in fix-up than it saves, and a model that could pick $g = 64$ or $g = 8$ would see that.

### The three on one picture

The figure shows the work on each of the 108 SMs for the schedules in this chapter, on the two shapes. Choose a shape and a schedule and compare the height of the tallest bar with the dashed line, the even share. The numbers come from `scripts/figures/29-gemm-scheduling.py`, which deals the CTAs of each schedule to SMs in wave order and prices them with the lab's model.

```viz
29-schedule-lanes
```

On the 117-tile shape, data-parallel has 9 tall bars of 128 iterations over 99 bars of 64. Split-K with $s = 2$ has 18 bars of 96. Stream-K has a flat profile of 69 and 70, with a model time of $123.2\ \mu\text{s}$ against $185.8\ \mu\text{s}$, a 1.5 times speedup. On the decode shape, data-parallel has 32 bars and 76 empty SMs, and stream-K has none empty and a model time of $28.8\ \mu\text{s}$ against $91.6\ \mu\text{s}$.

The model is kinder to stream-K than the hardware will be. It prices the fix-up by its bytes and not by the waiting, and it ignores the lost L2 locality that the hybrids exist to repair. The lab measures what is available on the A100 with Triton, which is split-K and the persistent kernel.

## Persistent kernels

This section answers what a persistent kernel is for, now that every schedule so far could run as one CTA per tile.

Picture the launch. A data-parallel kernel launches $T$ CTAs, and the hardware hands each to an SM when one frees up. A *persistent* kernel launches one CTA per SM, $g = 108$, and each CTA loops: fetch a work item, compute it, and fetch the next, until none remain. The CTA doesn't exit between tiles. It keeps its registers, its shared-memory layout, and its barriers.

A work item can be a tile, a (tile, split) pair, or a stream-K range. The loop is the same:

```python
for work in range(start, num_items, NUM_SMS):
    tile_id = work // SPLITS
    split = work % SPLITS
    # grouped map from tile_id to (pid_m, pid_n), then the K loop over the split's range
```

The Triton persistent-matmul tutorial launches a grid of about one program per SM, taken from the device's multiprocessor count, and loops each program over tiles in steps of that count.

### A static or a dynamic counter

There are two ways to hand out items. A *static* counter gives CTA $c$ the items $c$, $c + g$, $c + 2g$, and so on, as in the loop above. Nothing is shared, so it costs nothing, and the deal is the same on every run. A *dynamic* counter keeps one integer in global memory. A CTA that finishes does `atomicAdd` on it, receives the next item id, and so no CTA idles while another has a queue.

The difference shows when items cost different amounts, as with ragged batches or causal masks. Take six tiles with costs $5, 1, 1, 1, 1, 1$ and two CTAs:

| Counter | CTA 0 | CTA 1 | Makespan |
|---|---|---|---|
| Static round-robin | tiles 0, 2, 4: $5+1+1 = 7$ | tiles 1, 3, 5: $1+1+1 = 3$ | 7 |
| Dynamic | tile 0: 5 | tiles 1 to 5: $1+1+1+1+1 = 5$ | 5 |

The dynamic counter finishes at 5 because CTA 1 takes every cheap tile while CTA 0 is busy with the expensive one. The lab's `simulate_tile_counter` and `static_makespan` reproduce the table, with the rule that the lowest-numbered free CTA goes first on a tie. For a GEMM whose tiles all cost the same, the two are identical, so the static counter is the right default. The dynamic one pays for itself when costs vary.

### What persistence buys on a uniform GEMM

On the 117-tile shape a static persistent kernel with 108 CTAs gives 9 CTAs two tiles and 99 one, the same 128 iterations as data-parallel. Persistence by itself doesn't change the critical path, so it doesn't fix the tail. It gives you the following:

- **A launch you pay once.** Setup, such as descriptors and shared-memory layout, runs once per CTA instead of once per tile. With $T = 1024$ tiles and 108 CTAs, each CTA sets up once and runs 9 or 10 tiles.
- **Control of the order.** The CTA chooses the next tile. Mapping each id through grouped ordering keeps the tiles that run together in the same rows and columns of C, which is chapter 14's L2 argument applied to a loop.
- **Overlap of the epilogue with the next mainloop.** CUTLASS's Hopper kernels include a persistent ping-pong design where two consumer warp groups work on different tiles, so that one group's epilogue runs while the other's math does. That needs a persistent CTA that holds both groups' state.
- **A vehicle for the other schedules.** Split-K's (tile, split) items and stream-K's ranges are both work lists that a persistent CTA walks.

### Grouped order in the loop

The lab's `persistent_tile_order` combines the static counter with grouped ordering. CTA $c$ takes tile ids $c, c + g, \ldots$ and maps each id with the arithmetic of chapter 14, with one fix. Take the position inside the strip first, $w = \text{id} \bmod (G_M \cdot N_t)$, where $G_M$ is `GROUP_M` and $N_t$ is the number of tile columns, then the row is $w \bmod h$ and the column is $\lfloor w / h \rfloor$, where $h$ is the strip's height. The Triton matmul tutorial writes the row as `(pid % num_pid_in_group) % group_size_m`. Chapter 14's kernel writes `pid % group_size_m` from the global id. The two agree when the strip's height divides the number of tiles before the strip. Otherwise, on a short last strip, they number the rows in a different order, and both still visit every tile once. The lab follows the tutorial, and its test enumerates the strips with nested loops to check the map.

### Work items of tile and split

The lab's persistent kernel combines persistence with split-K. A work item is a (tile, split) pair, so a grid of 32 tiles with $s = 3$ is 96 items on 96 of the 108 CTAs. Consecutive items belong to the same tile, so the splits of a tile run at the same time. When $s > 1$ each item adds its fp32 partial with `tl.atomic_add`. That makes this kernel the atomic variant of split-K inside a persistent loop, and its output can differ from run to run in the last bits.

## Epilogue fusion

This section answers what happens to the accumulator after the K loop ends, and why where it goes matters.

Picture the end of the kernel. The accumulator is an fp32 tile in registers. A plain GEMM casts it to bf16 and stores it. A transformer's linear layer wants more: add a bias, apply an activation, add the residual stream, or scale the result. Done as separate PyTorch operations, each one is a kernel that reads the whole $M \times N$ output from HBM and writes it back. Done in the *epilogue*, they run on the accumulator while it is still in registers, and the output is written once.

### The bytes

Count only the C-sized traffic. Let $E = 2MN$ bytes be the bf16 output. A fused epilogue writes C once, which is $E$ bytes, and reads its operands: a bias row of $2N$ bytes, and a residual of $E$ bytes when it has one. An unfused epilogue writes C from the GEMM, which is $E$, and then runs one kernel per stage. Each stage reads $E$ and writes $E$:

$$
\boxed{B_{\text{fused}} = E + \text{operands}, \qquad B_{\text{unfused}} = E + \hla{n_{\text{st}}} \cdot 2E + \text{operands}}
$$

Here $\hla{n_{\text{st}}}$ is the number of separate stages. The saving is $2 n_{\text{st}} E$, independent of the operands, because both versions read the same bias and residual.

The lab defines six epilogues. A stage count of 1 is bias alone. The other five have two stages: bias then activation, bias then residual, and scale then residual. The following table gives the traffic for a $4096 \times 4096$ bf16 output, with $E = 33{,}554{,}432$ bytes:

| Epilogue | Operands | Fused bytes | Unfused bytes | Saved |
|---|---|---|---|---|
| `bias` | bias row | 33.6 MB | 100.7 MB | 67.1 MB |
| `bias_relu`, `bias_gelu`, `bias_silu` | bias row | 33.6 MB | 167.8 MB | 134.2 MB |
| `bias_residual` | bias row, residual | 67.1 MB | 201.3 MB | 134.2 MB |
| `scale_residual` | residual | 67.1 MB | 201.3 MB | 134.2 MB |

For the GELU case, fusing saves 134.2 MB, which is $105\ \mu\text{s}$ at 1275 GB/s. The 4096 cube's arithmetic floor at 312 TFLOP/s is $137.4 \times 10^9 / 312 \times 10^{12} = 440\ \mu\text{s}$, so the unfused epilogue adds about 24% to the compute floor. Each of its stages also pays a launch of about 5 microseconds, which is negligible here.

The other end of the shape range reverses which term matters. For the decode shape the output is $64 \times 4096 \times 2 = 524$ KB, and the bytes are small. But two extra launches cost $2 \times 5 = 10\ \mu\text{s}$, which is more than a third of the stream-K GEMM's $28.8\ \mu\text{s}$. For small outputs, fusion saves launches. For large ones, it saves bytes.

### The six epilogues

Every epilogue is a function of the fp32 accumulator, with at most a bias row and a residual tile as operands:

| Kind | Formula |
|---|---|
| `bias` | $x = \text{acc} + b$ |
| `bias_relu` | $\max(x, 0)$ |
| `bias_gelu` | $\text{gelu}_{\tanh}(x)$ |
| `bias_silu` | $x \cdot \sigma(x)$ |
| `bias_residual` | $x + r$ |
| `scale_residual` | $\alpha \cdot \text{acc} + \beta \cdot r$ |

The last is the form BLAS gives a GEMM: $\alpha AB + \beta C$, with $C$ the residual. Two points make these cheap to write in Triton. First, apply the epilogue to the fp32 accumulator and cast once at the store. Casting first rounds the product to bf16 and then rounds the result of the activation again, which loses bits that the fused kernel can keep. Second, the tanh form of GELU is $0.5\,x\,(1 + \tanh u)$ with $u = \sqrt{2/\pi}\,(x + 0.044715\,x^3)$, and $0.5\,(1 + \tanh u) = \sigma(2u)$, so the kernel needs only `tl.sigmoid`:

```python
u = 0.7978845608028654 * (x + 0.044715 * x * x * x)
out = x * tl.sigmoid(2.0 * u)
```

[Chapter 24](/c/24-cute-dsl-gemm) fuses a bias and the erf form of GELU in the CuTe DSL, and the lab here uses the tanh form so that the CPU reference is `torch.nn.functional.gelu(x, approximate="tanh")`. The structure is the one chapter 14's kernel already has: its `HAS_BIAS` and `ACTIVATION` branches are a two-epilogue version of this one. The lab's kernel extends them with a residual load and the scale form.

### Where fusion stops

An epilogue is elementwise, so it can't hold a reduction across the tile's columns. A softmax or a layer norm needs the whole row, which a tile doesn't have. That boundary is where [chapter 17](/c/17-fusion-and-compilers) and the normalization kernels of chapter 15 pick up. A split-K or stream-K kernel adds a second constraint: the epilogue must run after the reduction, on the sum of the partials. In the workspace variant the reduction kernel is the place for it. In stream-K the tile's owner applies it after adding the partials.

## Picking a schedule by shape

This section answers the practical question: given M, N, and K, which schedule do you launch?

The lab's `choose_schedule` scores each candidate with the cost model and applies one rule: a candidate other than data-parallel wins only if it beats data-parallel by at least 5%. The margin is a modelling choice, not a hardware number. A prediction within a few percent of the baseline is inside the model's error, and a simpler schedule is easier to trust. The following table shows the model's times and picks. The tile is $64 \times 128 \times 64$ for the first two rows, $256 \times 128 \times 64$ for the two 2304-row shapes, and $128 \times 128 \times 64$ for the others:

| Shape (M x N x K) | Tiles | Wave efficiency | Data-parallel | Best split-K | Stream-K | Pick, with stream-K | Pick, without |
|---|---|---|---|---|---|---|---|
| 16 x 4096 x 4096 | 32 | 29.6% | 89.5 us | 33.2 us ($s = 4$) | 28.2 us | stream-K | split-K, 4 |
| 64 x 4096 x 4096 | 32 | 29.6% | 91.6 us | 38.7 us ($s = 4$) | 28.8 us | stream-K | split-K, 4 |
| 128 x 4096 x 4096 | 32 | 29.6% | 94.4 us | 46.1 us ($s = 4$) | 31.3 us | stream-K | split-K, 4 |
| 256 x 4096 x 4096 | 64 | 59.3% | 50.0 us | 56.7 us ($s = 3$) | 36.2 us | stream-K | data-parallel |
| 1024 x 1024 x 8192 | 64 | 59.3% | 92.9 us | 87.2 us ($s = 3$) | 61.8 us | stream-K | split-K, 3 |
| 2304 x 1536 x 4096 | 108 | 100% | 92.9 us | 142.3 us ($s = 2$) | 92.9 us | data-parallel | data-parallel |
| 2304 x 1544 x 4096 | 117 | 54.2% | 185.8 us | 189.0 us ($s = 2$) | 123.2 us | stream-K | data-parallel |
| 4096 x 4096 x 4096 | 1024 | 94.8% | 464.6 us | 656.9 us ($s = 2$) | 451.5 us | data-parallel | data-parallel |
| 8192 x 8192 x 8192 | 4096 | 99.8% | 3531.0 us | 4378.1 us ($s = 2$) | 3535.4 us | data-parallel | data-parallel |

All times are the lab model's predictions on the A100's ratings and the 1275 GB/s ceiling, not measurements. Read the table as five regimes:

- **Too few tiles, M up to 128.** 32 tiles on 108 SMs. Both split-K and stream-K win by 2 to 3 times in the model. Split-K needs no coordination between CTAs, and stream-K needs cross-CTA flags.
- **A mid-size grid, M = 256.** At 64 tiles the model says split-K *loses*: the workspace traffic costs more than the iterations it saves. Only stream-K, with its small fix-up, wins.
- **A grid just over a multiple of 108.** The 117-tile shape is the quantization case. Stream-K recovers about a third of the time, and split-K doesn't.
- **A grid on a multiple of 108.** At 108 tiles every schedule ties or loses to data-parallel, which is already a full wave.
- **A big grid.** At 1024 tiles the model's stream-K gain is 2.8%, under the 5% margin. At 4096 tiles it is nothing.

Two cautions go with the table. The model says nothing about L2 behaviour, and chapter 14 shows grouped order moving the L2's work by tens of percent in some cases and a few percent in others. And the model's decode rows are predictions for an A100 whose copy ceiling is measured, and the rows assume the kernel gets 1275 GB/s on a skinny GEMM. The lab measures what Triton's split-K reaches, and the measured best-of-N is the check on the chooser.

The shapes with M at or below 128 are where inference lives, and the companion course, [learn-inference](https://qwen.ifkash.dev), owns that kernel in depth, including the weight-only GEMV. The scheduling idea here is the one it builds on.

## What goes wrong

- **Dropping the K remainder.** Giving every split `k_iters // splits` iterations covers 15 of 16 iterations when 16 iterations go to 5 splits. The result is wrong by one K-step, and small enough in a random test to look like noise.
- **Reducing partials in the output dtype.** A partial rounded to fp16 loses about $2^{-11}$ of its value before the sum. Keep partials in fp32 and cast once at the end. On the lab's fp16 test, the rounded version is off by about 3e-2, against under 1e-3 for the fp32 reduction.
- **Trusting the atomic variant to reproduce.** `tl.atomic_add` adds partials in an arrival order that changes between runs. The lab's workspace kernel is bit-identical across runs, and the persistent kernel with $s > 1$ is not.
- **Counting waves with the flat area.** A $130 \times 130$ output on $128 \times 128$ tiles needs 4 tiles. Dividing the areas gives 2, because each dimension rounds up on its own.
- **Splitting past the work.** With more splits than K-iterations, trailing splits are empty and still launch CTAs. The lab chooser stops at the iteration count.
- **A stream-K segment that crosses a tile.** A segment is one tile's K range. A CTA range that spans two tiles is two segments, and merging them reads the wrong rows of A and B.
- **Waiting on a CTA that is not resident.** Stream-K's owner spins on a flag. A grid larger than the resident capacity can deadlock it.
- **Applying the epilogue to a partial.** With split-K, an activation applied to each partial before the sum computes $f(a) + f(b)$ and not $f(a+b)$. Apply it once, after the reduction.
- **Fusing where the shape doesn't pay.** The 134 MB saving is a large-output number. At $M = 64$ the bytes are small and the saving is launches. A fusion that costs registers and lowers occupancy needs a measurement.

> [!RECAP]
> - The unit of work is the MAC-iteration. A schedule deals iterations to SMs, and the kernel's time is the busiest SM's count. For the 117-tile shape the even share is 69.3 iterations and data-parallel takes 128.
> - Split-K multiplies the grid by $s$ and pays $5\ \mu\text{s} + 2sMN \cdot 4 / \beta$ for the reduction. It wins on small outputs with long K and loses on the 117-tile shape. A fixed-order reduction is reproducible, and fp32 atomics are not.
> - Stream-K deals fixed-size ranges: 36 CTAs of 70 iterations and 72 of 69 on the 117-tile shape. Tiles that straddle a range boundary need a fix-up, 105 of 117 here, and a hybrid cuts that to 9.
> - A persistent kernel launches one CTA per SM and loops over work items. A static counter is free and a dynamic one balances uneven items, 5 against 7 in the lab's example. It doesn't fix the tail by itself.
> - A fused epilogue saves $2 n_{\text{st}} E$ bytes, 134.2 MB for a two-stage epilogue on a 4096 by 4096 bf16 output. Apply it to the fp32 accumulator, after any reduction, and cast once.
> - The lab's model picks stream-K for $64 \times 4096 \times 4096$ and data-parallel for $4096^3$. It is a prediction, and the lab measures how far the Triton kernels are from it.

## Check your understanding

> [!QUESTION] How many tiles, waves, and what efficiency for $M = 2304$, $N = 1544$ on $256 \times 128$ tiles with one CTA per SM?
> $9 \times 13 = 117$ tiles. The first wave takes 108 and the second takes 9, so 2 waves at $117/216 = 54.2\%$. The same grid is $117 \times 64 = 7488$ MAC-iterations at $K = 4096$ and $B_K = 64$, and the even share is 69.3 per SM.

> [!QUESTION] You have 16 K-iterations and 5 splits. Which ranges does `split_k_ranges` return, and what goes wrong if you use `16 // 5` for every split?
> $[0,4)$, $[4,7)$, $[7,10)$, $[10,13)$, $[13,16)$. Using 3 for every split covers 15 iterations and drops the 16th, so the output is missing one K-step of every tile.

> [!QUESTION] Why does the order of fp32 partials change the sum? Give a case.
> fp32 addition rounds after every add, so grouping matters. The fp32 spacing at $10^8$ is 8, so $10^8 + 1 = 10^8$. For partials $10^8, 1, -10^8, 1$, the order 0, 1, 2, 3 gives 1.0, the order 0, 2, 1, 3 gives 2.0, and the order 1, 3, 0, 2 gives 0.0. The atomic variant's order is the arrival order.

> [!QUESTION] 7488 MAC-iterations go to 108 CTAs. How many iterations does each CTA run, and how many tiles need a fix-up if a tile has 64 iterations?
> $7488 = 108 \times 69 + 36$, so 36 CTAs run 70 and 72 run 69. Each CTA's range exceeds a tile's 64 iterations, so nearly every tile straddles a boundary: 105 of 117 need a fix-up. A hybrid that runs 108 tiles data-parallel and stream-Ks the last 9 has 9 fix-ups and the same 70-iteration critical path.

> [!QUESTION] A persistent kernel launches 108 CTAs for 117 equal tiles. How long does it take, in iterations, and what did persistence change?
> Nine CTAs run two tiles, so the critical path is $2 \times 64 = 128$ iterations, the same as data-parallel. Persistence changes the setup cost, the tile order, and what you can overlap, but not the tail. Splitting the work finer, with split-K or stream-K, changes the tail.

> [!QUESTION] Costs $5, 1, 1, 1, 1, 1$ on two CTAs: what are the static and dynamic makespans?
> Static round-robin gives CTA 0 the tiles 0, 2, 4, so $5+1+1 = 7$, and CTA 1 gets 3, for a makespan of 7. A dynamic counter lets CTA 1 take tiles 1 to 5 while CTA 0 runs tile 0, for a makespan of 5.

> [!QUESTION] How many bytes does fusing a bias and a GELU save on a $4096 \times 4096$ bf16 output, and why does it matter less for $M = 64$?
> The unfused version writes C, then two stages each read and write $33.6$ MB, for 167.8 MB against the fused 33.6 MB. It saves 134.2 MB, which is $105\ \mu\text{s}$ at 1275 GB/s. At $M = 64$ the output is 524 KB, so bytes are small. The two saved launches, about $10\ \mu\text{s}$, are the larger term.

> [!QUESTION] Why does the model say split-K loses at $2304 \times 1544$ even though it cuts the critical path from 128 to 96 iterations?
> The workspace for two splits is $2 \times 2 \times 2304 \times 1544 \times 4 = 56.9$ MB of traffic, $44.6\ \mu\text{s}$ at 1275 GB/s, plus a $5\ \mu\text{s}$ launch. That cancels the $46.4\ \mu\text{s}$ the shorter critical path saves. The model may overstate the cost, since the 28.5 MB workspace can stay in the 40 MiB L2.

## Lab

> [!TRY] Schedule a GEMM on the CPU, then on the GPU
> Open `29-gemm-scheduling`. `wave_stats` is the one function the first check needs. Then write `split_k_ranges` and `split_k_matmul`, `reduce_partials`, `stream_k_plan` and `fixup_tiles`, `persistent_tile_order` with `simulate_tile_counter` and `static_makespan`, `apply_epilogue` and `epilogue_bytes`, and `schedule_cost_us` with `choose_schedule`. Every one of those runs on the CPU. On the A100, write `splitk_matmul`, `persistent_matmul`, and `fused_matmul` in Triton. They are checked against `torch.matmul` for $M$ from 16 to 128 and for six epilogues. A machine without a GPU skips those checks with a message and reports the metrics as skipped.

The lab reports four metrics when a GPU is present:

- `plain_ms_m64`: the time of a lab-14-style plain kernel at $64 \times 4096 \times 4096$, with the L2 flushed before each call.
- `splitk_speedup_m64` and `splitk_best_splits`: the plain time over the best split-K time, and the split count that won.
- `persistent_speedup_m64`: the plain time over the best persistent time.

The check on those metrics is loose: the best schedule must not be more than about 1.4 times slower than the plain kernel. The model predicts a speedup of 2 to 3 times at this shape, and the lab measures the real figure.

## Further reading

- Osama, Merrill, Cecka, Garland, and Owens, [Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU](https://arxiv.org/abs/2301.03598): the iteration-range partition, the partial-tile fix-up with a global workspace and flags, the hybrid schedules, and the A100 experiments.
- [Triton persistent matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html): a grid sized to the SM count, the tile loop with grouped ordering, epilogue subtiling, TMA, and warp specialization.
- [Triton matrix multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html): the grouped-ordering arithmetic, including the `(pid % num_pid_in_group) % group_size_m` form the lab uses.
- [CUTLASS efficient GEMM](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md): split-K as a partitioned-K GEMM and a batched reduction with a caller-provided workspace, sliced-K, and the persistent cooperative and ping-pong kernels.
- [CUTLASS GEMM API 3.x](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/gemm_api_3x.md): the tile scheduler parameter of a kernel, and the collective epilogue.
- [Matrix multiplication background](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html), NVIDIA Deep Learning Performance guide: tile quantization, wave quantization, and the 108-tile example.
- [Chapter 4](/c/04-roofline-and-latency), for the wave and stream-K counts this chapter builds on, [chapter 14](/c/14-triton), for the tile kernel and grouped order that the lab extends, and [chapter 24](/c/24-cute-dsl-gemm), for a fused epilogue in the CuTe DSL.
