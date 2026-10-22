---
title: The roofline and latency hiding
slug: 04-roofline-and-latency
part: "Part 1 — The machine"
summary: Predict how long any kernel takes from four floors, arithmetic, bandwidth, latency, and launch cost, and learn what makes a real kernel miss its prediction.
minutes: 120
gpu: true
objectives:
  - "Count the FLOPs and the compulsory bytes of any kernel and compute its arithmetic intensity."
  - "Draw the A100 roofline for each precision and place a kernel on it, using the rated and the measured bandwidth."
  - "Explain what a hierarchical roofline adds, and when the DRAM roofline alone misleads you."
  - "Apply Little's law to compute the bytes in flight that saturate memory, and translate it into warps per SM and loads per thread."
  - "Choose between thread-level and instruction-level parallelism to hide latency."
  - "Compute the number of waves and the wave efficiency of a grid, including a GEMM tile grid."
  - "Treat kernel launch cost as a floor, and combine all four floors into a predicted time."
lab: 04-roofline
---

# The roofline and latency hiding

> [!TLDR]
> - A kernel can't finish before its arithmetic is done, before its bytes arrive, before one memory round trip
>   completes, or before the launch itself returns. Four floors, one prediction.
> - The roofline handles the first two. Divide FLOPs by bytes to get the arithmetic intensity, then compare it with
>   the ridge point: 15.3 FLOP/byte for fp32 on the A100's CUDA cores against the measured 1275 GB/s, and 244.7 for
>   bf16 tensor cores.
> - Little's law handles the third: to sustain 1275 GB/s with a 330.7 ns memory latency, about 3.9 KB must be in
>   flight on each of the 108 SMs. Threads and independent loads both supply bytes in flight.
> - A grid runs in whole waves. 109 blocks on 108 SMs take two block-times, not one, and the same arithmetic explains
>   why a GEMM that grows by one tile row can run at half speed.
> - Launching a kernel costs about 5 microseconds on the lab card, so anything that moves under about 6.4 MB is
>   launch bound, not bandwidth bound.

Chapter 0 gave you two rates and the ratio between them. This chapter turns that ratio into a method you can run
on any kernel before you write it: count what it must do, divide by what the machine can do, and name the floor
that binds. When the measured time lands near the prediction, you're done optimizing. When it lands far above, the
gap tells you which assumption was wrong.

The method matters more than any single result, because the kernels in a model are mostly small and varied. A
LayerNorm, a rotary embedding, a gather, a softmax, and a GEMM with an awkward shape each hit a different floor.
The roofline model from [the companion course](https://qwen.ifkash.dev/c/10-roofline) places the operations of one
transformer on a plot. This chapter goes further and more generally: it works for any kernel, and it adds the
three things that plot leaves out, which are latency, quantization into waves, and launch cost.

By the end, you can predict a kernel's time to within a small factor before you run it. The lab makes you do it:
you write the predictors, then measure a family of kernels and read off the ridge point from your own data.

## Before you start

- **Arithmetic rate and bandwidth.** *Peak arithmetic rate* is FLOP per second, and *bandwidth* is bytes per
  second. [Chapter 0](/c/00-what-you-are-building) derived the ridge point from them, and [chapter 1](/c/01-the-gpu-at-a-glance)
  derived each peak from the SM count and the clock.
- **Latency versus bandwidth.** *Latency* is how long one request takes from issue to answer. Bandwidth is how many
  bytes per second the whole system moves. A pipe can be long, which is latency, and wide, which is bandwidth, and
  the two are independent.
- **Warps, blocks, and SMs from [chapter 2](/c/02-the-execution-model).** A block runs on one SM, an SM holds up to
  2048 resident threads (64 warps), and a warp issues one instruction for 32 threads together.
- **The memory path from [chapter 3](/c/03-memory-and-bandwidth).** A warp's coalesced load touches 128 contiguous
  bytes, which the memory system serves as 32-byte sectors.

## FLOP and byte counts

This section answers the first question of any prediction: how much work does the kernel do, and how much data must
it move? Every number later in the chapter depends on getting these two counts right.

Count FLOPs by operation, with one fused multiply-add (FMA) counted as two FLOPs, the convention behind the A100's
19.5 TFLOP/s fp32 rating. Count bytes as the *compulsory traffic*: every distinct byte the kernel must read from
DRAM once, plus every byte it must write once. Divide the first by the second:

$$
\hlc{I} = \frac{\hla{F}}{\hlb{B}}
$$

Here $\hla{F}$ is the FLOP count, $\hlb{B}$ is the compulsory DRAM traffic in bytes, and $\hlc{I}$ is the
*arithmetic intensity* in FLOP per byte. The compulsory count is a lower bound. A kernel that reads an input twice
from DRAM moves more, and the textbook count then understates its time.

The following table counts five kernels, all arithmetic you can check by hand:

| Kernel | FLOPs per element | Bytes per element | Intensity (FLOP/byte) |
|---|---|---|---|
| Copy, fp32 | 0 | 8 (read 4, write 4) | 0 |
| Vector add, fp32 | 1 | 12 (read 8, write 4) | 0.083 |
| `y = a*x + y`, fp32 | 2 | 12 | 0.167 |
| Sum reduction, fp32 | 1 | 4 (read only) | 0.25 |
| `k` FMAs per element, fp32 (the lab kernel) | $2k$ | 8 | $k/4$ |

The last row is the one the lab uses, because $k$ is a dial: turn it up and the kernel climbs the roofline
without changing its memory traffic.

A matrix multiplication differs because its data is reused. An $M \times K$ by $K \times N$ product does $2MNK$
FLOPs, and its compulsory traffic is the two inputs and the output, each read or written once. For a square
$n \times n$ product with elements of $b$ bytes, the intensity is:

$$
I = \frac{2n^{3}}{3 n^{2} b} = \frac{2n}{3b}
$$

The intensity grows with $n$, so a large GEMM is compute bound and a thin one isn't. For fp16 or bf16 ($b=2$),
that's $n/3$:

> [!EXAMPLE] Two GEMMs from NVIDIA's guide, recomputed
> A $4096^3$ bf16 GEMM does $2 \times 4096^3 = 137.4$ GFLOP and moves $3 \times 4096^2 \times 2 = 100.7$ MB, an
> intensity of $4096/3 = 1365$ FLOP/byte. An $8192 \times 128 \times 8192$ GEMM does
> $2 \times 8192 \times 128 \times 8192 = 17.18$ GFLOP and moves
> $2 \times (8192 \times 128 + 128 \times 8192 + 8192^2) = 138.4$ MB, an intensity of 124.1 FLOP/byte. NVIDIA's
> [matrix multiplication guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
> works out the same 124.1 and 2,730 (for $8192^3$) and compares them with a ridge point. The thin GEMM falls below
> the A100's bf16 ridge point of 161 FLOP/byte, so it's memory bound despite doing 17 GFLOP.

Two habits make these counts reliable. First, write the count before the code, because a count made after you see
the profiler tends to match it. Second, count the *dtype you actually move*: a bf16 tensor is 2 bytes per element
even if you accumulate in fp32 registers, because registers don't touch DRAM.

## The roofline

This section answers the second question: given $F$ and $B$, what is the fastest the kernel can run? The answer is
the larger of two times, and the picture that follows from it is the roofline.

The kernel needs $F / \hla{C}$ seconds to do its arithmetic at peak rate $\hla{C}$, and $B / \hlb{\beta}$ seconds to
move its bytes at bandwidth $\hlb{\beta}$. Hardware overlaps the two but can't beat either, so:

$$
\boxed{t \;\ge\; \max\!\left(\frac{\hla{F}}{\hla{C}},\; \frac{\hlb{B}}{\hlb{\beta}}\right)}
$$

Dividing $F$ by that time gives the throughput a kernel can attain, which depends on its intensity $\hlc{I}$ alone:

$$
P = \min\!\left(\hla{C},\; \hlb{\beta}\,\hlc{I}\right)
$$

Plotted on log axes, $P$ against $I$ is the roofline (Williams, Waterman, and Patterson, 2009): a slanted roof at
low intensity where throughput is bandwidth times intensity, and a flat roof at $\hla{C}$ where arithmetic is the
limit. They meet at the ridge point $I^{\ast} = \hla{C} / \hlb{\beta}$.

```viz
04-roofline
```

The figure starts on fp32 with the measured bandwidth and an intensity of 16 FLOP/byte, which is the lab kernel at
$k = 64$ and sits just right of its ridge point. Drag the slider left and the attainable throughput falls along the
slanted roof. Switch to bf16 and the same kernel is far to the left of the ridge, so it's memory bound.

### A ridge point for every precision

The A100 has one memory system and several arithmetic peaks, so it has several ridge points. The following table
divides each dense peak by the rated and by the measured bandwidth:

| Arithmetic peak | $C$ (TFLOP/s) | Ridge against 1935 GB/s | Ridge against 1275 GB/s |
|---|---|---|---|
| bf16 or fp16 tensor cores | 312 | 161.2 | 244.7 |
| TF32 tensor cores | 156 | 80.6 | 122.4 |
| INT8 tensor cores (TOPS) | 624 | 322.5 | 489.4 |
| fp32 CUDA cores | 19.5 | 10.1 | 15.3 |

Three consequences follow. First, an fp32 kernel on the CUDA cores becomes compute bound at only 15.3 FLOP/byte,
so the fp32 vector add, at 0.083, is 184 times short of it. Second, moving a GEMM to tensor cores moves its
ridge point right by 16 times (312 over 19.5), so a kernel that was compute bound in fp32 can become memory bound
in bf16 without changing a line of its memory access. Third, the hardware generations push the ridge right: the
H100 SXM's bf16 ridge is 295.3 FLOP/byte and the B200's is 281.3, from the figures in chapter 1.

> [!KEY] The binding side depends on the precision you chose
> The same bytes and the same FLOPs sit on different sides of the ridge for different arithmetic peaks. Always say
> which peak you divide by: a kernel doing 20 FLOP per byte is compute bound on the CUDA cores and memory bound on
> the tensor cores.

### The roofline as a time floor

The roofline in time form is the number you use most. Take a float32 vector add on $2^{24}$ elements, the
learn-inference lab's size. It does $2^{24}$ FLOPs and moves $12 \times 2^{24} = 201.3$ MB:

- **Arithmetic.** $2^{24} / 19.5 \times 10^{12} = 0.86\ \mu\text{s}$.
- **Memory at the measured ceiling.** $201.3 \times 10^{6} / 1275 \times 10^{9} = 157.9\ \mu\text{s}$.
- **Memory at the rating.** $201.3 \times 10^{6} / 1935 \times 10^{9} = 104.0\ \mu\text{s}$.

The floor is 157.9 microseconds against the copy ceiling. The learn-inference chapter 12 hand-written vector add
measured 1304 GB/s on this card, which is 154.4 microseconds: within 3% of that floor, and slightly under it,
because the copy ceiling is a good target and not a hard limit. The rated floor of 104.0 microseconds is the hard
one. A kernel at 154 microseconds is done: it runs at the memory system's speed, and no rewrite helps.

Now the same exercise on the $4096^3$ bf16 GEMM, 137.4 GFLOP over 100.7 MB:

| Floor | Arithmetic | Memory (1275 GB/s) |
|---|---|---|
| bf16 tensor cores, 312 TFLOP/s | 0.441 ms | 0.079 ms |
| TF32 tensor cores, 156 TFLOP/s | 0.881 ms | 0.079 ms |
| fp32 CUDA cores, 19.5 TFLOP/s | 7.05 ms | 0.079 ms |

Arithmetic binds in every row, so the prediction is 0.441 ms at best in bf16. The roofline says no more than that:
a good GEMM lands somewhere above 0.441 ms, and [chapters 10 to 14](/c/10-gemm-from-scratch) are about how far
above.

## Which ceiling to believe

The roofline is only as good as its two numbers, and neither is a number you can use as given. This section
answers: which rates do you divide by, and what does a single DRAM roofline hide?

### Rated peaks are not attainable

A rating assumes the best clock, the best instruction mix, and no power limit. On bandwidth, the A100 80GB PCIe is
rated at 1935 GB/s, but a plain device-to-device copy measured 1275 GB/s in learn-inference chapter 10, which is
66% of the rating. No source this course relies on explains the gap, so don't attribute it to a specific cause.
Use 1275 GB/s as the ceiling for a coalesced, memory-bound kernel. Your lab in
[chapter 3](/c/03-memory-and-bandwidth) measures it again.

On arithmetic, the 312 TFLOP/s rating of the tensor cores is the product $108 \times 2048 \times 1.41 \times 10^{9}$,
which assumes the 1410 MHz boost clock. The A100 PCIe card has a 300 W limit against 400 W on the SXM module, and
NVIDIA doesn't quantify the clock it sustains under a heavy tensor load. A measured `mma.sync` rate of 310.6 TFLOP/s
on an A100 PCIe 40GB (Luo et al.) shows the instruction alone can get close to the rating, but a whole kernel carries
other work, so plan for less.

So build two rooflines and keep both. The rated one is a hard floor that no kernel may beat: a measured time below it
means your byte count, FLOP count, or timer is wrong. The measured one is the target a good kernel approaches.

### One roof per level of the memory hierarchy

A DRAM roofline assumes the bytes you count are the bytes that hit DRAM. Reuse inside the chip breaks that
assumption, and the *hierarchical roofline* repairs it: draw one bandwidth roof for each memory level, and count
the bytes that cross each level.

Figures for the A100 differ by level and by source, so each carries its provenance:

| Level | Bandwidth | Source and status |
|---|---|---|
| HBM2e | 1935 GB/s rated, 1275 GB/s measured copy | Datasheet; learn-inference chapter 10 |
| L2 cache | about 2.8 TB/s measured on an A100 PCIe 40GB, 7.2 TB/s theoretical | Luo et al.; GA100 whitepaper |
| Shared memory | 19.5 TB/s theoretical across 108 SMs | $128\ \text{B/clock} \times 108 \times 1.41\ \text{GHz}$ |

The L2 figures disagree by a factor of 2.6 between the paper's measurement and the vendor's theory, and a
third source (Chips and Cheese) measured 4.5 TB/s on an SXM A100. When you quote an L2 bandwidth, cite the method.

Each level has its own ridge point. Against the measured 2.8 TB/s of L2, the fp32 ridge is $19.5 / 2.8 = 7.0$
FLOP/byte and the bf16 ridge is $312 / 2.8 = 111$. A kernel can sit left of the DRAM ridge and right of the L2 ridge:
its DRAM traffic is small, because it reuses data in cache, so the L2 isn't what binds it either. The hierarchical
roofline plots one dot per level for the same kernel, with a different intensity for each, since each level sees a
different number of bytes. Yang et al. (2020) describe the method for deep-learning kernels, and Nsight Compute draws
it for you ([chapter 5](/c/05-profiling)).

The fusion sweep from learn-inference chapter 13 is a measured case of the hierarchy mattering. On the lab A100, a
fused RMSNorm over a 41.9 MB tensor ran at 0.92 to 0.95 times the speed of the unfused version, although the byte
count predicts a 1.25 times win. The crossover came near the 40 MiB L2. That's consistent with the unfused
version's intermediate staying in cache below that size, so the extra DRAM traffic that the byte count charges it
mostly doesn't happen.

> [!NOTE] Counting bytes at the right level
> To use the DRAM roofline, count DRAM bytes: what a profiler reports as DRAM read and write bytes, not the bytes your
> code requests. For a kernel whose working set fits in the 40 MiB L2, the requested bytes and the DRAM bytes differ,
> and the DRAM roofline overpredicts its time.


## Latency and Little's law

This section answers a question the roofline can't: why does a kernel with plenty of bytes to move still run below the
bandwidth ceiling? The usual answer is that it doesn't keep enough bytes in flight.

Memory doesn't answer instantly. Every load spends a fixed time travelling from the SM through the L2 to HBM and back.
On an A100 PCIe 40GB, Luo et al. measured a global-memory load latency of 466.3 cycles, which at 1410 MHz is
$466.3 / 1.41 = 330.7$ ns. The same paper measured 261.5 cycles (185.5 ns) for an L2 hit, 37.9 cycles for an L1 hit, and
29.0 cycles for shared memory. The paper's card is the 40 GB part, and no 80 GB measurement exists in the sources this
course uses, so treat 330.7 ns as the right order of magnitude for the lab card.

Bandwidth is a rate, and latency is a delay, so a memory system sustains full bandwidth only when a new request
leaves every time an old one returns. Little's law (1961) makes that exact. In any steady system, the average number of
items in the system is the arrival rate times the time each one stays:

$$
\boxed{\hla{L} = \hlb{\lambda}\,\hlc{W}}
$$

For memory, $\hlb{\lambda}$ is the bandwidth $\beta$ in bytes per second, $\hlc{W}$ is the latency, and $\hla{L}$ is
the bytes in flight: requested but not yet returned. To sustain a bandwidth, you must keep that many bytes in flight.

> [!EXAMPLE] Bytes in flight on the A100
> At the rated 1935 GB/s and 330.7 ns:
> $L = 1935 \times 10^{9} \times 330.7 \times 10^{-9} = 639.9$ KB, which is 5.9 KB on each of the 108 SMs.
> At the measured 1275 GB/s ceiling: $1275 \times 10^{9} \times 330.7 \times 10^{-9} = 421.7$ KB, or 3.9 KB per SM.
> Your lab computes both and measures where a read kernel really saturates.

These are lower bounds. The 330.7 ns is an idle-memory latency, and latency under load rises as the queues fill, so a
real kernel needs more than the formula says. Whether a given access pattern can sustain that many outstanding requests
also depends on queue limits inside the memory system that NVIDIA doesn't publish. Treat the formula as the
requirement you must meet, not a promise that meeting it is enough. The lab measures how far above it the real
saturation point sits.

> [!DEEPDIVE] Why Little's law holds, in one paragraph
> Picture a pipe that holds requests. Requests enter at rate $\lambda$, and each stays for time $W$. Over a long time $T$,
> $\lambda T$ requests enter and the same number leave. Each spent $W$ in the pipe, so the total time spent inside is
> $\lambda T W$. Divide by $T$ to get the average occupancy, $\lambda W$. The argument needs no assumption about the
> distribution of latencies or the order of service, which is why the law applies to a memory system whose internals you
> can't see. It needs a steady state, so a kernel that runs for under a few microseconds, shorter than it takes to fill
> the pipe, never reaches it.

### From bytes in flight to warps and loads

Bytes in flight are supplied by threads that have issued a load and not yet received its data. So the 5.9 KB per SM
turns into a requirement on how many threads, and how many loads per thread, each SM keeps waiting.

A warp's coalesced 4-byte load is 128 bytes. A warp's `float4` load, 16 bytes per thread, is 512 bytes. The
requirement then reads as follows, using the rated 5.9 KB (5,925 bytes) per SM:

- With 4-byte loads, 46.3 warp-loads in flight per SM.
- With 16-byte loads, 11.6 warp-loads in flight per SM.
- At the full 64 warps per SM, each with one 4-byte load outstanding, an SM holds $64 \times 128 = 8192$ bytes in
  flight, which exceeds 5,925.

Divide the per-SM requirement by the number of resident threads and you get the bytes each thread must keep in flight:

| Resident threads per SM | Occupancy | Bytes in flight per thread | In loads |
|---|---|---|---|
| 2048 | 100% | 2.9 | one 4-byte load |
| 1024 | 50% | 5.8 | two 4-byte loads |
| 512 | 25% | 11.6 | three 4-byte loads, or one 16-byte load |
| 256 | 12.5% | 23.1 | six 4-byte loads, or two 16-byte loads |

Read the table as a trade. A thread has a fixed budget of registers and issue slots, so fewer threads need more loads
each. Full occupancy needs almost nothing from each thread; a quarter occupancy needs a 16-byte load or three
independent 4-byte ones. This is the quantitative form of the point Volkov made in 2010: occupancy is a means to
bytes in flight, and not a goal.

## Latency hiding with threads and with ILP

Two kinds of parallelism supply bytes in flight, and this section answers which to reach for. *Thread-level
parallelism* (TLP) keeps many warps resident, so while one waits for its load, the scheduler issues instructions from
another. *Instruction-level parallelism* (ILP) keeps many independent loads outstanding inside one thread.

The following kernel from the lab's fixture reads $n$ floats with
a configurable number of independent loads per thread, `ILP`, and sums them so the compiler can't drop the loads:

```cuda
template <int ILP>
__global__ void read_sum_kernel(const float* __restrict__ x, float* __restrict__ out, long n) {
  long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long nthreads = (long)gridDim.x * blockDim.x;
  float acc = 0.f;
  for (long base = tid; base < n; base += nthreads * ILP) {
    float v[ILP];
#pragma unroll
    for (int j = 0; j < ILP; ++j) {           // issue ILP independent loads...
      long i = base + (long)j * nthreads;
      v[j] = i < n ? x[i] : 0.f;
    }
#pragma unroll
    for (int j = 0; j < ILP; ++j) acc += v[j]; // ...then use them
  }
  out[tid] = acc;
}
```

The loads of one iteration don't depend on each other, so a thread issues all `ILP` of them before it waits for any.
Bytes in flight per SM are about `blocks_per_sm * threads * ILP * 4`. The lab sweeps that product from 512 bytes to
64 KB and reports the point where bandwidth stops rising. Consecutive threads read consecutive floats, and the `j`
offset steps by the whole grid, so every warp-load stays coalesced.

The two supplies cost different things:

| | TLP: more resident warps | ILP: more loads per thread |
|---|---|---|
| What it needs | Few registers and little shared memory per block, so that many blocks fit | Independent addresses, and registers to hold the in-flight values |
| What it costs | Each warp needs its own registers and shared memory | More registers per thread, so fewer threads fit |
| What it hides well | Any latency, including irregular loads | Latency of loads whose addresses are known early |
| Where it fails | A kernel with heavy register or shared-memory use can't raise occupancy | A loop-carried dependence: the next address depends on the last load |

A pointer-chasing kernel, such as a linked list or a gather through an index that was itself just loaded, gives ILP
nothing: the second load can't issue before the first returns. Only TLP helps there. A streaming kernel is the opposite
and gets cheap ILP from vector loads. In Triton you choose neither directly: a program with `BLOCK=4096` and
`num_warps=4` holds 32 elements per thread, which is ILP by construction, and `num_warps` sets TLP.

The same law applies to arithmetic, with a shorter pipe. An FMA's result isn't ready for several cycles, so a chain
of dependent FMAs in one warp leaves the arithmetic pipe idle unless other warps, or independent chains in the same
thread, fill the gaps. The latency is far shorter than a memory round trip, so a few resident warps per scheduler
are enough, which is why the lab's compute-bound kernel reaches its rate without any ILP tricks.

> [!TIP] Raise ILP before you raise occupancy
> When a profiler shows low achieved bandwidth and low occupancy, the cheaper fix is usually more independent loads per
> thread, with a `float4` load or an unrolled loop, not a smaller register budget. Registers spent on in-flight data
> buy the same bytes in flight as registers spent on extra warps.

## Waves and the tail

This section answers a question about quantity, not rate: what happens when the grid doesn't divide evenly into the
machine? The 108 SMs take blocks in rounds, and the last round can run nearly empty.

A *wave* is one full set of resident blocks. If each SM holds $r$ resident blocks, the GPU's capacity is $108\,r$ blocks,
and a grid of $G$ equal blocks takes $\lceil G / 108r \rceil$ block-times. Blocks of the same kernel run for about the
same time, so the grid's time is the number of waves multiplied by one block's time. The fraction of slots that do
useful work is the *wave efficiency*:

$$
\mathrm{waves} = \left\lceil \frac{\hla{G}}{\hlb{r}\cdot 108} \right\rceil, \qquad
\boxed{\eta = \frac{\hla{G}}{\mathrm{waves}\cdot \hlb{r}\cdot 108}}
$$

Here $\hla{G}$ is the grid size in blocks and $\hlb{r}$ is the resident blocks per SM, which
[chapter 2](/c/02-the-execution-model) computes from threads, registers, and shared memory.

The figure lets you pick the grid and the resident blocks per SM, and shows the last wave. It starts at 109 blocks with
one resident block per SM.

```viz
04-waves
```

With one resident block per SM, a grid of 108 blocks is exactly one wave at $\eta = 100\%$. Add one block, 109, and the
GPU needs a second wave for it: two block-times for $109$ blocks, $\eta = 109/216 = 50.5\%$. The following table shows how
quickly the penalty of a single extra block fades as the grid grows:

| Grid size ($r = 1$) | Waves | Efficiency |
|---|---|---|
| 109 | 2 | 50.5% |
| 217 | 3 | 67.0% |
| 433 | 5 | 80.2% |
| 865 | 9 | 89.0% |
| 1729 | 17 | 94.2% |

Each grid is $108w + 1$ blocks, the worst case for $w$ full waves. The tail costs one block-time out of $w+1$, so you
fix it either by making $w$ large, with many small blocks, or by making the grid an exact multiple of the capacity.

Most elementwise kernels never notice. The vector add on $2^{26}$ elements with 256-thread blocks has $2^{26}/256 =
262{,}144$ blocks. If 8 blocks are resident per SM (2048 threads over 256), the capacity is 864 and the grid takes 304
waves at $\eta = 262144/(304 \times 864) = 99.8\%$. A grid of thousands of waves hides the tail.

### The GEMM tile example

Waves hurt matrix multiplications, where the unit of work is a big output tile and the grid is small. NVIDIA's
[matrix multiplication guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
works a case on the A100's 108 SMs. A GEMM with $M = 2304$ and $K = 4096$, cut into $256 \times 128$ output tiles
with one block per tile:

- At $N = 1536$ the grid is $9 \times 12 = 108$ tiles: one full wave.
- At $N = 1544$ the grid is $9 \times 13 = 117$ tiles. The 9 extra tiles run in a second wave that uses
  $9 / 108 = 8.33\%$ of the SMs.

Efficiency falls from 100% to $117/216 = 54.2\%$, and NVIDIA reports throughput that roughly halves between the two
sizes, though the matrix grew by 8 columns. The lab reproduces the arithmetic with `gemm_wave_efficiency`.

There is a second quantization on top. A matrix side that isn't a multiple of the tile size still computes whole
tiles: with 128-wide tiles, $M = 130$ needs 2 tile rows and computes 256 rows to produce 130, so
$130/256 = 50.8\%$ of that dimension's work is useful. NVIDIA's guide also lists the tensor-core-friendly multiples:
64 for fp16, 32 for TF32, and 128 for INT8 on the A100.

Four fixes recur in later chapters. Smaller tiles raise the wave count and shrink the tail, at the cost of less reuse
per tile. A persistent kernel launches exactly one block per SM and loops over tiles. Split-K divides the reduction
dimension among several blocks to create more work units. Stream-K (Osama et al., 2023) generalizes the last two by
splitting the total work evenly across SMs regardless of tile boundaries. [Chapters 10 to 14](/c/10-gemm-from-scratch)
build the tiled kernels these choices apply to.

> [!WARNING] A grid smaller than the machine breaks Little's law as well
> A kernel launched with 32 blocks keeps at most 32 SMs busy. Bytes in flight are a whole-GPU requirement, 639.9 KB at
> the rating, so each of the 32 active SMs must now hold $639.9 / 32 = 20.0$ KB, nearly 3.4 times the per-SM figure
> for a full GPU. Small-batch decode kernels meet exactly this: their grids can't fill the machine, and they can't
> hold enough bytes in flight to saturate it either.

## Launch overhead as a floor

This section answers the question that bites small kernels: what is the least time any kernel can take? Even an empty
kernel has a cost, and when the kernel is small that cost is the whole answer.

A kernel launch is asynchronous: the host queues the work and returns, and the GPU starts it when it reaches the queue's
head. Learn-inference chapter 12 measured about 5 microseconds to queue a kernel on the lab A100, and its rule of thumb is
5 to 10 microseconds. A kernel that does less than that much work is launch bound, and the roofline's byte and FLOP floors
are irrelevant to it.

The crossover is a one-line calculation: the bytes the memory system moves in 5 microseconds at the copy ceiling.

$$
B_{\text{launch}} = \hlb{\beta} \times t_{\text{launch}} = 1275 \times 10^{9} \times 5 \times 10^{-6} = 6.4\ \text{MB}
$$

A kernel that moves under about 6.4 MB (under 12.8 MB at the 10-microsecond end of the range) spends more time in
launch cost than in transfer, so a sequence of such kernels is limited by how fast the host can queue them. Ten small
kernels in a layer cost at least 50 microseconds of host time at 5 microseconds each, which is where fusion
([chapter 17](/c/17-fusion-and-compilers)) and CUDA graphs earn their keep.

NVIDIA's CUDA graphs article gives the shape of the saving on a V100: a kernel that executed in 2.9 microseconds
took 9.6 microseconds per launch with a synchronization after every launch, 3.8 microseconds when launches overlapped,
and 3.4 microseconds per kernel inside a graph of 20 kernels. Those are V100 numbers, so only the ratios transfer:
launching back to back instead of waiting cuts the per-kernel cost by about 2.5 times, and a graph trims it further.

There is also a latency floor beneath the launch floor. A kernel whose threads follow $d$ dependent memory round trips can't
finish in under about $d \times 330.7$ ns, whatever its size: two dependent global loads cost at least 0.66 microseconds.
That's small next to the launch, but it's the floor for a kernel that reads an index and then the data the index points at.

## The four floors together

This section answers the practical question: how do you turn the four floors into one prediction you can write down
before you run the kernel?

Each floor is a lower bound on a different part of the time. Memory latency enters by lowering the bandwidth a
kernel can reach, so replace $\beta$ by the smaller of the ceiling and what Little's law allows from the bytes the
launch keeps in flight, and add the launch cost on top:

$$
t \;\approx\; \hld{t_0} + \max\!\left(\frac{\hla{F}}{\hla{C}},\; \frac{\hlb{B}}{\min(\beta,\; L/W)}\right)
$$

Here $\hld{t_0}$ is the launch cost, $L$ is the bytes the launch keeps in flight across the GPU, $W$ is the memory latency,
and $\beta$ is the ceiling you chose. The result is a prediction, and the sharp corner of a $\max$ is an idealization.
Near the ridge point, a real kernel usually falls below both ceilings, because memory and arithmetic overlap imperfectly. The lab
sweep shows the corner in your own data.

The following table lists the floors with the A100 number for each and how the lab checks it:

| Floor | Formula | A100 value | The lab |
|---|---|---|---|
| Arithmetic | $F / C$ | 19.5 TFLOP/s for fp32 FMAs | Reports your fp32 rate from the $k$ sweep |
| Bandwidth | $B / \beta$ | 1275 GB/s measured copy | Reports the rate at $k=1$ |
| Latency | $L = \beta W$ | 3.9 KB per SM at 1275 GB/s | Sweeps bytes in flight per SM |
| Launch | $t_0$ | about 5 microseconds | Times an empty kernel |
| Wave | $\lceil G/108r \rceil$ block-times | 2 waves at 109 blocks | Times 54 to 217 blocks |

### A worked prediction

Take the lab's kernel with $k = 64$ on $2^{26}$ float32 elements, which is 268 MB in and 268 MB out. It does
$2 \times 64 \times 2^{26} = 8.59$ GFLOP and moves $8 \times 2^{26} = 536.9$ MB. Walk the floors in order:

1. **Intensity.** $8.59 \times 10^{9} / 536.9 \times 10^{6} = 16.0$ FLOP/byte, against an fp32 ridge of 15.3 at the
   copy ceiling. The kernel sits just right of the ridge.
2. **Arithmetic.** $8.59 \times 10^{9} / 19.5 \times 10^{12} = 440.5\ \mu\text{s}$.
3. **Memory.** $536.9 \times 10^{6} / 1275 \times 10^{9} = 421.1\ \mu\text{s}$ at the copy ceiling, or $277.5\ \mu\text{s}$ at
   the rating.
4. **Latency.** A grid of 262,144 blocks of 256 threads gives each SM up to 2048 resident threads with one 4-byte load
   each, 8 KB per SM, above the 5.9 KB the rated bandwidth needs.
5. **Waves and launch.** 304 waves at 99.8% efficiency, and a 5 microsecond launch that is about 1% of the time.

The prediction is 440.5 microseconds, compute bound, and only just. The roofline's lower bound is 277.5 microseconds if
you believe the rating, which no kernel reaches. The same arithmetic for the lab's sweep gives:

| $k$ | Intensity (FLOP/byte) | Arithmetic floor | Memory floor (1275 GB/s) | Predicted time |
|---|---|---|---|---|
| 1 | 0.25 | 6.9 µs | 421.1 µs | 421.1 µs, memory |
| 16 | 4.0 | 110.1 µs | 421.1 µs | 421.1 µs, memory |
| 32 | 8.0 | 220.3 µs | 421.1 µs | 421.1 µs, memory |
| 64 | 16.0 | 440.5 µs | 421.1 µs | 440.5 µs, compute |
| 128 | 32.0 | 881.0 µs | 421.1 µs | 881.0 µs, compute |
| 512 | 128.0 | 3524.1 µs | 421.1 µs | 3524.1 µs, compute |

The ridge falls at $k = 61.2$, because $k/4 = 15.3$. These are predictions from the specs. Your lab prints the measured
column beside them and reports the measured ridge, the highest FLOP rate divided by the highest byte rate any run
reached.

## What goes wrong

- **Counting only the reads, or only the writes.** A copy moves $2N$ bytes. The lab's kernel moves $8$ per element,
  and a count of 4 halves the bandwidth it reports.
- **Benchmarking data that lives in the L2.** An array smaller than the 40 MiB cache is read from cache after the first
  iteration, and the bandwidth looks impossible. The lab uses 256 MiB, and `cuda_time(..., flush_l2=True)` exists for
  smaller inputs.
- **A flat time as $k$ grows.** If the time doesn't rise with $k$, the compiler has folded or deleted the arithmetic.
  Check that the loop's result reaches a store, and read the SASS ([chapter 5](/c/05-profiling)).
- **Using the wrong arithmetic peak.** A fp32 kernel measured against 312 TFLOP/s looks 16 times worse than it is.
- **Quoting the rating as the target.** A kernel at 1275 GB/s is at the copy ceiling. The SXM4 card, rated 2039 GB/s,
  may be the one you get, and the lab prints its SM count so you can tell.
- **Treating occupancy as a goal.** Low occupancy with enough bytes in flight per thread isn't a defect.
- **Wrong resident blocks per SM.** The wave count depends on $r$ from the register and shared-memory limits, not on
  the 32-block cap. The lab forces $r = 1$ with a 100 KiB shared-memory allocation so that you can count waves exactly.
- **Timing the launch.** Event timing around a single tiny kernel includes the host queueing time. The lab's launch
  floor is a launch-plus-empty-kernel figure, and it is what a user of PyTorch sees.
- **Expecting a clean corner.** Near the ridge point, memory and arithmetic overlap imperfectly, so a kernel there is
  slower than either floor alone says. Judge a result by the trend across the sweep.

> [!RECAP]
> - Predict a kernel's time from four floors: arithmetic $F/C$, memory $B/\beta$, latency through bytes in flight, and
>   launch cost.
> - Compute the intensity from compulsory bytes, and compare it with the ridge point for the precision you use: 15.3
>   FLOP/byte for fp32 and 244.7 for bf16, both against the measured 1275 GB/s.
> - Build a rated roofline as a hard floor and a measured one as the target, and use per-level rooflines when reuse in
>   the L2 or shared memory changes the DRAM bytes.
> - Little's law sets the bytes in flight: 3.9 KB per SM at 1275 GB/s, supplied by resident warps, independent loads per
>   thread, or both.
> - Grids run in whole waves, so efficiency is $G/(\mathrm{waves} \cdot 108r)$, and 109 blocks on 108 SMs run at 50.5%.
> - Below about 6.4 MB of traffic a kernel is launch bound, and fusion or CUDA graphs are the cure.

## Check your understanding

> [!QUESTION] A kernel reads 8 bytes and writes 4 bytes of float32 data per element and does 6 FLOPs. For $2^{26}$ elements, what does the roofline predict on the lab A100?
> Intensity is $6/12 = 0.5$ FLOP/byte, far below the fp32 ridge of 15.3, so memory binds. It moves
> $12 \times 2^{26} = 805.3$ MB, which takes $805.3 \times 10^{6} / 1275 \times 10^{9} = 631.6\ \mu\text{s}$ at the copy ceiling.
> The arithmetic needs $6 \times 2^{26} / 19.5 \times 10^{12} = 20.6\ \mu\text{s}$, a thirtieth of that.

> [!QUESTION] A kernel does 20 FLOP per byte. Is it compute bound or memory bound?
> It depends on the arithmetic peak. On the fp32 CUDA cores, the ridge against 1275 GB/s is 15.3, so 20 is right of it
> and the kernel is compute bound. On bf16 tensor cores the ridge is 244.7, so the same kernel is memory bound.

> [!QUESTION] A kernel runs at 25% occupancy, 512 resident threads per SM. How many bytes must each thread keep in flight to saturate the rated 1935 GB/s?
> Little's law needs 5,925 bytes in flight per SM. Divided over 512 threads, that is 11.6 bytes per thread: three
> independent 4-byte loads, or one 16-byte `float4` load.

> [!QUESTION] You launch 217 blocks of one block per SM on the 108-SM A100, and then 216. How do the wave counts and efficiencies compare?
> 217 blocks need 3 waves at $217/324 = 67.0\%$. 216 blocks need 2 waves at $216/216 = 100\%$. Dropping one block saves a
> wave's worth of time, one third of the runtime.

> [!QUESTION] A GEMM with $M = 2304$ and 256 x 128 tiles has $N = 1544$. How many tiles is that, and what happens to its throughput?
> $9 \times 13 = 117$ tiles. On 108 SMs with one block per SM that's 2 waves, with the second wave using 9 of 108 SMs.
> Efficiency is $117/216 = 54.2\%$, and the time is about that of 216 tiles, almost the same as the time for
> $N = 1664$, so the GEMM runs at roughly half the speed it had at $N = 1536$.

> [!QUESTION] A kernel moves 2 MB. What does the roofline predict, and what actually limits it?
> The memory floor is $2 \times 10^{6} / 1275 \times 10^{9} = 1.6\ \mu\text{s}$. A launch costs about 5 microseconds, so
> launch cost dominates, three times the data movement. Fuse the kernel with its neighbours or capture the sequence in a
> CUDA graph.

> [!QUESTION] A measured time falls below the roofline built from the rated peaks. What does that tell you?
> Something in the measurement or the count is wrong, because no kernel can beat the rated peaks. Check that you counted
> every byte, that the input didn't fit in the L2, that you timed the kernel and not only the launch, and that the
> compiler didn't delete work.

## Lab

> [!TRY] Predict, then measure
> In `starter.py`, write the predictors: `ridge_point`, `roofline_ms`, `fma_kernel_counts`, `num_waves`,
> `wave_efficiency`, `gemm_tiles`, `gemm_wave_efficiency`, `bytes_in_flight`, `predicted_read_gbs`, and
> `measured_ridge`. The tests first check them as pure arithmetic against the numbers in this chapter. They then
> launch three CUDA kernels, with $k$ FMAs per element, a fixed block workload on one block per SM, and a read kernel
> with a tunable number of loads in flight, and print each prediction beside the measured time.
>
> You pass when the predictors are right, no measured time beats the rated roofline, and the measured behavior agrees
> with the model within loose bounds: the $k=1$ kernel reaches half the copy ceiling, the measured ridge falls
> within a factor of four of the predicted 15.3 FLOP/byte, 109 blocks take at least 1.6 times as long as 108, and a
> load-starved read kernel runs at most half as fast as the best. The first run compiles with `nvcc` for 30 to 60
> seconds. The lab reports `ridge_measured`, `fp32_tflops`, `k1_bw_gbs`, `launch_floor_us`, `wave_step_ratio`,
> `read_peak_gbs`, and `inflight_kb_per_sm_90pct`.

## Further reading

- [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://zenodo.org/record/1236156):
  Williams, Waterman, and Patterson, 2009, the original model and the definition of arithmetic intensity.
- [Hierarchical Roofline Performance Analysis for Deep Learning Applications](https://arxiv.org/abs/2009.05257):
  Yang, Wang, Farrell, Kurth, and Williams, 2020, one roof per memory level, applied to GPU kernels.
- [Better Performance at Lower Occupancy](https://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf): Volkov,
  2010, the bytes-in-flight argument and why ILP can replace occupancy.
- [Benchmarking and Dissecting the Nvidia Hopper GPU Architecture](https://arxiv.org/abs/2402.13499): Luo and others,
  2024, the A100 PCIe latencies and cache throughputs this chapter uses.
- [Matrix Multiplication Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html):
  NVIDIA, arithmetic intensity, tile quantization, and the 108-tile wave example.
- [GPU Performance Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html):
  NVIDIA, math-bound versus memory-bound and the ops-to-byte ratio.
- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html): Horace He, 2022, compute,
  memory, and overhead regimes for ML operations.
- [Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU](https://arxiv.org/abs/2301.03598):
  Osama and others, 2023, a remedy for the wave tail in GEMM.
- [How to Scale Your Model: All About Rooflines](https://jax-ml.github.io/scaling-book/roofline/): Austin and others,
  2025, the roofline applied to transformer layers.
- [The roofline chapter of the companion course](https://qwen.ifkash.dev/c/10-roofline): the same model applied to the
  operations of one model, with the MLP, attention, and RMSNorm worked out.
