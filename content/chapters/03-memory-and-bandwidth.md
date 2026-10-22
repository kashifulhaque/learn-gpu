---
title: Memory and bandwidth
slug: 03-memory-and-bandwidth
part: "Part 1 — The machine"
summary: Where data lives on an A100, how a warp's load becomes 32-byte sectors, and how to write copy kernels that reach the card's measured bandwidth ceiling, then measure the gap between that ceiling and the datasheet.
minutes: 90
gpu: true
objectives:
  - "Name the four levels of the A100's memory hierarchy with their sizes, bandwidths, and measured latencies, and say which sources disagree."
  - "Count the 32-byte sectors and 128-byte lines a warp's load touches, and compute its efficiency for any stride and offset."
  - "Count the bytes a kernel moves, reads plus writes, and turn a time into a bandwidth you can compare with 1275 GB/s."
  - "Write a float4 vectorized copy that handles the alignment requirement and the tail."
  - "Write a grid-stride loop, and explain what it buys and what it doesn't."
  - "Use Little's law to say how many bytes must be in flight to saturate HBM."
  - "Describe the read-only path, cache hints, and L2 persistence, and say when each is worth trying."
  - "Run a size sweep, report the bandwidth plateau, and compare it honestly with the 1935 GB/s rating and the datasheet's 95% claim."
lab: 03-bandwidth-probe
---

# Memory and bandwidth

> [!TLDR]
> - The A100 has four places to keep data: registers, shared memory and L1, a 40 MB L2, and 80 GB of HBM.
>   Each level down is bigger, slower, and shared by more threads. HBM takes about 466 cycles to answer.
> - Memory moves in 32-byte sectors. A warp that reads 32 consecutive floats touches 4 sectors and
>   uses every byte. A warp that reads at a stride of 8 floats or more touches 32 sectors and uses
>   one eighth of what it fetches.
> - Count bytes as reads plus writes. A copy of $N$ bytes moves $2N$. A vector add moves $3N$.
>   Divide by CUDA-event time and compare with the measured 1275 GB/s copy ceiling, not the 1935 GB/s rating.
> - Vectorized `float4` loads, grid-stride loops, and read-only hints all aim at one thing: keeping
>   enough bytes in flight to cover the 330 ns HBM latency. The lab measures each one.
> - The datasheet claims 95% DRAM efficiency and the lab's copy reaches 66%. No source in this
>   course explains the gap, so the lab sweeps the copy size and you report where the plateau lands.

A transformer layer spends most of its time moving bytes. Chapter 0 gave the reason: a bandwidth rating of
1935 GB/s against 312 TFLOP/s of tensor-core arithmetic puts the ridge point at 161 FLOP per byte, and
most of the kernels in a model sit far to its left. For those kernels, the time is the bytes divided by
the bandwidth, and nothing else matters until you've made both numbers honest.

This chapter makes them honest. You learn what the memory hierarchy offers, how a warp's 32 loads turn
into traffic, and how to count that traffic so a measured time converts to a bandwidth you can trust. Then
you write the simplest bandwidth-bound kernel there is, a copy, four ways, and find the ceiling the
card actually allows.

The chapter ends on a question it can't fully answer. The card's datasheet says its HBM can run at
95% efficiency, and a copy on the lab card runs at 66% of the rating. You measure that gap yourself.

## Before you start

**Warp.** A group of 32 consecutive threads that execute one instruction together. The memory system sees a
warp's 32 loads as one request. [Chapter 2](/c/02-the-execution-model) covers warps in full.

**HBM.** High-bandwidth memory, the stacked DRAM beside the GPU die. On the A100 80GB PCIe it has a 5,120-bit
interface and a rated bandwidth of 1935 GB/s, in decimal units: $1.935 \times 10^{12}$ bytes per second.

**Latency and bandwidth.** Latency is how long one request takes to come back. Bandwidth is how many bytes per second the
system delivers once it has many requests in progress. They are different quantities, and this chapter uses both.

**The ridge point.** The arithmetic intensity, in FLOP per byte, above which a kernel is limited by arithmetic and
below which it is limited by memory. [Chapter 0](/c/00-what-you-are-building) derived it, and
[chapter 4](/c/04-roofline-and-latency) draws the full roofline.

## The memory hierarchy

The hierarchy answers one question: where can a thread keep a value, and what does each place cost?
Four levels matter on the A100, and the figures below come from the A100 datasheet, the Ampere GA100
whitepaper, and a microbenchmark paper that measured latency on the card.

| Level | Size | Scope | Latency, measured | Bandwidth |
|---|---|---|---|---|
| Registers | 256 KB per SM, 27,648 KB across 108 SMs | One thread | Part of the instruction | Not a limit |
| Shared memory and L1 | 192 KB per SM, of which up to 164 KB is shared memory | Block (shared), SM (L1) | 29.0 cycles shared, 37.9 cycles L1 | 128 B per clock per SM theoretical: 19.5 TB/s across the GPU |
| L2 cache | 40 MB | The whole GPU | 261.5 cycles | 7.2 TB/s theoretical, about 2.6 to 2.8 TB/s measured (see the next callout) |
| HBM2e | 80 GB | The whole GPU | 466.3 cycles | 1935 GB/s rated, 1275 GB/s measured on a copy |

Here is the arithmetic behind the bandwidth column. Shared memory has 32 banks of 4 bytes, so one SM
serves 128 bytes per clock. Across the card at the 1410 MHz boost clock:

$$
128\ \text{B/clk} \times 108\ \text{SMs} \times 1.41 \times 10^{9}\ \text{clk/s} = 19.5\ \text{TB/s}
$$

That is the theoretical rate for all 108 SMs reading shared memory in the same cycle, and it is ten times
the bandwidth of HBM. The whitepaper's L2 figure is 5,120 bytes per clock, which is
$5120 \times 1.41 \times 10^{9} = 7.2$ TB/s.

The latency column is a measurement, not a rating. Luo and colleagues measured them on an A100 PCIe **40 GB** card at 1410 MHz. (Jia et al.'s Volta paper
is the standard reference for the pointer-chasing method behind such numbers.)
Dividing cycles by 1.41 gives nanoseconds:

| Level | Cycles | Nanoseconds |
|---|---|---|
| L1 | 37.9 | 26.9 |
| Shared memory | 29.0 | 20.6 |
| L2 | 261.5 | 185.5 |
| HBM | 466.3 | 330.7 |

> [!WARNING] Two sources disagree, and both are right about something
> The companion course's hierarchy table rounds the latencies to about 30, 200, and 400 cycles for shared
> memory, L2, and HBM. That is a teaching table. This chapter uses the measured 29.0, 261.5, and 466.3
> cycles, and says "roughly 30, 260, and 470" when it rounds. The measurements are from the 40 GB
> card. No one in the sources measured the 80 GB card, so treat the numbers as close, not exact.
>
> The L2 bandwidth sources disagree too. The whitepaper's theoretical figure is 5,120 B per clock.
> Luo and colleagues measured 1,853.7 to 2,007.9 B per clock on an A100 PCIe, which is 2.6 to 2.8 TB/s.
> Chips and Cheese, a secondary source, measured 4.5 TB/s on an A100 SXM part with a different method. They tested different
> cards by different methods, so this chapter quotes the range and names the source each time.

Three facts from the table shape every kernel in the course.

First, ==each level down is roughly an order of magnitude slower in latency and several times narrower in
bandwidth==. A value that lives in a register costs nothing to reuse, and a value that must come from HBM
costs 466 cycles each time a warp asks for it.

Second, the fast levels are small. The whole card's register file is 27,648 KB, and its shared memory is
$108 \times 164\ \text{KB} = 17{,}712$ KB. Both are smaller than the 40 MB L2, and all three together are
under 1% of the 80 GB of HBM. A kernel that works on gigabytes streams them through these levels.

Third, L2 sits between the SMs and HBM and serves everything. Whether the data you're about to read is in
it decides whether the access costs 261.5 cycles or 466.3. That is why a benchmark that reuses a small
buffer measures the cache, not the memory, and why the lab sweeps sizes on both sides of 40 MB.

> [!NOTE] Where L1 and shared memory split
> L1 and shared memory share one 192 KB array per SM. A kernel picks a carveout of shared memory, from
> 0 to 164 KB in the steps 0, 8, 16, 32, 64, 100, 132, and 164, and the rest is L1.
> [Chapter 8](/c/08-shared-memory-and-transpose) uses it.

## What a load really moves

A load moves whole 32-byte sectors, never single floats, and the number of sectors a warp touches
decides how much of the traffic is useful. This section counts them.

The memory system works on two granularities. A *sector* is 32 bytes, the unit that travels between the
caches and DRAM, whether or not a global load is cached in L1, according to the CUDA C++ Best
Practices Guide. A *cache line* is 128 bytes, four sectors, and L1 and L2 track their hits at that
granularity. The current guide counts only 32-byte transactions, so the 128-byte line comes from an
archived edition of the Programming Guide. For counting traffic, the sector is the number to use.

The coalescing rule for compute capability 6.0 and later is that a warp's accesses are served by as
many 32-byte transactions as it takes to cover the addresses the lanes asked for.

### The aligned, sequential case

Lane $t$ of a warp loads one `float` at index $i_0 + t$, with the base address on a 128-byte boundary. The
32 lanes cover $32 \times 4 = 128$ contiguous bytes, which is exactly four sectors and one cache line.
Every fetched byte is used.

### Misalignment

Shift the base by one float, so lane 0 reads the last 4 bytes of a sector. The warp still needs 128 bytes, but now
they straddle five sectors: 160 bytes fetched for 128 used, an efficiency of 80%. The Best Practices
Guide gives the same count of five 32-byte segments. A base that isn't on a 128-byte boundary also spans two cache lines.
The cost is mild, because only the two ends of the span are partly wasted.

### Stride

Now let lane $t$ read element $s \cdot t$, with a stride of $s$ floats. The addresses are $4s$ bytes apart. While
$4s \le 32$, every sector in the span still has at least one lane's value in it, so the warp fetches the
whole span: $4s$ sectors for $s \le 8$. Once the addresses are 32 bytes apart or more, every lane
needs a sector of its own, and the count tops out at 32.

| Stride $s$ (floats) | Bytes between lanes | Sectors fetched | Bytes fetched | Useful bytes | Efficiency |
|---|---|---|---|---|---|
| 1 | 4 | 4 | 128 | 128 | 100% |
| 2 | 8 | 8 | 256 | 128 | 50% |
| 4 | 16 | 16 | 512 | 128 | 25% |
| 8 | 32 | 32 | 1,024 | 128 | 12.5% |
| 32 | 128 | 32 | 1,024 | 128 | 12.5% |

Efficiency is the useful bytes over the fetched bytes:

$$
\boxed{\text{efficiency} = \frac{\hla{32 \times 4}}{\hlb{32\,\text{B} \times \text{sectors touched}}}}
$$

The numerator, $\hla{128}$ bytes, is what the warp asked for. The denominator is the traffic: what
the memory system moved to deliver it. The figure lets you move the stride and the start offset and watch
the sectors fill in.

```viz
03-coalescing
```

Set the figure to a stride of 32 and read the readout: 32 sectors, 1,024 bytes fetched, 12.5%. That is the
case the companion course's lab measured. Then switch to 16 bytes per lane, which is `float4`, at a stride of 1, and
see that a warp's 512 bytes fill 16 consecutive sectors with no waste. Then try an offset of 1 at 16 bytes per lane: the readout flags
it, because a `float4` access must start on a 16-byte boundary, and the hardware raises a fault instead of fetching
extra sectors.

> [!KEY] The sector is the unit of waste
> A stride of 32 floats costs 8x on the read, not 32x, because the memory system fetches 32 bytes at a time
> and not one float. The loss grows with the stride until each lane has its own sector, and then it stops.

### The measured 6.9x

The companion course ([chapter 12](https://qwen.ifkash.dev/c/12-your-first-cuda-kernel)) measured this on the
lab card. At $n = 2^{24}$ floats, a coalesced vector add reached 1304 GB/s and a strided copy at a stride of 32
reached 189 GB/s, each counting its own logical bytes. The ratio is $1304 / 189 = 6.9$.

The sector model explains part of it. The strided copy reads a 32-byte sector for each 4 bytes it uses and
writes 4 bytes coalesced, so it moves $32 + 4 = 36$ bytes per element while counting $4 + 4 = 8$. That is an
amplification of $36 / 8 = 4.5$. If bytes were the only cost, the strided kernel would count
$1304 / 4.5 = 290$ GB/s. It measured 189, so a second factor of $290 / 189 = 1.53$ comes from
something other than bytes. The companion course attributes it to the 8x increase in sector requests per
warp, which queue in the SM and at L2, and that explanation is a model, not a measurement of the queues.

$$
6.9 = \hlc{4.5} \times \hld{1.53}
$$

The $\hlc{4.5}$ is fetched bytes thrown away, and the $\hld{1.53}$ is the extra cost of the requests.

Two cautions keep the 6.9 in perspective. First, its read footprint is small: $2^{24} / 32 = 524{,}288$ distinct
elements, one sector each, is 16.8 MB, which fits in the 40 MB L2. After warm-up the reads hit the cache, which
flatters the strided kernel. A strided pattern with a footprint above the L2 should do worse, and the lab
measures that second case (`strided_big_gbs`, a 64 MiB footprint). Second, the sector model gives the
right order of magnitude for the slowdown, and the 1.53 is the part it doesn't predict.

## Counting the bytes

A kernel's bandwidth is the bytes it moved divided by the time it took, so the answer is only as good as
the byte count. The count is reads plus writes, and the common mistake is to leave out half of it.

For a kernel that touches each of $n$ float32 elements once on each side, write the logical traffic as:

$$
\boxed{B = \hla{R} + \hlb{W}}
$$

where $\hla{R}$ is the bytes read from memory and $\hlb{W}$ is the bytes written. The bandwidth is
$B / t$, in bytes per second, and this course reports it in decimal GB/s.

| Kernel | Reads | Writes | Bytes moved |
|---|---|---|---|
| Copy, $n$ floats | $4n$ | $4n$ | $8n$ |
| Vector add, $n$ floats | $8n$ | $4n$ | $12n$ |
| Strided copy, counted | $4n$ | $4n$ | $8n$ |
| Strided copy, really moved at stride $\ge 8$ | $32n$ | $4n$ | $36n$ |

The last two rows are the same kernel counted two ways. The *counted* bytes are what the kernel logically
needs, and they are what you divide by the time to compare kernels fairly. The *really moved* bytes are what
the memory system fetched, and they explain why a strided kernel's counted bandwidth is low.

> [!EXAMPLE] A 256 MiB copy, by hand
> A tensor of $2^{26}$ floats is $2^{26} \times 4 = 268{,}435{,}456$ bytes, or 256 MiB. A copy reads one and
> writes one, so it moves $2 \times 268{,}435{,}456 = 536{,}870{,}912$ bytes.
>
> At the measured ceiling of 1275 GB/s that takes $536.87 \times 10^{6} / 1.275 \times 10^{12} = 0.421$ ms. At the rated
> 1935 GB/s the floor is $0.277$ ms. If your timer reads 0.421 ms and you divide only the 268 MB you copied, you report
> 638 GB/s and conclude something is wrong, when the kernel is at 1275 GB/s and done.

The lab's `bytes_moved` function is this table in code, and `sector_bytes_per_element` is the last row.

## Vectorized loads

A `float4` load moves 16 bytes per thread in one instruction, so a kernel needs a quarter of the
load and store instructions and keeps more bytes in flight per thread. It costs an alignment
requirement and a tail.

Widen the thread's access. The widest a single thread can load is 16 bytes, the `LDG.128` instruction,
which CUDA exposes through built-in vector types such as `float2` and `float4`. One warp-wide
`float4` load then touches $32 \times 16 = 512$ bytes, or four cache lines, in one instruction.

The change to the copy kernel is small. Reinterpret the pointers as `float4*` and divide the element count by 4:

```cuda
__global__ void copy_vec4_kernel(const float4* __restrict__ src,
                                 float4* __restrict__ dst, long long n4) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) dst[i] = src[i];          // one 16-byte load and one 16-byte store
}
```

Here `n4` is the number of `float4` elements, so a tensor of $n$ floats needs $\lfloor n / 4 \rfloor$ of them.

### The alignment requirement

A global access of 1, 2, 4, 8, or 16 bytes compiles to a single instruction only if the address is a multiple of
its size, which is the *natural alignment*. A `float4` access at an address that isn't a multiple of 16 is not slower: it
raises a misaligned-address error, and the process dies. Three facts keep you safe:

- Allocations from `cudaMalloc` are at least 256-byte aligned, and PyTorch's caching allocator hands out blocks from
  them, so a fresh tensor's `data_ptr()` is a multiple of 16.
- A view breaks the guarantee. `x[1:]` starts 4 bytes past an aligned address, and `x[3:]` starts 12 bytes past it.
- Checking is cheap. The host side tests the address modulo 16 and falls back to the scalar kernel when it isn't zero.

The lab's `copy_vec4` does exactly that, and its tests pass it a tensor sliced from an offset of 1 on purpose.

### The tail

If $n$ isn't a multiple of 4, the last $n \bmod 4$ floats don't fill a `float4`. The simplest correct answer is
a second launch of the scalar kernel on those few elements, offset by $4 \lfloor n / 4 \rfloor$. NVIDIA's
vectorized-access article instead lets one extra thread copy the leftovers with scalar loads. Either way, the
tail is a few elements, and the tests use $n = 1$, 3, 5, and 1,000,003 to check it.

```cuda
long long n4 = n / 4;
if (n4 > 0) copy_vec4_kernel<<<(n4 + 255) / 256, 256>>>(
    reinterpret_cast<const float4*>(s), reinterpret_cast<float4*>(d), n4);
launch_scalar(s + 4 * n4, d + 4 * n4, n - 4 * n4);   // the last n % 4 floats
```

### What it buys

Vectorizing cuts the instruction count: the loop that issued $n$ scalar loads issues $n / 4$ vector loads.
Luitjens' article says this lowers the executed instruction count by a factor of up to four, and that it matters
most for kernels limited by instruction issue or by latency. It also notes the cost: more registers per
thread, which can reduce occupancy. A kernel that is already register-limited gains little.

The next section turns "latency" into a number, and shows why a wider load can help even when the instruction
count is not the problem. Triton, from [chapter 14](/c/14-triton), emits wide loads on its own when it can prove alignment
and contiguity, so the choice you make by hand in CUDA is one the compiler makes for you there.

## Bytes in flight

A load takes about 466 cycles to return, so the memory system only reaches full bandwidth when enough
loads are in progress at once. This is Little's law, and it sets how many threads and how many bytes
per thread a bandwidth-bound kernel needs.

Little's law says that the average number of requests in a system equals the arrival rate times the time
each one stays. Here the rate is the bandwidth $\hla{\beta}$ and the time is the latency $\hlb{L}$, so the bytes in
flight must reach:

$$
\boxed{\text{bytes in flight} = \hla{\beta} \times \hlb{L}}
$$

Take $\hla{\beta} = 1935$ GB/s and the HBM latency of 466.3 cycles at 1.41 GHz, which is 330.7 ns:

$$
1.935 \times 10^{12}\ \text{B/s} \times 330.7 \times 10^{-9}\ \text{s} = 639{,}900\ \text{B} \approx 640\ \text{KB}
$$

Across 108 SMs, that is about 5,925 bytes in flight per SM. This is an arithmetic result from the card's rating and
the 40 GB card's latency, and it is a prediction, not a measurement. [Chapter 4](/c/04-roofline-and-latency)
develops the law and its limits.

Turn that into threads. If every thread has one 4-byte load outstanding, an SM needs
$5{,}925 / 4 \approx 1{,}480$ threads with a load in flight, which is 72% of the 2,048 an SM can hold. With
`float4` loads, each thread has 16 bytes outstanding, and the need falls to $5{,}925 / 16 \approx 370$ threads,
18% of the maximum. The same arithmetic applies to independent loads inside one thread: two outstanding
`float` loads per thread count the same as one `float2`.

```viz
03-bytes-in-flight
```

> [!INTUITION] Wider loads buy slack
> A scalar copy at full occupancy only just has the bytes in flight to cover HBM. Any stall, a register-limited
> block, or a partial last wave leaves it below the line. A `float4` copy has four times the margin at the same
> occupancy, so it can lose threads to resource limits and still keep the memory system fed. Volkov's talk on
> better performance at lower occupancy makes the same argument.

The prediction says wide loads should help a scalar kernel that is short of bytes in flight, and say nothing about
a kernel that already has plenty. The lab reports `scalar_gbs` and `vec4_gbs` at 256 MiB, and you judge which
side of the line the scalar kernel falls on. Don't assume the vector kernel wins: if the scalar copy already reaches
the ceiling, the two will match, and that is a finding too.

## Grid-stride loops

A grid-stride loop sizes the grid to the machine and lets each thread walk the array, which decouples the launch
from the data size. The pattern comes from Mark Harris's NVIDIA blog post, and the companion course shows it for a
vector add.

```cuda
__global__ void copy_grid_stride_kernel(const float* __restrict__ src,
                                        float* __restrict__ dst, long long n) {
    long long stride = (long long)blockDim.x * gridDim.x;   // threads in the whole grid
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        dst[i] = src[i];
    }
}
```

Each pass of the loop covers `stride` consecutive elements, one per thread, then the whole grid moves forward by
its own width. Within one pass, the 32 lanes of a warp are still on 32 consecutive floats, so the load stays
coalesced. That is the reason the stride is the grid's size and not something smaller.

What it gives you:

- **A launch that doesn't depend on $n$.** One grid size, tuned once, handles every array. For the A100 you
  would pick a multiple of the 108 SMs, and the lab uses $108 \times 8 = 864$ blocks of 256 threads, which is
  eight blocks, or 2,048 threads, per SM.
- **No separate tail guard.** The loop condition `i < n` is the guard.
- **Amortized index math.** Each thread computes its start once and adds a constant.
- **Debuggability.** Launch it with one block and one thread and it is a serial loop. The lab runs the kernel with
  one block of 256 threads on 1,000,003 elements for this reason.

What it doesn't give you is a guaranteed speedup. For a copy, each thread's loop is already memory-bound, and
the one-thread-per-element kernel launches millions of tiny blocks that the hardware schedules without trouble. The prediction is that the
two kernels reach about the same bandwidth, and the lab reports both so you can see. The pattern
pays off when per-thread setup is expensive, when the data size varies, or when you want a *persistent* kernel
that keeps its state across iterations, as persistent matmul kernels do.

> [!WARNING] Keep the index arithmetic in 64 bits
> `i += stride` and `i * stride` overflow a 32-bit `int` on large arrays. A 4 GiB tensor has $2^{30}$ floats, and a
> stride of 32 pushes `i * stride` to $2^{35}$. The kernels in this chapter use `long long` for every index. A
> signed overflow is undefined behavior in C++, so the kernel may read the wrong place without complaint.

## Read-only data, cache hints, and L2 persistence

This section covers three ways to tell the hardware how you'll use the data: that you won't write it, that you
won't reuse it, and that you want it to stay. None of them changes what a copy must move, and the chapter's
lab doesn't measure them. They are what to try after the byte count says you're at the limit.

### The read-only path

Declaring a pointer `const float* __restrict__` promises the compiler that the data is not written through any other
pointer during the kernel. The compiler can then load it through the read-only (non-coherent) path and reorder loads
more freely. The explicit form is the intrinsic `__ldg(ptr)`, available since compute capability 3.5. On Ampere the L1 and
texture caches share one array, so the difference between the paths is mostly about what the hardware is allowed
to assume. To see whether the compiler used it, read the SASS: [chapter 5](/c/05-profiling) shows how, and a load
through the read-only path typically carries a `.CONSTANT` suffix on its `LDG` instruction.

All four kernels in the lab take `const __restrict__` sources for this reason.

### Cache operators

The `__ldcg`, `__ldcs`, `__stcs`, and related intrinsics attach a cache operator to one access. `__ldcg` caches at L2 only,
skipping L1, and `__ldcs` marks the data as streaming, so it is the first to be evicted. A copy is the classic
streaming workload: each byte is read once and never reused, so evicting it early leaves room for data
that is reused. Whether that moves the number on a pure copy is a thing to measure, and the lab doesn't ask you to. If you try
`__ldcs` and `__stcs` in the copy kernel, the sweep is the right place to see the effect.

### L2 persistence

Ampere added programmer control of L2 residency. The mechanism has two parts: you set aside a portion of L2 for
*persisting* accesses with `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, bytes)`, and you mark a window of
memory as persisting by setting an `accessPolicyWindow` (a base pointer, a size in bytes, a `hitRatio`, and the
hit and miss properties) on a stream. Reads in the window then tend to stay in the reserved L2 region, instead of being
evicted by streaming traffic. The CUDA Programming Guide's L2 cache control section documents it, including the
device limits you should query before choosing a size.

The use case is a small hot array, such as a lookup table or an embedding block, read repeatedly beside a large
streaming array that would otherwise flush it. For a one-pass copy there is nothing to keep. This chapter doesn't
state limits or speedups for it, because the sources read for this course don't give figures for the A100, and PyTorch
doesn't expose it. Chapter 14's grouped matmul ordering gets a similar effect for matrix tiles without any API,
by ordering the work so that tiles that share data run near each other in time.

## The gap between the rating and a copy

The A100's datasheet and the lab's copy disagree by a third, and this section sets the numbers side by side
without choosing a cause. It is the one place in the chapter where the honest answer is "the sources don't say."

The numbers on the table are as follows:

| Figure | GB/s | Fraction of 1935 | Source and conditions |
|---|---|---|---|
| Rated bandwidth, A100 80GB PCIe | 1935 | 100% | NVIDIA datasheet |
| Datasheet's "95% DRAM utilization efficiency" | 1,838 | 95% | Datasheet text; $0.95 \times 1935$ |
| Lab copy, 42 MB per direction | 1275 | 65.9% | Measured in learn-inference, chapter 10 |
| Lab vector add, $2^{24}$ floats | 1304 | 67.4% | Measured in learn-inference, chapter 12, counting $3n$ |
| Luo et al., throughput test | 90% of theoretical | 90% | A100 PCIe 40GB (1555 GB/s), so about 1,400 GB/s, derived |

Three observations frame the question:

1. The datasheet's claim and the lab's copy are about 560 GB/s apart: $1838 - 1275 = 563$.
2. A different test reached about 90% of theoretical on the 40 GB card, so the hardware can get closer than a plain copy did.
3. The lab's 1275 was measured with 42 MB per direction. That is $2 \times 42 = 84$ MB of source plus destination,
   about twice the 40 MiB L2, and it sits right at the boundary where a sweep would show cache effects fade.
   Whether that number is on the plateau is not established.

The course's fact sheet lists candidate causes and marks each one unconfirmed. They are: the mix of reads and writes and
the bus turnaround between them, the overhead of ECC on HBM2e, a PCIe card limited to 300 W (against 400 W for the SXM4 module), and test
sizes too small to reach steady state. The companion course names DRAM refresh, row activation, and bus turnaround as general
properties of DRAM that a pin-rate rating ignores. ==No source ties the 66% to any one of these==, and this chapter does not
either.

What you can do is measure. The lab's sweep copies from 4 MiB to 4 GiB per tensor, from well below the L2 to far above
it, and reports three things:

- The **L2-resident** figure, at 8 MiB per tensor, where source and destination together (16 MiB) fit in the 40 MiB L2. Expect it
  to exceed the HBM rating if the cache is doing the work. The lab reports it as `l2_resident_gbs`, and you read what it shows.
- The **plateau**, the median over sizes at least twice the L2 (80 MiB and up), as `plateau_gbs`.
- The plateau as a fraction of 1935 GB/s, as `rating_fraction`.

If the plateau comes in at 1275, the companion's number was a steady-state figure. If it comes in higher, the 42 MB
measurement was still partly paying for something at its size. If it comes in lower, the 42 MB figure was
partly flattered by the cache. In every case, you have measured the thing the sources don't state, and you can tell which
way it moved.

> [!TIP] Two experiments the lab doesn't run
> A copy mixes reads and writes. If you suspect the mix, time a kernel that only reads, such as the sum reduction in
> [chapter 6](/c/06-reductions), and one that only writes, such as `dst.fill_(1.0)`, and compare each with the copy at the same size. A read-only kernel
> that gets near 1838 GB/s would point at the read-write mix and not at the card. Treat that as a hypothesis the
> measurement tests, not a conclusion this chapter has drawn.

## What goes wrong

The lab exposes these mistakes, and each one produces a number that looks like a result.

- **Counting $N$ instead of $2N$.** The copy ran at 1275 GB/s and you report 638. Count every byte read and written.
- **Timing a cached buffer.** A copy of 8 MiB per tensor fits in the L2, so repeated runs read the cache and
  report a bandwidth above the HBM rating. Use sizes well above 40 MiB, or `cuda_time(..., flush_l2=True)`, when the question is about DRAM.
- **Comparing with the rating.** A kernel at 1275 GB/s on this card is at its measured ceiling. Judging it against 1935 sends you
  looking for a speedup that the copy can't have.
- **A `float4` pointer that isn't 16-byte aligned.** A sliced view such as `x[1:]` crashes the kernel with a
  misaligned-address error. Check the address, and fall back to the scalar kernel.
- **Forgetting the tail.** With $n = 1{,}000{,}003$, dividing by 4 leaves 3 floats. A kernel that copies only $n / 4$ vectors
  leaves them uncopied, and a test at a multiple of 4 never notices.
- **32-bit index overflow.** `i * stride` wraps for large $n$ and reads the wrong element. Use `long long`.
- **Blaming the kernel for a launch-sized time.** A copy of 4 MiB per tensor moves 8.4 MB, which at 1275 GB/s is
  6.6 microseconds. Small sizes give noisy, low bandwidth, and the sweep's smallest entries show that.
- **One block, or too few.** A launch of fewer than about two blocks per SM, under 216 blocks of 256 threads on this
  card, can't fill 108 SMs. The grid-stride kernel needs `blocks >= 1` to be correct, and about $108 \times 8$ to be fast.

> [!RECAP]
> - The A100 has registers, shared memory and L1, a 40 MB L2, and 80 GB of HBM. The measured latencies are 29.0 (shared), 261.5 (L2), and 466.3 (HBM) cycles on a 40 GB card.
> - Memory moves in 32-byte sectors. A coalesced warp load touches 4 sectors, a stride of 8 floats or more touches 32, and the measured 6.9x strided penalty splits into 4.5x wasted bytes and 1.53x extra request cost.
> - Bandwidth is reads plus writes over event time. A copy of $N$ bytes moves $2N$, and a vector add moves $3N$.
> - A `float4` load needs a 16-byte aligned address and a tail loop. Wider loads and grid-stride loops keep the roughly 640 KB of bytes in flight that Little's law asks for.
> - The 1275 GB/s copy is 66% of the 1935 GB/s rating, and the datasheet claims 95%. No source explains the gap, so you report the plateau you measure.

## Check your understanding

> [!QUESTION] A warp of 32 lanes each loads one float at a stride of 4 floats from a 128-byte-aligned base. How many sectors does it fetch, and what is the efficiency?
> The addresses are 16 bytes apart, so the 32 lanes span $31 \times 16 + 4 = 500$ bytes. Every 32-byte sector in that span holds at
> least one lane's value, so the warp fetches $\lceil 500 / 32 \rceil = 16$ sectors, which is 512 bytes. Efficiency is $128 / 512 = 25\%$.

> [!QUESTION] You time a kernel that copies a 1 GiB tensor in 1.70 ms. What bandwidth do you report, and what fraction of the 1275 GB/s ceiling is it?
> A copy moves $2 \times 2^{30} = 2{,}147{,}483{,}648$ bytes. Divided by 1.70 ms that is $2.147 \times 10^{9} / 1.70 \times 10^{-3} = 1263$ GB/s,
> or 99% of 1275. Reporting only $2^{30}$ bytes would give 632 GB/s and a false alarm.

> [!QUESTION] Why does the lab's `copy_vec4` check the address modulo 16 before it launches?
> A `float4` access at an address that is not a multiple of 16 raises a misaligned-address error and kills the process. A view like `x[1:]` starts 4 bytes past
> an aligned address, so the check routes it to the scalar kernel, which needs only 4-byte alignment.

> [!QUESTION] How many threads per SM have to keep one 4-byte load in flight to cover HBM, and how does `float4` change that?
> Little's law gives $1.935 \times 10^{12} \times 330.7 \times 10^{-9} / 108 \approx 5{,}925$ bytes in flight per SM. At 4 bytes per thread that is about 1,480
> threads, 72% of the 2,048 maximum. At 16 bytes per thread it is about 370 threads, 18%. The estimate uses the 40 GB card's latency, so treat it as a prediction.

> [!QUESTION] A grid-stride copy launched with 1 block gives the right answer but a tiny bandwidth. Is the kernel wrong?
> No. One block of 256 threads on one SM has about 256 loads in flight, and the other 107 SMs sit idle. The loop is correct for any grid size, which is why it is
> useful for debugging, and fast only when the grid fills the card, such as $108 \times 8$ blocks.

> [!QUESTION] The strided copy reads a footprint of 16.8 MB at $n = 2^{24}$. Why might its 6.9x penalty understate the cost of striding?
> The footprint fits in the 40 MB L2, so after warm-up the strided reads can hit the cache. A stride whose footprint exceeds L2 must fetch from HBM every time, so it should
> do worse than 6.9x. The lab's `strided_big_gbs` at a 64 MiB footprint measures that case.

> [!QUESTION] Your sweep plateaus at 1290 GB/s. The datasheet says 95% efficiency. What can you say about the cause of the gap?
> Nothing definite. Your plateau is 67% of 1935, and the datasheet's figure would be 1,838 GB/s. The course's sources list candidates, including the read-write
> mix, ECC, a 300 W power limit, and small test sizes, but none is confirmed. You can test one by comparing a read-only kernel and a write-only kernel with the copy at the same size.

## Lab

> [!TRY] Bandwidth probe
> Write four copy kernels in CUDA C++: scalar (one element per thread), `float4` vectorized (with the alignment fallback and
> the tail), grid-stride, and strided. Write four Python functions too: `bytes_moved`, `sector_bytes_per_element`,
> `plateau_gbs`, and `time_copy_gbs`.
>
> Passing means every kernel matches PyTorch at sizes that aren't multiples of the block size or of 4 (including 1), the `float4` kernel
> survives a misaligned view, and the three coalesced kernels each reach at least half of the 1275 GB/s copy ceiling. The strided kernel must be at least
> twice as slow as the coalesced one at $2^{24}$ floats.
>
> The lab reports `scalar_gbs`, `vec4_gbs`, `grid_stride_gbs`, `strided_gbs`, `strided_big_gbs`, `coalescing_ratio`, `l2_resident_gbs`,
> `plateau_gbs`, and `rating_fraction`, and it prints the whole sweep from 4 MiB to 4 GiB. Read the table beside the
> chapter's: where does the bandwidth stop growing, and how does the plateau compare with 1275 and with the datasheet's 1838?
>
> The first run compiles CUDA with `nvcc` and takes 30 to 60 seconds. Open `content/labs/03-bandwidth-probe/starter.py`.

## Further reading

- [CUDA C++ Best Practices Guide: Memory Optimizations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html), NVIDIA. The 32-byte sector coalescing rule, with aligned and misaligned examples.
- [How to Access Global Memory Efficiently in CUDA C/C++ Kernels](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/), Mark Harris, NVIDIA, 2013. Offset and stride experiments on Tesla C870, C1060, and C2050, so the shapes carry over and the numbers don't.
- [CUDA Pro Tip: Increase Performance with Vectorized Memory Access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/), Justin Luitjens, NVIDIA, 2013. The `float4` copy, the alignment rule, and the tail.
- [CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/), Mark Harris, NVIDIA, 2013. The grid-stride pattern.
- [Benchmarking and Dissecting the Nvidia Hopper GPU Architecture](https://arxiv.org/abs/2402.13499), Luo et al., 2024. The A100 PCIe 40GB latencies and throughputs used in this chapter.
- [NVIDIA A100 Tensor Core GPU Architecture](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf), NVIDIA, 2020. The L2 structure, the 5,120 B per clock figure, and the L2 residency controls.
- [CUDA C++ Programming Guide, L2 cache control](https://docs.nvidia.com/cuda/cuda-c-programming-guide/04-special-topics/l2-cache-control.html), NVIDIA. The persistence API.
- [Nvidia's H100: Funny L2, and Tons of Bandwidth](https://chipsandcheese.com/p/nvidias-h100-funny-l2-and-tons-of-bandwidth), Chips and Cheese, 2023. Measured L2 and DRAM bandwidth, with a method different from Luo's.
- [Better Performance at Lower Occupancy](https://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf), Vasily Volkov, 2010. Bytes in flight and instruction-level parallelism.
- The companion course, [Your first CUDA kernel](https://qwen.ifkash.dev/c/12-your-first-cuda-kernel) and [The roofline](https://qwen.ifkash.dev/c/10-roofline). The 6.9x derivation and the 1275 GB/s copy measurement.
