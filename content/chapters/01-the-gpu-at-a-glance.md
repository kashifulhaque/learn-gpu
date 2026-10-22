---
title: The GPU at a glance
slug: 01-the-gpu-at-a-glance
part: "Part 1 — The machine"
summary: The anatomy of a datacenter GPU from the chip down to one SM, how its peak FLOP/s and bandwidth follow from that anatomy, and how Ampere, Hopper, and Blackwell change the numbers.
minutes: 90
gpu: true
objectives:
  - "Name the parts of a datacenter GPU, from the GPC and the streaming multiprocessor down to the sub-partition, and say what each one does."
  - "Derive the A100's peak FLOP/s for each precision from SM count, unit counts, and clock, and check the results against the datasheet."
  - "Derive the A100's HBM bandwidth from the stack count, bus width, and pin rate."
  - "Compute the ridge point for each precision on the A100, H100, and B200, and explain why it rises across generations."
  - "Say where a GPU's FLOPs come from, and where its bytes come from."
  - "Describe what Hopper (TMA, wgmma, clusters, FP8) and Blackwell (tcgen05, tensor memory, FP4, two dies) add, and which later chapters build on each."
  - "Read a spec sheet critically: dense versus sparse, preliminary versus final, and per-GPU versus per-system figures."
lab: 01-device-peaks
---

# The GPU at a glance

> [!TLDR]
> - A datacenter GPU is a few hundred small processors, the streaming
>   multiprocessors (SMs), sharing one L2 cache and one stack of HBM memory. The
>   A100 80GB PCIe has 108 SMs, a 40 MiB L2, and five HBM2e stacks.
> - Peak FLOP/s is SMs times FLOP per SM per clock times the clock. The
>   A100's 312 TFLOP/s for bf16 is $108 \times 2048 \times 1.41\ \text{GHz}$,
>   and almost all of it comes from the tensor cores: the CUDA cores reach 19.5.
> - Peak bandwidth is the HBM bus width times the pin rate: 5120 bits at 3.02
>   Gb/s per pin is 1935 GB/s. Each level of the memory hierarchy above HBM is
>   faster and smaller, and each link below it is slower.
> - The ridge point, peak FLOP/s over peak bandwidth, is 161 FLOP per byte on
>   the A100 and rises on newer parts (295 on the H100, 281 on the B200 in bf16),
>   because arithmetic has grown faster than memory.
> - Hopper adds asynchronous copies, warpgroup matrix instructions, and
>   clusters. Blackwell adds tensor memory, FP4, and a second die. Read every
>   headline number for its precision, its sparsity, and its power limit.

You write PyTorch, so a GPU looks like a device that runs `a @ b` quickly. Before
you write a kernel, you need a sharper picture: how many processors there are,
how they are grouped, where the data sits, and what each part can do per clock.
That picture is what lets you say, before you run anything, how fast a kernel
can go.

This chapter builds the picture from the top down, and then turns it into
numbers. You start with the chip, zoom into one SM, and derive the A100's rated
peaks from its parts. A rating on a datasheet stops being a magic constant once
you can rebuild it from the hardware. Then you take the same steps for the H100
and the B200, and see which part of each generation's gain came from where.

You use these numbers in every later chapter. The ridge point from this chapter
is the first thing you compute for any kernel, and the structure of the SM is
what [the execution model](/c/02-the-execution-model) and
[tensor cores](/c/11-tensor-cores) refer back to.

## Before you start

- **FLOP and FLOP/s.** A *FLOP* is one floating-point operation, an add or a
  multiply. A fused multiply-add (FMA) computes $a \times b + c$ in one
  instruction and counts as 2 FLOP. Every rate in this chapter uses that
  convention.
- **Warp.** A *warp* is 32 threads that execute the same instruction together.
  The SM schedules warps, not single threads. [Chapter 2](/c/02-the-execution-model)
  covers warps in full.
- **Dense and sparse peaks.** NVIDIA's tables often print a second, larger figure
  "with sparsity", which assumes a 2:4 structured-sparse weight matrix. A *dense*
  figure assumes no such structure, and it is the one this course uses. The dense
  value is half the sparse value.
- **Decimal and binary units.** Bandwidth is decimal, so 1935 GB/s is
  $1.935 \times 10^{12}$ bytes per second. Cache and memory sizes are binary where
  the hardware's are: a 40 MiB L2 is $40 \times 2^{20}$ bytes.
- **The two rates and the ridge point** from [chapter 0](/c/00-what-you-are-building):
  a peak arithmetic rate $\hla{C}$, a peak memory rate $\hlb{\beta}$, and their
  ratio.

## The chip, top-down

This section answers one question: what parts does a datacenter GPU have, and
how many of each? You can read the answer off the A100's die, and the other
generations reuse the same vocabulary.

A GPU is a hierarchy. The top of it is a set of *graphics processing clusters*
(GPCs), each holding a few *texture processing clusters* (TPCs), each holding two
*streaming multiprocessors* (SMs). For compute, the GPC and TPC matter little.
They are how the die is tiled and how the hardware that hands out work is wired.
The SM is the unit you program around.

NVIDIA's [Ampere architecture blog](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)
gives the counts. The full GA100 die has 8 GPCs, 8 TPCs per GPC, and 2 SMs per
TPC: $8 \times 8 \times 2 = 128$ SMs. The A100 product enables 7 GPCs and 108 of
those SMs, so 20 of the 128 are switched off. The same pattern repeats on later parts: the H100 SXM5 enables 132 SMs
and the B200 enables 148.

Around the SMs sit the memory system and the links:

| Part | What it does | A100 80GB PCIe |
|---|---|---|
| SMs | Run your threads. Each has its own register file, shared memory, and execution units. | 108 |
| L2 cache | One cache, shared by every SM, between the SMs and HBM. | 40 MiB |
| Memory controllers | Move data between L2 and the HBM stacks. | 10, each 512 bits wide |
| HBM stacks | The main memory, stacked DRAM dies beside the GPU die. | 5 stacks, 5120 bits in all |
| Host link | Carries data to and from the CPU. | PCIe Gen4 x16, 31.5 GB/s per direction |
| GPU link | Carries data to and from other GPUs. | NVLink bridge between two cards, 300 GB/s per direction |

Two details in the table are worth a second look. The 10 memory controllers
each drive a 512-bit slice of the bus, which matches the 5 stacks of 1024
bits: $10 \times 512 = 5120$ bits. And the NVLink and PCIe
figures are per direction. NVIDIA prints the sums of both directions, 600 and
64 GB/s, so the per-direction values here are those figures divided by 2. The
PCIe value is a direct calculation: 16 GT/s per lane, times $128/130$ for the
line code, times 16 lanes, divided by 8 bits per byte, is 31.5 GB/s.

```viz
01-chip-anatomy
```

The figure lets you flip between the three generations and between the whole
chip and one SM. Notice how little the structure changes: more SMs, more HBM
stacks, a bigger L2, and fatter units inside each SM.

Two more facts frame everything that follows. First, the SMs don't talk to each
other directly, with one exception on Hopper and later that you meet below.
They communicate by writing to memory that another SM reads, usually L2 or HBM.
Second, a hardware scheduler that NVIDIA calls the GigaThread Engine hands
thread blocks to SMs as they free up. You never place a block on an SM by hand.
[Chapter 2](/c/02-the-execution-model) explains how that works and what it
costs when the work doesn't divide evenly.

## Inside one streaming multiprocessor

The SM is where arithmetic happens, so this section answers: what units does one
SM contain, and how are they organized?

An A100 SM is split into four identical *sub-partitions*, which NVIDIA's
microbenchmark literature also calls sub-cores. Each sub-partition has its own
warp scheduler, its own slice of the register file, its own CUDA cores, and its
own tensor core. The four share the SM's L1 cache and shared memory.

Here is what one A100 sub-partition holds, with the per-SM total in parentheses:

| Unit | Per sub-partition | Per SM | What it does |
|---|---|---|---|
| Warp scheduler | 1 | 4 | Picks a ready warp each clock and issues its next instruction. |
| Register file | 64 KB (16,384 32-bit registers) | 256 KB (65,536 registers) | Holds the live values of every resident thread. |
| FP32 CUDA cores ("lanes") | 16 | 64 | Float add, multiply, and FMA. |
| INT32 cores | 16 | 64 | Integer arithmetic and address calculation. |
| FP64 cores | 8 | 32 | Double-precision arithmetic. |
| Tensor core | 1 | 4 | Multiplies small matrix tiles. |

The per-SM figures for FP32, INT32, FP64, and the register file are in the
[A100 whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf).
The per-sub-partition column divides them by 4, which assumes the four
sub-partitions are identical, as NVIDIA's descriptions say. Each SM also has
load/store units that move data to and from memory, and special function units
(SFUs) that compute transcendental functions such as $e^x$, $\sin x$, and
reciprocals. Neither the whitepaper's text nor this course needs their counts,
but the SFU matters later: softmax's exponentials run there, and
[chapter 15](/c/15-softmax-and-normalization) returns to it.

Notice the register file. At 256 KB per SM it is bigger than the L1 and shared
memory array, which is 192 KB. Across the whole A100 the register files hold
$108 \times 256\ \text{KB} = 27{,}648\ \text{KB} = 27\ \text{MiB}$, against 20.25
MiB of L1 and shared memory and 40 MiB of L2. The registers are the largest
fast memory on the chip and the one a kernel controls most directly. That is why
the fastest kernels in later chapters keep their working data in registers.

### How a warp uses a sub-partition

A warp has 32 threads, but a sub-partition has 16 FP32 lanes. So a warp's FP32
instruction needs two clocks of the lane group: $32 / 16 = 2$. Meanwhile the
scheduler can issue instructions from other warps to the other units, so the
integer, FP64, load/store, and tensor units are not idle during those two clocks.
Keeping all of these busy needs many warps resident at once, which is the subject
of [the execution model](/c/02-the-execution-model).

> [!NOTE] Sub-partitions are a throughput detail, not a programming model
> You never address a sub-partition from CUDA. The hardware assigns each resident
> warp to one of the four, and a block of 128 threads or more spreads across all
> four. The detail matters here because the peak rates below are sums over
> sub-partitions.

### The L1 cache and shared memory

The SM has one 192 KB array that serves two roles. A kernel that doesn't use
shared memory gets almost all of it as L1 cache. A kernel that declares shared
memory carves out part of it: the A100 supports carveouts of 0, 8, 16, 32, 64,
100, 132, and 164 KB, and the remainder is L1. One block can use at most 163 KB,
because CUDA reserves 1 KB per block.

Shared memory is organized in 32 banks, each serving 32 bits per clock, so one SM
can move $32 \times 4 = 128$ bytes per clock from it. Across the A100 that is

$$
128\ \text{B/clock} \times 108\ \text{SMs} \times 1.41 \times 10^{9}\ \text{clock/s} = 19.5\ \text{TB/s}.
$$

That is the theoretical bandwidth of all the shared memory on the chip, ten times
HBM's 1.935 TB/s. [Chapter 8](/c/08-shared-memory-and-transpose) shows how a
kernel loses most of it to bank conflicts.

You can read the SM's limits from the device. This CUDA C++ program prints the
fields this chapter uses:

```cpp
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("%s, compute capability %d.%d\n", p.name, p.major, p.minor);
    printf("SMs: %d\n", p.multiProcessorCount);
    printf("L2 cache: %.1f MiB\n", p.l2CacheSize / 1048576.0);
    printf("registers per SM: %d\n", p.regsPerMultiprocessor);
    printf("max threads per SM: %d\n", p.maxThreadsPerMultiProcessor);
    printf("shared memory per SM: %zu KiB\n", p.sharedMemPerMultiprocessor / 1024);
    return 0;
}
```

The same facts are available from PyTorch, which is what the lab uses:

```python
import torch

props = torch.cuda.get_device_properties(0)
print(props.name, props.major, props.minor)  # for example: ... 8 0
print(props.multi_processor_count)            # 108 on the course's A100
print(props.L2_cache_size / 2**20)            # L2 in MiB
```

Notice what these calls don't report: the clock, the number of tensor cores, and
how wide they are. The device properties give you the SM count and the limits;
the widths and the clock come from the whitepaper and the datasheet. Deriving a
peak therefore takes both sources, which is what the next section does and what
your lab does in code.

## Where the FLOPs come from

This section answers: how does the A100 reach 312 TFLOP/s, and why does a plain
fp32 kernel get only 19.5?

Peak arithmetic rate is a product of three things: how many SMs there are, how
many FLOP each SM completes per clock, and how many clocks happen per second.

$$
\boxed{\hla{C} = \hlc{N_{\text{SM}}} \times \hld{r} \times f}
$$

Here $\hla{C}$ is the peak rate in FLOP/s, $\hlc{N_{\text{SM}}}$ is the SM count,
$\hld{r}$ is FLOP per SM per clock, and $f$ is the clock in Hz. The SM count and the
clock are one number each. The interesting term is $\hld{r}$, and it depends on
which unit does the work.

### The CUDA cores

Start with the simple case. Each of an SM's 64 FP32 cores does one FMA per clock,
which is 2 FLOP. So the FP32 rate per SM is $64 \times 2 = 128$ FLOP per clock.
At the A100's boost clock of 1410 MHz:

> [!EXAMPLE] A100 FP32 peak
> $$108 \times 128 \times 1.41 \times 10^{9} = 1.949 \times 10^{13}\ \text{FLOP/s} = 19.49\ \text{TFLOP/s}$$
> The datasheet rates 19.5. The derivation matches the rating after rounding.

The same method gives the FP64 rate on the CUDA cores. There are 32 FP64 cores per
SM, so $r = 32 \times 2 = 64$, and $108 \times 64 \times 1.41 \times 10^{9} = 9.75$
TFLOP/s against a rating of 9.7.

### The tensor cores

Now the unit that matters for machine learning. The whitepaper states that each
A100 tensor core can execute 256 FP16 FMA operations per clock. That is 512 FLOP
per clock per tensor core, and each SM has four:

$$
\hld{r_{\text{bf16}}} = 4 \times 256 \times 2 = 2048\ \text{FLOP per SM per clock}
$$

The rate for bf16 is the same as for fp16. Multiply through:

> [!EXAMPLE] A100 bf16 tensor-core peak
> $$108 \times 2048 \times 1.41 \times 10^{9} = 3.119 \times 10^{14}\ \text{FLOP/s} = 311.9\ \text{TFLOP/s}$$
> The datasheet rates 312 for bf16 and fp16.

The other precisions are ratios of this one. TF32 runs at half the rate, 1024
FLOP per SM per clock, which gives $108 \times 1024 \times 1.41 \times 10^{9} =
155.9$ TFLOP/s against a rating of 156. INT8 runs at twice the rate, 4096 per SM
per clock, which gives 623.7 TOPS against a rating of 624. The whitepaper doesn't
print the per-clock TF32 and INT8 rates in the text this course read; they are the
datasheet's ratios applied to the bf16 rate, and the products match the ratings.

Here is every A100 peak, rebuilt and compared:

| Quantity | $r$ (FLOP per SM per clock) | Derived | Rated |
|---|---|---|---|
| bf16 or fp16 tensor | 2048 | 311.9 TFLOP/s | 312 |
| TF32 tensor | 1024 | 155.9 TFLOP/s | 156 |
| INT8 tensor | 4096 | 623.7 TOPS | 624 |
| FP64 tensor | 128 | 19.49 TFLOP/s | 19.5 |
| FP32 CUDA cores | 128 | 19.49 TFLOP/s | 19.5 |
| FP64 CUDA cores | 64 | 9.75 TFLOP/s | 9.7 |

A derivation that matches six ratings is evidence that you have the structure
right: 108 SMs, four tensor cores each, 1410 MHz. You can reuse the method on any
GPU whose whitepaper gives those three things.

### The consequences

The table says something blunt. ==The tensor cores are 94% of the A100's
arithmetic==: $2048$ FLOP per SM per clock for bf16 against $128$ for FP32 is a
ratio of 16, and $128 / 2048 = 6.25\%$. A kernel that does its matrix math on the
CUDA cores, in any precision, can reach at most 6.25% of the headline number.
Most of this course's matmul work is about getting data to the tensor cores fast
enough that they stay busy, and [chapter 11](/c/11-tensor-cores) shows how to
call them.

> [!WARNING] An fp32 matmul in PyTorch doesn't use the 156 TFLOP/s TF32 path by default
> `torch.backends.cuda.matmul.allow_tf32` is `False` by default, so `a @ b` on two
> float32 tensors runs on the CUDA cores, at a ceiling of 19.5 TFLOP/s, not on the
> tensor cores at 156. Your lab times an fp32 matmul with TF32 off to show you the
> lower ceiling. A bf16 matmul runs on the tensor cores.

The tensor-core rate also tells you how a single instruction behaves. The Ampere
instruction `mma.sync.m16n8k16` multiplies a 16 by 16 tile by a 16 by 8 tile. That
is $16 \times 8 \times 16 = 2048$ FMA, or 4096 FLOP. One tensor core at 256 FMA per
clock needs $2048 / 256 = 8$ clocks to finish one, while Luo and colleagues
measured a latency of 24.6 to 26.3 cycles on an A100 PCIe 40GB. So a tensor core
needs roughly three such instructions in flight to stay full, since $25.5 / 8
\approx 3$. This is the same latency-hiding argument as for memory, and
[chapter 4](/c/04-roofline-and-latency) makes it precise.

### The clock

The clock enters the peak linearly, so a card that can't hold its boost clock
can't reach its rating. The A100 PCIe has a 300 W limit and the SXM4 module has
400 W, yet NVIDIA's datasheet prints identical compute peaks for both. The
datasheet doesn't say how far the PCIe card's sustained clock falls under a heavy
tensor load, so treat 312 TFLOP/s as a ceiling that the PCIe card may sit below.
The lab can also hand you either card, and it prints which one you got.

> [!KEY] Peak FLOP/s is a product, and each factor is a lever
> $C = N_{\text{SM}} \times r \times f$. Each generation raises it by adding SMs,
> by widening the per-SM rate, or by raising the clock. When you read a spec sheet,
> factor the headline number into these three and check it.

## Where the bytes come from

This section answers the matching question for memory: how does the A100 reach
1935 GB/s, and how fast are the levels above and below HBM?

HBM is a stack of DRAM dies on a silicon base, placed beside the GPU die and
connected to it by a very wide bus. Bandwidth is the bus width times the data rate
of each wire. Each HBM stack has a 1024-bit interface, and the A100 80GB has five
of them:

$$
W = 5 \times 1024 = 5120\ \text{bits}
$$

The 80GB card's memory moves data at a rate that NVIDIA doesn't print per pin, but
you can solve for it. Write the bandwidth as $\hlb{\beta} = W \times p / 8$, where
$p$ is the pin rate in Gb/s and the 8 converts bits to bytes. Then:

$$
p = \frac{8\hlb{\beta}}{W} = \frac{8 \times 1935}{5120} = 3.02\ \text{Gb/s per pin}
$$

That is 1.512 GHz double data rate, which is a plausible HBM2e figure. The 40GB
A100's HBM2 runs slower, at 1215 MHz DDR per the whitepaper, and that is why its
bandwidth is 1555 GB/s instead of 1935.

Each stack therefore delivers $1935 / 5 = 387$ GB/s, and each SM's share of the
whole is $1935 \times 10^{9} / 108 = 17.9$ GB/s, or $17.9 \times 10^{9} / 1.41
\times 10^{9} = 12.7$ bytes per clock. Hold on to that last number. An SM can do
2048 tensor FLOP per clock and gets 12.7 bytes per clock from HBM.

### The bandwidth ladder

HBM is one rung of a ladder. The table puts the rungs of the A100 side by side,
fastest first. Latencies are from Luo and colleagues' microbenchmarks on an A100
PCIe 40GB at 1410 MHz; the 80GB card is not measured in that source but is
expected to be close.

| Level | Capacity | Bandwidth | Latency |
|---|---|---|---|
| Registers | 256 KB per SM, 27 MiB in all | Operands feed the units directly | Not measured here |
| Shared memory | Up to 164 KB per SM | 19.5 TB/s theoretical, whole chip | 29.0 cycles, 20.6 ns |
| L1 cache | Rest of 192 KB per SM | 128 B/clock per SM theoretical; 99.5 to 106.8 measured | 37.9 cycles, 26.9 ns |
| L2 cache | 40 MiB | 7.2 TB/s theoretical; 2.6 to 2.8 TB/s measured | 261.5 cycles, 185.5 ns |
| HBM | 80 GB | 1.935 TB/s rated; 1.275 TB/s measured on a copy | 466.3 cycles, 330.7 ns |
| NVLink bridge | other GPU's memory | 0.3 TB/s per direction | Not measured here |
| PCIe Gen4 x16 | host memory | 0.0315 TB/s per direction | Not measured here |

The L2 row hides a disagreement worth knowing about. The whitepaper's theoretical
figure is 5120 bytes per clock, which is $5120 \times 1.41 \times 10^{9} = 7.2$
TB/s. Luo and colleagues measured 1853.7 to 2007.9 bytes per clock, which is 2.6
to 2.8 TB/s. A third source, Chips and Cheese, measured 4.5 TB/s on an A100 SXM.
The methods and cards differ, so quote the one you used and name it. Whichever you
pick, L2 sits between shared memory and HBM, and its latency is 261.5 cycles
against HBM's 466.3.

> [!NOTE] The measured copy is 66% of the rating
> The datasheet describes a 95% DRAM utilization efficiency, but a plain
> device-to-device copy on the course's A100 reaches 1275 GB/s, which is
> $1275 / 1935 = 66\%$ of the rating (measured in learn-inference, [chapter 10](https://qwen.ifkash.dev/c/10-roofline)).
> No source this course read explains the gap. [Chapter 3](/c/03-memory-and-bandwidth)'s lab sweeps the size and
> reports where the copy plateaus. Judge a memory-bound kernel against 1275, not 1935.

Two ratios from the ladder are worth remembering. Shared memory's theoretical
19.5 TB/s is $19.5 / 1.935 = 10.1$ times HBM's rating, which is why you stage
reused data there. And HBM is $1935 / 31.5 = 61$ times the PCIe link, which agrees
with the roughly 60 times that learn-inference reports for a host-to-device copy
(it measured about 21 GB/s there). Moving a tensor from CPU memory costs
about as much as reading it from HBM sixty times over.

### Where FLOPs and bytes come from

Put the two derivations next to each other:

| | Peak FLOP/s | Peak bytes/s |
|---|---|---|
| Built from | SMs $\times$ FLOP per SM per clock $\times$ clock | Bus width $\times$ pin rate $\div$ 8 |
| A100 | $108 \times 2048 \times 1.41\ \text{GHz}$ | $5120\ \text{bits} \times 3.02\ \text{Gb/s} \div 8$ |
| Lives | In the SMs, spread across the die | In the HBM stacks, at the edge of the package |
| Grows by | More SMs, wider units, higher clock | More stacks, wider bus, faster pins |

The two sides scale independently, with different physics, and that is the origin
of everything in the next section. The die can put more arithmetic on every
square millimetre as the process shrinks. The memory bus grows by adding stacks
and raising the pin rate, both of which are limited by package area and by
signal integrity.

## The ridge point

This section answers: given the two peaks, how much arithmetic must a kernel do per
byte to use both?

Divide the arithmetic rate by the memory rate. The result is the *ridge point*,
the arithmetic intensity at which a kernel stops being limited by memory and
starts being limited by arithmetic:

$$
\boxed{I^{\ast} = \frac{\hla{C}}{\hlb{\beta}}}
$$

The A100's ridge points follow from the peaks you derived. Each row divides a peak
by 1.935 TB/s:

| Precision and unit | $\hla{C}$ (TFLOP/s) | Ridge point (FLOP per byte) |
|---|---|---|
| INT8 tensor core | 624 | 322.5 |
| bf16 or fp16 tensor core | 312 | 161.2 |
| TF32 tensor core | 156 | 80.6 |
| FP32 CUDA cores | 19.5 | 10.1 |
| FP64 CUDA cores | 9.7 | 5.0 |

The lab prints derived peaks, such as 311.9 for bf16, so its ridge points can
differ from this table in the last digit.

The ridge point depends on the precision because the numerator does. Lower
precision raises $\hla{C}$ without raising $\hlb{\beta}$, so a lower-precision
kernel needs more arithmetic per byte to be compute bound. Moving a kernel from
fp32 to bf16 doesn't just speed it up: it moves its ridge point from 10 to 161
FLOP per byte.

There's a per-SM way to see the same number. An SM does 2048 bf16 FLOP per clock
and gets 12.7 bytes per clock from HBM, and $2048 / 12.7 = 161$. In other words,
each byte the SM receives must feed 161 FLOP of tensor math. ==Everything an SM does
with a byte in a register, in a tile in shared memory, or in L2 is an attempt to
reuse it 161 times==, and the following chapters are about how.

### A worked example: one token against a weight matrix

Take a matrix multiplication with $m$ rows of activations and a weight matrix of
$d \times d$ in bf16, 2 bytes per element. It does $2md^2$ FLOP. The best case
traffic reads the weights once, $2d^2$ bytes, and reads and writes the activations,
$4md$ bytes. When $d$ is large the weights dominate, and the intensity is about

$$
I \approx \frac{2md^2}{2d^2} = m\ \text{FLOP per byte}
$$

So for a decode step that processes one token, $m = 1$ and the intensity is about
1 FLOP per byte. The roofline says the attainable rate is $1 \times 1935\ \text{GB/s}
= 1.935$ TFLOP/s, which is $1.935 / 312 = 0.62\%$ of the bf16 peak. To reach the ridge
point at 161 FLOP per byte, you need about 161 tokens sharing one read of the
weights. This is the quantitative reason that batching helps inference, and
[chapter 4](/c/04-roofline-and-latency) builds on it.

> [!EXAMPLE] A square GEMM and the ridge point
> A square bf16 GEMM with $m = n = k$ does $2n^3$ FLOP and, at best, moves
> $3n^2 \times 2 = 6n^2$ bytes: read $A$ and $B$, write $C$. Its intensity is $n/3$.
> It reaches the A100's ridge point of 161.2 when $n = 3 \times 161.2 = 484$. A
> 4096 by 4096 GEMM has an intensity of $4096/3 = 1365$, far above the ridge, so
> it is compute bound if its traffic is anywhere near the best case. Real kernels
> re-read tiles, so their intensity is lower than the ideal.

```viz
01-peaks-ridge
```

Use the figure to move the intensity along the x axis. At 1 FLOP per byte every
GPU is a small fraction of its peak, and the plot shows how far to the right each
ridge sits.

## Three generations

This section answers: what changed from the A100 to the H100 to the B200, and
which changes do later chapters rely on?

Here is the same structure, filled in for each part. The H100 column is the SXM5
module, the B200 column is the HGX B200 per GPU, and every compute figure is dense.

| | A100 80GB PCIe | H100 SXM5 | B200 (HGX) |
|---|---|---|---|
| Architecture, compute capability | Ampere, 8.0 | Hopper, 9.0 | Blackwell, 10.0 |
| SMs | 108 | 132 | 148 (74 per die) |
| bf16 tensor FLOP per SM per clock | 2048 | 4096 | 8192 |
| bf16 tensor peak, dense | 312 TFLOP/s | 989.4 | 2250 |
| FP8 tensor peak, dense | none | 1978.9 | 4500 |
| FP4 tensor peak, dense | none | none | 9000 |
| FP32 on the CUDA cores | 19.5 | 66.9 | 75 |
| HBM | 80 GB HBM2e, 5 stacks | 80 GB HBM3, 5 stacks | 180 GB HBM3e, 8 stacks |
| Memory bandwidth | 1935 GB/s | 3350 | 8000 |
| L2 cache | 40 MiB | 50 MB | 126 MB |
| Shared memory plus L1 per SM | 192 KB | 256 KB | 256 KB |
| Tensor memory per SM | none | none | 256 KB |
| GPU link, per direction | 300 GB/s (bridge) | 450 (NVLink 4) | 900 (NVLink 5) |

The H100 and B200 values for compute come from NVIDIA's product pages (the H100's
printed with sparsity, so halved) and from the HGX B200 totals divided by 8. The
caveats in the table matter, and the next-to-last section collects them.

### Where each gain came from

Factor the bf16 peak as $N_{\text{SM}} \times r \times f$ for each step:

| Step | SMs | $r$ per SM per clock | Clock | Product |
|---|---|---|---|---|
| A100 to H100 | $132/108 = 1.22$ | $4096/2048 = 2$ | $1.83/1.41 = 1.30$ | $3.17\times$ (989.4 / 312) |
| H100 to B200 | $148/132 = 1.12$ | $8192/4096 = 2$ | $1.856/1.83 = 1.01$ | $2.27\times$ (2250 / 989.4) |

The clocks for the H100 and B200 are not published in a document this course read.
They are the clocks implied by the rated throughput, so treat them as derived,
and the B200's as unconfirmed. Even so, the pattern is clear. ==Each generation
doubles the per-SM tensor rate==, and the rest comes from a few more SMs and, in
the H100's case, a faster clock.

Memory scales differently. The H100's bandwidth gain over the A100 is
$3350 / 1935 = 1.73\times$ with the same 5120-bit bus, so all of it comes from the
pin rate: $8 \times 3350 / 5120 = 5.23$ Gb/s against 3.02. The B200's gain over the
H100 is $8000 / 3350 = 2.39\times$, which is a wider bus ($8192 / 5120 = 1.6$) times
a faster pin ($7.81 / 5.23 = 1.49$). Memory grew by 1.73 and 2.39 while arithmetic
grew by 3.17 and 2.27, so the ridge point moves.

### Ridge points across the generations

Apply $I^{\ast} = C / \beta$ to each part and precision:

| GPU and precision | $C$ (TFLOP/s) | $\beta$ (TB/s) | Ridge point (FLOP per byte) |
|---|---|---|---|
| A100, bf16 | 312 | 1.935 | 161.2 |
| H100, bf16 | 989.4 | 3.35 | 295.3 |
| B200, bf16 | 2250 | 8.0 | 281.3 |
| A100, TF32 | 156 | 1.935 | 80.6 |
| H100, TF32 | 494.7 | 3.35 | 147.7 |
| H100, FP8 | 1978.9 | 3.35 | 590.7 |
| B200, FP8 | 4500 | 8.0 | 562.5 |
| B200, FP4 | 9000 | 8.0 | 1125 |
| A100, FP32 CUDA cores | 19.5 | 1.935 | 10.1 |
| H100, FP32 CUDA cores | 66.9 | 3.35 | 20.0 |

The bf16 ridge nearly doubles from the A100 to the H100, from 161 to 295, and
then falls slightly to 281 on the B200, whose bandwidth grew a little faster than
its arithmetic that step. The low-precision rows are the ones to notice: the
FP8 ridge on the H100 is 591 FLOP per byte, and the FP4 ridge on the B200 is 1125.
At the same bandwidth, each halving of the element size doubles the arithmetic
rate and so doubles the ridge point. A kernel that is comfortably compute bound
in bf16 can be memory bound in FP8 on the same chip.

The one-token decode example now has three answers. At 1 FLOP per byte the
attainable rate is $\beta$ itself in TFLOP/s: 1.935 on the A100, 3.35 on the H100,
and 8.0 on the B200. As fractions of each bf16 peak, those are 0.62%, 0.34%, and
0.36%. The newer parts are no better at this workload, in relative terms, because
a byte-limited kernel gains only what bandwidth gains.

### What Hopper adds

The H100 keeps the SM's shape but changes how data reaches the tensor cores. Four
additions matter, and each has a chapter that builds on it.

- **The Tensor Memory Accelerator (TMA).** A hardware unit that copies a
  multidimensional tile between global and shared memory. One thread issues a
  single `cp.async.bulk.tensor` instruction, covering tensors of one to five
  dimensions, and the copy completes through a barrier. It doesn't use registers or
  other SM instructions to move the data, which frees the rest of the SM for math.
  [Chapter 12](/c/12-pipelining-and-async-copies) covers it in prose.
- **Warpgroup matrix instructions (`wgmma`).** A warpgroup is four contiguous
  warps, 128 threads. The `wgmma.mma_async` instruction multiplies a tile whose M
  dimension is always 64, takes its B operand from shared memory, and runs
  asynchronously, so the warpgroup can do other work while it completes. Luo and
  colleagues measured an FP16 `wgmma` at 729.3 TFLOP/s of the 756.5 TFLOP/s rated
  peak (96%) on an H800 PCIe, a part with the same 114 SMs as the H100 PCIe.
- **Thread block clusters and distributed shared memory.** A cluster is a group of
  blocks that the hardware guarantees to schedule at the same time, on one GPC. Up
  to 8 blocks form a portable cluster, and 16 are possible on the H100 and B200
  with an opt-in. Blocks in a cluster can read and write each other's shared
  memory. This is the exception to "SMs don't talk to each other": on an H800,
  SM-to-SM latency was 180 cycles against 263 for L2, and peak SM-to-SM throughput
  was about 3.27 TB/s at a cluster size of 2 (Luo et al.).
- **FP8.** The tensor cores gain two 8-bit formats, E4M3 and E5M2, at twice the
  bf16 rate. The A100 has no FP8 path at all. [Chapter 9](/c/09-floating-point-on-the-gpu)
  covers the formats and [chapter 18](/c/18-quantized-kernels) their use.

Hopper also raises the shared memory per SM to 228 KB, with 227 KB for one block,
from the A100's 164 and 163.

### What Blackwell adds

The B200 changes where the matrix multiplication's result lives. Four changes
matter:

- **Tensor memory (TMEM).** Each SM has 256 KB of a memory that only the
  tensor cores use for accumulators. Its shape is 128 rows by 512 columns of 32-bit
  cells, so $128 \times 512 \times 4 = 262{,}144$ bytes. On earlier parts the
  accumulator lived in the register file, which competed with everything else for
  space.
- **`tcgen05.mma`.** A single thread issues the matrix instruction, which runs
  asynchronously and reads its operands from shared memory or tensor memory. The
  largest bf16 tile for one block is 128 by 256 by 16, and a pair of blocks can
  cooperate on 256 by 256 by 16. NVIDIA's CUTLASS documentation puts the throughput
  at 2 to 4 times Hopper's `wgmma`, depending on the data type.
- **Lower precisions.** FP6 and FP4, and block-scaled formats such as MXFP8, MXFP4,
  and NVFP4, where a shared scale accompanies each block of 16 or 32 values. FP4
  doubles the FP8 rate again, giving the table's 9000 TFLOP/s.
- **Two dies.** The B200 puts two reticle-sized dies in one package, joined by a
  link NVIDIA calls NV-HBI at 10 TB/s, and presents them as one CUDA device with
  208 billion transistors in all. The 148 SMs split as 74 per die. Chips and Cheese
  measured the cost of crossing: thread-pair latency of 90 to 100 ns inside one
  partition and 190 to 220 ns across partitions.

Blackwell also shows a ratio that has been building. In Tri Dao's FlashAttention-4
analysis, an SM's tensor cores do 8192 operations per clock while its shared memory
moves 128 bytes per clock, and its exponential unit does 16 per clock. On the
A100 the same shared-memory ratio is $2048 / 128 = 16$ FLOP per byte; on the B200
it is $8192 / 128 = 64$. The tensor cores have grown faster than the shared memory
that feeds them, which is why Hopper and Blackwell move data with TMA and keep
accumulators in tensor memory. [Chapter 16](/c/16-flash-attention) uses the exponential
figure to show how softmax competes with the tensor cores.

> [!NOTE] Where the generations are taught
> The labs run on an A100, so Hopper and Blackwell features appear here, in
> [chapter 12](/c/12-pipelining-and-async-copies) (TMA, `wgmma`, `tcgen05`), and in
> [chapter 23](/c/23-beyond-cuda) as prose, figures, and slides. When a later chapter
> says "on Hopper", it means the instruction exists on compute capability 9.0 and
> later and that your lab can't run it.

### Reading a spec sheet

Three places in this chapter's table hide a trap. All three are cases where two
documents or two products give different numbers for what looks like one GPU.

1. **Preliminary versus final.** The H100 whitepaper printed preliminary
   figures: about 1000 TFLOP/s for dense FP16 and 3 TB/s of bandwidth, which gives
   a ridge point of $1000 / 3 = 333$. NVIDIA's final product page gives 989.4 TFLOP/s
   dense and 3.35 TB/s, a ridge point of 295. Use the final figures.
2. **One name, several SKUs.** The H100 PCIe has 114 SMs, 350 W, and a rated bf16
   peak of 756.5 TFLOP/s, which corresponds to a clock of about 1.62 GHz.
   Luo and colleagues use that figure for an H800 PCIe with the same 114 SMs, and
   the final PCIe FP64 and FP32 figures come from a reseller summary, so treat
   the H100 PCIe numbers as secondary. The B200 in an
   HGX system, at about 1000 W, is rated at 2250 bf16 and 4500 FP8 TFLOP/s. NVIDIA's
   Blackwell architecture blog lists 5000 dense FP8 and 10,000 dense NVFP4 TFLOP/s
   for a version at up to 1200 W, which would put the FP8 ridge at 625 and the FP4
   ridge at 1250. This course uses the HGX values and names the second as the
   maximum-power part. B200 memory is 180 GB in HGX and cloud listings and 192 GB in
   the blog, and both are right for their SKU.
3. **Per-GPU versus per-system.** NVIDIA's HGX page prints totals for eight GPUs.
   Divide by 8 for one GPU. Its NVLink figures are sums of both directions; divide
   by 2 for one direction.

The B200's boost clock of about 1.856 GHz is itself unconfirmed: it is what the
rated 2250 TFLOP/s implies for 148 SMs at 8192 FLOP per clock.

## What goes wrong

- **Comparing a sparse figure to a dense one.** The A100 datasheet's 624 TFLOP/s
  for bf16 assumes 2:4 sparsity. A dense matmul can't reach it. Halve any table
  entry that says "with sparsity" before you compare it with a measurement.
- **Applying the tensor-core peak to a CUDA-core kernel.** A kernel that doesn't
  call tensor-core instructions, or a float32 matmul with TF32 off, has a ceiling of
  19.5 TFLOP/s, not 312. If you divide its throughput by 312 you will conclude it
  is 16 times worse than it is.
- **Using one precision's ridge point for another.** The bf16 ridge is 161 and the
  FP32 ridge is 10.1 on the same card. A kernel with an intensity of 50 is memory
  bound in bf16 and compute bound on the CUDA cores in fp32.
- **Treating the rating as the target.** Both ceilings are rated at the boost clock
  and, for memory, under ideal access. The copy measured 66% of the bandwidth
  rating, and a real GEMM reaches less than the single-instruction `mma.sync` loop
  does: Luo and colleagues measured 303.4 TFLOP/s for FP16 with FP32 accumulation,
  97% of 312, in a loop that issues nothing but the instruction.
- **Mixing card variants.** The SXM4 A100 is rated at 2039 GB/s against the PCIe
  card's 1935, a 5.4% difference. The ridge point is 153.0 for the SXM4 and 161.2 for
  the PCIe card. Print `torch.cuda.get_device_name()` with every measurement.
- **Reading `multi_processor_count` as cores.** It is the SM count, 108 here. The
  CUDA core count is that times 64, which is 6912, and the tensor core count is that
  times 4, which is 432.
- **Dropping the unit.** A peak in TFLOP/s ($10^{12}$) over a bandwidth in GB/s
  ($10^{9}$) is off by 1000 if you don't convert. 312 divided by 1935 is 0.161, not
  161.

> [!RECAP]
> - A GPU is a set of SMs sharing an L2 cache and HBM. The A100 80GB PCIe has 108
>   SMs, a 40 MiB L2, and five HBM2e stacks, and each SM has four sub-partitions.
> - Peak FLOP/s is $N_{\text{SM}} \times r \times f$. The A100's tensor cores give
>   311.9 TFLOP/s in bf16, and its CUDA cores give 19.49 in fp32.
> - Peak bandwidth is bus width times pin rate over 8: $5120 \times 3.02 / 8 = 1935$
>   GB/s, and a plain copy reaches 1275.
> - The ridge point is $C / \beta$, per precision: 161 for A100 bf16, 295 for H100
>   bf16, 281 for B200 bf16, and it doubles each time the element size halves.
> - Hopper added TMA, `wgmma`, clusters, and FP8. Blackwell added tensor memory,
>   `tcgen05`, FP4, and two dies.
> - Read a spec sheet for dense versus sparse, preliminary versus final, and per-GPU
>   versus per-system figures.

## Check your understanding

> [!QUESTION] The A100 has 108 SMs, runs at 1410 MHz, and each SM has four tensor cores doing 256 FMA per clock. What is its bf16 peak?
> Each tensor core does $256 \times 2 = 512$ FLOP per clock, and each SM does
> $4 \times 512 = 2048$. Then $108 \times 2048 \times 1.41 \times 10^{9} = 3.119
> \times 10^{14}$, which is 311.9 TFLOP/s, matching the 312 rating.

> [!QUESTION] An fp32 `a @ b` in PyTorch on an A100 with default settings is limited to what rate, and why?
> To 19.5 TFLOP/s. `allow_tf32` defaults to `False`, so the matmul runs on the FP32
> CUDA cores, which do $64 \times 2 = 128$ FLOP per SM per clock:
> $108 \times 128 \times 1.41 \times 10^{9} = 19.49$ TFLOP/s. The tensor cores would
> allow 156 TFLOP/s for TF32, but only if you enable it.

> [!QUESTION] An HBM bus is 5120 bits wide and the card's rated bandwidth is 1935 GB/s. What is the pin rate?
> $8 \times 1935 / 5120 = 3.02$ Gb/s per pin. Bandwidth is $W \times p / 8$, so
> $p = 8\beta / W$.

> [!QUESTION] Why does the ridge point of the H100 in FP8 (591) exceed its ridge point in bf16 (295)?
> FP8 doubles the tensor-core rate, 1978.9 against 989.4 TFLOP/s, while the HBM
> bandwidth stays at 3.35 TB/s. The ridge point is the ratio, so it doubles. A kernel
> needs twice the arithmetic per byte to be compute bound in FP8.

> [!QUESTION] A kernel multiplies one token's activations by a $d \times d$ bf16 weight matrix. Which ceiling limits it on an A100, and what rate does the roofline allow?
> Memory. Its intensity is about $m = 1$ FLOP per byte, far below the ridge point of
> 161. The attainable rate is $1 \times 1935$ GB/s $= 1.935$ TFLOP/s, which is 0.62%
> of 312.

> [!QUESTION] A blog post says an H100 does 1979 TFLOP/s in bf16. What should you check before comparing it with a measurement?
> Whether the figure is dense or with sparsity. NVIDIA's H100 page prints 1979 for
> bf16 with sparsity. The dense peak is half, 989.4 TFLOP/s, which is also what
> $132 \times 4096 \times 1.83\ \text{GHz}$ gives.

> [!QUESTION] The B200's bf16 peak is 2.27 times the H100's. Which factors make it up?
> The per-SM rate doubles, 8192 against 4096 FLOP per clock. The SM count grows by
> $148 / 132 = 1.12$ and the clock by about 1.01, and $2 \times 1.12 \times 1.01 =
> 2.27$. Almost all of the gain is wider tensor cores, not more of them.

> [!QUESTION] Why do the clocks for the H100 and B200 in this chapter come with a caveat?
> NVIDIA doesn't publish them in a document this course read. They are the clocks
> implied by dividing the rated throughput by SMs times FLOP per SM per clock, so they
> are derived, and the B200's is unconfirmed.

## Lab

> [!TRY] Device peaks
> Implement the formulas in `starter.py`: `flop_per_sm_clock`, `peak_tflops`,
> `hbm_bus_bits`, `pin_rate_gbps`, `ridge_point`, `roofline_ms`,
> `attainable_tflops`, `gemm_intensity`, and `compare_gpus`, plus
> `describe_local_device`, `local_peak_tflops`, and `measure_matmul_tflops`. The
> tests check that your peaks reproduce NVIDIA's ratings within 1% for the A100,
> H100, and B200, and that your ridge points match this chapter. On the GPU, your
> lab times a bf16 and an fp32 matmul and reports each as a fraction of the peak you
> derived. The roofline says neither can exceed 100%, and the lab fails a matmul that
> reaches under 30%. The measured figures are yours to report.

## Further reading

- [NVIDIA A100 Tensor Core GPU datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf):
  the rated figures this chapter rebuilds.
- [NVIDIA A100 architecture whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf):
  the SM's units, the 40 MB L2, and the tensor-core rate per clock.
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/):
  TMA, clusters, distributed shared memory, and FP8.
- [Inside NVIDIA Blackwell Ultra](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/):
  two dies, tensor memory, and NVFP4.
- [Nvidia's B200: Keeping the CUDA Juggernaut Rolling](https://chipsandcheese.com/p/nvidias-b200-keeping-the-cuda-juggernaut):
  measured B200 structure and latencies.
- [Benchmarking and Dissecting the Nvidia Hopper GPU Architecture](https://arxiv.org/abs/2402.13499):
  the latency and throughput measurements quoted here, on an A100 PCIe 40GB and an H800 PCIe.
- [Compute capabilities, CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html):
  per-architecture limits for threads, registers, and shared memory.
- [The companion course's roofline chapter](https://qwen.ifkash.dev/c/10-roofline):
  the same two rates applied to a language model.
