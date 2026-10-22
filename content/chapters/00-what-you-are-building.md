---
title: What you are building
slug: 00-what-you-are-building
part: "Part 1 — The machine"
summary: Why GPU kernels decide how fast a model trains and serves, how a GPU differs from a CPU, how to launch and time a first kernel from Python, and how the course and its labs are laid out.
minutes: 60
gpu: true
objectives:
  - Say what a GPU kernel is, and name the two rates, memory bandwidth and arithmetic throughput, that bound every kernel.
  - Compute the A100's ridge point of 161 FLOP per byte, and classify a kernel as memory bound or compute bound from its FLOP and byte counts.
  - Explain why a GPU is a throughput machine that hides latency with many threads, and use Little's law to say how many bytes must be in flight.
  - Compile and call a CUDA kernel from Python with load_inline, and read the launch's grid and block arguments.
  - Time GPU work correctly with CUDA events, and explain why a host timer without a synchronize measures the wrong thing.
  - Describe the six parts of the course, what each lab builds, and how a lab runs on the RunPod A100.
  - Say how this course relates to its companion, learn-inference, and what you need to know before you start.
lab: 00-hello-gpu
---

# What you are building

> [!TLDR]
> - A kernel is one function that thousands of GPU threads run at once. Training and serving a model is a long sequence of kernels, and the speed of the model is the speed of that sequence.
> - Every kernel is bound by one of two rates: bytes per second the memory delivers, or FLOP per second the arithmetic units sustain. On the A100 80GB PCIe those are rated at 1935 GB/s and 312 TFLOP/s, so a kernel needs 161 FLOP per byte to leave the memory side behind.
> - A GPU is a throughput machine. It hides a memory latency of about 330 ns by keeping hundreds of thousands of loads in flight, which is why a kernel needs many threads.
> - A launch is asynchronous. Time a kernel with CUDA events, or synchronize before you read a host clock.
> - The course has six parts and 23 labs, all on a real A100. Each lab predicts a time from the specs and then measures it.

You write PyTorch, and you know what a transformer is. This course teaches what happens underneath: the machine that runs your tensors, and the kernels that make the difference between a training run that takes a week and one that takes three days.

By the end you have written reductions, a tiled matrix multiplication, a tensor core GEMM, a FlashAttention kernel, and a ring all-reduce, and you can predict, before you run it, roughly how long each should take. The prediction is the skill. A kernel that is much slower than its prediction has a bug, and one that is much faster has a bug in the measurement.

This chapter is the front door. It gives you the picture of why kernels matter, the mental model of the machine, a first kernel and a first honest timing, and a map of the course. Later chapters assume everything here and go deeper on each piece.

## Before you start

You need less than you might think. The course assumes you write Python and PyTorch, know what a transformer layer does, and can read a few lines of C++. It doesn't assume you have written CUDA, and the labs give you a starter file with every TODO marked. This section fixes the notation the rest of the course uses.

**Bytes and FLOPs.** A *FLOP* is one floating-point operation, an add or a multiply. A fused multiply-add counts as two. *FLOP/s* is a rate. Write FLOP for a count and FLOP/s for a rate, because mixing them is the most common arithmetic slip in this subject. A float32 is 4 bytes, and bfloat16 and float16 are 2 bytes.

**Decimal and binary units.** Bandwidth and arithmetic rates are decimal: 1935 GB/s is $1.935 \times 10^{12}$ bytes per second, and 312 TFLOP/s is $3.12 \times 10^{14}$ FLOP per second. Capacities are binary where the hardware's are: the A100's 40 MB L2 cache is 40 MiB, which is $40 \times 2^{20}$ bytes. When a number has a unit, the course states which one.

**Rated, measured, and predicted.** A *rating* is the vendor's peak, which no kernel reaches. A *measurement* is a number someone ran on a card, and the course names the card. A *prediction* is arithmetic on ratings and measurements. Chapters are written without access to the GPU, so the numbers you see in a chapter are ratings, arithmetic, or quotations from the companion course or a cited paper. Your lab reports the measurements.

**The GPU in one paragraph.** An NVIDIA GPU is built from *streaming multiprocessors* (SMs). The A100 has 108. Each SM runs many *threads*, grouped in sets of 32 called *warps*, and a *kernel* launch creates a *grid* of *blocks* of threads. Chapters 1 and 2 build this picture properly, and this chapter uses only the words.

**Where to look up a word.** The [GPU Glossary](https://modal.com/gpu-glossary) gives short linked definitions of the device, memory, and software terms. Keep it open in a tab.

## Where the time goes

Training and serving a model are sequences of kernels, and two facts about those kernels explain most of the course. The section answers one question: why does a model's speed come down to a few kinds of kernel?

A PyTorch operation such as `x @ w` or `torch.softmax(x, -1)` doesn't compute anything itself. It picks a *kernel*, a function compiled for the GPU, and launches it. A transformer's forward pass launches that kind of function hundreds of times, and a training step launches them thousands of times, forward and backward. The profile of a step is a list of kernels with durations, and the step time is, to a first approximation, their sum plus the gaps between them.

Most of those kernels fall into three families, and the family sets what limits the kernel.

- **Matrix multiplications** (the projections, the MLP, the attention scores) do the bulk of the arithmetic. They are the only kernels that can use the tensor cores, and the only ones that are bound by arithmetic.
- **Elementwise and normalization kernels** (the residual add, the activation, RMSNorm, dropout, the optimizer update) do a few FLOP per element, so they spend their time reading and writing memory.
- **Reductions and data-movement kernels** (softmax, a loss, a transpose, a gather) are mostly memory traffic too, with some coordination between threads.

### A worked comparison: one matmul and four elementwise ops

Put numbers on the first two families with a single activation tensor. Take 4096 tokens of a 4096-wide hidden state in bfloat16. The tensor holds $4096 \times 4096 = 16{,}777{,}216$ values at 2 bytes each, which is 33.6 MB.

A matrix multiplication of that activation by a $4096 \times 4096$ weight does $2 \times 4096^3 = 137.4$ GFLOP. At the 312 TFLOP/s tensor core rating, the floor is

$$
t_{\text{matmul}} \ge \frac{137.4 \times 10^{9}\ \text{FLOP}}{312 \times 10^{12}\ \text{FLOP/s}} = 0.44\ \text{ms}.
$$

It must also move, at minimum, the two inputs and the output, $3 \times 33.6 = 100.7$ MB, which takes $100.7 \times 10^{6} / 1.935 \times 10^{12} = 52\ \mu\text{s}$ at the memory rating. So the arithmetic floor is about 8.5 times the memory floor, and the matmul is limited by arithmetic.

Now an elementwise op on the same tensor, say the activation function. It reads 33.6 MB and writes 33.6 MB, which is 67.1 MB. At the 1275 GB/s that a plain copy measured on this card in learn-inference, that is

$$
t_{\text{elementwise}} \ge \frac{67.1 \times 10^{6}\ \text{B}}{1.275 \times 10^{12}\ \text{B/s}} = 52.6\ \mu\text{s}.
$$

Its arithmetic, a few operations on each of 16.8 million values, finishes in a small fraction of that. The elementwise op is limited by memory.

Run four of those in a row, as a residual add, a normalization, an activation, and a scale do when each is its own PyTorch call, and the floor is $4 \times 52.6 = 210\ \mu\text{s}$. That is about half of the matmul's 440 µs, for operations that do almost no arithmetic. Each op writes its result to memory and the next reads it straight back.

> [!KEY] Kernels are where the speed is
> A single kernel that does all four elementwise steps reads the tensor once and writes it once, so its floor is 52.6 µs, four times lower than four separate kernels. Nothing about the arithmetic changed. The saving is bytes that never move. Fusion, tiling, and every other optimization in this course are ways of moving fewer bytes, or of keeping the arithmetic units fed while the bytes move.

This arithmetic is a *prediction*. Real kernels land at or above those floors, and Chapter 17 measures how close fusion gets on this card. In learn-inference, a fused RMSNorm on 4096 rows of 5120 values ran at 945 GB/s, which is 74% of the copy ceiling, and at 4096 rows the fusion speedup came out at 0.92x rather than the 1.25x the byte count predicts. The byte count is where you start, and the measurement tells you what else is going on.

### Serving: the same story at batch size 1

Serving a language model one token at a time shows the memory side at its starkest. To produce one token, the GPU must read every weight once. A 7-billion-parameter model in bfloat16 holds $7 \times 10^{9} \times 2 = 14$ GB of weights, so the floor for one token is

$$
t_{\text{token}} \ge \frac{14 \times 10^{9}\ \text{B}}{1.935 \times 10^{12}\ \text{B/s}} = 7.2\ \text{ms},
$$

or 11.0 ms at the 1275 GB/s copy ceiling. The arithmetic for that token is $2 \times 7 \times 10^{9} = 14$ GFLOP, which takes $14 \times 10^{9} / 312 \times 10^{12} = 0.045$ ms at peak. The memory floor is 161 times the arithmetic floor. That ratio is no accident: it is the ridge point you meet in the next sections, because at batch size 1 a bfloat16 model does 1 FLOP per byte of weights.

### Training: the arithmetic dominates, until it doesn't

Training is the opposite case for the matmuls and the same case for everything else. A common rule of thumb, from the [scaling-laws literature](https://arxiv.org/abs/2001.08361), is that a training step costs about $6ND$ FLOP for a model with $N$ parameters and $D$ tokens: two for the forward pass and four for the backward. For $N = 7 \times 10^{9}$ and $D = 4096$ tokens, that is $1.72 \times 10^{14}$ FLOP, and the floor at 312 TFLOP/s is 0.55 s. A step that takes much longer than that is losing time somewhere, and the profile shows where: elementwise kernels, small kernels whose launch cost dominates, and matmuls that don't reach the tensor core rate.

Horace He's [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) names the three places time goes: compute, memory bandwidth, and overhead. This course covers all three. Memory and compute get Parts 1 to 4. Overhead, the time the GPU sits idle between kernels, gets chapter 17.

## Two machines: latency and throughput

A CPU and a GPU solve the same problem with opposite strategies, and the difference explains how you have to write GPU code. The section answers one question: why does a GPU need so many threads?

A CPU core is built to finish one thread's work as soon as possible. It spends transistors on large caches, branch prediction, and out-of-order execution, so a single stream of instructions runs quickly even when it waits on memory. That's a *latency machine*: it minimizes the time to finish one task.

A GPU is built to finish as much total work as possible per second. It spends its transistors on arithmetic units and on the machinery to switch between threads, and it tolerates slow individual threads. That's a *throughput machine*. The A100 has 108 SMs, and each can hold 2048 resident threads, so up to $108 \times 2048 = 221{,}184$ threads are in flight at once, scheduled onto 6,912 FP32 cores. A thread that stalls on memory isn't a problem, because another warp runs while it waits.

### Why many threads: Little's law

Reading from HBM, the A100's main memory, takes about 466 cycles. At the 1410 MHz boost clock, that is 466.3 / 1.41 = 330.7 ns, measured on an A100 PCIe 40GB in the [Luo et al. microbenchmark paper](https://arxiv.org/abs/2402.13499). Your thread's load returns more than 300 nanoseconds after it asks.

If a single thread waited on each load, the memory system would sit idle. To keep it busy, the GPU needs enough loads outstanding at any moment that one finishes as each new one starts. Little's law states this: the data in flight equals the rate times the latency.

$$
\boxed{\hla{B_{\text{flight}}} = \hlb{\beta} \times \hlc{L} = 1935 \times 10^{9}\ \text{B/s} \times 330.7 \times 10^{-9}\ \text{s} \approx 640\ \text{KB}}
$$

Here $\hla{B_{\text{flight}}}$ is the bytes in flight across the whole GPU, $\hlb{\beta}$ is the bandwidth you want to sustain, and $\hlc{L}$ is the latency of one load. Spread across 108 SMs, that is 5.9 KB per SM. If each thread has one 4-byte load outstanding, you need $640{,}000 / 4 = 160{,}000$ threads waiting on memory at once, or 1,481 per SM. A kernel that launches 10,000 threads can't reach the rated bandwidth, whatever its code looks like.

```viz
00-threads-in-flight
```

The figure shows the prediction. Drag the thread count down and the predicted bandwidth falls in proportion. Switch the load width from 4 to 16 bytes and the same bandwidth needs a quarter of the threads, which is the reason chapter 3 vectorizes its copy kernel with `float4`. The model is an upper bound from the ratings and one latency; the real ceiling for a copy on this card is the measured 1275 GB/s, and chapter 4 returns to latency hiding with the full roofline.

> [!NOTE] The 330.7 ns is from a 40 GB card
> No source we read measured the 80 GB A100's latency, and the paper's card is the 40 GB PCIe model with 1555 GB/s of bandwidth. The latencies are expected to be close, and the figure uses them only to show the shape of the curve. Chapter 4 derives the same quantity and says which parts are measured.

### What this means for the code you write

Two habits follow from the throughput design, and you practice both from the first kernel.

1. **Give the GPU a lot to do.** One thread per element of a large array is the default, because the hardware is happiest with hundreds of thousands of threads. A kernel on a few thousand elements is dominated by launch cost and idle SMs.
2. **Think in warps.** A warp's 32 threads execute together, so the 32 threads next to each other should load next-to-each-other addresses. Chapters 2 and 3 explain why and measure the cost of getting it wrong.

## Your first kernel, from Python

You launch a kernel by writing a function in CUDA C++, compiling it, and calling it on tensors. The section answers one question: what is the smallest complete kernel you can call from PyTorch?

The labs compile CUDA from Python with `torch.utils.cpp_extension.load_inline`, which takes C++ source as a string, runs `nvcc`, and returns a module you import. The following code is a complete SAXPY, $y \leftarrow a x + y$, the first kernel in [An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/) adapted to tensors:

```python
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

__global__ void saxpy_kernel(const float* x, float* y, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

void saxpy(torch::Tensor x, torch::Tensor y, double a) {
    TORCH_CHECK(x.is_cuda() && y.is_cuda(), "x and y must be CUDA tensors");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 && y.scalar_type() == torch::kFloat32,
                "x and y must be float32");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous(), "x and y must be contiguous");
    TORCH_CHECK(x.numel() == y.numel(), "x and y must have the same size");
    int n = static_cast<int>(x.numel());
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    saxpy_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), static_cast<float>(a), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

module = load_inline(
    name="saxpy",
    cpp_sources="void saxpy(torch::Tensor x, torch::Tensor y, double a);",
    cuda_sources=cuda_src,
    functions=["saxpy"],
)

x = torch.randn(1_000_003, device="cuda")
y = torch.randn_like(x)
expected = 2.0 * x + y
module.saxpy(x, y, 2.0)
print(torch.allclose(y, expected))  # True
```

The first call takes 30 to 60 seconds, because `nvcc` and the host compiler work through the PyTorch headers. `load_inline` caches the result by a hash of the source, so later runs are instant, and editing one character costs you the full compile again.

### Reading the kernel

The kernel body has three lines, and each earns a sentence.

- `__global__` marks a function that runs on the GPU and is launched from the host. It returns `void`, so results leave through the pointers.
- `int i = blockIdx.x * blockDim.x + threadIdx.x;` is how a thread finds its own element. The launch creates `blocks` blocks of `threads` threads, every thread runs this same body, and the formula turns a block number and a position in the block into a global index.
- `if (i < n)` guards the tail. The grid has `blocks * threads` threads, which is at least `n` and usually more, and the extra threads must do nothing.

The host function does four jobs: it checks its inputs, rounds the block count up, launches with the `<<<blocks, threads>>>` syntax, and checks that the launch was accepted. For $n = 1{,}000{,}003$ and 256 threads per block, `blocks` is $\lceil 1{,}000{,}003 / 256 \rceil = 3{,}907$, and the grid holds $3{,}907 \times 256 = 1{,}000{,}192$ threads, so 189 of them fail the guard.

Chapter 2 derives the index formula and the grid geometry, and the [companion course's chapter 12](https://qwen.ifkash.dev/c/12-your-first-cuda-kernel) walks through the same kernel line by line. Here the point is to see the loop closed: Python tensor in, thousands of threads, Python tensor out.

> [!NOTE] Why the stream argument
> The fourth launch argument, `at::cuda::getCurrentCUDAStream()`, puts the kernel on the stream PyTorch is using. A launch with no stream argument goes to the default stream, which is the right stream until the day your code runs inside a `torch.cuda.stream(...)` block and the kernel runs on the wrong one. Chapter 22 covers streams.

### How much should it cost?

Apply the habit you came for: predict before you measure. For $n = 2^{24} = 16{,}777{,}216$ floats, SAXPY reads `x` and `y` and writes `y`, so it moves $3 \times 4 \times 2^{24} = 201.3$ MB, and it does 2 FLOP per element. That is 0.17 FLOP per byte, far below the ridge point, so the kernel is memory bound. The learn-inference course measured a hand-written vector add of the same shape, $3n$ values moved, at 1304 GB/s on this card, which gives $201.3 \times 10^{6} / 1.304 \times 10^{12} = 154\ \mu\text{s}$. Expect SAXPY to land in the same place, and the lab in chapter 3 measures it.

That 154 µs is also about 30 times the 5 µs a host spends queueing a launch, which is why the launch cost doesn't matter for one big kernel and matters a great deal for a thousand small ones.

## Timing it: launches are asynchronous

A CUDA launch returns to Python before the kernel finishes, so a clock read around the call measures the wrong interval. The section answers one question: how do you time GPU work correctly?

When Python calls `module.saxpy(x, y, 2.0)`, the host queues the launch on a stream and returns, in about 5 microseconds on the lab A100 (measured in learn-inference, chapter 12). The GPU starts the kernel when it reaches it in the queue. The host and the GPU run in parallel, and ==a host clock reads the queueing, not the kernel==.

```viz
00-async-timeline
```

Try the three clocks in the figure. With no synchronize, the host timer reads about 5 µs whatever the kernel does. With a synchronize before the second read, the host timer waits for the GPU and the reading is the queueing plus the kernel. With CUDA events, the reading is the kernel alone, as the GPU observed it. The figure is a model with the 5 µs queue time and a default kernel of 154 µs, the SAXPY estimate above.

### Time with CUDA events

A *CUDA event* is a marker the GPU records into the stream when it reaches that point. The time between two events is the time the GPU took to get from one to the other, measured on the device with a resolution of about half a microsecond. The following function is the pattern the lab asks you to write:

```python
import statistics
import torch

def time_ms(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    readings = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()          # wait until the GPU has passed `end`
        readings.append(start.elapsed_time(end))   # milliseconds
    return statistics.median(readings)
```

Each piece has a reason.

1. **Warm up.** The first call of a kernel loads it onto the device, and the first PyTorch allocation initializes the allocator. Neither belongs in the number. Warmup calls run first, followed by a full synchronize.
2. **`enable_timing=True`.** An event records a timestamp only when you ask. Events created without it are cheaper and can't be subtracted.
3. **`end.synchronize()`.** `record` only queues the event. `elapsed_time` needs both events to have occurred, so wait for `end` first. Skip this and `elapsed_time` raises an error, or on some paths reads a time before the kernel ended.
4. **The median.** A single iteration can be slowed by a clock change or an unrelated process. The median ignores the outliers, and the minimum, which is also reasonable, hides run-to-run variation you may want to see.

> [!WARNING] A host timer around a launch is wrong, and it looks right
> `time.perf_counter()` before and after `module.saxpy(x, y, 2.0)` produces a clean, stable, plausible number: the queue time. Nothing crashes. The only symptom is a kernel that appears to run in 5 µs and to beat the speed of light. If a result looks too good, check how you timed it first. A bandwidth above the card's rating is the clearest sign.

The shared helper `cuda_time` in `lab_common` does the same thing with a few extra options, such as flushing the L2 cache between calls, which chapter 3 needs when the data fits in the 40 MiB L2. In this lab you write the event timer yourself once, and every later lab uses the helper.

### Errors arrive late too

Asynchrony also delays errors. An out-of-bounds write in a kernel doesn't raise in the line that launched it; the error surfaces at the next synchronize, often on an unrelated line, and most CUDA errors then poison the context so every later call fails. `C10_CUDA_KERNEL_LAUNCH_CHECK()` catches launch-time errors, such as a bad grid size. For execution errors, set the environment variable `CUDA_LAUNCH_BLOCKING=1` while you debug, which makes every launch synchronous and puts the error on the right line. Later chapters return to debugging.

## Two rates and the ridge point

Every kernel moves bytes and does arithmetic, and the GPU has a ceiling for each. The section answers one question: when is a kernel waiting on memory?

The A100 80GB PCIe, the card the labs run on, is rated at the following peaks:

| Quantity | Value | Source |
|---|---|---|
| Streaming multiprocessors (SMs) | 108 | [NVIDIA A100 datasheet](https://www.nvidia.com/en-us/data-center/a100/) |
| HBM2e memory bandwidth | 1935 GB/s | Rated, same datasheet |
| bfloat16 tensor core throughput, dense | 312 TFLOP/s | Rated, same datasheet |
| fp32 throughput on the CUDA cores | 19.5 TFLOP/s | Rated, same datasheet |

The 312 TFLOP/s checks against the clock: $108 \times 1410 \times 10^{6} \times 2048$ FLOP per SM per clock $= 311.9 \times 10^{12}$. The fp32 figure uses 128 FLOP per SM per clock, which gives 19.5.

A rating is a ceiling that no kernel reaches. In learn-inference (its chapter 10), a plain device-to-device copy measured 1275 GB/s on this card, 66% of the rating. That measured figure, not the rating, is the ceiling a memory-bound kernel is judged against, and chapter 3's lab measures it again.

> [!WARNING] The SXM4 card is faster
> The 80GB A100 also ships as an SXM4 module rated at 2039 GB/s, and the GPU provider hands you whichever is free. Two runs of one lab can land on different cards. The lab below prints which one you got, and treat differences under about 5% as noise.

### The ridge point

Divide the arithmetic rate by the memory rate and you get the arithmetic a kernel must do per byte to keep both sides busy. Take the two peaks and divide:

$$
\boxed{I^{\ast} = \frac{\hla{C}}{\hlb{\beta}} = \frac{312 \times 10^{12}\ \text{FLOP/s}}{1.935 \times 10^{12}\ \text{byte/s}} = 161.2\ \text{FLOP/byte}}
$$

Here $\hla{C}$ is the peak arithmetic rate and $\hlb{\beta}$ is the peak memory bandwidth. A kernel whose *arithmetic intensity*, FLOP per byte moved, is below $I^{\ast}$ is limited by memory, and one above it is limited by arithmetic. The kernel's time floor is the larger of its two times:

$$
t \ge \max\left(\frac{\text{FLOP}}{\hla{C}},\ \frac{\text{bytes}}{\hlb{\beta}}\right).
$$

The kernel can't finish before either unit does, and a kernel that overlaps the two perfectly finishes at the larger. Chapter 4 draws this as a roofline and adds the measured ceiling.

> [!EXAMPLE] Four kernels against the ridge point
> The intensity of each kernel is its FLOP divided by its minimum bytes.
>
> - **Vector add, float32.** 1 FLOP per 12 bytes (two reads, one write), so $1/12 = 0.083$ FLOP/byte. At 1935 GB/s the most it can do is $1935 / 12 = 161$ GFLOP/s, which is 0.052% of 312 TFLOP/s.
> - **Decode at batch 1, bfloat16.** 2 FLOP per parameter against 2 bytes per parameter: 1 FLOP/byte.
> - **The 4096-cubed matmul above.** $137.4 \times 10^{9} / 100.7 \times 10^{6} = 1365$ FLOP/byte, 8.5 times the ridge point.
> - **A square matmul with side $n$ in bfloat16.** $2n^3$ FLOP over $3 \times 2n^2$ bytes is $n/3$ FLOP/byte, so it crosses the ridge at $n = 3 \times 161 \approx 484$.

This is why a course about the GPU is mostly about memory. Most of a model's kernels sit below the ridge point. The matmuls sit above it only when the matrices are large, and a matmul with a small side, such as the one at decode, falls below it again.

### Which precision, which peak

The ridge point depends on which arithmetic units run. The 312 TFLOP/s figure is the tensor cores in bfloat16 or float16, dense. The same $4096^3$ matmul in float32 on the CUDA cores has a floor of $137.4 \times 10^{9} / 19.5 \times 10^{12} = 7.0$ ms, which is 16 times the bfloat16 floor ($312 / 19.5 = 16$). By default PyTorch doesn't use the TF32 tensor cores for float32 matmuls (`torch.backends.cuda.matmul.allow_tf32` is `False`), so a float32 `torch.matmul` on the A100 runs at the 19.5 TFLOP/s peak. Chapters 9 and 11 take up the formats and the tensor cores.

## The course map

The course has six parts, and each one adds a layer to the same picture. The section answers one question: what do you build, and in what order?

Every chapter has a book chapter, a lab, and a slide deck. The labs are the point: you write the kernel, a test checks it against a PyTorch reference, and the lab reports how fast it ran next to what the specs predict.

| Part | Chapters | What you learn | Labs build |
|---|---|---|---|
| 1. The machine | 0 to 5 | SMs, warps, the memory hierarchy, the roofline, profiling | A device-peak calculator, an occupancy calculator, copy kernels, roofline predictions, a profile-and-fix exercise |
| 2. CUDA patterns | 6 to 9 | Reductions, scans, shared memory, floating point | Sum and max reductions, scan and histogram, a padded transpose, a numerics study |
| 3. Matrix multiplication | 10 to 14 | SGEMM, tensor cores, async pipelines, CUTLASS layouts, Triton | An SGEMM ladder, a WMMA GEMM, a cp.async pipeline, CuTe layouts, a Triton matmul |
| 4. Kernels for ML | 15 to 20 | Softmax, FlashAttention, fusion, quantization, training kernels, MoE | Online softmax and LayerNorm, FlashAttention-2, torch.compile and CUDA graphs, a W4A16 GEMM, a fused cross-entropy, a grouped GEMM |
| 5. Many GPUs | 21, 22 | Collectives, overlap, parallelism | A ring all-reduce, stream overlap |
| 6. Beyond | 23 | AMD, TPUs, Apple silicon, tile languages | None; the chapter is reading |

Part 1 is the machine, and it is the part everything else leans on, so chapters 1 to 5 are shorter and heavier on measurement than the rest. Part 2 teaches the patterns most kernels reuse. Part 3 is the longest, because matrix multiplication is where a model spends most of its arithmetic. Part 4 applies all of it to the kernels a transformer needs, and Part 5 adds more than one GPU.

> [!NOTE] Hopper and Blackwell
> The labs run on an Ampere A100 (compute capability 8.0). Hopper (H100) and Blackwell (B200) features, such as TMA, `wgmma`, and `tcgen05`, are taught in prose, figures, and slides, and where a lab can exercise the same idea on Ampere it does. Every hardware table names the part it describes.

### The two languages

The labs use two ways of writing kernels. *CUDA C++*, compiled from Python with `load_inline`, shows you the hardware directly: threads, blocks, shared memory, and registers. *Triton* is a Python-embedded language in which you write the program for one block of data and the compiler handles the thread-level details. Parts 1 and 2 are mostly CUDA, Part 3 starts in CUDA and moves to Triton, and Part 4 is mostly Triton. In practice, you write Triton for most production kernels and drop to CUDA when you need control Triton doesn't expose, and you only know which is which after writing both. The lab environment pins PyTorch 2.14.1, CUDA 13.0, and Triton 3.8.0.

## How the labs run

A lab is a Python file you edit in the browser, which a worker runs on a real A100. The section answers one question: what happens when you press Run?

1. You edit `starter.py`, which holds function stubs and docstrings that specify each function. A `TODO` marks every place you write code.
2. You press **Run**, or the keyboard shortcut the editor shows. The app sends your file to a RunPod serverless endpoint that has one A100 80GB, a CUDA toolkit with `nvcc`, PyTorch, and Triton.
3. The worker imports your file and runs the lab's `tests.py`. Output streams to the **Console** tab as it arrives.
4. When the run ends, the **Checks** tab lists each check with a pass or fail mark and a detail line, and the run reports its metrics, such as `copy_gbs`.

A few things about the environment are worth knowing now.

- **The first compile is slow.** `nvcc` takes 30 to 60 seconds the first time a source is built. Labs that compile CUDA have a timeout of at least 900 seconds for this reason. The compiled result is cached, so a second run of the same source is fast, and editing the source costs you the compile again.
- **The first run can wait for a GPU.** A serverless endpoint may need to start a worker. That wait isn't part of any timing.
- **A crash is contained.** An illegal memory access in your kernel kills only the lab's subprocess, and the app shows a message. You can fix the code and run again.
- **Correctness is a hard check, and speed is a metric.** Each lab checks your result against a PyTorch reference over several shapes, including awkward ones such as a size of 1 and sizes that don't divide the block size. It then reports performance and fails you only when you are far below what the roofline predicts, so a small difference in the card you land on doesn't fail a correct kernel.
- **Hints and the solution are on purpose.** The **Hints** tab reveals one hint at a time, and the worked solution is there when you need it. Try the lab first.

The lab for this chapter needs no CUDA C++ at all. It uses what PyTorch already provides.

## The companion course

This course shares a GPU, a voice, and a set of measured numbers with [learn-inference](https://qwen.ifkash.dev). The section answers one question: how do the two courses fit together?

learn-inference builds an inference engine for one 27-billion-parameter model, a Qwen model, on one rented A100. It covers the model's architecture, the memory arithmetic, the KV cache, the sampling, a roofline analysis, a first CUDA kernel, Triton fusion, FlashAttention, paged attention, continuous batching, quantization, tensor parallelism, and speculative decoding. It is organized around one model and one question, how fast can you serve it.

This course is organized around the machine and the kernels, and it goes deeper on every kernel topic that course touches. The following table maps the overlap:

| Topic | In learn-inference | In this course |
|---|---|---|
| The roofline and measured bandwidth | [Chapter 10](https://qwen.ifkash.dev/c/10-roofline) | Chapters 3 and 4 |
| A first CUDA kernel, coalescing | [Chapter 12](https://qwen.ifkash.dev/c/12-your-first-cuda-kernel) | Chapters 2, 3, and 6 |
| Fusion in Triton | [Chapter 13](https://qwen.ifkash.dev/c/13-fusion-in-triton) | Chapters 14, 15, and 17 |
| FlashAttention | [Chapter 14](https://qwen.ifkash.dev/c/14-flash-attention) | Chapter 16 |
| Quantization | [Chapter 18](https://qwen.ifkash.dev/c/18-quantization) | Chapter 18 |
| Tensor parallelism | [Chapter 19](https://qwen.ifkash.dev/c/19-tensor-parallelism) | Chapters 21 and 22 |

You don't need to take either course first. If you take both, learn-inference gives you the end-to-end view of one engine, and this course gives you the tools to build and tune its parts. The numbers this course quotes as "measured in learn-inference" came from that course's labs on the same A100 80GB PCIe: the 1275 GB/s copy, the 1304 GB/s vector add, and the roughly 5 µs launch.

## What goes wrong

The mistakes this chapter's lab provokes, and the ones that follow you through the course, share a theme: a number that looks plausible and is wrong.

- **Quoting the rating as the target.** A kernel at 1275 GB/s on this card is done. Comparing it with 1935 sends you hunting for a speedup that isn't there.
- **Timing the launch, not the kernel.** A CUDA launch returns before the GPU finishes, so a host timer measures how long the launch took to queue. Use CUDA events, as the lab does, or synchronize before you read the clock.
- **Forgetting the warmup.** The first call includes loading the kernel and initializing allocators. A lab that times one call reports a number that is several times too slow, and the error shrinks as you add warmup calls.
- **Counting half the bytes.** A copy reads and writes, so it moves $2N$ bytes for an $N$-byte tensor. Count only $N$ and your bandwidth is half the true value. The opposite error, counting the read and write twice, gives a figure above the rating.
- **Mixing units.** Bandwidth here is decimal, so 1935 GB/s is $1.935 \times 10^{12}$ bytes per second. Divide 312 TFLOP/s by 1935 without converting and you get 0.161 instead of 161. Dividing a byte count by $2^{30}$ gives GiB, not GB, and the two differ by 7%.
- **Comparing across cards.** A number measured on an H100 says nothing about the A100. When a chapter quotes a result from a paper, it names the hardware, and you should too.
- **Timing a tensor that fits in the cache.** A 16 MiB tensor sits inside the A100's 40 MiB L2, so a repeated copy of it can run faster than DRAM allows. The lab uses 512 MiB to stay well outside it.

> [!RECAP]
> - A kernel is a function that many GPU threads run at once, and a model's speed is the sum of its kernels' times and the gaps between them.
> - Memory bandwidth and arithmetic throughput are the two ceilings: 1935 GB/s and 312 TFLOP/s rated on the A100 80GB PCIe, with a measured copy ceiling of 1275 GB/s.
> - The ridge point is 161 FLOP per byte, and most model kernels sit below it, so most of the course is about moving fewer bytes.
> - A GPU hides a roughly 330 ns memory latency with about 640 KB of loads in flight, which takes hundreds of thousands of threads.
> - Launches are asynchronous. Time with CUDA events, warm up first, and count every byte read and written.
> - Predict a time from the specs, then measure it. The gap between them is where you learn.

## Check your understanding

> [!QUESTION] A kernel does 2 FLOP for every 4 bytes it moves. Which ceiling limits it on an A100?
> Its intensity is 0.5 FLOP/byte, far below the ridge point of 161, so memory bandwidth limits it. Its floor is $\text{bytes}/\beta$, not $\text{FLOP}/C$.

> [!QUESTION] Why does a host timer around a kernel launch give the wrong answer?
> The launch is asynchronous: the call returns once the work is queued, in about 5 µs on the lab A100. The timer measures the queueing, not the GPU's execution, unless you synchronize before the second read or time with CUDA events.

> [!QUESTION] The SXM4 A100 is rated at 2039 GB/s. What is its ridge point?
> The tensor core rating is the same, so $312 \times 10^{12} / 2.039 \times 10^{12} = 153.0$ FLOP/byte. A higher bandwidth lowers the ridge point, so more kernels are compute bound.

> [!QUESTION] How many bytes must be in flight to sustain 1935 GB/s with a 330.7 ns latency, and what does that say about launching 10,000 threads?
> Little's law gives $1935 \times 10^{9} \times 330.7 \times 10^{-9} \approx 640$ KB. With one 4-byte load per thread that is 160,000 threads, so 10,000 threads can keep at most 40 KB in flight, about 6% of what the memory system needs. The kernel can't reach the rating.

> [!QUESTION] You time `y = module.saxpy(x, y, 2.0)` with `time.perf_counter()` and get 8 µs for $2^{24}$ elements. Is that plausible?
> No. The kernel moves about 201 MB, which takes at least $201.3 \times 10^{6} / 1.935 \times 10^{12} = 104\ \mu\text{s}$ even at the rating. The 8 µs is the launch queue time. Time with events, or synchronize first.

> [!QUESTION] Why does fusing four elementwise ops into one kernel speed them up when the arithmetic is unchanged?
> Each unfused op reads and writes the whole tensor, so four ops move four times the bytes. A fused kernel reads once and writes once. For a 33.6 MB tensor at 1275 GB/s, the floor drops from $4 \times 52.6 = 210\ \mu\text{s}$ to 52.6 µs.

> [!QUESTION] At what size does a square bfloat16 matmul cross the A100's ridge point?
> Intensity is $n/3$ FLOP/byte for side $n$, from $2n^3$ FLOP over $6n^2$ bytes. Setting $n/3 = 161$ gives $n \approx 484$. Below that the matmul is memory bound by this count, which is why decode, with a batch of one, can't use the tensor cores well.

## Lab

> [!TRY] Hello, GPU
> Fill in `describe_device`, `time_ms`, `copy_bandwidth_gbs`, and `ceiling_fraction`. You write the CUDA-event timer yourself and use it to time a 512 MiB device-to-device copy.
>
> You pass when the device facts match what PyTorch reports, your timer waits for the GPU and agrees with `lab_common.cuda_time` within a factor of two, and the copy reaches at least half of the 1275 GB/s ceiling. The lab also prints what a host timer without a synchronize reads for the same copy, so you see the asynchronous launch in your own output. It reports `sm_count`, `copy_ms`, `copy_gbs`, `ceiling_fraction`, and `host_no_sync_ms`. On the lab A100, the roofline predicts a copy near the 1275 GB/s ceiling, and your lab measures the real figure. Allow a minute for the first call, which loads CUDA.

## Further reading

- [NVIDIA A100 datasheet](https://www.nvidia.com/en-us/data-center/a100/): the ratings this chapter quotes.
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/): the grid, block, thread, and kernel vocabulary and the asynchronous launch model, and the reference for everything from Part 2 onward.
- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html): Horace He's compute, bandwidth, and overhead framing.
- [GPU Glossary](https://modal.com/gpu-glossary): short, linked definitions for every term this course introduces.
- [An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/): Mark Harris's smallest complete CUDA program.
- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html): asynchronous execution, streams, and why timing needs events or a synchronize.
- [`torch.utils.cpp_extension.load_inline`](https://docs.pytorch.org/docs/2.14/cpp_extension.html) and [`torch.cuda.Event`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html): the two APIs this chapter and its lab use.
- [GPU MODE lectures](https://github.com/gpu-mode/lectures): the community's lecture series, starting with profiling CUDA kernels in PyTorch.
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html): the assess, parallelize, optimize, deploy cycle and what to measure first.
- [Your first CUDA kernel](https://qwen.ifkash.dev/c/12-your-first-cuda-kernel) in the companion course: the same kernel, with the coalescing measurement.
