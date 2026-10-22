---
title: What you are building
slug: 00-what-you-are-building
part: "Part 1 — The machine"
summary: What this course teaches, how the labs run on a real A100, and the two numbers, bandwidth and arithmetic rate, that every later chapter returns to.
minutes: 30
gpu: true
objectives:
  - Say what a GPU kernel is and why ML workloads spend most of their time in a handful of them.
  - Name the two rates, memory bandwidth and arithmetic throughput, and compute the ridge point that separates them.
  - Explain why a vector add can't use more than a sliver of the A100's peak FLOP/s.
  - Run your first lab, which times a copy with CUDA events.
lab: 00-hello-gpu
---

# What you are building

> [!TLDR]
> - A kernel is one function that thousands of GPU threads run at once. Training
>   and serving a model is a sequence of them.
> - Every kernel is limited by one of two rates: how fast it reads and writes
>   memory, or how fast it does arithmetic. The A100 80GB PCIe is rated at 1935
>   GB/s and 312 TFLOP/s for bfloat16 tensor cores.
> - The ridge point, 312 TFLOP/s divided by 1935 GB/s, is 161 FLOP per byte. A
>   kernel that does less arithmetic per byte than that waits on memory.
> - Most kernels in a model, from adds to normalizations to attention at short
>   lengths, fall below the ridge point.
> - The labs run on a real A100 and report what they measure.

You write PyTorch, and you know what a transformer is. This course teaches what
happens underneath: the machine that runs your tensors, and the kernels that
make the difference between a training run that takes a week and one that takes
three days.

By the end you have written reductions, a tiled matrix multiplication, a tensor
core GEMM, a FlashAttention kernel, and a ring all-reduce, and you can predict,
before you run it, roughly how long each should take. The prediction is the
skill. A kernel that is much slower than its prediction has a bug, and one that
is much faster has a bug in the measurement.

## Two rates

Every kernel moves bytes and does arithmetic, and the GPU has a ceiling for each.

The A100 80GB PCIe, the card the labs run on, is rated at the following peaks:

| Quantity | Value | Source |
|---|---|---|
| Streaming multiprocessors (SMs) | 108 | [NVIDIA A100 datasheet](https://www.nvidia.com/en-us/data-center/a100/) |
| HBM2e memory bandwidth | 1935 GB/s | Rated, same datasheet |
| bfloat16 tensor core throughput, dense | 312 TFLOP/s | Rated, same datasheet |
| fp32 throughput on the CUDA cores | 19.5 TFLOP/s | Rated, same datasheet |

A rating is a ceiling that no kernel reaches. In the companion course
learn-inference (its chapter 10), a plain device-to-device copy measured 1275
GB/s on this card, 66% of the rating. That measured figure, not the rating, is
the ceiling a memory-bound kernel is judged against, and chapter 3's lab measures
it again.

> [!WARNING] The SXM4 card is faster
> The 80GB A100 also ships as an SXM4 module rated at 2039 GB/s, and the GPU
> provider hands you whichever is free. Two runs of one lab can land on
> different cards. The lab below prints which one you got.

## The ridge point

Divide the arithmetic rate by the memory rate and you get the arithmetic a kernel
must do per byte to keep both sides busy. The section answers one question: when
is a kernel waiting on memory?

Take the two peaks and divide:

$$
\boxed{I^{\ast} = \frac{\hla{C}}{\hlb{\beta}} = \frac{312 \times 10^{12}\ \text{FLOP/s}}{1.935 \times 10^{12}\ \text{byte/s}} = 161.2\ \text{FLOP/byte}}
$$

Here $\hla{C}$ is the peak arithmetic rate and $\hlb{\beta}$ is the peak memory
bandwidth. A kernel whose arithmetic intensity, FLOP per byte moved, is below
$I^{\ast}$ is limited by memory, and one above it is limited by arithmetic.

> [!EXAMPLE] A vector add against the ridge point
> A vector add reads two float32 values and writes one for each single add: 12
> bytes per FLOP, an intensity of $1/12 = 0.083$ FLOP/byte. At 1935 GB/s that is
> at most $1935 / 12 = 161$ GFLOP/s, which is 0.052% of 312 TFLOP/s. The
> arithmetic units sit idle for all but a few parts in ten thousand of the time.

This is why a course about the GPU is mostly about memory. Chapter 4 draws the
full roofline, and the chapters after it build kernels against each side.

## What goes wrong

- **Quoting the rating as the target.** A kernel at 1275 GB/s on this card is
  done. Comparing it with 1935 sends you hunting for a speedup that isn't there.
- **Timing the launch, not the kernel.** A CUDA launch returns before the GPU
  finishes, so a host timer measures how long the launch took to queue. Use CUDA
  events, as the lab does.
- **Mixing units.** Bandwidth here is decimal, so 1935 GB/s is
  $1.935 \times 10^{12}$ bytes per second. Divide 312 TFLOP/s by 1935 without
  converting and you get 0.161 instead of 161.

> [!RECAP]
> - A kernel is a function that many GPU threads run at once.
> - Memory bandwidth and arithmetic throughput are the two ceilings.
> - The A100 80GB PCIe is rated at 1935 GB/s and 312 TFLOP/s.
> - The ridge point is 161 FLOP per byte, and most model kernels sit below it.
> - A rating is a ceiling; a measurement on the card is the target.

## Check your understanding

> [!QUESTION] A kernel does 2 FLOP for every 4 bytes it moves. Which ceiling limits it on an A100?
> Its intensity is 0.5 FLOP/byte, far below the ridge point of 161, so memory
> bandwidth limits it.

> [!QUESTION] Why does a host timer around a kernel launch give the wrong answer?
> The launch is asynchronous: the call returns once the work is queued. The timer
> measures the queueing, not the GPU's execution, unless you synchronize first or
> time with CUDA events.

> [!QUESTION] The SXM4 A100 is rated at 2039 GB/s. What is its ridge point?
> $312 \times 10^{12} / 2.039 \times 10^{12} = 153.0$ FLOP/byte.

## Lab

> [!TRY] Hello, GPU
> Fill in `describe_device`, `copy_bandwidth_gbs`, and `ceiling_fraction`. You
> pass when the device facts match what PyTorch reports and the copy's
> bandwidth is plausible. On the lab A100, expect a figure near the 1275 GB/s
> ceiling.

## Further reading

- [NVIDIA A100 datasheet](https://www.nvidia.com/en-us/data-center/a100/): the
  ratings this chapter quotes.
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/):
  the reference for everything from Part 2 onward.
