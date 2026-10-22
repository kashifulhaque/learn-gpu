---
title: Overlap and parallelism
slug: 22-overlap-and-parallelism
part: "Part 5 — Many GPUs"
summary: How CUDA streams, events, and pinned memory let copies and kernels run at the same time, and how data, sharded, tensor, pipeline, and expert parallelism each add kernels and collectives that frameworks then hide behind compute.
minutes: 120
gpu: true
objectives:
  - "Explain what a CUDA stream and an event are, state the default stream's synchronization rule, and say why a PyTorch side stream needs wait_stream."
  - "Say why a copy from pageable memory can't overlap with compute, and why a copy from pinned memory can, using the copy engines and the PCIe Gen4 rate of 31.5 GB/s per direction."
  - "Derive the time of a chunked copy-compute-copy pipeline, T = s1 + s2 + s3 + (n - 1) max(s), and compare it with the max-of-stages bound and the sum-of-stages baseline."
  - "Describe the caching allocator's cross-stream hazard and when record_stream or a wait is required."
  - "Say, for DDP, FSDP and ZeRO, tensor, pipeline, and expert parallelism, which collective or point-to-point kernel each one adds, how often, and how a framework overlaps it with compute."
  - "Compute the pipeline bubble for GPipe, 1F1B, and the interleaved schedule, and the activation memory each one holds."
  - "Estimate whether a communication is hidden behind compute by comparing its time over a given link with the compute it overlaps."
  - "Build a chunked pinned-memory pipeline with one stream and with several, and measure its efficiency against the max(copy, compute) bound."
lab: 22-streams-overlap
---

# Overlap and parallelism

> [!TLDR]
> - A GPU runs queues of work called streams. Work in one stream runs in order, and work in different streams can run at the same time, if the hardware has a free engine for it: a copy engine for a copy, the SMs for a kernel.
> - A host-to-device copy overlaps with a kernel only when the host memory is pinned (page-locked), the copy is asynchronous, and the two sit in different streams. On the A100 80GB PCIe, PCIe Gen4 x16 moves 31.5 GB/s per direction, about 61 times slower than HBM's 1935 GB/s rating, so these copies are worth hiding.
> - Cut a job into $n$ chunks and the three stages (copy in, compute, copy out) pipeline: the time falls from the sum of the stages to $s_1 + s_2 + s_3 + (n-1)\max(s)$, which approaches the longest stage.
> - Every parallelism strategy is a set of kernels and collectives added around your compute: DDP adds one all-reduce per gradient bucket, FSDP an all-gather and a reduce-scatter per layer, tensor parallelism two all-reduces per layer, pipeline parallelism point-to-point sends, and expert parallelism two all-to-alls per MoE layer. Frameworks win by putting those on their own streams and starting them early.
> - Overlap is never free: a communication kernel takes SMs and memory bandwidth, a bubble can't be hidden at all, and the caching allocator needs `record_stream` when a tensor crosses streams.

A training step isn't one long kernel. It's a mix of kernels, copies, and network transfers, and the GPU can do several of those at once: its SMs run a kernel while a DMA engine moves bytes over PCIe and another engine pushes bytes over NVLink. A step that does them one after another leaves most of that hardware idle. The same step, rearranged so the engines run together, finishes in the time of the slowest one.

This chapter is about that rearrangement, in two halves. The first half is the mechanism on one GPU: streams, events, pinned memory, and the chunked pipeline, which you build in the lab. The second half asks what changes when a model spans many GPUs, and looks at data, sharded, tensor, pipeline, and expert parallelism from the kernel's point of view. Each one adds specific kernels and collectives to the timeline, and the difference between a framework that scales and one that doesn't is how well it overlaps them.

The mechanism is the same at every scale. A DDP all-reduce hiding behind a backward pass, a `cp.async` hiding behind an MMA ([chapter 12](/c/12-pipelining-and-async-copies)), and a PCIe copy hiding behind a kernel are all the same idea: two things that don't depend on each other should not run one after the other.

## Before you start

**Asynchronous launches.** A kernel launch returns to Python as soon as the work is queued, and the GPU runs it later. [Chapter 0](/c/00-what-you-are-building) uses this to explain why a host timer without a synchronize measures the queue and not the kernel. Everything in this chapter builds on that: because a launch is only a queue entry, you can queue work for several engines before any of it runs.

**Collectives.** An *all-reduce* sums a tensor across ranks and leaves the result on every rank. An *all-gather* concatenates each rank's shard on every rank, and a *reduce-scatter* sums and leaves each rank one shard. An *all-to-all* sends a different slice to every rank. [Chapter 21](/c/21-collective-communication) derives their costs. This chapter uses two results from it: a ring all-reduce of $n$ bytes over $p$ ranks sends $2(p-1)/p \times n$ bytes per rank, and all-gather, reduce-scatter, and all-to-all each send $(p-1)/p \times n$.

**Links, per direction.** The A100's host link is PCIe Gen4 x16, which carries 16 GT/s per lane with 128b/130b encoding on 16 lanes. NVIDIA's datasheet prints 64 GB/s, which is the two directions added together. Per direction:

$$
\frac{16 \times 10^{9} \times \tfrac{128}{130} \times 16}{8}\ \text{B/s} = 31.5\ \text{GB/s}
$$

NVLink 3 on the A100 is 600 GB/s total across 12 links, which is 300 GB/s per direction. Use the per-direction figure whenever you divide a transfer by a link, because a copy in one direction uses one direction only. The course's fact sheet lists both; the datasheet figures are sums.

**Pinned and pageable memory.** Ordinary host memory is *pageable*: the operating system can move it or swap it out. *Pinned* (page-locked) memory is pinned to fixed physical pages, so a DMA engine can read it directly. You get it in PyTorch from `torch.empty(n, pin_memory=True)` or `tensor.pin_memory()`.

## Streams: the GPU's queues

A stream is an ordered queue of GPU work, and this section answers one question: what does the GPU guarantee about the order of work you launch?

Work in a single stream executes in the order you issued it. A kernel in a stream starts after the previous operation in that stream finishes. Work in two different streams has no guaranteed order, and the hardware is free to run it at the same time or in either order. That's the whole model, and the rest is how to add ordering where you need it.

Each device has a *default stream*. A kernel launch with no stream argument, or a PyTorch operation outside a `torch.cuda.stream` block, goes there. Under the legacy semantics, the default stream is special: an operation in it waits for all earlier work in every other blocking stream on the device, and later work in those streams waits for it, per [Mark Harris's NVIDIA overview of overlapping transfers](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/). It acts as a barrier. A program that uses only the default stream is correct and has no overlap.

PyTorch's side streams (`torch.cuda.Stream()`) don't take part in that barrier, so the guarantee goes away, and PyTorch's [CUDA semantics notes](https://docs.pytorch.org/docs/2.14/notes/cuda.html) say so directly: on a non-default stream, you are responsible for synchronization. The notes' own example shows the failure. A `sum()` on a new stream can start before the `normal_()` that fills its input finishes on the default stream, and it reads garbage.

### Ordering across streams with events

An *event* is a marker in a stream. Recording it puts a flag in the queue, and the flag is set when the GPU reaches it. Another stream can wait on the flag without involving the host, which is how you express "this kernel needs that copy" across streams.

```python
import torch

copy_stream = torch.cuda.Stream()
compute_stream = torch.cuda.Stream()

ready = torch.cuda.Event()               # no timing: cheaper than a timing event
with torch.cuda.stream(copy_stream):
    x_dev.copy_(x_host, non_blocking=True)
    ready.record(copy_stream)            # flag set when the copy completes

compute_stream.wait_event(ready)         # GPU-side wait: the host does not block
with torch.cuda.stream(compute_stream):
    y_dev = kernel(x_dev)
```

`wait_event` and `wait_stream` are GPU-side: they enqueue a dependency and return. The host moves on and queues more work. `event.synchronize()` and `torch.cuda.synchronize()` are host-side, and they stop the Python thread until the GPU gets there. The difference decides whether your queueing runs ahead of the GPU, which is the precondition for any overlap.

Events also time things. An event created with `enable_timing=True` records a GPU timestamp, and `start.elapsed_time(end)` gives the milliseconds between two of them, as in the lab in chapter 0. An event for ordering doesn't need timing, so create it without.

### The side-stream rule

Because a side stream isn't ordered against the default stream, PyTorch's rule has two halves. Before work on a side stream reads something the default stream produced, the side stream waits for it. After the side stream's work, the default stream waits for the side stream before using the result. The notes' corrected pattern:

```python
s = torch.cuda.Stream()
A = torch.empty((100, 100), device="cuda").normal_(0.0, 1.0)
s.wait_stream(torch.cuda.default_stream())   # s now waits for normal_()
with torch.cuda.stream(s):
    B = torch.sum(A)
A.record_stream(s)                           # see the allocator section below
torch.cuda.default_stream().wait_stream(s)   # before the default stream reads B
```

Don't join right after the launch: waiting at once turns the two streams back into one. Put independent work between the fork and the join.

### Priorities

A stream can have a priority. `torch.cuda.Stream(priority=-1)` makes a high-priority stream, where a lower number means a higher priority and 0 is the default. When the GPU has blocks from several streams waiting for SMs, it prefers the higher-priority stream's blocks as SMs free up. Priority doesn't preempt a running block and doesn't reserve SMs. Frameworks use it so a communication kernel doesn't queue behind a long compute kernel.

> [!KEY] A stream is an ordering, not a speed
> Streams don't make anything faster by themselves. They remove ordering that you don't need, so that independent work can use independent hardware. Overlap appears only when the work in the two streams needs different engines, or the same engine with room to spare.

## Copy engines and pinned memory

This section answers: when does a host-to-device copy run at the same time as a kernel? Three conditions have to hold, and the third is the one people miss.

A GPU has separate hardware for each kind of work. The SMs run kernels. *Copy engines* (DMA engines) move bytes between host and device memory without using the SMs. A GPU with two copy engines can run a host-to-device copy and a device-to-host copy at once. Harris's article describes a Tesla GPU with two, one per direction. The CUDA runtime reports the count for your card as `asyncEngineCount` in `cudaDeviceProp`.

For a copy and a kernel to overlap, all of the following must hold:

1. The device can overlap copy and execution, which every GPU in this course can.
2. The copy and the kernel are in different, non-default streams (or the default stream has been made per-thread).
3. The host memory in the copy is pinned.

### Why pageable memory serializes

A DMA engine reads physical addresses, and a pageable buffer has no stable ones: the operating system may move or swap a page at any time. So the driver can't hand a pageable buffer to the engine. For a copy from pageable memory, the CUDA runtime first copies the data into a pinned staging buffer it owns, and the engine copies from there. The runtime's [API synchronization behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html) page documents that a copy to or from pageable memory can be synchronous with respect to the host, even when it's an asynchronous call.

The result is a copy that blocks the host, drains the stream, and takes two passes over the data. With a pageable source, `cudaMemcpyAsync` or `copy_(..., non_blocking=True)` can behave like a synchronous copy, and work behind it in the stream can't overlap with it. The same holds for a copy back to pageable memory.

Pinned memory removes both costs. The engine reads the pages directly, and the call returns as soon as the copy is queued. The price is that pinned pages can't be reclaimed by the operating system, and allocating them is slow: the PyTorch notes warn that pinning is often expensive, and that too much of it can leave the host short of memory. Allocate pinned buffers once, at start-up, and reuse them. A `DataLoader(..., pin_memory=True)` does this for batches.

### What the copy costs

Put numbers on it with the lab's tensor, $2^{26}$ float32 values, which is 256 MiB. That's $2^{26} \times 4 = 268{,}435{,}456$ bytes. Over PCIe Gen4 x16 at 31.5 GB/s per direction:

$$
t_{\text{H2D}} \ge \frac{268.4 \times 10^{6}\ \text{B}}{31.5 \times 10^{9}\ \text{B/s}} = 8.52\ \text{ms}
$$

The companion course's paged-attention chapter uses a planning figure of about 21 GB/s, which would make this copy 12.8 ms. A real transfer lands below the link's ceiling, so 8.52 ms is a floor and your lab reports the measured time. The same 256 MiB on the GPU, copied within HBM at the measured 1275 GB/s, moves $2 \times 268.4 \times 10^{6}$ bytes in $0.42$ ms. The host link is about 61 times slower than the rated HBM ($1935 / 31.5 = 61.4$), which is why a kernel that reads data fresh from the host spends most of its time waiting on PCIe, and why hiding that wait is worth the work.

### non_blocking in PyTorch

`tensor.to("cuda", non_blocking=True)` and `dst.copy_(src, non_blocking=True)` queue the copy and return. For a pinned source, the copy is asynchronous. For a pageable source, PyTorch calls `cudaMemcpyAsync` all the same, and the runtime's staging makes it behave synchronously, so you get no error and no overlap. Check with `tensor.is_pinned()`.

Reading the result has the opposite hazard. A `non_blocking=True` copy to the host returns before the data arrives. If you read the destination before an event or a synchronize says the copy finished, you read stale bytes. In the lab, the tests fill the output with NaNs first, so a missing wait shows up as NaNs in the result.

## A pipeline of chunks

This section answers: how much faster is a chunked, overlapped pipeline than a sequential one, and what is its limit?

Say a job has three stages in sequence: copy $n$ bytes to the GPU, run a kernel on them, copy the results back. Run it as one piece and the stages run one after another:

$$
T_{\text{sequential}} = t_{\text{H2D}} + t_{\text{kernel}} + t_{\text{D2H}}
$$

Each stage uses different hardware, though. The host-to-device copy uses one copy engine, the kernel the SMs, and the device-to-host copy another copy engine. While one chunk is in the kernel, the next can be in flight to the GPU and the previous can be on its way back.

Cut the data into $n$ chunks, each taking $s_1 = t_{\text{H2D}}/n$, $s_2 = t_{\text{kernel}}/n$, and $s_3 = t_{\text{D2H}}/n$. The first chunk goes through all three stages alone. After that, the pipeline emits one chunk every time the slowest stage finishes one:

$$
\boxed{T_n = \hla{s_1 + s_2 + s_3} + \hlb{(n - 1)\max(s_1, s_2, s_3)}}
$$

Read it as a fill cost, $s_1 + s_2 + s_3$ for the first chunk, plus one slowest-stage time for each of the remaining $n - 1$ chunks. Expressed with the stage totals, the second term is $(n-1)/n \times \max(t_{\text{H2D}}, t_{\text{kernel}}, t_{\text{D2H}})$, so as $n$ grows $T_n$ approaches the longest stage's total, the bound $\max(t)$.

> [!EXAMPLE] A balanced pipeline on the lab tensor
> Take all three stages at 8.52 ms (the kernel's iteration count is chosen to match the copy, as the lab does). Sequential is $3 \times 8.52 = 25.57$ ms and the bound is 8.52 ms. For equal stages, $T_n = t(n + 2)/n$:
>
> | Chunks $n$ | $T_n$ (ms) | Speedup over sequential | Efficiency, bound / $T_n$ |
> |---|---|---|---|
> | 1 | 25.57 | 1.00x | 33% |
> | 2 | 17.04 | 1.50x | 50% |
> | 4 | 12.78 | 2.00x | 67% |
> | 8 | 10.65 | 2.40x | 80% |
> | 16 | 9.59 | 2.67x | 89% |
>
> The ceiling is 3x, reached only with infinitely many chunks. In practice every chunk costs a launch and a stream switch, so chunks that are too small give some of the gain back. These are model values; your lab measures the real ones.

The model assumes the three engines don't interfere. The two copies don't share PCIe bandwidth, because the link is full duplex in principle. A server or container can deliver less than the full rate in both directions at once, so compare your measured `d2h_ms` alone against the pipeline's time. The kernel and the copies share HBM, but the copies move 31.5 GB/s against HBM's 1935, so the contention is small for a compute-heavy kernel and noticeable for a bandwidth-bound one.

```viz
22-stream-timeline
```

### Two ways to build the pipeline

Both of the following issue the same operations. They differ in which streams hold what.

**A stream per chunk.** Chunk $i$ goes in stream $i \bmod S$, and each stream does copy-in, kernel, copy-out in order. The ordering inside a stream gives you the dependencies for free, and the streams run against each other. With $S = 1$ the chunks serialize and the chunked version runs in $T_{\text{sequential}}$ plus launch overhead, no better. 

```python
streams = [torch.cuda.Stream() for _ in range(n_streams)]   # create once, reuse
cur = torch.cuda.current_stream()
for s in streams:
    s.wait_stream(cur)                 # the buffers were allocated on cur

for i, (a, b) in enumerate(chunk_bounds(n, n_chunks)):
    s = streams[i % n_streams]
    with torch.cuda.stream(s):
        d_in[a:b].copy_(h_in[a:b], non_blocking=True)
        launch_kernel(d_in[a:b], d_out[a:b], iters)
        h_out[a:b].copy_(d_out[a:b], non_blocking=True)

for s in streams:
    cur.wait_stream(s)                 # join: GPU-side, no host stall
```

**A stream per stage.** One stream for host-to-device copies, one for kernels, and one for device-to-host copies, with events carrying the chunk dependency across them. Chunk $i$'s kernel waits for its copy-in event, and its copy-out waits for its kernel event. This mirrors the hardware engines and is how real data pipelines are written. It also exercises events, which the stream-per-chunk version never needs. This is the lab's `pipeline_staged`.

Harris's article compares two issue orders for the stream-per-chunk shape, and the winner differed between a GPU with one copy engine and a GPU with two. The issue order can matter, so measure it on your card.

> [!WARNING] A copy that blocks hides the whole pipeline
> If any call in the loop synchronizes the host, such as `.item()`, `.cpu()`, `print(tensor)`, or a copy from a pageable tensor, the Python thread stops queueing, the engines starve, and the pipeline degrades to the sequential time. The lab checks that your pipeline function returns to the host long before the GPU finishes.

## PyTorch, streams, and the caching allocator

This section answers: why can a tensor be freed, reused, and overwritten while a kernel in another stream is still reading it?

PyTorch's caching allocator keeps freed GPU memory in per-stream pools. When you drop the last reference to a tensor, the block goes back to the pool *at once*, without waiting for the GPU, because the allocator assumes every use of the tensor is already ordered before whatever comes next in the tensor's stream. That assumption makes `del x` cheap, and it's exactly the one that breaks when you use the tensor on another stream.

Here is the failure. Allocate `A` on the default stream, launch a kernel on a side stream `s` that reads `A`, and drop `A`. The block returns to the default stream's pool. The next allocation on the default stream can get that block, write to it, and the write can run while `s` is still reading. No error is raised. The kernel on `s` reads whatever the new tensor holds.

`Tensor.record_stream(s)` fixes it. It tells the allocator the tensor is also in use on stream `s`, so the block isn't reused until `s` has passed the point where you called it. The PyTorch notes use it in the corrected example in this chapter's streams section. It adds one event to the block, and the allocator checks the event before it reuses the block.

> [!DEEPDIVE] The reverse hazard, and what the allocator does to FSDP
> `record_stream` protects a tensor that crosses streams. The reverse case is a tensor *created* on a side stream and read on the default stream. The notes' answer is the `wait_stream` join: the default stream waits for `s` before it reads the result.
>
> A third hazard is memory the side stream allocates for itself. The allocator keeps blocks per stream, so a side stream that allocates large buffers, as the all-gather stream does in FSDP, holds those blocks in its own pool, and the compute stream can't use them. The FSDP paper reports that the producer stream can over-reserve and force expensive `cudaMalloc` retries on the compute stream, and it adds a *rate limiter* that allows at most two all-gathers in flight, the minimum that still overlaps. It also reports the limiter's effects varied by model, from up to 5x on T5-11B to about 5% overhead on DeepViT. The lesson for your own code: a side stream that allocates is part of your memory budget, and `torch.cuda.memory_stats()` shows the retries.

The practical rules are short:

- Allocate your buffers once and reuse them across iterations, as the lab does. A buffer that lives for the whole run can't be freed under a running kernel.
- If a tensor is used on a stream other than the one that allocated it, and it might be freed before the work finishes, call `record_stream`, or keep a reference until you've joined.
- Join with `wait_stream` or an event, not with a host synchronize, so the host keeps queueing.
- Backward ops run on the stream of the matching forward op, and the notes warn that using gradients outside the stream that ran `backward()` needs a `wait_stream` first.

## Data parallelism: all-reduce behind the backward pass

This section answers: how does DDP hide the gradient all-reduce, and when can't it?

In data parallelism every GPU holds a full copy of the model, runs forward and backward on its own slice of the batch, and averages gradients with an all-reduce so the replicas stay identical. The kernel's view is a backward pass followed by one big all-reduce, and if the all-reduce runs after the backward finishes, the GPU is idle for the whole of it.

Run the numbers for a 7-billion-parameter model with bf16 gradients, which is $7 \times 10^{9} \times 2 = 14$ GB. A ring all-reduce over $p = 8$ ranks sends $2(p - 1)/p \times 14$ GB $= 1.75 \times 14 = 24.5$ GB per rank. Over NVLink 3 at 300 GB/s per direction:

$$
t_{\text{allreduce}} \ge \frac{24.5 \times 10^{9}}{300 \times 10^{9}} = 82\ \text{ms}
$$

Over PCIe at 31.5 GB/s the same transfer takes $24.5 \times 10^{9} / 31.5 \times 10^{9} = 778$ ms. The backward pass for 8192 tokens on one GPU is about $4 \times 7 \times 10^{9} \times 8192 = 2.29 \times 10^{14}$ FLOP, which is at least $2.29 \times 10^{14} / 312 \times 10^{12} = 0.735$ s at the tensor-core peak, and longer at the efficiency real kernels reach. So over NVLink the all-reduce is about 11% of the backward floor and fits behind it comfortably, and over PCIe it's about the whole backward floor and doesn't. Overlap is how the first case becomes free, and it can't rescue the second. These are predictions from the link and compute ratings, not measurements.

### Buckets and hooks

The backward pass produces gradients layer by layer, from the last layer to the first. A layer's gradient is final as soon as that layer's backward finishes, so the all-reduce for the last layers can start while the first layers are still computing. DDP does this with gradient *buckets*, described in [PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://arxiv.org/abs/2006.15704) and the [DDP design notes](https://docs.pytorch.org/docs/2.14/notes/ddp.html):

- A `Reducer` registers one autograd hook per parameter. The hook fires when the parameter's gradient is ready.
- Parameters are grouped into buckets, filled roughly in the reverse order of `model.parameters()`, because gradients tend to become ready in that order. The size is the `bucket_cap_mb` argument, which defaults to 25 MiB in the constructor. (The reducer is also reported to use a smaller first bucket so the first all-reduce starts early. The constructor documentation doesn't say, so check the source before relying on it.)
- When every gradient in a bucket is ready, the reducer launches an asynchronous all-reduce on that bucket, on NCCL's own stream. The backward kernels keep running on the compute stream.
- After the last bucket, the reducer waits for all the all-reduces and writes the averaged gradients to `param.grad`.

Every process launches the all-reduces in bucket order, not in the order buckets happen to fill, so every rank issues the same collectives in the same order. A mismatch hangs or corrupts the result, which is why the bucket assignment is fixed at construction. If the real order differs from the reverse of `parameters()`, the first bucket fills late and the overlap shrinks.

The bucket size trades two costs from chapter 21's alpha-beta model. A small bucket starts sooner but pays the latency $\alpha$ of a collective more times. A large bucket amortizes $\alpha$ but starts later and leaves a long last all-reduce exposed after the final kernel. The exposed tail, the part that can't overlap, is roughly the last bucket's time plus any backlog.

> [!NOTE] Compilation can break the overlap
> The DDP notes warn that compiling the whole backward graph as one unit delays every hook until the backward finishes. PyTorch's `DDPOptimizer` splits the graph at bucket boundaries so hooks fire between pieces. If a compiled DDP model's profile shows all-reduces bunched at the end, check this first.

## Sharding the state: ZeRO and FSDP

This section answers: what does it cost, in kernels, to stop keeping a full copy of the model on every GPU?

DDP replicates everything, and replication is wasteful. [ZeRO](https://arxiv.org/abs/1910.02054) counts what a mixed-precision Adam run keeps per parameter: 2 bytes of fp16 weights, 2 of fp16 gradients, and 12 of fp32 optimizer state (an fp32 copy of the weights, the first moment, and the second), which totals $16\Psi$ bytes for $\Psi$ parameters. Sharding those across $N$ ranks in stages gives the ZeRO table. The paper's worked example is $\Psi = 7.5$ billion and $N = 64$:

| Stage | What is sharded | Bytes per GPU | At $\Psi = 7.5\text{B}$, $N = 64$ |
|---|---|---|---|
| Baseline (DDP) | nothing | $16\Psi$ | 120 GB |
| ZeRO-1 | optimizer state | $4\Psi + 12\Psi/N$ | 31.4 GB |
| ZeRO-2 | plus gradients | $2\Psi + 14\Psi/N$ | 16.6 GB |
| ZeRO-3 | plus parameters | $16\Psi/N$ | 1.9 GB |

The arithmetic for one row: ZeRO-2 is $2 \times 7.5 + 14 \times 7.5 / 64 = 15 + 1.64 = 16.6$ GB.

Memory isn't free. At stage 3 every rank holds only a shard of each weight, so a layer's weights must be assembled before the layer can run. The ZeRO paper counts the communication: DDP moves $2\Psi$ per step (a reduce-scatter and an all-gather, which together make an all-reduce), and ZeRO-3 moves $3\Psi$, which is 1.5 times as much.

*FSDP*, PyTorch's implementation of the same idea, adds kernels in a fixed pattern per unit of layers, as the [FSDP paper](https://arxiv.org/abs/2304.11277) describes:

1. **All-gather the unit's parameters** before its forward pass, run the forward, and free the unsharded copy.
2. **All-gather again** before its backward pass, since the copy was freed.
3. **Reduce-scatter its gradients** after the backward, so each rank keeps only its gradient shard, then free the unsharded copy.

Three collectives per unit is the price of $1/N$ memory. The three can hide behind compute, but only if they're issued early, so FSDP does three things:

- **A separate stream for communication.** FSDP issues the all-gathers on their own CUDA stream. Without it, NCCL would wait for the default stream's computation, and an all-gather couldn't overlap with it.
- **Prefetching.** Before the current unit's reduce-scatter, FSDP issues the next unit's all-gather, so two exposed communications don't land back to back on the critical path. Eager execution can't see the next unit, so FSDP uses the reverse of the recorded forward order as a proxy. The paper reports about 18% speedup from backward prefetching on GPT-175B. Forward prefetching issues the next all-gather before the current unit's forward, and helps when the CPU is too slow to queue it in time.
- **The rate limiter**, from the allocator discussion, which caps in-flight all-gathers at two.

Because the all-gather and the reduce-scatter share one process group and one NCCL stream, they run one after the other, never together. When the model is deep enough, the all-gather for layer $k + 1$ hides behind the compute of layer $k$. When a layer's compute is shorter than its all-gather, nothing hides it, and the step is communication bound.

## Tensor parallelism: the all-reduce on the critical path

This section answers: why does tensor parallelism need the fastest link, and what can be overlapped?

Tensor parallelism splits each weight matrix across ranks. The companion course's [tensor parallelism chapter](https://qwen.ifkash.dev/c/19-tensor-parallelism) derives the split for inference. Here is the training version, from the [Megatron-LM paper](https://arxiv.org/abs/1909.08053).

A *column-parallel* linear layer splits the weight $W$ by output columns. Each rank computes its own slice of the output from the full input, and no communication is needed. A *row-parallel* layer splits $W$ by input rows. Each rank has part of the input and computes a partial sum of the output, and the partial sums have to be added across ranks, which is an all-reduce. An MLP is two linear layers, so Megatron makes the first column-parallel and the second row-parallel. The activation between them is applied to each rank's slice with no communication, and the pair costs one all-reduce. Attention works the same way, splitting by head, with one all-reduce after the output projection.

So a transformer layer has two all-reduces in the forward pass (after attention and after the MLP) and two in the backward pass, which is four per layer. [Narayanan et al.](https://arxiv.org/abs/2104.04473) put the communication at $8\,b\,s\,h\,(t-1)/t$ per layer per microbatch for tensor-parallel size $t$, where $b$ is the microbatch size, $s$ the sequence length, and $h$ the hidden size. That's four all-reduces of a $b \times s \times h$ tensor, each moving $2(t-1)/t$ times the tensor's element count per rank in a ring.

The trouble is where they sit. These all-reduces are on the critical path, because the next layer needs their result. Unlike DDP's, they can't start early. Compute their size for $t = 8$, $b \cdot s = 8192$ tokens, and $h = 8192$ in bf16. The tensor is $8192 \times 8192 \times 2 = 134.2$ MB. A ring all-reduce sends $2 \times 7/8 \times 134.2 = 234.9$ MB per rank, which over 300 GB/s NVLink takes $234.9 \times 10^{6} / 300 \times 10^{9} = 0.78$ ms. The MLP's FLOP on those tokens, with a $4h$ intermediate size, is $2 \times 8192 \times (2 \times 8192 \times 32768) = 8.8 \times 10^{12}$, or $1.1 \times 10^{12}$ per rank at $t = 8$, which takes at least 3.5 ms at 312 TFLOP/s. So an exposed all-reduce adds about 22% to the MLP's floor over NVLink. Over PCIe the same bytes take $234.9 \times 10^{6} / 31.5 \times 10^{9} = 7.5$ ms, longer than the compute. That is the reason tensor parallelism stays inside one NVLink-connected node. The companion course reaches the same conclusion for inference with its own link numbers.

### Sequence parallelism

Megatron's *sequence parallelism*, from [Korthikanti et al.](https://arxiv.org/abs/2205.05198), targets what tensor parallelism leaves replicated: the LayerNorm and dropout between the parallel blocks, which run on the full sequence on every rank. It splits those along the sequence dimension and replaces each all-reduce with a reduce-scatter before them and an all-gather after. The pair moves the same bytes as the all-reduce, since a ring all-reduce is a reduce-scatter followed by an all-gather, and it cuts the activation memory of those regions by $t$.

## Pipeline parallelism: the bubble

This section answers: how much of a pipeline's time is idle, and how do schedules shrink it?

Pipeline parallelism cuts the model into $p$ *stages* of consecutive layers, one per GPU, and sends each stage's output activations to the next stage. The kernel's view is mostly compute, with a *point-to-point* send and receive between stages, a NCCL `send` and `recv` kernel pair on a communication stream. The cost is idle time. Stage 2 can't start until stage 1 produces something, so at the start of a batch most stages wait, and at the end they wait again.

Split the batch into $m$ *microbatches* so the stages can work on different ones at once. The idle time at the start and end is the *bubble*. [GPipe](https://arxiv.org/abs/1811.06965) runs all $m$ forward passes and then all $m$ backward passes. Per the Megatron paper, the bubble time as a fraction of the ideal compute time is

$$
\boxed{\frac{t_{\text{bubble}}}{t_{\text{ideal}}} = \frac{p - 1}{m}}
$$

It's independent of how long the forward and backward passes take. With $p = 8$ stages and $m = 8$ microbatches the bubble is $7/8 = 87.5\%$ of the ideal time. That's the same as $7/15 = 46.7\%$ of the pipeline's wall-clock time being idle, because wall-clock is ideal plus bubble, $(m + p - 1)/m$ times ideal. With $m = 32$ the bubble is $7/32 = 21.9\%$ of ideal, and $7/39 = 17.9\%$ of wall-clock. The fix for a bubble is more microbatches.

### 1F1B

More microbatches in GPipe cost memory, because the schedule keeps the activations of every microbatch until its backward pass runs, and all $m$ forwards come first. The *1F1B* schedule, one forward then one backward, starts a microbatch's backward pass as soon as it can: each stage runs a few warm-up forwards, then alternates one forward and one backward. The bubble is the same, $(p-1)/m$, per the Megatron paper, but the number of microbatches a stage has in flight is at most $p$, not $m$. At $m = 32$ and $p = 8$ that's 8 sets of stashed activations instead of 32.

```viz
22-pipeline-bubble
```

### Interleaving

An *interleaved* schedule gives each GPU $v$ smaller chunks of layers instead of one big one, for example layers 1 to 2 and 9 to 10 instead of 1 to 4. The pipeline is then $v$ times deeper and each step $v$ times shorter, and the Megatron paper gives the bubble as $\frac{1}{v}\frac{p-1}{m}$. At $p = 8$, $m = 32$, and $v = 4$, that's $7/128 = 5.5\%$ of ideal, down from 21.9%. The cost is communication, which goes up by a factor of $v$ because activations cross a GPU boundary $v$ times as often, and the schedule requires $m$ to be a multiple of $p$.

Between stages, the Megatron paper's scatter-gather optimization avoids sending the same tensor from every tensor-parallel rank. The sender splits the activation into $t$ chunks, each rank sends its chunk over its own network card, and the receiver all-gathers over NVLink. That cuts the cross-node bytes from $b\,s\,h$ to $b\,s\,h / t$, and the paper reports up to 11% higher throughput for communication-heavy schedules.

## Expert parallelism: the all-to-all

This section answers: what happens when a mixture-of-experts layer spans GPUs?

In a mixture-of-experts layer ([chapter 20](/c/20-moe-and-sparse-kernels)), a router sends each token to its top-$k$ experts. With *expert parallelism* the experts live on different GPUs, so each GPU must send each token to the GPU that holds its experts, wait for the experts to run, and get the outputs back. [GShard](https://arxiv.org/abs/2006.16668) uses an *all-to-all* for the exchange. A forward pass has two per MoE layer, a *dispatch* before the experts and a *combine* after, and the backward pass has two more.

Size one for 16,384 tokens per rank, top-$k = 2$, hidden size 4096, in bf16: each rank dispatches $16384 \times 2 \times 4096 \times 2 = 268.4$ MB. With 8 ranks, $7/8$ of it leaves the GPU, which is $234.9$ MB, and over 300 GB/s NVLink that takes $0.78$ ms, once for dispatch and once for combine. This assumes the router balances tokens evenly. It doesn't, which is the second difficulty: the busiest expert's rank sets the pace, and the all-to-all's message sizes differ per pair.

Two things set expert parallelism apart in the kernel's view. The all-to-all is on the critical path like tensor parallelism's all-reduce, since the experts need the tokens before they can run. And the work between the two exchanges is a grouped GEMM over uneven expert batches, as in chapter 20, whose shape depends on the router. The standard remedy is to split the batch into two microbatches and offset them: while one microbatch's tokens are in the all-to-all, the other's experts compute. The [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437) describes two versions of this. In training, its DualPipe schedule pairs a forward chunk with a backward chunk so that one's all-to-all runs while the other computes, and the report says communication uses 20 of the H800's 132 SMs. In inference prefill, it overlaps one microbatch's attention and MoE compute with the other's dispatch and combine. Both need communication that runs on few enough SMs that the compute kernel doesn't slow down, which leads to the next section.

## Overlap inside a kernel

This section answers: why does stream-level overlap stop short, and what do kernels that communicate do instead?

Stream-level overlap has two costs. A communication kernel is a kernel: NCCL's collectives run on the GPU's SMs, so an all-reduce on a side stream takes some SMs from the compute stream's GEMM. They also share HBM bandwidth, so a bandwidth-bound compute kernel slows as it overlaps with a collective. And overlap needs independent work: in tensor parallelism, the all-reduce after a layer and the next layer's matmul are dependent, so there's nothing in another stream to run alongside.

The way past the dependence is to cut finer. If a matmul's output is split into tiles, the first tiles are done long before the last, and their reduce-scatter can start while the rest still compute. That's the same idea as chapter 12's `cp.async` pipeline and this chapter's chunked copies, applied to communication. It appears in two forms:

- **Decomposition at the host level.** PyTorch's [async tensor parallelism](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487) splits the matmul into chunks and communicates each chunk's result while the next chunk computes, using the `SymmetricMemory` abstraction so that GPUs write directly into each other's memory through peer-to-peer access. The announcement reports up to about 29% forward speedup and 8% end-to-end on Llama 3 7B, and up to about 20% forward and 8% end-to-end on Llama 3 70B; the post names the hardware. The `SymmetricMemory` API is documented as alpha, so check [its page](https://docs.pytorch.org/docs/2.14/symmetric_memory.html) before relying on a signature.
- **Fusion into one kernel.** [Flux](https://arxiv.org/abs/2406.06858) maps each tile of the GEMM and its communication to one thread block of a single fused kernel, on CUTLASS. The authors argue that stream-level decomposition gives no precise control over timing and leaves the GPU underused when the pieces are small. They report that a fused kernel can potentially overlap up to 96% of the communication, with up to 1.24x training speedup over Megatron-LM on a cluster of 128 GPUs; the paper states the hardware.

Both show the trade of the lab's chunk count: a chunk too small wastes time on launch and synchronization, and one too large leaves a long exposed tail.

### Choosing a strategy

The strategies compose, and the question at each step is what has to move and what runs in the meantime.

| Strategy | Collective, per step | On the critical path? | How it overlaps |
|---|---|---|---|
| DDP | One all-reduce per bucket (25 MiB default) | No: starts during backward | Hooks launch each bucket on the NCCL stream |
| ZeRO-3 and FSDP | All-gather twice and reduce-scatter once per unit | Only if not prefetched | A separate stream, and prefetching the next unit |
| Tensor parallelism | Two all-reduces per layer forward, two backward | Yes | Decomposed or fused matmul with communication |
| Pipeline parallelism | A send and a receive per microbatch per stage boundary | Partly: the bubble is idle time | More microbatches, 1F1B, interleaving |
| Expert parallelism | Two all-to-alls per MoE layer forward, two backward | Yes | Offset two microbatches so one computes while the other communicates |

A common layout keeps tensor parallelism inside a node, where NVLink carries its critical-path all-reduces, and puts pipeline or data parallelism across nodes.

## What goes wrong

- **Pageable memory.** A tensor that isn't pinned makes `non_blocking=True` a no-op. The code runs, gives the right answer, and shows no overlap. Check `is_pinned()` and time the copy alone first.
- **Reading a result too early.** A `non_blocking` copy to the host returns before the data is there. Read the destination only after an event or a stream wait that is ordered after the copy. NaN-filling the output in a test turns the silent bug into a visible one.
- **No wait before the side stream starts.** A side stream isn't ordered against the default stream. If the default stream produced the input, `s.wait_stream(default)` must come first, and the same holds even when the side stream only writes, because the allocator may have handed it memory with pending work from a freed tensor.
- **A host synchronize in the loop.** A `.item()`, `.cpu()`, or `torch.cuda.synchronize()` between chunks stops the host from queueing, and the pipeline collapses to the sequential time.
- **One stream, many chunks.** Chunks in a single stream run in order, so chunking with `n_streams = 1` adds launch overhead and no overlap.
- **Freed under a running kernel.** A tensor used on a side stream and dropped on the allocating stream can be reused while the kernel still reads it. Use `record_stream`, or keep the reference until the join.
- **Chunks too small.** Past some chunk count the per-launch cost dominates and the time rises again. The model says the time falls with $n$, and the measurement says where it stops.
- **Counting hidden communication as free.** An overlapped all-reduce still takes SMs and memory bandwidth, so the compute kernel beside it runs slower than it did alone.
- **A bubble you tried to hide.** Pipeline idle time is waiting on a dependency, and a different stream can't fill it. Only more microbatches or a better schedule shrink it.

> [!RECAP]
> - A stream orders work within itself and nothing across streams. An event marks a point, and `wait_event` or `wait_stream` makes another stream wait for it without blocking the host.
> - A copy overlaps with a kernel when the host memory is pinned, the copy is asynchronous, and the two are in different non-default streams. Pageable memory goes through a staging buffer and synchronizes.
> - A chunked pipeline takes $T_n = s_1 + s_2 + s_3 + (n-1)\max(s)$ and approaches $\max(t)$. For three equal 8.52 ms stages that's 25.6 ms sequential, 10.65 ms at 8 chunks, and 8.52 ms at the limit.
> - The caching allocator reuses a freed block without waiting for other streams, so a tensor used across streams needs `record_stream` or a join before it's freed.
> - DDP overlaps one all-reduce per 25 MiB bucket with backward. FSDP and ZeRO-3 add all-gathers and a reduce-scatter per unit and prefetch them. Tensor parallelism puts two all-reduces per layer on the critical path, so it stays on NVLink. Pipeline parallelism has a bubble of $(p-1)/m$, and expert parallelism has two all-to-alls per layer.
> - Communication overlaps only if it's independent of the compute beside it, and it costs SMs and bandwidth. Chunked, decomposed, and fused matmul-plus-communication kernels cut the dependency finer.

## Check your understanding

> [!QUESTION] A copy uses `non_blocking=True`, but the profile shows the kernel starting only after the copy ends. Name two causes.
> The host tensor may be pageable, which makes the copy synchronous in effect (check `is_pinned()`), or the copy and the kernel may be in the same stream, where they run in order. A third cause is a host synchronize between them.

> [!QUESTION] The lab tensor is 256 MiB and the link is PCIe Gen4 x16. What is the shortest possible host-to-device time, and why is the real one longer?
> $268.4 \times 10^{6} / 31.5 \times 10^{9} = 8.52$ ms is the per-direction link rate's floor. A real transfer pays for protocol overhead and the host's memory system, so it lands above the floor, and the lab reports the measured time.

> [!QUESTION] Three stages take 8, 12, and 6 ms. What are the sequential time, the bound, and the time with 6 chunks?
> Sequential is $8 + 12 + 6 = 26$ ms and the bound is 12 ms. With 6 chunks the stages are $1.33$, $2$, and $1$ ms, so $T_6 = 4.33 + 5 \times 2 = 14.33$ ms, which is 84% efficient against the bound.

> [!QUESTION] A kernel on stream `s` reads tensor `A`, which was allocated on the default stream. The code then runs `del A`. What can go wrong, and what fixes it?
> The allocator returns the block to the default stream's pool at once, so a later allocation can reuse it while the kernel on `s` still reads it. Call `A.record_stream(s)` before the `del`, or keep `A` alive until the default stream has waited for `s`.

> [!QUESTION] Why does DDP fill buckets in the reverse order of `model.parameters()`?
> Backward visits layers from the last to the first, so gradients become ready in roughly the reverse of the parameter order. Filling buckets that way lets the first bucket complete early and start its all-reduce while the rest of backward runs.

> [!QUESTION] Compare the all-reduce time of a 14 GB gradient over 8 ranks on NVLink (300 GB/s per direction) and on PCIe (31.5 GB/s), and say which one a 0.735 s backward pass can hide.
> The ring sends $1.75 \times 14 = 24.5$ GB per rank, so NVLink takes 82 ms and PCIe 778 ms. The backward floor of 0.735 s hides the NVLink all-reduce with room to spare, and not the PCIe one, which is slightly longer than the backward floor itself.

> [!QUESTION] With $p = 8$ pipeline stages, how many microbatches keep the bubble under 10% of ideal time, and what changes with an interleave of $v = 4$?
> The bubble is $(p-1)/m = 7/m$, so under 10% needs $m > 70$, say 71 or more. With $v = 4$ the bubble is $7/(4m)$, which falls under 10% at $m > 17.5$. The schedule needs $m$ to be a multiple of $p$, so $m = 24$ is the first choice, at the price of four times the point-to-point communication.

> [!QUESTION] Why can't a second stream hide the tensor-parallel all-reduce the way it hides DDP's?
> The next layer's matmul needs the all-reduce's result, so no independent compute is available to run beside it. The remedy is to cut the matmul and the communication into tiles, so tile $k$'s communication overlaps tile $k+1$'s compute.

## Lab

> [!TRY] A chunked pinned-memory pipeline
> The starter gives you a Triton kernel that applies an iterated fused multiply-add to a float32 tensor, and the helpers that time it. You write `make_pinned`, `chunk_bounds`, and two pipelines, `pipeline_streams` (a stream per chunk) and `pipeline_staged` (a stream per stage, with events), plus the time model `pipeline_time_model` and `overlap_efficiency`. You pass when the output matches a float64 reference for sizes that include 1 and a size that isn't a multiple of the chunk count, your pipeline returns to the host before the GPU finishes, the overlapped pipelines take less than 85% of the one-stream time, and the best one reaches at least 40% of the `max(copy, compute)` bound. The lab reports the stage times, the pipeline times, the efficiency, and the copy rate against the 31.5 GB/s per-direction PCIe figure. Budget about two minutes for Triton's first compile.

## Further reading

- [How to overlap data transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/): Mark Harris on streams, pinned memory, copy engines, and the two ways to order a chunked pipeline.
- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html): asynchronous execution, `pin_memory` and `non_blocking`, stream semantics, `record_stream`, and the backward pass.
- [CUDA runtime API: API synchronization behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html): which copies are synchronous and when.
- [PyTorch Distributed: experiences on accelerating data parallel training](https://arxiv.org/abs/2006.15704): gradient bucketing and overlap with the backward pass.
- [ZeRO](https://arxiv.org/abs/1910.02054) and [PyTorch FSDP](https://arxiv.org/abs/2304.11277): sharding optimizer state, gradients, and parameters, and FSDP's streams, prefetching, and rate limiter.
- [Megatron-LM](https://arxiv.org/abs/1909.08053), [Efficient large-scale training on GPU clusters](https://arxiv.org/abs/2104.04473), and [Reducing activation recomputation](https://arxiv.org/abs/2205.05198): tensor, pipeline, and sequence parallelism, with the bubble formulas.
- [GPipe](https://arxiv.org/abs/1811.06965) and [GShard](https://arxiv.org/abs/2006.16668): micro-batching and the pipeline bubble, and expert parallelism with all-to-all.
- [Flux](https://arxiv.org/abs/2406.06858) and [PyTorch async tensor parallelism](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487): communication overlap inside and around matmul kernels.
- [The ultra-scale playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook): a survey of data, tensor, pipeline, context, and expert parallelism.
- The companion course's [tensor parallelism chapter](https://qwen.ifkash.dev/c/19-tensor-parallelism), and this course's [collective communication](/c/21-collective-communication) and [fusion and compilers](/c/17-fusion-and-compilers) chapters.
