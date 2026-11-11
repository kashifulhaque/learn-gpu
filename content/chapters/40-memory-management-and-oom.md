---
title: GPU memory management and OOM debugging
slug: 40-memory-management-and-oom
part: "Part 4 — Kernels for ML"
summary: Count every byte a training step holds, model PyTorch's caching allocator to see why reserved memory outgrows allocated memory, read a memory snapshot, tune the allocator, share a GPU, and choose which layers to checkpoint to fit a byte budget.
minutes: 120
gpu: true
objectives:
  - "Write the byte formula for each thing a training step keeps in GPU memory: weights, gradients, optimizer state, activations, workspace, and the CUDA context."
  - "Describe the PyTorch caching allocator: blocks, segments, the small and large pools, the 512-byte rounding, splitting, merging, and why a freed block is cached and not returned to the device."
  - "Tell memory_allocated from memory_reserved, and explain how fragmentation makes reserved grow while allocated stays flat."
  - "Choose among expandable_segments, max_split_size_mb, and garbage_collection_threshold for a given failure, and say what cudaMallocAsync changes."
  - "Record and read a memory snapshot: the largest free block, the fragmentation ratio, and the allocation behind the peak."
  - "Follow a fixed order to debug an out-of-memory error, and pick the layers to checkpoint so the peak fits a byte budget at the lowest recompute cost."
lab: 40-memory-budget
---

# GPU memory management and OOM debugging

> [!TLDR]
> - A training step holds six kinds of memory. Weights, gradients, and optimizer state cost 16 bytes per parameter for Adam in fp32 or in mixed precision. Activations scale with the batch and the sequence. Workspace and the CUDA context are the rest.
> - PyTorch's caching allocator takes memory from the device in segments and cuts them into blocks. It rounds every request to 512 bytes, keeps freed blocks for reuse, and ==never returns a segment during normal operation==. So `memory_reserved` never falls below its earlier level until something releases the cache, while `memory_allocated` tracks live tensors.
> - Fragmentation is free memory in blocks too small for the next request. The reserved-but-unallocated gap tells you it exists, and a memory snapshot shows where it sits and which allocation caused it.
> - `expandable_segments`, `max_split_size_mb`, and `garbage_collection_threshold` each target one failure. Try them after you have found the cause, not before.
> - To fit a byte budget, choose which layers to checkpoint. The peak is either the stored total or the rebuild of one layer on top of the layers below it, and a small dynamic program finds the cheapest set.

An out-of-memory error is the most common way a training run fails, and the message is the least informative part of it. It names the size of the allocation that failed. It doesn't say whether the model is too big, the batch is too big, a tensor leaked three hundred steps ago, or the allocator broke free memory into pieces that no longer fit the request. Each of those needs a different fix, and a wrong guess costs a restart.

This chapter gives you the tools to tell them apart. [Chapter 19](/c/19-training-kernels) counted the activations of one layer and showed recomputation as a trade of arithmetic for bytes. [Chapter 22](/c/22-overlap-and-parallelism) explained why the allocator reuses a freed block before a side stream finishes with it. Here you go further on both: you write down every byte a step holds, build a working model of the allocator, and read the file PyTorch writes when you ask where the memory went.

The lab has you build the calculator, a small allocator simulator, a snapshot parser, and a checkpoint chooser, all in plain Python. If an A100 is attached, the lab then measures the real peak with and without checkpointing and compares it with the calculator.

## Before you start

**Recomputation.** Activation checkpointing keeps a layer's input, drops its internals, and runs the layer's forward pass again when the backward pass needs them. [Chapter 19](/c/19-training-kernels) explains the technique and its cost. This chapter takes the byte accounting for granted and asks which layers to apply it to.

**Streams and the allocator.** [Chapter 22](/c/22-overlap-and-parallelism) covers per-stream pools, `record_stream`, and the rule that `del x` returns a block to the pool at once. You need that rule here, because it is why a freed tensor doesn't lower `memory_reserved`.

**Fusion and launch overhead.** [Chapter 17](/c/17-fusion-and-compilers) shows how `torch.compile` and CUDA graphs change what a step allocates. A fused kernel never writes the intermediate tensor, which is the cheapest memory saving there is.

**Units.** Sizes of memory are binary in this chapter: 1 MiB is $2^{20}$ bytes and 1 GiB is $2^{30}$ bytes. The A100 80GB is sold as 80 GB, and the chapter treats it as 80 GiB. Treat the figure as a round number: the usable amount is lower, as the next section explains.

## What sits in GPU memory

This section answers one question: when a training step runs out of memory, what was the memory holding? Six things compete for it, and three of them depend only on the model.

| Item | Bytes | Depends on | When it exists |
|---|---|---|---|
| Weights | $b_w N$ | The parameter count $N$ | The whole run |
| Gradients | $b_g N$ | $N$ | From the first backward pass on, until you free them |
| Optimizer state | $b_o N$ | $N$ and the optimizer | After the first optimizer step |
| Activations | $\sum_i A_i$ | Batch, sequence, architecture | Built during forward, released during backward |
| Workspace | Kernel scratch | The kernels chosen | Short-lived, but it counts toward the peak |
| CUDA context | A fixed cost | The driver and the kernels loaded | Once per process |

### The three that scale with parameters

For Adam, the optimizer state holds two numbers per parameter: a running mean of the gradient, $m$, and a running mean of its square, $v$. Both are kept in fp32, so the state costs 8 bytes per parameter. Counting them as one copy is the most common error in a hand estimate, and the lab's first check catches it.

Add the weights and gradients. With fp32 training, each is 4 bytes. With mixed precision, the weights and gradients are 2 bytes each in bf16, and the optimizer also keeps a 4-byte fp32 master copy of the weights, so that a small update isn't rounded away. Write $b_w$ for the weight bytes, $b_g$ for the gradient bytes, and $b_m$ for the master-copy bytes:

$$
\boxed{M_{\text{state}} = \big(\hla{b_w} + \hlb{b_g} + \hlc{b_m} + 8\big)\,N}
$$

The three coloured terms are the weights, the gradients, and the master copy, and the 8 is Adam's two fp32 moments. For fp32, $4 + 4 + 0 + 8 = 16$. For mixed precision, $2 + 2 + 4 + 8 = 16$. The two recipes cost the same per parameter, which surprises people who expect mixed precision to halve the model's footprint. It halves the weights that the forward and backward GEMMs read and the activations they save, and it leaves the optimizer's share unchanged. Chapter 19's table gives the same 16.

A pure bf16 recipe that keeps no master copy and holds Adam's moments in fp32 costs $2 + 2 + 0 + 8 = 12$ bytes, with the rounding risk that the master copy exists to remove.

> [!EXAMPLE] A 1.2 billion parameter model
> Take a decoder with hidden size $h = 2048$ and 24 layers. A layer holds about $12h^{2}$ parameters, from the four attention projections and a 4h MLP, so $N = 12 \times 2048^{2} \times 24 = 1{,}207{,}959{,}552$. At 16 bytes per parameter the state is $19{,}327{,}352{,}832$ bytes, which is exactly 18.00 GiB. The embedding table adds to this and the example leaves it out.
>
> That is 22.5% of an 80 GiB card, before one activation exists. `scripts/figures/40-memory-management-and-oom.py` prints these numbers.

ZeRO shards the optimizer state, the gradients, and eventually the weights across ranks, which divides the matching terms by the number of GPUs. [Chapter 25](/c/25-ddp-and-zero) writes the first stage. The formula above is what each rank holds before sharding.

### Activations, workspace, and the context

Activations are the term you control. [Chapter 19](/c/19-training-kernels) derived that a GPT-style layer saves $34sbh + 5as^{2}b$ bytes for backward, with the second term gone under fused attention, where $s$ is the sequence length, $b$ the batch, $h$ the hidden size, and $a$ the head count. The stack adds up across layers, and checkpointing changes what each layer keeps. The last section of this chapter chooses which layers to checkpoint.

*Workspace* is memory a kernel borrows while it runs. A GEMM may ask for scratch space for a split-K reduction, a convolution for an algorithm's temporary buffers, an optimizer's multi-tensor kernel for staging lists. Workspace is short-lived, so it doesn't appear in the sum of live tensors for long, but it counts toward the peak when it overlaps the biggest activations. You measure it. The calculator can't derive it.

The *CUDA context* is the per-process state the driver creates on first use of the GPU: the loaded kernels and the driver's bookkeeping. Its size depends on the driver and on how many kernels your libraries load, and it is typically hundreds of MiB (U in `docs/FACTS.md`: no source in this chapter fixes the figure). `torch.cuda.mem_get_info()` returns the driver's free and total bytes, so the difference between the card's size and the total it reports, plus what `nvidia-smi` shows before you allocate a tensor, is your context. Measure it once on your own stack.

### Why the first step under-reports

Two of these terms, the optimizer state and the gradients, don't exist at the start of a run. The first forward pass shows a peak lower than the steady state, and the OOM often arrives at step 2 or step 3, when the optimizer state has appeared. A run that survives one full step with its optimizer has, for the first time, shown you its real floor.

```viz
40-budget-bar
```

The figure puts the 1.2 billion parameter example on an 80 GiB card. State is 18.00 GiB. At batch 12 with fused attention, 24 layers save 76.50 GiB of activations, so the step needs 94.50 GiB before headroom and doesn't fit. Checkpointing layers 0 to 6 cuts the activation peak to 55.50 GiB. With 4 GiB of headroom, an assumption of this chapter and not a measurement, the step needs 77.5 GiB and fits.

> [!KEY] Count first, then measure
> The formulas give you the floor of a step in minutes. If `torch.cuda.max_memory_allocated()` lands far above your count, something you didn't count is alive: a second copy of the model, a logits tensor, a list of losses that kept their graphs.

## The caching allocator

This section answers what happens between `torch.empty` and the driver. The answer decides whether a freed tensor lowers the number `nvidia-smi` shows, and it is why the allocator exists.

A call to `cudaMalloc` is slow and implicit-synchronizing, and a training step allocates thousands of tensors. PyTorch's *caching allocator* asks the driver for large pieces, hands out parts of them, and keeps the pieces it has been given. The constants below come from the PyTorch allocator source, `c10/cuda/CUDACachingAllocator.cpp`, and from the allocator's configuration header. Chapter 22 describes the same allocator for the stream rules.

### Blocks, segments, and two pools

A *segment* is one `cudaMalloc` allocation. A *block* is a piece of a segment. A block is either in use by a tensor or free. The allocator keeps two pools, and each request goes to exactly one:

| Property | Small pool | Large pool |
|---|---|---|
| A request goes here when its rounded size is | at most 1 MiB | above 1 MiB |
| The segment it takes from the driver | 2 MiB | 20 MiB if the request is under 10 MiB, otherwise the request rounded up to a multiple of 2 MiB |
| A block splits when the remainder is | at least 512 bytes | above 1 MiB |

The allocator also rounds every request first. A size under 512 bytes becomes 512, and anything else rounds up to the next multiple of 512. A 1000-byte tensor takes a 1024-byte block, a 1-byte tensor takes 512, and a request of 1,000,000 bytes takes 1,000,448, which is $1954 \times 512$.

That rounding is why the numbers `memory_allocated` reports are a little larger than the sum of your tensors' sizes, and why a large allocation placed in a 20 MiB segment has a tail that other requests can use.

> [!NOTE] What the lab confirms, and what it assumes
> The lab's allocator uses these constants, which the PyTorch source defines: a minimum block of 512 bytes, a 1 MiB small-request limit, 2 MiB small segments, a 10 MiB threshold, 20 MiB large segments, and 2 MiB rounding above the threshold. The PyTorch CUDA notes document the 20 MB default for the large segment size, and the source comment says small requests pack into 2 MB buffers. The split rule, the choice of the smallest fitting block, and counting a block's rounded size in `memory_allocated` come from reading the source; none of the three is stated in the user documentation, and `docs/FACTS.md` marks them U.

### A request, step by step

Here is a request in words. The allocator rounds the size. It picks the pool. It looks for the smallest free block in that pool that fits, and takes it. If the block is much larger than the request, it splits the block in two: the front part serves the request, and the remainder stays free in the pool. If no free block fits, the allocator calls `cudaMalloc` for a new segment of the size in the table, and carves the request from that.

When you free a tensor, the allocator marks its block free. If the block's neighbors inside the same segment are free too, it merges them into one larger block. The block goes back to the pool and the segment stays reserved. Chapter 22's rule holds: this happens at once, without waiting for the GPU.

If `cudaMalloc` fails, the allocator doesn't give up yet. It frees the cached segments that hold no live tensor and tries again. Only when the retry fails does it raise `OutOfMemoryError`. The source has helpers that release whole free segments, and the lab models this rule: a segment that is a single free block goes back to the device, and anything with a live block in it stays.

```viz
40-allocator-steps
```

The figure runs the lab's recorded trace through that model. Four 12 MiB tensors take four segments. Freeing the first and third leaves 24 MiB cached but in two separate segments, and the 16 MiB request that follows fits in neither, so it takes a new 16 MiB segment. Reserved climbs to 64 MiB while allocated is 40.

The next request, 4 MiB, takes the smallest free block that fits, which is a 12 MiB one, and splits it: 4 live and 8 free. The last request, 20 MiB, finds free blocks of 8, 12, and 16 MiB, and none holds 20. The allocator reserves a new segment. At the end 48 MiB is live and 84 MiB is reserved.

### Allocated and reserved

PyTorch reports the two numbers that matter. The documentation says `memory_allocated()` monitors the memory occupied by tensors, and `memory_reserved()` monitors the total memory the caching allocator manages, including cached blocks no tensor uses. `nvidia-smi` shows reserved plus the context, because the cached memory still belongs to your process. The documentation notes this: unused memory that the allocator manages still shows as used.

The gap between the two is the cache:

$$
\text{cached} = \hla{\text{reserved}} - \hlb{\text{allocated}}
$$

Here $\hla{\text{reserved}}$ is the sum of the segment sizes and $\hlb{\text{allocated}}$ is the sum of the live block sizes. A gap is normal. A gap that is large and that coincides with an OOM is the signature of fragmentation, which the next section explains.

`torch.cuda.empty_cache()` releases cached segments that hold no live tensor. It doesn't touch live tensors. Use it between phases that have very different memory shapes, such as after evaluation, or to let another process in. Don't call it every step: the next allocation pays `cudaMalloc` again.

> [!WARNING] Reserved never shrinks on its own
> A freed tensor leaves reserved unchanged. A run that holds `reserved` at 60 GiB after a spike isn't leaking: the allocator is keeping what it was given. Read `memory_allocated` to learn what your code holds, and `max_memory_reserved` to learn what the run needed.

## Why reserved grows: fragmentation

This section answers why a step can fail with gigabytes of free memory in the cache. The cause is that free memory is not one pool of bytes: it is a set of blocks, and a request needs one block at least as large as itself.

Two kinds of waste appear. *Internal* fragmentation is the part of a block that the tensor doesn't use: the rounding to 512 bytes, the tail of a 20 MiB segment that is too small to split off, the case where a request of 11 MiB takes a 12 MiB segment and the 1 MiB remainder isn't above the split threshold, so the block keeps all 12 MiB. *External* fragmentation is free blocks that no request can use because they are too small or sit in different segments.

External fragmentation is the one that causes OOMs, and it has three common sources.

- **Sizes that change.** A variable batch, a variable sequence length, or a dynamic-shape model asks for a different size each step. A freed block of the old size doesn't fit the new, larger request, so the allocator adds a segment. Over many steps, the cache fills with blocks of sizes that no longer occur.
- **Lifetimes that interleave.** A long-lived tensor allocated between two short-lived ones pins the segment. When the short ones die, the free blocks on each side of the long-lived tensor can't merge. Freed memory is contiguous only if nothing live sits between its pieces.
- **Per-stream pools.** The allocator keeps blocks per stream, as chapter 22 explains. A side stream that allocates large buffers, as the all-gather stream does in FSDP, holds them in its own pool, and the compute stream can't use them.

Segments, once created, stay reserved, so a run that hit a large spike early keeps the segments from the spike, split into whatever sizes the later steps asked for.

### A number for it

A single number summarizes how scattered free memory is. Take the free bytes $F$ across all inactive blocks and the largest free block $L$:

$$
\boxed{\text{fragmentation} = 1 - \frac{\hla{L}}{\hlb{F}}}
$$

If all free memory sits in one block, $L = F$ and the ratio is 0. If free memory is in many equal pieces, $L$ is small against $F$ and the ratio approaches 1. The ratio says nothing about whether you are about to fail: that depends on the next request. A request of size $r$ fits if $r \le L$, and an OOM despite $r \le F$ is the fragmentation failure.

> [!EXAMPLE] The lab's snapshot
> The lab's fixture has four segments of 2, 20, 20, and 64 MiB, so 106 MiB is reserved. Live blocks add up to 55 MiB. The inactive blocks are 1, 4, 6, 8, 8, and 24 MiB, so $F = 51$ MiB and $L = 24$ MiB. The ratio is $1 - 24/51 = 0.529$. A 30 MiB request is below $F$ and above $L$, so it fails although 51 MiB is free: the OOM is a fragmentation failure and not a capacity failure.

The fixture is synthesized, with the shape of a real snapshot, and labeled so in its file.

## Tuning the allocator

This section answers which environment variable to reach for. `PYTORCH_CUDA_ALLOC_CONF` is kept as an alias for the current `PYTORCH_ALLOC_CONF`, and its value is a comma-separated list of `option:value` pairs. Three options address the failures above, and one switches the allocator.

| Option | What it does | Use it when |
|---|---|---|
| `expandable_segments:True` | Starts with one segment per stream and grows it by mapping more memory, so a freed region can serve a larger request later. Experimental, default off. | Reserved-but-unallocated memory is large and your sizes vary |
| `max_split_size_mb:N` | Stops the allocator splitting blocks larger than $N$ MiB. Default is unlimited. | Many oversized blocks stay inactive and large requests fail |
| `garbage_collection_threshold:F` | When usage passes the fraction $F$ of the device, frees unused blocks, oldest first. $F$ is between 0 and 1. | Throughput drops from repeated `cudaMalloc` retries |
| `backend:cudaMallocAsync` | Replaces the native allocator with CUDA's stream-ordered one. | You want driver-managed pools, and accept the next section's trade |

All of these apply to the whole process. `max_split_size_mb` and `garbage_collection_threshold` are ignored under `cudaMallocAsync`.

### Expandable segments

Without the option, an allocation above 2 MiB calls `cudaMalloc` with a fixed size, and the segment keeps that size. With it, the allocator reserves one growing segment per stream and extends it by mapping more memory as needed. A freed region at the end of the range can then serve a later, larger request, which removes the main source of external fragmentation under changing shapes. It is experimental, so test it on your workload before you rely on it, and expect a different set of failure modes from native segments.

### Splitting and collection

`max_split_size_mb` works against fragmentation by keeping big blocks whole. A block at or above the limit isn't split, and the allocator won't hand such a block to a request below the limit. That reserves large blocks for large requests, at the price of internal waste: a small request that would have used part of a large block now takes a new segment. The PyTorch notes recommend it only as a last resort for an OOM with many inactive split blocks.

`garbage_collection_threshold` addresses a different problem. When the cache is nearly full, every new segment triggers the retry path: the allocator frees all cached segments, which is slow, and tries again. The threshold makes it reclaim old unused blocks earlier and in smaller amounts. The effect is on speed, not on the maximum a step can hold.

> [!TIP] Change one option, then compare snapshots
> Record a snapshot before and after you set an option, and compare the fragmentation ratio and the peak reserved. An option that doesn't move either number isn't helping, however plausible the theory.

## Memory snapshots

This section answers where the memory went. A snapshot records every segment, every block, and, if you ask, the call stack of every allocation.

### Record and dump

Start the recorder before the code you want to watch, and dump after:

```python
import torch

torch.cuda.memory._record_memory_history(max_entries=100_000)
run_a_few_steps()
torch.cuda.memory._dump_snapshot("snapshot.pickle")
torch.cuda.memory._record_memory_history(enabled=None)
```

`_record_memory_history` takes `enabled`, where `"all"` keeps both the current state and the history of allocations and frees. It also takes `context` and `stacks`, which control how much of the Python and C++ call stack each event keeps, and `max_entries`, the size of a ring buffer of events. Set a limit for a long job so the history doesn't grow without bound. `_snapshot()` returns the dictionary, and `_dump_snapshot(filename)` writes it as a pickle file. These underscored functions are PyTorch's documented interface for the feature.

A job that fails with an OOM can dump the snapshot from inside its exception handler, which gives you the allocator's state at the moment of failure. The history's `oom` entry includes `device_free`, the free bytes the driver reported.

### The shape of a snapshot

The dictionary has two parts. `segments` lists the `cudaMalloc` segments, each with its `address`, `total_size`, `stream`, `segment_type` (`"small"` or `"large"`), `allocated_size`, `active_size`, and `blocks`. A block has a `size`, a `requested_size`, an `address`, a `state`, and `frames`, the stack that allocated it. The state is `"active_allocated"` for a live tensor, `"active_awaiting_free"` for a block freed by the user but still waiting on another stream, or `"inactive"` for a block that is free for reuse.

`device_traces` holds one list of events per device. Each has an `action`: `segment_alloc`, `alloc`, `free_requested`, `free_completed`, `segment_free`, `oom`, and a few more. These are the entries you replay to find the peak.

The lab's fixture, `fixture_snapshot.py`, is a trimmed dictionary with this shape. It is synthesized: nobody recorded it. Its first frame of each stack is the innermost call, an ordering the documentation doesn't spell out, so the lab states the convention and `docs/FACTS.md` marks it U.

### Read the viewer, then replay in code

PyTorch hosts a viewer at `pytorch.org/memory_viz`. You drag the pickle onto the page, and it runs in the browser without uploading the file. It offers an *Active Memory Timeline*, which plots live tensors over time and shows an allocation's stack on hover, and an *Allocator State History*, which steps through events and shows the segments and blocks at each.

Use this reading order:

1. Find the plateau. The flat band under everything is the weights, and the optimizer state once it exists. It should match your formula.
2. Find the sawtooth. Each step builds activations and releases them. A saw whose peak creeps upward from step to step is a leak: something holds a reference, such as a loss accumulated without `.item()`.
3. Find the peak. Hover over the widest block at the top, and read its stack. That is the allocation to attack.
4. Find the OOM entry. In the Allocator State History, go to the `oom` event, and compare the requested size with the largest `inactive` block. If a block that big exists, something else failed. If the free bytes add up to enough but no block does, you have fragmentation.

The viewer is for looking. For a number you can track across runs or put in a test, replay the history in code. Add each `alloc`, subtract each `free_requested`, and take the maximum. The lab does exactly that, and it ignores `free_completed`, because the tensor stopped existing at the request. Counting both subtracts the block twice and undercounts the peak.

> [!EXAMPLE] The peak in the fixture
> The fixture's trace holds 55 MiB at the end. Replaying it finds a peak of 79 MiB at the 24 MiB allocation from `loss.py:31`, the log-softmax buffer that lives on top of the 55 MiB. The allocation behind the peak isn't the biggest tensor in the model: it is the one that arrived last, on top of everything else. The final state alone, 55 MiB, would have missed it.

## Beyond the native allocator

This section answers what else the memory system offers: the stream-ordered allocator, pinned memory, and unified memory. Each trades something against the native allocator.

### cudaMallocAsync and stream-ordered pools

`cudaMallocAsync` and `cudaFreeAsync` order allocation and free within a stream, so they don't force the device-wide synchronization that `cudaMalloc` and `cudaFree` do. The CUDA programming guide describes the memory pool behind them. Each device has a default pool. Memory freed in a stream can be reused at once by a later allocation in that stream, and after the host synchronizes with the stream, it is available to any stream.

A pool's *release threshold* sets how many bytes it keeps before returning memory to the operating system. Set it to the maximum value and the pool never shrinks on its own, which is the native allocator's behavior. `cudaMemPoolTrimTo` shrinks it on request. Three reuse policies decide when the pool may hand freed memory to another stream, and turning off the opportunistic and internal-dependency ones makes the pattern of reuse more deterministic.

PyTorch exposes it as `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync`, available from CUDA 11.4. It applies to every device in the process. The native allocator's tuning options don't apply to it. What you gain is the driver's pool in place of PyTorch's blocks: one allocator across libraries that share a process. What you lose is the native allocator's knobs and its snapshot tooling as documented, so check that your debugging flow still works before you switch.

### Pinned and unified memory

*Pinned* host memory is page-locked, which lets the copy engines transfer to and from it asynchronously. [Chapter 22](/c/22-overlap-and-parallelism) covers the copy engines and the overlap they allow. For the memory budget, note two points. Pinned memory comes out of host RAM that the operating system can't page out, so pinning too much starves the host. PyTorch also caches pinned allocations, and the history recorder can include them with `record_pinned_host_memory`.

*Unified* (managed) memory, from `cudaMallocManaged`, is one address space that the CPU and GPU both use. Pages migrate to the processor that touches them, by a fault-and-migrate mechanism. On systems with full support it lets an array exceed GPU memory, at the cost of page faults and migrations. `cudaMemPrefetchAsync` moves a range ahead of use, and `cudaMemAdvise` gives hints. The CUDA guide notes that the hints cost host time and recommends them only when they measurably help. Treat managed memory as a way to make something fit, not to make it fast: a training step that needs it is paging over PCIe, and the roofline of chapter 4 says what that costs.

## Sharing one GPU

This section answers how several jobs can use one GPU, and what each method does to memory. The two NVIDIA mechanisms differ in what they isolate.

### MPS

The *Multi-Process Service* lets kernels from different processes run concurrently, where without it they time-slice. A control daemon starts a server, and the server is the clients' shared connection to the GPU and the point of arbitration among them. Use it when no single process fills the GPU, such as many small inference workers.

On memory, NVIDIA's documentation says MPS clients have fully isolated GPU address spaces. Device memory limits can be set per server, and a client can tighten its own with `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT`, after which an allocation over the limit returns an out-of-memory error. The isolation of faults is weaker: a fatal GPU fault affects every client sharing the GPU, and the clients are told, without being told which one caused it. `nvidia-smi` attributes all client activity to the server process.

### MIG

*Multi-Instance GPU* partitions a supported GPU into isolated instances, each with dedicated compute and memory. On the A100 80GB the documented profiles are 1g.10gb (up to seven instances), 1g.20gb (four), 2g.20gb (three), 3g.40gb (two), 4g.40gb (one), and 7g.80gb (the whole card). The number before `g` is the fraction of the card's streaming multiprocessors, in sevenths, and the number before `gb` is the memory. An instance's memory is a hard ceiling: a process in a 1g.10gb instance runs out of memory at 10 GB however much the rest of the card has free.

The table summarizes the choice.

| Property | MPS | MIG |
|---|---|---|
| Memory isolation | Separate address spaces, optional per-client limits | Dedicated memory per instance |
| Fault isolation | A fatal fault reaches every client on the GPU | Separate instances |
| Granularity | Any number of clients, up to a documented limit | Fixed profiles |
| Use it for | Many small processes that share a GPU's compute | Predictable slices with a hard memory ceiling |

For this chapter's purpose, the point is the budget. Under MIG, your budget is the instance's memory, and the formulas above apply to it. A model whose state is 18 GiB doesn't fit a 1g.10gb instance at all, and the OOM message reports the instance's size, not the card's.

## An order for debugging an OOM

This section answers what to do when the error arrives. Work down the list, and stop at the first step that explains the failure. The order puts the cheap, high-information steps first.

1. **Read the message.** PyTorch's `OutOfMemoryError` reports the size of the failed request, the device's free and total bytes, and how much PyTorch holds as allocated and as reserved but unallocated. It also suggests `expandable_segments` when the unallocated figure is large; `docs/FACTS.md` marks the exact wording U, so read your own version's message. Two numbers decide the next step: reserved-but-unallocated, and the request size.
2. **Classify.** If reserved-but-unallocated is smaller than the request, the failure is capacity: you need less memory, and the allocator isn't to blame. If it is larger than the request, it is fragmentation: the memory exists, in the wrong shape. If neither PyTorch figure accounts for the device's total, something outside the allocator holds memory, such as another process, the CUDA context, or NCCL buffers.
3. **Count.** Add up the terms in this chapter's first section and compare with `max_memory_allocated`. A count far below the measurement means something unexpected is alive.
4. **Find the peak.** Record a snapshot and read the allocation behind the peak. A transient spike, like the log-softmax buffer in the fixture, is a fusion or chunking problem, the kind chapter 19's fused loss addresses. A tall plateau is a state problem.
5. **Cut.** In order of cost: reduce the micro-batch and accumulate gradients, use fused attention and a fused loss, checkpoint layers, shard the optimizer state across ranks with [chapter 25](/c/25-ddp-and-zero), then reduce precision.
6. **Reshape.** For fragmentation, make sizes repeat: pad sequences to a few bucket lengths, allocate long-lived buffers first and once, and try `expandable_segments`. Reach for `max_split_size_mb` last.
7. **Look outside.** Compare `torch.cuda.mem_get_info()` with PyTorch's reserved figure. A large difference points at the context, at other processes, or at a library with its own allocator.

Steps 1 to 3 take a minute and fix a large share of failures. The snapshot is the next most informative tool, and the allocator options come last because they treat a symptom you have already named.

## Fitting a model in a budget

This section answers which layers to checkpoint when you can't afford to checkpoint all of them. Each checkpoint frees bytes and costs a second forward pass through that layer, so you want the cheapest set that fits.

### The peak with some layers checkpointed

Write $F_i$ for the bytes layer $i$ saves when it keeps everything, $I_i$ for the boundary bytes it keeps when checkpointed, and $R_i$ for the cost of running it again. Let $C$ be the set of checkpointed layers. Layer $i$ stores $s_i = I_i$ if $i \in C$ and $s_i = F_i$ otherwise.

Two moments set the peak. After the forward pass, every layer's stored bytes are held, so the total is $\sum_i s_i$. Then the backward pass runs from the last layer to the first. When it reaches a checkpointed layer $i$, the layers above are done and have released their bytes, the layers up to and including $i$ are still stored, and the rebuild brings back $F_i$:

$$
\boxed{\text{peak}(C) = \max\Big(\hla{\textstyle\sum_{j} s_j},\ \max_{i \in C}\big(\hlb{\textstyle\sum_{j \le i} s_j} + \hlc{F_i}\big)\Big)}
$$

The first term is the stored total after the forward pass. The second is the largest rebuild, with its stored prefix and the layer coming back. For all 24 layers checkpointed in the earlier example, the first term is $24 \times 192$ MiB and the second adds one 3264 MiB layer on top of the whole stored prefix.

Where you put a checkpoint matters, and the formula shows why. Checkpointing an early layer costs a rebuild on top of a short prefix. Checkpointing a late layer costs a rebuild on top of nearly every stored byte. The lab's small case has four layers of 10 units each and a boundary of 2. Checkpointing layers 0 and 1 peaks at 24, and checkpointing layers 2 and 3 peaks at 34, though both sets store 24 units. The same count of checkpoints, in a different place, gives a different peak.

The calculator code is short. This is `peak_activation_bytes` from the lab:

```python
def peak_activation_bytes(full, boundary, ckpt):
    ckpt = set(ckpt)
    stored = [boundary[i] if i in ckpt else full[i] for i in range(len(full))]
    peak = sum(stored)
    prefix = 0
    for i in range(len(full)):
        prefix += stored[i]
        if i in ckpt:
            peak = max(peak, prefix + full[i])
    return peak
```

### Choosing the set

With equal layers the answer is a count. In the 24-layer example at batch 12, the card leaves $80 - 18.00 - 4 = 58.00$ GiB for activations. Checkpointing layers 0 to $k - 1$ gives a peak of $76.50 - 3.00\,k$ GiB, since each checkpoint frees $3264 - 192 = 3072$ MiB, which is 3.00 GiB. At $k = 6$ the peak is 58.50 GiB, over the room. At $k = 7$ it is 55.50 GiB, which fits. The cost is 7 of 24 layers rebuilt, which is 29% of one forward pass and about 9.7% of a step that costs three forward-equivalents.

Real models aren't uniform. Layers differ in size, and their rebuild costs differ, because an attention layer and an MLP layer do different amounts of arithmetic. Then the choice is a search. There are $2^{L}$ subsets, and a greedy rule fails. The lab uses six layers with $F$ of 900, 900, 600, 600, 1200, and 1200 MiB, a boundary of 100 MiB each, and costs 9, 9, 6, 6, 12, and 12. With nothing checkpointed the peak is 5400 MiB. For a budget of 4200 MiB the cheapest set is layers 1 and 3, at cost 15. The greedy choice, ranking layers by bytes freed per unit cost, picks layer 4 first, which frees 1100 MiB for 12. It still needs 100 MiB more, so it adds a cheap layer, and ends at 18. The cheapest answer uses two modest layers that together free enough.

An exact method walks the layers in order. Keep a table from "bytes stored by the layers so far" to the cheapest set that reaches that total. For each layer there are two moves. Keeping it whole adds $F_i$ to the total. Checkpointing it adds $I_i$ and is allowed only if the new prefix plus $F_i$ is within the budget. Totals over the budget are dropped. At the end, the cheapest surviving entry is the answer. That checks both terms of the peak: the stored total is bounded by the last prefix, and every rebuild is checked when it is chosen. The table has at most one entry per distinct total, so it is small when sizes are rounded to a unit like 1 MiB.

For the lab's six layers, the results are:

| Budget (MiB) | Layers to checkpoint | Cost |
|---|---|---|
| 5400 | none | 0 |
| 4200 | 1, 3 | 15 |
| 3000 | 1, 3, 4 | 27 |
| 2200 | 0, 1, 3, 4 | 36 |
| 1500 | not feasible | |

The last row is a result too: when even the best set can't fit, the answer is to change something else, such as the batch or the sharding, and not to checkpoint harder. `scripts/figures/40-memory-management-and-oom.py` prints the table, and the lab's check compares the chooser with a brute force over every subset on forty seeded cases.

> [!WARNING] A chooser that ignores the rebuild
> A chooser that checks only the stored total will accept a set whose rebuild of one layer overflows the budget. The step fits until the backward pass reaches that layer, then fails with an OOM that the planning said couldn't happen. The lab's brute-force check catches this by asking for the true peak of the set the chooser returns.

## What goes wrong

- **Trusting `nvidia-smi` for the live size.** It shows reserved plus the context. After a spike it stays high. Use `memory_allocated` for what your code holds and `max_memory_allocated` for its peak.
- **Calling `empty_cache` every step.** It returns segments that the next allocation asks the driver for again. You pay `cudaMalloc` and its synchronization for nothing.
- **Counting one Adam state.** The optimizer holds $m$ and $v$. A fp32 run holds 16 bytes per parameter and not 12.
- **Measuring after the first step only.** The optimizer state appears at the first `optimizer.step()`, and the gradients at the first backward pass. A first-step reading under-reports the floor by up to 12 of the 16 bytes per parameter.
- **Keeping a graph alive.** Appending `loss` to a list keeps its autograd graph and every saved activation. Append `loss.item()`. The snapshot shows this as a peak that creeps up with each step.
- **Setting every option at once.** `max_split_size_mb` and `garbage_collection_threshold` fix different problems, and `max_split_size_mb` can raise reserved memory. Change one, then compare snapshots.
- **Planning from the stored total.** A checkpoint plan that fits the forward pass can overflow while it rebuilds a layer. Check the peak formula, which has both terms.
- **Reading a snapshot's last state as the peak.** The state at the end holds what survived. The peak is in the history.
- **Forgetting that MIG shrinks the card.** Your budget is the instance's memory, and the OOM message names that size.

> [!RECAP]
> - A step holds weights, gradients, optimizer state, activations, workspace, and the CUDA context. State costs $(b_w + b_g + b_m + 8)N$ bytes: 16 per parameter for Adam in fp32 and in mixed precision.
> - The caching allocator rounds to 512 bytes, serves requests from a small and a large pool, takes segments of 2 MiB or 20 MiB or a rounded request, splits blocks, merges frees, and keeps segments until something releases the cache.
> - Reserved minus allocated is the cache. A fragmentation failure is a request below the free total and above the largest free block.
> - A snapshot gives segments, blocks, and a history. Replay the history to find the peak and its allocation, and read the `oom` entry to classify the failure.
> - `expandable_segments`, `max_split_size_mb`, and `garbage_collection_threshold` each address one failure. `cudaMallocAsync` swaps the allocator. MPS shares compute and isolates address spaces, and MIG gives hard slices.
> - The checkpoint peak is the larger of the stored total and the biggest rebuild on its prefix, and a dynamic program over stored bytes finds the cheapest set.

## Check your understanding

> [!QUESTION] A 3 billion parameter model trains in bf16 with a fp32 master copy and Adam. How many bytes do weights, gradients, and optimizer state take, and does it fit an 80 GiB card?
> The recipe holds $2 + 2 + 4 + 8 = 16$ bytes per parameter, so $3 \times 10^{9} \times 16 = 48 \times 10^{9}$ bytes, which is 44.7 GiB. It fits with 35 GiB to spare, and the activations, workspace, and context share that 35 GiB.

> [!QUESTION] After a spike, `memory_allocated` falls to 20 GiB and `memory_reserved` stays at 60 GiB. Is something leaking?
> No. The allocator keeps freed segments for reuse, so reserved doesn't fall when tensors die. The 40 GiB gap is the cache. It becomes a problem only if a later request can't fit in any cached block. Call `torch.cuda.empty_cache()` if another process needs the memory.

> [!QUESTION] The allocator holds blocks of 8, 12, and 16 MiB free in three segments, and a 20 MiB request arrives. What happens, and what does it do to reserved?
> No single free block holds 20 MiB, so the allocator calls `cudaMalloc` for a new 20 MiB segment, and reserved grows by 20 MiB. If that call fails, the allocator releases the cached segments that contain no live block and retries before it raises an error.

> [!QUESTION] A snapshot's inactive blocks are 3 MiB and 2 MiB, and an OOM requests 6 MiB. What is the fragmentation ratio, and what kind of failure is this?
> Free memory is 5 MiB and the largest block is 3 MiB, so the ratio is $1 - 3/5 = 0.4$. The request is above the free total, 6 against 5 MiB, so this is a capacity failure and not a fragmentation failure. Reshaping the allocator won't help.

> [!QUESTION] You replay a trace and subtract the size on both `free_requested` and `free_completed`. What goes wrong?
> You subtract each freed block twice, the running total sits too low after every free, and the maximum you report is below the real peak. The tensor stopped counting at `free_requested`, so that is the only event that subtracts.

> [!QUESTION] Two plans checkpoint two of four layers, each layer saving 10 units with a boundary of 2. One checkpoints layers 0 and 1, the other layers 2 and 3. Which has the lower peak, and why?
> Layers 0 and 1: the peak is 24. Layers 2 and 3 peak at 34. Both store 24 units after the forward pass, but the second plan rebuilds layer 3 on top of the stored prefix of 24, which adds 10 to reach 34, while the first rebuilds layer 1 on a prefix of 4.

## Lab

> [!TRY] Memory budget, allocator, and snapshots
> Open the lab `40-memory-budget`. Nine checks run in pure Python and torch-CPU, and a tenth runs on the A100.
>
> 1. **Count the bytes.** `state_bytes` returns 16 bytes per parameter for fp32 and for mixed precision. `activation_layer_bytes` returns 3,254,779,904 bytes for $s = 4096$, $b = 1$, $h = 4096$, $a = 32$, and 570,425,344 with fused attention. `mlp_stack_saved_bytes` matches the bytes PyTorch's saved-tensor hooks report on CPU.
> 2. **Peak and chooser.** `peak_activation_bytes` returns 40, 18, 24, and 34 for the four-layer cases, and `choose_checkpoints` matches a brute force on forty seeded cases.
> 3. **Simulate the allocator.** `CachingAllocator` reproduces allocated and reserved on the recorded trace: 48 MiB allocated and 84 MiB reserved at the end, and reserved doesn't fall when everything is freed. With an 80 MiB capacity it releases two free segments and finishes at 56 MiB reserved.
> 4. **Read a snapshot.** On the fixture, the largest free block is 24 MiB, the fragmentation ratio is 0.529, the peak is 79 MiB at `loss.py:31`, and the OOM is a fragmentation failure.
> 5. **Measure.** On a GPU, `measure_peaks` runs a 16-layer stack with and without checkpointing and the test compares the reduction with the calculator's prediction of 448 MiB, within 35%. Without a GPU the tenth check prints a skip and the other nine decide the result.
>
> The tests catch a calculator that counts Adam as one copy, and an allocator that returns a freed segment to the device so that `reserved` shrinks.

The slide deck is [GPU memory management and OOM debugging](/slides/40-memory-management-and-oom).

## Further reading

- PyTorch, [CUDA semantics: memory management](https://docs.pytorch.org/docs/2.15/notes/cuda.html). `memory_allocated`, `memory_reserved`, `empty_cache`, and every `PYTORCH_ALLOC_CONF` option this chapter lists.
- PyTorch, [Understanding CUDA memory usage](https://docs.pytorch.org/docs/2.15/torch_cuda_memory.html). `_record_memory_history`, `_snapshot`, the snapshot fields, and the viewer.
- PyTorch, [`c10/cuda/CUDACachingAllocator.cpp`](https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDACachingAllocator.cpp). The constants, `should_split`, `get_allocation_size`, and the release helpers behind the lab's simulator.
- NVIDIA, [Stream-ordered memory allocation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html). `cudaMallocAsync`, pools, the release threshold, and the reuse policies.
- NVIDIA, [Unified memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html). `cudaMallocManaged`, migration, prefetch, and advice.
- NVIDIA, [MPS documentation](https://docs.nvidia.com/deploy/mps/index.html) and [when to use MPS](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html). Address-space isolation, memory limits, and fault behavior.
- NVIDIA, [MIG user guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html) and its [supported profiles](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html). The A100 80GB profiles.
- Tianqi Chen and others, [Training deep nets with sublinear memory cost](https://arxiv.org/abs/1604.06174), and Korthikanti and others, [Reducing activation recomputation in large transformer models](https://arxiv.org/abs/2205.05198). The checkpointing trade and the per-layer count that [chapter 19](/c/19-training-kernels) uses.
- Rajbhandari and others, [ZeRO](https://arxiv.org/abs/1910.02054). The per-parameter byte counts for sharded state.
- GPU MODE, lecture 32, [Unsloth: LLM systems engineering](https://github.com/gpu-mode/lectures), by Daniel Han, from the lecture index. The index lists it by title only, so the chapter doesn't claim what it covers beyond the topic. Stanford CS336's second assignment, [systems](https://github.com/stanford-cs336/assignment2-systems), has a handout PDF that wasn't read for this chapter, so the chapter cites the assignment without a section.
- Chapters [19](/c/19-training-kernels), [22](/c/22-overlap-and-parallelism), and [25](/c/25-ddp-and-zero): recomputation, the allocator's stream rules, and ZeRO.
