---
title: Profiling and reading the machine code
slug: 05-profiling
part: "Part 1 — The machine"
summary: A four-question workflow for finding out why a GPU program is slow, from a timer to torch.profiler, Nsight Systems, and Nsight Compute, plus how to read the PTX and SASS the compiler produces.
minutes: 120
gpu: true
objectives:
  - "Run the profiling funnel in order: time end to end, find the hot kernel, read the timeline, then examine one kernel."
  - "Benchmark a kernel without fooling yourself: warm up, flush the L2, take medians, and synchronize or use events."
  - "Find the dominant kernel and the idle gaps in a PyTorch program with torch.profiler and a Chrome trace."
  - "Read an Nsight Compute report: Speed of Light, memory workload, warp state, occupancy, and source counters."
  - "Tell a memory-bound kernel from a latency-bound one, and a launch-bound program from a kernel-bound one."
  - "Get PTX, SASS, and register and spill counts out of a CUDA build, and recognize LDG.E.128, FFMA, HMMA, BAR.SYNC, and predicated instructions."
  - "Diagnose uncoalesced access, host synchronization, and warp divergence from evidence, and fix each."
lab: 05-diagnose
---

# Profiling and reading the machine code

> [!TLDR]
> - Profile in a funnel. Time the whole program first, find the one kernel that
>   dominates with `torch.profiler`, look at the timeline for gaps and syncs,
>   and only then open one kernel in Nsight Compute.
> - Every measurement needs hygiene: warm up, flush the 40 MiB L2 when the data
>   fits in it, synchronize or use CUDA events, and report the median.
> - A slow program is one of three things: kernels that move too many bytes,
>   a GPU left idle between kernels, or lanes that sit masked off inside a
>   warp. Each leaves a different signature in the profile.
> - PTX is the virtual assembly and SASS is what the GPU runs. A handful of
>   instructions, `LDG.E.128`, `FFMA`, `HMMA`, `BAR.SYNC`, and predicated
>   `@P0` instructions, tell you most of what you need to know.
> - Your lab profiles with `torch.profiler` and events. Nsight Compute needs
>   hardware counters that a cloud container usually can't reach, so this
>   chapter teaches it from described reports instead.

Chapters 1 to 4 gave you a model of the machine and a prediction for how long a
kernel should take. This chapter is what you do when the prediction is wrong.
A kernel that runs at a third of its roofline isn't mysterious: some resource
is saturated, or some resource is idle, and a profiler can say which.

Profiling is a skill you apply the same way to a hand-written kernel, a Triton
kernel, and a whole PyTorch training step. The workflow doesn't change, only
the tool you pick for the question you are asking. Learning it now pays off in
every later chapter, because each lab's performance metric is a profiling
question in disguise.

The chapter has two halves. The first is the funnel of tools, from a stopwatch
to a counter-level report. The second is reading the machine code the compiler
generated, which is the ground truth when a profiler's summary isn't enough.
The lab then hands you three slow pieces of code and asks you to diagnose and
fix each.

## Before you start

**Asynchronous launches.** A kernel launch returns to Python as soon as the
work is queued, and the GPU runs it later. [Chapter 0](/c/00-what-you-are-building)
showed that a host timer without a synchronize measures the queueing, not the
kernel.

**Warps and divergence.** A warp is 32 threads that execute one instruction
together. If lanes of one warp take different branches, the warp runs both
paths one after the other. [Chapter 2](/c/02-the-execution-model) covers this.

**Coalescing.** A warp's 32 loads are served as 32-byte sectors, so a
contiguous float load is 4 sectors and a scattered one can be 32.
[Chapter 3](/c/03-memory-and-bandwidth) derives it.

**The roofline.** A memory-bound kernel is limited by bytes per second and a
compute-bound one by FLOP per second. [Chapter 4](/c/04-roofline-and-latency)
turns that into a predicted time, which is what you compare a measurement
against.

## A workflow of four questions

The section answers one question: which tool do you reach for first? The answer
is the cheapest tool that can still rule something out.

A GPU program can be slow for reasons at four different scales, and each scale
has a tool that sees it. Starting at the finest scale wastes hours, because a
counter-level report of the wrong kernel tells you nothing about the program.
Start at the coarsest scale and narrow down.

| Step | Question | Tool | What it costs you |
|---|---|---|---|
| 1 | How long does the whole thing take, and is that what I expect? | A timer with a synchronize, or CUDA events | One line of code |
| 2 | Which kernel dominates? | `torch.profiler` with `ProfilerActivity.CUDA`, and a Chrome trace | A few lines; slows the run a little |
| 3 | Is the GPU busy, or waiting on the host? | Nsight Systems, or the Chrome trace's gaps | A separate tool and a report file |
| 4 | Why is this one kernel slow? | Nsight Compute | Hardware counters; replays the kernel many times |

Step 1 gives you the number to beat and the prediction to compare it with.
Step 2 gives you a name. Step 3 tells you whether the name matters: if the hot
kernel runs for 40 percent of the step and the GPU is idle for the other 60,
tuning the kernel recovers less than the idle time would. Step 4 is for the one
kernel that survives steps 2 and 3.

> [!KEY] Rule out the cheap explanations first
> Each step is cheaper than the next and can rule out a whole class of
> problems. A host synchronization found in step 3 explains a slowdown that no
> amount of work in step 4 would find, because the kernel itself is fine.

The lab follows this funnel. Its three slow pieces of code each fail at a
different step: one at step 4, one at step 3, and one that you can see at
step 4 only if you know what to look for.

## Measure end to end first

The section answers one question: how do you time GPU code so the number means
something? The rules are few, and each one removes a specific lie.

The first measurement of any optimization is the end-to-end time of the unit
you care about, a training step or a forward pass, measured the way the user
experiences it. Every later number is compared with it. A kernel that gets 2x
faster but was 5 percent of the step has bought you 2.6 percent.

Four habits make a timing trustworthy.

**Synchronize or use events.** A launch is asynchronous, so a host timer needs
`torch.cuda.synchronize()` before it stops. CUDA events avoid blocking the
host: you record one on the stream before the work and one after, and the GPU
timestamps both. This is what `cuda_time` in `lab_common` does:

```python
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
start.record()
fn()
end.record()
torch.cuda.synchronize()          # wait for `end` to happen
ms = start.elapsed_time(end)      # GPU time between the two events
```

Events measure the GPU timeline between two points on the stream. That
includes any gaps where the GPU waited for the host, which is exactly what you
want when the question is "how long does this take," and why the lab times its
host-synchronization case with a wall clock and a final synchronize instead.

**Warm up.** The first call pays for things that later calls don't: CUDA
context creation, `nvcc` or Triton compilation, library handle setup, and cold
instruction caches. Run the function a few times and throw those timings away.

**Take the median of many runs.** A single run is a sample from a distribution
that has a long tail, because of clock changes, other processes, and the
operating system. The median ignores the tail. Report the spread too when you
compare two kernels that are close.

**Flush the L2 when the data fits in it.** The A100's L2 is 40 MiB. A kernel
that reads a 16 MiB array a second time finds it in cache, so a timing loop over
a small array measures cache bandwidth, not HBM. The learn-inference course
saw this directly: its strided copy's read footprint was 16.8 MB, which fits in
the L2, and that is part of why its measurement doesn't match a bytes-only
prediction (see [the companion chapter on your first CUDA kernel](https://qwen.ifkash.dev/c/12-your-first-cuda-kernel)).
Writing twice the L2's size into a scratch buffer before each timed call
guarantees a cold start, and `cuda_time(fn, flush_l2=True)` does it.

$$
\text{scratch} = 2 \times 40\ \text{MiB} = 80\ \text{MiB} \gg \text{any line left in the cache}
$$

There is a fifth habit that you can't fully control: clocks. The A100's rated
peaks assume a boost clock of 1410 MHz, and the PCIe card has a 300 W power
limit against the SXM4 module's 400 W, so a long, power-hungry kernel may run
below boost. The datasheet doesn't say by how much, so don't assume a number.
What you can do is run the benchmark more than once, compare the medians, and
treat differences under about 5 percent as noise, as the companion course does.

> [!TIP] Time what you will ship
> Time the function exactly as the program calls it, with realistic shapes. A
> kernel timed on one huge array can look great while the model's real
> shapes, many small calls, are launch-bound.

## Find the hot kernel with torch.profiler

The section answers one question: of all the kernels your program launched,
which one is the problem? `torch.profiler` records every CPU operator and every
GPU kernel in a window and tells you where the time went.

You turn it on with a context manager. The activities list asks for the CPU
side and the GPU side:

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with record_function("my_step"):
        out = model(x)
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=10))
prof.export_chrome_trace("trace.json")
```

Three details matter. First, `ProfilerActivity.CUDA` is the argument that
records kernels. The old `use_cuda=True` flag is gone, so code that passes it
fails on PyTorch 2.14. Second, synchronize inside the `with` block, or the
profiler closes before the GPU finishes and the last kernels are missing.
Third, the table's time column was renamed `self_cuda_time_total` to
`self_device_time_total` when PyTorch added non-CUDA backends, so the lab's
solution tries both names.

The table has one row per distinct operator or kernel name, and the columns
mean the following:

| Column | Meaning |
|---|---|
| Self CPU | Host time spent in this operator, not counting operators it called |
| CPU total | Host time including the operators it called |
| Self CUDA (device) | GPU time of this row's own kernels. Zero for a pure CPU operator |
| CUDA total | GPU time of this row and everything beneath it |
| # of Calls | How many times it ran in the window |

Sort by self device time and the top row is the hot kernel. Rows for CPU
operators such as `aten::add` carry the device time of the kernels they launched
in their total column, but only a kernel's own row has self device time, which
is why "self" is the right sort key for finding a kernel.

The names tell you a lot before you read any code. A kernel named like
`vectorized_elementwise_kernel` is a PyTorch pointwise op, one whose
name says "vectorized" is moving 16 bytes per thread, a name with `gemm` or
`cutlass` is a matrix multiply, and your own kernel is whatever you called it,
mangled with its argument types. If the hot kernel is one you didn't expect, you
have already learned something.

**The Chrome trace is the picture.** `export_chrome_trace` writes JSON that you
open in `chrome://tracing` or at ui.perfetto.dev. It shows the host thread on
top and each CUDA stream below, with every kernel as a box on a time axis. Zoom
in and the gaps are visible. A row of tiny kernels separated by gaps is a
launch-bound program. A long gap on the GPU lane under a long `cudaStreamSynchronize`
box on the host lane is a synchronization. You can answer step 3's question
from this trace for most programs without a second tool.

> [!NOTE] The profiler perturbs what it measures
> Recording each kernel and operator adds overhead on the host. Use the
> profiler to rank kernels and to see structure. Take absolute timings from
> events or a synchronized timer without the profiler running.

For a loop, `torch.profiler.schedule(wait=1, warmup=1, active=3)` skips the
first steps, which absorbs warmup, and records only the next few. You then call
`prof.step()` at the end of each iteration. Use `record_function("name")` around
regions of your own so they appear by name in the table and the trace.

## Read the timeline: gaps, syncs, and launch overhead

The section answers one question: is the GPU busy? A kernel can be perfect and
the program still slow, because the GPU spent most of the step waiting for the
host to hand it work.

The host queues each launch in about 5 microseconds, measured on the lab A100
by the companion course, and its rule of thumb is 5 to 10. That is free when a
kernel runs for hundreds of microseconds. An elementwise kernel over $2^{24}$
floats moves $3 \times 64\ \text{MiB} = 192\ \text{MiB}$, which at the
measured 1304 GB/s is $2.01 \times 10^{8} / 1.304 \times 10^{12} = 154\ \mu$s,
about 30 times the launch cost. It matters when kernels are tiny: a kernel that
runs for 2 microseconds and costs 5 to queue leaves the GPU idle at least 60
percent of the time, however well the kernel is written.

There are two ways a program ends up with a mostly idle GPU, and the trace
distinguishes them.

- **Launch-bound.** The host can't queue work as fast as the GPU finishes it.
  The GPU lane shows short kernels with gaps, and the host lane is solid:
  it's always busy launching. The fixes are fewer, bigger kernels (fusion,
  batching) or [CUDA graphs](/c/17-fusion-and-compilers), which submit many
  kernels in one launch.
- **Sync-bound.** The host stops to wait for the GPU, and then the GPU waits for
  the host to queue the next thing. A call that reads a GPU value into Python,
  such as `.item()`, `.tolist()`, `bool(tensor)`, or a `print` of a tensor, is a
  synchronization. The host lane shows a wait, and the GPU lane is empty right
  after the wait ends, while the host does its bookkeeping and queues more work.

NVIDIA measured the difference on a V100 with a kernel that executes for 2.9
microseconds: 9.6 microseconds per kernel when the host synchronizes after
every launch, 3.8 when the launches overlap with one synchronization at the end,
and 3.4 inside a CUDA graph of 20 kernels
([NVIDIA's CUDA graphs article](https://developer.nvidia.com/blog/cuda-graphs/)).
Those are a V100's numbers, so don't carry the microseconds to the A100, but
the shape carries: the sync costs more than the kernel. The figure below builds
the same timeline from those constants.

```viz
05-launch-gaps
```

Start with the default, a 2.9 microsecond kernel and a sync after every launch:
the GPU is busy $2.9 / 9.6 = 30$ percent of the time. Switch to overlapped
launches and the kernels sit back to back as far as the host's launch rate
allows. Then raise the kernel time: the gaps stay the same size and shrink in
proportion, which is why long kernels hide launch overhead and short ones don't.

**Nsight Systems** is the dedicated timeline tool. It records the CPU threads,
the CUDA API calls, every kernel and memory copy on every stream, and NVTX
ranges you add yourself, and it shows them on one time axis. You run it as
`nsys profile -o report python train.py` and open the report in the GUI, or
read a table with `nsys stats`. Reach for it when the Chrome trace isn't enough:
a multi-process job, a question about which CPU thread is blocking, or the exact
sequence of API calls around a stall. Its output answers step 3, never step 4.
It tells you that a kernel took 300 microseconds, not why.

> [!WARNING] A sync hides in innocent code
> `if loss.item() > threshold:` in a training loop synchronizes every step. So
> does indexing a CUDA tensor with a boolean mask, because the output size
> depends on the data, and printing a metric every iteration. In PyTorch you
> can turn each one into an error with `torch.cuda.set_sync_debug_mode("error")`,
> which the lab's tests use.

## One kernel under Nsight Compute

The section answers one question: given a single kernel that you know matters,
what is it limited by? Nsight Compute (`ncu`) answers by running the kernel
many times while it reads the GPU's hardware counters, and then presenting the
counters as sections.

You point it at one kernel, usually by name, and skip the warmup launches:

```bash
ncu --set full -k regex:slow_scale -s 3 -c 1 -o report python lab.py
```

`-k` filters kernels by name, `-s` skips the first launches, `-c` limits how
many it profiles, and `-o` writes a report you open in the GUI. `--set full`
collects every section; `--list-sections` shows which exist. Compile with
`-lineinfo`, which adds source line information without disabling optimization,
if you want the source view below.

Two behaviors of `ncu` surprise people. It replays the kernel once per group of
counters, saving and restoring memory around each replay, so a report is several
runs of the same kernel, and by default it flushes caches between replays and
locks the GPU clocks to a fixed base value. A kernel's duration in `ncu` is
therefore not the duration in your benchmark, and its cache hit rates describe a
cold start. Use `ncu` for ratios and causes, and use your own timings for how
long.

> [!NOTE] The lab doesn't run ncu, and here is why
> Collecting counters needs permission to read the GPU's performance counters,
> and a managed container usually doesn't grant it. If you run `ncu` there you
> typically see an error that names `ERR_NVGPUCTRPERM`. The lab uses
> `torch.profiler` and CUDA events, which need no special permission, and this
> section teaches `ncu` through descriptions. Run it yourself on a machine you
> control; the same kernels from the lab compile there.

A report is a stack of sections. The first few, in the order to read them:

**Speed of Light** (`SpeedOfLight`). Two percentages, the *compute throughput*
(`sm__throughput`) and the *memory throughput*, each as a fraction of the
hardware's peak for that kernel's run. A roofline chart sits next to them. This
section answers the first question: is the kernel near a ceiling, and which one?

- A **memory-bound** kernel shows memory throughput high, close to its peak, and
  compute throughput low. Nothing about the instruction stream can help, because
  the memory system is the limit. The remedy is fewer bytes: vectorize, fuse,
  cache, or compress.
- A **compute-bound** kernel shows the reverse, with the arithmetic pipelines
  near their peak.
- A **latency-bound** kernel shows both percentages low. Neither resource is
  saturated, which means the warps spend their time waiting for each other, for a
  barrier, or for memory that arrived late, and there aren't enough other warps
  ready to cover the wait. This is the case where the next sections matter most.

**Memory Workload Analysis** (`MemoryWorkloadAnalysis`). The traffic through
each level: L1, L2, and device memory (DRAM), with hit rates, and the number of
*sectors per request*. A coalesced float load from a warp is one request for 4
sectors. A scattered one is up to 32. Two counter names give you the ratio
directly: `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` divided by
`l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`. A ratio of 4 for 4-byte
loads is perfect, and a ratio near 32 is the signature of the lab's first slow
kernel. `ncu` also prints a rule here, in words, that names uncoalesced
accesses and estimates the speedup of fixing them.

**Warp State Statistics** (`WarpStateStats`). For each warp, how many cycles per
issued instruction it spent in each state. The states are the *stall reasons*,
and the big ones to know are as follows:

| Stall reason | The warp is waiting for |
|---|---|
| Long Scoreboard | A load from global or local memory to return |
| Short Scoreboard | A result from shared memory or another short-latency unit |
| Barrier | The other warps of its block to reach a `__syncthreads()` |
| Math Pipe Throttle | An arithmetic pipeline that is full |
| LG Throttle | The queue of global and local memory instructions to drain |
| Not Selected | Nothing: it was ready, and the scheduler picked another warp |

Read this section as a diagnosis. A memory-bound kernel is dominated by long
scoreboard stalls. A kernel with a lot of Barrier stalls has uneven work
between warps or too many synchronizations. A kernel dominated by Math Pipe
Throttle is compute-bound, and mostly Not Selected means the scheduler has more
ready warps than it can issue, which is the healthy end state.

**Occupancy** (`Occupancy`). The ratio of active warps to the 64 an A100 SM can
hold, as a *theoretical* value, limited by the kernel's block size, registers
per thread, and shared memory per block, and as an *achieved* value, averaged
over the run. The limiters are named: `launch__occupancy_limit_registers` is the
occupancy the register count allows. [Chapter 2](/c/02-the-execution-model)
works the arithmetic: a 256-thread block at 64 registers per thread fits 4
blocks, which is 32 warps, or 50 percent, and at 255 registers it fits one block
and 12.5 percent. Low occupancy is a problem only when the warps that are there can't hide
latency, so read it beside the stall reasons.

**Source Counters** (`SourceCounters`) and the source view. With `-lineinfo`,
the GUI shows each source line next to its SASS and the counters attributed to
it: instructions executed, the memory sectors it touched, and sampled stalls. It
also reports *branch efficiency* and the average number of lanes active per
instruction (`smsp__thread_inst_executed_per_inst_executed`), which is the
number that exposes divergence: close to 32 when every lane participates, and
near 16 for the lab's third slow kernel.

Counter names change between Nsight Compute versions, so treat the ones above
as the ones this chapter's tooling recognizes and run `ncu --query-metrics` for
your installed version.

### What the two commonest reports look like

Without inventing a measurement, here is how to tell the two common kernel
failures apart, because the first screen of each report looks different.

A **memory-bound** vector add. Speed of Light shows memory throughput near its
peak and compute throughput in the low single digits, because the kernel does
1 FLOP per 12 bytes against a ridge point of 161. Memory Workload Analysis shows
4 sectors per request and DRAM traffic close to the 12 bytes per element you
counted. Warp State shows most cycles in Long Scoreboard. The advice is that
there is nothing to fix: you are at the ceiling, and the target is the copy
ceiling, 1275 GB/s on the lab card.

A **latency-bound** kernel. Speed of Light shows both percentages well below
peak. Occupancy is limited by registers or shared memory, or the grid is too
small to fill 108 SMs. Warp State shows Barrier or Short Scoreboard stalls, or
Long Scoreboard with too few warps to hide it. The advice is to raise the
parallelism: more independent loads in flight per thread, a smaller register
footprint, or a grid that fills the machine. [Chapter 4](/c/04-roofline-and-latency)
derived the amount: roughly 640 KB of memory in flight across the GPU, or 5.9 KB
per SM, to cover HBM latency.

## Read the machine code: PTX and SASS

The section answers one question: when the report says a kernel is wrong, how
do you see what the compiler built? You read the assembly, and you only need a
small vocabulary to do it.

CUDA compiles in two stages. `nvcc` first lowers your code to **PTX**, a virtual
instruction set with unlimited registers that is stable across GPU generations.
Then `ptxas` lowers PTX to **SASS**, the real instruction set of one
architecture, with a fixed register file and real scheduling decisions. A
*cubin* holds SASS, and a *fatbin* can hold SASS for several architectures plus
PTX that the driver can compile at load time for a GPU that didn't exist when you
built. When you ask what the hardware did, SASS is the answer. When you ask what
the compiler understood, PTX is easier to read.

### Getting them out

For a standalone `.cu` file, three commands cover it:

```bash
nvcc -arch=sm_80 -ptx   k.cu -o k.ptx          # PTX, readable
nvcc -arch=sm_80 -cubin k.cu -o k.cubin        # SASS, in an ELF file
cuobjdump -sass k.cubin                        # print the SASS
nvdisasm k.cubin                               # an alternative disassembler, with more detail
```

`cuobjdump -sass` also works on a built library or executable, and
`cuobjdump -res-usage` prints registers, shared memory, and local memory per
function. `nvdisasm` reads only cubins; its `-g` annotates SASS with source
lines and `-plr` adds register live ranges.

For the **register and spill counts**, which are the first things to check, ask
`ptxas` to be verbose:

```bash
nvcc -arch=sm_80 --ptxas-options=-v -c k.cu
```

and it prints, for each kernel, a line like `Used 40 registers` and a line
reading `N bytes spill stores, N bytes spill loads`. In the lab's `load_cuda`
helper, pass `extra_cuda_cflags=("-Xptxas", "-v")` and `verbose=True` to see the
same text during the build. The lab's `parse_ptxas_info` reads exactly this
output.

> [!TIP] Compile tiny kernels in Compiler Explorer
> [Compiler Explorer](https://godbolt.org) shows PTX and SASS for a CUDA
> function beside the source, with matching colors. It is the quickest way to
> see what a one-line change does to the instructions, and it needs no GPU.

### The instructions worth knowing

You don't need to read all of SASS. A few instruction families answer most
questions, and the table shows them with what each tells you. The fragments in
this chapter are written by hand in the style of `cuobjdump -sass`, to show the
shape, and they are not the output of a particular compile.

| SASS | PTX analogue | What it tells you |
|---|---|---|
| `LDG.E` | `ld.global.f32` | A 32-bit global load per thread |
| `LDG.E.128` | `ld.global.v4.f32` | A 16-byte load per thread: the vectorized form of a `float4` |
| `STG.E`, `STG.E.128` | `st.global` | Global stores, scalar or 16 bytes wide |
| `FFMA` | `fma.rn.f32` | A fused multiply-add: 2 FLOP in one instruction |
| `HMMA` | `mma.sync` | A tensor-core matrix multiply-accumulate on fp16 or bf16 |
| `BAR.SYNC` | `bar.sync` | A `__syncthreads()` |
| `LDS`, `STS` | `ld.shared`, `st.shared` | Shared-memory loads and stores |
| `LDL`, `STL` | `ld.local`, `st.local` | Local memory: registers that spilled, or an indexed per-thread array |
| `@P0 EXIT`, `@!P1 BRA` | `@%p1 bra` | A predicated instruction; `!` negates the predicate |

Here is what you do with them.

**Count the loads and check their width.** A kernel that reads `float` with a
scalar pointer compiles to `LDG.E`: each thread asks for 4 bytes, so a warp asks
for 128. Cast the pointer to `float4` and the compiler emits one `LDG.E.128` per
thread, so a warp asks for 512 bytes with a quarter of the instructions.
Seeing `LDG.E.128` in a memory-bound kernel's inner loop is the evidence that
vectorization took, and not seeing it means the compiler couldn't prove the
alignment. The lab's `max_load_bits` is this check.

```text
/*0070*/        LDG.E.128 R4, [R2.64] ;            one 16-byte load per thread
/*0080*/        FFMA R4, R4, 2, 1 ;                four FMAs on the four lanes of data
/*0090*/        FFMA R5, R5, 2, 1 ;
/*00a0*/        FFMA R6, R6, 2, 1 ;
/*00b0*/        FFMA R7, R7, 2, 1 ;
/*00d0*/        STG.E.128 [R2.64], R4 ;            one 16-byte store per thread
```

**Count the arithmetic.** The ratio of `FFMA` to loads in the hot loop is the
kernel's arithmetic intensity as the machine sees it. A loop with 256 `FFMA`s
per element, like the lab's divergent kernel, is compute-bound, and one `FFMA`
per `LDG` is memory-bound. An `HMMA` in the loop means the kernel reached the
tensor cores, and its absence from a matmul means it didn't. [Chapter 11](/c/11-tensor-cores)
is about writing them.

**Find the barriers.** Each `BAR.SYNC` is a point where every warp in the block
waits for the slowest one. A loop with a barrier per iteration shows up in
Warp State as Barrier stalls, and the SASS tells you which line of source to
blame.

**Spot spills.** A `LDL` or `STL` in a hot loop is a register that didn't fit.
The compiler gives a thread at most 255 registers, and a kernel that wants more
keeps the overflow in *local memory*, which is per-thread storage that lives in
the same device memory as global data, cached in L1 and L2. A spill is slower
than a register by the latency of an L1 or L2 hit, and the extra loads and
stores also compete for the memory pipeline. `ptxas -v` reports spills in bytes,
and any nonzero count in the inner loop is worth a look. A cheaper register
budget also raises occupancy: at 255 registers a 256-thread block is one per SM
and 12.5 percent occupancy, against 64 warps at 32 registers.

You limit registers with `__launch_bounds__(maxThreads, minBlocks)` on the
kernel or `-maxrregcount` on the compile. Both can introduce spills, so
re-run `ptxas -v` after you use either. A spill you accept to gain occupancy
is a trade, and only the profile can say whether it won.

**Recognize predication.** Short `if` bodies compile to predicated
instructions instead of branches. `@P0 FADD R0, R0, R1` executes on the lanes
where `P0` is true and does nothing on the others. Predication costs the
instruction's issue slot even for lanes that are off, and no branch. A long
body compiles to a real `BRA`, and the warp then runs both sides in turn when
its lanes disagree. The pattern with every lane in the same direction is
harmless in both forms.

```text
/*0040*/        ISETP.GE.AND P0, PT, R0, c[0x0][0x170], PT ;   compare i with n
/*0050*/   @P0  EXIT ;                                         lanes past the end leave
```

That pair is the `if (i < n) return;` guard in nearly every kernel, and it's why
the guard costs almost nothing: it is two instructions and no divergence in all
but the last warp.

PTX shows the same ideas one level up. A float4 load in PTX is
`ld.global.v4.f32 {%f1, %f2, %f3, %f4}, [%rd5];`, a fused multiply-add is
`fma.rn.f32`, and a guard is a `setp` that writes a predicate register
followed by `@%p1 bra`. PTX registers are virtual, so a PTX file shows hundreds of
named registers that `ptxas` later packs into a small number of real ones.
Don't read a register count off PTX.

## Three patterns, three diagnoses

The section answers one question: how do the tools turn a symptom into a cause?
Each of the lab's three slow pieces of code is a common mistake, and each one
shows up at a different step of the funnel.

### Uncoalesced access: the evidence is sectors

The first slow kernel computes `2 * x + 1` over a row-major matrix, and its thread
index changes the row fastest. Thread $t$ and thread $t+1$ read addresses
`cols` floats apart. With `cols` equal to 1024, that is 4096 bytes.

A warp's 32 loads then touch 32 different 32-byte sectors. The warp needs
$32 \times 4 = 128$ bytes and the memory system serves $32 \times 32 = 1024$:

$$
\boxed{\frac{\text{sectors fetched}}{\text{sectors needed}} = \frac{32 \times 32\ \text{B}}{128\ \text{B}} = 8}
$$

The reads are eight times too expensive. The writes are scattered the same way,
and the companion course counted the whole copy: 32 bytes read plus a 4-byte
write for every 8 bytes the kernel counts as useful, a 4.5x amplification of
bytes. It measured 6.9x less bandwidth, 189 GB/s against 1304 GB/s, for a
stride of 32 floats, and decomposed it as 4.5x for bytes times 1.53x for the
extra request cost. This lab's stride is larger, so don't expect exactly 6.9x.
The lesson from that decomposition holds: the bytes predict a floor on the loss,
and the real loss is bigger.

What does the profile show? In `torch.profiler`, one kernel dominates and
nothing is idle. The kernel runs far longer than the copy-ceiling prediction.
For a 16384 by 1024 matrix, that is $2^{24}$ floats, and the bytes are
$2 \times 2^{24} \times 4 = 134{,}217{,}728$ B, one read and one write. At the
measured copy ceiling:

$$
t_{\text{floor}} = \frac{134.2 \times 10^{6}\ \text{B}}{1.275 \times 10^{12}\ \text{B/s}} \approx 105\ \mu\text{s}
$$

The coalesced kernel should land near 105 microseconds. If the 6.9x ratio from
the companion course carried over, the slow one would take about
$105 \times 6.9 \approx 726\ \mu$s, and your lab reports both. Under Nsight Compute
the evidence is the sectors-per-request ratio near 32 instead of 4, and a
Memory Workload rule that names the uncoalesced access.

The fix is to swap the mapping so consecutive threads get consecutive elements.
That is the one-line change from `row = tid % rows` to a flat index
`i = tid`, and the lab's tests require at least a 2x speedup and 50 percent of
the copy ceiling.

### Host synchronization: the evidence is gaps

The second slow piece of code is Python, not CUDA. It clips a list of tensors so
each one's largest absolute value is at most a threshold, and it decides per
tensor with `if t.abs().max().item() > thresh`. Each `.item()` copies one number
to the host and blocks until every queued kernel has finished.

In the trace you see this directly: a host-side wait after each small group of
kernels, and an empty GPU lane while the host launches the next group. The
program isn't doing too much GPU work. It is doing too little at a time. For
256 tensors, each iteration queues three kernels, the `abs`, the `max`, and the
multiply, at about 5 microseconds of host time each, so even before any waiting
the launches alone cost $256 \times 3 \times 5\ \mu\text{s} = 3.84$ ms. A
batched version does a handful of launches in total.

There are two ways to fix it, and the lab accepts both. Compute the scale on the
GPU, with no decision on the host:

```python
scale = (thresh / m).clamp(max=1.0)     # m is a GPU tensor; no .item()
out = t * scale
```

Dividing by a zero maximum gives infinity, which the clamp turns into 1, so an
all-zero tensor needs no special case. And if the tensors have the same size,
`torch.stack` them and do the whole job on one 2-D tensor with a handful of
kernels. The V100 numbers above predict at least 2.5x for removing the syncs
alone, and batching beats that by a wide margin. The test enforces 1.5x, which
is 60 percent of the ratio, and also runs your function under
`torch.cuda.set_sync_debug_mode("error")` so a hidden sync fails loudly.

### Divergence: the evidence is active lanes

The third slow kernel is the subtle one, because nothing is wrong with its
memory access and nothing is idle on the timeline. It applies a long arithmetic
loop to every element, and the loop depends on whether the index is even or odd:

```cuda
out[i] = (i & 1) ? path_b(x[i]) : path_a(x[i]);
```

Even and odd lanes sit next to each other in every warp, so every warp takes
both branches. The hardware runs path A with the odd lanes masked off, then path
B with the even lanes masked off. Each path is 256 `FFMA` instructions.
Here is the arithmetic. The kernel's useful work, $2^{22}$ elements at 256 FMAs
of 2 FLOP each, divided by the FP32 peak of 19.5 TFLOP/s, is:

$$
t_{\text{compute}} = \frac{2^{22} \times 256 \times 2\ \text{FLOP}}{19.5 \times 10^{12}\ \text{FLOP/s}} = \frac{2.147 \times 10^{9}}{19.5 \times 10^{12}} \approx 110\ \mu\text{s}
$$

The memory floor is $2^{22} \times 8\ \text{B} / 1.275 \times 10^{12} = 26\ \mu$s, so
the kernel is compute-bound by about 4x. The divergent version issues the same
instructions for half as many useful lanes, so the prediction is
$2 \times 110 = 220\ \mu$s against 110 for the fixed version. Your lab reports
both. The figure shows why: the warp pays for every path that any lane takes.

```viz
05-warp-divergence
```

Try the patterns in order. With lanes alternating, the warp makes two passes
and half its lanes are masked in each. With lanes 0 to 15 against 16 to 31, it
is still two passes, because the 32 lanes of one warp are one unit. Only the
warp-uniform pattern makes one pass, and it costs half.

The fix isn't to remove the branch. It's to make the branch uniform across each
warp, and the cheapest way here is to give each thread both an even element and
the odd one beside it. Every lane then runs path A and then path B with all 32
active, and the instruction count per element halves. You can see it in
Nsight Compute as the lanes-active-per-instruction figure rising from about 16 to
about 32.

Not all divergence is a bug. The tail guard `if (i < n)` diverges only in the
last warp. A branch on `blockIdx` never diverges, because all of a block's
threads agree. Check the evidence before you restructure anything: divergence
is expensive only when the diverging paths are long, and a one-instruction
`if` compiles to a predicate and costs a single extra issue slot.

| Symptom | Where it shows | Evidence | Fix |
|---|---|---|---|
| Kernel far above its byte floor | Step 2: one dominant kernel | About 32 sectors per request instead of 4 | Remap threads so a warp reads contiguous memory |
| GPU idle between kernels, host waiting | Step 3: gaps in the trace | `cudaStreamSynchronize` or a copy on the host lane | Remove `.item()` and friends; batch; use CUDA graphs |
| Compute-bound kernel slower than its FLOP floor | Step 4: a source view and the SASS | About 16 active lanes per instruction, two paths in SASS | Make the branch warp-uniform; give each thread both paths |
| Register-hungry kernel with low occupancy | Step 4: Occupancy, and `ptxas -v` | Spills in `ptxas -v`; `LDL` and `STL` in the loop | Cut live state, tune `__launch_bounds__`, re-measure |

## What goes wrong

- **Profiling the warmup.** The first launch includes compilation and context
  setup, and it dominates a short trace. Call the function once before you open
  the profiler, as `hot_kernel` does.
- **Missing the last kernels.** Without a synchronize inside the profiler's
  `with` block, the window closes while the GPU is still running, and the table
  omits the work you most wanted.
- **Trusting a cached measurement.** A benchmark on a 16 MiB array without an L2
  flush times the cache. If your kernel looks faster than the HBM bandwidth
  allows, suspect this before you believe it.
- **Reading a mean.** One slow outlier, from a clock change or another process,
  moves a mean a lot and a median not at all. Report medians.
- **Tuning a kernel that is a small share of the step.** A 2x win on a kernel
  that is 5 percent of the step is worth 2.6 percent. Do step 2 before step 4.
- **Using `ncu` numbers as benchmark numbers.** `ncu` replays the kernel with
  flushed caches and locked clocks, so its durations and hit rates describe a
  different run than your benchmark.
- **Reading register counts off PTX.** PTX has unlimited virtual registers. Use
  `ptxas -v` or `cuobjdump -res-usage`, which report what `ptxas` allocated.
- **Fixing a divergence that isn't there.** Count the lanes first. A branch
  whose paths are one instruction long is predicated, not serialized.
- **Forgetting the sync that the profiler itself needs.** Python code that reads
  a GPU value inside a profiled region adds a synchronization that your
  program might not have without the profiler, so review your own `print`s and
  `.item()`s before you blame the model.

> [!RECAP]
> - Profile in a funnel: end-to-end time, then the hot kernel, then the
>   timeline, then one kernel's counters. Each step can rule out a whole class
>   of problems before you pay for the next.
> - Time with a synchronize or events, after a warmup, as a median, with the
>   L2 flushed when the data fits in 40 MiB.
> - A GPU program is slow because it moves too many bytes, because the GPU sits
>   idle between kernels, or because lanes are masked off. The sector count, the
>   gaps in the trace, and the active lanes per instruction are the evidence.
> - Nsight Compute's Speed of Light separates memory-bound, compute-bound, and
>   latency-bound kernels. Warp State names the stall, and Occupancy names the
>   limiter.
> - PTX is virtual and SASS is real. `ptxas -v` gives registers and spills,
>   `cuobjdump -sass` gives the instructions, and `LDG.E.128`, `FFMA`, `HMMA`,
>   `BAR.SYNC`, and `@P0` are the ones to look for.

## Check your understanding

> [!QUESTION] You add a CUDA kernel call to a loop and time the loop with `time.perf_counter()` around it. It reports 6 microseconds per iteration. What did you measure, and how do you fix it?
> You measured the host's cost to queue the launch, about 5 microseconds on the
> lab A100, not the kernel. Synchronize before you read the timer, or record
> CUDA events around the call and read `elapsed_time` after a synchronize. If the
> loop does many launches, time the whole loop with one synchronize at the end.

> [!QUESTION] A benchmark of a 12 MiB copy reports 3000 GB/s, above the A100's 1935 GB/s rating. What's wrong, and what do you change?
> The data fits in the 40 MiB L2, so after the first iteration the timing loop
> reads from cache and measures L2 bandwidth. Flush the L2 before each timed
> call by writing twice its size, 80 MiB, into a scratch buffer, or use a
> working set much larger than 40 MiB.

> [!QUESTION] A profile shows the GPU idle for 60 percent of a training step, with the host lane full of `cudaLaunchKernel` calls and no waits. Is the program sync-bound or launch-bound, and what do you try?
> It is launch-bound. The host is busy queueing, and the GPU is starved because
> each kernel is shorter than the time to queue the next. A sync-bound program
> has host waits instead. Reduce the number of launches by fusing operations or
> batching, or replay the whole sequence with a CUDA graph, which cuts the
> per-kernel overhead from a launch to a replay.

> [!QUESTION] An Nsight Compute report shows memory throughput at a small fraction of peak and compute throughput at a small fraction of peak. Which of the three kernel types is it, and what do you look at next?
> It is latency-bound: neither resource is saturated. Look at Occupancy to see
> whether registers, shared memory, or a small grid limit the warps, and at Warp
> State for the stall reason. Barrier or Short Scoreboard stalls point at the
> kernel's structure, and Long Scoreboard with few warps points at too little
> memory in flight to cover the latency.

> [!QUESTION] A warp of 32 threads loads a float each at addresses 4096 bytes apart. How many 32-byte sectors does it fetch, and what is the sectors-per-request ratio against a contiguous load?
> It fetches 32 sectors, one per thread, which is 1024 bytes for the 128 it
> needs, an 8x read amplification. A contiguous warp load of floats is 128 bytes,
> which is 4 sectors per request, so the ratio is 32 against 4.

> [!QUESTION] `cuobjdump -sass` shows `LDG.E` in your kernel's hot loop, where you wrote a `float4` load. What does that tell you, and what are the likely causes?
> The compiler didn't emit a 16-byte load: the loop uses 4-byte loads, and a
> warp asks for 128 bytes per instruction instead of 512. Likely causes are a
> pointer the compiler can't prove is 16-byte aligned, a cast that loses the
> `float4` type, or a loop that doesn't iterate in steps of four. Fix the
> alignment or the type and check for `LDG.E.128` again.

> [!QUESTION] `ptxas -v` reports 255 registers and 48 bytes of spill stores for a 256-thread kernel. What is the occupancy limit, and is the spill the problem?
> At 255 registers, charged as 256, one 256-thread block fits per SM: 8 of 64
> warps, 12.5 percent. The spill is a symptom: the kernel wants more registers
> than the hardware allows. Whether it is the problem depends on whether the
> spilled values are used in the hot loop (`LDL` and `STL` there), so check the
> SASS and the stall reasons before you change anything.

## Lab

> [!TRY] Diagnose three slow kernels
> `fixture_slow.py` holds an uncoalesced scale kernel, a clipping function
> that synchronizes on every tensor, and a divergent compute kernel. Fill in
> `hot_kernel`, which uses `torch.profiler` to name the dominant kernel. Then
> write `fast_scale`, `clip_to_max_abs`, and `fast_branchy`, and implement three
> small parsers that read `ptxas -v` and SASS text. You pass when every fixed
> version matches the slow one's results, including awkward sizes, and clears a
> loose speedup threshold: 2x for the scale kernel and 50 percent of the 1275 GB/s
> copy ceiling, 1.5x for the clip, and 1.3x for the divergent kernel. Your lab
> reports each speedup so you can compare it with this chapter's predictions.
> Two CUDA modules compile on the first run, which takes a minute or two.

## Further reading

- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html):
  the metric names, the section reference, and the roofline chart.
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html):
  the timeline view, NVTX ranges, and the `nsys` command line.
- [PyTorch Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html):
  tables, schedules, and the Chrome trace export.
- [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html):
  `cuobjdump`, `nvdisasm`, and the SASS instruction tables.
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html):
  the reference for every PTX instruction.
- [Getting started with CUDA graphs](https://developer.nvidia.com/blog/cuda-graphs/):
  the V100 launch-overhead measurements this chapter quotes.
- [Compiler Explorer](https://godbolt.org): PTX and SASS beside your source.
- [GPU MODE lectures](https://github.com/gpu-mode/lectures): worked Nsight
  Compute sessions.
- [The companion course's CUDA kernel chapter](https://qwen.ifkash.dev/c/12-your-first-cuda-kernel):
  the strided-copy measurement and its 4.5x by 1.53x decomposition.
