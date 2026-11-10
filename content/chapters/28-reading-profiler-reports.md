---
title: Reading profiler reports
slug: 28-reading-profiler-reports
part: "Part 1 — The machine"
summary: Turn the raw counters in an Nsight Compute report into a roofline placement and a diagnosis, read stall reasons, occupancy, sector, bank-conflict, and wave counters, and read an Nsight Systems stats export for stream gaps and launch overhead.
minutes: 110
gpu: false
objectives:
  - "Parse an ncu --csv report, with its banner lines, quoted cells, and large numbers, into a dictionary from kernel to metric."
  - "Place a kernel on the A100 roofline from raw counters: DRAM bytes, FP32 FLOPs counted from FADD, FMUL, and FFMA, and the duration."
  - "Read the stall reasons, and say what long scoreboard, short scoreboard, barrier, MIO throttle, and math pipe throttle each point at."
  - "Compute theoretical occupancy from the launch limits, name the limiter, and compare it with the achieved occupancy."
  - "Detect uncoalesced access from sector efficiency, bank conflicts from wavefront counters, and a tail effect from the wave count."
  - "Write an ordered rule that labels a kernel memory-bound, compute-bound, latency-bound, uncoalesced, bank-conflicted, or tail-limited."
  - "Read an nsys stats export: the union of busy time across streams, the gaps between kernels, and the launch call time, and name a timeline launch-bound or sync-bound."
lab: 28-profile-reports
---

# Reading profiler reports

> [!TLDR]
> - A profiler report is a table of counters. Every conclusion in it is arithmetic on those counters, so you can write the conclusion as a function and test it without a GPU.
> - The roofline placement needs three quantities: DRAM bytes, FP32 FLOPs (`FADD + FMUL + 2 × FFMA`), and the duration. A vector add lands at 0.083 FLOP per byte and 1280 GB/s, and a tiled SGEMM at 683 FLOP per byte and 16.2 TFLOP/s, against a ridge of 10.08 FLOP per byte.
> - A Speed of Light percentage is not a diagnosis. A strided copy sits at 62 percent DRAM throughput and looks memory-bound, but it fetches 8 times the bytes it uses. Check sector efficiency, bank conflicts, and the wave count before you read the two percentages.
> - Stall reasons say what the warps wait for, occupancy says how many warps are there to wait, and the wave count says whether the last wave is empty. Read the three together.
> - An Nsight Systems stats export gives you the GPU's busy fraction and its gaps. Take the union of the intervals across streams, use the median gap, and compare the kernel time with the 5 µs launch call to tell a launch-bound trace from a sync-bound one.

[Chapter 5](/c/05-profiling) taught the funnel and showed what an Nsight Compute report contains. It graded one thing, a rule that separates a memory-bound report from a latency-bound one. This chapter is about the skill that comes after you open a report: going from counters to a cause, and being able to say why.

You do that by treating the report as data. The recorded reports in this chapter's lab are CSV files. You write the parser, the roofline placement, and the classifier as plain Python, and the tests run them on kernels you have not seen. That is also how you would automate a regression check on a real kernel: profile it in CI, parse the counters, and fail the build when the diagnosis changes.

The fixtures are synthesized. No GPU recorded them. A script chose each kernel's launch shape and counters so they agree with one another on an A100 80GB PCIe, and the sections below say how. The counter names and the file layouts are the ones the tools use, but treat any single number as an example, not a measurement.

## Before you start

You need two chapters and a short list of ideas from them.

**The roofline and the ridge point.** A kernel is limited by bytes per second below the ridge and by FLOP per second above it. On the A100 80GB PCIe the FP32 peak is 19.5 TFLOP/s and the HBM2e rating is 1935 GB/s, so the FP32 ridge is $19.5 \times 10^{12} / 1.935 \times 10^{12} = 10.08$ FLOP per byte. [Chapter 4](/c/04-roofline-and-latency) derives it and also gives the wave formula this chapter reads from counters.

**The profiling funnel.** Time the program, find the hot kernel, read the timeline, then open one kernel. [Chapter 5](/c/05-profiling) built it, listed the sections of an `ncu` report, and graded a two-report classifier. This chapter assumes you have run through that lab.

**Sectors and wavefronts.** A global load is served in 32-byte sectors, so a warp of fp32 loads is 4 sectors when it's contiguous. A shared-memory access takes one *wavefront* per bank-conflict-free pass, and 32 banks serve 128 bytes per clock. [Chapter 3](/c/03-memory-and-bandwidth) and [chapter 8](/c/08-shared-memory-and-transpose) derive both.

**What is not rebuilt here.** This chapter doesn't re-derive the stall taxonomy, the occupancy calculator, or the wave formula. It uses them as functions of counters. It also doesn't teach Nsight Compute's GUI or its source view.

## What a counter report looks like as data

This section answers one question: how do you get an `ncu` report into a form a program can read?

Start with the command. You ask for specific metrics with `--metrics`, you pick the kernel with `-k`, and you write CSV with `--csv`:

```bash
ncu --csv --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum \
    -k regex:vec_add -c 1 python lab.py > report.csv
```

According to the [Nsight Compute CLI documentation](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html), `--csv` makes the console output comma-separated and implies `--print-units base`. Without it the default `--print-units auto` scales each value to a fitting order of magnitude, with SI factors of 1000 and the prefixes K, M, G for bytes and n, u, m for time. You want base units in a file a program reads, so that a value of `805306368` always means bytes. The report's `Metric Unit` column still names the unit.

The file is one row per kernel and metric, in a long layout. The fixture `fixture_ncu.py` records six kernels in that layout. This is the head of it:

```text
==PROF== Connected to process 4127 (/usr/bin/python3.12)
"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit","Metric Value"
"0","4127","python3.12","labhost","vec_add","1","7","(256, 1, 1)","(262144, 1, 1)","NVIDIA A100 80GB PCIe","8.0","","gpu__time_duration.sum","nsecond","629,146"
"0","4127","python3.12","labhost","vec_add","1","7","(256, 1, 1)","(262144, 1, 1)","NVIDIA A100 80GB PCIe","8.0","","dram__bytes_read.sum","byte","536,870,912"
```

Four things in those lines trip a first parser, and each one is a check in the lab:

- **Banner lines.** `ncu` prints `==PROF==` lines to standard output before the table and after it. They aren't rows, and `csv.DictReader` would read the first one as the header unless you skip to the line that starts with `"ID"`.
- **Quoted cells.** Every cell is in quotes, and the `Block Size` cell, `(256, 1, 1)`, holds commas. `line.split(",")` shifts every column to the right of it. Use the `csv` module.
- **Thousands separators.** The recorded values carry them, as in `"805,306,368"`, so `float("805,306,368")` raises. Remove the commas first. The CLI documentation doesn't say whether a given version prints separators, so a parser that strips them is correct either way.
- **Values that aren't numbers.** A metric a kernel can't report comes back as `n/a`. Drop it, and don't turn it into zero, because a zero would later look like a measurement.

The result you want is a nested dictionary, kernel first, then metric:

```python
report = parse_ncu_csv(text)
report["vec_add"]["gpu__time_duration.sum"]   # 629146.0, nanoseconds
report["vec_add"]["dram__bytes_read.sum"]     # 536870912.0, bytes
```

> [!WARNING] Names change between Nsight Compute versions
> The metric names in this chapter are the ones the fixtures use. The [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) lists the stall-reason names `smsp__pcsamp_warps_issue_stalled_<reason>`, the occupancy names `sm__maximum_warps_per_active_cycle_pct` and `launch__occupancy_limit_<resource>`, and `launch__waves_per_multiprocessor`. The guide names `l1tex__data_bank_conflicts_pipe_lsu.sum` as an example of a rollup, and describes sectors, wavefronts, and shared-memory instruction counts as memory-table quantities. The author could not confirm the exact sector, FLOP, and bank-conflict counter names on the guide page, so they are the names the fixtures use. Run `ncu --query-metrics` on your installed version before you build a tool on a name, and expect to rename a few.

The lab's tests never open the CSV themselves. They hand your parser the text, so the same classifier works on a report you recorded yourself.

## The roofline from raw counters

This section answers one question: given the counters of one kernel, where does it sit on the roofline, and how far from its roof?

Picture a plot with two axes. Horizontal is arithmetic intensity, FLOP per byte of DRAM traffic, on a log scale. Vertical is the FLOP/s the kernel reached, also on a log scale. The roof is a line that slopes up at 1935 GB/s until the ridge at 10.08 FLOP per byte, then runs flat at 19.5 TFLOP/s. A kernel is a point under the roof, and its distance from the roof is how much a perfect kernel with the same intensity would gain.

In words: the point's horizontal position comes from how many FLOPs it does per byte it moves, and its vertical position comes from how fast it did them. Both come out of three counter groups, and you don't need a profiler's chart to find the point.

```viz
28-roofline-counters
```

The figure shows three kernels from the recorded report with FLOPs, and the readout shows all six. Pick `vec_add` and `sgemm_tiled` and read the readout against the arithmetic below.

### The three quantities

The first quantity is the **DRAM bytes**: `dram__bytes_read.sum` plus `dram__bytes_write.sum`. These count traffic between the L2 cache and device memory. A load that hits in L2 is not in them. That is what the classic roofline wants, because the roof is HBM bandwidth.

The second is the **duration**, `gpu__time_duration.sum`, in nanoseconds. Remember from [chapter 5](/c/05-profiling) that `ncu` locks clocks and flushes caches, so this duration describes the profiled run and not your benchmark. The ratios that come out of it are still good.

The third is the **FLOP count**, and it isn't a single counter. The SASS instruction counters count thread-level instructions that executed with their predicate on:

| Counter | What it counts | FLOPs each |
|---|---|---|
| `smsp__sass_thread_inst_executed_op_fadd_pred_on.sum` | FP32 adds | 1 |
| `smsp__sass_thread_inst_executed_op_fmul_pred_on.sum` | FP32 multiplies | 1 |
| `smsp__sass_thread_inst_executed_op_ffma_pred_on.sum` | FP32 fused multiply-adds | 2 |

An FFMA is a multiply and an add in one instruction, so it counts twice. Forget the 2 and every GEMM reports half its work. Tensor-core kernels use different counters, and their peak is 312 TFLOP/s for bf16, so a tensor-core kernel needs its own FLOP count and its own roof. The lab's reports are FP32, and `roofline_from_counters` takes the peak as an argument so you can pass 312 for a bf16 kernel.

With the three quantities, the rest is division. State it before the numbers:

$$
I = \frac{\hla{F}}{\hlb{B}}, \qquad
\text{achieved} = \frac{\hla{F}}{t}, \qquad
\text{roof}(I) = \min\bigl(P,\; I \cdot \beta\bigr)
$$

$\hla{F}$ is the FLOP count, $\hlb{B}$ is the DRAM bytes, $t$ is the duration, $P$ is the FP32 peak, and $\beta$ is the bandwidth. The intensity $I$ places the point horizontally, the achieved rate places it vertically, and the roof at that intensity is the best any kernel at that intensity could do.

> [!EXAMPLE] A vector add and a tiled SGEMM, from their counters
> The recorded `vec_add` reads 536,870,912 bytes and writes 268,435,456, so $B = 805{,}306{,}368$. Its FADD counter is $2^{26} = 67{,}108{,}864$ and it has no FMULs or FFMAs, so $F = 67{,}108{,}864$. The duration is 629,146 ns.
>
> $$
> I = \frac{67{,}108{,}864}{805{,}306{,}368} = \boxed{0.0833}\ \text{FLOP/B}, \qquad
> \frac{805{,}306{,}368\ \text{B}}{629{,}146 \times 10^{-9}\ \text{s}} = \boxed{1280}\ \text{GB/s}
> $$
>
> 0.0833 is $1/12$: one add for 12 bytes. It's far below the ridge of 10.08, so the kernel is memory-bound and its roof is $0.0833 \times 1935 = 161$ GFLOP/s. It achieved $67.1 \times 10^{6} / 629{,}146 \times 10^{-9} = 107$ GFLOP/s, which is 66.2 percent of that roof, and the same 66 percent of the 1935 GB/s rating. In bandwidth terms it is at 100.4 percent of the 1275 GB/s copy ceiling that learn-inference measured on this card. That's within a rounding of done. Because the fixture is synthesized, treat that as a pattern to recognize and not as a result.
>
> The recorded `sgemm_tiled` does 4096 cubed FFMAs, $68{,}719{,}476{,}736$, plus $4096^2$ FMULs for the scaling, so $F = 2 \times 68{,}719{,}476{,}736 + 16{,}777{,}216 = 137{,}455{,}730{,}688$. It moves 201,326,592 bytes, 134,217,728 read and 67,108,864 written, in 8,484,922 ns. Its intensity is $683$ FLOP per byte, 68 times the ridge, so it's compute-bound. It achieved $16.2$ TFLOP/s, which is 83.1 percent of the 19.5 TFLOP/s peak.

The same function handles a kernel with no FLOPs. A copy has $F = 0$, so $I = 0$ and the roof is zero, and the "percent of the roof" for a memory-bound kernel is better stated as the bandwidth share: achieved GB/s over 1935. The lab's `pct_of_ceiling` does that, so `copy_strided` reports 62.0 percent instead of dividing zero by zero.

### Which roof to believe

The rated peaks are ceilings, not targets. [Chapter 4](/c/04-roofline-and-latency) already made the point that a plain copy on this card measured 1275 GB/s, 66 percent of the rating, so a memory-bound kernel at 66 percent of 1935 is at its ceiling. A tiled FP32 SGEMM at 83 percent of 19.5 TFLOP/s is a good kernel. When you read a "percent of roof" figure, say which roof you divided by.

The roofline also has a blind spot that the next sections fill. A point near the roof tells you the kernel moves its bytes efficiently, and says nothing about whether it moves more bytes than it needs. A kernel that fetches 8 times the useful data can sit at a respectable DRAM throughput, because the bytes it wastes still cross the memory bus.

## Stall reasons

This section answers one question: when a warp can't issue an instruction, what is it waiting for, and what does that say about the fix?

A scheduler on an A100 SM picks one ready warp per cycle. A warp that isn't ready is *stalled* for a reason, and Nsight Compute samples the program counter of every warp over the kernel's run to count them. The metrics are named `smsp__pcsamp_warps_issue_stalled_<reason>`, and the Profiling Guide lists the reasons: barrier, branch resolving, dispatch stall, drain, IMC miss, LG throttle, long scoreboard, math pipe throttle, membar, MIO throttle, misc, no instructions, not selected, selected, short scoreboard, sleeping, TEX throttle, wait, and warpgroup arrive. The recorded fixtures keep nine of them as shares of the samples, so the nine sum to 100.

Read the reasons as a picture of one warp's life. It issues an instruction, then waits for the result or for a resource, then issues again. The share a reason holds is the share of the warp's time spent waiting for that.

| Reason | The warp waits for | What a large share points at |
|---|---|---|
| Long scoreboard | A global or local load to return | Memory latency or bandwidth, with too few loads in flight |
| Short scoreboard | A result from shared memory or another short-latency unit | Shared-memory traffic, often with bank conflicts |
| MIO throttle | The queue in front of shared memory and special-function units | Too many shared-memory instructions at once |
| Barrier | The other warps of the block at `__syncthreads()` | Uneven work between warps, or too many barriers |
| Math pipe throttle | An arithmetic pipeline that is full | The kernel is using the math units, which is the compute-bound end state |
| LG throttle | The queue of global and local memory instructions | Too many small, uncoalesced loads queued up |
| Not selected | Nothing: the warp was ready, and another was picked | More ready warps than the scheduler can issue |
| Selected | Nothing: the warp issued | Healthy |
| Wait | A fixed-latency dependency on the previous instruction | A short dependency chain that more warps or more ILP would hide |

`not_selected` and `selected` aren't stalls. A kernel where 70 percent of the samples are `not_selected` has more ready warps than it can issue, and you read the next-largest reason instead. The lab's `dominant_stall` leaves both out for that reason.

The six recorded kernels show what each reason looks like. The dominant stall of each is in the table, with the share it holds:

| Kernel | Dominant stall | Share | What it says |
|---|---|---|---|
| `vec_add` | Long scoreboard | 61% | Warps wait for DRAM, as they should in a memory-bound kernel |
| `sgemm_tiled` | Math pipe throttle | 38% | The FP32 pipes are full, the compute-bound end state |
| `copy_strided` | Long scoreboard | 72% | Waiting for loads that each fetch a mostly wasted sector, with LG throttle at 12% behind it |
| `transpose_naive_smem` | MIO throttle | 34% | The shared-memory queue is full, with short scoreboard at 29% and barrier at 14% |
| `reduce_tail` | Math pipe throttle | 22% | The math pipes are the wait, though they sit idle half the time because of the tail |
| `scan_lookback` | Long scoreboard | 41% | Waiting for loads, with barrier at 20% and too few warps to hide them |

Two of these rows teach a rule. `reduce_tail`'s dominant stall is math pipe throttle, which in isolation says "compute-bound", and the kernel isn't. The stall shares describe the warps that are running, and they say nothing about the SMs that sit empty during the second wave. And `copy_strided` has the same dominant stall as `vec_add`, long scoreboard, though one is at its memory ceiling and the other is wasting seven eighths of its bytes. A stall reason names a symptom. You need the counters in the following sections to name the cause.

> [!KEY] A stall reason is a symptom, a counter pair is a cause
> Long scoreboard tells you the warps wait for memory. Sector efficiency tells you why: they wait for bytes they don't use. MIO throttle tells you the shared-memory queue is full. The wavefront counters tell you why: bank conflicts. Read the stall first to pick the counters to look at, and let the counters decide.

## Occupancy: what the launch allows, and what it got

This section answers one question: how many warps is each SM able to hold, how many did it hold, and does the difference matter?

*Theoretical occupancy* is a property of the launch, with no run needed. An SM holds at most 64 warps, 2048 threads, and a block needs registers, shared memory, and warp slots. The SM fits as many blocks as the scarcest resource allows. Nsight Compute reports each resource's limit as `launch__occupancy_limit_registers`, `_shared_mem`, `_warps`, and `_blocks`, each in blocks per SM, and the theoretical occupancy as `sm__maximum_warps_per_active_cycle_pct`. [Chapter 2](/c/02-the-execution-model) works the calculator, so this section only reads its result.

State the rule before you apply it. The resident blocks per SM is the smallest of the four limits $L_r, L_s, L_w, L_b$. A block of $T$ threads has $\lceil T/32 \rceil$ warps:

$$
\text{occupancy} = \frac{\min(L_r, L_s, L_w, L_b) \cdot \lceil T/32 \rceil}{64}
$$

The smallest limit is the limiter. The register limit comes from how the registers are allocated: the SM has 65,536 and a warp takes its thread registers times 32, rounded up to a unit of 256 registers. [The fact sheet](../../docs/FACTS.md) rates that rounding rule as secondary, from the retired occupancy-calculator spreadsheet. The report prints the result as `launch__occupancy_limit_registers`, so read the limit directly and use the rounding rule only to predict it.

> [!EXAMPLE] Three launches, three limiters
> `sgemm_tiled` runs 256 threads, 8 warps, at 128 registers per thread. One warp holds $128 \times 32 = 4096$ registers, so the SM fits $65{,}536 / 4096 = 16$ warps, which is 2 blocks. The shared-memory limit is 10 blocks, the warp limit is 8, and the block limit is 32. The smallest is 2, from registers:
>
> $$
> \frac{2 \times 8}{64} = \boxed{25\%}
> $$
>
> `scan_lookback` runs 128 threads, 4 warps, at 168 registers. A warp holds $168 \times 32 = 5376$ registers, and $\lfloor 65{,}536 / 5376 \rfloor = 12$ warps fit, which is 3 blocks. Occupancy is $3 \times 4 / 64 = 18.75\%$, limited by registers.
>
> `reduce_tail` runs 256 threads, but each block asks for about 100 KB of shared memory, and the SM has 164 KB. One block fits, 8 warps, $8/64 = 12.5\%$, limited by shared memory.

Each of those matches the report's `sm__maximum_warps_per_active_cycle_pct`: 25, 18.75, and 12.5.

*Achieved occupancy*, `sm__warps_active.avg.pct_of_peak_sustained_active`, is what the run actually held, averaged over the SM's active cycles. It's never above the theoretical value and it drops for three reasons: blocks of unequal length, SMs that run out of blocks at the end of the grid, and a grid smaller than the machine. The recorded numbers show two of them:

| Kernel | Theoretical | Achieved | Gap |
|---|---|---|---|
| `sgemm_tiled` | 25.0% | 24.4% | Small: the grid is 4.7 waves, and the tail is short |
| `scan_lookback` | 18.75% | 16.9% | 10%: blocks of unequal length, and the last wave |
| `reduce_tail` | 12.5% | 6.9% | 45%: the second wave runs 12 blocks on 108 SMs |

Occupancy is a means. The question is whether the warps that are resident can hide the latency their kernel has. [Chapter 4](/c/04-roofline-and-latency) derived how many bytes must be in flight, about 5.9 KB per SM for HBM latency. `sgemm_tiled` runs at 25 percent occupancy and 83 percent of the FP32 peak, because each thread has many independent FFMAs to issue. `scan_lookback` runs at 18.75 percent and 14 percent DRAM throughput, because each thread has one load at a time. The same few warps are plenty in one kernel and too few in the other. The stall reason says which: math pipe throttle for the first, long scoreboard with a barrier for the second.

> [!TIP] Raise occupancy last
> A latency-bound kernel with low occupancy has three fixes: reduce registers per thread, which can spill, shrink the block's shared memory, or issue more independent loads per thread, which raises the bytes in flight without raising the warps. The third usually wins. Occupancy rises for free in the first two, and the kernel may not get faster.

## Coalescing counters: sectors, requests, and bytes

This section answers one question: how much of what a load fetches does the kernel use?

A warp's global load is one *request*. The L1 cache splits it into the 32-byte *sectors* it touches. Three counters give you the picture for loads, with an `st` twin for stores:

- `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`, the number of warp-level load requests.
- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`, the sectors those requests touched.
- `smsp__sass_data_bytes_mem_global_op_ld.sum`, the bytes the instructions asked for.

The first ratio people compute is sectors per request. A contiguous fp32 warp load is 128 bytes, 4 sectors, and a load whose lanes are 128 bytes apart touches 32. That's the signature of the strided kernel, and it's in the report: `copy_strided` has 524,288 requests and 16,777,216 sectors, so 32 sectors per request, against 4 for `vec_add`.

But sectors per request alone is not a test. A warp of `float4` loads moves $32 \times 16 = 512$ bytes, and 512 contiguous bytes are 16 sectors. A coalesced `float4` kernel runs at 16 sectors per request and wastes nothing. A rule that flags anything above 8 flags the fastest copy kernel you will write.

The quantity that doesn't depend on the access width is the *sector efficiency*, the useful bytes over the fetched bytes:

$$
\eta_{\text{sector}} = \frac{\hla{U}}{32 \cdot \hlb{S}}
$$

$\hla{U}$ is the bytes the instructions asked for and $\hlb{S}$ is the sectors fetched, so $32\,\hlb{S}$ is the bytes that crossed. Efficiency is 1 when every fetched byte was used. A coalesced fp32 load has 128 useful bytes over 4 sectors, which is 128 / 128 = 1. A coalesced `float4` load has 512 over 16 sectors, which is 1. A stride-32 fp32 load has 128 useful bytes over 32 sectors, which is 128 / 1024 = 0.125.

> [!EXAMPLE] `copy_strided`, counted
> The kernel copies $2^{24}$ floats, reading with a stride of 32 floats and writing contiguously. The load counters are 524,288 requests, 16,777,216 sectors, and 67,108,864 useful bytes. The fetched bytes are $32 \times 16{,}777{,}216 = 536{,}870{,}912$, so
>
> $$
> \eta_{\text{sector}} = \frac{67{,}108{,}864}{536{,}870{,}912} = \boxed{0.125}
> $$
>
> The DRAM counters agree with the sectors: `dram__bytes_read.sum` is 536,870,912, which is the fetched bytes and 8 times the useful ones. The writes are fine, 4 sectors per request and efficiency 1. The kernel moves $536{,}870{,}912 + 67{,}108{,}864 = 603{,}979{,}776$ bytes in 503,316 ns, which is 1200 GB/s, or 62 percent of 1935 GB/s.

That last figure is the trap. 62 percent DRAM throughput passes a memory-bound rule, and the SM throughput is 4 percent. The report's Speed of Light looks like `vec_add`'s. The difference is in the bytes: a coalesced version of the same copy moves $2 \times 2^{24} \times 4 = 134{,}217{,}728$ bytes, which at the 1275 GB/s copy ceiling is a prediction of 105 µs, 4.8 times shorter than the 503 µs recorded. [Chapter 5](/c/05-profiling)'s first slow kernel is this pattern, and its lab measures the real speedup.

A few details keep the counters honest. Predicated-off lanes inflate the requested-byte count in some kernels, which is why `ncu` also has derived "excessive sectors" quantities. A load that hits in L1 doesn't reach DRAM, so a high sector count with low DRAM bytes is a cache doing its job. And a read-only broadcast, all lanes loading one address, is 1 sector for 128 useful-looking bytes, so the efficiency can pass 1. The lab's rule asks only whether efficiency is below one half.

## Shared memory counters: wavefronts and bank conflicts

This section answers one question: how many extra passes through the shared-memory pipe did the access pattern cost?

Shared memory has 32 banks of 4 bytes, and it serves 128 bytes per clock per SM. A warp access that puts each of its lanes in a different bank, or on the same word, is served in one pass. If $N$ lanes hit $N$ different words in one bank, the pipe takes $N$ passes. Nsight Compute counts a pass as a *wavefront*. [Chapter 8](/c/08-shared-memory-and-transpose) derives the rule and fixes the transpose with padding and a swizzle, so this chapter reads the counters.

Picture the transpose's 32 by 32 fp32 tile. A thread writes its element to `tile[row][col]` and later reads `tile[col][row]`. The write is one row, so the 32 lanes land in 32 consecutive words, 32 banks. The read is one column: every lane's word is $32$ words past the last, which is the same bank. All 32 lanes hit bank 0, in 32 different words, so the load takes 32 wavefronts.

Four counters record it, with the shared loads and stores counted apart:

- `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum` and `..._st.sum`, the wavefronts.
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` and `..._st.sum`, the wavefronts beyond the minimum that conflicts caused.

The test is a fraction, and the same reasoning as for sectors applies. The wavefront count can't be compared with the request count, because a conflict-free 128-bit access takes four wavefronts and isn't a conflict. The conflict counter already subtracts the minimum, so the fraction of wavefronts that are conflicts is the width-independent test:

$$
f_{\text{conflict}} = \frac{\hla{C_{ld}} + \hla{C_{st}}}{\hlb{W_{ld}} + \hlb{W_{st}}}
$$

$\hla{C}$ is the conflict wavefronts and $\hlb{W}$ is all the wavefronts, and you add loads and stores before you divide, so a kernel with clean stores and conflicted loads is judged on both.

> [!EXAMPLE] The naive transpose, counted
> `transpose_naive_smem` moves an 8192 by 8192 fp32 matrix, which is $2^{26}$ elements and $2^{21} = 2{,}097{,}152$ warp accesses in each direction. The store is conflict-free: 2,097,152 wavefronts and no conflicts. The column load is a 32-way conflict: $32 \times 2{,}097{,}152 = 67{,}108{,}864$ wavefronts, of which 31 per access, 65,011,712, are conflicts.
>
> $$
> f_{\text{conflict}} = \frac{65{,}011{,}712}{67{,}108{,}864 + 2{,}097{,}152} = \boxed{0.939}
> $$
>
> The stalls agree. The dominant one is MIO throttle at 34 percent, short scoreboard follows at 29, and barrier at 14. A padded tile reads each column in one wavefront, so the total drops from 69,206,016 wavefronts to $2 \times 2{,}097{,}152 = 4{,}194{,}304$, 16.5 times fewer, and the conflict fraction is zero.

The other side matters as much. `sgemm_tiled` also has conflict counters, 2,684,355 conflicted wavefronts out of 301,989,888, which is 0.9 percent, and it's compute-bound at 83 percent of peak. A rule that flags any nonzero count blames the one kernel that isn't the problem. The lab sets the threshold at a quarter of the wavefronts, and its held-out kernels include a padded transpose with 0.4 percent conflicts and a 2-way conflict at 50 percent.

The memory table also hides one more case. If the kernel never touches shared memory, every counter is zero, and the fraction is 0 / 0. Define it as 0, and never divide without a guard.

## Waves and the tail effect from counters

This section answers one question: how does the report show that the last wave of blocks runs nearly empty?

[Chapter 4](/c/04-roofline-and-latency) derived wave efficiency, $\eta = G / (\text{waves} \cdot r \cdot 108)$ for $G$ blocks and $r$ resident blocks per SM, and the table that shows 109 blocks at 50.5 percent. The report gives you $G$ as `launch__grid_size` and $r$ as the smallest of the four occupancy limits, so you compute the wave count from the same dictionary:

$$
\text{waves} = \frac{G}{108\, r}, \qquad \eta = \frac{\text{waves}}{\lceil \text{waves} \rceil}
$$

The report also prints `launch__waves_per_multiprocessor`, which is the same wave count. The lab checks that your computed value matches it, so a wrong $r$ shows.

```viz
28-wave-sawtooth
```

The figure sweeps the grid. Pick one resident block per SM and the efficiency touches 1.0 at every multiple of 108 and drops to a sliver of a wave above it. With 8 resident blocks the wave is 864 blocks, and the dips are shallower because the same extra block is a small part of a large wave.

> [!EXAMPLE] `reduce_tail`: 120 blocks on 108 SMs
> The kernel launches 120 blocks of 256 threads, and each asks for about 100 KB of shared memory, so one block is resident per SM. The wave holds 108 blocks, and the grid is $120 / 108 = 1.111$ waves.
>
> $$
> \eta = \frac{1.111}{\lceil 1.111 \rceil} = \frac{1.111}{2} = \boxed{0.556}
> $$
>
> The first wave runs 108 blocks and the second runs 12, on 12 SMs, while 96 SMs sit idle. The report shows it three ways. `sm__cycles_active.avg` over `sm__cycles_elapsed.avg` is 0.556. `sm__throughput` is 33 percent. And achieved occupancy is 6.9 percent against 12.5 percent theoretical.

Each block takes about 1 ms and the grid takes 2.0 ms. If the 120 blocks of work could be split evenly across 108 SMs, the floor would be $120/108 \times 1\ \text{ms} = 1.11$ ms. That is a prediction from the fixture's arithmetic, and the saving is 44 percent.

The tail effect is the case that most wants a counter. Its Speed of Light percentages are low, because half the SMs are idle half the time, and a classifier that reads only them says "latency-bound" and sends you to raise occupancy. The wave count says the opposite: the machine is fine and the grid is the wrong shape. The fixes are in the grid and not the block:

- **Make the grid a whole number of waves.** 108 blocks, or 216, at one resident block per SM.
- **Make more, smaller blocks.** The tail costs one block-time out of $w + 1$ waves, and a smaller block is a smaller block-time.
- **Use a persistent kernel.** Launch exactly $108\,r$ blocks and have each loop over tiles. [Chapter 4](/c/04-roofline-and-latency) names the tile grid case, and the course's other GEMM chapters build the schedules.

Two limits keep you from over-reading the number. When the grid has less than one wave, $\eta = \text{waves}$ is below one for a different reason, because the grid is smaller than the machine, and the fix is more parallelism, not a better grid shape. The lab calls that case latency-bound and reserves tail-effect for grids above one wave. And a grid of 303.4 waves, like `vec_add`'s, loses $1 - 303.407/304 = 0.2$ percent to its tail, which no one should fix.

## From counters to a label

This section answers one question: in what order do you apply the tests, and why does the order matter?

You now have five groups of numbers: the two Speed of Light percentages, sector efficiency, the conflict fraction, and the wave efficiency. Each names a different cause. The question is which to test first, and the answer is the structural ones before the saturation ones. Speed of Light describes how busy the machine was. The structural counters describe whether the kernel was doing the right work. A busy machine doing the wrong work, 8 times the bytes or 32 passes per load, looks the same as a busy machine doing the right work.

The lab's `diagnose` applies six rules in this order and returns at the first that fires:

| Order | Label | Fires when | Why here |
|---|---|---|---|
| 1 | `uncoalesced` | The smaller of the load and store sector efficiencies is below 0.5 | Wasted bytes inflate DRAM throughput |
| 2 | `bank-conflicts` | At least 25% of shared wavefronts are conflicts | Conflicts stall the pipe with DRAM and SM both low |
| 3 | `tail-effect` | More than one wave, and wave efficiency below 0.8 | An empty last wave halves SM throughput |
| 4 | `memory-bound` | DRAM throughput at least 50% of peak | The kernel is at its bandwidth roof |
| 5 | `compute-bound` | SM throughput at least 60% of peak | The kernel is at its arithmetic roof |
| 6 | `latency-bound` | Everything else | Neither resource is busy, and no structural cause was found |

Here is what each recorded kernel returns, with the numbers that decide it:

| Kernel | DRAM % | SM % | Deciding evidence | Label |
|---|---|---|---|---|
| `vec_add` | 66.2 | 9.4 | Efficiency 1.0, no shared memory, 303.4 waves | `memory-bound` |
| `sgemm_tiled` | 1.2 | 83.5 | Conflicts 0.9%, waves 4.74 at $\eta$ 0.948 | `compute-bound` |
| `copy_strided` | 62.0 | 4.2 | Load efficiency 0.125 | `uncoalesced` |
| `transpose_naive_smem` | 29.8 | 21.2 | Conflict fraction 0.939 | `bank-conflicts` |
| `reduce_tail` | 2.2 | 33.0 | 1.111 waves, $\eta$ 0.556 | `tail-effect` |
| `scan_lookback` | 14.0 | 6.1 | Nothing structural, occupancy 18.75% | `latency-bound` |

Read the table against what a Speed-of-Light-only rule would say. `copy_strided` would be memory-bound at 62 percent, and the fix would be "move fewer bytes", which is right but misses that the fix is a different index map. `transpose_naive_smem` would be latency-bound at 30 and 21 percent, and you would raise occupancy of a kernel whose pipe is the problem. `reduce_tail` would be latency-bound at 2 and 33 percent, and a better grid would be the fix.

The thresholds are rules of thumb for this course. NVIDIA doesn't publish them. The 50 percent DRAM threshold is 967.5 GB/s of the 1935 rating, which is 76 percent of the measured 1275 GB/s copy ceiling, so it flags a kernel that is well on its way to the ceiling. The 60 percent SM threshold fits a compute-bound GEMM that reaches 83 percent. If you port the rules to another GPU, scale the peaks and keep the order.

> [!NOTE] The fallback label is a statement of ignorance
> `latency-bound` is what remains after the structural tests, so it's the weakest of the six. It says that neither bandwidth nor arithmetic is the limit and that nothing structural was found. The next step is the stall reasons and occupancy: long scoreboard with low occupancy points at too little memory in flight, barrier points at the kernel's structure, and a small grid points at the launch shape.

The lab also grades the classifier on held-out kernels that you can't see, and they are the cases that break a rule of thumb. A coalesced `float4` copy has 16 sectors per request and is memory-bound. A triad with 1000.5 waves has a half-full last wave and is memory-bound. A padded transpose with 0.4 percent conflicts is memory-bound. A kernel whose 54 blocks fill half the machine is latency-bound. A kernel with no global traffic at all has zero requests and zero sectors, and it is compute-bound, with no division by zero. If your classifier passes the six recorded kernels and fails these, it learned the six.

## Reading an Nsight Systems stats export

This section answers one question: given a timeline as a table, how do you measure how idle the GPU is and why?

Nsight Compute answers "why is this kernel slow." Nsight Systems answers the earlier question from [chapter 5](/c/05-profiling): is the GPU busy at all? You record a timeline, and you can export its tables without the GUI with `nsys stats`:

```bash
nsys profile -o report python train_step.py
nsys stats --report cuda_gpu_trace --format csv report.nsys-rep > gpu_trace.csv
nsys stats --report cuda_api_sum   --format csv report.nsys-rep > api_sum.csv
```

The [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) lists `cuda_api_sum` and `cuda_gpu_kern_sum` among the default reports, and says `--format` accepts csv among several. It doesn't print the column headers in the part of the guide this chapter's author read, so the fixtures' headers are a synthesized layout: `cuda_gpu_trace` has `Start (ns)`, `Duration (ns)`, `Strm`, and `Name` among its columns, and `cuda_api_sum` has `Total Time (ns)`, `Num Calls`, `Avg (ns)`, and `Name`. Run `nsys stats --help-reports` on your version, and read each column by its header name and not its position.

Two traps again. The export starts with a progress line, `Processing [...]`, before the header, so the header is the first line that holds a comma. And a kernel's template name holds commas inside quotes, `vectorized_elementwise_kernel<(int)4, at::native::AddFunctor<float>, ...>`, so the `csv` module is again the parser. A memory copy appears in the same table with a name that starts `[CUDA memcpy`, and you separate it from a kernel by that prefix.

The lab ships three exports. They're synthesized, and they tell three stories:

| Fixture | The story | Kernels | Median kernel | Median gap |
|---|---|---|---|---|
| `LAUNCH_BOUND` | 60 small kernels, issued one launch at a time | 60 | 2.87 µs | 4.69 µs |
| `SYNC_BOUND` | 12 steps of a 350 µs kernel, a 4-byte copy of the loss, then a host pause | 12 | 357.8 µs | 901.7 µs |
| `OVERLAP` | Two streams, each busy, running at once | 20 | 202.4 µs | no gaps |

### Busy time is a union

The first number you want is the *duty cycle*: the fraction of the trace's span in which the GPU ran anything. Picture each event as an interval on a time axis. The GPU is busy where at least one interval covers the axis, so the busy time is the length of the union of the intervals. The span is the end of the last event minus the start of the first.

$$
\text{duty} = \frac{\left|\bigcup_i [s_i,\ s_i + d_i]\right|}{\max_i (s_i + d_i) - \min_i s_i}
$$

The numerator is the union of the event intervals, each from its start $s_i$ for its duration $d_i$. The denominator is the span. To compute the union, sort the intervals by start, merge each interval that begins before the previous one ends, and add up the merged lengths. The gaps are the spaces between the merged intervals.

Adding the durations is the plausible wrong answer, and `OVERLAP` is built to catch it. Its two streams each run 10 kernels back to back, offset by 100 µs. The durations sum to 4,027,997 ns, and the span is 2,127,008 ns, a ratio of 1.894. That's a GPU busy 189 percent of the time, which can't be. The union is 2,127,008 ns, a duty of 1.0, because at every moment at least one stream is running.

> [!KEY] Overlapped streams are busy once
> Total kernel time over wall time can exceed 1 when streams overlap. That number is concurrency, and the duty cycle is the union. Use the union for "is the GPU idle", and the sum for "how much work ran".

### The median gap, and the 5 µs launch

A gap is idle GPU time between two kernels. Its size says who's responsible, and the comparison is with the launch call. The `cuda_api_sum` export gives the average time of `cudaLaunchKernel`: 5120 ns in the `LAUNCH_BOUND` export, 5400 ns in `SYNC_BOUND`. FACTS.md records about 5 µs as the host cost of queueing a kernel on the lab card, from learn-inference chapter 12. The host spends that long on each launch, and the GPU can't start a kernel until the host has queued it.

If the host launches the kernels one after another and each kernel runs for less than a launch call, the GPU drains its queue as fast as the host fills it. The kernel waits on the launch, and the gap between kernels is about a launch call. The `LAUNCH_BOUND` trace is that. Its kernels run a median 2.87 µs, shorter than the 5.12 µs launch call, and its median gap is 4.69 µs, close to it. The 60 kernels run for 176 µs across a span of 451 µs, a duty of 0.39. Fix this with fewer, larger kernels, by fusing, or by recording the sequence once and replaying it as a graph, which [chapter 17](/c/17-fusion-and-compilers) builds.

A different gap has a different cause. In `SYNC_BOUND` the kernels run 358 µs, 66 launch calls long, and the gaps are 902 µs. A launch call can't explain a gap that long. The host is doing something else, and the `cuda_api_sum` shows what: 12 calls to `cudaStreamSynchronize` with an average of 905,000 ns. The step copies a loss value to the host, a `.item()`, and the host waits, decides, and only then launches the next kernel. [Chapter 5](/c/05-profiling) showed how a sync hides in innocent code. The duty is 0.30, as low as in the launch-bound trace, and the fix is the opposite: more kernels in flight, not fewer launches.

The lab's `diagnose_timeline` encodes the difference in three tests over the kernel events, in order:

1. The duty is at least 0.85: `kernel-bound`. The GPU is busy and the kernels are the work.
2. The median gap is at least 10 launch calls: `sync-bound`. The host is blocked or busy elsewhere.
3. The median kernel is shorter than 2 launch calls: `launch-bound`. Otherwise `kernel-bound`.

Two choices in that rule come from the traps in the data. It uses only the kernel events, because the 4-byte copy in `SYNC_BOUND` sits inside the host's pause. If you include it, it splits each long gap into a 1.5 µs sliver and a 900 µs gap, and the median of the gaps becomes the sliver. And it uses the median gap, not the mean. The held-out trace is 100 small kernels spaced like `LAUNCH_BOUND`, then one 20 ms gap before the last kernel, as if the program printed something at the end. The mean gap of that trace is about 200 µs, forty launch calls, and it looks sync-bound. The median is 4.5 µs. One stall at the end of a trace is an event, and the median is the pattern.

The thresholds are again this course's. The 85 percent duty, the 10 launch calls, and the 2 launch calls are rules of thumb, and the 10 is wide on purpose: it separates "the host was queueing" from "the host was elsewhere" by an order of magnitude.

## Putting the two tools together

This section answers one question: which report do you open first, and what does each one rule out?

Follow [chapter 5](/c/05-profiling)'s funnel with the two exports in hand. The Nsight Systems export is steps 2 and 3: it names the hot kernels by total time and tells you whether the GPU is idle. The Nsight Compute report is step 4. The functions of this chapter automate the two decisions that pick the next step.

| You see | In | It means | Next |
|---|---|---|---|
| Duty below 0.85, median gap about one launch call | `nsys` | Launch-bound | Fuse, batch, or capture a graph |
| Duty below 0.85, median gap far above a launch call | `nsys` | Sync-bound | Find the `cudaStreamSynchronize` and the `.item()` behind it |
| Duty near 1, one kernel dominates | `nsys` | Kernel-bound | Open that kernel in `ncu` |
| Load efficiency below 0.5 | `ncu` | Uncoalesced | Fix the index map |
| Conflict fraction at least a quarter | `ncu` | Bank conflicts | Pad or swizzle the tile |
| Waves above 1 with efficiency below 0.8 | `ncu` | Tail effect | Change the grid or persist the blocks |
| DRAM at least 50% | `ncu` | Memory-bound | Move fewer bytes |
| SM at least 60% | `ncu` | Compute-bound | Use tensor cores, or accept |
| Neither | `ncu` | Latency-bound | Read the stalls and the occupancy |

> [!TIP] Run the classifier on every kernel you tune
> Write the parser once and run `diagnose` before and after each change. When the label changes, say from `uncoalesced` to `memory-bound`, you fixed the cause. When the time improves and the label doesn't change, you might have fixed a symptom.

## What goes wrong

- **Trusting a Speed of Light percentage alone.** 62 percent DRAM throughput on a strided kernel is wasted bandwidth. Count sectors before you decide the kernel is at its limit.
- **Flagging sectors per request above 8.** A coalesced `float4` load is 16. Use useful bytes over fetched bytes, which doesn't depend on the access width.
- **Counting an FFMA once.** Every GEMM then reports half its FLOP/s and half its intensity, and it lands on the wrong side of the ridge.
- **Dividing by zero.** A kernel with no FLOPs has no arithmetic intensity, and a kernel with no shared memory has no conflict fraction. Guard each ratio, and define it as zero where there is no traffic.
- **Treating a conflict count as a conflict.** `sgemm_tiled` has a few percent conflicted wavefronts and is at 83 percent of peak. Judge by the fraction, with a threshold, and not by whether the count is nonzero.
- **Calling any fractional wave count a tail effect.** A grid of 303.4 waves loses 0.2 percent. The tail matters when the grid is a few waves and the last is mostly empty.
- **Reading a dominant stall as a cause.** `reduce_tail` shows math pipe throttle and sits half idle. The stall describes the warps that ran, not the SMs that didn't.
- **Adding durations across streams.** Concurrent streams add up to more than the wall time. Merge the intervals.
- **Averaging gaps.** One long gap, from a final sync or a print, turns a launch-bound trace into a sync-bound one. Use the median.
- **Reading a column by position.** The `nsys` and `ncu` column sets change between versions. Read by header name, and skip the lines before the header.
- **Quoting a profiled duration as a benchmark.** `ncu` locks clocks and flushes caches. The durations in this chapter's fixtures are examples of that profile, and nothing here is a benchmark of an A100.

> [!RECAP]
> - A report is data. Parse it with the `csv` module, skip the banner lines, strip thousands separators, drop `n/a`, and keep base units.
> - The roofline placement is DRAM bytes, FP32 FLOPs as `FADD + FMUL + 2 × FFMA`, and the duration. `vec_add` is at 0.083 FLOP per byte and 1280 GB/s, and `sgemm_tiled` at 683 FLOP per byte and 16.2 TFLOP/s, on either side of the 10.08 ridge.
> - A stall reason names a symptom. Sector efficiency, the conflict fraction, and the wave count name the cause. Test them before the two Speed of Light percentages.
> - Theoretical occupancy is the smallest launch limit times the warps per block over 64. Achieved occupancy falls below it when blocks end unevenly, and a tail effect halves it.
> - The classifier order is uncoalesced, bank conflicts, tail effect, memory-bound, compute-bound, latency-bound.
> - From an `nsys` export, the busy time is the union of the intervals, the gap statistic is the median, and the comparison is against a roughly 5 µs launch call: gaps of one launch call mean launch-bound, and gaps of many launch calls mean the host is elsewhere.

## Check your understanding

> [!QUESTION] A kernel reports 70 percent DRAM throughput and 5 percent SM throughput. Is it memory-bound?
> You can't tell yet. Check the sector efficiency first. If the kernel fetches 8 times the bytes it uses, 70 percent of the bandwidth is mostly waste, and the label is uncoalesced. If the efficiency is 1 and the conflict and wave counters are clean, it's memory-bound and at its roof.

> [!QUESTION] The counters say 1,000,000 FADD, 500,000 FMUL, and 2,000,000 FFMA in 10 µs, with 4,000,000 DRAM bytes. What is the arithmetic intensity and the achieved GFLOP/s?
> The FLOPs are $1{,}000{,}000 + 500{,}000 + 2 \times 2{,}000{,}000 = 5{,}500{,}000$. The intensity is $5.5 \times 10^6 / 4 \times 10^6 = 1.375$ FLOP per byte, below the FP32 ridge of 10.08, so the kernel is memory-bound by arithmetic. It did $5.5 \times 10^6 / 10^{-5}\ \text{s} = 550$ GFLOP/s. The roof at 1.375 is $1.375 \times 1935 = 2661$ GFLOP/s, so the kernel is at 20.7 percent of it.

> [!QUESTION] A kernel uses 96 registers per thread, 128-thread blocks, and no shared memory. What is its theoretical occupancy and its limiter?
> A warp holds $96 \times 32 = 3072$ registers, which is a multiple of 256, so $\lfloor 65{,}536 / 3072 \rfloor = 21$ warps fit. A block has 4 warps, so 5 blocks fit, which is 20 warps. The warp limit is 16 blocks and the block limit is 32, so registers limit it to 5 blocks, and the occupancy is $20/64 = 31.25$ percent.

> [!QUESTION] A 32 by 32 fp32 tile is read by column. How many wavefronts does one warp load take, and what is the conflict fraction if the tile is stored by row?
> The column read puts all 32 lanes in one bank, so it takes 32 wavefronts, 31 more than the minimum of 1. The row store takes 1 wavefront and has no conflict. Per pair of accesses, the conflict fraction is $31 / (32 + 1) = 0.939$. Padding the row to 33 words spreads the column across all 32 banks, and the fraction is 0.

> [!QUESTION] A kernel runs 217 blocks with 1 resident block per SM on 108 SMs. What is its wave efficiency, and should you fix it?
> It needs $217/108 = 2.009$ waves, so 3 whole waves, and the efficiency is $2.009/3 = 0.670$. That is below 0.8 and above one wave, so the lab labels it a tail effect. Whether to fix it depends on the time: the third wave runs one block, so you pay a third of the kernel's time for one block of work. Making the grid 216 blocks, or splitting the work into smaller blocks, recovers that.

> [!QUESTION] A trace has 500 kernels of 4 µs, a 5 µs launch call, and a duty of 0.4. Another has 20 kernels of 2 ms and a duty of 0.4. Both are 60 percent idle. How do the fixes differ?
> The first is launch-bound: the kernels are shorter than 2 launch calls, and the gaps are about a launch. Fuse the kernels or replay them as a graph. The second has gaps of about 3 ms, hundreds of launch calls, so the host is blocked or doing something else. Look for a synchronization, such as a `.item()`, or host work between launches, and keep more kernels in flight.

> [!QUESTION] Two streams each run for 1 second, and they overlap for 0.5 second. What is the busy time of the GPU?
> The union is $1 + 1 - 0.5 = 1.5$ seconds. The sum of the durations is 2 seconds, and dividing it by the 1.5 second span gives a ratio of 1.33, which can't be a duty cycle. Merge the intervals, and the duty is 1.0.

## Lab

> [!TRY] Read profiler reports
> `fixture_ncu.py` holds a recorded `ncu --csv` report of six kernels, and `fixture_nsys.py` holds three `nsys stats` exports. Write `parse_ncu_csv`, then `flops` and `roofline_from_counters`, then the occupancy, sector, conflict, wave, and stall functions, and last `diagnose`. For the timelines, write `parse_nsys_csv`, `gpu_events`, `busy_and_gaps`, `launch_avg_ns`, and `diagnose_timeline`. You pass when the roofline numbers match the arithmetic in this chapter, the six recorded kernels get their labels, and the classifier also labels held-out kernels and traces that you don't see: a coalesced `float4` copy, a padded transpose, a half-full grid, and a trace with one long gap at the end. The lab reports `sgemm_achieved_gflops`, `sgemm_pct_of_roof`, `overlap_duty`, and `launch_bound_duty`. It runs entirely on a CPU. A final check profiles one PyTorch add with `ncu` when a GPU, `ncu`, and permission to read performance counters are present, and otherwise skips with a message.

The recorded reports are synthesized, and that's the point and the limit of this lab. No GPU produced the counters. Their shapes follow what the tools write, and their values agree with one another on an A100 80GB PCIe, so the arithmetic in this chapter closes. A real report has noise, extra sections, and counters this lab doesn't read. When you profile a kernel of your own, run your parser on it before you trust the label.

## Further reading

- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html): the metric names for stall reasons, occupancy limits, and waves per multiprocessor, and the memory-table quantities for sectors, wavefronts, and bank conflicts.
- [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html): `--csv`, `--print-units`, `--metrics`, `-k`, `-s`, and `-c`.
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html): `nsys profile`, `nsys stats`, and the report names, including `cuda_api_sum`.
- GPU MODE, lecture 1, "Profiling and Integrating CUDA kernels in PyTorch" (Mark Saroufim), lecture 8, "CUDA Performance Checklist" (Mark Saroufim), and lecture 16, "On Hands profiling" (Taylor Robbie), from the [lecture index](https://github.com/gpu-mode/lectures). Only the titles were read, from the index.
- [Chapter 5](/c/05-profiling), for the funnel and the report sections, [chapter 4](/c/04-roofline-and-latency) for the roofline and wave formulas, and [chapter 8](/c/08-shared-memory-and-transpose) for the bank model.
