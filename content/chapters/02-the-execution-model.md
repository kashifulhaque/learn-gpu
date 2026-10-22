---
title: The execution model
slug: 02-the-execution-model
part: "Part 1 — The machine"
summary: How a launch becomes grids, blocks, and warps; what a warp costs when its threads disagree; and how registers and shared memory cap the number of warps an SM can hold.
minutes: 120
gpu: true
objectives:
  - "Map a launch's grid, blocks, and threads onto the SMs, and say what the hardware promises about the order blocks run in."
  - "Explain how a warp executes under SIMT, and why four warp schedulers per SM matter."
  - "Predict what branch divergence costs, and tell a divergent branch from a warp-uniform one."
  - "Say what independent thread scheduling changed on Volta and later, and when you must call __syncwarp."
  - "Compute occupancy for the A100 from threads, warps, blocks, registers, and shared memory, and name the limiter."
  - "Argue, with Little's law and Volkov's results, why lower occupancy can be faster."
  - "Use __launch_bounds__ and -maxrregcount, and recognize when register spilling costs more than the occupancy it buys."
  - "Explain launch overhead and why the CPU and GPU run asynchronously."
lab: 02-occupancy
---

# The execution model

> [!TLDR]
> - A launch is a grid of blocks, and a block is a group of threads. The hardware places each block on one SM, keeps it there until it finishes, and promises nothing about the order blocks run in.
> - The SM runs threads in warps of 32. Each of the A100's four warp schedulers per SM issues one instruction per cycle for one warp, and a warp whose threads take different branches runs each branch in turn.
> - Occupancy is the resident warps divided by 64. Registers, shared memory, block size, and the 32-block cap set it, and the smallest of those four limits wins. A kernel with 255 registers per thread and 256-thread blocks holds 8 of the 64 warps, 12.5%.
> - Occupancy is a means, not a goal. A thread with several independent loads in flight can saturate memory at a fraction of full occupancy, so raising it can slow a kernel down.
> - A launch costs about 5 microseconds of CPU time and returns before the GPU finishes, so a kernel shorter than that is limited by the launch, and a host timer measures the wrong thing.

[Chapter 1](/c/01-the-gpu-at-a-glance) described the A100 as a pile of resources: 108 SMs, a 256 KB register file per SM, 164 KB of shared memory per SM. This chapter describes the rules for spending them. When you launch a kernel, the hardware decides how many of your threads run at the same time, which ones run together, and what happens when they disagree. Those rules decide whether a kernel that is correct also runs fast.

You write a kernel as though one thread runs alone, and you launch it for millions. The distance between those two pictures is the execution model. A warp that diverges, a block that asks for too many registers, or a launch that does too little work each shows up as a kernel that is slower than the roofline predicts, and none of them is visible in the arithmetic. By the end you can compute which resource caps your kernel before you run it, and your lab checks the computation against the driver.

## Before you start

**Launch syntax.** A CUDA launch writes `kernel<<<blocks, threads>>>(args)`: the first number is how many blocks, the second is how many threads per block. [Chapter 12 of the companion course](https://qwen.ifkash.dev/c/12-your-first-cuda-kernel) writes a first kernel line by line, and this chapter assumes you can read one.

**The SM.** The streaming multiprocessor is the unit that runs blocks. The A100 80GB PCIe has 108 of them, and each has four sub-partitions with their own warp scheduler, register file, FP32 cores, and tensor core. See [chapter 1](/c/01-the-gpu-at-a-glance).

**Occupancy.** The fraction of an SM's warp slots that hold a resident warp. This chapter defines it exactly in the section on the four limits.

## Grids, blocks, and threads

A launch creates one *grid*, the grid is made of *blocks*, and each block is made of *threads*. The section answers one question: what does the programmer choose, and what does the hardware choose?

You choose the shape. A grid has up to three dimensions of blocks, and a block has up to three dimensions of threads. The hardware limits on compute capability 8.0 are 1,024 threads per block, a grid of up to $2^{31}-1$ blocks in x and 65,535 in y and z, and block dimensions of at most 1,024 in x, 1,024 in y, and 64 in z, with the product capped at 1,024 ([CUDA Programming Guide, compute capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)).

Inside the kernel, each thread learns where it is from built-in variables, and the standard one-dimensional index is the first thing every kernel computes:

```cuda
__global__ void scale(const float* x, float* y, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's element
    if (i < n) y[i] = a * x[i];                     // guard the tail
}

// Host side: round the block count up so every element has a thread.
int threads = 256;
int blocks = (n + threads - 1) / threads;
scale<<<blocks, threads>>>(x, y, 2.0f, n);
```

The hardware chooses everything after the launch: which SM gets which block, when, and in what order. The CUDA programming model requires that blocks can run in any order, in parallel or in series, and it gives you no way to ask a block what another block is doing. That freedom is what lets the same binary fill 108 SMs on an A100 or 132 on an H100.

> [!KEY] A block is the unit of scheduling, and a thread is the unit of code
> You write per-thread code, but the hardware places whole blocks on SMs and runs the threads of a block on one SM. Threads of one block can share memory and synchronize with `__syncthreads()`. Threads of different blocks can't, so any algorithm that needs them to cooperate needs a second kernel or an atomic.

A Triton program plays the role of a CUDA block. When you write `pid = tl.program_id(0)` in [chapter 14](/c/14-triton), `pid` is a block index, and `num_warps` sets how many warps run that block's code. The execution model in this chapter applies to Triton kernels too, because Triton compiles to the same machine.

## How blocks reach SMs

The block scheduler hands blocks to SMs with free resources, and a block stays on its SM until its last thread exits. The section answers one question: how many blocks run at once, and what happens to the rest?

A block is *resident* on an SM from the moment the SM accepts it until it finishes. An SM accepts a block only if it has room for all of the block's threads, registers, and shared memory, and if it holds fewer than 32 blocks. When a block finishes, the scheduler places a waiting block in its place. A block never moves, and it is never paused to make room for another.

> [!EXAMPLE] A vector add over $2^{24}$ floats
> With 256 threads per block, the grid has $2^{24} / 256 = 65{,}536$ blocks. If the kernel uses 32 registers per thread or fewer, the 2,048-thread cap is the limit, so each SM holds $2048 / 256 = 8$ blocks, and the GPU holds $108 \times 8 = 864$ blocks at once. The grid runs in $65{,}536 / 864 = 75.9$, so 76, waves of resident blocks. The last wave has $65{,}536 - 75 \times 864 = 736$ blocks, 85% of a full wave.
>
> Those 864 resident blocks hold $864 \times 256 = 221{,}184$ threads, which is $108 \times 2048$. Most of them are waiting on memory at any instant, and that waiting is the point of the next sections.

Two consequences follow. First, a grid with fewer blocks than SMs leaves SMs idle: a 64-block launch on 108 SMs uses at most 64 of them. Second, a grid whose block count just exceeds a multiple of a wave leaves a nearly empty last wave. [Chapter 4](/c/04-roofline-and-latency) computes this *wave quantization* for matrix multiplies, where NVIDIA's own guide gives a worked example of throughput roughly halving when 108 tiles become 117.

On Hopper (compute capability 9.0 and later) a group of blocks called a *cluster* is guaranteed to run at the same time on neighboring SMs, which lets them share memory. The A100 has no clusters, so the labs don't use them.

## Warps and SIMT

The SM doesn't run threads one at a time. It runs them in groups of 32 called *warps*, and a warp is the unit the scheduler issues instructions for. The section answers one question: what does it mean that 32 threads execute together?

The threads of a block are numbered in a line, with `threadIdx.x` varying fastest, then y, then z:

$$
\text{linear id} = x + y \cdot D_x + z \cdot D_x D_y
$$

Warp 0 holds linear ids 0 to 31, warp 1 holds 32 to 63, and so on. A block of 256 threads is 8 warps, and a block of 100 threads is 4 warps with the last one 4 threads full. The warp size is 32 on every NVIDIA architecture in this course.

A warp follows the *SIMT* model, single instruction, multiple threads. All 32 threads, called *lanes*, execute the same instruction at the same time, each on its own registers. One `FFMA` instruction is 32 fused multiply-adds, one per lane, and a load is 32 addresses, one per lane. A warp's instruction is the smallest thing the hardware issues.

Each A100 SM has four sub-partitions, which the programming guide and the Ampere whitepaper call sub-cores, and each owns a warp scheduler, a quarter of the register file, 16 FP32 cores, and one tensor core. Every cycle, a scheduler picks one resident warp that is ready and issues one instruction for it. These are the numbers that follow:

| Quantity per SM | Value | Arithmetic |
|---|---|---|
| Warp schedulers | 4 | one per sub-partition |
| Warp instructions issued per cycle | at most 4 | 4 schedulers x 1 |
| FP32 cores per sub-partition | 16 | 64 per SM / 4 |
| Cycles of FP32 pipe per warp `FFMA` | 2 | 32 lanes / 16 cores |
| FP32 FLOP per cycle | 128 | 64 cores x 2 FLOP per FMA |

The last row connects to the peak you met in [chapter 0](/c/00-what-you-are-building): $108 \times 1.41 \times 10^{9}\ \text{Hz} \times 128 = 19.5$ TFLOP/s, the rated FP32 rate. A single warp can use a sub-partition's FP32 pipe at most half the time, because one warp `FFMA` occupies the 16 cores for two cycles. The other issue slots go to loads, integer arithmetic, and other warps.

> [!NOTE] A thread is not a core
> "6,912 CUDA cores" and "221,184 resident threads" are different kinds of number. A core is one lane of an arithmetic pipe. A resident thread is a set of registers and a program counter parked on the SM, and an SM keeps far more of them than it has lanes so that something is always ready to issue.

## Branch divergence

When the lanes of a warp reach an `if` and disagree, the warp runs both sides, one after the other, with the lanes that didn't take each side switched off. The section answers one question: what does disagreement cost?

A warp has one instruction stream. Where lanes take different paths, the hardware executes each path in turn with an *active mask* that names the lanes it applies to, and the warp's *reconvergence* point is where the paths rejoin. The cost of a branch is the sum of the lengths of the paths that at least one lane takes. The CUDA guide states the scope precisely: divergence happens only within a warp, and different warps can take different paths at no cost.

```cuda
// Diverged: lanes of one warp split on lane parity.
// The warp executes path_a, then path_b, with half the lanes idle in each.
if (threadIdx.x % 2 == 0) y = path_a(x); else y = path_b(x);

// Uniform at warp granularity: whole warps split on warp parity.
// Each warp executes one path and nothing else.
if ((threadIdx.x / 32) % 2 == 0) y = path_a(x); else y = path_b(x);
```

Both statements split the block's threads in half, and the work and results are the same shape. The first costs $t_a + t_b$ for every warp, and the second costs $t_a$ for half the warps and $t_b$ for the other half. This is the shape of the experiment in your lab.

With $k$ paths of equal length taken by lanes of one warp, the warp's time is $k$ times one path's time:

$$
\boxed{\text{slowdown} = k \quad\text{(throughput-bound, equal paths)}}
$$

The *lane efficiency* is $1/k$: of every 32 lanes' worth of issue slots, only $32/k$ lanes do useful work. The slowdown is the prediction for a kernel that is limited by instruction issue or by the FP32 pipe, as the lab's is. A kernel limited by memory hides some of it, because the memory system, not the instruction stream, is the bottleneck.

```viz
02-divergence
```

Three things make real kernels more forgiving than the worst case.

- **Predication.** For a short branch, such as `x = (c > 0) ? a : b`, the compiler emits both sides unconditionally and uses a predicate to choose. No branch exists, so no divergence exists, and the cost is the instructions on both sides.
- **Uniform conditions.** A condition that is the same for all 32 lanes, such as a loop bound that comes from a kernel argument or from the block index, never diverges.
- **The tail guard.** `if (i < n)` diverges only in the one warp that straddles `n`, so it costs one warp's partial pass per launch.

Real divergence comes from data-dependent control flow. If each lane loops a different number of times, the warp loops as many times as its slowest lane. If a mixture-of-experts kernel gives each lane a different expert, the warp runs one expert's code per distinct expert it holds. The fix is usually to arrange data so that the lanes of a warp agree: sort tokens by expert, group sequences by length, or assign one row to a warp and let the warp's lanes share the loop.

> [!TIP] Divergence is per warp, so look at your indices
> Before blaming divergence, check what the condition depends on. A condition on `blockIdx.x` or on `threadIdx.x / 32` is free. A condition on `threadIdx.x % 32`, or on a loaded value that varies across neighboring elements, is a candidate. [Chapter 5](/c/05-profiling) shows the profiler counters that count divergent branches.

## Independent thread scheduling and `__syncwarp`

Before Volta, a warp had one program counter, and the lanes of a warp stayed in lockstep by construction. Volta (compute capability 7.0) and every architecture since, including the A100, give each thread its own program counter and call stack. The section answers one question: what code does that change break?

Per-thread program counters let the hardware interleave the paths of a divergent branch instead of finishing one before starting the other, and they let a thread that waits on a lock yield to the thread that holds it. A schedule optimizer still groups threads at the same instruction into SIMT units, so performance in the common case is unchanged. The cost is that the convergence you used to get for free is no longer promised. NVIDIA's guide states that threads "can yield execution at a per-thread granularity", and the warp-primitives article it links says that convergence is only guaranteed inside the explicit synchronous warp-level primitives ([Lin and Grover, 2018](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)).

The practical rule is this: code that assumed the 32 lanes of a warp ran in lockstep must call `__syncwarp()`, or use a `*_sync` primitive, wherever it relied on that. The classic case is the last warp of a shared-memory reduction, which pre-Volta code wrote with `volatile` and no barrier. The following version is correct on Volta and later:

```cuda
// Sum smem[0..63] using the first warp. Lane = threadIdx.x, in 0..31.
if (lane < 32) {
    float v = smem[lane] + smem[lane + 32];
    for (int k = 16; k > 0; k >>= 1) {
        __syncwarp();            // every lane has finished reading smem
        smem[lane] = v;
        __syncwarp();            // every lane's write is visible
        v += smem[lane + k];
    }
    // lane 0 now holds the sum
}
```

The first `__syncwarp()` in each iteration keeps a fast lane from overwriting `smem[lane]` while a slower lane still needs to read the old value. The second makes sure every write has landed before any lane reads a neighbor's. Without them the hardware is free to run the lanes in any interleaving, and the sum is wrong only sometimes.

For a reduction that stays in registers, the shuffle intrinsics both exchange data and synchronize the lanes that the mask names:

```cuda
__device__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;  // lane 0 holds the sum of all 32 lanes
}
```

The mask `0xffffffff` names all 32 lanes, so every lane must reach the call. The `*_sync` rules in the programming guide are strict: each calling thread must have its own bit set in the mask, every thread the mask names must execute the intrinsic with the same mask, and breaking either is undefined behavior. A mask that names a lane that exited or took another branch can hang or return garbage. [Chapter 6](/c/06-reductions) builds the whole reduction from these two pieces.

> [!WARNING] Don't build the mask from `__activemask()`
> `__activemask()` reports which lanes happen to be converged right now, and with independent scheduling that set can be smaller than the lanes you meant. Lin and Grover's advice is explicit: don't use it as the mask value. Compute the mask from the program's logic, or use the full mask when the whole warp reaches the call.

> [!TIP] Test whether scheduling is your bug
> If old warp-synchronous code gives different results on Volta and later, rebuild it for the earlier scheduling with `-arch=compute_60 -code=sm_70`. If the bug disappears, the code depended on lockstep and needs `__syncwarp`.

## Why warps are cheap: hiding latency

A load from HBM takes hundreds of cycles, and the SM doesn't wait: it issues another warp's instruction in the meantime. The section answers one question: how does the SM stay busy while one warp is stalled?

A warp that waits on a load, or on the result of a previous instruction, is *stalled*. The scheduler skips it and issues from a warp that is *eligible*, one whose next instruction has its operands ready. Switching between warps costs nothing, because every resident warp's registers stay in the register file and no state is saved or restored. That is the opposite of a CPU thread switch, and it is why the GPU keeps many warps resident.

How many warps does it take? Little's law gives the amount of work that must be in flight to keep a pipe full:

$$
\boxed{N_{\text{in flight}} = \hla{L} \times \hlb{\lambda}}
$$

where $\hla{L}$ is the latency of an operation and $\hlb{\lambda}$ is the throughput the hardware can sustain. [Chapter 4](/c/04-roofline-and-latency) develops it fully. Here are two instances.

**Memory.** Luo et al. measured a global-memory latency of 466.3 cycles on an A100 PCIe 40GB, which is 330.7 ns at 1.41 GHz. To sustain the rated 1935 GB/s the GPU needs $1935 \times 10^{9} \times 330.7 \times 10^{-9} = 640$ KB in flight, which is 5.9 KB per SM across 108 SMs. To sustain the 1275 GB/s that a copy measured in learn-inference, the need is $1275 \times 10^{9} \times 330.7 \times 10^{-9} / 108 = 3.9$ KB per SM.

**Arithmetic.** A dependent chain of FP32 operations has a latency of a few cycles. Microbenchmarks of Volta ([Jia et al.](https://arxiv.org/abs/1804.06826)) report about 4 cycles, and this chapter assumes the A100 is similar. A sub-partition completes 0.5 warp instructions per cycle (the 16-core pipe in the table above), so it needs $4 \times 0.5 = 2$ independent warp instructions in flight, which is 2 warps when each warp has no instruction-level parallelism. That is 8 warps per SM, 12.5% of the 64 slots, to keep the FP32 pipes busy on dependent arithmetic.

The two numbers say different things. Arithmetic needs few warps. Memory needs 5.9 KB per SM in flight, and how many warps that takes depends on how many bytes each warp has outstanding, which is the subject of the section on Volkov's results.

## Occupancy: four limits

Occupancy is the number of warps resident on an SM divided by the most it could hold, and four resources set it. The section answers one question: given a kernel's resources, how many blocks fit on one SM?

$$
\text{occupancy} = \frac{\text{resident warps per SM}}{64}
$$

The A100's limits come from compute capability 8.0, and the table adds other architectures for contrast. Every value is from the [CUDA Programming Guide, compute capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html), and the 10.0 column is from NVIDIA's Blackwell tuning guide.

| Limit per SM | CC 7.5 | CC 8.0 (A100) | CC 8.6 | CC 9.0 (H100) | CC 10.0 (B200) |
|---|---|---|---|---|---|
| Resident blocks | 16 | 32 | 16 | 32 | 32 |
| Resident warps | 32 | 64 | 48 | 64 | 64 |
| Resident threads | 1,024 | 2,048 | 1,536 | 2,048 | 2,048 |
| 32-bit registers | 64K | 64K | 64K | 64K | 64K |
| Shared memory | 64 KB | 164 KB | 100 KB | 228 KB | 228 KB |
| Shared memory per block | 64 KB | 163 KB | 99 KB | 227 KB | 227 KB |

Every compute capability allows at most 255 registers per thread. The A100's 64K registers are 256 KB, and the same file is 256 KB on every architecture in the table.

A block fits as many times as the tightest of four resources allows. Let $w$ be the warps in a block:

$$
\text{blocks per SM} = \min\Big(\hla{\Big\lfloor \tfrac{64}{w} \Big\rfloor},\ \hlb{32},\ \hlc{\Big\lfloor \tfrac{W_{\text{reg}}}{w} \Big\rfloor},\ \hld{\Big\lfloor \tfrac{164\ \text{KiB}}{s} \Big\rfloor}\Big)
$$

Here $\hla{\lfloor 64/w \rfloor}$ is the warp limit (equivalently the 2,048-thread limit), $\hlb{32}$ is the block cap, $\hlc{W_{\text{reg}}}$ is the number of warps the register file can hold, and $s$ is the shared memory charged to one block. The tightest term is the *limiter*, and it names what to change.

**Registers.** The compiler assigns each thread a fixed number of registers, and the warp is charged for all 32 lanes. A kernel that uses $r$ registers per thread charges its warp $32r$ registers, rounded up to a multiple of 256. Registers are allocated to a warp inside one of the four sub-partitions, so the SM holds a quarter of the 65,536 registers in each. The rule NVIDIA's retired occupancy-calculator spreadsheet used is to count whole warps that fit per sub-partition, and then multiply by 4.

**Shared memory.** A block is charged what it requests, rounded up to 128 bytes, plus 1 KiB that the runtime reserves for each block. The SM has up to 164 KiB, and one block may request at most 163 KiB.

> [!WARNING] The granularity rules are an approximation
> The 256-register unit, the grouping by sub-partition, the 128-byte shared-memory unit, and the 1 KiB reservation are the rules of the retired calculator, which the course's sources confirm only through a developer-forum post, not a primary NVIDIA document. The tables in this chapter use them, and the limits in the table above are confirmed. Your lab compares the rules with `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, the function the driver itself answers, and a mismatch there beats a rule here.

> [!EXAMPLE] 256-thread blocks on the A100
> A block of 256 threads is $w = 8$ warps. The warp limit is $64/8 = 8$ blocks, and the block cap of 32 is never the limit. Registers decide the rest:
>
> | Registers per thread | Registers per warp (rounded) | Warps by registers | Blocks by registers | Resident blocks | Occupancy |
> |---|---|---|---|---|---|
> | 32 | 1,024 | 64 | 8 | 8 | 100% |
> | 64 | 2,048 | 32 | 4 | 4 | 50% |
> | 128 | 4,096 | 16 | 2 | 2 | 25% |
> | 255 | 8,192 (charged as 256 per thread) | 8 | 1 | 1 | 12.5% |
>
> At 64 registers, the register file holds $65{,}536 / 2{,}048 = 32$ warps, and 32 of 64 slots is 50%.

For shared memory, take a block of 256 threads that declares a 48 KiB array. It is charged $48 + 1 = 49$ KiB, so $164 / 49 = 3.35$, which rounds down to 3 blocks. That is $3 \times 8 = 24$ warps, 37.5% occupancy, regardless of registers. A block that asks for the maximum 163 KiB is charged 164 KiB and is the only block on its SM.

```viz
02-occupancy-calculator
```

You rarely compute this by hand. `nvcc -Xptxas -v` prints each kernel's register count and spills, Nsight Compute reports theoretical and achieved occupancy and names the limiter ([chapter 5](/c/05-profiling)), and the CUDA runtime answers directly:

```cuda
int blocks_per_sm = 0;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocks_per_sm, my_kernel, /*blockSize=*/256, /*dynamicSmemBytes=*/0);
float occupancy = blocks_per_sm * (256 / 32) / 64.0f;
```

Two cautions apply to every theoretical figure. Theoretical occupancy assumes the grid is big enough to fill every SM, and a short grid or a ragged last wave lowers *achieved* occupancy below it. And the limiter changes as you change the kernel: cut registers and the register limit relaxes, but the shared-memory or warp limit may become the new ceiling.

## Lower occupancy can be faster

Occupancy is a way to hide latency, not a measure of how busy the machine is. The section answers one question: when does chasing occupancy lose?

The standard advice is to run more threads per SM because that is the way to hide latency. Vasily Volkov's 2010 GTC talk, [Better Performance at Lower Occupancy](https://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf), shows that it isn't the only way. Work in a single thread can also hide latency, if the thread has several independent operations ready at once. That is *instruction-level parallelism*, ILP, and by Little's law it substitutes for thread-level parallelism, TLP.

The talk's results are measured on older GPUs, and the hardware is named with each:

| Experiment | GPU | Result |
|---|---|---|
| CUFFT 2.2 to 2.3: 256 to 64 threads per block | G80 | Occupancy 33% to 17%, performance 45 to 93 GFLOP/s (2x) |
| CUBLAS 1.1 to 2.0: 512 to 64 threads per block | G80 | Occupancy 67% to 33%, performance 128 to 204 GFLOP/s (1.6x) |
| Dependent FMA chain, 1 independent chain per thread | GTX 480 | 576 threads per SM to reach peak |
| Same, 2 independent chains per thread | GTX 480 | 320 threads per SM |
| Same, 4 independent chains per thread | GTX 480 | 192 threads per SM |
| Copy, 4 floats loaded per thread before any is used | GTX 480 | Full bandwidth at 25% occupancy |

The numbers don't transfer to the A100, because the latencies and the hardware differ, but the mechanism does. A thread that issues four independent loads before it uses any of them has four loads in flight, so a quarter as many threads keep the same number of bytes outstanding. Volkov's measured GTX 480 copy needed only 25% occupancy once each thread loaded four floats.

The same arithmetic applies on the A100 as a prediction. The memory section needed 5.9 KB per SM in flight for the full rating, and the table shows how it divides:

| Per-thread load | Threads per SM for 5.9 KB | Occupancy |
|---|---|---|
| 1 float (4 B) | 1,482 | 72% |
| 1 `float4` (16 B) | 371 | 18% |
| 2 `float4` (32 B) | 186 | 9% |

Divide 5,925 B by the bytes one thread keeps in flight, and round up. With 1 `float4` load per thread, 371 threads per SM, 12 warps, are enough to cover HBM latency at the rating on paper. The roofline rule you will use again in [chapter 3](/c/03-memory-and-bandwidth) is that bytes in flight, not occupancy, saturates the memory system, and your lab in that chapter measures the effect.

A measurement on the course's lab A100 points the same way. In the companion course's fused RMSNorm for rows of 5,120 elements, the number of warps per block, `num_warps`, swept as 8, 16, and 32, ran in 0.0888, 0.0908, and 0.0955 ms ([chapter 13](https://qwen.ifkash.dev/c/13-fusion-in-triton)). More warps made the kernel slower, because each extra warp had less of the row to work on and added reduction cost, not because it hid more latency.

> [!KEY] Occupancy is a floor you need, not a score you maximize
> A kernel needs enough warps, or enough work per warp, to cover its latency. Above that point, more occupancy adds nothing, and it usually costs registers or shared memory that the kernel would rather spend on tiles. Raise occupancy when the profiler says the schedulers have no eligible warp and the stalls are latency. Lower it, or leave it, when the registers buy larger tiles or more ILP.

That trade is why the fastest matrix multiplication kernels hold a big tile of the output in registers and run few warps per SM. [Chapter 10](/c/10-gemm-from-scratch) builds that kernel, and Volkov's own case study of SGEMM makes the same argument.

## Registers: `__launch_bounds__`, `-maxrregcount`, and spilling

The compiler picks the register count, and you can ask it to pick a different one. The section answers one question: how do you trade registers for occupancy, and what does it cost?

If a kernel needs more registers than the budget allows, the compiler *spills* the extra values to *local memory*. Local memory is private to a thread but physically global memory, cached through L1 and L2, and each thread may use up to 512 KB of it. A spill is a store and a load that the unconstrained kernel didn't have, and the loads sit on the critical path.

You have two controls. The first is per kernel, `__launch_bounds__`:

```cuda
// Promise at most 256 threads per block, and ask for at least 2 resident blocks per SM.
__global__ void __launch_bounds__(256, 2) my_kernel(float* x) { /* ... */ }
```

The first argument is the largest block you will launch. The second, optional, is a minimum number of blocks per SM that you want, and the compiler uses the register count that makes it possible: for 256 threads and 2 blocks that is $65{,}536 / (2 \times 256) = 128$ registers per thread at most. Even without the second argument, the first lets the compiler know that blocks are small, so it need not leave room for 1,024 threads.

The second is the compile flag `-maxrregcount=N`, which caps every kernel in the file:

```python
module = load_cuda("probe", cuda_src, cpp_src, ["run"],
                   extra_cuda_cflags=["-maxrregcount=64", "-Xptxas", "-v"], verbose=True)
```

With `-Xptxas -v` and `verbose=True`, the build log prints `registers`, `spill stores`, and `spill loads` for each kernel. The same numbers come from Triton: the object a launch returns exposes `n_regs` and `n_spills`.

> [!EXAMPLE] Capping registers on a 256-thread kernel
> A kernel at 128 registers has 25% occupancy, from the table above. Capping it at 64 doubles occupancy to 50%, and each thread now has half the registers. If the kernel's live values fit, the cap is free, and the kernel gains whatever the extra warps buy. If they don't, the spills cost more than the occupancy returns, and the kernel is slower.
>
> The test is measurement. Compile both ways, read `n_spills`, and time both with the same inputs. Occupancy has nothing to say about which one wins.

The rule of thumb is to leave the compiler alone until a profile points at registers, then cap in steps and watch the spill count. A cap that produces zero spills is free. One that spills a few values in a cold path is usually fine. One that spills inside the hot loop is not.

## Launch overhead and asynchrony

A kernel launch isn't the kernel. The section answers one question: what does the CPU do, and when does it wait?

When Python calls a kernel, the CPU writes a launch command into a queue, called a *stream*, and returns. The GPU reads the queue and runs the kernel later. The call finishes when the work is queued, not when it is done, and that is why the CPU can queue the next layer's kernels while the GPU runs this one. A kernel that fails also reports late, because the error surfaces at a later call.

Queueing costs CPU time. The companion course measured about 5 microseconds to queue a kernel on this lab A100, and it uses 5 to 10 microseconds as a rule of thumb ([chapter 12](https://qwen.ifkash.dev/c/12-your-first-cuda-kernel)). The next table puts that against kernel durations:

| Kernel | Time | Launch as a share |
|---|---|---|
| Elementwise op on $2^{24}$ floats on the A100 | 154 microseconds at 1,304 GB/s, from $192\ \text{MiB} / 1304\ \text{GB/s}$ | 5 / 154 = 3.2% |
| Kernel that runs 2.9 microseconds | 2.9 microseconds | larger than the kernel |

The second row is NVIDIA's V100 measurement from the [CUDA graphs article](https://developer.nvidia.com/blog/cuda-graphs/): a kernel with 2.9 microseconds of execution took 9.6 microseconds per kernel when the host synchronized after every launch, 3.8 microseconds when launches overlapped, and 3.4 microseconds per kernel inside a graph of 20 kernels. Those are V100 numbers, not A100 numbers, and they show the shape: when the kernel is that short, the GPU waits on the CPU. [Chapter 17](/c/17-fusion-and-compilers) shows how CUDA graphs and `torch.compile` take the per-launch cost away.

Asynchrony has one rule that every measurement obeys. A host timer around a launch measures the time to queue it. To time the GPU, record CUDA events on the stream, as `cuda_time` in the labs does, or call `torch.cuda.synchronize()` before you stop the clock. [Chapter 22](/c/22-overlap-and-parallelism) builds on streams and events to overlap copies with compute.

## What goes wrong

- **Reading divergence off the source line.** A condition on `blockIdx.x` is free, and so is one on `threadIdx.x / 32`. One on `threadIdx.x % 32` is not. Work out which threads of one warp disagree, not which threads of the block.
- **Warp-synchronous code without `__syncwarp`.** The reduction that worked for years can return a wrong sum on Volta and later, and only on some runs. The fix is the `__syncwarp()` pair or a `*_sync` shuffle with a mask the whole warp reaches.
- **A shuffle mask that names a lane that doesn't arrive.** The full mask hangs or gives garbage if a lane left early, for example through an early `return` for out-of-range indices. Keep every lane in the warp until the shuffles finish, and mask the values, not the threads.
- **Maximizing occupancy by reflex.** A kernel at 100% theoretical occupancy and a kernel at 25% can run at the same speed, or the 25% one can win. Look for the stall reason before you change a register count.
- **Capping registers into spills.** `-maxrregcount` helps only if the live values still fit. Read the spill counts in the build log, and time the kernel.
- **Trusting theoretical occupancy.** It assumes every SM has resident blocks all the time. A grid of 64 blocks never exceeds 64 SMs, whatever the calculator says.
- **Timing the launch.** The call returns before the GPU starts. Use events, and warm up first.
- **Treating the granularity rules as law.** The 256-register unit and the 1 KiB shared-memory reservation are the retired calculator's rules. When your arithmetic and `cudaOccupancyMaxActiveBlocksPerMultiprocessor` disagree, the API is right.

> [!RECAP]
> - A launch is a grid of blocks. Each block runs on one SM, stays there until it finishes, and runs in an order the hardware doesn't promise.
> - The SM executes warps of 32 threads. Four schedulers issue one warp instruction per cycle each, and lanes that take different branches make the warp run each path in turn, so $k$ equal paths cost $k$ times as much.
> - Volta and later give each thread its own program counter, so lockstep is no longer free: use `__syncwarp()` and `*_sync` primitives with masks the whole warp reaches.
> - Occupancy is resident warps over 64. Blocks per SM is the minimum of the warp, block, register, and shared-memory limits, and a 256-thread block at 255 registers holds 1 block, 12.5%.
> - Occupancy hides latency, but ILP does too, so lower occupancy can be faster. Cap registers with `__launch_bounds__` or `-maxrregcount` only after a profile, and read the spill count.
> - A launch costs about 5 microseconds of CPU time and returns before the GPU finishes: time with events, not a host clock.

## Check your understanding

> [!QUESTION] A kernel launches 100 blocks of 256 threads on an A100. Why can't it use all 108 SMs, and what does the hardware do with the extra capacity?
> There are only 100 blocks, and each block runs on exactly one SM, so at most 100 SMs have work and at least 8 sit idle. The hardware can't split a block across SMs. The fix is more, smaller blocks, or a different division of the work, so that the block count is at least the SM count and preferably a multiple of the number of resident blocks per wave.

> [!QUESTION] The first statement below diverges and the second does not. Why?
> `if (threadIdx.x % 2 == 0) {...}` versus `if ((threadIdx.x / 32) % 2 == 0) {...}`.
>
> The first splits the lanes of every warp: lanes 0, 2, 4, ... take one path and lanes 1, 3, 5, ... take the other, so every warp runs both paths. The second splits whole warps: all 32 lanes of warp 0 agree, all 32 lanes of warp 1 agree, so each warp runs one path. Divergence happens only inside a warp.

> [!QUESTION] A block of 128 threads uses 40 registers per thread and no shared memory. How many blocks and warps are resident on an A100 SM, and what is the limiter?
> A warp is charged $40 \times 32 = 1280$ registers, already a multiple of 256. Each sub-partition holds $16{,}384 / 1{,}280 = 12.8$, so 12 warps, and the SM holds $12 \times 4 = 48$ warps. A block is 4 warps, so the register limit is $48 / 4 = 12$ blocks. The warp limit is $64 / 4 = 16$ blocks and the cap is 32, so registers limit it: 12 blocks, 48 warps, 75% occupancy. The lab's calculator checks this case.

> [!QUESTION] You must reduce 64 values in shared memory with the first warp, and the old code uses no barrier. Why is it wrong on an A100, and what is the minimum fix?
> Since Volta each thread has its own program counter, so the hardware may interleave the lanes instead of running them in lockstep. A lane can overwrite `smem[lane]` before a slower lane has read the old value. Add `__syncwarp()` between the reads and the writes and between the writes and the next reads, or hold the value in a register and use `__shfl_down_sync` with a mask the full warp reaches.

> [!QUESTION] The calculator says a kernel reaches 100% occupancy, and it runs slower than a version at 50%. What are the likely explanations?
> Occupancy is not utilization. The 50% version may have spent the freed registers on a larger tile or on more independent loads per thread, so it keeps as many bytes in flight with fewer warps and re-uses more data. The 100% version may be limited by something else, such as bandwidth, bank conflicts, or divergence. Check stall reasons in the profiler before changing anything.

> [!QUESTION] On paper, how many threads per SM keep 5.9 KB of loads in flight when each thread issues one `float4` load, and why is that only a floor?
> $5{,}925 / 16 = 370.3$, so 371 threads, which is 18% of 2,048. It means that, by Little's law and the 466.3-cycle latency, the loads in flight cover the rating if every thread has its `float4` load outstanding at all times. A real kernel also needs the loads to be coalesced, issued before they're used, and issued continuously. The number is a floor from a model. Your lab in chapter 3 measures what the card does.

> [!QUESTION] Why can't you time a kernel with `time.perf_counter()` around a PyTorch call, and what do you use instead?
> The call returns once the kernel is queued, about 5 microseconds on the lab A100, long before a longer kernel finishes. The timer measures queueing. Record CUDA events on the stream around the call and read the elapsed time after a synchronize, as `cuda_time` does, or synchronize before you stop the clock.

## Lab

> [!TRY] Occupancy and divergence
> Fill in `occupancy`, `max_regs_for_occupancy`, `branch_time_ms`, `divergence_slowdown`, and the one-line `group_of` in the CUDA source.
>
> - **The calculator.** `occupancy(threads, regs, smem)` returns blocks per SM, warps, occupancy, and the limiter. It passes when it reproduces the hand-worked cases in this chapter, then agrees with `cudaOccupancyMaxActiveBlocksPerMultiprocessor` on at least 80% of the configurations that use no shared memory. The lab prints every disagreement, so a missing rule has a place to start.
> - **The divergence cost.** The lab launches a kernel whose lanes take 1, 2, 4, 8, 16, or 32 branches of equal length. The prediction from this chapter is a slowdown equal to the number of branches, and the roofline floor for the uniform launch is $8.59 \times 10^{9}$ FLOP at 19.5 TFLOP/s, 0.44 ms, so about 14 ms at 32 ways. Your lab reports the measured values. It passes if each slowdown is at least half the number of branches, and if choosing the branch by warp instead of by lane costs at most 1.5 times a uniform launch.
>
> The first compile takes 30 to 60 seconds. The lab reports `api_agree_pct`, `slowdown_2way`, `slowdown_32way`, and `by_warp_32way_ratio`.

## Further reading

- [CUDA Programming Guide: programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html): the thread hierarchy, warps, and how divergence is defined.
- [CUDA Programming Guide: advanced kernel programming](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html): independent thread scheduling and why `__syncwarp` exists.
- [CUDA Programming Guide: compute capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html): the occupancy limits table this chapter uses.
- [Better Performance at Lower Occupancy](https://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf), Vasily Volkov, 2010: ILP against TLP, with the GTX 480 and G80 measurements quoted here.
- [Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/), Yuan Lin and Vinod Grover: masks, `__syncwarp`, and porting pre-Volta code.
- [Inside Volta](https://developer.nvidia.com/blog/inside-volta/): an account of independent thread scheduling.
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html): block-size guidance and the occupancy API.
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html): how theoretical and achieved occupancy are reported.
- [Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking](https://arxiv.org/abs/1804.06826), Jia et al.: the method for measuring instruction latency on a real SM.
