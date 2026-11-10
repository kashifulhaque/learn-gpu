---
title: Debugging GPU kernels
slug: 27-debugging-gpu-kernels
part: "Part 1 — The machine"
summary: Find the bugs that a correct-looking answer hides, with compute-sanitizer's four tools, printf and assert in kernels, synchronous launches, the Triton interpreter, and cuda-gdb, and read a sanitizer report down to the thread and the line.
minutes: 120
gpu: true
objectives:
  - "Explain why a GPU bug can return the right answer, fail only sometimes, or fail at a line that isn't the bug, and use CUDA_LAUNCH_BLOCKING to move an asynchronous error to its source."
  - "Pick the compute-sanitizer tool for a symptom: memcheck for out-of-bounds accesses, racecheck for shared-memory hazards, synccheck for barriers, and initcheck for reads of memory nobody wrote."
  - "Read a sanitizer report: infer the tool from the header, find the kernel, the source line, and the thread and block, and turn a thread's coordinates into the element index it was working on."
  - "Find the reads that race in a shared-memory tree reduction, say which barrier ends each hazard, and say why a missed __syncwarp is a bug on every GPU since Volta."
  - "Say why a __syncthreads() inside divergent code is undefined, count how many threads of a tail block reach it, and restructure the kernel so every thread does."
  - "Use printf and assert in a kernel, TRITON_INTERPRET and tl.device_print for Triton, and the basic cuda-gdb commands, and say what each one cannot see."
lab: 27-sanitize
---

# Debugging GPU kernels

> [!TLDR]
> - A GPU bug is often quiet. A kernel that writes 96 bytes past its output can return the right answer, and a race can be wrong one run in ten. A correct output is not evidence that a kernel is correct.
> - Launches are asynchronous, so an error shows up at a later call, on a line that isn't the bug. `CUDA_LAUNCH_BLOCKING=1` runs each launch to completion, and the error then lands on the line that launched the kernel.
> - `compute-sanitizer` has four tools. `memcheck` finds out-of-bounds accesses, `racecheck` finds shared-memory hazards, `synccheck` finds barriers that not every thread reaches, and `initcheck` finds reads of memory that nothing wrote.
> - A report names the kernel, the source line, and the thread and block coordinates, and it ends with an error count that can be larger than the number of findings it printed. A thread's coordinates give the element it worked on: block 3, thread 232 of 256 is element 1000.
> - Use `printf` and `assert` to look inside a kernel, `TRITON_INTERPRET=1` to run a Triton kernel as ordinary Python, and `cuda-gdb` to stop a kernel and inspect one thread.

[Chapter 5](/c/05-profiling) taught you to find out why a kernel is slow. This chapter teaches you to find out why it is wrong, and wrong is harder. A slow kernel shows up on a timeline. A wrong kernel can pass every test you wrote, because the bug lives in a thread that wrote one element too far, or in a warp that read a value one instruction too early.

You will meet five bugs in this chapter, one for each tool: an out-of-bounds write, a race on shared memory, a missed `__syncwarp`, a `__syncthreads` that only part of a block reaches, and a read of shared memory that nothing initialized. Every one of them appears in code from the earlier chapters when someone leaves out a line, and the lab asks you to fix all five. The lab runs your fixed kernels under `compute-sanitizer` and requires zero findings, so a kernel that gets the right answer is not enough to pass.

The lab has a part that needs no GPU. It gives you recorded sanitizer reports and asks you to write the parser that turns each one into a finding: which tool, which kind of error, which kernel, which line, which thread. It also asks for two small pure-Python functions that find the racing reads and the out-of-bounds threads from the index arithmetic alone. On a machine without a GPU or without `compute-sanitizer`, those checks still run, and the two kernel checks skip with a message.

## Before you start

**Asynchronous launches.** A kernel launch returns to the CPU before the GPU finishes, as [chapter 2](/c/02-the-execution-model) explained. The CPU keeps running while the kernel is queued or in flight. The chapter builds on that: it is why an error arrives late.

**Warps and `__syncwarp`.** [Chapter 2](/c/02-the-execution-model) explained independent thread scheduling: since Volta the lanes of a warp are not promised to run in lockstep, so code that relied on that must call `__syncwarp()`. This chapter shows what the sanitizer says when it doesn't.

**Shared memory and `__syncthreads`.** [Chapter 8](/c/08-shared-memory-and-transpose) introduced shared memory, the barrier `__syncthreads()`, and why a block must wait before it reads what its neighbors wrote. [Chapter 6](/c/06-reductions) wrote the tree reduction whose missing barrier is the race below.

**Profiling versus debugging.** [Chapter 5](/c/05-profiling) used Nsight Compute to answer why a kernel is slow. The tools here answer a different question, whether the kernel is correct, and they slow the program down to do it. Run them on small inputs, not on a benchmark.

**The Triton interpreter.** [Chapter 14](/c/14-triton) introduced `TRITON_INTERPRET`. A later section here shows when to reach for it and what it can't see.

## Why a GPU bug stays quiet

This section answers why a kernel with a bug can still pass a test.

A CPU program that writes past the end of an array often crashes with a segmentation fault, because the operating system maps memory in pages and a stray write hits an unmapped one. A GPU gives you less. Device memory is allocated in large blocks, and PyTorch's caching allocator hands out slices of them, so a write 96 bytes past a tensor's end usually lands in a neighboring tensor or in padding. The kernel returns, the output is right, and the damage is somewhere else.

Races are worse. A race depends on the order in which warps happen to run, so the same binary can give the right sum 9 times and the wrong sum on the tenth. A missing barrier between two steps of a tree reduction is invisible when the two steps run in the same warp, and wrong when they don't. The bug is real on every run, and the symptom appears on some.

Three properties make GPU bugs hard to find:

- **Silent corruption.** An out-of-bounds write corrupts memory that something else owns, so the symptom appears later and elsewhere.
- **Nondeterminism.** A race depends on scheduling, so a test can pass while the bug is present.
- **Late errors.** An error that the hardware raises is reported at some later call, as the next section shows.

> [!KEY] A correct output is not evidence of a correct kernel
> A kernel can return the exact right answer while it reads or writes memory it doesn't own, or while it races. Test the answer against a reference, and also run the kernel under a tool that watches the accesses.

The following table lists the five bugs of this chapter. Each has a symptom a test might see, and a tool that finds it without depending on the symptom.

| Bug | Symptom a test may see | Tool |
|---|---|---|
| Write past the end of a buffer | Nothing, or a wrong neighbor | `memcheck` |
| Missing `__syncthreads` between tree steps | A sum that is wrong on some runs | `racecheck` |
| Missing `__syncwarp` in the last warp | A sum that is wrong only on newer GPUs | `racecheck` |
| `__syncthreads` that not every thread reaches | A hang, or wrong data in the tail block | `synccheck` |
| Read of shared memory nobody wrote | Garbage that changes between runs | `initcheck` |

## Make the error appear where it happened

This section answers why the traceback points at the wrong line, and how to fix that.

A kernel launch queues work and returns. If the kernel then faults, say with an illegal address, the hardware records the error, and the runtime reports it at the next call that checks: a `cudaMemcpy`, a `cudaDeviceSynchronize`, a `.item()` in PyTorch, or any other call that waits. The call that reports the error isn't the call that caused it. PyTorch's message for this case says that CUDA errors may be reported asynchronously at another call, so the stack trace can be wrong, and it suggests `CUDA_LAUNCH_BLOCKING=1`.

That variable makes each launch run to completion before the call returns. The CUDA documentation describes it as forcing GPU work to run synchronously from the CPU's point of view. The error then surfaces on the line that launched the faulting kernel, and the traceback is right.

```bash
CUDA_LAUNCH_BLOCKING=1 python train.py
```

Two cautions apply. A synchronous launch changes timing, so a race that depends on overlapping kernels can disappear, and a benchmark run this way measures the wrong thing. And some errors are *sticky*: after a kernel faults with an illegal address, the context is unusable, and every later call fails too. Read the first error in the log, not the last.

In a plain CUDA program the same idea is two calls. Check `cudaGetLastError()` right after the launch to catch a bad configuration, such as too many threads per block, and call `cudaDeviceSynchronize()` and check its return value to catch a fault in the kernel itself. The lab's host functions check the first.

```cuda
scale_shift_kernel<<<grid, block>>>(x, y, n);
cudaError_t launch = cudaGetLastError();        // bad launch configuration
cudaError_t run = cudaDeviceSynchronize();      // a fault inside the kernel
```

> [!TIP] Turn it on first
> When a PyTorch job dies with an unexplained CUDA error, rerun the failing step with `CUDA_LAUNCH_BLOCKING=1` before you read any code. The error often names the kernel, and you learn the line without a tool.

## `printf` and `assert` inside a kernel

This section answers how to look at values in a kernel that you can't step through.

A kernel can call `printf`, and each thread that executes the call produces a line. That is a lot of lines. Print from one thread, one block, or one condition, and print the coordinates with the value, or you can't tell the lines apart:

```cuda
if (blockIdx.x == 3 && threadIdx.x == 232) {
    printf("block %d thread %d: i=%d n=%d\n", blockIdx.x, threadIdx.x, i, n);
}
```

The output isn't immediate. The device writes into a fixed-size buffer that the runtime copies to the host at synchronization points, so a line from a kernel that is still running isn't visible yet, and a kernel that prints a great deal can fill the buffer. The programming guide documents the buffer, and `cudaDeviceSetLimit` with `cudaLimitPrintfFifoSize` raises its size. Call `cudaDeviceSynchronize()` before you read the output, or run with `CUDA_LAUNCH_BLOCKING=1`.

An `assert` checks a condition inside the kernel. The CUDA programming guide says a failing assert stops the kernel, and each failing thread then prints a message with the file, line, block, thread, and the expression. It also says the assert corrupts the CUDA context, so every later call fails with `cudaErrorAssert`. That's why a PyTorch process that trips a device-side assert reports the same error from everything it does afterward, and why you must restart it.

```cuda
assert(i < n);   // stops the kernel when any thread has i >= n
```

An assert costs instructions, and the programming guide notes that you disable them by defining `NDEBUG`, which removes the check entirely. Keep the expression free of side effects, because a build that defines `NDEBUG` would otherwise behave differently. Use `printf` to look and `assert` to state an invariant. When a bug involves many threads, neither beats a tool that watches every access, which is the next section.

## compute-sanitizer: four tools, one command

This section answers how to run the sanitizer, and what each of its four tools watches.

`compute-sanitizer` ships with the CUDA toolkit, so the course's devel image already has it. It runs your program as a child process and watches every kernel it launches. It replaced `cuda-memcheck`, which CUDA 12.0 removed. You choose a tool with `--tool`:

```bash
compute-sanitizer --tool memcheck python run.py       # the default
compute-sanitizer --tool racecheck python run.py
compute-sanitizer --tool synccheck python run.py
compute-sanitizer --tool initcheck python run.py
```

The four tools answer four different questions, and each finds bugs the others can't:

| Tool | Watches | Reports | Option you will need |
|---|---|---|---|
| `memcheck` | Every load and store | An access outside an allocation, a misaligned access, and CUDA API failures | `--report-api-errors no` mutes failures from library setup |
| `racecheck` | Shared memory only | Write-after-write, write-after-read, and read-after-write hazards between threads | `--racecheck-report all` prints hazards and a summary |
| `synccheck` | Barriers | `__syncthreads`, `__syncwarp`, and Cooperative Groups barriers used with divergent threads | none |
| `initcheck` | Reads | Reads of device global memory that nothing wrote, or of shared memory with the option | `--initcheck-address-space shared` |

The options come from the [Compute Sanitizer documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html). It also documents `--error-exitcode`, which makes the tool return a chosen exit code when the program succeeds but the tool found errors. That's the option for a test suite, and the lab uses it.

Four habits make a report readable.

**Compile with line information.** Add `-lineinfo` to `nvcc`. Without it a report shows an offset into the kernel, `+0x90`, and no source line. The lab compiles with it. Don't use `-G` for the sanitizer: it turns off optimizations and changes the code you are checking.

**Make allocations exact.** PyTorch's caching allocator hands out slices of large `cudaMalloc` blocks, and `memcheck` can see only an access that leaves the whole block. A write 96 bytes past a tensor's end then lands inside the pool and goes unreported. Set `PYTORCH_NO_CUDA_MEMORY_CACHING=1` and every tensor becomes its own `cudaMalloc` of exactly its size, so the tool sees the boundary. The lab sets this variable for you.

**Keep the program small.** The tool instruments every access, and it slows the program, sometimes by a large factor. Use a few thousand elements, the smallest shape that shows the bug, and a handful of blocks.

**Run one tool at a time.** The tools instrument the program differently and can't share a run. A bug that `memcheck` misses is not proof of a clean kernel; run all four.

> [!WARNING] The recorded reports in this chapter are representative
> The reports below follow the format in NVIDIA's documentation. They were written for the lab, trimmed to the first findings, and not captured from a run on the lab's A100, so treat the addresses and byte offsets as examples. The tool, the kind, the kernel, the line, and the thread and block coordinates are what you read. Your own runs print the real values.

## memcheck: the write that looks fine

This section answers how a kernel that returns the right answer gets caught, and how to read the report.

The first bug is the one from every first kernel. You launch enough blocks to cover $n$ elements, and the last block has threads past the end. The number of blocks is a ceiling division:

$$
\text{grid} = \left\lceil \frac{n}{B} \right\rceil
$$

where $n$ is the element count and $B$ is the block size. Take $n = 1000$ and $B = 256$. The grid is $\lceil 1000/256 \rceil = 4$ blocks, so $4 \times 256 = 1024$ threads run, and the 24 threads with indices 1000 to 1023 have no element.

An unguarded kernel lets those 24 threads read and write:

```cuda
__global__ void scale_shift_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    y[i] = 2.0f * x[i] + 1.0f;          // no guard: threads 1000..1023 run this too
}
```

The output has 1000 floats, 4000 bytes. Each of the 24 threads reads 4 bytes past the end of `x` and writes 4 bytes past the end of `y`, which is 96 bytes in all on each tensor. The first 1000 outputs are right, and a test that compares them passes. Here is what `memcheck` prints for this kernel, trimmed:

```text
========= COMPUTE-SANITIZER
========= Invalid __global__ read of size 4 bytes
=========     at scale_shift_kernel(const float *, float *, int)+0x70 in /tmp/lab27/oob.cu:11
=========     by thread (232,0,0) in block (3,0,0)
=========     Address 0x7f3a5ce00fa0 is out of bounds
=========     and is 0 bytes after the nearest allocation at 0x7f3a5ce00000 of size 4,000 bytes
=========
========= Invalid __global__ write of size 4 bytes
=========     at scale_shift_kernel(const float *, float *, int)+0x90 in /tmp/lab27/oob.cu:11
=========     by thread (232,0,0) in block (3,0,0)
...
========= ERROR SUMMARY: 48 errors
```

Read it from the top. The first line of a finding says what happened: an invalid read, of 4 bytes, in global memory. The `at` line gives the kernel and the source line, `oob.cu:11`. The `by thread` line gives the thread, $(232, 0, 0)$, and the block, $(3, 0, 0)$. The last two lines say how far out of bounds: this address is 0 bytes after the nearest allocation, which is 4,000 bytes, so it is the first element past the end.

The coordinates give you the index. The thread's global index is $\text{blockIdx} \times \text{blockDim} + \text{threadIdx}$, and for this thread that is $3 \times 256 + 232 = 1000$. The first thread past the end of a 1000-element array is thread 1000, as the arithmetic predicted, and it confirms the bug is a missing guard, not a wrong offset.

Two details in the report matter. The first finding is the *read* of `x[i]`, not the write, because the kernel reads before it writes. And the summary says 48 errors although the report shows two findings: 24 threads each make one invalid read and one invalid write, and the tool prints only the first few. The number at the bottom counts failing accesses, not blocks of text.

The fix guards both accesses:

```cuda
if (i < n) {
    y[i] = 2.0f * x[i] + 1.0f;
}
```

> [!INTUITION] Read the error count last
> The findings tell you what is wrong, and the count tells you how widespread it is. A count of 48 with two findings means the same mistake repeats for many threads. A count that matches the number of findings means each is a separate access.

## racecheck: two threads, one address, no barrier

This section answers which reads race in a shared-memory reduction, and how `racecheck` reports them.

A *hazard* is two accesses to the same address by different threads, at least one of them a write, with nothing that orders them. There are three kinds: read-after-write (RAW), write-after-read (WAR), and write-after-write (WAW). `racecheck` watches shared memory, tracks which threads touched each address, and reports a pair that no barrier separates.

The standard victim is the tree reduction from [chapter 6](/c/06-reductions). A block of $B$ threads loads $B$ values into shared memory `s`, then halves the active threads each step:

```cuda
s[t] = x[blockIdx.x * 256 + t];
__syncthreads();
for (int d = 128; d > 0; d >>= 1) {
    if (t < d) {
        s[t] += s[t + d];
    }
    // __syncthreads() is missing here
}
```

Step $k$ uses the stride $d = B \gg k$. Thread $t < d$ reads `s[t]` and `s[t + d]`, and writes `s[t]`. Which of those reads depends on another thread's write? The address $t + d$ is at least $d$, and the previous step's stride was $2d$, so thread $t + d < 2d$ was active and wrote `s[t + d]`. Each active thread reads one value that a different thread wrote one step earlier. Without a barrier, nothing guarantees that write has landed.

Count them for $B = 8$ with no barrier at all. The load phase has every thread write its own cell. Step 1, with $d = 4$, has threads 0 to 3 read cells 4 to 7, written in the load phase by threads 4 to 7. That is 4 hazards. Step 2, with $d = 2$, has threads 0 and 1 read cells 2 and 3, written in step 1 by threads 2 and 3: 2 hazards. Step 3 has thread 0 read cell 1, written in step 2 by thread 1: 1 hazard. The total is $4 + 2 + 1 = 7$.

In general, step $k$ contributes $B/2^{k}$ hazards, and the sum over the load-to-step-1 boundary and every later boundary is

$$
\sum_{k=1}^{\log_2 B} \frac{B}{2^{k}} = \boxed{B - 1}
$$

so a block of 8 has 7 pairs, a block of 16 has 15, and a block of 256 has 255. A barrier between two phases removes exactly the pairs that cross it. One barrier after the load removes the first 4 and leaves 3, and a barrier after every step leaves none. The following figure computes the pairs with a trace of the algorithm, and lets you place the barriers.

```viz
27-race-trace
```

The figure shows two models of a warp. In the *lockstep* model, which describes GPUs before Volta, the lanes of a warp can't get out of step, so a hazard between two threads of one warp can't happen. Choose it for a block of 8 and the figure shows no hazard at all, which is why the bug passed on old hardware. In the *independent-lane* model, which is what Volta and every later architecture promise, the lanes of a warp are ordered only by `__syncwarp()` or a barrier. [Chapter 2](/c/02-the-execution-model) explains why. Choose a block of 64, no barriers, and lockstep, and 32 hazards between warp 0 and warp 1 remain, because no model orders two different warps without a barrier.

Here is the report for the missing-barrier kernel, trimmed to two hazards:

```text
========= Error: Potential RAW hazard detected at __shared__ 0x100 in block (0,0,0) :
=========     Write Thread (64,0,0) at 0x1f0 in /tmp/lab27/race.cu:17:block_sum_kernel(const float *, float *)
=========     Read Thread (0,0,0) at 0x1d0 in /tmp/lab27/race.cu:17:block_sum_kernel(const float *, float *)
=========     Current Value : 0, Incoming Value : 3
=========
========= RACECHECK SUMMARY: 2 hazards displayed (2 errors, 0 warnings)
========= ERROR SUMMARY: 2 errors
```

The header names the hazard kind, RAW, the shared-memory address, `0x100`, and the block. Then come the two accesses, each with a thread, an instruction offset, a file and line, and the kernel. Here the write is by thread 64 and the read by thread 0, and both are on line 17, because `s[t] += s[t + d]` is a single statement that reads and writes. Address `0x100` is byte 256, which is the float `s[64]`: thread 64 wrote it in step 1, and thread 0 reads it in step 2, as the arithmetic said.

The fix is the line the kernel left out:

```cuda
for (int d = 128; d > 0; d >>= 1) {
    if (t < d) {
        s[t] += s[t + d];
    }
    __syncthreads();      // every thread must reach this, so it sits outside the if
}
```

Note where the barrier goes. It sits outside the `if (t < d)`, so every thread of the block reaches it each iteration. A barrier inside the `if` is the bug of the synccheck section below.

## The missed `__syncwarp`

This section answers why the last warp of a reduction is still a race, and what `racecheck` prints for it.

The final steps of a reduction run in one warp, and a famous optimization drops the barriers there. When $t < 32$, the last six steps read and write cells that only the first warp touches, so older code wrote them without a barrier, with `volatile` shared memory:

```cuda
if (t < 32) {
    volatile float* v = s;
    v[t] += v[t + 32];
    v[t] += v[t + 16];
    v[t] += v[t + 8];
    v[t] += v[t + 4];
    v[t] += v[t + 2];
    v[t] += v[t + 1];
}
```

That code was correct on a GPU that kept the 32 lanes of a warp in lockstep. It isn't promised to work on Volta and later, for two reasons that [chapter 2](/c/02-the-execution-model) explained. A lane can run ahead of another between two statements, so `v[t + 16]` can be read before lane $t + 16$ wrote its partial sum. And a single statement `v[t] += v[t + 16]` is a load, a load, an add, and a store, so lane $t + 16$ can store its result to `v[t + 16]` before lane $t$ loads it. The first is a RAW hazard and the second is a WAR hazard, both inside one warp.

The sanitizer reports these at *warning* level, and that has a consequence worth knowing. The error count stays at zero, because a warning isn't an error. Here is the shape of the report:

```text
========= Warning: (Warp Level Programming) Potential RAW hazard detected at __shared__ 0x40 in block (0,0,0) :
=========     Write Thread (16,0,0) at 0x2a0 in /tmp/lab27/warp.cu:21:warp_block_sum_kernel(const float *, float *)
=========     Read Thread (0,0,0) at 0x2c0 in /tmp/lab27/warp.cu:22:warp_block_sum_kernel(const float *, float *)
=========     Current Value : 0, Incoming Value : 7
=========
========= RACECHECK SUMMARY: 1 hazard displayed (0 errors, 1 warning)
========= ERROR SUMMARY: 0 errors
```

The summary says 0 errors, and the kernel is racy. A test that reads only the last line calls this report clean. The lab's `is_clean` function looks at the findings, and its checks fail a parser that trusts the summary alone.

The fix is the one from chapter 2. Each lane keeps its running value in a register, so a statement has no load and store to interleave. `__syncwarp()` before the store makes every lane finish its read first, and `__syncwarp()` after it makes every write visible before the next read:

```cuda
if (t < 32) {
    float v = s[t] + s[t + 32];
    for (int d = 16; d > 0; d >>= 1) {
        __syncwarp();            // all lanes have read
        s[t] = v;
        __syncwarp();            // all lanes have written
        v += s[t + d];
    }
}
```

A warp shuffle, `__shfl_down_sync`, avoids shared memory in the last warp, and it passes `racecheck` because there is no shared memory to race on. [Chapter 6](/c/06-reductions) builds that version. The lab accepts either fix.

## synccheck and the deadlocked barrier

This section answers what goes wrong when a `__syncthreads()` sits in code that only some threads reach.

`__syncthreads()` is a barrier for the whole block: no thread continues until every thread of the block has arrived. The CUDA programming guide allows it in conditional code only when the condition evaluates the same way across the entire block. Otherwise, it says, execution can hang or have unintended side effects, and its examples label a thread-dependent condition as undefined behavior.

The bug appears most often at the end of an array, where the last block is partly empty. A thread whose element doesn't exist returns early, before the barrier:

```cuda
template <int B>
__global__ void tail_pair_sum_kernel(const float* x, float* out, int n) {
    __shared__ float tile[B];
    int t = threadIdx.x;
    int i = blockIdx.x * B + t;
    if (i >= n) {
        return;                         // threads past the end leave here ...
    }
    tile[t] = x[i];
    __syncthreads();                    // ... so only part of the last block arrives
    out[i] = tile[t] + ((t + 1 < B) ? tile[t + 1] : 0.0f);
}
```

Take $n = 1000$ and $B = 128$. The grid is $\lceil 1000/128 \rceil = 8$ blocks. Block 7 starts at element $7 \times 128 = 896$, so it holds elements 896 to 999, which is $1000 - 896 = 104$ valid elements. Threads 104 to 127 return early, and only 104 of the block's 128 threads reach the barrier. Blocks 0 to 6 are full and fine.

Look at the warps of block 7. Warp 3 holds threads 96 to 127, which are elements 992 to 1023. Elements 992 to 999 exist, so 8 of its 32 lanes reach the barrier and 24 don't. That is a warp whose lanes diverge at the barrier, and it's the case `synccheck` reports:

```text
========= Barrier error detected. Divergent thread(s) in warp
=========     at void tail_pair_sum_kernel<128>(const float *, float *, int)+0x110 in /tmp/lab27/sync.cu:14
=========     by thread (96,0,0) in block (7,0,0)
=========
========= ERROR SUMMARY: 1 error
```

The kernel name carries its template argument, and a parser that wants the function's name has to cut at the `<`.

What happens when you run this without the tool depends on the hardware and the compiler, and the guide gives no promise. The barrier can hang the kernel, and a hung kernel can leave the GPU unresponsive until a watchdog resets it. It can also appear to work, because on current hardware a thread that has exited may not hold up the barrier. Even then, the last valid thread reads `tile[t + 1]`, a cell that the exited thread never wrote. The kernel "works" and reads an uninitialized cell, and `synccheck` calls it a bug in both cases.

The fix is to make every thread reach the barrier, and to guard only the memory accesses. A thread past the end loads zero, waits at the barrier, and doesn't store:

```cuda
tile[t] = (i < n) ? x[i] : 0.0f;
__syncthreads();
if (i < n) {
    out[i] = tile[t] + ((t + 1 < B) ? tile[t + 1] : 0.0f);
}
```

This is the rule to carry out of the section: **put the guard around the work, and keep the barrier outside the guard.** A `return` before a barrier is the same bug as an `if` around it. [Chapter 8](/c/08-shared-memory-and-transpose) states the rule for a barrier in a transpose tile.

> [!WARNING] A hang is not always a hang you can see
> A kernel stuck at a barrier holds its SM, and the process can look idle while the GPU is busy. If a kernel never returns, run it under `synccheck` on a tiny input first. A tool that reports one divergent warp is faster than waiting for a timeout.

## initcheck: reading what nobody wrote

This section answers how to find a read of memory that no thread initialized.

Shared memory starts with whatever the previous block on that SM left in it. A kernel that reads a cell it never wrote gets garbage, and the garbage can be zero on one run, so the bug hides. `initcheck` tracks which cells have been written and reports a read of one that hasn't. By default it watches global memory, and with `--initcheck-address-space shared` it watches shared memory, which is what the lab uses.

The fifth kernel is a three-point stencil, $\text{out}_i = x_{i-1} + x_i + x_{i+1}$, with zero outside the array. Each block loads a tile with one halo cell on each side, so the tile has $B + 2$ cells and thread $t$ reads the cells $t$, $t+1$, and $t+2$. The buggy kernel loads only the middle:

```cuda
__shared__ float tile[B + 2];
tile[t + 1] = (i < n) ? x[i] : 0.0f;      // fills tile[1] .. tile[B]
__syncthreads();
out[i] = tile[t] + tile[t + 1] + tile[t + 2];
```

Thread 0 reads `tile[0]`, and thread $B - 1$ reads `tile[B + 1]`. Neither cell is ever written. With $B = 128$ the tile has 130 cells, the two bad cells are `tile[0]` and `tile[129]`, and every one of the 8 blocks reads both, so there are 16 bad reads:

```text
========= Uninitialized __shared__ memory read of size 4 bytes
=========     at void stencil3_kernel<128>(const float *, float *, int)+0x1c0 in /tmp/lab27/init.cu:15
=========     by thread (0,0,0) in block (0,0,0)
=========     Address 0x0
...
========= ERROR SUMMARY: 16 errors
```

The thread, $(0,0,0)$, and the address, `0x0`, point at `tile[0]`. A second finding names thread 127 and address `0x204`, which is byte 516, the float `tile[129]`. The error count, 16, is the 8 blocks times 2 halo reads. The coordinates alone told you which cells; the count told you the same mistake repeats in every block.

The fix loads the halos, each with a bounds check, and keeps every thread at the barrier:

```cuda
tile[t + 1] = (i < n) ? x[i] : 0.0f;
if (t == 0)     { tile[0]     = (i > 0)     ? x[i - 1] : 0.0f; }
if (t == B - 1) { tile[B + 1] = (i + 1 < n) ? x[i + 1] : 0.0f; }
__syncthreads();
```

Note that the halo load for thread $B - 1$ in the last block is allowed to see $i + 1 \ge n$, and gets zero. A halo that reads `x[i + 1]` with no check would be the first bug again, and `memcheck` would find it.

## Reading a report, step by step

This section answers what to do with a report once you have it.

The five reports share a shape, and a fixed routine turns any of them into a diagnosis. The lab's parser does the first four steps in code.

1. **Infer the tool from the first line.** A report never names its tool. `Invalid __global__ ...` is `memcheck`. `Uninitialized __shared__ ...` is `initcheck`. `Barrier error detected` is `synccheck`. `Potential RAW hazard` is `racecheck`.
2. **Find the kernel and the line.** The `at` line, or a racecheck access line, gives both. Cut the name at the first `(` or `<`, and drop a leading `void`.
3. **Find the thread and the block.** For a race, read both threads: the write and the read, or the two writes.
4. **Compare the finding count with the error count.** The last line counts failing accesses, and the tool prints only the first few.
5. **Turn the coordinates into an element.** Compute $\text{blockIdx} \times \text{blockDim} + \text{threadIdx}$ and ask what that thread was working on. For the out-of-bounds report it was element 1000 of 1000.
6. **Ask which line the thread executed.** The line plus the thread usually identifies the missing guard, barrier, or load.

The table maps the tool and the kind to the bug and the fix. The lab's `classify` function is this table.

| Tool | Header says | Bug | Fix |
|---|---|---|---|
| `memcheck` | Invalid read or write, N bytes after the allocation | Missing bounds guard, or a wrong index | Guard with `i < n`, or correct the index |
| `racecheck` | Potential RAW, WAR, or WAW hazard, error | Missing `__syncthreads` between phases | Add the barrier, outside any thread-dependent branch |
| `racecheck` | The same, with Warp Level Programming and a warning | Missing `__syncwarp` | `__syncwarp()` around the read and the write, or a shuffle |
| `synccheck` | Divergent thread(s) in warp or block | A barrier that some threads skip | Move the guard to the memory access, keep the barrier outside |
| `initcheck` | Uninitialized shared memory read | A cell that no thread wrote | Write it, or don't read it |

> [!EXAMPLE] From thread coordinates to the bug
> The memcheck report names block 3, thread 232 of 256 threads. That is element $3 \times 256 + 232 = 1000$. The allocation is 4,000 bytes, which is 1000 floats, and the access is 0 bytes after it. So the thread is the first past the end, and the missing guard is `if (i < n)`. A report that said the access was 8 bytes after the allocation would be element 1002 and still the same bug.

Two traps are worth knowing before you start. A finding in a library kernel, such as a `cublas` or `at::native` function, is usually a symptom: you passed it a bad pointer or a wrong size. Look at your call, not at the library. And a report from `memcheck` is not a report from `racecheck`: a clean `memcheck` run says nothing about races.

## Triton: `TRITON_INTERPRET`, `device_print`, and `device_assert`

This section answers how to debug a Triton kernel, where you don't write the threads.

[Chapter 14](/c/14-triton) introduced `TRITON_INTERPRET=1`. It makes Triton run each kernel on the CPU, in Python, one program at a time, so you can use `print`, a debugger, and ordinary exceptions. The [Triton debugging guide](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html) states what it does: it simulates the kernel with NumPy equivalents instead of compiling it.

Use it for the bugs a Triton kernel has, which are index and mask bugs: an offset that is off by a stride, a `mask` that forgets the tail, a pointer that points at the wrong row. Run a tiny shape, a few programs, and print the offsets:

```bash
TRITON_INTERPRET=1 python test_kernel.py
```

Inside a kernel, `tl.device_print("offs", offs)` prints at run time on the GPU, and `tl.static_print` prints at compile time, which is where you check a `tl.constexpr`. `tl.device_assert(cond)` checks a condition on the GPU, but the guide says it runs only when `TRITON_DEBUG` is set to 1. Set that variable when you want the check, and expect the kernel to be slower while it is on.

The interpreter is a different program from the GPU kernel, and the guide lists two limits: it doesn't support `bfloat16` operations, so you cast to `float32` with `tl.cast`, and it doesn't support indirect memory access, where you load a pointer and then load through it. It also has no shared memory, no layouts, and no parallelism, so it can't reproduce a race, a bank conflict, or a spill. A kernel that is right in the interpreter and wrong on the GPU points at something device-specific.

A compiled Triton kernel is still a CUDA kernel, so `compute-sanitizer` can watch the process. The report's kernel name is the generated function, and its line numbers refer to generated code unless Triton keeps debug information, so expect the kernel and the thread coordinates to help more than the line. A Triton kernel with a missing mask, which reads past the end of a row, shows up in `memcheck` exactly like the first kernel in this chapter.

> [!TIP] Compare before you instrument
> Before any tool, run the Triton kernel against a float32 PyTorch reference on several shapes, as chapter 14 recommends. Wrong only when $K$ isn't a multiple of the block means the mask. Wrong only on the GPU means the interpreter can't help, and `compute-sanitizer` and `device_print` can.

## cuda-gdb basics

This section answers when you stop a kernel and look, and which commands to start with. It is prose only: the lab doesn't run a debugger.

`cuda-gdb` is `gdb` for the GPU. It stops a kernel at a breakpoint and lets you inspect one thread's variables, and it is the tool to reach for when the sanitizer says where the bug is and you want to see why. The [CUDA-GDB documentation](https://docs.nvidia.com/cuda/cuda-gdb/index.html) describes it. It needs a debug build, which is slow, so use it on a small input.

Compile with `-g -G`: `-g` adds host debug information, and `-G` adds device debug information and turns off device optimizations. Because `-G` changes the code, it can hide a race that depends on instruction timing, and it can change a register count. A race that `racecheck` found in the optimized build may not reproduce under `-G`.

```bash
nvcc -g -G -o prog prog.cu
cuda-gdb ./prog
```

Inside the debugger the commands are `gdb`'s, with a few for the GPU's hierarchy:

- `break scale_shift_kernel` or `break oob.cu:11` sets a breakpoint in device code. A breakpoint in a kernel stays pending until the kernel's code loads.
- `break oob.cu:11 if threadIdx.x == 232` is a conditional breakpoint, which is how you stop in one thread of thousands.
- `cuda block 3 thread 232` moves the focus to that thread, and `cuda thread (15)` changes only the thread. `info cuda threads` lists the active threads. After a switch, `print i` and `print n` show that thread's values.
- `set cuda break_on_launch application` stops at the first instruction of each kernel you launch.
- Watchpoints on device code aren't supported, so you can't ask the debugger to stop when a cell changes.

Two more tools connect it to the sanitizer. Setting `CUDA_DEVICE_WAITS_ON_EXCEPTION=1` makes the program run normally until a device exception, and then wait so that you can attach `cuda-gdb`, which lets you stop at a fault in a program that you didn't start under the debugger. And `compute-sanitizer --generate-coredump yes` writes a GPU core dump on an error, which you can open in `cuda-gdb`. Reach for the sanitizer first: it finds the thread and the line without a debug build, and the debugger is for the question the report leaves open.

## A debugging order

This section answers which tool to use first, because the cheapest one that finds the bug is the right one.

Work from the cheapest tool to the most expensive, and stop when the bug has a name.

1. **Reproduce small.** Find the smallest input that shows the problem, and check it against a reference in float32 on several shapes, including ones that aren't multiples of the block size. A wrong tail points at a guard.
2. **Make errors synchronous.** Run with `CUDA_LAUNCH_BLOCKING=1`. If the error moves to a different line, you found the real launch.
3. **Run `memcheck`.** An out-of-bounds access corrupts everything else, so rule it out before you chase a race.
4. **Run `racecheck`, then `synccheck`, then `initcheck`.** Each answers a question the others can't, and each takes a few minutes on a small input.
5. **Add `printf` for one thread.** Use the coordinates the report gave you.
6. **Use `cuda-gdb` for the question left over.** Stop in the one thread, and read its values.

Fix the first finding and rerun, because later findings are often consequences. An out-of-bounds write can make a later read look uninitialized, and a barrier fix often removes a dozen hazards at once.

## What goes wrong

- **Trusting a correct answer.** The out-of-bounds kernel returns the right 1000 values. Test the answer and run the tools.
- **Reading the error count as the finding count.** The OOB report prints 2 findings and says 48 errors. A parser that counts findings reports the wrong number.
- **Trusting a summary of 0 errors.** A warp-level hazard is a warning, and the summary says 0 errors. Read the findings, not just the last line.
- **Running `memcheck` through the caching allocator.** A write that lands inside PyTorch's pool is invisible. Set `PYTORCH_NO_CUDA_MEMORY_CACHING=1`.
- **Putting `__syncthreads()` inside the guard.** A barrier in `if (i < n)`, or after an early `return`, is reached by part of the block. Guard the accesses and leave the barrier outside.
- **Fixing a race with `volatile`.** `volatile` forces a memory access, and it doesn't order two lanes. Use `__syncwarp()` or a shuffle.
- **Profiling or benchmarking under the sanitizer.** The tool changes the timing by a large factor, and `CUDA_LAUNCH_BLOCKING=1` serializes launches. Measure with neither.
- **Checking only the optimized build with `cuda-gdb`.** `-G` changes the code. A bug that vanishes under `-G` may be a race, not a fix.
- **Debugging in a process after a device assert.** The context is corrupted and every later call fails. Restart the process.
- **Expecting the Triton interpreter to find a race.** It runs programs one at a time on the CPU, so there is nothing to race.

> [!RECAP]
> - A GPU bug can return the right answer. Test against a reference, and run the kernel under a tool that watches the accesses.
> - Errors arrive late. `CUDA_LAUNCH_BLOCKING=1` makes them arrive at the launch that caused them.
> - `memcheck`, `racecheck`, `synccheck`, and `initcheck` answer four different questions. A clean run of one says nothing about the others.
> - A report names the tool by its header, the kernel and line by its `at` line, and the thread and block by their coordinates. The error count at the end can be larger than the number of findings printed, and a warning doesn't count in it.
> - A tree reduction of $B$ threads has $B - 1$ racing reads without a barrier. A barrier between two phases removes the reads that cross it, and `__syncwarp()` does the same inside a warp. Put every barrier where all threads reach it.

## Check your understanding

> [!QUESTION] A kernel with 1000 elements and 256 threads per block returns the right answer. Why can `memcheck` still report it, and how many threads are out of bounds?
> The grid is $\lceil 1000/256 \rceil = 4$ blocks, which is 1024 threads, so 24 threads have indices 1000 to 1023 and no element. Each reads 4 bytes past the input and writes 4 bytes past the output. The first 1000 results are right, so a comparison passes, but the accesses leave the allocations, and `memcheck` sees every one. The summary counts 48 errors, 24 reads and 24 writes.

> [!QUESTION] A memcheck finding says block 5, thread 17, with 128 threads per block. Which element was the thread working on?
> $5 \times 128 + 17 = 657$. The coordinates give the element when the kernel maps one element to one thread. If the kernel gives each thread several elements, multiply by the elements per thread and look at the loop variable too.

> [!QUESTION] An 8-thread tree reduction has no barrier. How many read-after-write hazards are there, and how many remain if you add one barrier after the load?
> There are $8 - 1 = 7$: 4 in step 1, 2 in step 2, and 1 in step 3. A barrier after the load removes the 4 between the load and step 1 and leaves 3. Barriers after every step leave none. A barrier only helps for the pairs that cross it.

> [!QUESTION] The last-warp reduction with `volatile` and no `__syncwarp` passed on a Pascal GPU and fails `racecheck` on an A100. Is the A100 broken?
> No. Pascal kept the lanes of a warp in lockstep, so the lanes could not get out of step, and the code was correct there by the hardware's design. Volta and later give each thread its own program counter, so the lanes may interleave, and the code now races. The A100 is exposing a bug that the older hardware hid. The fix is `__syncwarp()` around the read and the write.

> [!QUESTION] The summary line says `ERROR SUMMARY: 0 errors`, and `racecheck` printed a warning about a hazard. Is the kernel clean?
> No. The sanitizer reports a warp-level hazard at warning level, and a warning isn't counted as an error. The kernel has a race. A test that decides by the summary alone would pass it, so decide by the findings as well.

> [!QUESTION] A kernel with 1000 elements and 128 threads per block returns early for `i >= n` before a `__syncthreads()`. How many threads of block 7 reach the barrier, and which warp is divergent?
> Block 7 starts at element 896 and holds $1000 - 896 = 104$ elements, so 104 threads arrive and 24 leave. Warp 3 holds threads 96 to 127, which are elements 992 to 1023. Elements 992 to 999 exist, so 8 lanes arrive and 24 don't. `synccheck` reports that warp as divergent.

> [!QUESTION] A Triton kernel is right under `TRITON_INTERPRET=1` and wrong on the GPU for large inputs. What can the interpreter not tell you, and what do you try?
> The interpreter runs programs one at a time on the CPU with NumPy, so it can't show a race, a layout problem, a register spill, or a dtype that the hardware rounds differently, and it doesn't support `bfloat16` or indirect loads. For a large input, a 32-bit offset can also overflow on the GPU. Run `compute-sanitizer --tool memcheck` on the process and compare against a float32 reference on several shapes.

## Lab

> [!TRY] Sanitize five buggy kernels
> Open `starter.py`. It has five CUDA kernels that each contain one bug: an out-of-bounds write, a shared-memory race, a missed `__syncwarp`, a `__syncthreads` that part of a block skips, and an uninitialized shared read. Fix each, then write the report tools. The report tools are `error_count`, `parse_report`, `classify`, and `is_clean`, graded against recorded reports in `fixture_reports.py`. Add `min_grid`, `oob_indices`, `barrier_arrivals`, and `shared_hazards`.
>
> You pass the CPU checks when `error_count` returns 48 for the out-of-bounds report and `None` when there is no summary, `parse_report` returns the kernel `scale_shift_kernel`, line 11, thread $(232,0,0)$ and block $(3,0,0)$ for the first finding, a templated kernel is cut at its `<`, the warp-level report is a warning that `is_clean` rejects, and `shared_hazards(8, False)` returns 7 hazards while a barrier after the load leaves 3.
>
> You pass the GPU checks when each fixed kernel gives the exact right answer and runs clean under its own tool: `memcheck`, `racecheck` twice, `synccheck`, and `initcheck`. A machine without CUDA, `nvcc`, or `compute-sanitizer` skips those two checks with a message, and the seven CPU checks still run. Allow a few minutes the first time, because five modules compile.

The slide deck is [Debugging GPU kernels](/slides/27-debugging-gpu-kernels).

## Further reading

- [Compute Sanitizer documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html): the four tools, the report formats, `--tool`, `--error-exitcode`, `--racecheck-report`, and `--initcheck-address-space`. The recorded reports in this chapter follow its formats.
- [CUDA-GDB documentation](https://docs.nvidia.com/cuda/cuda-gdb/index.html): `-g -G`, device breakpoints, the `cuda` focus commands, `CUDA_DEVICE_WAITS_ON_EXCEPTION`, and the note that Compute Sanitizer replaced `cuda-memcheck`.
- [CUDA programming guide, C++ language extensions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html): `__syncthreads()` in conditional code, `__syncwarp()`, and `assert`, including the corrupted context after a failure.
- [CUDA programming guide, environment variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html): `CUDA_LAUNCH_BLOCKING`.
- [Using CUDA warp-level primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/): Yuan Lin and Vinod Grover, 2018, on why lockstep code needs `__syncwarp` after Volta.
- [Triton debugging guide](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html): `TRITON_INTERPRET`, `device_print`, `static_print`, `device_assert` with `TRITON_DEBUG`, and the interpreter's limits.
- [GPU MODE, lecture 1: Profiling and Integrating CUDA kernels in PyTorch](https://github.com/gpu-mode/lectures): Mark Saroufim. The lecture index lists no session on debugging or `compute-sanitizer`, so this chapter's tool coverage comes from NVIDIA's documentation.
- [GPU MODE, lecture 16: On Hands profiling](https://github.com/gpu-mode/lectures): Taylor Robbie. A profiling session, and the complement to this chapter's correctness tools.
- [Chapter 2](/c/02-the-execution-model), [chapter 6](/c/06-reductions), [chapter 8](/c/08-shared-memory-and-transpose), and [chapter 14](/c/14-triton): independent thread scheduling, the tree reduction, the barrier, and `TRITON_INTERPRET`.
