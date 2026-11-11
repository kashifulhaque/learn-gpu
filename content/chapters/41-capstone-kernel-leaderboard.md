---
title: "Capstone: a kernel leaderboard"
slug: 41-capstone-kernel-leaderboard
part: "Part 6 — Beyond"
summary: Pull the course together on three kernels under a fixed grader. Compute the roofline floor first, change one thing at a time, time honestly with warmup and a flushed L2, and turn a time into a score the way a kernel leaderboard does.
minutes: 180
gpu: true
objectives:
  - "Write the speed-of-light time for a kernel from its ideal bytes and FLOPs, and say which roof it sits under."
  - "Run an optimization loop that predicts, changes one thing, measures, and logs, and say when to stop."
  - "Name the benchmark traps that make a kernel look fast, and describe the timing discipline that avoids each one."
  - "Explain how a kernel leaderboard grades: correctness first, then time against a reference, over several shapes."
  - "Compute a leaderboard score from a time, a floor, and a reference, and combine scores over shapes with a geometric mean."
  - "Plan a fused grayscale and blur, a row-wise top-k, and a fused MLP block, each with its floor and its first three moves."
lab: 41-capstone
---

# Capstone: a kernel leaderboard

> [!TLDR]
> - A fast kernel is a kernel close to its *speed-of-light* time: the larger of the compute time and the memory time for the bytes and FLOPs it can't avoid. Compute that floor before you write a line of kernel code. It tells you how much headroom exists and which resource to chase.
> - Optimize in a loop: predict, change one thing, measure, write it in a log. A change you didn't predict and can't explain is luck, and luck doesn't survive a new shape.
> - Most "fast" results come from a broken benchmark. Time with events, not the host clock. Warm up. Flush the L2 when the data fits in it. Use shapes that aren't tile multiples. Never time the first launch.
> - A leaderboard grades correctness first and speed second, over several shapes, against a reference. This chapter's lab scores a kernel on a log scale between the reference and the floor, then takes a geometric mean over the three tasks.
> - The three starting points are a fused RGB-to-grayscale and blur (bandwidth bound, floor 0.2105 ms), a row-wise top-8 over (4096, 32000) logits (bandwidth bound, floor 0.4115 ms), and a fused MLP block (compute bound, floor 0.2203 ms). Every floor is a prediction on the A100 80GB PCIe, not a result.

Every chapter so far taught one idea and graded it in a lab. This one teaches no new kernel idea. It teaches the method that turns the earlier chapters into a fast kernel when nobody tells you which chapter applies: you get a slow, correct function, a grader, and a clock.

That is the setting of a kernel leaderboard. A reference implementation defines the right answer. Your kernel must match it on every test shape, and then the leaderboard ranks the time. The skill isn't knowing a trick. It's knowing how fast the kernel can possibly be, finding what stands between you and that number, and not fooling yourself about the measurement.

## Before you start

The chapter uses the whole course, and these chapters carry the most weight.

- **The roofline.** [Chapter 4](/c/04-roofline-and-latency) derives the time floor $\max(\text{FLOPs}/\pi, \text{bytes}/\beta)$ and the ridge point. This chapter applies it to three kernels and uses the measured 1275 GB/s copy ceiling in place of the 1935 GB/s rating.
- **Profiling.** [Chapter 5](/c/05-profiling) shows how to time with CUDA events and how to read a profile. Here the profile decides what to change next.
- **GEMM.** [Chapter 10](/c/10-gemm-from-scratch) explains tiling and reuse, and [chapter 14](/c/14-triton) writes the matmul in Triton. The MLP task is two of those with an epilogue.
- **Softmax and fusion.** [Chapter 15](/c/15-softmax-and-normalization) shows a row-wise kernel that reads a row once. The top-k task has the same shape.
- **Top-k and its tie rule.** [Chapter 30](/c/30-sort-top-k-and-sampling) defines top-k by radix select and fixes the tie rule: the lowest index wins.
- **Stencils and fusion of neighbors.** [Chapter 31](/c/31-convolution-and-stencils) builds the halo tile and counts the traffic of fusing a convolution with its neighbors. The blur task is a $3 \times 3$ stencil.

## What a leaderboard measures

This section answers what a leaderboard actually grades, so you can tell what your kernel must do before it is allowed to be fast.

A leaderboard task has three parts: an input generator, a reference implementation, and a list of test shapes. Your submission gets inputs, returns outputs, and the grader compares them with the reference's outputs. A wrong answer scores nothing, however fast it was. This is the rule to design around: ==correctness is a gate, not a term in the score==. A kernel that returns garbage in 10 microseconds ranks below a slow kernel that is right.

Once a kernel passes, the grader times it. Two choices decide what the number means: which shapes to time and how to rank them. Timing several shapes stops a kernel that is tuned for one size from winning. Ranking can use the time on each shape, or a single summary of the times, such as a geometric mean. This chapter's lab uses a geometric mean over tasks, for the reason in the next section.

### What the public leaderboards say

GPU MODE publishes its kernel problems in the open [reference-kernels repository](https://github.com/gpu-mode/reference-kernels). Each problem folder holds a PyTorch `reference.py`, a `task.yml` that generates test cases for different shapes, and a `task.py` that defines the input and output schema. The repository says the problems run on KernelBot, and that the competitions are announced on the GPU MODE Discord. The pages I read don't state how a problem ranks submissions, so read a problem's own files and its leaderboard page for its exact rule before you rely on one. This chapter's score is the lab's own design, and it doesn't claim to copy theirs.

Stanford CS149's assignment 5 uses the same machinery with a different goal. On the page I read, students choose one or more AI-related kernels from a list that includes histogram, FlashAttention, a 3D heat-equation benchmark, and SwiGLU. Each kernel has a PyTorch baseline, and submissions go through the `popcorn-cli` queue to H100 GPUs. The page has a `test` mode that checks correctness and a `benchmark` mode that reports runtime, and it keeps a class leaderboard per kernel. It also says there's no performance bar that sets the grade: most of the grade comes from a work log of steps, measurements, hypotheses, and reasoning.

That second point is the reason for the method in the next section. Even when speed is graded, the log is what shows you understood the speed.

### Where this course's leaderboard lives

This course already stores what a leaderboard needs. Each lab run records its metrics, such as `gray_ms` and `mlp_tflops`, in the app's SQLite database. Those rows are the leaderboard's data: a view that sorts them by a metric would be a small addition to the frontend. This chapter and lab don't build that view. They produce the numbers it would sort, and they define the score.

> [!NOTE] The lab measures; the chapter predicts
> No number in this chapter is a measurement of the lab kernels. Every floor is arithmetic on the A100 80GB PCIe constants in `lab_common.py`: 1275 GB/s measured for a device-to-device copy (learn-inference, chapter 10), 19.5 TFLOP/s for fp32, and 312 TFLOP/s dense for bf16. The lab prints the measured times. If a measurement disagrees with a floor, the floor is the part to check first.

## The speed-of-light floor

This section answers how fast a kernel can be at best, from the bytes it must move and the FLOPs it must do.

A kernel can't finish before the memory system delivers its inputs and accepts its outputs, and it can't finish before the arithmetic units do their work. Those two times overlap on a good kernel, so the floor is the larger of them:

$$
\hla{t_{\text{floor}}} = \max\left(\frac{\hlb{F}}{\pi},\ \frac{\hlc{B}}{\beta}\right)
$$

Here $\hlb{F}$ is the FLOPs the kernel must do, $\pi$ is the peak arithmetic rate, $\hlc{B}$ is the bytes it must move, and $\beta$ is the bandwidth. The floor is the time of the slower resource, and the faster resource hides behind it.

Two details turn this from a formula into a tool.

**Count the ideal bytes, not the baseline's.** The ideal kernel reads every input once, writes every output once, and writes nothing else. An intermediate tensor that the baseline writes and reads back is traffic that a fused kernel removes. If you count the baseline's bytes you compute the baseline's floor, and a kernel can't tell you it has headroom.

**Use the bandwidth a kernel can reach.** The 80GB PCIe card is rated at 1935 GB/s, and a plain device-to-device copy measured 1275 GB/s (learn-inference, chapter 10). The rating is a ceiling no kernel reaches. The copy is the best any kernel did that moved bytes and did nothing else. Judge memory-bound kernels against the copy, and say so when you quote a floor.

For compute, use the peak that the kernel's arithmetic can use: 312 TFLOP/s dense for bf16 on tensor cores, 19.5 TFLOP/s for fp32 on the CUDA cores. A kernel that does fp32 multiply-adds can't reach the tensor-core peak, whatever its tile size.

> [!KEY] The floor tells you what to chase
> If $B/\beta > F/\pi$, the kernel is memory bound, and the only ways to go faster are fewer bytes or more bandwidth. If $F/\pi > B/\beta$, it is compute bound, and fewer bytes won't help. Compute the floor first and you know which resource to profile.

### The three floors

The lab's tasks have these ideal costs. Section by section the chapter derives each one, but here they are together, so you can see that they sit under different roofs.

| Task | Shape | FLOPs | Ideal bytes | Memory floor at 1275 GB/s | Compute floor | Floor |
|---|---|---|---|---|---|---|
| Grayscale and blur | $4096 \times 4096$ image | 369,098,752 | 268,435,456 | 0.2105 ms | 0.0189 ms at 19.5 TFLOP/s | 0.2105 ms |
| Row-wise top-8 | 4096 rows of 32,000 | 0 | 524,681,216 | 0.4115 ms | none | 0.4115 ms |
| Fused MLP | $M=4096$, $d=1024$, $h=4096$ | 68,719,476,736 | 33,564,672 | 0.0263 ms | 0.2203 ms at 312 TFLOP/s | 0.2203 ms |

The first two are memory bound by a wide margin, and the third is compute bound by a wide margin: its arithmetic intensity is 2047 FLOP per byte against a ridge point of $312 \times 10^{12} / 1275 \times 10^{9} = 244.7$. So the first two tasks are about not wasting bytes, and the third is about keeping the tensor cores fed.

Each floor is a prediction. The lab measures the real times, and the interesting part of the exercise is the gap.

## The optimization loop

This section answers what to do between a correct baseline and a fast kernel, one step at a time.

The loop has five steps, and the discipline is in doing all five every time.

1. **Floor.** Write the floor once, at the start. Everything else is a fraction of it.
2. **Profile.** Find what the current kernel is limited by: achieved bandwidth, achieved FLOP/s, occupancy, stall reasons ([chapter 5](/c/05-profiling) and [chapter 28](/c/28-reading-profiler-reports)). Place it on the roofline.
3. **Predict.** Pick one change and write down what it should do to the time, with a number. "Fusing the blur removes the intermediate image, which cuts 134 MB, so the time should fall by about 0.1 ms."
4. **Change one thing.** One change per measurement. Two changes together hide which one helped, and a regression hides inside a gain.
5. **Measure and log.** Time it with the discipline in the next section. Write down the prediction, the measurement, and whether they agree.

The log is not paperwork. When the measurement disagrees with the prediction, you have learned something about the machine, and that's the only way you'll find the next bottleneck. A row where the prediction held tells you the model is right. A row where it didn't tells you where to look.

### A log you can fill in

Here is the shape of a log for the grayscale task. The predictions come from the byte counts in this chapter. The measurement column is empty, because the lab measures it, and you fill it in.

| Step | Change | Predicted effect | Measured | Verdict |
|---|---|---|---|---|
| 0 | Eager baseline: matmul for luma, then `conv2d` | Moves 402,653,184 bytes, a floor of 0.3158 ms | | |
| 1 | Fuse luma and blur in one Triton kernel | Moves 268,435,456 bytes, a floor of 0.2105 ms. The time falls by about a third | | |
| 2 | Wider block along the row | Fewer program launches, same bytes. Small change | | |
| 3 | Keep a row of luma in registers instead of recomputing it nine times | Fewer loads from L1, same DRAM bytes. Matters only if step 1 is not near the floor | | |

Step 3's prediction is a conditional on purpose. If step 1 already sits near the floor, step 3 has no headroom to win, and the log says to stop.

### When to stop

Stop when one of three things is true. The kernel is within the noise of its floor, so there's nothing left to win. The profile shows a limit you can't move without leaving the task, such as a precision the grader doesn't allow. Or the next change is predicted to gain less than the run-to-run spread of the measurement, so you couldn't tell if it worked.

## Measuring honestly

This section answers why a kernel appears faster than it is, and what a timing harness has to do about each cause.

A leaderboard turns your time into rank, so the temptation to measure badly is built in, and most bad measurements aren't dishonest. They're accidents that favor you. Learn the list, because the same accidents will favor you when you measure your own work.

### Trap 1: timing without a synchronize

A kernel launch returns to the host in a few microseconds, and the GPU runs the kernel later ([chapter 0](/c/00-what-you-are-building) shows this). A host clock that brackets a launch measures the queueing. A 2 ms kernel can look like a 5 microsecond one.

The fix is CUDA events on the stream, as `cuda_time` does, or a `torch.cuda.synchronize()` before each clock read. An event pair measures the GPU's own time between two points in the stream, and it doesn't care how long the host took to queue the work.

### Trap 2: the cache remembers your input

The A100's L2 holds 40 MiB. An input that fits in it, read again by the next iteration, is read from the cache and not from DRAM. The benchmark then reports a bandwidth above what the memory can do. A $2048 \times 2048$ float32 image is 16 MiB and fits twice.

The fix is to flush: write to a buffer larger than the L2 before each timed call, so every call starts cold. `cuda_time(..., flush_l2=True)` writes twice the L2 size. The lab's image is 201 MB and its logits are 524 MB, both larger than the L2. The MLP's inputs are 25 MB, which fit, so the flush matters there.

### Trap 3: shapes that are tile multiples

A kernel tuned on $4096 \times 4096$ has no ragged edges: every tile is full, and every mask is true. A leaderboard's test shapes include sizes that aren't multiples of your tile, such as $1023 \times 1025$, and a kernel with no masking reads and writes out of bounds there. Correct kernels pay for their boundary handling, so a benchmark that uses only round sizes flatters the kernels that skip it.

Test odd shapes for correctness, and time the shape the task names. The lab's correctness checks use $1 \times 1$, a single row, a single column, and $1023 \times 1025$ for the image, and sizes such as $(257, 192, 520)$ for the MLP.

### Trap 4: timing the first launch

The first call of a Triton kernel compiles it, which takes seconds. The first call of a `load_inline` extension compiles it with `nvcc`, which takes 30 to 60 seconds. The first call of a cuBLAS routine may initialize a handle and pick an algorithm. A time that includes any of these is a measurement of the compiler.

The fix is warmup: run the function several times before the timed loop, and drop those times. Warmup also lets the GPU clocks settle and fills the instruction cache. The lab's `cuda_time` runs a fixed number of warmup calls first, and the CPU check `robust_time` in this lab writes the same rule for recorded times: drop the first few, then take the median.

### Trap 5: the mean of a noisy sample

GPU timings have a floor and a long tail. Most iterations take about the same time, and a few take much longer because another process used the GPU, a clock changed, or the OS ran something. An average of 50 runs with one 30 ms outlier among 1 ms runs reports 1.6 ms, which is wrong by 60%.

Report the median. It ignores the tail, and it is stable between runs. For a leaderboard that wants a spread, also record the minimum and the 90th percentile. In the lab, `robust_time` takes a list of per-iteration times, drops `warmup` of them, and returns the median of the rest.

> [!EXAMPLE] Why robust_time uses the median
> Take the recorded times `[50.0, 9.0, 1.2, 1.0, 1.1, 1.0, 30.0, 1.05]` ms, with two warmup iterations. Drop the first two. Six remain: `1.2, 1.0, 1.1, 1.0, 30.0, 1.05`. Sorted, they are `1.0, 1.0, 1.05, 1.1, 1.2, 30.0`, and the median is $(1.05 + 1.1)/2 = 1.075$ ms. The mean of the same six is 5.9 ms, and the median of all eight, warmup included, is 1.15 ms.

### Trap 6: an impossible number

Every trap above makes a kernel look fast. A time below the floor is how you notice it. If a kernel that must read 524 MB reports 0.2 ms, it moved 2.6 TB/s, which is above the 1935 GB/s the card is rated for. The kernel isn't magic: the timer is wrong, the input was cached, or the kernel skipped work and returned the wrong answer on a shape the grader didn't test.

The lab's timing check applies this test. It computes the floor at the rated bandwidth and peak, with no allowance for what a kernel reaches, and fails a kernel that reports less than 95% of it. A correct kernel can't be faster than the hardware's rating, so a time below it means the measurement is broken.

> [!WARNING] A speedup above the roofline is a bug report
> When a result beats the floor, don't celebrate and don't submit it. Find out why. In order of likelihood: the timer measures queueing, the L2 holds the input, the kernel does less work than the reference, or the reference was slow for a reason that has nothing to do with your change.

## From a time to a score

This section answers how a leaderboard turns a time into a number you can compare across tasks, and what the lab's score rewards.

A time in milliseconds can't be compared across tasks: 0.4 ms on top-k and 0.2 ms on the blur are different amounts of work. A score divides out the work by comparing your time with two reference points. The reference time is what a plain implementation does, and the floor is what the hardware allows. Those bracket the range of results, and your kernel lands somewhere between them.

The lab's score is the fraction of that range you cover, on a log scale:

$$
\boxed{\text{score} = \operatorname{clip}\left(\frac{\ln(\hla{t_{\text{ref}}} / \hlb{t})}{\ln(\hla{t_{\text{ref}}} / \hlc{t_{\text{floor}}})},\ 0,\ 1\right)}
$$

The numerator is how many times faster than the reference you are, as a logarithm. The denominator is how many times faster than the reference the floor is. A kernel as slow as the reference scores 0, and one at the floor scores 1.

### Why a log scale

Speedups are ratios. Going from 8 ms to 4 ms and from 4 ms to 2 ms are the same achievement, one halving each, and they should score the same. On a log scale they do. On a linear scale the first halving gains 4 ms of a 7 ms range and the second gains only 2 ms, so the linear form rewards the first speedup twice as much.

A log score also fits the way kernels get faster. The easy changes give large ratios and the last ones give small ratios, because the remaining headroom shrinks. A score that is flat in the ratio keeps the last 20% of effort visible.

> [!EXAMPLE] One time, two scores
> Take a reference at 4 ms and a floor at 1 ms. A kernel at 2 ms is halfway to the floor in ratio, since $4/2 = 2$ and $4/1 = 4$, and $\ln 2 / \ln 4 = 0.5$. The linear form gives $(4 - 2)/(4 - 1) = 0.667$. The two agree only at the ends. At 3 ms, the log score is $\ln(4/3)/\ln 4 = 0.208$ and the linear score is 0.333.

The figure lets you move the reference and your time, in multiples of the floor, and watch the two forms.

```viz
41-score-curve
```

The clip at 1 handles a time below the floor, and it deserves suspicion, as the last section said. The lab reports the score with the clip and separately fails a kernel whose time is below the hardware's rated limit.

### Combining shapes and tasks

A leaderboard that times several shapes needs one number per kernel. An arithmetic mean of scores lets a great result on one shape cover for a bad result on another. A *geometric mean* doesn't:

$$
\text{geomean}(s_1, \ldots, s_n) = \exp\left(\frac{1}{n}\sum_{i=1}^{n} \ln s_i\right)
$$

Three scores of 0.5, 0.8, and 0.2 have an arithmetic mean of 0.5 and a geometric mean of $0.08^{1/3} = 0.431$. A zero anywhere gives a zero, which is the point: a kernel that fails to improve one task hasn't finished the capstone. The lab's `capstone_score` is the geometric mean of the three task scores.

For speedups the geometric mean has a second property. The geometric mean of ratios doesn't depend on which implementation you call the reference, and it doesn't change if you measure in seconds instead of milliseconds. That's why it is the usual summary for benchmark suites that compare many programs.

## Starting point one: grayscale and blur

This section answers how fast a fused RGB-to-grayscale and $3 \times 3$ blur can be, and what stands between the eager baseline and that floor.

**The task.** The input is a float32 image of shape $(H, W, 3)$. The luma is $0.299R + 0.587G + 0.114B$. Then a blur with the kernel $[1,2,1] \otimes [1,2,1] / 16$, with zeros outside the image. The output is a float32 image of shape $(H, W)$. The lab's shape is $4096 \times 4096$, and its correctness shapes include $1 \times 1$, a single row, a single column, and $1023 \times 1025$.

**The floor.** Each pixel is read as 12 bytes and written as 4. The image has $4096^2 = 16{,}777{,}216$ pixels, so the ideal traffic is $16 \times 16{,}777{,}216 = 268{,}435{,}456$ bytes. At 1275 GB/s that's $0.2105$ ms. The arithmetic is 22 FLOP per pixel (3 multiplies and 2 adds for the luma, 9 multiplies and 8 adds for the blur), or 369 MFLOP, which at 19.5 TFLOP/s is 0.0189 ms. The kernel is memory bound by a factor of 11, and the floor is 0.2105 ms. Against the card's 1935 GB/s rating the same bytes would take 0.1387 ms, which no kernel will reach.

**The baseline.** The eager version computes the luma with a matrix-vector product and then calls `conv2d`. The first kernel reads 12 bytes per pixel and writes 4, and the second reads 4 and writes 4, so the pair moves 24 bytes per pixel, or 402,653,184 bytes. At the copy ceiling that's a floor of 0.3158 ms for the *unfused* program, and it is a lower bound: a library convolution may move more, through a layout change or a workspace. The gap between the unfused and fused floors is one intermediate image, written and read once, which is 134,217,728 bytes or 0.1053 ms.

### The first three moves

1. **Fuse.** Compute the luma inside the blur kernel. The intermediate image never reaches memory, and the traffic falls from 24 to 16 bytes per pixel. The prediction is a speedup of at most $24/16 = 1.5$ over the baseline's own floor. The simplest fusion has each output pixel load its nine neighbors' RGB values and compute nine lumas. That reads $9 \times 12 = 108$ bytes per pixel from the cache hierarchy, and only the first read of each cache line comes from DRAM. Whether the L1 and L2 absorb the repeats is the thing to profile.

2. **Reuse the luma.** Nine neighbor loads is nine times the instruction traffic. A tile that stages the luma values with a one-pixel halo, as in [chapter 31](/c/31-convolution-and-stencils), computes each luma once per tile. In Triton you do this by processing a strip and keeping the previous row's values in registers. In CUDA you use shared memory. The halo overhead for a $16 \times 16$ tile is $(18/16)^2 = 1.266$ luma computations per output.

3. **Widen the loads.** The image is interleaved: each pixel is three consecutive floats. A thread that loads R, G, and B separately issues three loads with a stride of 12 bytes, and a warp's R loads span 384 bytes. Treating the row as a flat array of $3W$ floats and loading contiguous spans, then separating the channels in registers, makes every load a full contiguous transaction ([chapter 3](/c/03-memory-and-bandwidth)).

> [!TIP] Count loads, not just bytes
> The DRAM traffic of the fused kernel is 16 bytes per pixel whether it loads each neighbor 9 times or once, as long as the cache catches the repeats. The loads still take issue slots and L1 bandwidth. If the profile shows the kernel is limited by L1 throughput and not by DRAM, the next move is the second one, whatever the byte count says.

The zero-padding rule is a correctness trap. A kernel that clamps coordinates to the image reads the border pixel again and returns 1.0 at the corner of an all-ones image, where the reference returns $9/16 = 0.5625$. The lab's tests include a reference check for this by hand: on a $3 \times 3$ all-ones image the blur is 9/16 at the corners, 12/16 on the edges, and 1 at the center.

## Starting point two: row-wise top-k

This section answers how to select the 8 largest of 32,000 logits per row at the speed of reading them.

**The task.** The input is a float32 tensor of shape $(4096, 32000)$, which is 524,288,000 bytes. The output is, for each row, the 8 largest values in descending order and their int64 indices. The tie rule is the one from [chapter 30](/c/30-sort-top-k-and-sampling): among equal values, the lowest index wins, which matches the first 8 entries of a stable descending sort.

**The floor.** Every logit must be read once, and nothing else is large. The output is $4096 \times 8 \times 12 = 393{,}216$ bytes, so the ideal traffic is $524{,}681{,}216$ bytes. At 1275 GB/s that's $0.4115$ ms. The arithmetic is comparisons, which aren't FLOPs, and the card has plenty of integer throughput, so the floor is set by memory. At the rating it would be 0.2712 ms.

**The baseline.** A full stable sort of 32,000 values per row, then a slice. A sort does many passes over the data and writes sorted keys and indices that the slice throws away. It does far more than 8 rows' worth of selection. The lab keeps it as the baseline because it satisfies the tie rule by construction, and because the gap is the lesson: selecting $k$ of $n$ doesn't need the order of the other $n - k$.

### The shape of the problem

Each row is independent, so 4096 rows are 4096 blocks of parallel work. That's about 38 blocks per SM across 108 SMs, enough to fill the machine, so you don't need to split a row across blocks. A row is 128,000 bytes, which fits in a block's registers and shared memory. The design is one block per row that reads its row exactly once.

The read has to run at memory speed, and the selection must hide behind it. That makes the cost question: how much work does the block do per element after the load?

### The first three moves

1. **Repeat the maximum.** Hold the row in registers. Find the maximum, find the lowest index among the elements equal to it, record both, replace that element with minus infinity, and repeat 8 times. This is the lab's reference solution. It reads DRAM once, which meets the floor's byte count, and then does 8 rounds of two block-wide reductions each. The reductions are the cost: a block-wide reduction takes several synchronizations, so the kernel is limited by how fast it can do 16 of them, not by memory, until enough blocks are resident to hide each other's latency.

2. **Keep a thread-local top 8.** Give each thread a slice of the row and a sorted list of its 8 best candidates. A new element enters the list only if it beats the current 8th, and in a random row that becomes rare quickly. Most elements cost one comparison. Then merge the 1024 threads' lists with a tree. The loads are the same, and the reductions drop from 16 per row to one merge tree. The tie rule needs care here: compare $(\text{value}, -\text{index})$ pairs, so that a tied element with a lower index counts as larger.

3. **Choose the block size from the occupancy.** A block of 1024 threads holding 32 elements each limits the SM to two blocks, and a block of 256 threads with 128 elements each allows more but uses more registers per thread. The occupancy arithmetic of [chapter 2](/c/02-the-execution-model) decides which wins, and the profile shows whether the kernel stalls on loads or on synchronizations.

The alternative, radix select from [chapter 30](/c/30-sort-top-k-and-sampling), narrows the 8th largest key one digit at a time in 4 passes over the row. It fits when $k$ is large. With $k = 8$ the thread-local list is cheaper, and radix select's 4 passes are 4 reads of the row from the L1 or the shared memory.

### The tie rule decides the grader

The lab compares the values exactly with the stable sort's values, and compares the indices as sets under the tie rule. The set comparison is the reason `topk_matches` is a function you write: a kernel may return its 8 indices in any order, and a comparison in order would reject a correct kernel.

PyTorch's `torch.topk` doesn't promise a stable tie order, which [chapter 30](/c/30-sort-top-k-and-sampling) quotes from its documentation. The lab reports its time as a metric, `topk_torch_topk_ms`, and doesn't use it as the grader's reference. A faster library call that breaks the tie rule is a faster wrong answer under this grader.

> [!WARNING] The all-equal row
> A row of 32,000 equal values has one correct answer: indices 0 to 7. A kernel that keeps every element at or above the threshold returns 32,000 indices, and a kernel that picks ties from the wrong end returns the last 8. The lab includes a constant row and rows of values rounded to a quarter, so ties are common.

## Starting point three: a fused MLP block

This section answers how close a transformer's feed-forward block can get to the tensor-core peak, and what fusing it can and can't save.

**The task.** The inputs are bf16: $x$ of shape $(M, d)$, $W_1$ of shape $(d, h)$, $b_1$ of length $h$, $W_2$ of shape $(h, d)$, and $b_2$ of length $d$. The output is bf16:

$$
y = \operatorname{gelu}(x W_1 + b_1)\, W_2 + b_2
$$

Each matrix product accumulates in fp32, and the GELU is the exact one, with the error function. The lab's shape is $M = 4096$, $d = 1024$, $h = 4096$, a four-times expansion like the feed-forward block of a small transformer.

**The floor.** The FLOPs are two GEMMs: $2 M d h$ each, so $4 \times 4096 \times 1024 \times 4096 = 68{,}719{,}476{,}736$. At 312 TFLOP/s dense bf16 that's $0.2203$ ms. The ideal bytes are $x$ and $y$ at 2 bytes each, both weight matrices, and the biases: $2(2 \cdot 4096 \cdot 1024 + 2 \cdot 1024 \cdot 4096 + 4096 + 1024) = 33{,}564{,}672$ bytes, or 0.0263 ms at 1275 GB/s. The intensity is 2047 FLOP per byte against a ridge of 244.7, so the block is compute bound by a factor of 8, and the floor is 0.2203 ms.

Treat that floor with care. It assumes the tensor cores run at the rated peak for the whole kernel. [Chapter 10](/c/10-gemm-from-scratch) grades SGEMM as a percent of cuBLAS, and no real GEMM runs at the rated peak. A score of 1 is out of reach for any implementation you'll write, and that is fine. The score measures how much of the headroom you close, and the lab reports `mlp_fraction_of_peak` so you can see the ceiling you're actually working against.

**The baseline.** Eager PyTorch calls `addmm` for the first product, then a separate GELU kernel, then `addmm` for the second. The GELU reads and writes the hidden layer, which has $M h = 16{,}777{,}216$ values, or 33,554,432 bytes in bf16. The first GEMM writes that tensor once, the GELU reads and writes it, and the second GEMM reads it: four transfers of 33.5 MB, or 134 MB, which is 0.1053 ms at the copy ceiling. A fused program removes some of it.

### What fusion can save here

Fusing the GELU into the first GEMM's epilogue removes the GELU kernel and two of the four transfers: 67,108,864 bytes, or 0.0526 ms at 1275 GB/s. The GEMM computes a tile of $xW_1 + b_1$ in fp32 registers anyway, so the bias and the GELU are a few instructions on data that is already there, and the tile is written to memory once, as bf16. The same holds for $b_2$ in the second GEMM. This is the epilogue fusion of [chapter 24](/c/24-cute-dsl-gemm) and [chapter 29](/c/29-gemm-scheduling).

The lab's solution stops there, with two kernels. It writes the hidden layer once and reads it once. That is 67 MB of traffic that the ideal fused kernel doesn't have, and it costs nothing in the floor, because the memory time is already 8 times smaller than the compute time and the loads overlap with the tensor-core work.

Fusing further, so that the hidden layer never reaches memory, means each block computes a tile of the hidden layer and multiplies it by $W_2$ at once. Here the shapes work against it. A block that owns $B_M$ rows of the output needs to accumulate a $B_M \times d$ fp32 tile, and the register file of an SM holds $65536 \times 4 = 262{,}144$ bytes. At $d = 1024$, a tile of 128 rows needs 524,288 bytes, twice the register file, and even 64 rows fill it completely. A smaller tile fits, but each block then streams both weight matrices, 16,777,216 bytes in all, once per tile: with 16 rows per block that is $256$ blocks and 4.3 GB of weight reads. A fully fused kernel at this size trades 67 MB of hidden-layer traffic for gigabytes of weight traffic, so it doesn't win unless the weights are shared between blocks.

> [!KEY] Fuse the epilogue, not the whole block
> At $d = 1024$ the right fusion is the epilogue of each GEMM. The hidden layer is cheap to write and read, and the weights are expensive to re-read. A rule for any fusion: count the bytes you remove against the bytes the fusion makes you re-read.

### The first three moves

1. **Replace the eager chain with two tensor-core GEMMs that have epilogues.** This is the chapter 14 matmul with `tl.dot`, bf16 inputs, and an fp32 accumulator, plus the bias and the GELU before the store. The prediction is the removal of the GELU kernel's two transfers, 0.0526 ms in the best case. The comparison to cuBLAS is the hard part: the eager baseline's GEMMs are cuBLAS's, so the speedup can be below 1 if your GEMM is slower than its. The score is 0 in that case, and the log says so.

2. **Fix the wave count of the second GEMM.** The second product has $M = 4096$ rows and only $N = d = 1024$ columns, so $128 \times 128$ tiles give $32 \times 8 = 256$ tiles for 108 SMs: $256/108 = 2.37$ waves, which runs as 3 and wastes the rest of the last one. The efficiency is $2.37/3 = 79\%$ if one tile runs per SM at a time. Tiles of $128 \times 64$ give 512 tiles, $512/108 = 4.74$ waves, which runs as 5, an efficiency of 95%. The first GEMM, with 1024 tiles, is already at $9.48/10 = 95\%$. These are predictions from the tile counts. [Chapter 4](/c/04-roofline-and-latency) introduces wave quantization and [chapter 29](/c/29-gemm-scheduling) schedules around it with split-K and stream-K.

3. **Tune the tile and the pipeline.** Block sizes, the number of software-pipeline stages, and the number of warps trade shared memory against latency hiding. [Chapter 12](/c/12-pipelining-and-async-copies) explains the pipeline, and Triton's autotuner searches the space. Autotuning on the timed shape is allowed in this lab, but the tuned configuration has to be correct on every shape, and an autotune that runs inside the timed loop is a trap of its own, the first-launch trap.

The correctness check for this task needs a decision about error. The reference is float64 with an exact GELU and no rounding of the hidden layer. Your kernel rounds the hidden layer to bf16, which has an 8-bit significand, a relative error of about $2^{-9}$ per element. Those errors are independent, so they add as a root sum of squares and stay small. The lab accepts a maximum error up to 2% of the largest output, and the eager bf16 block, run on the CPU while the lab was written, lands near 0.3%. A kernel that drops $b_1$, or applies the GELU after the second GEMM, is wrong by a large fraction of the output scale, and the check catches it with room to spare.

## Putting the three together

This section answers how to spend your time across three tasks that have different floors.

The three tasks reward different habits. The grayscale kernel is a lesson in not moving bytes you don't need: after fusion, the headroom left is the gap to the copy ceiling. The top-k kernel is a lesson in latency: the bytes are fixed, and the work is hiding reductions behind loads. The MLP is a lesson in not losing tensor-core time: the bytes don't matter, and the gap to cuBLAS does.

Spend the effort where the score moves. The score is a geometric mean, so a task at 0 zeroes the whole thing. Raising one task from 0.1 to 0.3 triples the product of the scores and multiplies the geometric mean by $3^{1/3} = 1.44$. Raising one from 0.8 to 0.9 multiplies the mean by $1.125^{1/3} = 1.04$. Raise the weakest task first.

### What to put in the log

For each task, record at least these columns for every step: the change, the predicted time, the measured median, the fraction of the floor, and what the profile said. Keep the failed attempts. A leaderboard shows only the winning time, and your log is where the reasoning lives, which is also what CS149's assignment grades on the page I read: the steps, the measurements, the hypotheses.

## What goes wrong

This section lists the mistakes the lab is likely to provoke, with the symptom of each.

- **The kernel is faster than the floor.** The timing is wrong, so go back to the traps. Check that the timer uses events or a synchronize, that the L2 is flushed, and that the kernel does all the work. The lab's timing check fails this on purpose.
- **A correct kernel scores 0.** Its time is at or above the reference's. For the MLP this means your GEMM is no faster than cuBLAS with a separate GELU. The score isn't broken. The reference was strong.
- **The kernel passes the round shape and fails the odd one.** A missing mask reads or writes past the edge. Test $1 \times 1$, a single row, and a width that isn't a multiple of your block.
- **The first run takes a minute.** `nvcc` or the Triton compiler is running. Compile in a warmup call. A time that includes it is wrong.
- **The measured time moves by 10% between runs.** One run is a sample. Use the median of many iterations, and compare two kernels in the same run, with the same flush.
- **A tie goes to the wrong index.** A kernel that breaks ties by thread order or by the last index fails the tie rule. The set comparison catches it, and the values alone don't.
> [!RECAP]
> - The floor is $\max(F/\pi, B/\beta)$ for the ideal fused kernel, at the 1275 GB/s copy ceiling and the peak that the arithmetic can use. Compute it before you optimize.
> - Predict, change one thing, measure, and log. Stop when you're within the noise of the floor, or when the next change can't beat the noise.
> - A fast result needs the timing discipline: events, warmup, a flushed L2, odd shapes, and the median. A time below the rated limit is a bug in the measurement.
> - A leaderboard gates on correctness, then ranks time over several shapes. The lab's score is a log fraction of the headroom between the reference and the floor, combined by a geometric mean.
> - The three floors are 0.2105 ms for the fused grayscale and blur, 0.4115 ms for the top-8 over $(4096, 32000)$, and 0.2203 ms for the fused MLP. They are predictions, and the lab measures the gaps.

## Check your understanding

> [!QUESTION] A kernel reads 1 GB and writes 1 GB, and does 4 GFLOP in fp32. What is its floor on the A100 80GB PCIe, and what limits it?
> The bytes are $2 \times 10^{9}$, so the memory time is $2 \times 10^{9} / 1.275 \times 10^{12} = 1.569$ ms. The compute time is $4 \times 10^{9} / 19.5 \times 10^{12} = 0.205$ ms. The floor is 1.569 ms and memory limits it.

> [!QUESTION] The fused MLP's FLOPs are $4Mdh$. Where does the 4 come from, and which terms does the count ignore?
> Each GEMM does $2Mdh$ FLOPs, one multiply and one add per product term, and there are two of them. The count ignores the bias adds and the GELU, which are $O(Mh)$ and $O(Md)$ against $O(Mdh)$ for the products.

> [!QUESTION] A kernel's benchmark loop calls it 100 times on the same 16 MiB input and reports 2 TB/s. Name two causes and a fix for each.
> The 16 MiB input fits in the 40 MiB L2, so every iteration after the first reads from it, and the fix is to flush the L2 between iterations. Or the timer brackets an asynchronous launch and reads the clock before the kernel runs, and the fix is to record CUDA events or synchronize before reading the clock.

> [!QUESTION] With a reference at 6 ms and a floor at 1.5 ms, what does a kernel at 3 ms score on the lab's scale, and what would a linear score say?
> The ratio is $6/3 = 2$ against $6/1.5 = 4$, so the log score is $\ln 2/\ln 4 = 0.5$. The linear score is $(6 - 3)/(6 - 1.5) = 0.667$.

> [!QUESTION] Why does the lab combine task scores with a geometric mean, and what does a task at 0 do?
> A geometric mean lets no strong task cover for a weak one, and it doesn't depend on units or on which implementation is the reference. A task at 0 gives an aggregate of 0.

> [!QUESTION] The top-k kernel returns indices `[0, 3]` for the row `[3, 1, 3, 3, 0]` with $k = 2$. Is it correct, and why?
> It isn't. The values match, since both selected elements are 3. The tie rule picks the lowest indices among the tied 3s, which are 0 and 2, so the right set is $\{0, 2\}$. A comparison that checks values only would accept it.

> [!QUESTION] Why doesn't fusing the whole MLP block into one kernel win at $d = 1024$?
> A block that owns $B_M$ rows of the output needs a $B_M \times d$ fp32 accumulator, which at 128 rows is 524,288 bytes, twice an SM's 262,144-byte register file. Smaller tiles fit, but then each block re-reads the 16,777,216 bytes of weights, which costs far more than the 67 MB of hidden-layer traffic it saves.

> [!QUESTION] The second GEMM has a $4096 \times 1024$ output. How many $128 \times 128$ tiles is that, and what's the wave efficiency on 108 SMs?
> $32 \times 8 = 256$ tiles. $256/108 = 2.37$ waves, which runs as 3, so the efficiency is $2.37/3 = 79\%$ when one tile runs on each SM at a time.

## Lab

> [!TRY] Write the grader, then beat the baselines
> Open `41-capstone`. Part 1 runs on any machine. Write `traffic` first, which passes the first check. Then `speed_of_light`, `robust_time`, `score`, `geomean`, and `topk_matches`. Every check shows an expected and an actual value, and the values are worked by hand in the tests.
>
> Part 2 starts from three correct, slow functions: `grayscale_blur`, `topk_rows`, and `mlp_block`. Replace each with a Triton kernel or a CUDA kernel built with `load_cuda`. The tests compare each against a float64 reference on odd shapes, with the stable-sort tie rule for top-k. A last check times the kernels with warmup and a flushed L2, and fails a kernel that beats the rated hardware limit or runs more than twice as slow as the eager baseline.
>
> On a machine without a GPU, Part 1's six checks run and the GPU checks skip with a message. The starter passes none of Part 1 except the grader's own reference check, so it fails until you write the functions.

The lab reports these metrics, none of which fails the lab on its own:

- `gray_ms`, `topk_ms`, and `mlp_ms`: the median time of each kernel at the lab shape.
- `gray_gbs`, `gray_fraction_of_ceiling`, and `topk_gbs`: ideal bytes over time, and for the image the fraction of the 1275 GB/s copy ceiling.
- `mlp_tflops` and `mlp_fraction_of_peak`: ideal FLOPs over time, and the fraction of the 312 TFLOP/s bf16 peak.
- `gray_speedup`, `topk_speedup`, and `mlp_speedup`: the eager baseline's time over yours.
- `gray_score`, `topk_score`, `mlp_score`, and `capstone_score`: the scores on the scale in this chapter.
- `topk_torch_topk_ms`: the time of `torch.topk`, for comparison. It doesn't promise the tie rule.

## Further reading

- [GPU MODE reference-kernels](https://github.com/gpu-mode/reference-kernels). The open problems behind the GPU MODE kernel competitions. Each problem has a PyTorch `reference.py`, a `task.yml` of test shapes, and a `task.py` for the input and output schema. Read a problem's own files for its ranking rule.
- [Stanford CS149, assignment 5](https://github.com/stanford-cs149/asst5-kernels). An open-ended kernel assignment with PyTorch baselines, a `test` and a `benchmark` mode, a class leaderboard per kernel, and a grade that depends mostly on a work log. The page names H100 GPUs and the `popcorn-cli` queue.
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html), NVIDIA. Section 9.1, Timing: the warning that kernel launches are asynchronous, and timing with CUDA events.
- Samuel Williams, Andrew Waterman, and David Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures," Communications of the ACM, 2009. The model behind every floor in this chapter. Cited from memory; the page could not be fetched.
- [`torch.topk`](https://docs.pytorch.org/docs/2.14/generated/torch.topk.html), PyTorch. The documentation on the order of tied elements.
- [Chapter 4](/c/04-roofline-and-latency), for the roofline, [chapter 14](/c/14-triton), for the Triton matmul, [chapter 30](/c/30-sort-top-k-and-sampling), for top-k and its tie rule, and [chapter 31](/c/31-convolution-and-stencils), for halo tiles and fusion traffic.
