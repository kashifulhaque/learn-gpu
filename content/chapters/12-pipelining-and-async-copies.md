---
title: Pipelining and asynchronous copies
slug: 12-pipelining-and-async-copies
part: "Part 3 — Matrix multiplication"
summary: How a GEMM hides memory latency behind tensor-core work with cp.async and a multistage software pipeline, and how Hopper's TMA and wgmma and Blackwell's tcgen05 and tensor memory take the same idea further.
minutes: 120
gpu: true
objectives:
  - "Explain why a GEMM block that loads, waits, and then computes leaves the tensor cores idle, and estimate the idle fraction from a load latency and a tile's compute time."
  - "Write cp.async with 16-byte copies, zero-fill for tails, and commit_group and wait_group, and say what each one guarantees."
  - "Build a prologue, steady-state, and epilogue software pipeline, and derive the wait_group count for any number of stages."
  - "Choose the stage count from the shared-memory capacity, the tile size, and the number of blocks per SM."
  - "Describe Hopper's TMA, mbarriers with transaction counts, wgmma, clusters, warp specialization, and persistent kernels, and what each one removes from the instruction stream."
  - "Describe Blackwell's tcgen05.mma, tensor memory, and 2-CTA MMA, and how the accumulator moves out of the register file."
  - "Extend a tensor-core GEMM with a 2 to 4 stage cp.async pipeline and measure the speedup over one stage."
lab: 12-cp-async
---

# Pipelining and asynchronous copies

> [!TLDR]
> - A GEMM block that loads a tile, waits, and then computes leaves the tensor cores idle for the length of the load. On the A100, a 128 x 128 x 32 tile takes 512 cycles of tensor-core work, and a trip to HBM takes about 466.
> - `cp.async` (Ampere) copies 4, 8, or 16 bytes from global to shared memory without going through registers, so a thread issues the copy and keeps going. `commit_group` and `wait_group` tell you when it has landed.
> - A multistage pipeline keeps S - 1 tile loads in flight while the tensor cores work on one. The wait is always `wait_group S - 2`, and the shared memory it costs sets the stage count.
> - Hopper moves the copy to a hardware unit (TMA), tracks it with mbarrier byte counts, and runs the MMA asynchronously (`wgmma`). Producer and consumer warps then specialize.
> - Blackwell puts the accumulator in tensor memory and lets one thread issue the MMA (`tcgen05.mma`), so registers stop being the bottleneck.

A fast GEMM does two things at once: it feeds the tensor cores with operands, and it computes on them. The previous chapter ended with a kernel that did the second well. [Tensor cores](/c/11-tensor-cores) showed `mma.sync` and `ldmatrix` reaching most of an SM's arithmetic rate when the operands were already in shared memory. Getting them there is this chapter.

In a transformer, nearly every large FLOP is a GEMM: the QKV and output projections, the MLP, the LM head. How close a GEMM gets to peak decides how long a training run takes. Much of the gap is this chapter's subject, overlapping the loads of the next tile with the arithmetic of the current one. Every fast GEMM since Ampere does it, and so do Triton, FlashAttention, and CUTLASS.

## Before you start

**Tile and k-tile.** A block computes a tile of C, for example 128 x 128, by walking down the K dimension in slices of BK. Each slice needs a BM x BK piece of A and a BK x BN piece of B in shared memory. This chapter calls one such step a *k-tile*. [GEMM from scratch](/c/10-gemm-from-scratch) builds the tiling.

**`mma.sync` and `ldmatrix`.** One warp executes `mma.sync.m16n8k16`, a 16 x 8 x 16 multiply-accumulate, on fragments held in registers, and `ldmatrix` fills those registers from shared memory. [Tensor cores](/c/11-tensor-cores) covers both.

**Shared-memory banks.** Thirty-two banks serve 4 bytes each per clock, and a warp that hits one bank twice serializes. Padding a row by 16 bytes, as this chapter's kernel does, removes the conflict. [Shared memory and the transpose](/c/08-shared-memory-and-transpose) explains why.

**Latency and occupancy.** A global load takes hundreds of cycles to return, and the SM hides that by running other warps. [The roofline and latency hiding](/c/04-roofline-and-latency) derives how much work must be in flight.

## Why the tensor cores wait

The section answers one question: what does a block do while its next tile is in flight? In the kernel from chapter 11 the answer is nothing, and the arithmetic shows how expensive that is.

Take the lab's block tile, 128 x 128 with BK = 32. One k-tile is a 128 x 32 slice of A times a 32 x 128 slice of B:

$$
\hla{\text{FLOP per k-tile}} = 2 \times 128 \times 128 \times 32 = 1{,}048{,}576
$$

An A100 SM does 2048 bf16 FLOP per clock (108 SMs x 2048 x 1.41 GHz gives the rated 312 TFLOP/s, per the A100 whitepaper's tensor-core rate). If the block had the tensor cores to itself, the tile would take:

$$
\hlb{T} = \frac{1{,}048{,}576\ \text{FLOP}}{2048\ \text{FLOP/clock}} = 512\ \text{clocks}
$$

Now count what the block must load for that work. It needs 128 x 32 elements of A and 32 x 128 of B, both bf16:

$$
(128 \times 32 + 32 \times 128) \times 2\ \text{bytes} = 16{,}384\ \text{bytes}
$$

Here is the sequence the chapter 11 kernel runs for every k-tile:

```cuda
for (int kt = 0; kt < k_tiles; ++kt) {
  // 1. global -> register -> shared
  float4 a = *reinterpret_cast<const float4*>(&A[...]);   // load: ~hundreds of cycles
  float4 b = *reinterpret_cast<const float4*>(&B[...]);
  *reinterpret_cast<float4*>(&As[...]) = a;               // stalls here until `a` arrives
  *reinterpret_cast<float4*>(&Bs[...]) = b;
  __syncthreads();
  // 2. shared -> registers (ldmatrix) -> tensor cores (mma.sync)
  compute_tile(As, Bs, acc);
  __syncthreads();
}
```

The first line returns a register, and the store on the third line can't issue until that register holds data, so the warp waits. Then the barrier makes every warp wait for the slowest. Only after that do the tensor cores get work. For the whole length of the load, they idle.

The [Luo et al. microbenchmark paper](https://arxiv.org/abs/2402.13499) measured global-memory latency at 466.3 cycles and L2 latency at 261.5 cycles on an A100 PCIe 40GB. The 80GB card isn't measured in the sources, so treat both as close approximations. With $\hlc{L}$ the load latency, a block that serializes load and compute spends a fraction $L / (L + T)$ of its time idle:

| Where the tile comes from | $L$ (cycles) | Idle fraction $L/(L+T)$ with $T = 512$ |
|---|---|---|
| L2 hit | 261.5 | 261.5 / 773.5 = 33.8% |
| HBM | 466.3 | 466.3 / 978.3 = 47.7% |

Both rows are optimistic, because they ignore the time to move 16 KiB through a busy memory system. ==One block that serializes load and compute leaves the tensor cores idle a third to a half of the time.== A second resident block hides part of that, since its tensor work fills the gap, and the lab measures how much.

There is a second ceiling hiding in the same numbers. Every block loads its tiles from L2, even when HBM sees each value once, so the SMs together pull 16,384 bytes every 512 clocks each, or 32 bytes per clock per SM. Across 108 SMs at 1.41 GHz:

$$
32 \times 108 \times 1.41 \times 10^{9} = 4.9\ \text{TB/s}
$$

The Luo paper measured 1,853.7 to 2,007.9 bytes per clock of L2 throughput on an A100 PCIe 40GB, about 2.8 TB/s (the same paper's Table V). A 128 x 128 tile does 64 FLOP per byte it loads ($128 \times 128 / (128 + 128)$), so if that measured L2 rate held, it would cap the tile at $64 \times 2.8 \times 10^{12} \approx 179$ TFLOP/s. This is a bound from one microbenchmark on another card, not a prediction. It is one reason production kernels use 128 x 256 tiles: at 85 FLOP per byte the same arithmetic allows about 239 TFLOP/s.

> [!KEY] The loads aren't the problem; waiting for them is
> A block that issues the copy for tile $k+1$ and then computes tile $k$ doesn't shorten the load. It makes the load overlap with arithmetic the block had to do anyway. Pipelining buys time without adding bandwidth.

## Overlapping copy and compute

The section answers one question: how many tiles must be in flight to cover the latency? The idea is old (a CPU does it with prefetching) and the recipe is mechanical.

Double buffering is the smallest form. Keep two slots in shared memory. While the tensor cores read slot 0, the loads for the next tile fill slot 1. When the compute finishes, swap. Each slot is read or written, never both at once, so one barrier per iteration suffices.

With $S$ slots, the block keeps $S - 1$ copies in flight and computes on one tile. Each copy takes $L$ cycles. In steady state a new tile lands every $L / (S - 1)$ cycles, and a tile takes $T$ cycles to compute, so the loop runs at the slower of the two:

$$
\boxed{t_{\text{iter}} = \max\left(\hla{T},\ \frac{\hlb{L}}{S - 1}\right)}
$$

Reading it: once $L / (S - 1) \le T$, the tensor cores never wait and more stages buy nothing. For $T = 512$, two stages cover a latency up to 512 cycles, three cover 1024, and four cover 1536.

> [!INTUITION] Stages are a latency budget
> Each extra stage lets the block commit to one more tile of future work before it needs the first. The stage count is the number of latencies you can overlap, and it costs shared memory you could have spent on a bigger tile or a second resident block.

The 466-cycle number is the latency of an idle memory system. A GEMM fills every SM's load queue at once, so the latency each copy sees is higher, and it varies. That variation is why kernels use three or four stages although the idle-latency model says two would do. Drag the sliders and watch where the tensor-core row stops showing gaps:

```viz
12-pipeline-timeline
```

The figure is a model, not a measurement: it has no bandwidth limit, and it assumes the load of tile $t + S - 1$ starts the moment the compute of tile $t$ does. Your lab reports the real speedup.

The synchronous kernel can't be pipelined as written. A load into a register has to complete before the store to shared memory, so the thread stalls at the store, and the copy is serialized with the compute no matter where you place it. The fix isn't a clever ordering. It's a copy instruction that doesn't need the register.

## Asynchronous copies with cp.async

The section answers one question: how does a thread start a copy and keep going? Ampere's answer is `cp.async`, which moves data from global to shared memory without staging it in a register.

The whitepaper describes it as "eliminating the need for intermediate register file usage," and the copy can bypass L1. ==A thread issues the copy, the hardware tracks it, and the thread proceeds to the next instruction.== In SASS it shows up as `LDGSTS`, a name you can look for with the tools from [chapter 5](/c/05-profiling).

The PTX form, from the PTX ISA, copies 4, 8, or 16 bytes:

```ptx
cp.async.ca.shared.global [smem_addr], [gmem_addr], 16;   // cache at all levels
cp.async.cg.shared.global [smem_addr], [gmem_addr], 16;   // cache in L2 only (16 bytes only)
cp.async.cg.shared.global [smem_addr], [gmem_addr], 16, src_size;   // zero-fill the rest
```

Four rules cover the details:

- **Size.** `.ca` accepts 4, 8, or 16 bytes. `.cg` accepts only 16. Both addresses must be aligned to the copy size.
- **Cache hint.** `.ca` keeps the line in L1 too. For a GEMM tile that each block reads once, `.cg` avoids polluting L1, so the lab's kernel uses it.
- **Zero-fill.** The optional `src_size` operand says how many bytes to read. The rest of the destination becomes zero. With `src_size = 0` nothing is read at all, so a thread past the matrix edge still issues the instruction and gets a zero tile. This is how the kernel handles an M, N, or K that isn't a multiple of the tile, with no special-case code.
- **Completion.** The thread, not the instruction, tracks it. `cp.async.commit_group` closes the copies issued so far into a *group*. `cp.async.wait_group N` blocks until at most N of this thread's most recent groups are still pending, and all older ones are done. `cp.async.wait_all` is a commit followed by `wait_group 0`.

Wrappers for them take a few lines of inline assembly. This is the lab's code:

```cuda
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// Copy 16 bytes global -> shared. When `valid` is false, read 0 bytes and zero-fill.
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
  const int src_size = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :: "r"(smem_u32(dst)), "l"(src), "r"(src_size) : "memory");
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}
```

The shared-memory address has to be a 32-bit shared-window address, which `__cvta_generic_to_shared` produces. The `"memory"` clobber stops the compiler from moving shared-memory reads across the wait.

> [!WARNING] wait_group covers your copies, not the block's
> `wait_group` only tells a thread that its own copies have landed. Other threads' copies into the same tile may still be in flight, so you must follow the wait with `__syncthreads()` before any thread reads data another thread copied. Skipping the barrier gives wrong results on some runs and correct ones on others.

The copy also has a bandwidth side. The register-staged load needs registers for every in-flight byte: a thread holding 16 bytes in flight uses four. With `cp.async` the in-flight data sits in the memory system, so a block can have a whole tile of copies outstanding without a register to its name. That is what lets the pipeline be deep.

## The multistage pipeline

The section answers one question: what does the main loop look like, and where does the wait count come from? The loop has three parts, a prologue that fills the pipeline, a steady state, and a drain.

Number the k-tiles $0, 1, \dots, n-1$ and the ring slots $0, \dots, S-1$. Tile $t$ lives in slot $t \bmod S$. The schedule is:

1. **Prologue.** Issue the copies for tiles $0$ to $S - 2$, one `commit_group` each. Now $S - 1$ groups are in flight.
2. **Steady state, for each $k$ from 0 to $n - 1$.**
   - Wait until tile $k$ has landed.
   - Barrier.
   - Issue the copy for tile $k + S - 1$ into slot $(k + S - 1) \bmod S$, the slot tile $k - 1$ used. Commit.
   - Compute tile $k$.
3. **Drain.** Past the last tile there is nothing to load, so the loop issues empty commits, and a final `wait_group 0` makes sure nothing is outstanding.

The one number you have to derive is the argument of the wait. At the top of iteration $k$, the thread has committed groups for tiles $0$ through $k + S - 2$. That is $k + S - 1$ groups. Tile $k$ is the group at index $k$, and `wait_group N` guarantees everything except the newest $N$ is done, so tile $k$ is safe when $N$ covers the groups after it:

$$
\boxed{N = (k + S - 1) - (k + 1) = \hla{S - 2}}
$$

==It doesn't depend on $k$.== With four stages, the loop always waits for at most two groups pending: the groups for tiles $k+1$ and $k+2$ may still be in flight, and tile $k$ and everything before it must be done. Two stages gives `wait_group 0`, which is double buffering.

Two details make the counting work for every thread and every shape:

- **Commit even when there's nothing to load.** Near the end, tile $k + S - 1$ doesn't exist. The loop still calls `commit_group`, which creates an empty group. Without it, the count of groups behind the wait would shift, and the wait would let tile $k$ go before it landed.
- **Zero-fill covers the tails.** A tile that is partly outside the matrix is still a valid copy with a smaller `src_size`. A tile with fewer than $S - 1$ total k-tiles just has empty groups in the prologue.

Here is the lab kernel's loop, with the one-stage case shown for contrast:

```cuda
if constexpr (STAGES == 1) {
  for (int kt = 0; kt < k_tiles; ++kt) {
    load_tile(0, kt);
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();
    compute_tile(0);
    __syncthreads();            // the slot is about to be overwritten
  }
} else {
  for (int s = 0; s < STAGES - 1; ++s) {          // prologue
    if (s < k_tiles) load_tile(s, s);
    cp_async_commit();                            // commit even for an empty group
  }
  for (int kt = 0; kt < k_tiles; ++kt) {          // steady state
    cp_async_wait<STAGES - 2>();                  // tile kt has landed (this thread's copies)
    __syncthreads();                              // ...everyone's copies, and everyone is done with kt - 1
    const int next = kt + STAGES - 1;
    if (next < k_tiles) load_tile(next % STAGES, next);   // reuse the slot of tile kt - 1
    cp_async_commit();
    compute_tile(kt % STAGES);
  }
}
cp_async_wait<0>();                               // drain
```

Why is one `__syncthreads()` per iteration enough? The barrier sits between the wait and the new load. Everything before it in iteration $k$ is the wait for tile $k$, and everything before it in the previous iteration includes the compute of tile $k - 1$. So when a thread passes the barrier, every warp has finished reading slot $(k - 1) \bmod S$, and the new copy can overwrite it. The one-stage loop needs two barriers because the slot is overwritten right after it is read.

A concrete trace helps. Take $S = 4$ and a long K, so $n = 128$ k-tiles (K = 4096 with BK = 32). The prologue loads tiles 0, 1, 2. Iteration 0 waits until at most 2 groups are pending (tile 0 lands), loads tile 3 into slot 3, and computes tile 0 from slot 0. Iteration 1 waits for tile 1, loads tile 4 into slot 0, which tile 0 just freed, and computes tile 1. From then on, three loads are always in flight and one tile is computing. The following figure steps through the same loop for two, three, or four stages:

```viz
12-ring-buffer
```

Check it against the rule. At any moment after the new load is issued, $S$ slots hold one tile being computed and $S - 1$ in flight, so no slot is free. That is the whole cost of the pipeline, $S$ tiles of shared memory, and the whole benefit, $S - 1$ copies overlapped.

> [!EXAMPLE] The wait count, step by step, for four stages
> Before iteration 5 starts, the thread has committed 3 prologue groups plus 5 loop groups, 8 in all, numbered 0 to 7. Tile 5 is group 5. For it to be done, groups 0 to 5 must complete, so at most groups 6 and 7 may be pending: `wait_group 2`, which is $S - 2$. In iteration 5 the loop then commits group 8, for tile 8.

## Choosing the stage count

The section answers one question: how many stages should a kernel use? Two limits fight: the latency model asks for enough stages, and the shared memory and occupancy limit how many fit.

The lab's stage holds a 128 x 40 tile of A and a 32 x 136 tile of B in bf16. The 8 extra elements of padding per row (16 bytes) keep the `ldmatrix` reads conflict-free: with 80-byte rows (A) and 272-byte rows (B), eight consecutive rows land in eight different 16-byte positions of the 128-byte bank window. The sizes are:

| Piece | Elements | Bytes |
|---|---|---|
| A tile, $128 \times (32 + 8)$ | 5,120 | 10,240 |
| B tile, $32 \times (128 + 8)$ | 4,352 | 8,704 |
| One stage | 9,472 | 18,944 |

An A100 SM has 164 KiB (167,936 bytes) of shared memory, a block may opt in to 163 KiB (166,912 bytes), and CUDA reserves 1 KB per block (A100 whitepaper and CUDA tuning guides). Dividing:

| Blocks per SM | Shared memory per block | Most stages that fit | Arithmetic |
|---|---|---|---|
| 1 | 166,912 B | 8 | $\lfloor 166{,}912 / 18{,}944 \rfloor$ |
| 2 | 82,944 B | 4 | $\lfloor (167{,}936 / 2 - 1024) / 18{,}944 \rfloor$ |
| 3 | 54,954 B | 2 | $\lfloor (167{,}936 / 3 - 1024) / 18{,}944 \rfloor$ |

Four stages cost 75,776 bytes, and two blocks of that, with the reserved bytes, take 153,600 of the 167,936. A fifth stage pushes a block to 95,744 bytes, and only one block fits per SM.

So the stage count is a trade between two ways to hide latency. More stages hide it inside a block. More resident blocks hide it across blocks, because while one block waits, another computes. The lab kernel is built for two blocks per SM, since `__launch_bounds__(256, 2)` caps registers at 128 per thread and 4 stages leave room for two blocks. [Chapter 2's](/c/02-the-execution-model) occupancy calculation applies unchanged, and shared memory is the limit.

| Choice | What you gain | What you pay |
|---|---|---|
| 2 stages | Smallest shared-memory cost, easiest to reason about | Covers a latency up to $T$ cycles only |
| 3 to 4 stages | Covers latency up to $2T$ to $3T$, absorbs variation | 55.5 KiB to 74 KiB per block in the lab kernel |
| 5 or more | Deeper cover | Fewer resident blocks, and for BK = 32 tiles a returns curve that has flattened |

You can't read the best stage count off a formula, because the loaded latency isn't a constant. That is why Triton exposes it as `num_stages` in its autotuner and CUTLASS as a template parameter. When you write `tl.dot` in a loop in chapter 14, Triton's compiler builds this same prologue, steady state, and drain on sm_80, with `cp.async` for the loads, so `num_stages=3` there is the S in this chapter.

A larger BK also lengthens $T$ and shrinks the number of stages needed, at the cost of a bigger stage. A tile with BK = 64 takes 1,024 cycles of tensor work, so two stages cover a 466-cycle latency with room to spare, but a stage doubles to hold twice the K. The lab fixes BK = 32 so the stage count is the variable.

## The same idea in libcu++

The section answers one question: is there a portable way to write this without inline PTX? CUDA 11 added one, `cuda::memcpy_async` with `cuda::pipeline`, and the compiler lowers it to `cp.async` on sm_80.

```cuda
#include <cuda/pipeline>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// A sketch of the same loop; `tile_bytes` is a multiple of 16.
auto block = cg::this_thread_block();
__shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, STAGES> state;
auto pipe = cuda::make_pipeline(block, &state);

for (int kt = 0; kt < k_tiles; ++kt) {
  pipe.producer_acquire();                               // wait for a free stage
  cuda::memcpy_async(block, smem_slot(kt), gmem_tile(kt),
                     cuda::aligned_size_t<16>(tile_bytes), pipe);
  pipe.producer_commit();                                // closes the stage's copies

  pipe.consumer_wait();                                  // oldest stage has landed
  compute_tile(oldest_slot);
  pipe.consumer_release();                               // slot is free again
}
```

The library hides the commit and wait counting, the ring index, and the barrier, and it extends to Hopper, where the same `memcpy_async` can use TMA. Its costs are less control, since the zero-fill predicate, the choice of `.cg`, and the per-thread chunking are decided for you, and it needs the pipeline state in shared memory. The lab uses raw PTX because the point is to see the counts. Production code often uses the library or, further up, CUTLASS.

## Hopper: TMA and mbarriers

The section answers one question: what does Hopper change about the copy? The H100 SXM has the same 4-tensor-core SM layout, but twice the per-SM tensor rate, 4,096 bf16 FLOP per clock, for 989.4 TFLOP/s on 132 SMs (NVIDIA's H100 page, halved to dense). Twice the compute per byte means copies have half the time to arrive, and the instructions that issue them compete with the MMAs for the same issue slots. Hopper's answer is to move the copy off the threads entirely.

The **Tensor Memory Accelerator (TMA)** is a copy engine in each SM. One thread issues a single instruction, and the engine moves a whole tile between global and shared memory:

- **A tensor map describes the tensor.** On the host, you encode a *tensor map* with the driver API: the base address, the dimensions (1D to 5D), the strides, the box size to copy, and a swizzle mode. It lives in global memory and is passed to the kernel. The instruction `cp.async.bulk.tensor` then takes the map and the tile's coordinates, not 128 addresses.
- **One thread issues it.** The tile that took 256 threads two copies each in the lab takes one instruction. The address arithmetic, the bounds checks, and the registers that held them are gone, which is what the Hopper tuning guide means by TMA enabling warp specialization.
- **The engine handles edges and layout.** Out-of-bounds elements are zero-filled by the engine, the descriptor can request a swizzled shared-memory layout that avoids bank conflicts, and a copy from shared to global memory can apply an add, min, or max reduction on write.
- **It supports plain bulk copies.** `cp.async.bulk` moves a contiguous block, with a size that is a multiple of 16 bytes and 16-byte-aligned addresses.

A TMA copy doesn't complete through `commit_group` and `wait_group`. It completes through an **mbarrier**, a barrier object in shared memory. An mbarrier counts two things: *arrivals* (threads that call arrive) and a *transaction count* (bytes that have landed). The pattern for a tile load:

1. A thread calls `mbarrier.arrive.expect_tx` with the number of bytes the copies will deliver, for example 49,152 for a 128 x 64 tile of A and a 256 x 64 tile of B in bf16.
2. The TMA unit executes the copies and, as each byte lands, decrements the barrier's pending-byte count.
3. When the arrival count is met and the byte count is zero, the barrier's *phase* flips, and every thread that waits on that phase, with `mbarrier.try_wait.parity`, proceeds.

The phase bit is what makes a ring buffer cheap: a slot's barrier is reused every $S$ iterations, and a waiter says "wait for phase parity $p$" with $p$ flipping each lap.

> [!NOTE] mbarrier, not `__syncthreads()`
> A block-wide barrier makes every thread stop. An mbarrier lets one group of warps wait on data while another keeps computing, which is the property that warp specialization needs. In Ampere, the equivalent of the barrier is the combination of `wait_group` and `__syncthreads()`, and each thread must wait on its own copies.

The `expect_tx` count is a 32-bit unsigned byte count, and the barrier scope can be the CTA or the whole cluster (PTX ISA). Getting the byte count wrong is the classic TMA bug: expect too few and the barrier flips early, expect too many and it never flips.

## Hopper: wgmma and warpgroups

The section answers one question: what does Hopper do about the MMA instruction itself? On Ampere, `mma.sync` is a warp-level instruction whose operands sit in registers. Every `mma.sync` needs `ldmatrix` first, and a warp's fragment loads and MMAs share the same issue slots and registers.

Hopper adds `wgmma.mma_async`, executed by a *warpgroup*: four contiguous warps, 128 threads, the first warp's rank a multiple of 4 (PTX ISA). Its properties:

- **A big shape.** M is always 64. N runs from 8 to 256 in steps of 8, and K is 16 per instruction for bf16. One `m64n256k16` instruction does $64 \times 256 \times 16 = 262{,}144$ FMA, or 524,288 FLOP.
- **Operands from shared memory.** B always comes from shared memory through a descriptor, so no `ldmatrix` and no register copy. A comes from registers or from shared memory. The accumulator stays in registers, spread across the 128 threads.
- **Asynchronous.** The instruction returns as soon as it's issued. A sequence of `wgmma.fence`, one or more `wgmma.mma_async`, `wgmma.commit_group`, and `wgmma.wait_group N` brackets the work, with the same group-counting logic as `cp.async`. Touching the accumulator before the matching wait is undefined.

The arithmetic checks against a measurement. At 4,096 FLOP per SM per clock, one `m64n256k16` takes $524{,}288 / 4096 = 128$ clocks, and the Luo microbenchmark paper measured 128 cycles per instruction on an H800 PCIe ([Luo et al.](https://arxiv.org/abs/2402.13499)). The same paper measured 729.3 TFLOP/s for FP16 `m64n256k16` against a 756.5 peak, 96%, on that card. Hopper's peak is meant to be reached with `wgmma`, and the large N helps because a small instruction can't amortize its issue cost.

Because the accumulator is in registers, its size now limits the tile. A 64 x 256 fp32 accumulator is 16,384 values over 128 threads, 128 registers per thread. Two consumer warpgroups each owning a 64 x 256 slice make a 128 x 256 block tile and use half the SM's register file, which is why Hopper kernels move registers between warpgroups, as the next section shows.

## Hopper: clusters, warp specialization, and persistent kernels

The section answers one question: how do the pieces combine into a kernel that keeps the tensor cores busy? Four ideas do it, and each one builds on the previous.

**Thread block clusters.** A cluster is a group of blocks guaranteed to be scheduled at once, on SMs of one GPC (PTX ISA). The portable maximum is 8 blocks, and 16 on H100 and B200 with an opt-in. You launch with `__cluster_dims__(x, y, z)` or `cudaLaunchKernelEx`, and the grid must be a multiple of the cluster size. A block can read and write its cluster-mates' shared memory directly, *distributed shared memory*, through `mapa` or `cluster.map_shared_rank`. The Hopper blog claims about 7x faster data exchange than through global memory, and the Luo paper measured 180 cycles SM-to-SM latency on an H800 PCIe, 32% below L2.

**Multicast.** A cluster lets one TMA load deliver the same tile to several blocks. Take 128 x 256 x 64 tiles: A is 16 KiB and B is 32 KiB, 48 KiB per stage. At 4,096 FLOP per clock the tile's $2 \times 128 \times 256 \times 64 = 4{,}194{,}304$ FLOP take 1,024 clocks, so one SM pulls $49{,}152 / 1024 = 48$ bytes per clock from L2. If two blocks in a cluster compute neighboring M tiles that share the same B columns, each can fetch half of B and multicast it to both: 16 KiB for A plus 16 KiB for its half of B is 32 KiB, or 32 bytes per clock. L2 traffic drops by a third. That arithmetic is why multicast matters at Hopper's tensor rate.

**Warp specialization.** Because TMA needs one thread and `wgmma` is asynchronous, a block no longer needs every thread to do both jobs. A *producer* warp runs the loop that waits for a free slot and issues TMA. One or two *consumer* warpgroups wait for a full slot, run `wgmma`, and release the slot. The two groups communicate through pairs of mbarriers per slot, a "full" barrier the producer's copies complete, and an "empty" barrier the consumers arrive on. In pseudocode:

```cuda
// Producer (one elected thread of the producer warp):
for (int kt = 0; kt < k_tiles; ++kt) {
  wait(empty[slot], phase ^ 1);                    // a consumer released this slot
  arrive_expect_tx(full[slot], 49152);             // 16 KiB of A + 32 KiB of B
  tma_load(smem_A[slot], &map_A, kt * 64, m0, full[slot]);
  tma_load(smem_B[slot], &map_B, kt * 64, n0, full[slot]);
  advance(slot, phase);
}

// Consumer warpgroup:
for (int kt = 0; kt < k_tiles; ++kt) {
  wait(full[slot], phase);                         // both tiles have landed
  wgmma_fence();
  for (int k = 0; k < 4; ++k) wgmma(acc, smem_A[slot], smem_B[slot], k);  // 4 x k16 = BK 64
  wgmma_commit();
  wgmma_wait<0>();                                 // or 1, to keep one MMA group in flight
  arrive(empty[slot]);
  advance(slot, phase);
}
```

This is pseudocode, with CUTLASS-style helper names, not a kernel you can paste. The ring is the one from this chapter. The change is that the *roles* are split, so no instruction slot or register in the consumer's path goes to address arithmetic. The producer needs few registers and the consumers need many, so Hopper lets a warpgroup rebalance its register allocation at run time with `setmaxnreg`.

**Persistent kernels and ping-pong.** A kernel launched with one block per tile pays block launch, prologue, and epilogue every time. A *persistent* kernel launches about one block per SM and loops over tiles with a scheduler, so the pipeline stays full across tile boundaries, and the epilogue of tile $i$ overlaps with the loads of tile $i + 1$. *Ping-pong* goes one step further: two consumer warpgroups take alternate tiles, so while one runs its epilogue (the softmax in attention, the store in a GEMM), the other uses the tensor cores. The alternative, *cooperative*, has both warpgroups split a single tile. CUTLASS ships both as kernel schedules.

FlashAttention-3 applies the same two ideas, warp specialization and ping-pong scheduling, to attention, and reports up to about 740 TFLOP/s in FP16 on an H100, about 75% of peak, a 1.5x to 2.0x gain over FlashAttention-2. That is an H100 result, not a number the lab A100 can reproduce.

> [!NOTE] Stage count on Hopper
> A 128 x 256 x 64 bf16 stage is 48 KiB (16 KiB for A, 32 KiB for B). A block may use 227 KiB of shared memory on an H100, so four stages (192 KiB) fit and five (240 KiB) don't. The same shared-memory arithmetic as the A100's, with bigger tiles.

## Blackwell: tensor memory and tcgen05

The section answers one question: what is left to fix after Hopper? The accumulator. On Hopper it lives in the register file, which holds both the accumulator and everything else the consumer warps need, and the epilogue shares those registers with the next tile's mainloop.

Blackwell (sm_100) moves the accumulator into a new on-chip memory, **tensor memory (TMEM)**:

- **Size and shape.** TMEM is 128 lanes (rows) by 512 columns of 32-bit cells: $128 \times 512 \times 4 = 262{,}144$ bytes, 256 KiB per SM. It is separate from the 256 KiB register file and the shared memory.
- **Allocation.** One warp allocates it with `tcgen05.alloc`, in a power-of-two number of columns between 32 and 512, and frees it with `tcgen05.dealloc`.
- **One thread issues the MMA.** `tcgen05.mma` is issued by a single thread, runs asynchronously, and signals completion through `tcgen05.commit` and an mbarrier. The accumulator is always in TMEM. A comes from shared memory or from TMEM, and B from shared memory.
- **Bigger tiles.** For one CTA, M is 64 or 128 and N goes up to 256. A bf16 `tcgen05.mma` of $128 \times 256 \times 16$ is about twice the largest Hopper `wgmma` atom. NVIDIA reports the instruction at 2x to 4x Hopper's `wgmma` throughput, depending on the data type.
- **Data movement.** `tcgen05.ld` and `tcgen05.st` move values between TMEM and registers, and `tcgen05.cp` copies from shared memory into TMEM.

The consequence for the pipeline: the MMA warp holds no accumulator registers at all. The epilogue warps read finished tiles from TMEM with `tcgen05.ld` while the MMA warp accumulates the next tile into another region of TMEM. A $128 \times 256$ fp32 accumulator takes 256 of the 512 columns, so TMEM holds two of them, which is exactly the double buffering the epilogue needs, with no register pressure from either side. The register file goes back to being for the epilogue's math.

**2-CTA MMA.** With `cta_group::2`, a pair of CTAs on two SMs cooperatively executes one MMA with M = 128 or 256. A pair computing $256 \times 256 \times 16$ in bf16 does 2,097,152 FLOP. At 8,192 bf16 FLOP per SM per clock the pair does 16,384 per clock, so the instruction takes 128 clocks, the same as one CTA's $128 \times 256 \times 16$ on a single SM. The gain is operand traffic: each CTA stages its own 128 rows of A and half of the B tile, so each SM stages half as much of B as a single CTA computing the whole tile would.

Everything else from Hopper carries over: TMA loads, mbarrier transaction counts, producer and consumer roles (now a loader warp, an MMA warp, and epilogue warps), and clusters. The shape of the pipeline is the ring from this chapter, with the compute stage moved off the registers.

> [!WARNING] sm_100 is not sm_120
> `tcgen05` and tensor memory are on data-center Blackwell (compute capability 10.0). Consumer Blackwell parts (12.x) have a smaller shared memory (99 KB per block) and run block-scaled `mma.sync` instead. A kernel tuned for B200 won't run on them.

The chapter's lab runs on an A100, and none of the Hopper and Blackwell machinery is available there. What carries over is the idea: separate the copy from the compute, track completion explicitly, keep $S - 1$ tiles in flight, and pay for it in on-chip memory. The A100 does it with `cp.async`, registers for the accumulator, and `__syncthreads()`.

## What goes wrong

- **A wait that's one too many.** Writing `wait_group<STAGES - 1>` lets the loop read tile $k$ while it may still be in flight. The kernel computes on stale or partial data, and the result is wrong on some runs and right on others. Test with a long K and several stage counts, as the lab does.
- **A wait that's one too few.** `wait_group<STAGES - 3>` or `wait_group<0>` is correct but stalls: it waits for copies the loop could have overlapped. The result is right and the speedup is gone, and the lab shows it as a flat TFLOP/s curve.
- **No barrier after the wait.** `wait_group` covers the thread's own copies. Without `__syncthreads()`, a warp can read a tile whose other chunks another warp's thread is still copying.
- **A barrier in the wrong place.** If the new load is issued before the barrier, a fast warp overwrites the slot a slow warp is still reading. The load must come after the barrier that follows the previous compute.
- **Skipping the empty commit.** Without a commit for tiles past the end, the group count falls behind and the last $S - 1$ tiles are waited on too early.
- **Unaligned or oversized copies.** `.cg` is 16 bytes only, and both addresses must be 16-byte aligned. A row stride that isn't a multiple of 16 bytes, such as K = 1001 in bf16, breaks it, which is why the lab's wrapper pads K and N to multiples of 8.
- **Forgetting the shared-memory opt-in.** A kernel that uses more than 48 KB of dynamic shared memory fails to launch unless you call `cudaFuncSetAttribute` with `cudaFuncAttributeMaxDynamicSharedMemorySize`. Three stages of the lab tile are 56,832 bytes.
- **Too many stages.** At five stages the lab tile takes 95,744 bytes with the reserved bytes, only one block fits per SM, and a kernel can get slower while it looks more pipelined.
- **Expecting a fixed speedup.** If two blocks already share an SM, one block's wait is partly hidden by the other's compute, so the pipeline's win over one stage is smaller than the idle fractions above suggest. The prediction is a direction and a ceiling. The lab reports the measurement.

> [!RECAP]
> - A block that loads, waits, and computes idles the tensor cores for the length of the load: $L / (L + T)$, which is 34% to 48% for the lab's tile.
> - `cp.async` copies 4, 8, or 16 bytes global to shared without registers. `commit_group` closes a group, `wait_group N` waits until at most N are pending, and `src_size` zero-fills tails.
> - With S stages, a block keeps S - 1 copies in flight, waits with `wait_group S - 2`, and issues the next load after one barrier into the slot the previous tile used.
> - The stage count is set by shared memory and occupancy. The lab tile costs 18,944 bytes per stage, which allows 4 stages with two blocks per SM.
> - Hopper adds TMA (one-thread tile copies tracked by mbarrier byte counts), `wgmma` (asynchronous warpgroup MMA), clusters, warp specialization, and persistent kernels.
> - Blackwell adds tensor memory for the accumulator and a single-thread `tcgen05.mma`, with 2-CTA MMA to halve per-CTA operand traffic.

## Check your understanding

> [!QUESTION] A block's tile takes 512 cycles of tensor work and a load takes 466. How idle are the tensor cores with one stage, and what do two stages model as?
> With one stage the idle fraction is $466 / (466 + 512) = 47.7\%$. With two stages the loop runs at $\max(512, 466 / 1) = 512$ cycles per tile, so the model says the tensor cores never wait. Real loaded latencies are higher, which is why kernels use three or four.

> [!QUESTION] Why is the wait always `wait_group S - 2`, not `S - 1`?
> At the top of iteration $k$ the thread has committed $k + S - 1$ groups, and tile $k$ is group $k$. Only the groups for tiles $k + 1$ to $k + S - 2$ may stay pending, so $S - 2$ groups. `wait_group S - 1` would also leave group $k$ pending.

> [!QUESTION] The loop commits an empty group when there's no tile left to load. What breaks if you skip the commit?
> The thread's group count falls behind by one per skipped iteration. `wait_group S - 2` then counts the wrong groups, and near the end of K it lets the loop compute on a tile that hasn't landed.

> [!QUESTION] With 18,944 bytes per stage, how many stages fit when three blocks share an SM?
> Each block gets $167{,}936 / 3 = 55{,}978$ bytes, minus 1,024 reserved is 54,954, and $\lfloor 54{,}954 / 18{,}944 \rfloor = 2$ stages.

> [!QUESTION] Why does `wait_group` need a `__syncthreads()` after it?
> It only tells a thread that its own copies landed. Other threads copied other chunks of the tile, and the barrier makes every warp wait for all of them before any warp reads.

> [!QUESTION] What does an mbarrier's transaction count do that `wait_group` doesn't?
> It lets one thread announce how many bytes a TMA copy will deliver, and lets any thread wait for those bytes, without each thread tracking its own copies. That is what allows a producer warp to issue loads and consumer warps to wait on them.

> [!QUESTION] One wgmma `m64n256k16` is 524,288 FLOP. How many clocks does it take on an H100 SM at 4,096 FLOP per clock?
> $524{,}288 / 4096 = 128$ clocks, matching the 128 cycles per instruction measured on an H800 PCIe.

> [!QUESTION] Why can Blackwell double-buffer the accumulator for free when Hopper can't?
> A 128 x 256 fp32 accumulator is 128 KiB, half of Hopper's register file, so two copies would take all of it. On Blackwell it occupies 256 of tensor memory's 512 columns, so two fit, in memory the registers don't have to share.

## Lab

> [!TRY] A multistage cp.async pipeline
> The starter gives you a tensor-core GEMM, bf16 in and fp32 out, with the tile loads, the `ldmatrix` fragment loads, the MMA, and a working one-stage loop. Fill in `stage_smem_bytes`, `max_stages`, and `schedule` in Python, then write the pipelined main loop in the CUDA source. You pass when the output matches a PyTorch fp32 reference for 1 to 5 stages across awkward shapes (M = 1, K and N not multiples of 8 or of the tile, K shorter than the pipeline is deep) and the pipelined kernel runs above the 19.5 TFLOP/s FP32 CUDA-core peak and at least 0.9x as fast as one stage. Your lab reports `tflops_1stage`, `tflops_best`, `best_stages`, and `speedup`, and, for context, the same GEMM through cuBLAS. The first run compiles five kernel variants, which takes up to a minute or two.

## Further reading

- [Controlling data movement to boost performance on the NVIDIA Ampere architecture](https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/): NVIDIA's account of `cp.async`, pipelines, and the multistage mainloop.
- [PTX ISA: `cp.async`, `cp.async.bulk`, and `tcgen05`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html): the exact semantics of `commit_group` and `wait_group`, bulk and tensor copies, and tensor memory.
- [CUTLASS tutorial: mastering the Tensor Memory Accelerator](https://research.colfax-intl.com/tutorial-hopper-tma/): Colfax's walk through tensor maps, mbarrier transaction counts, and multicast.
- [CUTLASS tutorial: fast matrix multiplication with WGMMA on Hopper](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/): the `wgmma` issue model, warpgroups, and fences.
- [FlashAttention-3](https://arxiv.org/abs/2407.08608): warp specialization and ping-pong scheduling on an H100.
- [tcgen05 MMA programming guide](https://docs.nvidia.com/cutlass/4.6.0/media/docs/pythonDSL/mma_docs/tcgen05_programming.html): tensor memory, single-thread issue, and CTA pairs.
- [CUDA Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) and [Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html): shared-memory sizes, clusters, and TMA facts.
- [How to write a fast matrix multiplication from scratch with tensor cores](https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html): software pipelining on Ampere in practice.
- [GPU glossary: tensor memory](https://modal.com/gpu-glossary/device-hardware/tensor-memory): a short definition with links to the instructions.
- The companion course's [FlashAttention chapter](https://qwen.ifkash.dev/c/14-flash-attention), where the same overlap shows up in an attention kernel.
