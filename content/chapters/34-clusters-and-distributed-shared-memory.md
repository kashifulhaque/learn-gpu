---
title: Clusters and distributed shared memory
slug: 34-clusters-and-distributed-shared-memory
part: "Part 3 — Matrix multiplication"
summary: Launch a thread block cluster, address a peer block's shared memory with map_shared_rank, build a cluster histogram and reduction, and count how many clusters fit and how many L2 bytes TMA multicast saves. The kernels run only on an H100.
minutes: 100
gpu: true
objectives:
  - "Launch a kernel with __cluster_dims__ and with cudaLaunchKernelEx cluster attributes, and say why the grid must be a multiple of the cluster size."
  - "Explain why a cluster of at most 8 blocks is portable, why 16 is an opt-in, and why a large cluster lowers occupancy because every block of a cluster must fit in one GPC."
  - "Map a bin to its owning rank and offset, and map a rank to the block that cluster.map_shared_rank reaches, using the rank inside the cluster and not the block index in the grid."
  - "Write a cluster-wide histogram and a cluster reduction, and place the cluster.sync calls that keep a block's shared memory alive while its peers read or write it."
  - "Compute the L2 bytes a GEMM tile grid loads with and without TMA multicast, and state the saving as (cluster size - 1) / cluster size of the shared operand."
lab: 34-clusters-dsmem
---

# Clusters and distributed shared memory

> [!TLDR]
> - A thread block cluster is a group of blocks that the hardware schedules at the same time, inside one GPC. You launch one with `__cluster_dims__` or with a `cudaLaunchKernelEx` attribute. The portable maximum is 8 blocks, and 16 is an opt-in on an H100.
> - Blocks in a cluster can read, write, and do atomics on each other's shared memory. `cluster.map_shared_rank(ptr, rank)` turns your pointer into the same variable in the block with that rank. The rank is the index inside the cluster, not the index in the grid.
> - Two barriers keep this safe. `cluster.sync()` before a remote access guarantees the peer has started and initialized. `cluster.sync()` before exit guarantees the peer's shared memory is still alive.
> - Larger clusters lower occupancy, because all blocks of a cluster must fit in one GPC. On an assumed split of 132 SMs into 8 GPCs, clusters of 3 fit 42 times and leave 6 SMs idle.
> - With TMA multicast, one load fills the same tile in several blocks. Cluster size $c$ cuts the shared operand's L2 loads by $(c-1)/c$. A 2 by 1 cluster on a 4096 cubed bf16 GEMM cuts the total from 1.5 GiB to 1 GiB. The lab checks the model on the CPU and runs kernels only on an H100.

Every block you have launched so far lived alone. Blocks cooperate through global memory and the L2, never through each other's shared memory. That rule made kernels simple, and it costs you when two blocks want the same data or the same counter. Chapter 7's histogram flushes one private copy per block into global atomics. Chapter 12's GEMM tiles load the same B tile from L2 once per block.

A Hopper cluster adds a level between the block and the grid. A few blocks run together on neighboring SMs, and they can address each other's shared memory. This chapter builds the addressing rule, the two barriers that make it safe, a histogram and a reduction on top of it, and the byte count that explains why a Hopper GEMM uses clusters at all.

The kernels need an H100, and the lab gates them the way [chapter 26](/c/26-hopper-gemm) does. The addressing, the histogram, the reduction, the occupancy arithmetic, and the multicast byte count run in Python on any machine. This chapter states every speed as a prediction from the specs. The lab measures the real figure only when an H100 is configured.

## Before you start

You need three tools from earlier chapters, and you need them by name:

- **A private histogram.** [Chapter 7](/c/07-scan-and-histogram) gives every block its own shared-memory histogram, then adds the copies into global memory with atomics. This chapter shares one histogram across a cluster instead.
- **TMA, the mbarrier, and the Hopper roles.** [Chapter 12](/c/12-pipelining-and-async-copies) names TMA, the transaction-byte barrier, and multicast. [Chapter 26](/c/26-hopper-gemm) builds the box and the barrier count. This chapter does not rebuild either.
- **Occupancy.** [Chapter 2](/c/02-the-execution-model) counts how many blocks fit on an SM. A cluster adds one more limit to that count.

## A cluster is a group of blocks that the GPC schedules together

This section answers one question: what does the hardware guarantee about the blocks of a cluster?

Picture the grid you know, a rectangle of blocks. Cut it into small groups of adjacent blocks, say four in a row. Each group is a *cluster*. The CUDA Programming Guide says the blocks of one cluster execute on a single GPC and are scheduled simultaneously. A GPC is the group of SMs that sit physically close together, which [chapter 1](/c/01-the-gpu-at-a-glance) introduced.

The guarantee that all blocks run at once is the point. Without it, block 0 of a cluster could wait at a barrier for block 3, while block 3 waits in the queue for an SM that block 0 occupies. That is a deadlock. With a cluster, all four blocks are resident before any of them runs, so a barrier across them is safe.

Hopper (compute capability 9.0) is where clusters begin. The H100 has 8 GPCs and 132 SMs on the SXM5 part, per NVIDIA's Hopper in-depth blog. The A100 has no clusters, so the labs before this one don't use them.

> [!KEY] A cluster is co-scheduled, so cross-block barriers are safe
> All blocks of a cluster are resident together, on SMs of one GPC. That guarantee is why `cluster.sync()` can't deadlock, and why one block can read another's shared memory without asking whether the other has started.

You pick the shape when you launch. A cluster is one, two, or three dimensional, like a block. This chapter uses a one-dimensional cluster along x, so the cluster size is one number, $c$.

The hardware sets the largest $c$. The Hopper tuning guide gives a portable maximum of 8 blocks, and 16 on an H100 only if the kernel opts in with `cudaFuncAttributeNonPortableClusterSizeAllowed`. Portable means the kernel is valid on every GPU that supports clusters. A GPU with fewer SMs per GPC might not run 16.

## Launching a cluster

This section answers one question: how do you tell the hardware that a kernel runs in clusters?

There are two ways. The first fixes the cluster shape at compile time with `__cluster_dims__`, and you launch with the usual triple-chevron syntax. The Programming Guide's example puts the attribute before the function name.

```cuda
// Compile-time cluster of 4 blocks along x. Compute capability 9.0 and higher.
__global__ void __cluster_dims__(4, 1, 1) kernel(float* data);

// The grid is counted in blocks and must be a multiple of the cluster size.
kernel<<<num_blocks /* a multiple of 4 */, 256>>>(data);
```

The second way sets the shape at launch time. It is the way to choose a cluster size from a runtime value, or to launch the same kernel with two sizes. You fill a `cudaLaunchConfig_t`, add a `cudaLaunchAttributeClusterDimension` attribute, and call `cudaLaunchKernelEx`:

```cuda
cudaLaunchConfig_t config = {0};
config.gridDim = dim3{num_blocks};   // blocks, a multiple of the cluster size
config.blockDim = dim3{256};
config.dynamicSmemBytes = 0;

cudaLaunchAttribute attribute[1];
attribute[0].id = cudaLaunchAttributeClusterDimension;
attribute[0].val.clusterDim.x = 4;   // cluster size in x
attribute[0].val.clusterDim.y = 1;
attribute[0].val.clusterDim.z = 1;
config.attrs = attribute;
config.numAttrs = 1;

cudaLaunchKernelEx(&config, kernel, data);
```

If a kernel has no compile-time cluster shape, you must give one at launch. Omitting it is a launch error, per the Programming Guide.

The grid is still counted in blocks, and it must divide evenly into clusters. The Programming Guide words this as a recommendation, and Colfax's TMA tutorial reports that a grid that is not a multiple of the cluster shape fails to launch. Treat it as a requirement and round the grid up.

Two more calls matter once you leave the portable limit. To run a cluster of 9 to 16 blocks, set the opt-in attribute on the kernel before launching:

```cuda
cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
```

To find out how many clusters can be resident at once, ask the driver instead of computing it. The Hopper tuning guide recommends `cudaOccupancyMaxActiveClusters` for cluster kernels. The next section explains why the answer is not simple division.

> [!NOTE] Where the numbers in this chapter come from
> The 8 and the 16 are from the Hopper tuning guide. The `__cluster_dims__` and `cudaLaunchKernelEx` forms are from the CUDA Programming Guide. The Programming Guide pages this chapter read don't say that a grid that isn't a multiple of the cluster size fails to launch, so the chapter takes that statement from Colfax's tutorial and marks it as reported rather than confirmed.

## Why large clusters lower occupancy

This section answers one question: how many clusters of size $c$ are resident at once, and why does that number shrink faster than $1/c$?

Picture a GPC as a row of SMs. A cluster needs $c$ block slots, and all of them must be in the same row. If the row has 16 SMs and each SM holds one block, a cluster of 3 takes 3 slots, and five clusters take 15. The sixteenth SM has one slot left, which is less than a cluster needs. It sits idle.

The arithmetic runs per GPC, so you round down once per GPC, not once for the whole GPU. A GPC with $S$ SMs and $q$ resident blocks per SM has $S \cdot q$ slots, and it holds $\lfloor S\,q / c \rfloor$ clusters:

$$
N_{\text{clusters}} = \sum_{g=1}^{G} \left\lfloor \frac{\hla{S_g}\,\hlc{q}}{\hlb{c}} \right\rfloor
$$

$\hla{S_g}$ is the SM count of GPC $g$, $\hlc{q}$ is the blocks per SM that the block's threads, registers, and shared memory allow, and $\hlb{c}$ is the cluster size. The floor is inside the sum. That is the whole difference from dividing the GPU's total slots by $c$.

NVIDIA publishes 8 GPCs and 132 SMs for the H100 SXM5. It doesn't publish how many SMs each GPC keeps. This chapter assumes six GPCs of 16 SMs and two of 18, which sums to 132. The split is an assumption, and the real counts come from `cudaOccupancyMaxActiveClusters`. The model also assumes that two blocks of one cluster may share an SM when the resources allow, which this chapter did not confirm in a source.

> [!EXAMPLE] Clusters of 3 on the assumed split, one block per SM
> A GPC of 16 SMs holds $\lfloor 16/3 \rfloor = 5$ clusters, using 15 SMs. A GPC of 18 holds $\lfloor 18/3 \rfloor = 6$ clusters, using all 18. The GPU holds $6 \times 5 + 2 \times 6 = 42$ clusters, which is 126 blocks. Dividing 132 slots by 3 gives 44, which is wrong by two clusters. Six SMs are idle.

The loss depends on how $c$ divides the GPC. The following table is the same arithmetic for other sizes, on the same assumed split, at one resident block per SM. Sizes above 8 need the non-portable opt-in.

| Cluster size | Clusters resident | Blocks resident | Idle SMs |
|---|---|---|---|
| 1 | 132 | 132 | 0 |
| 2 | 66 | 132 | 0 |
| 3 | 42 | 126 | 6 |
| 4 | 32 | 128 | 4 |
| 8 | 16 | 128 | 4 |
| 16 | 8 | 128 | 4 |

A size of 6 holds 18 clusters and leaves 24 SMs idle, and a size of 10 holds 8 and leaves 52 idle. Divisors of the GPC size lose little, and sizes that don't divide it lose much. Hopper's published portable and non-portable maximums give the sizes you can ask for, and the occupancy query tells you the sizes that are worth asking for.

```viz
34-cluster-fit
```

A cluster also ties its blocks' lifetimes together. With the exit barrier from the section on barriers below, no block of a cluster exits before its slowest block arrives, so a kernel whose blocks do uneven work leaves more SMs waiting with $c = 16$ than with $c = 2$.

> [!NOTE] Blocks per SM still matters
> The $q$ in the formula comes from the usual limits: 2,048 threads, 64K registers, 228 KiB of shared memory with 1 KiB reserved per block, and 32 blocks, on compute capability 9.0. A cluster kernel that asks for 100 KiB of shared memory per block gets $\lfloor 233{,}472 / 103{,}424 \rfloor = 2$ resident blocks per SM, and the formula doubles the slots.

## Addressing distributed shared memory

This section answers one question: given a block and a target rank, which block's shared memory does a pointer reach?

Picture clusters of 4 along x in a grid of 12 blocks. Blocks 0 to 3 are cluster 0, blocks 4 to 7 are cluster 1, and blocks 8 to 11 are cluster 2. Inside each cluster the blocks have *ranks* 0 to 3. Block 5 is rank 1 of cluster 1.

A pointer to your own `__shared__` variable names the copy in your block. Cooperative Groups lets you ask for the same variable in a peer, by rank. The call is `cluster.map_shared_rank(ptr, target_rank)`. It returns a pointer you can load from, store to, or pass to `atomicAdd`, in the shared memory of the block with that rank. Underneath, it is the PTX `mapa` instruction, which maps a shared-memory address to a rank.

The rank is relative to your cluster. Block 5, asking for rank 2, reaches block 6, not block 2. Take the first block of your own cluster, then add the rank:

$$
\boxed{\text{peer} = \bigl(\hla{b} - (\hla{b} \bmod \hlb{c})\bigr) + \hlc{r}}
$$

$\hla{b}$ is your block index in the grid, $\hlb{c}$ is the cluster size, and $\hlc{r}$ is the target rank. The bracketed part is the index of the first block of your cluster. Your own rank is $\hla{b} \bmod \hlb{c}$, which is what `cluster.block_rank()` returns for a one-dimensional cluster.

```cuda
namespace cg = cooperative_groups;

cg::cluster_group cluster = cg::this_cluster();
unsigned rank = cluster.block_rank();                  // blockIdx.x % cluster_size
int* mine = smem;                                      // my copy
int* theirs = cluster.map_shared_rank(smem, 2);        // rank 2's copy of smem
```

> [!WARNING] The rank is not a grid index
> A kernel that computes the target block as `target_rank` reaches block 2 from block 5, which belongs to another cluster. In the kernel you never compute a grid index at all. You pass the rank to `map_shared_rank`, and the hardware resolves it inside your cluster. The lab's `peer_block` function names the grid index so you can check your rank arithmetic on the CPU.

`this_cluster()` needs compute capability 9.0 or greater, and on a non-cluster launch it assumes a 1 by 1 by 1 cluster, per the Programming Guide's Cooperative Groups table.

### How fast is a peer's shared memory?

Luo and colleagues measured distributed shared memory on an H800 PCIe, a part with the same 114 SMs as the H100 PCIe. SM-to-SM latency was 180 cycles, 32% below the 263-cycle L2 latency on the same card. Peak SM-to-SM throughput was about 3.27 TB/s at a cluster size of 2 and 2.65 TB/s at a cluster size of 4. NVIDIA's Hopper blog claims that DSMEM accelerates data exchange between blocks by about 7x compared with global memory.

Quote these as measurements on an H800 PCIe and a vendor claim, not as H100 SXM5 figures. They tell you the order of magnitude. On that card, shared memory takes 29 cycles, a peer's shared memory 180, and the L2 263.

The tuning guide adds one rule for access patterns. Keep distributed shared memory accesses coalesced and aligned to 32-byte segments, and avoid non-unit strides where you can. A histogram's atomics are scattered, so the histogram below runs on atomic throughput, not on that rule.

## Two barriers keep it safe

This section answers one question: when is it safe to touch a peer's shared memory, and when is it safe to exit?

A remote access is safe only while the peer block is alive and its shared memory is initialized. Two events bound that window. The peer must have started and zeroed or filled its data. The peer must not have exited. Each needs a barrier.

The cluster barrier is `cluster.sync()`. Every thread of every block in the cluster must arrive before any of them continues, and shared-memory writes before the barrier are visible to every block after it. The Programming Guide's split form, `barrier_arrive()` and `barrier_wait()`, lets a block do local work between the two halves to hide the latency.

```cuda
__shared__ int smem[N];
cg::cluster_group cluster = cg::this_cluster();

init(smem);          // zero or fill my own shared memory
cluster.sync();      // barrier 1: every block started and initialized

use_remote(cluster); // read, write, or atomicAdd on peers' shared memory

cluster.sync();      // barrier 2: every remote access finished; nobody exits early
```

The Programming Guide states the second rule directly: you must ensure that all distributed shared memory operations happen before a block exits. A block that returns frees its shared memory. A peer that writes to it afterward writes to nothing, or to the next block that lands on that SM.

Here is the failure with real numbers. Rank 0 finishes at step 3, rank 2 at step 5, rank 3 at step 6, and rank 1 at step 9. Rank 1 sends one update to rank 0 at step 8 and rank 2 at step 9. Without a final barrier, rank 0 has exited at step 3 and rank 2 at step 5, so both updates are lost. With the barrier, every block exits at step 9, the step the slowest block arrives, and no update is lost. The lab's `exit_steps` function is this rule.

> [!WARNING] A missing exit barrier fails silently on some runs
> The kernel returns the right answer when the peers happen to finish at the same time. It loses counts when one block is slow. The bug depends on timing, so a small test often passes. [Chapter 27](/c/27-debugging-gpu-kernels) covers tools for races. The cheap defense is to write the second `cluster.sync()` first, before you write the loop that needs it.

## A cluster-wide histogram

This section answers one question: how does a cluster count into bins that don't fit in one block's shared memory?

Start with what you know from [chapter 7](/c/07-scan-and-histogram). A global histogram does one atomic per element, and every block fights over the same hot addresses in L2. A private histogram gives each block its own copy in shared memory, counts into it with fast atomics, and adds the copy into global memory at the end. The private copy has to fit in one block's shared memory, at most 227 KiB per block on compute capability 9.0.

A cluster raises that limit. Spread one histogram across the $c$ blocks of a cluster. Each rank owns a contiguous slice of the bins in its own shared memory, and any block adds to any bin by reaching the owner's slice. The cluster's histogram is $c$ times larger than one block's.

The rule that picks the owner is a division:

$$
\hla{B_r} = \left\lceil \frac{N_{\text{bins}}}{\hlb{c}} \right\rceil,
\qquad
\text{owner}(i) = \left\lfloor \frac{i}{\hla{B_r}} \right\rfloor,
\qquad
\text{offset}(i) = i \bmod \hla{B_r}
$$

$\hla{B_r}$ is the number of bins per rank, $N_{\text{bins}}$ is the bin count, and $\hlb{c}$ is the cluster size. Bin $i$ lives at `smem[offset]` of the block with rank $\text{owner}(i)$.

> [!EXAMPLE] 1,001 bins in clusters of 4
> $B_r = \lceil 1001/4 \rceil = 251$. Bin 1,000 belongs to rank $\lfloor 1000/251 \rfloor = 3$ at offset $1000 - 753 = 247$. Bin 250 is rank 0 at offset 250. Bin 251 is rank 1 at offset 0. Rank 3 holds bins 753 to 1,000 at offsets 0 to 247, so its slots 248 to 250 never hold a bin. The `bin < num_bins` guard in the flush covers them.

Each block loops over its share of the input and does one atomic per element, on a pointer from `map_shared_rank`. The two barriers from the last section surround the loop. After the second barrier, every slice is complete, so each block flushes its own slice to global memory.

```cuda
constexpr int CLUSTER = 4;
constexpr int BINS_PER_RANK_MAX = 4096;   // 16 KiB of int32 per block

__global__ void __cluster_dims__(CLUSTER, 1, 1)
cluster_hist(const int* x, int64_t n, int num_bins, int* out) {
    __shared__ int smem[BINS_PER_RANK_MAX];
    cg::cluster_group cluster = cg::this_cluster();
    const unsigned rank = cluster.block_rank();
    const int bins_per_rank = (num_bins + CLUSTER - 1) / CLUSTER;

    for (int i = threadIdx.x; i < bins_per_rank; i += blockDim.x) smem[i] = 0;
    cluster.sync();                       // every slice is zeroed and live

    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const int b = x[i];
        const unsigned owner = b / bins_per_rank;     // a rank in the cluster
        const int offset = b - owner * bins_per_rank;
        atomicAdd(cluster.map_shared_rank(smem, owner) + offset, 1);
    }
    cluster.sync();                       // every remote add landed; nobody exits early

    for (int i = threadIdx.x; i < bins_per_rank; i += blockDim.x) {
        const int bin = rank * bins_per_rank + i;
        if (bin < num_bins && smem[i] != 0) atomicAdd(&out[bin], smem[i]);
    }
}
```

Count what moved, for an input of $2^{24}$ elements and 1,024 bins, in a grid of 1,056 blocks. A global histogram does 16,777,216 atomics to L2. A cluster histogram does the same number of updates, all to shared memory, local or remote, plus the flush. Each cluster flushes its $N_{\text{bins}}$ bins once, so clusters of 4 do $264 \times 1024 = 270{,}336$ global atomics. One private copy per block, as in chapter 7, would do $1056 \times 1024 = 1{,}081{,}344$. The cluster cuts the flush by a factor of $c$.

The capacity gain is larger than the flush gain. A block can hold $\lfloor 232{,}448 / 4 \rfloor = 58{,}112$ four-byte counters in 227 KiB. A cluster of 8 holds $8 \times 58{,}112 = 464{,}896$. A histogram with 64K bins needs 256 KiB, which no single block can hold. A cluster of 2 holds it at 128 KiB per block. That is the case where a cluster histogram replaces a global one.

Two costs come with it. A fraction of the updates cross to a peer, and a remote access has a higher latency than your own shared memory, per the H800 figures above. For uniformly distributed values, the fraction that goes to a peer is $(c-1)/c$: 75% for $c = 4$. The lab's simulator counts local and remote updates so you can see it.

Skewed input is the second cost. Take values drawn as $\lfloor 1024\,u^4 \rfloor$ for uniform $u$, the skewed input the lab uses. The chance that a value lands below bin 256 is $(1/4)^{1/4} = 0.7071$. Rank 0, with the first 256 bins, receives 70.7% of the updates. Ranks 1, 2, and 3 receive 13.4%, 9.0%, and 6.9%. All the contention moves from the L2's atomic units to one block's shared memory, where it is cheaper but not free. A layout that interleaves bins across ranks, `owner = bin % c`, spreads a low-bin hotspot, at the cost of a modulo and a different offset. The lab uses contiguous slices, and the chapter marks the interleaved layout as an exercise you can try.

> [!NOTE] This histogram is a prediction, not a measurement
> The cluster version should beat global atomics when the bins are many or the data is skewed enough to serialize global atomics. It should not when there are few bins and the data is uniform, because a block's own private histogram already does that job. The lab reports the ratio as a metric when an H100 runs it, and doesn't grade it.

## A cluster reduction

This section answers one question: how do the blocks of a cluster combine their partial sums without every block touching global memory?

Each block reduces its slice of the input to one number, its *partial*, using the warp and block reduction from [chapter 6](/c/06-reductions). The usual ending is one global atomic per block. A cluster replaces that with one global atomic per cluster, by combining the partials through distributed shared memory first.

Picture eight blocks, ranks 0 to 7, each holding a partial in `smem`. At stride 1, ranks 0, 2, 4, and 6 each add the partial of the rank one above. At stride 2, ranks 0 and 4 add ranks 2 and 6. At stride 4, rank 0 adds rank 4. Rank 0 now holds the sum of all eight. That is $\log_2 8 = 3$ levels.

A rank takes part at stride $s$ when $r \bmod 2s = 0$, and it reads rank $r + s$ only if $r + s < c$. The bound matters for clusters whose size isn't a power of two. For $c = 5$ the levels are strides 1, 2, and 4, and at stride 4 rank 0 reads rank 4, which exists, while ranks 1 to 3 never read. The number of levels is $\lceil \log_2 c \rceil$.

A cluster barrier follows each level, because a rank must not read a partial that its owner is still updating. The count is one barrier to publish the partials, plus one per level, so $1 + \lceil \log_2 c \rceil$ barriers. The last one doubles as the exit barrier, since the peers must stay alive until rank 0 has finished reading them.

The lab's kernel takes the simpler route for one number per block: rank 0 reads all $c$ partials in a loop. For $c = 8$ that is seven remote reads by one thread, and it needs two barriers, the publish and the exit. The tree uses more barriers and spreads the reads across ranks. It starts to matter when each rank holds a whole vector. The lab's Python simulator models the tree, and the lab doesn't measure which one wins.

```cuda
__global__ void __cluster_dims__(8, 1, 1)
cluster_reduce(const float* x, int64_t n, float* out) {
    __shared__ float partial;
    cg::cluster_group cluster = cg::this_cluster();

    float v = 0.0f;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) v += x[i];
    v = block_sum(v);                       // warp shuffles, then one warp over warp sums
    if (threadIdx.x == 0) partial = v;

    cluster.sync();                         // publish: every partial is written
    if (cluster.block_rank() == 0 && threadIdx.x == 0) {
        float total = 0.0f;
        for (unsigned r = 0; r < cluster.num_blocks(); ++r)
            total += *cluster.map_shared_rank(&partial, r);
        atomicAdd(out, total);              // one global atomic per cluster
    }
    cluster.sync();                         // peers stay alive until rank 0 is done
}
```

For a grid of 1,056 blocks, a per-block atomic does 1,056 global float atomics, and the cluster version does $1056/8 = 132$. The reduction in atomics is $c$. The input still crosses HBM once, so a large reduction stays bandwidth bound, and the saved atomics matter mostly when the grid is small or the partials are large. The lab's metric is the throughput in GB/s of the cluster kernel, and it prints the per-block-atomic time beside it.

A float atomic's order isn't fixed, so two runs can differ in the last bits. The lab compares against an fp64 sum with a relative tolerance of $10^{-4}$, which is loose enough for the order and tight enough to catch a lost block.

## TMA multicast: one load for several blocks

This section answers one question: how many L2 bytes does a GEMM tile grid save when the blocks of a cluster share operands?

Picture the output of a GEMM as a grid of tiles. Block $(i, j)$ computes tile $(i, j)$. To do it, the block loads row-slice $i$ of $A$ and column-slice $j$ of $B$ for the whole $K$ dimension. Block $(i, j+1)$, next to it, loads the same slice of $A$. Without clusters, each of them fetches it from L2 separately.

[Chapter 12](/c/12-pipelining-and-async-copies) introduced the fix. In a cluster, one TMA load can deliver the same tile into the shared memory of several blocks. You issue the load once, with a bitmask that names the destination blocks, and the copy engine writes the tile into each one. Colfax's tutorial describes the mask as a 16-bit value where each set bit marks a block that receives the data, and it adds that only blocks in one cluster can take part in a multicast.

Take a 4096 by 4096 by 4096 bf16 GEMM with $128 \times 256$ output tiles. The grid has $32 \times 16 = 512$ blocks. Each block loads a $128 \times 4096$ slice of $A$, which is 1 MiB at 2 bytes per element, and a $4096 \times 256$ slice of $B$, which is 2 MiB. That is 3 MiB per block and $512 \times 3 = 1536$ MiB for the grid.

Now form clusters of $c_m$ blocks along $M$ and $c_n$ along $N$. The $c_n$ blocks of a cluster row share one $A$ slice. The $c_m$ blocks of a cluster column share one $B$ slice. With multicast, each distinct slice is loaded once for the cluster:

$$
\text{bytes} = \underbrace{\frac{\hla{A_{\text{tot}}}}{c_n}}_{A\text{ slices}} + \underbrace{\frac{\hlc{B_{\text{tot}}}}{c_m}}_{B\text{ slices}}
$$

$\hla{A_{\text{tot}}}$ and $\hlc{B_{\text{tot}}}$ are the bytes the grid loads without multicast, 512 MiB and 1024 MiB here. The saving on each operand is $(c-1)/c$ with $c$ the number of blocks that share it:

$$
\boxed{\text{saved} = A_{\text{tot}}\,\frac{c_n - 1}{c_n} + B_{\text{tot}}\,\frac{c_m - 1}{c_m}}
$$

The saving is a fraction of the *shared operand's* loads, not of the whole. A cluster of 2 along $M$ shares $B$ and nothing else. It halves $B$'s 1024 MiB and leaves $A$'s 512 MiB. The total drops from 1536 to 1024 MiB, which is a third.

| Cluster ($c_m$ by $c_n$) | A loaded | B loaded | Total | Saved |
|---|---|---|---|---|
| 1 by 1 | 512 MiB | 1024 MiB | 1536 MiB | 0 |
| 2 by 1 | 512 MiB | 512 MiB | 1024 MiB | 33.3% |
| 2 by 2 | 256 MiB | 512 MiB | 768 MiB | 50% |
| 4 by 1 | 512 MiB | 256 MiB | 768 MiB | 50% |
| 4 by 2 | 256 MiB | 256 MiB | 512 MiB | 66.7% |
| 4 by 4 | 128 MiB | 256 MiB | 384 MiB | 75% |

```viz
34-multicast-traffic
```

The 2 by 1 row matches chapter 12's per-block arithmetic. There, a block pulls 48 KiB per stage and one that fetches half of $B$ pulls 32 KiB, a third less. The grid total in this chapter is the same saving counted over all 512 blocks.

Why does a third of L2 traffic matter? Chapter 12 derived that one SM at the full bf16 tensor rate pulls 48 bytes per clock for these tiles. On 132 SMs at the implied tensor clock of 1.83 GHz, that is $132 \times 48 \times 1.83 \times 10^{9} = 11.6$ TB/s. The FlashAttention-3 paper's hardware table lists the H100 SXM5 L2 at 12 TB/s. The demand is close to the L2's limit. A 2 by 1 multicast lowers it to $132 \times 32 \times 1.83 \times 10^{9} = 7.7$ TB/s. These are predictions from the specs and from a cited table. The lab doesn't run a GEMM, so the chapter doesn't claim a speedup.

The edge of a grid that isn't a multiple of the cluster needs care. With 3 tiles along $M$ and clusters of 2, the second cluster has one tile, so it has nothing to share. The lab's `multicast_l2_bytes` counts the distinct slices per cluster, which handles that edge. A formula that divides the total by $c$ everywhere gets it wrong.

> [!KEY] Multicast saves the shared operand, not the whole
> A cluster of $c$ blocks that share an operand loads it once instead of $c$ times, a saving of $(c-1)/c$ of that operand's bytes. The other operand is unchanged. Pick the cluster shape that shares the larger operand first.

## Who uses clusters

This section answers one question: where do you meet clusters in kernels you didn't write?

CUTLASS 3's Hopper GEMMs take a cluster shape as a template parameter, and its TMA multicast copies take a multicast mask. Colfax's TMA tutorial walks through the CuTe pieces: a `SM90_TMA_LOAD_MULTICAST` copy atom, `block_rank_in_cluster()`, and `cluster_sync()`. Each block slices the tile by its rank, so a single multicast fills both blocks' shared memory. The tutorial's example uses a cluster of 2 along one axis. The CUTLASS pipeline documentation lists thread block cluster synchronization and query APIs among its abstractions.

FlashAttention-3 targets the H100. Its paper lists clusters in its hardware hierarchy table, and says that blocks of a cluster are co-scheduled on one GPC. A search of the paper's text finds no mention of multicast or distributed shared memory, so this chapter doesn't claim that FlashAttention-3 uses them. The paper's gains come from warp specialization, overlapping softmax with the matrix multiply, and FP8, as [chapter 16](/c/16-flash-attention) describes. Whether a particular FlashAttention-3 kernel launches with a cluster shape is in its source code, which this chapter did not read.

Distributed shared memory appears in attention elsewhere. [Chapter 16](/c/16-flash-attention) summarizes a Blackwell backward pass that exchanges half of $dS$ through distributed shared memory so that each CTA owns half the rows and the global atomic adds halve. That is the cluster histogram's idea, applied to a reduction over gradients, on a different part.

A rule of thumb follows from this chapter. Reach for a cluster when blocks share an operand, as in multicast, or a counter or a reduction, as in the histogram. Skip it when blocks are independent, because the occupancy cost is real and a cluster adds nothing.

## What goes wrong

- **A grid index used as a rank.** The target of `map_shared_rank` is a rank inside your cluster. Computing `target_rank` as a grid index reaches block 2 from block 5, in another cluster. The lab's `global_index_as_rank` returns `target_rank` for `peer_block`, and the correct answer for block 5, cluster size 4, rank 2 is block 6.
- **No barrier before exit.** A block that finishes early frees its shared memory while a slower peer still writes to it. The update is lost, and the kernel passes whenever the timing is kind. The lab's `forgot_final_sync` lets every block exit at its own step.
- **No barrier before the first remote access.** A peer that has not started has not zeroed its slice. Your add lands on garbage, and the later zeroing erases it. The first `cluster.sync()` after initialization prevents it.
- **A grid that isn't a multiple of the cluster.** The launch fails, or does not run as intended. Round the grid up and guard the tail with the usual bounds check.
- **A size above 8 without the opt-in.** The launch fails on an H100 unless the kernel sets `cudaFuncAttributeNonPortableClusterSizeAllowed`. Even then, 16 may leave SMs idle, and a cluster size that doesn't divide the GPC loses more.
- **Total slots divided by the cluster size.** On the assumed 132-SM split, clusters of 3 fit 42 times. The lab's `total_slots_over_cluster` returns 44, because it ignores that a cluster stays inside one GPC.
- **Dividing both operands by the cluster size.** A 2 by 1 cluster saves 512 MiB of 1536, a third. The lab's `divide_everything_by_cluster` returns 768 MiB saved, half.
- **A bin index that doesn't fit the owner's slice.** A value at or above $N_{\text{bins}}$ computes an owner past the last rank, and `map_shared_rank` with a rank outside the cluster is invalid. Clamp or validate the input, as the lab's kernel assumes.

> [!RECAP]
> - A cluster is co-scheduled inside one GPC. You launch it with `__cluster_dims__` or a `cudaLaunchKernelEx` attribute, with a grid that is a multiple of the cluster size. The portable maximum is 8, and 16 is an opt-in.
> - A large cluster lowers occupancy because the floor applies per GPC. On an assumed 8-GPC, 132-SM split, clusters of 3 fit 42 times and leave 6 SMs idle.
> - `map_shared_rank(ptr, rank)` reaches the peer with that rank in your cluster. The grid index of that peer is the first block of your cluster plus the rank.
> - Two `cluster.sync()` calls bracket remote access: one so peers are initialized, and one so no block exits while a peer still accesses it.
> - A cluster histogram holds $c$ times the bins of one block and cuts the global flush by $c$. A cluster reduction does one global atomic per cluster. TMA multicast saves $(c-1)/c$ of the shared operand's L2 loads.

## Check your understanding

> [!QUESTION] Block 13 is in clusters of 8. What are its rank, and the grid index of the block that `map_shared_rank(ptr, 0)` reaches?
> The rank is $13 \bmod 8 = 5$. The first block of its cluster is $13 - 5 = 8$, so rank 0 is block 8. A rank-as-index mistake returns block 0, which is in cluster 0.

> [!QUESTION] A histogram has 1,001 bins and clusters of 4. Which rank owns bin 1,000, and at what offset?
> $B_r = \lceil 1001/4 \rceil = 251$. The owner is $\lfloor 1000/251 \rfloor = 3$ and the offset is $1000 - 3 \times 251 = 247$.

> [!QUESTION] Rank 1 sends an update to rank 0 at step 8. Rank 0 finishes at step 3 and has no final barrier. What happens, and what does the barrier change?
> Rank 0 has exited at step 3, so its shared memory is gone and the update is lost. A final `cluster.sync()` holds rank 0 until the slowest block, here rank 1, arrives, so no block exits before step 9 and the update lands.

> [!QUESTION] How many clusters of 3 fit on six GPCs of 16 SMs and two of 18, at one block per SM, and why isn't the answer 132 / 3?
> Each GPC floors separately: $6 \times \lfloor 16/3 \rfloor + 2 \times \lfloor 18/3 \rfloor = 30 + 12 = 42$. Dividing 132 by 3 gives 44, which treats the leftover SM of each 16-SM GPC as part of a cluster. A cluster can't span two GPCs, so six SMs are idle.

> [!QUESTION] A cluster reduction has 5 blocks per cluster. How many levels and how many `cluster.sync()` calls does the tree take?
> The levels are strides 1, 2, and 4, so $\lceil \log_2 5 \rceil = 3$. Rank 0 reads rank 4 at stride 4, and ranks 1 to 3 never read. The barriers are one to publish plus one per level, which is 4.

> [!QUESTION] A 4096 cubed bf16 GEMM uses $128 \times 256$ tiles and clusters of 2 by 2. How many MiB does multicast load, and what is the saving?
> A slices: $512/2 = 256$ MiB. B slices: $1024/2 = 512$ MiB. The total is 768 MiB against 1536 MiB, a saving of half. The saving on each operand is $(2-1)/2$.

> [!QUESTION] Why is $(c-1)/c$ not the saving of the whole GEMM's L2 traffic?
> It applies to the operand that the cluster shares. A 2 by 1 cluster shares only B. It saves half of B's 1024 MiB, which is 512 MiB out of a 1536 MiB total, a third.

## Lab

> [!TRY] Build a cluster in Python, then run it on an H100
> Open `34-clusters-dsmem`. `cluster_rank` is the one function the first check needs. Then implement `peer_block`, `bin_owner`, `simulate_dsmem_histogram`, `exit_steps`, `simulate_cluster_reduction`, `blocks_per_sm` with `max_active_clusters`, and `multicast_l2_bytes`. The histogram check compares your simulator with `torch.bincount` on four shapes, including a bin count that clusters don't divide. The last four functions are real sm_90 kernels that compile only for an H100.

The first eight checks run on any machine, including a laptop with no GPU. They catch four plausible mistakes: the rank used as a grid index, the missing exit barrier, total slots divided by the cluster size, and both operands divided by the cluster size. The two kernel checks, the cluster histogram against `torch.bincount` and the cluster reduction against an fp64 sum, need an H100. A machine without one prints a skip message, and the job is not sent to the A100 endpoint.

The lab reports two metrics when an H100 runs the kernels:

- `hist_speedup_vs_global_atomics`: the time of the global-atomic histogram divided by the time of the cluster histogram, on a skewed input of $2^{24}$ values in 1,024 bins. The metric is a record, not a threshold.
- `reduce_gbs`: the bytes of $2^{26}$ floats divided by the cluster reduction's time. The check prints the one-atomic-per-block time beside it. A machine without an H100 records `skipped (no H100)` for both.

## Further reading

- [CUDA Programming Guide, thread block clusters](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html), NVIDIA. The statement that all blocks of a cluster run on one GPC and are scheduled simultaneously, and that clusters need compute capability 9.0.
- [CUDA Programming Guide, C++ language extensions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html), NVIDIA. `__cluster_dims__`, and the `cudaLaunchKernelEx` example with `cudaLaunchAttributeClusterDimension`.
- [CUDA Programming Guide, Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html), NVIDIA. `this_cluster()` and the `barrier_arrive` and `barrier_wait` split.
- [CUDA C++ Programming Guide 12.4, distributed shared memory](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html), NVIDIA. Section 3.2.5 is the source for `map_shared_rank` and for the rule that all distributed shared memory operations finish before a block exits.
- [CUDA Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html), NVIDIA. The portable maximum of 8, the opt-in to 16, the advice to use `cudaOccupancyMaxActiveClusters`, and the access-pattern rule for distributed shared memory.
- [NVIDIA Hopper architecture in depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/), NVIDIA. 8 GPCs and 132 SMs on the H100 SXM5, and the claim that distributed shared memory exchanges data about 7x faster than global memory.
- [Benchmarking and dissecting the Nvidia Hopper GPU architecture](https://arxiv.org/abs/2402.13499), Luo and others. The 180-cycle SM-to-SM latency and the 3.27 TB/s throughput at cluster size 2, measured on an H800 PCIe.
- [CUTLASS tutorial: mastering the Tensor Memory Accelerator](https://research.colfax-intl.com/tutorial-hopper-tma/), Colfax Research. TMA multicast, the multicast mask, and the cluster-shape constraint on the grid.
- [FlashAttention-3](https://arxiv.org/abs/2407.08608), Shah and others. The H100 SXM5 hierarchy table with the 12 TB/s L2 figure.
- [GPU MODE, lecture 36: CUTLASS and FlashAttention-3](https://github.com/gpu-mode/lectures/tree/main/lecture_036), Jay Shah. The lecture index titles this session CUTLASS and FlashAttention-3. This chapter did not check whether the slides discuss clusters.
- [Chapter 12](/c/12-pipelining-and-async-copies) for the multicast arithmetic per block, and [chapter 26](/c/26-hopper-gemm) for the TMA box and barrier count that a multicast load builds on.
