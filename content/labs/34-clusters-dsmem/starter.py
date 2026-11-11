"""Lab 34: a thread block cluster, modeled in Python and run on an H100.

Do the CPU pieces first. They model how blocks in a cluster address each
other's shared memory, how a histogram and a reduction use it, how many
clusters fit on the GPU, and what TMA multicast saves. The last four
functions are real sm_90 kernels. A machine without an H100 skips them.

1. cluster_rank
2. peer_block
3. bin_owner
4. simulate_dsmem_histogram
5. exit_steps
6. simulate_cluster_reduction
7. blocks_per_sm and max_active_clusters
8. multicast_l2_bytes
9. cluster_histogram, global_atomic_histogram, cluster_reduce, block_atomic_reduce
"""

from __future__ import annotations

# Compute capability 9.0 limits.
MAX_THREADS_PER_SM = 2048
REGISTERS_PER_SM = 65536
SMEM_PER_SM = 228 * 1024
SMEM_PER_BLOCK_MAX = 227 * 1024
SMEM_RESERVED_PER_BLOCK = 1024
MAX_BLOCKS_PER_SM = 32
MAX_PORTABLE_CLUSTER = 8
MAX_NONPORTABLE_CLUSTER = 16

# The H100 SXM5 has 8 GPCs and 132 SMs. The per-GPC split is an assumption
# that sums to 132.
H100_SXM5_GPC_SMS = (16, 16, 16, 16, 16, 16, 18, 18)


def cluster_rank(block_idx: int, cluster_size: int) -> int:
    """Rank of a block inside its cluster: `cluster.block_rank()` for a 1D cluster.

    Blocks 0..3 of a grid of clusters of 4 are ranks 0..3 of cluster 0, and
    blocks 4..7 are ranks 0..3 of cluster 1.
    """
    return block_idx % cluster_size


def peer_block(block_idx: int, cluster_size: int, target_rank: int) -> int:
    """Grid index of the block that `cluster.map_shared_rank(ptr, target_rank)` reaches.

    The rank is relative to your own cluster. Block 5 in clusters of 4,
    asking for rank 2, reaches block 6.
    """
    # global_index_as_rank: the rank is used as if it were a grid index.
    return target_rank


def bin_owner(bin_index: int, bins_per_rank: int) -> tuple[int, int]:
    """Rank that owns a bin, and the bin's offset inside that rank's slice.

    Rank r owns bins r * bins_per_rank up to (r + 1) * bins_per_rank - 1.
    """
    raise NotImplementedError


def simulate_dsmem_histogram(values, num_bins: int, cluster_size: int, num_blocks: int) -> dict:
    """Run the cluster histogram on the CPU.

    Value `i` is handled by block `i % num_blocks`. `bins_per_rank` is
    `ceil(num_bins / cluster_size)`. Each block adds 1 to the slice of the
    bin's owner, in the shared memory of `peer_block(...)`. After the cluster
    barrier, each block flushes its own slice, bins
    `rank * bins_per_rank + offset`, to the global histogram, once per bin.

    Return a dict with `hist` (a list of `num_bins` counts), `local` and
    `remote` (updates that stayed in the block's own shared memory, and
    updates that crossed to a peer), and `global_atomics` (flush additions:
    one per bin per cluster). Raise ValueError if `num_blocks` is not a
    multiple of `cluster_size`. The result must equal `torch.bincount`.
    """
    raise NotImplementedError


def exit_steps(finish_steps: list[int], final_sync: bool) -> list[int]:
    """Step at which each block of a cluster may exit.

    `finish_steps[r]` is the step at which rank `r` finishes its own work.
    A final `cluster.sync()` holds every block until the slowest one
    arrives. Without it, a block exits as soon as it finishes.
    """
    # forgot_final_sync: every block exits at its own finish step.
    return list(finish_steps)


def simulate_cluster_reduction(block_partials: list[float], cluster_size: int) -> tuple[list[float], int]:
    """Tree-reduce each cluster's partial sums through DSMEM.

    At stride s = 1, 2, 4, ..., rank r with `r % (2 * s) == 0` adds the value
    of rank `r + s`, when `r + s < cluster_size`. Return one sum per cluster
    and the number of `cluster.sync()` calls per cluster: one to publish the
    partials, then one after each level.
    """
    raise NotImplementedError


def blocks_per_sm(threads: int, regs_per_thread: int, smem_per_block: int) -> int:
    """Resident blocks per SM on compute capability 9.0.

    The minimum of the thread, register, shared-memory (with 1 KiB reserved
    per block), and 32-block limits. Return 0 when the block needs more than
    227 KiB of shared memory. Ignore register allocation granularity.
    """
    raise NotImplementedError


def max_active_clusters(
    cluster_size: int,
    gpc_sms: tuple[int, ...],
    per_sm_blocks: int,
    allow_nonportable: bool = False,
) -> int:
    """Clusters that are resident at once.

    All blocks of a cluster sit in one GPC, so pack clusters inside each GPC.
    A size above 8 needs the non-portable opt-in, and a size above 16 never
    launches. Return 0 when the launch fails.
    """
    raise NotImplementedError


def multicast_l2_bytes(
    m: int,
    n: int,
    k: int,
    tile_m: int,
    tile_n: int,
    elem_bytes: int,
    cluster_m: int,
    cluster_n: int,
) -> tuple[int, int]:
    """L2 bytes loaded for A and B, without and with TMA multicast.

    A block loads a `tile_m` by `k` slice of A and a `k` by `tile_n` slice of
    B. A cluster holds `cluster_m` tiles along M and `cluster_n` along N. The
    blocks of a cluster row share one A slice, and the blocks of a cluster
    column share one B slice. With multicast, each distinct slice is loaded
    once per cluster. Edge tiles load a full box. Return
    `(baseline_bytes, multicast_bytes)`.
    """
    raise NotImplementedError


def cluster_histogram(x, num_bins: int):
    """Histogram of an int32 CUDA tensor, with the bins in DSMEM, on an H100.

    Use a cluster of 4 with `__cluster_dims__`. Each rank owns
    `ceil(num_bins / 4)` bins in its shared memory and adds to the owner with
    `cluster.map_shared_rank`. Call `cluster.sync()` after zeroing and after
    the update loop. Return an int32 tensor, or print a skip message and
    return None on a machine without an H100. Compile with arch `sm_90`
    through `load_cuda`. Do not compile for sm_80.
    """
    raise NotImplementedError


def global_atomic_histogram(x, num_bins: int):
    """The baseline: one `atomicAdd` per element into a global histogram."""
    raise NotImplementedError


def cluster_reduce(x):
    """Sum of a float32 CUDA tensor with a cluster of 8, on an H100.

    Each block reduces its slice and publishes a partial in shared memory.
    After `cluster.sync()`, rank 0 reads the 8 partials with
    `map_shared_rank` and does one global atomic. A final `cluster.sync()`
    keeps the peers alive until rank 0 is done. Return a 1-element tensor, or
    print a skip message and return None on a machine without an H100.
    """
    raise NotImplementedError


def block_atomic_reduce(x):
    """The baseline: every block does one global `atomicAdd` of its partial."""
    raise NotImplementedError
