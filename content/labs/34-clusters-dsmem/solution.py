"""Lab 34 solution.

The CPU functions model a thread block cluster: which block a
`cluster.map_shared_rank` pointer reaches, a histogram whose bins live in
distributed shared memory, a tree reduction across the blocks of a cluster,
how many clusters fit on an H100, and how much L2 traffic TMA multicast
removes from a GEMM. The kernels are CUDA for sm_90. A machine without an
H100 skips them and does not compile for sm_80.
"""

from __future__ import annotations

# Compute capability 9.0 limits, from docs/FACTS.md section 4.2.
MAX_THREADS_PER_SM = 2048
REGISTERS_PER_SM = 65536
SMEM_PER_SM = 228 * 1024
SMEM_PER_BLOCK_MAX = 227 * 1024
SMEM_RESERVED_PER_BLOCK = 1024
MAX_BLOCKS_PER_SM = 32
MAX_PORTABLE_CLUSTER = 8
MAX_NONPORTABLE_CLUSTER = 16

# The H100 SXM5 has 8 GPCs and 132 SMs (NVIDIA Hopper in-depth blog). NVIDIA
# does not publish how many SMs each GPC keeps. This split is an assumption
# that sums to 132: six GPCs with 16 SMs and two with 18.
H100_SXM5_GPC_SMS = (16, 16, 16, 16, 16, 16, 18, 18)


def cluster_rank(block_idx: int, cluster_size: int) -> int:
    """Rank of a block inside its cluster: `cluster.block_rank()` for a 1D cluster."""
    return block_idx % cluster_size


def peer_block(block_idx: int, cluster_size: int, target_rank: int) -> int:
    """Grid index of the block that `cluster.map_shared_rank(ptr, target_rank)` reaches."""
    first = block_idx - cluster_rank(block_idx, cluster_size)
    return first + target_rank


def bin_owner(bin_index: int, bins_per_rank: int) -> tuple[int, int]:
    """Rank that owns a bin, and the bin's offset inside that rank's slice."""
    return bin_index // bins_per_rank, bin_index % bins_per_rank


def simulate_dsmem_histogram(values, num_bins: int, cluster_size: int, num_blocks: int) -> dict:
    """Run the cluster histogram on the CPU.

    Value `i` is handled by block `i % num_blocks`. Each block adds into the
    slice of the bin's owner, in the shared memory of `peer_block(...)`. After
    the cluster barrier, each block flushes its own slice to the global
    histogram. Returns the global histogram and the update counts.
    """
    if num_blocks % cluster_size:
        raise ValueError(f"num_blocks {num_blocks} must be a multiple of cluster_size {cluster_size}")
    bins_per_rank = -(-num_bins // cluster_size)
    smem = [[0] * bins_per_rank for _ in range(num_blocks)]
    local = remote = 0
    for i, value in enumerate(values):
        block = i % num_blocks
        owner, offset = bin_owner(int(value), bins_per_rank)
        target = peer_block(block, cluster_size, owner)
        smem[target][offset] += 1
        if target == block:
            local += 1
        else:
            remote += 1
    hist = [0] * num_bins
    global_atomics = 0
    for block in range(num_blocks):
        rank = cluster_rank(block, cluster_size)
        for offset in range(bins_per_rank):
            bin_index = rank * bins_per_rank + offset
            if bin_index < num_bins:
                hist[bin_index] += smem[block][offset]
                global_atomics += 1
    return {"hist": hist, "local": local, "remote": remote, "global_atomics": global_atomics}


def exit_steps(finish_steps: list[int], final_sync: bool) -> list[int]:
    """Step at which each block of a cluster may exit.

    `finish_steps[r]` is the step at which rank `r` finishes its own work.
    A final `cluster.sync()` holds every block until the slowest one
    arrives. Without it, a block exits as soon as it finishes.
    """
    if final_sync:
        last = max(finish_steps)
        return [last] * len(finish_steps)
    return list(finish_steps)


def simulate_cluster_reduction(block_partials: list[float], cluster_size: int) -> tuple[list[float], int]:
    """Tree-reduce each cluster's partial sums through DSMEM.

    Returns one sum per cluster and the number of `cluster.sync()` calls per
    cluster: one to publish the partials, then one after each level. The last
    one keeps every block alive until rank 0 has read its peers.
    """
    if len(block_partials) % cluster_size:
        raise ValueError(f"{len(block_partials)} blocks is not a multiple of cluster_size {cluster_size}")
    sums: list[float] = []
    syncs = 1
    for start in range(0, len(block_partials), cluster_size):
        smem = list(block_partials[start : start + cluster_size])
        syncs = 1
        stride = 1
        while stride < cluster_size:
            for rank in range(0, cluster_size, 2 * stride):
                if rank + stride < cluster_size:
                    smem[rank] += smem[rank + stride]
            syncs += 1
            stride *= 2
        sums.append(smem[0])
    return sums, syncs


def blocks_per_sm(threads: int, regs_per_thread: int, smem_per_block: int) -> int:
    """Resident blocks per SM on compute capability 9.0.

    The model ignores register allocation granularity. Zero means the block
    does not fit.
    """
    if threads <= 0 or smem_per_block > SMEM_PER_BLOCK_MAX:
        return 0
    by_threads = MAX_THREADS_PER_SM // threads
    by_regs = REGISTERS_PER_SM // (threads * regs_per_thread)
    by_smem = SMEM_PER_SM // (smem_per_block + SMEM_RESERVED_PER_BLOCK)
    return min(by_threads, by_regs, by_smem, MAX_BLOCKS_PER_SM)


def max_active_clusters(
    cluster_size: int,
    gpc_sms: tuple[int, ...],
    per_sm_blocks: int,
    allow_nonportable: bool = False,
) -> int:
    """Clusters that are resident at once.

    Every block of a cluster sits in one GPC, so each GPC holds
    `sms * per_sm_blocks // cluster_size` clusters. A size above 8 needs the
    non-portable opt-in. A size above 16 never launches. Zero means the
    launch fails. The real number comes from `cudaOccupancyMaxActiveClusters`.
    """
    if cluster_size < 1 or cluster_size > MAX_NONPORTABLE_CLUSTER:
        return 0
    if cluster_size > MAX_PORTABLE_CLUSTER and not allow_nonportable:
        return 0
    return sum(sms * per_sm_blocks // cluster_size for sms in gpc_sms)


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

    A block computes one `tile_m` by `tile_n` output tile and loads a
    `tile_m` by `k` slice of A and a `k` by `tile_n` slice of B. A cluster
    holds `cluster_m` tiles along M and `cluster_n` along N. The blocks of a
    cluster row share one A slice, and the blocks of a cluster column share
    one B slice. With multicast, each distinct slice is loaded once per
    cluster. Edge tiles load a full box.
    """
    tiles_m = -(-m // tile_m)
    tiles_n = -(-n // tile_n)
    a_slice = tile_m * k * elem_bytes
    b_slice = tile_n * k * elem_bytes
    baseline = tiles_m * tiles_n * (a_slice + b_slice)
    clusters_m = -(-tiles_m // cluster_m)
    clusters_n = -(-tiles_n // cluster_n)
    multicast = 0
    for im in range(clusters_m):
        distinct_m = min(cluster_m, tiles_m - im * cluster_m)
        for in_ in range(clusters_n):
            distinct_n = min(cluster_n, tiles_n - in_ * cluster_n)
            multicast += distinct_m * a_slice + distinct_n * b_slice
    return baseline, multicast


def _skip_reason() -> str | None:
    msg = "SKIP: the cluster kernels were not run. "
    try:
        import torch
    except ImportError:
        return msg + "torch is not installed, so there is no H100 to compile them on."
    if not torch.cuda.is_available():
        return msg + "No CUDA GPU is visible. They need an H100 (sm_90) and are not sent to the A100 endpoint."
    major, minor = torch.cuda.get_device_capability()
    if major < 9:
        return msg + f"The visible GPU is sm_{major}{minor}, not sm_90. They are not sent to the A100 endpoint."
    return None


_CPP = """
torch::Tensor cluster_hist_cuda(torch::Tensor x, int64_t num_bins);
torch::Tensor global_hist_cuda(torch::Tensor x, int64_t num_bins);
torch::Tensor cluster_reduce_cuda(torch::Tensor x);
torch::Tensor block_atomic_reduce_cuda(torch::Tensor x);
"""

# Real sm_90 CUDA. No TMA or wgmma, so plain sm_90 is enough. The histogram
# cluster has 4 blocks and each rank owns a slice of the bins in its shared
# memory. The reduction cluster has 8 blocks, the portable maximum.
_CUDA = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

constexpr int HIST_CLUSTER = 4;
constexpr int BINS_PER_RANK_MAX = 4096;  // 16 KiB of int32 per block
constexpr int REDUCE_CLUSTER = 8;
constexpr int THREADS = 256;

__global__ void __cluster_dims__(HIST_CLUSTER, 1, 1)
cluster_hist_kernel(const int* __restrict__ x, int64_t n, int num_bins, int* __restrict__ out) {
    __shared__ int smem[BINS_PER_RANK_MAX];
    cg::cluster_group cluster = cg::this_cluster();
    const unsigned rank = cluster.block_rank();
    const int bins_per_rank = (num_bins + HIST_CLUSTER - 1) / HIST_CLUSTER;

    for (int i = threadIdx.x; i < bins_per_rank; i += blockDim.x) smem[i] = 0;
    // Every block's slice is zeroed before any block adds to it.
    cluster.sync();

    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const int b = x[i];
        const unsigned owner = (unsigned)(b / bins_per_rank);
        const int offset = b - (int)owner * bins_per_rank;
        // owner is a rank inside the cluster, not a block index in the grid.
        int* dst = cluster.map_shared_rank(smem, owner);
        atomicAdd(dst + offset, 1);
    }
    // Every remote add has landed, and no block exits while a peer may still
    // write into its shared memory.
    cluster.sync();

    for (int i = threadIdx.x; i < bins_per_rank; i += blockDim.x) {
        const int bin = (int)rank * bins_per_rank + i;
        if (bin < num_bins && smem[i] != 0) atomicAdd(&out[bin], smem[i]);
    }
}

__global__ void global_hist_kernel(const int* __restrict__ x, int64_t n, int* __restrict__ out) {
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        atomicAdd(&out[x[i]], 1);
    }
}

__device__ float block_sum(float v) {
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    v = (threadIdx.x < (blockDim.x + 31) / 32) ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    }
    return v;  // valid in thread 0
}

__global__ void __cluster_dims__(REDUCE_CLUSTER, 1, 1)
cluster_reduce_kernel(const float* __restrict__ x, int64_t n, float* __restrict__ out) {
    __shared__ float partial;
    cg::cluster_group cluster = cg::this_cluster();
    float v = 0.0f;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) v += x[i];
    v = block_sum(v);
    if (threadIdx.x == 0) partial = v;
    // Publish: every partial is written before rank 0 reads it.
    cluster.sync();
    if (cluster.block_rank() == 0 && threadIdx.x == 0) {
        float total = 0.0f;
        for (unsigned r = 0; r < cluster.num_blocks(); ++r) total += *cluster.map_shared_rank(&partial, r);
        atomicAdd(out, total);  // one global atomic per cluster
    }
    // Final sync: peers keep their shared memory alive until rank 0 is done.
    cluster.sync();
}

__global__ void block_atomic_reduce_kernel(const float* __restrict__ x, int64_t n, float* __restrict__ out) {
    float v = 0.0f;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) v += x[i];
    v = block_sum(v);
    if (threadIdx.x == 0) atomicAdd(out, v);  // one global atomic per block
}

static int grid_for(int64_t n, int cluster) {
    int64_t blocks = (n + THREADS * 8 - 1) / (THREADS * 8);
    if (blocks > 1056) blocks = 1056;
    if (blocks < cluster) blocks = cluster;
    return (int)(((blocks + cluster - 1) / cluster) * cluster);
}

torch::Tensor cluster_hist_cuda(torch::Tensor x, int64_t num_bins) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kInt32 && x.is_contiguous(), "x must be a contiguous CUDA int32 tensor");
    TORCH_CHECK(num_bins >= 1 && num_bins <= (int64_t)HIST_CLUSTER * BINS_PER_RANK_MAX, "num_bins must be in 1..16384");
    c10::cuda::CUDAGuard guard(x.device());
    auto out = torch::zeros({num_bins}, x.options());
    cluster_hist_kernel<<<grid_for(x.numel(), HIST_CLUSTER), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<int>(), x.numel(), (int)num_bins, out.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor global_hist_cuda(torch::Tensor x, int64_t num_bins) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kInt32 && x.is_contiguous(), "x must be a contiguous CUDA int32 tensor");
    c10::cuda::CUDAGuard guard(x.device());
    auto out = torch::zeros({num_bins}, x.options());
    global_hist_kernel<<<grid_for(x.numel(), 1), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<int>(), x.numel(), out.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor cluster_reduce_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.is_contiguous(), "x must be a contiguous CUDA float32 tensor");
    c10::cuda::CUDAGuard guard(x.device());
    auto out = torch::zeros({1}, x.options());
    cluster_reduce_kernel<<<grid_for(x.numel(), REDUCE_CLUSTER), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), x.numel(), out.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor block_atomic_reduce_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.is_contiguous(), "x must be a contiguous CUDA float32 tensor");
    c10::cuda::CUDAGuard guard(x.device());
    auto out = torch::zeros({1}, x.options());
    block_atomic_reduce_kernel<<<grid_for(x.numel(), REDUCE_CLUSTER), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), x.numel(), out.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

_MODULE = None


def _module():
    global _MODULE
    if _MODULE is None:
        from lab_common import load_cuda

        _MODULE = load_cuda(
            "clusters_dsmem_34",
            _CUDA,
            _CPP,
            ["cluster_hist_cuda", "global_hist_cuda", "cluster_reduce_cuda", "block_atomic_reduce_cuda"],
            arch="sm_90",
        )
    return _MODULE


def cluster_histogram(x, num_bins: int):
    """Histogram of an int32 CUDA tensor with the bins in DSMEM. Returns int32, or None when skipped."""
    reason = _skip_reason()
    if reason is not None:
        print(reason, flush=True)
        return None
    return _module().cluster_hist_cuda(x, num_bins)


def global_atomic_histogram(x, num_bins: int):
    """The baseline: one global atomic per element. Returns int32, or None when skipped."""
    reason = _skip_reason()
    if reason is not None:
        print(reason, flush=True)
        return None
    return _module().global_hist_cuda(x, num_bins)


def cluster_reduce(x):
    """Sum of a float32 CUDA tensor with a cluster of 8. Returns a 1-element tensor, or None when skipped."""
    reason = _skip_reason()
    if reason is not None:
        print(reason, flush=True)
        return None
    return _module().cluster_reduce_cuda(x)


def block_atomic_reduce(x):
    """The baseline: one global atomic per block. Returns a 1-element tensor, or None when skipped."""
    reason = _skip_reason()
    if reason is not None:
        print(reason, flush=True)
        return None
    return _module().block_atomic_reduce_cuda(x)
