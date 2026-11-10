"""Lab 32: sparse formats, SpMV load balance, and BFS. Starter.

Fill in each function that raises NotImplementedError, smallest piece first.
The docstrings are the specification. The CUDA part at the end has two kernel
bodies for you to write: wpr_kernel and merge_kernel.

CPU pieces, plain Python, checked on any machine:

    coo_to_csr, spmv_csr             build and multiply a CSR matrix
    coo_bytes, csr_bytes, ell_bytes, hyb_width, hyb_bytes   what each format costs
    merge_path_coordinate, merge_path_partition, spmv_merge_path   equal rows + nonzeros per worker
    warp_work, imbalance             max/mean work per warp for three SpMV schemes
    bfs_levels_frontier, bottom_up_step, should_switch, bfs_direction_optimizing   BFS
    simulate_work_queue              static against atomic-counter scheduling

GPU pieces, which need CUDA and are skipped without it:

    spmv_thread_per_row, spmv_warp_per_row, spmv_merge
"""

from __future__ import annotations

import heapq

from lab_common import load_cuda

WARP = 32


# ---------------------------------------------------------------------------
# Formats
# ---------------------------------------------------------------------------


def coo_to_csr(n_rows, rows, cols, vals):
    """Build CSR from COO triples. Sort by (row, col) and sum duplicates.

    Returns (row_ptr, col_idx, values). row_ptr has n_rows + 1 entries, and row i
    owns the half-open range row_ptr[i] .. row_ptr[i + 1] of col_idx and values.
    """
    raise NotImplementedError


def spmv_csr(row_ptr, col_idx, values, x):
    """y = A x for a CSR matrix. One output per row."""
    raise NotImplementedError


def coo_bytes(nnz, idx_bytes=4, val_bytes=4):
    """COO stores a row index, a column index, and a value per nonzero."""
    raise NotImplementedError


def csr_bytes(n_rows, nnz, idx_bytes=4, val_bytes=4):
    """CSR stores n_rows + 1 row pointers, then a column index and a value per nonzero."""
    raise NotImplementedError


def ell_bytes(n_rows, width, idx_bytes=4, val_bytes=4):
    """ELL stores n_rows * width column indices and n_rows * width values, padding included."""
    raise NotImplementedError


def hyb_width(row_lengths):
    """The ELL width K of a hybrid format, by the rule in Bell and Garland (2008).

    Add a K-th column to the ELL part while at least one third of the rows have
    K or more entries. Return the largest such K, or 0 when no row has an entry.
    """
    raise NotImplementedError


def hyb_bytes(row_lengths, idx_bytes=4, val_bytes=4):
    """ELL part of width hyb_width, plus a COO entry for every nonzero past that width."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Merge path
# ---------------------------------------------------------------------------


def merge_path_coordinate(row_ptr, diagonal):
    """Where the merge path stands after `diagonal` steps: (rows finished, nonzeros consumed).

    Merge list A, the row end offsets row_ptr[1:], with list B, the nonzero
    indices 0, 1, 2, ... A row end is taken before a nonzero when it is <=.
    The two counts always add to `diagonal`. Binary search the row count.
    """
    raise NotImplementedError


def merge_path_partition(row_ptr, workers):
    """Boundaries of `workers` equal shares of rows + nonzeros: workers + 1 pairs (row, nz).

    Worker w covers the steps from boundary w to boundary w + 1. Each share holds
    ceil((n_rows + nnz) / workers) steps, except the last, which holds what remains.
    """
    raise NotImplementedError


def spmv_merge_path(row_ptr, col_idx, values, x, workers):
    """SpMV where each worker walks an equal share of the merge path.

    A worker's last row may continue in the next worker's share, so it adds its
    partial sum to y with +=, the CPU version of the fix-up a GPU kernel does
    with an atomic add or a second pass.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Load imbalance
# ---------------------------------------------------------------------------


def warp_work(row_ptr, scheme, warps):
    """Iterations each warp runs, in a lockstep model.

    thread_per_row: warp g owns rows 32g .. 32g + 31, one per lane. The lanes run
        in lockstep, so the warp takes max(row length) iterations, at least 1.
        `warps` is ignored: there are ceil(n_rows / 32) warps.
    warp_per_row: row i goes to warp i % warps. A row takes ceil(length / 32)
        iterations, at least 1, because the 32 lanes split the row.
    merge_path: each of `warps` warps takes an equal share of rows + nonzeros from
        merge_path_partition and runs ceil(share / 32) iterations.
    """
    raise NotImplementedError


def imbalance(work):
    """max / mean of per-warp work. 1.0 is perfect balance."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# BFS
# ---------------------------------------------------------------------------


def bfs_levels_frontier(row_ptr, col_idx, source):
    """Level-synchronous top-down BFS. level[v] is the distance from source, or -1.

    Each round expands every vertex of the frontier. A neighbor joins the next
    frontier only when it is unvisited. On a GPU that test is an atomic
    compare-and-swap, so exactly one thread claims each vertex.
    """
    raise NotImplementedError


def bottom_up_step(row_ptr, col_idx, level, depth):
    """One bottom-up round on an undirected graph.

    Every unvisited vertex scans its neighbors and stops at the first one whose
    level is `depth` (the current frontier). That vertex gets level depth + 1.
    Returns (new frontier, edges examined). `level` is updated in place.
    """
    raise NotImplementedError


def should_switch(direction, frontier_edges, unexplored_edges, frontier_size, n, alpha=4, beta=18):
    """The direction for the next round: "top_down" or "bottom_up".

    From top_down, switch when the frontier's edge count m_f exceeds the edge count of
    the unvisited vertices m_u divided by alpha. From bottom_up, switch back when the
    frontier size n_f drops below n / beta. Otherwise keep the direction.
    """
    raise NotImplementedError


def bfs_direction_optimizing(row_ptr, col_idx, source, alpha=4, beta=18):
    """BFS that picks top-down or bottom-up each round with should_switch.

    Returns (level, directions, edges_examined). directions has one entry per
    round. A top-down round examines every edge of the frontier's vertices.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Work queue
# ---------------------------------------------------------------------------


def simulate_work_queue(costs, workers, dynamic):
    """Makespan of running tasks with the given costs on `workers` workers.

    dynamic=False: task i goes to worker i % workers before anything runs.
    dynamic=True: a persistent worker takes the next task from a shared counter
    each time it finishes one, so the earliest-free worker takes task 0, 1, 2, ...
    in order. Returns the time the last worker finishes.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# GPU kernels. A float32 CSR matrix, y = A x.
# ---------------------------------------------------------------------------

CPP_SOURCE = """
torch::Tensor spmv_thread_per_row(torch::Tensor row_ptr, torch::Tensor col_idx, torch::Tensor values, torch::Tensor x);
torch::Tensor spmv_warp_per_row(torch::Tensor row_ptr, torch::Tensor col_idx, torch::Tensor values, torch::Tensor x);
torch::Tensor spmv_merge(torch::Tensor row_ptr, torch::Tensor col_idx, torch::Tensor values, torch::Tensor x, int64_t items_per_thread);
"""

FUNCTIONS = ["spmv_thread_per_row", "spmv_warp_per_row", "spmv_merge"]

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

constexpr int THREADS = 256;

// One thread per row. Lanes of a warp read different rows, so loads are scattered,
// and a warp runs as long as its longest row.
__global__ void tpr_kernel(int n_rows, const int* __restrict__ rp, const int* __restrict__ ci,
                           const float* __restrict__ v, const float* __restrict__ x, float* __restrict__ y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    float acc = 0.0f;
    for (int k = rp[row]; k < rp[row + 1]; ++k) acc += v[k] * x[ci[k]];
    y[row] = acc;
}

// One warp per row. The 32 lanes read consecutive entries of the row, then a shuffle tree sums them.
// TODO: the warp w takes rows w, w + warps, w + 2 * warps, ... Lane l sums entries rp[row] + l, + 32, ...
// Reduce the 32 partial sums with __shfl_down_sync, and let lane 0 write y[row].
__global__ void wpr_kernel(int n_rows, const int* __restrict__ rp, const int* __restrict__ ci,
                           const float* __restrict__ v, const float* __restrict__ x, float* __restrict__ y) {
}

// Binary search on diagonal `diag` of the merge path of row ends (list A) and nonzero indices (list B).
__device__ void merge_coord(long long diag, int n_rows, int nnz, const int* __restrict__ rp, int& row, int& nz) {
    long long lo = diag - nnz > 0 ? diag - nnz : 0;
    long long hi = diag < n_rows ? diag : n_rows;
    while (lo < hi) {
        long long mid = (lo + hi) >> 1;
        if (rp[mid + 1] <= diag - 1 - mid) lo = mid + 1; else hi = mid;
    }
    row = (int)lo;
    nz = (int)(diag - lo);
}

// One thread walks `items` steps of the merge path. y starts at zero.
// TODO: d0 = thread * items and d1 = min(d0 + items, n_rows + nnz). Find (i, j) at d0 and (i_end, j_end) at d1
// with merge_coord. At each step, if i < n_rows and rp[i + 1] <= j, the row i ends: atomicAdd the running sum
// into y[i], reset it, and move to the next row. Otherwise add v[j] * x[ci[j]] and move to the next nonzero.
// After the loop, atomicAdd the leftover running sum into y[i] if i < n_rows.
__global__ void merge_kernel(int n_rows, int nnz, int items, const int* __restrict__ rp, const int* __restrict__ ci,
                             const float* __restrict__ v, const float* __restrict__ x, float* __restrict__ y) {
}

static void check_inputs(torch::Tensor row_ptr, torch::Tensor col_idx, torch::Tensor values, torch::Tensor x) {
    TORCH_CHECK(row_ptr.is_cuda() && col_idx.is_cuda() && values.is_cuda() && x.is_cuda(), "tensors must be on the GPU");
    TORCH_CHECK(row_ptr.scalar_type() == torch::kInt32 && col_idx.scalar_type() == torch::kInt32, "indices must be int32");
    TORCH_CHECK(values.scalar_type() == torch::kFloat32 && x.scalar_type() == torch::kFloat32, "values and x must be float32");
    TORCH_CHECK(row_ptr.is_contiguous() && col_idx.is_contiguous() && values.is_contiguous() && x.is_contiguous(), "contiguous");
}

torch::Tensor spmv_thread_per_row(torch::Tensor row_ptr, torch::Tensor col_idx, torch::Tensor values, torch::Tensor x) {
    check_inputs(row_ptr, col_idx, values, x);
    int n_rows = (int)row_ptr.numel() - 1;
    auto y = torch::zeros({n_rows}, values.options());
    int blocks = (n_rows + THREADS - 1) / THREADS;
    tpr_kernel<<<blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_rows, row_ptr.data_ptr<int>(), col_idx.data_ptr<int>(), values.data_ptr<float>(), x.data_ptr<float>(), y.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor spmv_warp_per_row(torch::Tensor row_ptr, torch::Tensor col_idx, torch::Tensor values, torch::Tensor x) {
    check_inputs(row_ptr, col_idx, values, x);
    int n_rows = (int)row_ptr.numel() - 1;
    auto y = torch::zeros({n_rows}, values.options());
    int warps_per_block = THREADS / 32;
    int blocks = (n_rows + warps_per_block - 1) / warps_per_block;
    if (blocks > 108 * 16) blocks = 108 * 16;
    wpr_kernel<<<blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_rows, row_ptr.data_ptr<int>(), col_idx.data_ptr<int>(), values.data_ptr<float>(), x.data_ptr<float>(), y.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor spmv_merge(torch::Tensor row_ptr, torch::Tensor col_idx, torch::Tensor values, torch::Tensor x, int64_t items_per_thread) {
    check_inputs(row_ptr, col_idx, values, x);
    int n_rows = (int)row_ptr.numel() - 1;
    int nnz = (int)col_idx.numel();
    auto y = torch::zeros({n_rows}, values.options());
    long long total = (long long)n_rows + nnz;
    long long threads = (total + items_per_thread - 1) / items_per_thread;
    int blocks = (int)((threads + THREADS - 1) / THREADS);
    merge_kernel<<<blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_rows, nnz, (int)items_per_thread, row_ptr.data_ptr<int>(), col_idx.data_ptr<int>(), values.data_ptr<float>(), x.data_ptr<float>(), y.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
"""


def build():
    """Compile the CUDA source (cached) and return the extension module."""
    return load_cuda("lab32_spmv", CUDA_SOURCE, CPP_SOURCE, FUNCTIONS)


def spmv_thread_per_row(row_ptr, col_idx, values, x):
    return build().spmv_thread_per_row(row_ptr, col_idx, values, x)


def spmv_warp_per_row(row_ptr, col_idx, values, x):
    return build().spmv_warp_per_row(row_ptr, col_idx, values, x)


def spmv_merge(row_ptr, col_idx, values, x, items_per_thread=16):
    return build().spmv_merge(row_ptr, col_idx, values, x, items_per_thread)
