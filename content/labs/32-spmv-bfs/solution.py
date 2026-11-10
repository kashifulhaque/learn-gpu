"""Lab 32: sparse formats, SpMV load balance, and BFS. Reference solution.

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
    merged: dict[tuple[int, int], float] = {}
    for r, c, v in zip(rows, cols, vals):
        merged[(r, c)] = merged.get((r, c), 0.0) + v
    keys = sorted(merged)
    row_ptr = [0] * (n_rows + 1)
    for r, _ in keys:
        row_ptr[r + 1] += 1
    for i in range(n_rows):
        row_ptr[i + 1] += row_ptr[i]
    return row_ptr, [c for _, c in keys], [merged[k] for k in keys]


def spmv_csr(row_ptr, col_idx, values, x):
    """y = A x for a CSR matrix. One output per row."""
    y = []
    for i in range(len(row_ptr) - 1):
        acc = 0.0
        # A row ends at row_ptr[i + 1], exclusive. An end of row_ptr[i + 1] - 1 drops the last entry.
        for k in range(row_ptr[i], row_ptr[i + 1]):
            acc += values[k] * x[col_idx[k]]
        y.append(acc)
    return y


def coo_bytes(nnz, idx_bytes=4, val_bytes=4):
    """COO stores a row index, a column index, and a value per nonzero."""
    return nnz * (2 * idx_bytes + val_bytes)


def csr_bytes(n_rows, nnz, idx_bytes=4, val_bytes=4):
    """CSR stores n_rows + 1 row pointers, then a column index and a value per nonzero."""
    return (n_rows + 1) * idx_bytes + nnz * (idx_bytes + val_bytes)


def ell_bytes(n_rows, width, idx_bytes=4, val_bytes=4):
    """ELL stores n_rows * width column indices and n_rows * width values, padding included."""
    return n_rows * width * (idx_bytes + val_bytes)


def hyb_width(row_lengths):
    """The ELL width K of a hybrid format, by the rule in Bell and Garland (2008).

    Add a K-th column to the ELL part while at least one third of the rows have
    K or more entries. Return the largest such K, or 0 when no row has an entry.
    """
    n = len(row_lengths)
    best = 0
    for k in range(1, max(row_lengths, default=0) + 1):
        if 3 * sum(1 for length in row_lengths if length >= k) >= n:
            best = k
    return best


def hyb_bytes(row_lengths, idx_bytes=4, val_bytes=4):
    """ELL part of width hyb_width, plus a COO entry for every nonzero past that width."""
    width = hyb_width(row_lengths)
    overflow = sum(max(0, length - width) for length in row_lengths)
    return ell_bytes(len(row_lengths), width, idx_bytes, val_bytes) + coo_bytes(overflow, idx_bytes, val_bytes)


# ---------------------------------------------------------------------------
# Merge path
# ---------------------------------------------------------------------------


def merge_path_coordinate(row_ptr, diagonal):
    """Where the merge path stands after `diagonal` steps: (rows finished, nonzeros consumed).

    Merge list A, the row end offsets row_ptr[1:], with list B, the nonzero
    indices 0, 1, 2, ... A row end is taken before a nonzero when it is <=.
    The two counts always add to `diagonal`. Binary search the row count.
    """
    n_rows = len(row_ptr) - 1
    nnz = row_ptr[-1]
    lo = max(0, diagonal - nnz)
    hi = min(diagonal, n_rows)
    while lo < hi:
        mid = (lo + hi) // 2
        if row_ptr[mid + 1] <= diagonal - 1 - mid:
            lo = mid + 1
        else:
            hi = mid
    return lo, diagonal - lo


def merge_path_partition(row_ptr, workers):
    """Boundaries of `workers` equal shares of rows + nonzeros: workers + 1 pairs (row, nz).

    Worker w covers the steps from boundary w to boundary w + 1. Each share holds
    ceil((n_rows + nnz) / workers) steps, except the last, which holds what remains.
    """
    total = len(row_ptr) - 1 + row_ptr[-1]
    share = -(-total // workers)
    return [merge_path_coordinate(row_ptr, min(w * share, total)) for w in range(workers + 1)]


def spmv_merge_path(row_ptr, col_idx, values, x, workers):
    """SpMV where each worker walks an equal share of the merge path.

    A worker's last row may continue in the next worker's share, so it adds its
    partial sum to y with +=, the CPU version of the fix-up a GPU kernel does
    with an atomic add or a second pass.
    """
    n_rows = len(row_ptr) - 1
    y = [0.0] * n_rows
    bounds = merge_path_partition(row_ptr, workers)
    for w in range(workers):
        (i, j), (i_end, j_end) = bounds[w], bounds[w + 1]
        running = 0.0
        while i < i_end or j < j_end:
            if i < n_rows and row_ptr[i + 1] <= j:
                y[i] += running
                running = 0.0
                i += 1
            else:
                running += values[j] * x[col_idx[j]]
                j += 1
        if i < n_rows:
            y[i] += running
    return y


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
    n_rows = len(row_ptr) - 1
    lengths = [row_ptr[i + 1] - row_ptr[i] for i in range(n_rows)]
    if scheme == "thread_per_row":
        return [max(1, max(lengths[g : g + WARP])) for g in range(0, n_rows, WARP)]
    if scheme == "warp_per_row":
        work = [0] * warps
        for i, length in enumerate(lengths):
            work[i % warps] += max(1, -(-length // WARP))
        return work
    if scheme == "merge_path":
        bounds = merge_path_partition(row_ptr, warps)
        out = []
        for w in range(warps):
            steps = (bounds[w + 1][0] - bounds[w][0]) + (bounds[w + 1][1] - bounds[w][1])
            out.append(-(-steps // WARP))
        return out
    raise ValueError(f"unknown scheme {scheme!r}")


def imbalance(work):
    """max / mean of per-warp work. 1.0 is perfect balance."""
    return max(work) / (sum(work) / len(work))


# ---------------------------------------------------------------------------
# BFS
# ---------------------------------------------------------------------------


def bfs_levels_frontier(row_ptr, col_idx, source):
    """Level-synchronous top-down BFS. level[v] is the distance from source, or -1.

    Each round expands every vertex of the frontier. A neighbor joins the next
    frontier only when it is unvisited. On a GPU that test is an atomic
    compare-and-swap, so exactly one thread claims each vertex.
    """
    n = len(row_ptr) - 1
    level = [-1] * n
    level[source] = 0
    frontier = [source]
    depth = 0
    while frontier:
        nxt = []
        for u in frontier:
            for k in range(row_ptr[u], row_ptr[u + 1]):
                v = col_idx[k]
                if level[v] == -1:
                    level[v] = depth + 1
                    nxt.append(v)
        frontier = nxt
        depth += 1
    return level


def bottom_up_step(row_ptr, col_idx, level, depth):
    """One bottom-up round on an undirected graph.

    Every unvisited vertex scans its neighbors and stops at the first one whose
    level is `depth` (the current frontier). That vertex gets level depth + 1.
    Returns (new frontier, edges examined). `level` is updated in place.
    """
    nxt = []
    examined = 0
    # Mark the new vertices after the scan, so a vertex found this round is not a parent this round.
    found = []
    for v in range(len(row_ptr) - 1):
        if level[v] != -1:
            continue
        for k in range(row_ptr[v], row_ptr[v + 1]):
            examined += 1
            if level[col_idx[k]] == depth:
                found.append(v)
                break
    for v in found:
        level[v] = depth + 1
        nxt.append(v)
    return nxt, examined


def should_switch(direction, frontier_edges, unexplored_edges, frontier_size, n, alpha=4, beta=18):
    """The direction for the next round: "top_down" or "bottom_up".

    From top_down, switch when the frontier's edge count m_f exceeds the edge count of
    the unvisited vertices m_u divided by alpha. From bottom_up, switch back when the
    frontier size n_f drops below n / beta. Otherwise keep the direction.
    """
    if direction == "top_down":
        return "bottom_up" if frontier_edges * alpha > unexplored_edges else "top_down"
    return "top_down" if frontier_size * beta < n else "bottom_up"


def bfs_direction_optimizing(row_ptr, col_idx, source, alpha=4, beta=18):
    """BFS that picks top-down or bottom-up each round with should_switch.

    Returns (level, directions, edges_examined). directions has one entry per
    round. A top-down round examines every edge of the frontier's vertices.
    """
    n = len(row_ptr) - 1
    degree = [row_ptr[v + 1] - row_ptr[v] for v in range(n)]
    level = [-1] * n
    level[source] = 0
    frontier = [source]
    unexplored = sum(degree) - degree[source]
    depth = 0
    direction = "top_down"
    directions = []
    examined_total = 0
    while frontier:
        m_f = sum(degree[u] for u in frontier)
        direction = should_switch(direction, m_f, unexplored, len(frontier), n, alpha, beta)
        directions.append(direction)
        if direction == "top_down":
            nxt = []
            for u in frontier:
                for k in range(row_ptr[u], row_ptr[u + 1]):
                    examined_total += 1
                    v = col_idx[k]
                    if level[v] == -1:
                        level[v] = depth + 1
                        nxt.append(v)
        else:
            nxt, examined = bottom_up_step(row_ptr, col_idx, level, depth)
            examined_total += examined
        unexplored -= sum(degree[v] for v in nxt)
        frontier = nxt
        depth += 1
    return level, directions, examined_total


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
    if not dynamic:
        totals = [0] * workers
        for i, cost in enumerate(costs):
            totals[i % workers] += cost
        return max(totals)
    free_at = [(0, w) for w in range(workers)]
    heapq.heapify(free_at)
    finish = 0
    for cost in costs:
        t, w = heapq.heappop(free_at)
        heapq.heappush(free_at, (t + cost, w))
        finish = max(finish, t + cost)
    return finish


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
__global__ void wpr_kernel(int n_rows, const int* __restrict__ rp, const int* __restrict__ ci,
                           const float* __restrict__ v, const float* __restrict__ x, float* __restrict__ y) {
    int lane = threadIdx.x & 31;
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int warps = (gridDim.x * blockDim.x) >> 5;
    for (int row = warp; row < n_rows; row += warps) {
        float acc = 0.0f;
        for (int k = rp[row] + lane; k < rp[row + 1]; k += 32) acc += v[k] * x[ci[k]];
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
        if (lane == 0) y[row] = acc;
    }
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

// One thread walks `items` steps of the merge path. y must start at zero: a row that spans
// threads receives one atomic add from each, so the sum order, and the last float bits, can vary.
__global__ void merge_kernel(int n_rows, int nnz, int items, const int* __restrict__ rp, const int* __restrict__ ci,
                             const float* __restrict__ v, const float* __restrict__ x, float* __restrict__ y) {
    long long total = (long long)n_rows + nnz;
    long long d0 = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * items;
    if (d0 >= total) return;
    long long d1 = d0 + items < total ? d0 + items : total;
    int i, j, i_end, j_end;
    merge_coord(d0, n_rows, nnz, rp, i, j);
    merge_coord(d1, n_rows, nnz, rp, i_end, j_end);
    float running = 0.0f;
    for (long long d = d0; d < d1; ++d) {
        if (i < n_rows && rp[i + 1] <= j) {
            atomicAdd(&y[i], running);
            running = 0.0f;
            ++i;
        } else {
            running += v[j] * x[ci[j]];
            ++j;
        }
    }
    if (i < n_rows && running != 0.0f) atomicAdd(&y[i], running);
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
