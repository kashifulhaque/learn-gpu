"""Lab 27: debug five GPU kernels, and read what compute-sanitizer prints.

There are two parts. Part 1 has five CUDA kernels that each contain one bug.
Edit the CUDA strings until the kernel is correct and its sanitizer is quiet.
Part 2 is plain Python: functions that reason about indices and hazards, and a
parser for the text compute-sanitizer prints. Part 2 runs on any machine.

The tests run each fixed kernel in a subprocess under
`compute-sanitizer --tool <X>` and require zero findings. The tool for each
kernel is in the comment above it. Keep each host function's name and its
signature: the tests call them. The first run compiles five modules, which
takes a few minutes.

Reports for the parser are in `fixture_reports.py`. Read them before you write
the parser.
"""

from __future__ import annotations

import re  # noqa: F401  (the parser is easiest with regular expressions)

# ---------------------------------------------------------------------------
# Part 1. Five kernels, one bug each.
# ---------------------------------------------------------------------------

# Bug 1. Tool: memcheck. Run on 1000 floats with 256 threads per block.
# The answer looks right, and the kernel still touches memory it does not own.
OOB_CUDA = r"""
#include <torch/extension.h>

__global__ void scale_shift_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    y[i] = 2.0f * x[i] + 1.0f;
}

torch::Tensor scale_shift(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.is_contiguous());
    int n = static_cast<int>(x.numel());
    auto y = torch::empty_like(x);
    int block = 256;
    int grid = (n + block - 1) / block;
    scale_shift_kernel<<<grid, block>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "scale_shift launch failed");
    return y;
}
"""

# Bug 2. Tool: racecheck. A sum of 256 values in shared memory, one tree step
# at a time.
RACE_CUDA = r"""
#include <torch/extension.h>

__global__ void block_sum_kernel(const float* x, float* out) {
    __shared__ float s[256];
    int t = threadIdx.x;
    s[t] = x[blockIdx.x * 256 + t];
    __syncthreads();
    for (int d = 128; d > 0; d >>= 1) {
        if (t < d) {
            s[t] += s[t + d];
        }
    }
    if (t == 0) {
        out[blockIdx.x] = s[0];
    }
}

torch::Tensor block_sums(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.is_contiguous());
    TORCH_CHECK(x.numel() % 256 == 0, "length must be a multiple of 256");
    int blocks = static_cast<int>(x.numel() / 256);
    auto out = torch::empty({blocks}, x.options());
    block_sum_kernel<<<blocks, 256>>>(x.data_ptr<float>(), out.data_ptr<float>());
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "block_sums launch failed");
    return out;
}
"""

# Bug 3. Tool: racecheck. The last warp is unrolled by hand and relies on its
# lanes running in lockstep. Hint: read into a register, __syncwarp(), write,
# __syncwarp(). Using __shfl_down_sync instead is also a fix.
WARP_CUDA = r"""
#include <torch/extension.h>

__global__ void warp_block_sum_kernel(const float* x, float* out) {
    __shared__ float s[128];
    int t = threadIdx.x;
    s[t] = x[blockIdx.x * 128 + t];
    __syncthreads();
    if (t < 64) {
        s[t] += s[t + 64];
    }
    __syncthreads();
    if (t < 32) {
        volatile float* v = s;
        v[t] += v[t + 32];
        v[t] += v[t + 16];
        v[t] += v[t + 8];
        v[t] += v[t + 4];
        v[t] += v[t + 2];
        v[t] += v[t + 1];
        if (t == 0) {
            out[blockIdx.x] = v[0];
        }
    }
}

torch::Tensor warp_block_sums(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.is_contiguous());
    TORCH_CHECK(x.numel() % 128 == 0, "length must be a multiple of 128");
    int blocks = static_cast<int>(x.numel() / 128);
    auto out = torch::empty({blocks}, x.options());
    warp_block_sum_kernel<<<blocks, 128>>>(x.data_ptr<float>(), out.data_ptr<float>());
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "warp_block_sums launch failed");
    return out;
}
"""

# Bug 4. Tool: synccheck. Each output is a value plus its right neighbor in the
# same block (0 past the block's end). 1000 floats, 128 threads per block.
SYNC_CUDA = r"""
#include <torch/extension.h>

template <int B>
__global__ void tail_pair_sum_kernel(const float* x, float* out, int n) {
    __shared__ float tile[B];
    int t = threadIdx.x;
    int i = blockIdx.x * B + t;
    if (i >= n) {
        return;
    }
    tile[t] = x[i];
    __syncthreads();
    out[i] = tile[t] + ((t + 1 < B) ? tile[t + 1] : 0.0f);
}

torch::Tensor tail_pair_sum(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.is_contiguous());
    int n = static_cast<int>(x.numel());
    auto out = torch::empty_like(x);
    int grid = (n + 127) / 128;
    tail_pair_sum_kernel<128><<<grid, 128>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "tail_pair_sum launch failed");
    return out;
}
"""

# Bug 5. Tool: initcheck (shared memory). A 3-point stencil, zero outside the
# array: out[i] = x[i-1] + x[i] + x[i+1]. The tile holds one halo cell per side.
INIT_CUDA = r"""
#include <torch/extension.h>

template <int B>
__global__ void stencil3_kernel(const float* x, float* out, int n) {
    __shared__ float tile[B + 2];
    int t = threadIdx.x;
    int i = blockIdx.x * B + t;
    tile[t + 1] = (i < n) ? x[i] : 0.0f;
    __syncthreads();
    if (i < n) {
        out[i] = tile[t] + tile[t + 1] + tile[t + 2];
    }
}

torch::Tensor stencil3(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.is_contiguous());
    int n = static_cast<int>(x.numel());
    auto out = torch::empty_like(x);
    int grid = (n + 127) / 128;
    stencil3_kernel<128><<<grid, 128>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "stencil3 launch failed");
    return out;
}
"""


# ---------------------------------------------------------------------------
# Part 2. Index and hazard reasoning in plain Python.
# ---------------------------------------------------------------------------


def min_grid(n: int, block_dim: int) -> int:
    """Blocks needed to give every one of n elements a thread (a ceiling division)."""
    raise NotImplementedError("min_grid")


def oob_indices(n: int, block_dim: int, grid_dim: int, guarded: bool) -> list[int]:
    """Global thread indices at or past n that read or write memory.

    The global index is blockIdx * blockDim + threadIdx. An unguarded kernel
    lets every launched thread with index >= n touch memory. A guarded kernel
    (`if (i < n)`) lets none of them. Return the indices in increasing order.
    """
    raise NotImplementedError("oob_indices")


def barrier_arrivals(n: int, block_dim: int, block_idx: int, early_return: bool) -> tuple[int, bool]:
    """How many threads of one block reach a __syncthreads(), and whether that is divergent.

    With `early_return`, a thread whose global index is >= n leaves before the
    barrier. Return (arrivals, divergent). A barrier is divergent when some but
    not all of the block's threads arrive. A block in which no thread arrives
    left as a whole, and is not divergent.
    """
    raise NotImplementedError("barrier_arrivals")


def shared_hazards(block_dim, barriers, lockstep=False, warp_size=32):
    """Read-after-write hazards in a shared-memory tree reduction.

    Phase 0: every thread t writes s[t].
    Phase k, for k = 1 .. log2(block_dim): let d = block_dim >> k. Each thread
    t < d reads s[t] and s[t + d], then writes s[t].

    `barriers[k]` is a __syncthreads() between phase k and phase k + 1, so a
    block of 8 threads has 3 boundaries. A bool applies to every boundary.

    A hazard is a tuple (address, writer, reader): in some phase, `reader`
    reads `address`, the latest earlier write to it came from a different
    thread `writer`, and no barrier separates the two phases. With
    `lockstep=True`, treat threads in the same warp (thread // warp_size) as
    ordered. That was true before Volta and is not guaranteed now, so the
    default is False.

    Return the hazards as a sorted list. For 8 threads and no barriers there
    are 7: (1, 1, 0), (2, 2, 0), (3, 3, 1), (4, 4, 0), ...
    """
    raise NotImplementedError("shared_hazards")


# ---------------------------------------------------------------------------
# Part 3. Read a compute-sanitizer report.
# ---------------------------------------------------------------------------


def error_count(text: str):
    """The N in the `ERROR SUMMARY: N errors` line, or None when there is none.

    The line says `1 error` when N is 1. N counts every failing access, not
    the number of reports the tool printed, because the tool prints only the
    first few.
    """
    raise NotImplementedError("error_count")


def parse_report(text: str) -> list[dict]:
    """One dict per finding, in the order the tool printed them.

    A finding starts at a header line and runs until the next blank
    `=========` line. The report never names its tool, so you infer it from the
    header. Return a dict with these keys, using None when a report does not
    carry the field:

      tool         "memcheck" | "racecheck" | "synccheck" | "initcheck"
      kind         "invalid-global-read", "invalid-global-write",
                   "uninitialized-shared-read", "divergent-barrier",
                   "raw-hazard", "war-hazard", or "waw-hazard". The
                   memcheck and initcheck kinds are lower case: the space
                   without its underscores, then read or write.
      severity     "error" or "warning". Only a racecheck header says Warning.
      kernel       The function name alone: no "void ", no template
                   arguments, no parameter list.
      file         The source file's base name, for example "oob.cu".
      line         The source line, an int.
      thread       (x, y, z) of the thread. For a hazard, the first access
                   listed (the Write Thread of a RAW hazard).
      block        (x, y, z) of the block.
      size         The access size in bytes, for memcheck and initcheck.
      bytes_after  How many bytes past the nearest allocation, for memcheck.
      other_line, other_thread   The second access of a hazard.
      warp_level   True when a racecheck header says "(Warp Level Programming)".

    Lines such as "Host Frame:" and "Saved host backtrace" belong to a finding
    and carry nothing you grade.
    """
    raise NotImplementedError("parse_report")


def classify(finding: dict) -> str:
    """Name the bug behind one finding. Return one of these exact strings:

    "out-of-bounds write", "out-of-bounds read", "shared-memory race",
    "missing __syncwarp", "divergent __syncthreads", "uninitialized shared read",
    "uninitialized global read".

    A racecheck hazard at warp level is a missing __syncwarp. Any other
    racecheck hazard is a shared-memory race.
    """
    raise NotImplementedError("classify")


def is_clean(text: str) -> bool:
    """True when the run is clean: a summary of 0 errors and no finding at all.

    A warning is not an error in the summary, and the lab still counts it.
    """
    raise NotImplementedError("is_clean")
