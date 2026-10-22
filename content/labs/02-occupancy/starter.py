"""Lab 02: Occupancy and divergence.

You write three things. In Part 1, a Python occupancy calculator for the A100
(`occupancy`, `max_regs_for_occupancy`). In Part 2, which is already written,
the file compiles CUDA kernels so the tests can compare your calculator with
the CUDA occupancy API. In Part 3, you fix one line of CUDA (`group_of`) and
measure what warp divergence costs (`branch_time_ms`, `divergence_slowdown`).
The tests import the functions below.
"""

import torch
from lab_common import A100, cuda_time, load_cuda

# ---------------------------------------------------------------------------
# Part 1: an occupancy calculator for compute capability 8.0.
# ---------------------------------------------------------------------------

# The limits in lab_common.A100 come from the CUDA Programming Guide. The four
# numbers below are allocation granularities. They come from NVIDIA's retired
# occupancy-calculator spreadsheet, not from a
# table in the Programming Guide, so your lab cross-checks them against the
# CUDA occupancy API.
REG_ALLOC_UNIT = 256  # registers are charged to a warp in units of 256
SUBPARTITIONS = 4  # warps are placed in four groups, one per warp scheduler
SMEM_ALLOC_UNIT = 128  # shared memory is charged in 128-byte units
SMEM_RESERVED_PER_BLOCK = 1024  # the runtime reserves 1 KiB of shared memory per block

MAX_WARPS_PER_SM = A100.MAX_THREADS_PER_SM // A100.WARP_SIZE  # 64


def _round_up(value: int, unit: int) -> int:
    return -(-value // unit) * unit


def occupancy(threads_per_block: int, regs_per_thread: int, smem_per_block: int) -> dict:
    """Resident blocks per SM and occupancy for one launch configuration.

    Returns a dict with these keys:

    - "limits": a dict with the number of blocks each resource allows, under the
      keys "warps", "blocks", "registers", and "shared_memory"
    - "blocks_per_sm": the smallest of the four limits, an int (0 if the block
      cannot launch at all)
    - "warps_per_sm": blocks_per_sm times warps per block
    - "occupancy": warps_per_sm divided by the 64 warps an SM can hold, a float
    - "limiter": the name of the smallest limit. On a tie, take the first of
      "registers", "shared_memory", "blocks", "warps".

    `smem_per_block` is static plus dynamic shared memory, in bytes.
    """
    # TODO: work out four limits, in blocks per SM.
    #   warps:         64 warps per SM, divided by the warps in one block.
    #   blocks:        the 32-block cap (A100.MAX_BLOCKS_PER_SM).
    #   registers:     charge each warp max(regs, 1) * 32 registers, rounded up
    #                  to a multiple of REG_ALLOC_UNIT. A quarter of the
    #                  register file belongs to each of the SUBPARTITIONS, and a
    #                  warp lives in one of them. Count whole warps per
    #                  sub-partition, multiply by SUBPARTITIONS, and divide by
    #                  the warps in one block.
    #   shared_memory: charge round_up(smem_per_block, SMEM_ALLOC_UNIT) plus
    #                  SMEM_RESERVED_PER_BLOCK, and divide the SM's
    #                  A100.SMEM_PER_SM_BYTES by it.
    # A block that breaks a per-block limit (more than 1024 threads, more than
    # 255 registers per thread, more than A100.SMEM_PER_BLOCK_MAX_BYTES of
    # shared memory) cannot launch: set the matching limit to 0.
    # Then blocks_per_sm is the minimum, and the limiter is the limit that
    # equals it.
    raise NotImplementedError


def max_regs_for_occupancy(threads_per_block: int, target: float) -> int:
    """The most registers per thread that still reach `target` occupancy.

    Try every count from 255 down to 1 with no shared memory and return the
    first whose occupancy is at least `target`. Return 0 if none does.
    """
    # TODO: try 255, 254, ... 1 and return the first count that reaches `target`.
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 2: CUDA kernels for the occupancy API and for divergence.
# ---------------------------------------------------------------------------

CPP_DECLARATIONS = """
torch::Tensor branchy(torch::Tensor x, torch::Tensor coef, int64_t ways, int64_t by_warp, int64_t iters);
int64_t probe_registers(int64_t variant);
int64_t probe_static_smem(int64_t variant);
int64_t api_blocks_per_sm(int64_t variant, int64_t block, int64_t dyn_smem);
"""

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Which branch this thread takes. `ways` branches exist, numbered 0 to ways-1.
// by_warp == 0: neighbouring lanes of one warp take different branches.
// by_warp == 1: every lane of a warp takes the same branch, and different
//               warps take different ones.
__device__ __forceinline__ int group_of(int ways, int by_warp) {
    // TODO: return the branch number, from 0 to ways-1.
    //   by_warp == 0: derive it from the lane, threadIdx.x % 32, so the lanes
    //                 of one warp spread across all `ways` branches.
    //   by_warp == 1: derive it from the warp number, threadIdx.x / 32, so
    //                 all 32 lanes of a warp agree.
    // Until you write it, every thread takes branch 0 and the results are wrong.
    return 0;
}

// Every branch runs the same amount of work: `iters` dependent fused
// multiply-adds with its own coefficient. A warp whose lanes want different
// branches runs them one after another.
__global__ void branchy_kernel(const float* x, float* y, const float* coef,
                               int n, int ways, int by_warp, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int g = group_of(ways, by_warp);
    float v = x[i];
    for (int p = 0; p < ways; ++p) {
        if (g == p) {
            float c = coef[p];
            for (int k = 0; k < iters; ++k) v = fmaf(v, c, 1.0f);
        }
    }
    y[i] = v;
}

// Block size must stay 256: group_of reads threadIdx.x directly.
torch::Tensor branchy(torch::Tensor x, torch::Tensor coef, int64_t ways,
                      int64_t by_warp, int64_t iters) {
    auto y = torch::empty_like(x);
    int n = x.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    branchy_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), coef.data_ptr<float>(),
        n, (int)ways, (int)by_warp, (int)iters);
    return y;
}

// Four kernels that need different numbers of registers: ACC independent
// values are loaded first and combined afterwards, so about ACC registers are
// live at once. The compiler decides the real count; the host reads it back.
// They are never launched, only passed to the occupancy API.
template <int ACC>
__global__ void probe_kernel(const float* x, float* y, int n) {
    extern __shared__ float scratch[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float acc[ACC];
    #pragma unroll
    for (int j = 0; j < ACC; ++j) acc[j] = x[(i + j * 977) % n];
    float s = 0.0f;
    #pragma unroll
    for (int j = 0; j < ACC; ++j) s = fmaf(acc[j], acc[(j + 1) % ACC], s);
    if (n < 0) scratch[threadIdx.x] = s;  // keeps the dynamic shared array alive
    y[i] = s;
}

typedef void (*probe_fn)(const float*, float*, int);

static const void* probe_by_variant(int64_t variant) {
    probe_fn f;
    switch (variant) {
        case 0: f = probe_kernel<2>; break;
        case 1: f = probe_kernel<16>; break;
        case 2: f = probe_kernel<64>; break;
        default: f = probe_kernel<128>; break;
    }
    return (const void*)f;
}

int64_t probe_registers(int64_t variant) {
    cudaFuncAttributes a;
    TORCH_CHECK(cudaFuncGetAttributes(&a, probe_by_variant(variant)) == cudaSuccess,
                "cudaFuncGetAttributes failed");
    return a.numRegs;
}

int64_t probe_static_smem(int64_t variant) {
    cudaFuncAttributes a;
    TORCH_CHECK(cudaFuncGetAttributes(&a, probe_by_variant(variant)) == cudaSuccess,
                "cudaFuncGetAttributes failed");
    return (int64_t)a.sharedSizeBytes;
}

// What the driver says: how many blocks of `block` threads and `dyn_smem`
// bytes of dynamic shared memory fit on one SM. Returns -1 on an API error.
int64_t api_blocks_per_sm(int64_t variant, int64_t block, int64_t dyn_smem) {
    int n = 0;
    cudaError_t e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &n, probe_by_variant(variant), (int)block, (size_t)dyn_smem);
    if (e != cudaSuccess) {
        cudaGetLastError();
        return -1;
    }
    return n;
}
"""

FUNCTIONS = ["branchy", "probe_registers", "probe_static_smem", "api_blocks_per_sm"]

_module = None
_coef = None


def _ext():
    global _module
    if _module is None:
        _module = load_cuda("occupancy_lab", CUDA_SRC, CPP_DECLARATIONS, FUNCTIONS)
    return _module


def coefficients() -> torch.Tensor:
    """The 32 branch coefficients, 0.5 + 0.01 * p. Each is below 1, so the chain converges."""
    global _coef
    if _coef is None:
        _coef = (0.5 + 0.01 * torch.arange(32, dtype=torch.float32)).cuda()
    return _coef


def run_branchy(x: torch.Tensor, ways: int, by_warp: bool, iters: int = 1024) -> torch.Tensor:
    """Run the branchy kernel on a float32 CUDA tensor and return the result."""
    return _ext().branchy(x, coefficients(), ways, int(by_warp), iters)


def probe_registers(variant: int) -> int:
    """Registers per thread the compiler gave probe kernel `variant` (0 to 3)."""
    return int(_ext().probe_registers(variant))


def probe_static_smem(variant: int) -> int:
    """Static shared memory, in bytes, of probe kernel `variant`."""
    return int(_ext().probe_static_smem(variant))


def api_blocks_per_sm(variant: int, block: int, dyn_smem: int) -> int:
    """Blocks per SM according to cudaOccupancyMaxActiveBlocksPerMultiprocessor."""
    return int(_ext().api_blocks_per_sm(variant, block, dyn_smem))


# ---------------------------------------------------------------------------
# Part 3: what divergence costs.
# ---------------------------------------------------------------------------


def branch_time_ms(ways: int, by_warp: bool, n: int = 2**22, iters: int = 1024) -> float:
    """Median milliseconds of one launch of the branchy kernel on `n` floats.

    Time `run_branchy(x, ways, by_warp, iters)` with `cuda_time`.
    """
    # TODO: make x = torch.rand(n, device="cuda") and return cuda_time of a call to
    # run_branchy(x, ways, by_warp, iters). warmup=3 and iters=10 are plenty.
    raise NotImplementedError


def divergence_slowdown(ways: int, n: int = 2**22, iters: int = 1024) -> float:
    """How many times slower a `ways`-way lane-diverged warp is than a uniform one.

    Divide the time with `ways` branches taken by lanes of the same warp by the
    time with one branch.
    """
    # TODO: time ways=1 and `ways` with by_warp=False, and return diverged / uniform.
    raise NotImplementedError
