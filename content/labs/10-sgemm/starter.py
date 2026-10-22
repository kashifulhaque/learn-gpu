"""Lab 10: SGEMM from scratch.

Write fp32 kernels for C = A @ B on row-major matrices. Every kernel takes
float32 CUDA tensors A (M, K) and B (K, N) and returns a new (M, N) tensor, and
it must be correct for every M, N, and K, including sizes that are not
multiples of the tile sizes.

The tests compile `cuda_source()` with `CPP_DECLARATIONS`, check every function
named in `FUNCTIONS` against a float64 reference, then time each one at
4096 x 4096 x 4096 against torch.matmul with TF32 turned off (cuBLAS SGEMM on
the CUDA cores).

What to do, in order:
  1. Fill in the four pure-Python helpers in Part 1.
  2. Read the naive kernel in Part 2. It is complete, so you can see the
     conventions: int32 indexing, the host wrapper, and the launch.
  3. Write `tiled_kernel` (shared-memory tiling).
  4. Write `regblock_kernel` (a 128 x 128 block tile, a thread tile of 8 x 8).
  5. Optional: add `sgemm_coalesced` and `sgemm_vectorized`. Define each
     function in the CUDA source, then add its name to FUNCTIONS and its
     signature to CPP_DECLARATIONS. The tests measure whatever FUNCTIONS lists.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Part 1: the arithmetic (pure Python)
# ---------------------------------------------------------------------------


def gemm_flops(m: int, n: int, k: int) -> int:
    """Return the FLOPs of C = A @ B, counting a multiply and an add as two."""
    # TODO
    raise NotImplementedError


def min_bytes(m: int, n: int, k: int, dtype_bytes: int = 4) -> int:
    """Return the least DRAM traffic in bytes: read A and B once, write C once."""
    # TODO
    raise NotImplementedError


def tile_intensity(bm: int, bn: int) -> float:
    """Return FLOP per byte of global traffic for a (bm x bn) output tile, fp32.

    One step along K loads bm values of A and bn values of B, and does bm * bn
    FMAs. Count 4 bytes per value and 2 FLOP per FMA.
    """
    # TODO
    raise NotImplementedError


def smem_bytes_per_fma(tm: int, tn: int) -> float:
    """Return the shared-memory bytes one thread reads per FMA.

    A thread that owns a (tm x tn) tile of C reads tm values of A and tn values
    of B per k step, and does tm * tn FMAs with them. Count 4 bytes per value.
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 2: the kernels
# ---------------------------------------------------------------------------

# The functions the tests compile, check, and time. "sgemm_naive",
# "sgemm_tiled", and "sgemm_regblock" are required.
FUNCTIONS = ["sgemm_naive", "sgemm_tiled", "sgemm_regblock"]

CPP_DECLARATIONS = """
torch::Tensor sgemm_naive(torch::Tensor a, torch::Tensor b);
torch::Tensor sgemm_tiled(torch::Tensor a, torch::Tensor b);
torch::Tensor sgemm_regblock(torch::Tensor a, torch::Tensor b);
"""

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// ---------------------------------------------------------------------------
// Kernel 1: naive (complete). One thread per element of C. Consecutive threads
// in a warp (threadIdx.x) take consecutive ROWS, so a warp's loads of A are K
// floats apart: uncoalesced. Swapping the roles of x and y fixes it, which is
// the optional sgemm_coalesced.
// ---------------------------------------------------------------------------
__global__ void naive_kernel(int M, int N, int K, const float* A, const float* B, float* C) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: shared-memory tiling.
//
// Launch: a (TILE, TILE) block per (TILE x TILE) tile of C; grid.x walks the
// columns of C and grid.y walks the rows. Thread (tx, ty) owns C[row][col].
//
// TODO: for each step t along K in units of TILE:
//   1. Load one element of A and one of B into the shared tiles As and Bs.
//      An element outside the matrix loads as 0.f, so edge tiles add nothing.
//   2. __syncthreads()
//   3. acc += As[ty][k] * Bs[k][tx] for k in [0, TILE)
//   4. __syncthreads() again, before the next step overwrites the tiles.
// Then store acc if (row, col) is inside C. Every thread must reach both
// __syncthreads() calls, so guard the loads and the store, not the loop.
// ---------------------------------------------------------------------------
#define TILE 32

__global__ void tiled_kernel(int M, int N, int K, const float* A, const float* B, float* C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    // TODO
}

// ---------------------------------------------------------------------------
// Kernel 4: 2D register blocking.
//
// A block of NT = 256 threads computes a BM x BN = 128 x 128 tile of C. Thread
// `tid` owns a TM x TN = 8 x 8 sub-tile of it, held in registers:
//     threadRow = tid / (BN / TN),  threadCol = tid % (BN / TN)   (both 0..15)
//     rows  rowBase + threadRow * TM + [0, TM)
//     cols  colBase + threadCol * TN + [0, TN)
//
// TODO: for each step k0 along K in units of BK = 8:
//   1. Cooperatively load the BM x BK tile of A and the BK x BN tile of B
//      into shared memory, zero-filling outside the matrix. Store A
//      TRANSPOSED, As[k * BM + m], so a thread's TM values for one k are
//      contiguous. The 256 threads load 1024 values of each tile, four each:
//      use `for (int i = tid; i < BM * BK; i += NT)`.
//   2. __syncthreads()
//   3. for each kk in [0, BK): read regA[TM] from As and regB[TN] from Bs,
//      then acc[i][j] += regA[i] * regB[j] for every i, j (an outer product).
//   4. __syncthreads()
// Then write the TM x TN results to C, skipping rows and columns outside it.
// ---------------------------------------------------------------------------
constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
constexpr int NT = (BM / TM) * (BN / TN);  // 256 threads

__global__ void __launch_bounds__(NT)
regblock_kernel(int M, int N, int K, const float* __restrict__ A,
                const float* __restrict__ B, float* __restrict__ C) {
    __shared__ __align__(16) float As[BK * BM];  // As[k * BM + m]
    __shared__ __align__(16) float Bs[BK * BN];  // Bs[k * BN + n]
    // TODO
}

// ---------------------------------------------------------------------------
// Host wrappers (complete).
// ---------------------------------------------------------------------------
static void check_inputs(const torch::Tensor& a, const torch::Tensor& b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32 && b.scalar_type() == torch::kFloat32,
                "inputs must be float32");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(0),
                "shapes must be (M, K) and (K, N)");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(a.numel() < (1LL << 31) && b.numel() < (1LL << 31) &&
                    a.size(0) * b.size(1) < (1LL << 31),
                "int32 indexing: each matrix must have fewer than 2^31 elements");
}

#define GEMM_PROLOGUE                                  \
    check_inputs(a, b);                                \
    const int M = (int)a.size(0), K = (int)a.size(1), N = (int)b.size(1); \
    auto c = torch::empty({M, N}, a.options());        \
    if (M == 0 || N == 0) return c;                    \
    auto stream = c10::cuda::getCurrentCUDAStream();

torch::Tensor sgemm_naive(torch::Tensor a, torch::Tensor b) {
    GEMM_PROLOGUE
    dim3 block(32, 32);
    dim3 grid(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    naive_kernel<<<grid, block, 0, stream>>>(M, N, K, a.data_ptr<float>(), b.data_ptr<float>(),
                                             c.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}

torch::Tensor sgemm_tiled(torch::Tensor a, torch::Tensor b) {
    GEMM_PROLOGUE
    dim3 block(TILE, TILE);
    dim3 grid(CEIL_DIV(N, TILE), CEIL_DIV(M, TILE));
    tiled_kernel<<<grid, block, 0, stream>>>(M, N, K, a.data_ptr<float>(), b.data_ptr<float>(),
                                             c.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}

torch::Tensor sgemm_regblock(torch::Tensor a, torch::Tensor b) {
    GEMM_PROLOGUE
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    regblock_kernel<<<grid, NT, 0, stream>>>(M, N, K, a.data_ptr<float>(), b.data_ptr<float>(),
                                             c.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}
"""


def cuda_source() -> str:
    """Return the CUDA source that defines every function in FUNCTIONS."""
    return _CUDA_SOURCE
