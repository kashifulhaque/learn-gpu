"""Lab 10: SGEMM from scratch (worked solution).

Five fp32 kernels for C = A @ B on row-major matrices, from the naive one-thread-
per-output kernel to a vectorized 2D register-blocked kernel. Every kernel takes
float32 CUDA tensors A (M, K) and B (K, N) and returns a new (M, N) tensor, and
handles any M, N, and K, including sizes that don't divide the tile sizes.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Part 1: the arithmetic (pure Python)
# ---------------------------------------------------------------------------


def gemm_flops(m: int, n: int, k: int) -> int:
    """FLOPs of C = A @ B: one multiply and one add per (i, j, p), so 2*M*N*K."""
    return 2 * m * n * k


def min_bytes(m: int, n: int, k: int, dtype_bytes: int = 4) -> int:
    """Least DRAM traffic: read A and B once, write C once."""
    return dtype_bytes * (m * k + k * n + m * n)


def tile_intensity(bm: int, bn: int) -> float:
    """FLOP per byte of global traffic for a (bm x bn) output tile, fp32.

    Each step along K loads (bm + bn) floats and does bm * bn FMAs.
    """
    return (2 * bm * bn) / (4 * (bm + bn))


def smem_bytes_per_fma(tm: int, tn: int) -> float:
    """Shared-memory bytes a thread reads per FMA with a (tm x tn) register tile."""
    return 4 * (tm + tn) / (tm * tn)


# ---------------------------------------------------------------------------
# Part 2: the kernels
# ---------------------------------------------------------------------------

FUNCTIONS = [
    "sgemm_naive",
    "sgemm_coalesced",
    "sgemm_tiled",
    "sgemm_regblock",
    "sgemm_vectorized",
]

CPP_DECLARATIONS = """
torch::Tensor sgemm_naive(torch::Tensor a, torch::Tensor b);
torch::Tensor sgemm_coalesced(torch::Tensor a, torch::Tensor b);
torch::Tensor sgemm_tiled(torch::Tensor a, torch::Tensor b);
torch::Tensor sgemm_regblock(torch::Tensor a, torch::Tensor b);
torch::Tensor sgemm_vectorized(torch::Tensor a, torch::Tensor b);
"""

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// ---------------------------------------------------------------------------
// Kernel 1: naive. One thread per element of C. Consecutive threads in a warp
// (threadIdx.x) take consecutive ROWS, so a warp's loads of A are K floats
// apart: uncoalesced.
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
// Kernel 2: coalesced. Swap the roles: threadIdx.x walks along a row of C.
// A warp reads one A element (a broadcast) and 32 consecutive B elements.
// ---------------------------------------------------------------------------
__global__ void coalesced_kernel(int M, int N, int K, const float* A, const float* B, float* C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: shared-memory tiling. A 32 x 32 block computes a 32 x 32 tile of C
// and walks along K in steps of 32, staging one tile of A and one of B in
// shared memory per step. Out-of-range elements load as zero, so the partial
// tiles at the edges add nothing to the sum.
// ---------------------------------------------------------------------------
#define TILE 32

__global__ void tiled_kernel(int M, int N, int K, const float* A, const float* B, float* C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.f;
    for (int t = 0; t < K; t += TILE) {
        As[ty][tx] = (row < M && t + tx < K) ? A[row * K + t + tx] : 0.f;
        Bs[ty][tx] = (t + ty < K && col < N) ? B[(t + ty) * N + col] : 0.f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// Kernels 4 and 5: 2D register blocking, scalar and vectorized.
//
// A block computes a BM x BN tile of C with 256 threads. Each thread owns a
// TM x TN sub-tile held in registers. Per step along K (BK values), the block
// stages an BM x BK tile of A and a BK x BN tile of B in shared memory. A is
// stored transposed (As[k][m]) so that, for a fixed k, a thread's TM values
// are contiguous and can be read with two 128-bit loads. Each k step is an
// outer product: TM + TN shared-memory reads feed TM * TN FMAs.
//
// VEC = true loads global memory as float4, which needs K % 4 == 0, N % 4 == 0
// and 16-byte aligned base pointers. The host checks that and falls back to
// VEC = false (scalar loads, per-element guards) when it doesn't hold.
// ---------------------------------------------------------------------------
constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
constexpr int NT = (BM / TM) * (BN / TN);  // 256 threads

template <bool VEC>
__global__ void __launch_bounds__(NT)
regblock_kernel(int M, int N, int K, const float* __restrict__ A,
                const float* __restrict__ B, float* __restrict__ C) {
    __shared__ __align__(16) float As[BK * BM];  // As[k * BM + m]
    __shared__ __align__(16) float Bs[BK * BN];  // Bs[k * BN + n]

    const int tid = threadIdx.x;
    const int threadRow = tid / (BN / TN);  // 0..15
    const int threadCol = tid % (BN / TN);  // 0..15
    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;
    float regA[TM], regB[TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
        // ---- global -> shared ------------------------------------------------
        if (VEC) {
            // A tile: 128 rows x 8 columns = 256 float4, one per thread.
            const int m = tid / 2;
            const int k4 = (tid % 2) * 4;
            const int gr = rowBase + m, gk = k0 + k4;
            float4 va = make_float4(0.f, 0.f, 0.f, 0.f);
            if (gr < M && gk < K) va = *reinterpret_cast<const float4*>(&A[gr * K + gk]);
            As[(k4 + 0) * BM + m] = va.x;
            As[(k4 + 1) * BM + m] = va.y;
            As[(k4 + 2) * BM + m] = va.z;
            As[(k4 + 3) * BM + m] = va.w;
            // B tile: 8 rows x 128 columns = 256 float4, one per thread.
            const int kb = tid / 32;
            const int n4 = (tid % 32) * 4;
            const int gkb = k0 + kb, gc = colBase + n4;
            float4 vb = make_float4(0.f, 0.f, 0.f, 0.f);
            if (gkb < K && gc < N) vb = *reinterpret_cast<const float4*>(&B[gkb * N + gc]);
            *reinterpret_cast<float4*>(&Bs[kb * BN + n4]) = vb;
        } else {
            for (int i = tid; i < BM * BK; i += NT) {
                const int m = i / BK, k = i % BK;
                const int gr = rowBase + m, gk = k0 + k;
                As[k * BM + m] = (gr < M && gk < K) ? A[gr * K + gk] : 0.f;
            }
            for (int i = tid; i < BK * BN; i += NT) {
                const int k = i / BN, n = i % BN;
                const int gk = k0 + k, gc = colBase + n;
                Bs[k * BN + n] = (gk < K && gc < N) ? B[gk * N + gc] : 0.f;
            }
        }
        __syncthreads();

        // ---- shared -> registers -> FMAs --------------------------------------
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            if (VEC) {
                const float4* pa = reinterpret_cast<const float4*>(&As[kk * BM + threadRow * TM]);
                const float4* pb = reinterpret_cast<const float4*>(&Bs[kk * BN + threadCol * TN]);
                const float4 a0 = pa[0], a1 = pa[1], b0 = pb[0], b1 = pb[1];
                regA[0] = a0.x; regA[1] = a0.y; regA[2] = a0.z; regA[3] = a0.w;
                regA[4] = a1.x; regA[5] = a1.y; regA[6] = a1.z; regA[7] = a1.w;
                regB[0] = b0.x; regB[1] = b0.y; regB[2] = b0.z; regB[3] = b0.w;
                regB[4] = b1.x; regB[5] = b1.y; regB[6] = b1.z; regB[7] = b1.w;
            } else {
#pragma unroll
                for (int i = 0; i < TM; ++i) regA[i] = As[kk * BM + threadRow * TM + i];
#pragma unroll
                for (int j = 0; j < TN; ++j) regB[j] = Bs[kk * BN + threadCol * TN + j];
            }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += regA[i] * regB[j];
        }
        __syncthreads();
    }

    // ---- registers -> global ----------------------------------------------------
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gr = rowBase + threadRow * TM + i;
        if (gr >= M) continue;
        if (VEC) {
#pragma unroll
            for (int j = 0; j < TN; j += 4) {
                const int gc = colBase + threadCol * TN + j;
                if (gc < N)
                    *reinterpret_cast<float4*>(&C[gr * N + gc]) =
                        make_float4(acc[i][j], acc[i][j + 1], acc[i][j + 2], acc[i][j + 3]);
            }
        } else {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int gc = colBase + threadCol * TN + j;
                if (gc < N) C[gr * N + gc] = acc[i][j];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host wrappers.
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

torch::Tensor sgemm_coalesced(torch::Tensor a, torch::Tensor b) {
    GEMM_PROLOGUE
    dim3 block(32, 32);
    dim3 grid(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
    coalesced_kernel<<<grid, block, 0, stream>>>(M, N, K, a.data_ptr<float>(),
                                                 b.data_ptr<float>(), c.data_ptr<float>());
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
    regblock_kernel<false><<<grid, NT, 0, stream>>>(M, N, K, a.data_ptr<float>(),
                                                    b.data_ptr<float>(), c.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}

torch::Tensor sgemm_vectorized(torch::Tensor a, torch::Tensor b) {
    GEMM_PROLOGUE
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    const bool aligned = (K % 4 == 0) && (N % 4 == 0) &&
                         (reinterpret_cast<uintptr_t>(a.data_ptr<float>()) % 16 == 0) &&
                         (reinterpret_cast<uintptr_t>(b.data_ptr<float>()) % 16 == 0) &&
                         (reinterpret_cast<uintptr_t>(c.data_ptr<float>()) % 16 == 0);
    if (aligned) {
        regblock_kernel<true><<<grid, NT, 0, stream>>>(M, N, K, a.data_ptr<float>(),
                                                       b.data_ptr<float>(), c.data_ptr<float>());
    } else {
        regblock_kernel<false><<<grid, NT, 0, stream>>>(M, N, K, a.data_ptr<float>(),
                                                        b.data_ptr<float>(), c.data_ptr<float>());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}
"""


def cuda_source() -> str:
    """Return the CUDA source that defines every function in FUNCTIONS."""
    return _CUDA_SOURCE
