"""Lab 11: tensor-core GEMM with WMMA, then mma.sync with ldmatrix. Reference solution."""

import torch
from lab_common import load_cuda

CPP_SRC = """
torch::Tensor wmma_gemm_padded(torch::Tensor a, torch::Tensor b);
torch::Tensor mma_gemm_padded(torch::Tensor a, torch::Tensor b);
"""

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <stdint.h>

using namespace nvcuda;
typedef __nv_bfloat16 bf16;

// Block tile 128 x 128, K step 32, eight warps. Each warp owns a 64 x 32 tile
// of C: warps are laid out 2 (down M) by 4 (across N).
constexpr int BM = 128, BN = 128, BK = 32, THREADS = 256;
// Shared-memory row strides, in elements. The 8 extra bf16 (16 bytes) skew
// each row so that ldmatrix and wmma loads do not hit the same banks.
constexpr int A_LD = BK + 8;   // 40 elements = 80 bytes
constexpr int B_LD = BN + 8;   // 136 elements = 272 bytes

// Copy the A tile (BM x BK) and the B tile (BK x BN) from global to shared
// memory with 16-byte loads. M, N, K are padded, so no bounds checks.
__device__ __forceinline__ void load_tiles(const bf16* A, const bf16* B, int N, int K,
                                           int block_m, int block_n, int k0,
                                           bf16* As, bf16* Bs) {
    int tid = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int c = tid + i * THREADS;          // 0..511
        int ra = c >> 2, ca = (c & 3) * 8;  // A: 128 rows x 4 chunks of 8 bf16
        *reinterpret_cast<uint4*>(&As[ra * A_LD + ca]) =
            *reinterpret_cast<const uint4*>(&A[(size_t)(block_m + ra) * K + k0 + ca]);
        int rb = c >> 4, cb = (c & 15) * 8; // B: 32 rows x 16 chunks of 8 bf16
        *reinterpret_cast<uint4*>(&Bs[rb * B_LD + cb]) =
            *reinterpret_cast<const uint4*>(&B[(size_t)(k0 + rb) * N + block_n + cb]);
    }
}

// ---------------------------------------------------------------- WMMA ----
__global__ void __launch_bounds__(THREADS)
wmma_kernel(const bf16* A, const bf16* B, float* C, int M, int N, int K) {
    __shared__ __align__(128) bf16 As[BM * A_LD];
    __shared__ __align__(128) bf16 Bs[BK * B_LD];

    int block_m = blockIdx.y * BM, block_n = blockIdx.x * BN;
    int warp = threadIdx.x >> 5;
    int wm = warp >> 2, wn = warp & 3;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][2];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        load_tiles(A, B, N, K, block_m, block_n, k0, As, Bs);
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a[4];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> b[2];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                wmma::load_matrix_sync(a[i], &As[(wm * 64 + i * 16) * A_LD + kk], A_LD);
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                wmma::load_matrix_sync(b[j], &Bs[kk * B_LD + wn * 32 + j * 16], B_LD);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(
                &C[(size_t)(block_m + wm * 64 + i * 16) * N + block_n + wn * 32 + j * 16],
                acc[i][j], N, wmma::mem_row_major);
}

// ------------------------------------------------------ mma.sync + ldmatrix
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// Four 8x8 matrices of 16-bit values, one 32-bit register per matrix per thread.
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

// The same, but each 8x8 matrix is delivered transposed.
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

// D = A * B + D for one 16x8x16 tile, bf16 inputs, fp32 accumulate.
__device__ __forceinline__ void mma_16816(float (&d)[4], const uint32_t (&a)[4],
                                          uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__global__ void __launch_bounds__(THREADS)
mma_kernel(const bf16* A, const bf16* B, float* C, int M, int N, int K) {
    __shared__ __align__(128) bf16 As[BM * A_LD];
    __shared__ __align__(128) bf16 Bs[BK * B_LD];

    int block_m = blockIdx.y * BM, block_n = blockIdx.x * BN;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int wm = warp >> 2, wn = warp & 3;

    // Lane l addresses row (l % 8) of matrix (l / 8). For both the A tile and
    // the B tile the four matrices are: (rows 0-7, cols 0-7), (rows 8-15,
    // cols 0-7), (rows 0-7, cols 8-15), (rows 8-15, cols 8-15).
    int mat = lane >> 3;
    int row_off = (lane & 7) + (mat & 1) * 8;
    int col_off = (mat >> 1) * 8;

    float acc[4][4][4];  // [m-tile][n-tile][c0..c3]
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            #pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        load_tiles(A, B, N, K, block_m, block_n, k0, As, Bs);
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t a[4][4];  // four 16x16 A tiles
            uint32_t b[2][4];  // two 16x16 B tiles = four 16x8 B fragments
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                ldmatrix_x4(a[i], smem_u32(&As[(wm * 64 + i * 16 + row_off) * A_LD + kk + col_off]));
            #pragma unroll
            for (int jp = 0; jp < 2; ++jp)
                ldmatrix_x4_trans(b[jp], smem_u32(&Bs[(kk + row_off) * B_LD + wn * 32 + jp * 16 + col_off]));
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    mma_16816(acc[i][j], a[i], b[j >> 1][(j & 1) * 2], b[j >> 1][(j & 1) * 2 + 1]);
        }
        __syncthreads();
    }

    // C fragment: lane = 4 * g + t holds rows g and g + 8, columns 2t and 2t + 1.
    int g = lane >> 2, t = lane & 3;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int row = block_m + wm * 64 + i * 16 + g;
            int col = block_n + wn * 32 + j * 8 + 2 * t;
            *reinterpret_cast<float2*>(&C[(size_t)row * N + col]) = make_float2(acc[i][j][0], acc[i][j][1]);
            *reinterpret_cast<float2*>(&C[(size_t)(row + 8) * N + col]) = make_float2(acc[i][j][2], acc[i][j][3]);
        }
}

// ------------------------------------------------------------- launchers --
static void check_inputs(const torch::Tensor& a, const torch::Tensor& b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(a.scalar_type() == torch::kBFloat16 && b.scalar_type() == torch::kBFloat16, "inputs must be bfloat16");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(a.size(1) == b.size(0), "inner dimensions differ");
    TORCH_CHECK(a.size(0) % BM == 0 && b.size(1) % BN == 0 && a.size(1) % BK == 0,
                "the padded sizes must be multiples of 128, 128, 32");
}

torch::Tensor wmma_gemm_padded(torch::Tensor a, torch::Tensor b) {
    check_inputs(a, b);
    int M = a.size(0), K = a.size(1), N = b.size(1);
    auto c = torch::empty({M, N}, a.options().dtype(torch::kFloat32));
    dim3 grid(N / BN, M / BM);
    wmma_kernel<<<grid, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const bf16*>(a.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bf16*>(b.data_ptr<at::BFloat16>()), c.data_ptr<float>(), M, N, K);
    return c;
}

torch::Tensor mma_gemm_padded(torch::Tensor a, torch::Tensor b) {
    check_inputs(a, b);
    int M = a.size(0), K = a.size(1), N = b.size(1);
    auto c = torch::empty({M, N}, a.options().dtype(torch::kFloat32));
    dim3 grid(N / BN, M / BM);
    mma_kernel<<<grid, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const bf16*>(a.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bf16*>(b.data_ptr<at::BFloat16>()), c.data_ptr<float>(), M, N, K);
    return c;
}
"""

BM, BN, BK = 128, 128, 32


def _module():
    return load_cuda(
        "lab11_gemm", CUDA_SRC, CPP_SRC, ["wmma_gemm_padded", "mma_gemm_padded"]
    )


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def _pad(a: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Zero-pad a 2-D tensor to (rows, cols). Returns the input if it already fits."""
    if tuple(a.shape) == (rows, cols):
        return a.contiguous()
    out = torch.zeros(rows, cols, dtype=a.dtype, device=a.device)
    out[: a.shape[0], : a.shape[1]] = a
    return out


def _run(kernel: str, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    k2, n = b.shape
    assert k == k2, "inner dimensions differ"
    mp, np_, kp = _round_up(m, BM), _round_up(n, BN), _round_up(k, BK)
    out = getattr(_module(), kernel)(_pad(a, mp, kp), _pad(b, kp, np_))
    if (mp, np_) != (m, n):
        out = out[:m, :n].contiguous()
    return out


def wmma_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = A @ B with the WMMA API. a: (M, K) bf16, b: (K, N) bf16, returns (M, N) fp32."""
    return _run("wmma_gemm_padded", a, b)


def mma_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = A @ B with ldmatrix and mma.sync. Same contract as wmma_gemm."""
    return _run("mma_gemm_padded", a, b)


def percent_of_torch(kernel_tflops: float, torch_tflops: float) -> float:
    """Your kernel's throughput as a percentage of torch.matmul's."""
    return 100.0 * kernel_tflops / torch_tflops
