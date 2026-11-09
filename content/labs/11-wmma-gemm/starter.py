"""Lab 11: tensor-core GEMM with WMMA, then mma.sync with ldmatrix.

Do the CPU pieces first. They need no GPU.

1. fragment_cell
2. warp_tile_registers
3. ldmatrix_byte

Then fill in the TODOs in the CUDA source below. The scaffolding (the global
to shared-memory copy, the padding wrapper, the launchers) is done. The tile
geometry is fixed: a 128 x 128 block tile, K step 32, eight warps in a 2 x 4
grid, each warp owning a 64 x 32 piece of C. A is (M, K) row-major bf16, B is
(K, N) row-major bf16, C is fp32. Shared memory holds A as [128][40] and B as
[32][136] bf16.
"""

from __future__ import annotations

from lab_common import load_cuda


def fragment_cell(operand: str, lane: int, index: int) -> tuple[int, int]:
    """Row and column of one element in an m16n8k16 fragment.

    `operand` is `"A"`, `"B"`, or `"C"` (C and D share the layout). `index`
    is the scalar inside the lane's registers: 0 to 7 for A, two bf16 values
    in each of `a0` to `a3`, and 0 to 3 for B and for C. Group `g` is
    `lane // 4` and thread `t` is `lane % 4`.

    A, element `index`: row `g` when `index` is 0, 1, 4, or 5, and row `g + 8`
    otherwise. Column is `2 * t + (index % 2)`, plus 8 when `index >= 4`.
    B, element `index`: row `2 * t + (index % 2)`, plus 8 when `index >= 2`,
    and column `g`. C, element `index`: row `g` when `index < 2` and row
    `g + 8` otherwise, column `2 * t + (index % 2)`.

    Lane 5 on A is the worked case: rows 1 and 9, columns 2 and 3, then
    columns 10 and 11.
    """
    raise NotImplementedError("fragment_cell")


def warp_tile_registers(warp_m: int, warp_n: int) -> tuple[int, int, int]:
    """Accumulator, A-fragment, and B-fragment registers per lane.

    One K step is 16. Accumulators are `warp_m * warp_n / 32` fp32 registers.
    A fragments are `(warp_m / 16) * 4` registers. B fragments are
    `(warp_n / 8) * 2` registers. Return those three counts, in that order.

    A 64 by 32 tile is `(64, 16, 8)`. A 16 by 8 tile, one `mma.sync`, is
    `(4, 4, 2)`. A 128 by 64 tile is `(256, 32, 16)`.
    """
    raise NotImplementedError("warp_tile_registers")


def ldmatrix_byte(lane: int, stride_elements: int, swizzle: bool) -> int:
    """Byte address of the 16-byte row this lane supplies to `ldmatrix.x4`.

    The 16 by 16 tile starts at the origin of a bf16 shared-memory tile whose
    row stride is `stride_elements`. Lane `l` addresses row `l % 8` of matrix
    `l // 8`:

    `row = (lane % 8) + 8 * ((lane // 8) % 2)` and `col = 8 * (lane // 16)`.

    When `swizzle` is true the row is 64 bf16 elements and the 8-row swizzle
    applies: chunk `col // 8` becomes `chunk XOR (row % 8)`. The byte address
    is `2 * (row * stride_elements + col)`.

    Lane 1 with stride 64 starts at byte 128. With the swizzle on, that row
    starts at byte 144.
    """
    raise NotImplementedError("ldmatrix_byte")

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

    // TODO 1 (WMMA): the warp owns a 64 x 32 tile of C, which is 4 x 2 tiles of 16 x 16.
    //   - Declare wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][2]
    //     and fill each with 0.
    //   - For each k0 in steps of BK: load_tiles(...), __syncthreads(); then for each
    //     kk in {0, 16}: load 4 matrix_a fragments from As and 2 matrix_b fragments
    //     from Bs with load_matrix_sync (the leading dimension is A_LD or B_LD),
    //     and call mma_sync for each of the 8 (i, j) pairs; then __syncthreads().
    //   - Store each accumulator to C with store_matrix_sync (leading dimension N,
    //     wmma::mem_row_major).
    //   Fragment pointers must be 32-byte aligned; the skewed strides keep them so.
    (void)A; (void)B; (void)C; (void)M; (void)N; (void)K;
    (void)block_m; (void)block_n; (void)wm; (void)wn;
}

// ------------------------------------------------------ mma.sync + ldmatrix
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// Four 8x8 matrices of 16-bit values, one 32-bit register per matrix per thread.
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
    // TODO 2: ldmatrix.sync.aligned.m8n8.x4.shared.b16 {r0, r1, r2, r3}, [addr];
    r[0] = r[1] = r[2] = r[3] = 0; (void)addr;
}

// The same, but each 8x8 matrix is delivered transposed.
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], uint32_t addr) {
    // TODO 3: the same with the .trans qualifier (before .shared).
    r[0] = r[1] = r[2] = r[3] = 0; (void)addr;
}

// D = A * B + D for one 16x8x16 tile, bf16 inputs, fp32 accumulate.
__device__ __forceinline__ void mma_16816(float (&d)[4], const uint32_t (&a)[4],
                                          uint32_t b0, uint32_t b1) {
    // TODO 4: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 with D and C both
    // the four accumulator floats ("+f" constraints), A the four registers in a,
    // and B the two registers b0 and b1.
    (void)d; (void)a; (void)b0; (void)b1;
}

__global__ void __launch_bounds__(THREADS)
mma_kernel(const bf16* A, const bf16* B, float* C, int M, int N, int K) {
    __shared__ __align__(128) bf16 As[BM * A_LD];
    __shared__ __align__(128) bf16 Bs[BK * B_LD];

    int block_m = blockIdx.y * BM, block_n = blockIdx.x * BN;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int wm = warp >> 2, wn = warp & 3;

    // TODO 5 (ldmatrix addressing): lane l supplies the row address of matrix (l / 8),
    // row (l % 8). For both the A tile and the B tile, the four 8x8 matrices are
    // (rows 0-7, cols 0-7), (rows 8-15, cols 0-7), (rows 0-7, cols 8-15), and
    // (rows 8-15, cols 8-15). Work out row_off and col_off from `lane`.
    //
    // TODO 6 (main loop): keep float acc[4][4][4] = 0 (4 m-tiles x 4 n-tiles of
    // m16n8, 4 floats each). For each k0 in steps of BK: load_tiles, __syncthreads;
    // for each kk in {0, 16}: ldmatrix_x4 four A tiles from As, ldmatrix_x4_trans two
    // B tiles from Bs (each gives two n8 fragments: registers 0-1 and 2-3), then 16
    // calls to mma_16816; __syncthreads.
    //
    // TODO 7 (epilogue): lane = 4 * g + t holds C rows g and g + 8, columns 2t and
    // 2t + 1 of each m16n8 tile: c0, c1 in row g and c2, c3 in row g + 8. Store each
    // pair as one float2.
    (void)A; (void)B; (void)C; (void)M; (void)N; (void)K; (void)lane;
    (void)block_m; (void)block_n; (void)wm; (void)wn; (void)As; (void)Bs;
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
    import torch

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
