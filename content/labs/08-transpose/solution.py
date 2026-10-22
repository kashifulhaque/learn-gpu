"""Lab 08 solution: bank model in Python, four transposes in CUDA."""

from collections import defaultdict

# ---------------------------------------------------------------------------
# Part 1: the bank model
# ---------------------------------------------------------------------------

LAYOUTS = ("plain", "padded", "swizzled")


def tile_dim(elem_bytes: int) -> int:
    """Elements per tile side: 32 for 4-byte elements, 64 for 2-byte elements."""
    return 32 if elem_bytes == 4 else 64


def bank(byte_address: int) -> int:
    """Shared-memory bank of a byte address: 32 banks, each 4 bytes wide."""
    return (byte_address // 4) % 32


def conflict_degree(byte_addresses) -> int:
    """Transactions a warp's access needs: the most distinct words in one bank."""
    words_per_bank = defaultdict(set)
    for address in byte_addresses:
        words_per_bank[bank(address)].add(address // 4)
    return max(len(words) for words in words_per_bank.values())


def smem_index(layout: str, r: int, c: int, elem_bytes: int) -> int:
    """Element index of tile[r][c] inside the shared-memory array."""
    t = tile_dim(elem_bytes)
    w = 4 // elem_bytes  # elements per 32-bit word
    if layout == "plain":
        return r * t + c
    if layout == "padded":
        return r * (t + w) + c
    if layout == "swizzled":
        return r * t + (c ^ ((r & 31) * w))
    raise ValueError(layout)


def tile_bytes(layout: str, elem_bytes: int) -> int:
    """Bytes of shared memory the tile occupies."""
    t = tile_dim(elem_bytes)
    w = 4 // elem_bytes
    pitch = t + w if layout == "padded" else t
    return t * pitch * elem_bytes


def column_read_degree(layout: str, elem_bytes: int, c: int, row0: int = 0) -> int:
    """Degree of the warp access in which lane l reads tile[row0 + l][c]."""
    addresses = [smem_index(layout, row0 + lane, c, elem_bytes) * elem_bytes for lane in range(32)]
    return conflict_degree(addresses)


# ---------------------------------------------------------------------------
# Part 2: the CUDA kernels
# ---------------------------------------------------------------------------

CPP_DECLARATIONS = """
torch::Tensor transpose(torch::Tensor x, int64_t variant);
"""


def cuda_source() -> str:
    return r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

// One block is 32 x 8 threads. Each block owns a TILE x TILE tile of the input,
// so each thread handles (TILE / 8) * (TILE / 32) elements.
constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 8;

enum Layout { PLAIN = 0, PADDED = 1, SWIZZLED = 2 };

// Pitch of one tile row, in elements. The padded tile adds one 32-bit word.
template <typename T, int TILE, int LAYOUT>
__host__ __device__ constexpr int tile_pitch() {
    return LAYOUT == PADDED ? TILE + 4 / (int)sizeof(T) : TILE;
}

template <typename T, int TILE, int LAYOUT>
__host__ __device__ constexpr int tile_elems() {
    return TILE * tile_pitch<T, TILE, LAYOUT>();
}

// Element index of tile[r][c]. This mirrors smem_index in Python.
template <typename T, int TILE, int LAYOUT>
__device__ __forceinline__ int smem_index(int r, int c) {
    constexpr int W = 4 / (int)sizeof(T);  // elements per 32-bit word
    if constexpr (LAYOUT == SWIZZLED) {
        return r * TILE + (c ^ ((r & 31) * W));
    } else {
        return r * tile_pitch<T, TILE, LAYOUT>() + c;
    }
}

// Reads rows of the input, writes columns of the output. Reads coalesce; each
// warp's write touches one 4- or 2-byte element in each of 32 different rows.
template <typename T, int TILE>
__global__ void transpose_naive(const T* __restrict__ in, T* __restrict__ out, int M, int N) {
    const int n0 = blockIdx.x * TILE + threadIdx.x;
    const int m0 = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BLOCK_Y) {
        for (int i = 0; i < TILE; i += BLOCK_X) {
            const int m = m0 + j;
            const int n = n0 + i;
            if (m < M && n < N) {
                out[(size_t)n * M + m] = in[(size_t)m * N + n];
            }
        }
    }
}

// Stages the tile in shared memory so both the global read and the global
// write are coalesced. LAYOUT decides where each element lives in the tile.
template <typename T, int TILE, int LAYOUT>
__global__ void transpose_tiled(const T* __restrict__ in, T* __restrict__ out, int M, int N) {
    __shared__ T tile[tile_elems<T, TILE, LAYOUT>()];
    const int n_base = blockIdx.x * TILE;  // first input column of this tile
    const int m_base = blockIdx.y * TILE;  // first input row of this tile

    // Phase 1: lane x reads consecutive input columns of row r.
    for (int j = 0; j < TILE; j += BLOCK_Y) {
        for (int i = 0; i < TILE; i += BLOCK_X) {
            const int r = threadIdx.y + j;
            const int c = threadIdx.x + i;
            const int m = m_base + r;
            const int n = n_base + c;
            if (m < M && n < N) {
                tile[smem_index<T, TILE, LAYOUT>(r, c)] = in[(size_t)m * N + n];
            }
        }
    }
    __syncthreads();  // Every thread reaches this, so it sits outside the guards.

    // Phase 2: lane x writes consecutive output columns of output row o_row, so
    // it reads tile[o_col][o_row]: a column of the tile.
    for (int j = 0; j < TILE; j += BLOCK_Y) {
        for (int i = 0; i < TILE; i += BLOCK_X) {
            const int o_row = threadIdx.y + j;  // input column within the tile
            const int o_col = threadIdx.x + i;  // input row within the tile
            const int n = n_base + o_row;
            const int m = m_base + o_col;
            if (n < N && m < M) {
                out[(size_t)n * M + m] = tile[smem_index<T, TILE, LAYOUT>(o_col, o_row)];
            }
        }
    }
}

template <typename T, int TILE>
void launch(const T* in, T* out, int M, int N, int variant) {
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    TORCH_CHECK(grid.y <= 65535, "too many tile rows for one launch");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    switch (variant) {
        case 0: transpose_naive<T, TILE><<<grid, block, 0, stream>>>(in, out, M, N); break;
        case 1: transpose_tiled<T, TILE, PLAIN><<<grid, block, 0, stream>>>(in, out, M, N); break;
        case 2: transpose_tiled<T, TILE, PADDED><<<grid, block, 0, stream>>>(in, out, M, N); break;
        default: transpose_tiled<T, TILE, SWIZZLED><<<grid, block, 0, stream>>>(in, out, M, N); break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// variant: 0 naive, 1 tiled, 2 padded, 3 swizzled.
torch::Tensor transpose(torch::Tensor x, int64_t variant) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "x must be 2-D");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(variant >= 0 && variant <= 3, "variant must be 0, 1, 2, or 3");
    const c10::cuda::OptionalCUDAGuard device_guard(x.device());
    const int M = (int)x.size(0);
    const int N = (int)x.size(1);
    auto out = torch::empty({N, M}, x.options());
    if (x.numel() == 0) return out;
    if (x.scalar_type() == torch::kFloat32) {
        launch<float, 32>(x.data_ptr<float>(), out.data_ptr<float>(), M, N, (int)variant);
    } else if (x.scalar_type() == torch::kBFloat16) {
        // A transpose only moves bits, so treat bfloat16 as a 16-bit integer.
        launch<unsigned short, 64>(
            reinterpret_cast<const unsigned short*>(x.data_ptr()),
            reinterpret_cast<unsigned short*>(out.data_ptr()), M, N, (int)variant);
    } else {
        TORCH_CHECK(false, "x must be float32 or bfloat16");
    }
    return out;
}
"""
