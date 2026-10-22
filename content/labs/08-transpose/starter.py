"""Lab 08: shared memory and the transpose.

Part 1 is a model of shared-memory banks in plain Python. It runs without a
GPU, and it tells you in advance which tile layouts conflict.

Part 2 is CUDA C++: four transposes of a row-major (M, N) tensor into a new
(N, M) tensor, for float32 and bfloat16 and for any M and N. The tests check
that every variant is exact on awkward shapes, then time each one at
8192 x 8192 against the copy ceiling.

The Python functions and the CUDA `smem_index` describe the same layouts. Keep
them in step: the tests check the Python, and the Python tells you what the
CUDA does.
"""

LAYOUTS = ("plain", "padded", "swizzled")


def tile_dim(elem_bytes: int) -> int:
    """Elements per tile side: 32 for 4-byte elements, 64 for 2-byte elements.

    Either way a tile row is 128 bytes of float32 or bfloat16 data, which is
    one cache line. (Given; the CUDA below uses the same numbers.)
    """
    return 32 if elem_bytes == 4 else 64


# ---------------------------------------------------------------------------
# Part 1: the bank model
# ---------------------------------------------------------------------------


def bank(byte_address: int) -> int:
    """Return the shared-memory bank that holds `byte_address`.

    There are 32 banks. Successive 4-byte words go to successive banks.
    """
    raise NotImplementedError


def conflict_degree(byte_addresses) -> int:
    """Return how many transactions a warp's access needs.

    `byte_addresses` holds one byte address per lane (32 of them). Lanes that
    read the same 4-byte word share one transaction (a broadcast). Lanes that
    touch different words in the same bank queue up. The degree is the largest
    number of distinct words that fall in any one bank: 1 means conflict free.
    """
    raise NotImplementedError


def smem_index(layout: str, r: int, c: int, elem_bytes: int) -> int:
    """Return the element index of tile[r][c] inside the shared-memory array.

    A tile is tile_dim(elem_bytes) elements on a side. The three layouts:

    - "plain":    row r starts at r * tile_dim.
    - "padded":   each row is one 32-bit word longer than the tile is wide, so
                  the pitch is tile_dim + (4 // elem_bytes) elements.
    - "swizzled": the pitch is tile_dim, but the column is XORed with the row:
                  c ^ ((r & 31) * (4 // elem_bytes)). The factor keeps the two
                  bfloat16 halves of a word together.
    """
    raise NotImplementedError


def tile_bytes(layout: str, elem_bytes: int) -> int:
    """Return the bytes of shared memory one tile occupies in this layout."""
    raise NotImplementedError


def column_read_degree(layout: str, elem_bytes: int, c: int, row0: int = 0) -> int:
    """Return the conflict degree when lane l reads tile[row0 + l][c].

    This is the access pattern of the transpose's second phase. Build the 32
    byte addresses with smem_index and pass them to conflict_degree.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 2: the CUDA kernels
# ---------------------------------------------------------------------------

CPP_DECLARATIONS = """
torch::Tensor transpose(torch::Tensor x, int64_t variant);
"""


def cuda_source() -> str:
    """Return CUDA source that defines `transpose(x, variant)`.

    variant 0: naive. Each thread copies elements straight from the input to
               the transposed position in the output. No shared memory.
    variant 1: tiled, plain layout.
    variant 2: tiled, padded layout.
    variant 3: tiled, swizzled layout.

    The launcher at the bottom is written for you: a 32 x 8 block per
    TILE x TILE tile, with TILE = 32 for float32 and 64 for bfloat16. You write
    `smem_index`, `transpose_naive`, and the two phases of `transpose_tiled`.
    """
    return r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 8;

enum Layout { PLAIN = 0, PADDED = 1, SWIZZLED = 2 };

// Pitch of one tile row, in elements. (Given.)
template <typename T, int TILE, int LAYOUT>
__host__ __device__ constexpr int tile_pitch() {
    return LAYOUT == PADDED ? TILE + 4 / (int)sizeof(T) : TILE;
}

template <typename T, int TILE, int LAYOUT>
__host__ __device__ constexpr int tile_elems() {
    return TILE * tile_pitch<T, TILE, LAYOUT>();
}

// TODO: return the element index of tile[r][c] for this LAYOUT, the same
// formula as smem_index in Python. W = 4 / sizeof(T) is the number of
// elements in one 32-bit word.
template <typename T, int TILE, int LAYOUT>
__device__ __forceinline__ int smem_index(int r, int c) {
    return r * TILE + c;
}

// TODO: the naive transpose. The block's tile starts at input column
// blockIdx.x * TILE and input row blockIdx.y * TILE. Thread (x, y) handles
// rows y, y + 8, ... and columns x, x + 32, ... of the tile. Skip elements
// outside the M x N input. Read in[m * N + n], write out[n * M + m], and use
// size_t for the products so a large tensor can't overflow.
template <typename T, int TILE>
__global__ void transpose_naive(const T* __restrict__ in, T* __restrict__ out, int M, int N) {
}

// TODO: the tiled transpose, in two phases with __syncthreads() between them.
//
// Phase 1: load the input tile into `tile` with the loop shape of the naive
//   kernel. Lane x reads consecutive columns of one row, so the load coalesces.
//   Store each element at smem_index(r, c).
//
// Phase 2: write the output tile. Lane x must write consecutive output
//   columns, which are consecutive input rows, so it reads the tile element
//   smem_index(o_col, o_row), where o_row = threadIdx.y + j and
//   o_col = threadIdx.x + i. The output row is n_base + o_row and the output
//   column is m_base + o_col.
//
// Guard both phases against the M x N bounds, but keep __syncthreads()
// outside every guard.
template <typename T, int TILE, int LAYOUT>
__global__ void transpose_tiled(const T* __restrict__ in, T* __restrict__ out, int M, int N) {
    __shared__ T tile[tile_elems<T, TILE, LAYOUT>()];
}

// ---- Given: the launcher and the PyTorch binding. ----

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
