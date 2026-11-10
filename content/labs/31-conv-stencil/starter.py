"""Lab 31: convolution, stencils, and implicit GEMM.

Part 1 runs on any machine, in plain Python and torch-CPU. Do the functions in
order: the first check needs only `conv_out_size`.

1. conv_out_size
2. halo_overhead, active_fraction
3. halo_tile
4. im2col, im2col_blowup
5. conv2d_im2col
6. implicit_gemm_coords, implicit_gemm_tile, conv2d_implicit
7. stencil7_reference
8. conv_relu_maxpool

Part 2 is CUDA: a tiled conv2d with a constant-memory filter, and a 3D
7-point stencil with thread coarsening along z. Part 3 is an optional Triton
implicit-GEMM conv. Parts 2 and 3 run only on a GPU, and skip without one.

PyTorch convolution is cross-correlation: it does not flip the filter.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Part 1: shapes, halos, im2col, implicit GEMM, stencil, fusion
# ---------------------------------------------------------------------------


def conv_out_size(size: int, kernel: int, stride: int = 1, padding: int = 0, dilation: int = 1) -> int:
    """Output length along one axis of a convolution.

    PyTorch floors the division. Dilation spreads the filter taps apart, so the
    filter spans dilation * (kernel - 1) + 1 input elements.
    """
    raise NotImplementedError


def halo_overhead(out_tile: int, k: int, dims: int = 2) -> float:
    """Input elements loaded per output element, for an output tile of
    `out_tile` per side and a filter of `k` per side (output-tile strategy).

    The block loads (out_tile + k - 1) ** dims elements and writes
    out_tile ** dims outputs.
    """
    raise NotImplementedError


def active_fraction(block: int, k: int, dims: int = 2) -> float:
    """Input-tile strategy: the block is `block` threads per side and loads one
    input element each. Only the threads that have a full filter's worth of
    neighbours write an output. Return the fraction that write, which is 0.0
    when the filter is wider than the block.
    """
    raise NotImplementedError


def halo_tile(x: torch.Tensor, row0: int, col0: int, h: int, w: int) -> torch.Tensor:
    """Return the h by w tile of the 2D tensor `x` whose top-left corner is
    (row0, col0). Any coordinate outside `x`, including a negative one, reads 0.

    This is the halo load of a tiled convolution at the image border: the zero
    padding. Beware: x[-1] does not fail in Python or torch. It wraps to the
    last row.
    """
    raise NotImplementedError


def im2col(x: torch.Tensor, kh: int, kw: int, stride: int = 1, padding: int = 0, dilation: int = 1) -> torch.Tensor:
    """Lower (N, C, H, W) to (N, C*kh*kw, P*Q), the layout of F.unfold.

    Row index is c*kh*kw + i*kw + j, where (i, j) is the filter tap. Column
    index is p*Q + q. Build it from slices of the padded input; do not call
    F.unfold.
    """
    raise NotImplementedError


def im2col_blowup(in_shape, kernel, stride: int = 1, padding: int = 0, dilation: int = 1) -> float:
    """Elements of the im2col matrix divided by elements of the input.

    `in_shape` is (N, C, H, W) and `kernel` is (kh, kw).
    """
    raise NotImplementedError


def conv2d_im2col(x, w, bias=None, stride: int = 1, padding: int = 0, dilation: int = 1) -> torch.Tensor:
    """Convolution as one GEMM per image: w.reshape(K, C*kh*kw) @ im2col(x).

    x is (N, C, H, W), w is (K, C, kh, kw). PyTorch's convention is
    cross-correlation: do not flip the filter. Return (N, K, P, Q).
    """
    raise NotImplementedError


def implicit_gemm_coords(m, k, in_shape, kernel, stride=1, padding=0, dilation=1, layout="nchw"):
    """Map GEMM coordinates to an input coordinate.

    m indexes (n, p, q): m = (n * P + p) * Q + q. k indexes the reduction.
    Return (n, c, h, w), or None when (h, w) falls in the zero padding.

    For layout "nchw" the filter is (K, C, R, S) and k = (c * R + r) * S + s.
    For "nhwc" the filter is (K, R, S, C) and k = (r * S + s) * C + c.
    The input row is h = p * stride - padding + r * dilation, and the column
    follows the same way.
    """
    raise NotImplementedError


def implicit_gemm_tile(x, m0, m1, k0, k1, kernel, stride=1, padding=0, dilation=1):
    """Rows m0..m1 and columns k0..k1 of the im2col matrix (layout "nchw"),
    shape (m1 - m0, k1 - k0), built straight from `x` with the index map.

    No full im2col matrix may exist. A padded entry is 0. Index with clamped
    coordinates and mask afterwards: a negative index would wrap.
    """
    raise NotImplementedError


def conv2d_implicit(x, w, stride=1, padding=0, dilation=1, block_m=16, block_k=8):
    """Implicit-GEMM convolution. Loop over (block_m, block_k) tiles of the
    virtual im2col matrix, gather each with implicit_gemm_tile, and accumulate
    a @ w_matrix[:, k0:k1].T. Return (N, K, P, Q).
    """
    raise NotImplementedError


def stencil7_reference(x: torch.Tensor, c0: float, c1: float) -> torch.Tensor:
    """7-point stencil on a (Z, Y, X) grid. An interior cell becomes
    c0 * centre + c1 * (sum of its six face neighbours). A cell on the boundary
    of the grid is copied unchanged, so a grid with a side under 3 is returned
    as it is. Do not use torch.roll: it wraps the boundary.
    """
    raise NotImplementedError


def conv_relu_maxpool(x, w, bias, stride=1, padding=0, pool=2):
    """max_pool2d(relu(conv2d(x, w) + bias), pool) without building the full
    conv output.

    For each offset (di, dj) inside the pool window, take the matching strided
    columns of the im2col matrix and run one GEMM, keeping a running max.
    Bias is the same across a window and ReLU is monotonic, so apply both once
    after the max. A partial window at an odd edge is dropped, as max_pool2d
    does.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 2: CUDA
# ---------------------------------------------------------------------------

CPP_DECLARATIONS = """
torch::Tensor conv2d_tiled(torch::Tensor x, torch::Tensor w);
torch::Tensor stencil7(torch::Tensor x, double c0, double c1, int64_t variant);
"""


def cuda_source() -> str:
    """CUDA source that defines `conv2d_tiled` and `stencil7`.

    conv2d_tiled(x, w): x is (N, H, W) float32, w is a (K, K) odd filter with
    K at most 15. Single channel, "same" zero padding of K/2, stride 1,
    cross-correlation. The filter lives in __constant__ memory. Each 16 by 16
    block loads a (16 + K - 1) squared input tile into shared memory, with
    zeros outside the image, then writes a 16 by 16 output tile (the
    output-tile strategy).

    stencil7(x, c0, c1, variant): x is (Z, Y, X) float32. Variant 0 is one
    thread per cell. Variant 1 coarsens along z: a 32 by 8 block holds an x-y
    plane in shared memory and walks 16 planes, keeping the z neighbours in
    registers (prev, curr, next). Only the inner 30 by 6 threads write.

    The launchers are given. You write the two conv pieces and the two
    stencil kernels.
    """
    return r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

constexpr int TILE = 16;
constexpr int MAXK = 15;
constexpr int IN_DIM = TILE + MAXK - 1;

__constant__ float c_filter[MAXK * MAXK];

// TODO: out[y][x] = sum over a, b of c_filter[a * K + b] * in[y + a - K/2][x + b - K/2],
// with in = 0 outside the image.
//
// 1. Load the (TILE + K - 1) squared input tile into `tile`. The tile starts
//    at row blockIdx.y * TILE - K/2 and column blockIdx.x * TILE - K/2. A
//    thread loads several elements: stride by TILE in both directions. Write 0
//    for a coordinate outside [0, H) x [0, W). This is the halo and the zero
//    padding at once.
// 2. __syncthreads(), outside any if.
// 3. If the output cell is inside the image, accumulate the K * K products
//    from `tile[threadIdx.y + a][threadIdx.x + b]` and the constant filter.
//
// blockIdx.z selects the image. Use size_t for image offsets.
__global__ void conv2d_tiled_kernel(const float* __restrict__ in, float* __restrict__ out,
                                    int H, int W, int K) {
    __shared__ float tile[IN_DIM][IN_DIM];
}

// Given: copies the filter to constant memory and launches.
torch::Tensor conv2d_tiled(torch::Tensor x, torch::Tensor w) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "x and w must be CUDA tensors");
    TORCH_CHECK(x.dim() == 3, "x must be (N, H, W)");
    TORCH_CHECK(w.dim() == 2 && w.size(0) == w.size(1) && w.size(0) % 2 == 1 && w.size(0) <= MAXK,
                "w must be a square odd filter of side at most 15");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 && w.scalar_type() == torch::kFloat32, "float32 only");
    const c10::cuda::OptionalCUDAGuard device_guard(x.device());
    x = x.contiguous();
    w = w.contiguous();
    const int N = (int)x.size(0), H = (int)x.size(1), W = (int)x.size(2), K = (int)w.size(0);
    auto out = torch::empty_like(x);
    if (x.numel() == 0) return out;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemcpyToSymbolAsync(c_filter, w.data_ptr<float>(), (size_t)K * K * sizeof(float), 0,
                                           cudaMemcpyDeviceToDevice, stream));
    dim3 block(TILE, TILE);
    dim3 grid((W + TILE - 1) / TILE, (H + TILE - 1) / TILE, N);
    TORCH_CHECK(grid.y <= 65535 && grid.z <= 65535, "grid too large for one launch");
    conv2d_tiled_kernel<<<grid, block, 0, stream>>>(x.data_ptr<float>(), out.data_ptr<float>(), H, W, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// TODO: variant 0. One thread per cell: x = blockIdx.x * blockDim.x + threadIdx.x,
// y likewise, z = blockIdx.z. A boundary cell (x, y, or z on a face of the grid)
// is copied. An interior cell is
//   c0 * in[i] + c1 * (in[i-1] + in[i+1] + in[i-X] + in[i+X] + in[i-S] + in[i+S])
// with S = X * Y. Use size_t for i.
__global__ void stencil7_naive(const float* __restrict__ in, float* __restrict__ out,
                               int X, int Y, int Z, float c0, float c1) {
}

constexpr int TX = 32;
constexpr int TY = 8;
constexpr int ZC = 16;

// TODO: variant 1. Block (TX, TY). The block advances by TX-2 by TY-2 cells, so
//   gx = blockIdx.x * (TX - 2) + threadIdx.x - 1, gy likewise. A thread whose
//   (gx, gy) is inside the grid loads. Only threads with 1 <= tx < TX-1 and
//   1 <= ty < TY-1 write ("owners").
// The block handles planes z0 = blockIdx.z * ZC up to min(z0 + ZC, Z).
//   Keep prev, curr, next in registers. Per plane: load next from global,
//   store curr into the shared plane, __syncthreads(), compute from the plane
//   (x and y neighbours) and the registers (z neighbours), __syncthreads(),
//   then prev = curr, curr = next. Keep both barriers outside every if.
//   A boundary cell is copied from curr.
__global__ void stencil7_coarse(const float* __restrict__ in, float* __restrict__ out,
                                int X, int Y, int Z, float c0, float c1) {
    __shared__ float plane[TY][TX];
}

// Given: the launcher.
torch::Tensor stencil7(torch::Tensor x, double c0, double c1, int64_t variant) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 3 && x.is_contiguous(), "x must be a contiguous CUDA (Z, Y, X) tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "float32 only");
    TORCH_CHECK(variant == 0 || variant == 1, "variant must be 0 or 1");
    const c10::cuda::OptionalCUDAGuard device_guard(x.device());
    const int Z = (int)x.size(0), Y = (int)x.size(1), X = (int)x.size(2);
    auto out = torch::empty_like(x);
    if (x.numel() == 0) return out;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (variant == 0) {
        dim3 block(32, 8, 1);
        dim3 grid((X + 31) / 32, (Y + 7) / 8, Z);
        TORCH_CHECK(grid.y <= 65535 && grid.z <= 65535, "grid too large for one launch");
        stencil7_naive<<<grid, block, 0, stream>>>(x.data_ptr<float>(), out.data_ptr<float>(), X, Y, Z,
                                                    (float)c0, (float)c1);
    } else {
        dim3 block(TX, TY, 1);
        dim3 grid((X + TX - 3) / (TX - 2), (Y + TY - 3) / (TY - 2), (Z + ZC - 1) / ZC);
        TORCH_CHECK(grid.y <= 65535 && grid.z <= 65535, "grid too large for one launch");
        stencil7_coarse<<<grid, block, 0, stream>>>(x.data_ptr<float>(), out.data_ptr<float>(), X, Y, Z,
                                                     (float)c0, (float)c1);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""


# ---------------------------------------------------------------------------
# Part 3: optional Triton implicit-GEMM conv
# ---------------------------------------------------------------------------


def conv2d_triton(x: torch.Tensor, w: torch.Tensor, stride: int = 1, padding: int = 0) -> torch.Tensor:
    """Optional. NCHW fp32 conv as an implicit GEMM in Triton: x is (N, C, H, W),
    w is (K, C, R, S), dilation is 1. Decode m into (n, p, q) and the reduction
    index into (c, r, s) inside the kernel, mask the padded loads, and
    accumulate with tl.dot. tl.dot uses TF32 by default on Ampere, so expect an
    error near 1e-2. The lab records the error and never fails on it.
    """
    raise NotImplementedError
