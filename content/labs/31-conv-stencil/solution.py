"""Lab 31 solution: convolution, stencils, and implicit GEMM.

Part 1 is the arithmetic and the index maps, in plain Python and torch-CPU.
Part 2 is CUDA: a tiled conv2d with a constant-memory filter, and a 3D
7-point stencil with thread coarsening along z. Part 3 is an optional Triton
implicit-GEMM conv. The CUDA and Triton parts run only on a GPU.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Part 1: shapes, halos, im2col, implicit GEMM, stencil, fusion
# ---------------------------------------------------------------------------


def conv_out_size(size: int, kernel: int, stride: int = 1, padding: int = 0, dilation: int = 1) -> int:
    """Output length along one axis of a convolution."""
    return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def halo_overhead(out_tile: int, k: int, dims: int = 2) -> float:
    """Input elements loaded per output element, for an output tile of
    `out_tile` per side and a filter of `k` per side."""
    return ((out_tile + k - 1) / out_tile) ** dims


def active_fraction(block: int, k: int, dims: int = 2) -> float:
    """Input-tile strategy: the block is `block` threads per side and loads
    one input element each, and only the threads that have a full filter's
    worth of neighbours write an output. Return the fraction that write."""
    return (max(block - k + 1, 0) / block) ** dims


def halo_tile(x: torch.Tensor, row0: int, col0: int, h: int, w: int) -> torch.Tensor:
    """Return the h by w tile of the 2D tensor `x` whose top-left corner is
    (row0, col0). Coordinates outside `x`, including negative ones, read 0."""
    H, W = x.shape
    out = torch.zeros(h, w, dtype=x.dtype)
    r_lo, r_hi = max(row0, 0), min(row0 + h, H)
    c_lo, c_hi = max(col0, 0), min(col0 + w, W)
    if r_lo < r_hi and c_lo < c_hi:
        out[r_lo - row0 : r_hi - row0, c_lo - col0 : c_hi - col0] = x[r_lo:r_hi, c_lo:c_hi]
    return out


def im2col(x: torch.Tensor, kh: int, kw: int, stride: int = 1, padding: int = 0, dilation: int = 1) -> torch.Tensor:
    """Lower (N, C, H, W) to (N, C*kh*kw, P*Q), the layout of F.unfold.

    Row index is c*kh*kw + i*kw + j. Column index is p*Q + q."""
    N, C, H, W = x.shape
    P = conv_out_size(H, kh, stride, padding, dilation)
    Q = conv_out_size(W, kw, stride, padding, dilation)
    xp = F.pad(x, (padding, padding, padding, padding))
    taps = []
    for i in range(kh):
        for j in range(kw):
            r0, c0 = i * dilation, j * dilation
            taps.append(xp[:, :, r0 : r0 + (P - 1) * stride + 1 : stride, c0 : c0 + (Q - 1) * stride + 1 : stride])
    cols = torch.stack(taps, dim=2)  # (N, C, kh*kw, P, Q)
    return cols.reshape(N, C * kh * kw, P * Q)


def im2col_blowup(in_shape, kernel, stride: int = 1, padding: int = 0, dilation: int = 1) -> float:
    """Elements of the im2col matrix divided by elements of the input."""
    N, C, H, W = in_shape
    kh, kw = kernel
    P = conv_out_size(H, kh, stride, padding, dilation)
    Q = conv_out_size(W, kw, stride, padding, dilation)
    return (N * C * kh * kw * P * Q) / (N * C * H * W)


def conv2d_im2col(x, w, bias=None, stride: int = 1, padding: int = 0, dilation: int = 1) -> torch.Tensor:
    """Convolution as one GEMM per image. PyTorch's convention is
    cross-correlation: the filter is not flipped."""
    N = x.shape[0]
    K, C, kh, kw = w.shape
    P = conv_out_size(x.shape[2], kh, stride, padding, dilation)
    Q = conv_out_size(x.shape[3], kw, stride, padding, dilation)
    cols = im2col(x, kh, kw, stride, padding, dilation)
    out = w.reshape(K, C * kh * kw) @ cols  # (N, K, P*Q)
    out = out.reshape(N, K, P, Q)
    if bias is not None:
        out = out + bias.reshape(1, K, 1, 1)
    return out


def implicit_gemm_coords(m, k, in_shape, kernel, stride=1, padding=0, dilation=1, layout="nchw"):
    """Map GEMM coordinates to an input coordinate.

    m indexes (n, p, q) and k indexes the reduction. Return (n, c, h, w), or
    None when (h, w) falls in the zero padding. For layout "nchw" the filter
    is (K, C, R, S) and k = (c*R + r)*S + s. For "nhwc" the filter is
    (K, R, S, C) and k = (r*S + s)*C + c."""
    N, C, H, W = in_shape
    R, S = kernel
    P = conv_out_size(H, R, stride, padding, dilation)
    Q = conv_out_size(W, S, stride, padding, dilation)
    n, rem = divmod(m, P * Q)
    p, q = divmod(rem, Q)
    if layout == "nchw":
        c, rs = divmod(k, R * S)
    else:
        rs, c = divmod(k, C)
    r, s = divmod(rs, S)
    h = p * stride - padding + r * dilation
    wv = q * stride - padding + s * dilation
    if not (0 <= h < H and 0 <= wv < W):
        return None
    return (n, c, h, wv)


def implicit_gemm_tile(x, m0, m1, k0, k1, kernel, stride=1, padding=0, dilation=1):
    """The A tile rows m0..m1 and columns k0..k1 of the im2col matrix, built
    straight from `x` with the index map. No full im2col matrix exists."""
    N, C, H, W = x.shape
    R, S = kernel
    P = conv_out_size(H, R, stride, padding, dilation)
    Q = conv_out_size(W, S, stride, padding, dilation)
    m = torch.arange(m0, m1)[:, None]
    k = torch.arange(k0, k1)[None, :]
    n, rem = m // (P * Q), m % (P * Q)
    p, q = rem // Q, rem % Q
    c, rs = k // (R * S), k % (R * S)
    r, s = rs // S, rs % S
    h = p * stride - padding + r * dilation
    w = q * stride - padding + s * dilation
    valid = (h >= 0) & (h < H) & (w >= 0) & (w < W)
    flat = ((n * C + c) * H + h.clamp(0, H - 1)) * W + w.clamp(0, W - 1)
    vals = x.reshape(-1)[flat.expand(m1 - m0, k1 - k0)]
    return torch.where(valid.expand(m1 - m0, k1 - k0), vals, torch.zeros((), dtype=x.dtype))


def conv2d_implicit(x, w, stride=1, padding=0, dilation=1, block_m=16, block_k=8):
    """Implicit-GEMM convolution: loop over (BM, BK) tiles of the virtual
    im2col matrix, gather each from `x`, and accumulate."""
    N, C, H, W = x.shape
    K, _, R, S = w.shape
    P = conv_out_size(H, R, stride, padding, dilation)
    Q = conv_out_size(W, S, stride, padding, dilation)
    M, Kg = N * P * Q, C * R * S
    wmat = w.reshape(K, Kg)
    out = torch.zeros(M, K, dtype=x.dtype)
    for m0 in range(0, M, block_m):
        m1 = min(m0 + block_m, M)
        acc = torch.zeros(m1 - m0, K, dtype=x.dtype)
        for k0 in range(0, Kg, block_k):
            k1 = min(k0 + block_k, Kg)
            a = implicit_gemm_tile(x, m0, m1, k0, k1, (R, S), stride, padding, dilation)
            acc += a @ wmat[:, k0:k1].T
        out[m0:m1] = acc
    return out.reshape(N, P, Q, K).permute(0, 3, 1, 2).contiguous()


def stencil7_reference(x: torch.Tensor, c0: float, c1: float) -> torch.Tensor:
    """7-point stencil on a (Z, Y, X) grid. An interior cell becomes
    c0*centre + c1*(sum of its six face neighbours). A cell on the boundary
    of the grid is copied unchanged."""
    out = x.clone()
    if min(x.shape) < 3:
        return out
    c = x[1:-1, 1:-1, 1:-1]
    nb = (
        x[:-2, 1:-1, 1:-1] + x[2:, 1:-1, 1:-1]
        + x[1:-1, :-2, 1:-1] + x[1:-1, 2:, 1:-1]
        + x[1:-1, 1:-1, :-2] + x[1:-1, 1:-1, 2:]
    )
    out[1:-1, 1:-1, 1:-1] = c0 * c + c1 * nb
    return out


def conv_relu_maxpool(x, w, bias, stride=1, padding=0, pool=2):
    """max_pool2d(relu(conv2d(x, w) + bias), pool), without building the
    full conv output. For each offset inside the pool window, take the
    matching strided columns of the im2col matrix and run one GEMM, keeping a
    running max. The bias is the same across a window and ReLU is monotonic,
    so both can be applied after the max."""
    N = x.shape[0]
    K, C, kh, kw = w.shape
    P = conv_out_size(x.shape[2], kh, stride, padding)
    Q = conv_out_size(x.shape[3], kw, stride, padding)
    Pp, Qp = P // pool, Q // pool
    cols = im2col(x, kh, kw, stride, padding).reshape(N, C * kh * kw, P, Q)
    wmat = w.reshape(K, C * kh * kw)
    best = None
    for di in range(pool):
        for dj in range(pool):
            sub = cols[:, :, di : di + (Pp - 1) * pool + 1 : pool, dj : dj + (Qp - 1) * pool + 1 : pool]
            y = wmat @ sub.reshape(N, C * kh * kw, Pp * Qp)
            best = y if best is None else torch.maximum(best, y)
    best = best.reshape(N, K, Pp, Qp) + bias.reshape(1, K, 1, 1)
    return torch.relu(best)


# ---------------------------------------------------------------------------
# Part 2: CUDA
# ---------------------------------------------------------------------------

CPP_DECLARATIONS = """
torch::Tensor conv2d_tiled(torch::Tensor x, torch::Tensor w);
torch::Tensor stencil7(torch::Tensor x, double c0, double c1, int64_t variant);
"""


def cuda_source() -> str:
    return r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

// ---------------- conv2d: output tile, constant-memory filter ----------------

constexpr int TILE = 16;                  // output tile per side, and block side
constexpr int MAXK = 15;                  // largest supported filter side
constexpr int IN_DIM = TILE + MAXK - 1;   // shared tile side for the largest filter

__constant__ float c_filter[MAXK * MAXK];

// Single channel, "same" zero padding of K/2, stride 1, cross-correlation:
//   out[y][x] = sum_{a,b} w[a][b] * in[y + a - K/2][x + b - K/2]
__global__ void conv2d_tiled_kernel(const float* __restrict__ in, float* __restrict__ out,
                                    int H, int W, int K) {
    __shared__ float tile[IN_DIM][IN_DIM];
    const int r = K / 2;
    const int in_dim = TILE + K - 1;
    const float* img = in + (size_t)blockIdx.z * H * W;
    float* dst = out + (size_t)blockIdx.z * H * W;
    const int row0 = blockIdx.y * TILE - r;
    const int col0 = blockIdx.x * TILE - r;

    // The halo load: a thread may load several elements, and any element
    // outside the image is 0, which is the zero padding.
    for (int i = threadIdx.y; i < in_dim; i += TILE) {
        for (int j = threadIdx.x; j < in_dim; j += TILE) {
            const int gr = row0 + i, gc = col0 + j;
            tile[i][j] = (gr >= 0 && gr < H && gc >= 0 && gc < W) ? img[(size_t)gr * W + gc] : 0.0f;
        }
    }
    __syncthreads();

    const int orow = blockIdx.y * TILE + threadIdx.y;
    const int ocol = blockIdx.x * TILE + threadIdx.x;
    if (orow < H && ocol < W) {
        float acc = 0.0f;
        for (int a = 0; a < K; ++a)
            for (int b = 0; b < K; ++b)
                acc += c_filter[a * K + b] * tile[threadIdx.y + a][threadIdx.x + b];
        dst[(size_t)orow * W + ocol] = acc;
    }
}

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

// ---------------- 7-point stencil ----------------
// x is (Z, Y, X), X fastest. An interior cell becomes c0*centre + c1*(six
// neighbours); a boundary cell is copied.

// Variant 0: one thread per cell, every neighbour read from global memory.
__global__ void stencil7_naive(const float* __restrict__ in, float* __restrict__ out,
                               int X, int Y, int Z, float c0, float c1) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z;
    if (x >= X || y >= Y) return;
    const size_t S = (size_t)X * Y;
    const size_t i = (size_t)z * S + (size_t)y * X + x;
    if (x == 0 || y == 0 || z == 0 || x == X - 1 || y == Y - 1 || z == Z - 1) {
        out[i] = in[i];
    } else {
        out[i] = c0 * in[i] + c1 * (in[i - 1] + in[i + 1] + in[i - X] + in[i + X] + in[i - S] + in[i + S]);
    }
}

// Variant 1: input-tile in x and y with a shared-memory plane, and thread
// coarsening along z. The z neighbours of a cell live in registers: prev,
// curr, next slide forward one plane per iteration, so each input element is
// read from global memory once per block, not three times.
constexpr int TX = 32;
constexpr int TY = 8;
constexpr int ZC = 16;  // planes per block

__global__ void stencil7_coarse(const float* __restrict__ in, float* __restrict__ out,
                                int X, int Y, int Z, float c0, float c1) {
    __shared__ float plane[TY][TX];
    const int tx = threadIdx.x, ty = threadIdx.y;
    // The tile advances by TX-2 by TY-2 cells: the outer ring only loads.
    const int gx = blockIdx.x * (TX - 2) + tx - 1;
    const int gy = blockIdx.y * (TY - 2) + ty - 1;
    const bool inside = gx >= 0 && gx < X && gy >= 0 && gy < Y;
    const bool owner = inside && tx >= 1 && tx < TX - 1 && ty >= 1 && ty < TY - 1;
    const int z0 = blockIdx.z * ZC;
    const int z1 = min(z0 + ZC, Z);
    const size_t S = (size_t)X * Y;
    const size_t xy = inside ? (size_t)gy * X + gx : 0;

    float prev = 0.0f, curr = 0.0f, next = 0.0f;
    if (inside) {
        if (z0 > 0) prev = in[(size_t)(z0 - 1) * S + xy];
        curr = in[(size_t)z0 * S + xy];
    }
    for (int z = z0; z < z1; ++z) {
        next = (inside && z + 1 < Z) ? in[(size_t)(z + 1) * S + xy] : 0.0f;
        plane[ty][tx] = curr;
        __syncthreads();
        if (owner) {
            const bool edge = gx == 0 || gx == X - 1 || gy == 0 || gy == Y - 1 || z == 0 || z == Z - 1;
            out[(size_t)z * S + xy] = edge
                ? curr
                : c0 * curr + c1 * (prev + next + plane[ty][tx - 1] + plane[ty][tx + 1] +
                                    plane[ty - 1][tx] + plane[ty + 1][tx]);
        }
        __syncthreads();
        prev = curr;
        curr = next;
    }
}

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
    """NCHW fp32 conv as an implicit GEMM in Triton. x is (N, C, H, W), w is
    (K, C, R, S). Dilation is 1."""
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(x_ptr, w_ptr, y_ptr, C, H, W, K, R, S, P, Q, stride, pad, M, Kg,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        offs_m = tl.program_id(0) * BM + tl.arange(0, BM)
        offs_n = tl.program_id(1) * BN + tl.arange(0, BN)
        n = offs_m // (P * Q)
        pq = offs_m % (P * Q)
        p = pq // Q
        q = pq % Q
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, Kg, BK):
            offs_k = k0 + tl.arange(0, BK)
            c = offs_k // (R * S)
            rs = offs_k % (R * S)
            r = rs // S
            s = rs % S
            h = p[:, None] * stride - pad + r[None, :]
            wd = q[:, None] * stride - pad + s[None, :]
            a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < Kg) & (h >= 0) & (h < H) & (wd >= 0) & (wd < W)
            a = tl.load(x_ptr + ((n[:, None] * C + c[None, :]) * H + h) * W + wd, mask=a_mask, other=0.0)
            b_mask = (offs_k[:, None] < Kg) & (offs_n[None, :] < K)
            b = tl.load(w_ptr + offs_n[None, :] * Kg + offs_k[:, None], mask=b_mask, other=0.0)
            acc += tl.dot(a, b)
        y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < K)
        y_ptrs = y_ptr + ((n[:, None] * K + offs_n[None, :]) * P + p[:, None]) * Q + q[:, None]
        tl.store(y_ptrs, acc, mask=y_mask)

    N, C, H, W = x.shape
    K, _, R, S = w.shape
    P = conv_out_size(H, R, stride, padding)
    Q = conv_out_size(W, S, stride, padding)
    M, Kg = N * P * Q, C * R * S
    x = x.contiguous()
    w = w.contiguous()
    y = torch.empty(N, K, P, Q, device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(M, 32), triton.cdiv(K, 32))
    kernel[grid](x, w, y, C, H, W, K, R, S, P, Q, stride, padding, M, Kg, BM=32, BN=32, BK=32)
    return y
