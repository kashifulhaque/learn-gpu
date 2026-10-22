"""Lab 03: Bandwidth probe. Reference solution."""

import torch
from lab_common import A100, bandwidth_gbs, cuda_time

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

#define CHECK_F32_CUDA(x) \
    TORCH_CHECK((x).is_cuda() && (x).scalar_type() == at::kFloat && (x).is_contiguous(), \
                #x " must be a contiguous float32 CUDA tensor")

constexpr int THREADS = 256;

// One element per thread. The grid rounds up, so the guard covers the tail.
__global__ void copy_scalar_kernel(const float* __restrict__ src,
                                   float* __restrict__ dst, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// One float4 (16 bytes) per thread. n4 counts float4 elements, not floats.
__global__ void copy_vec4_kernel(const float4* __restrict__ src,
                                 float4* __restrict__ dst, long long n4) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) dst[i] = src[i];
}

// A fixed grid walks the whole array. The stride is the size of the grid, so
// the 32 lanes of a warp stay on 32 consecutive floats in every iteration.
__global__ void copy_grid_stride_kernel(const float* __restrict__ src,
                                        float* __restrict__ dst, long long n) {
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        dst[i] = src[i];
    }
}

// Coalesced write, strided read: dst[i] = src[(i * stride) % n].
__global__ void copy_strided_kernel(const float* __restrict__ src,
                                    float* __restrict__ dst, long long n, long long stride) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[(i * stride) % n];
}

static void launch_scalar(const float* src, float* dst, long long n, cudaStream_t stream) {
    if (n <= 0) return;
    long long blocks = (n + THREADS - 1) / THREADS;
    copy_scalar_kernel<<<(unsigned int)blocks, THREADS, 0, stream>>>(src, dst, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void copy_scalar(torch::Tensor src, torch::Tensor dst) {
    CHECK_F32_CUDA(src);
    CHECK_F32_CUDA(dst);
    TORCH_CHECK(src.numel() == dst.numel(), "src and dst must have the same size");
    launch_scalar(src.data_ptr<float>(), dst.data_ptr<float>(), src.numel(),
                  at::cuda::getCurrentCUDAStream());
}

void copy_vec4(torch::Tensor src, torch::Tensor dst) {
    CHECK_F32_CUDA(src);
    CHECK_F32_CUDA(dst);
    TORCH_CHECK(src.numel() == dst.numel(), "src and dst must have the same size");
    long long n = src.numel();
    const float* s = src.data_ptr<float>();
    float* d = dst.data_ptr<float>();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // float4 needs 16-byte alignment. A view such as t[1:] starts 4 bytes past
    // an aligned address, so it falls back to the scalar kernel.
    bool aligned = (reinterpret_cast<uintptr_t>(s) % 16 == 0) &&
                   (reinterpret_cast<uintptr_t>(d) % 16 == 0);
    if (!aligned) {
        launch_scalar(s, d, n, stream);
        return;
    }
    long long n4 = n / 4;
    if (n4 > 0) {
        long long blocks = (n4 + THREADS - 1) / THREADS;
        copy_vec4_kernel<<<(unsigned int)blocks, THREADS, 0, stream>>>(
            reinterpret_cast<const float4*>(s), reinterpret_cast<float4*>(d), n4);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    // The last n % 4 floats, copied one per thread.
    launch_scalar(s + 4 * n4, d + 4 * n4, n - 4 * n4, stream);
}

void copy_grid_stride(torch::Tensor src, torch::Tensor dst, int64_t blocks) {
    CHECK_F32_CUDA(src);
    CHECK_F32_CUDA(dst);
    TORCH_CHECK(src.numel() == dst.numel(), "src and dst must have the same size");
    TORCH_CHECK(blocks >= 1, "blocks must be at least 1");
    long long n = src.numel();
    if (n == 0) return;
    copy_grid_stride_kernel<<<(unsigned int)blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        src.data_ptr<float>(), dst.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void copy_strided(torch::Tensor src, torch::Tensor dst, int64_t stride) {
    CHECK_F32_CUDA(src);
    CHECK_F32_CUDA(dst);
    TORCH_CHECK(src.numel() == dst.numel(), "src and dst must have the same size");
    long long n = src.numel();
    if (n == 0) return;
    long long blocks = (n + THREADS - 1) / THREADS;
    copy_strided_kernel<<<(unsigned int)blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        src.data_ptr<float>(), dst.data_ptr<float>(), n, stride);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

CPP_DECLARATIONS = """
void copy_scalar(torch::Tensor src, torch::Tensor dst);
void copy_vec4(torch::Tensor src, torch::Tensor dst);
void copy_grid_stride(torch::Tensor src, torch::Tensor dst, int64_t blocks);
void copy_strided(torch::Tensor src, torch::Tensor dst, int64_t stride);
"""


def bytes_moved(kind: str, n: int) -> int:
    """Bytes a kernel logically moves, counting every read and every write."""
    if kind == "copy":
        return 2 * n * 4
    if kind == "add":
        return 3 * n * 4
    raise ValueError(f"unknown kind {kind!r}; expected 'copy' or 'add'")


def sector_bytes_per_element(stride_floats: int) -> int:
    """Bytes the memory system moves per element of a strided copy."""
    read = min(4 * stride_floats, 32)
    write = 4
    return read + write


def plateau_gbs(sweep, min_mib: float = 2 * A100.L2_MB) -> float:
    """Median GB/s over the sweep entries whose size is at least `min_mib`."""
    values = sorted(gbs for mib, gbs in sweep if mib >= min_mib)
    if not values:
        raise ValueError(f"no sweep entry is at least {min_mib} MiB")
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2


def time_copy_gbs(fn, n_floats: int) -> float:
    """GB/s of `fn(src, dst)` copying `n_floats` float32 values."""
    src = torch.randn(n_floats, device="cuda")
    dst = torch.empty_like(src)
    ms = cuda_time(lambda: fn(src, dst), warmup=10, iters=50)
    return bandwidth_gbs(bytes_moved("copy", n_floats), ms)
