"""Lab 03: Bandwidth probe.

You write four copy kernels, count the bytes they move, and let the tests
sweep the copy size to find where bandwidth stops growing. The tests compile
`CUDA_SOURCE` with `CPP_DECLARATIONS`, call the four functions it defines, and
call the four Python functions below.

The first run compiles with nvcc and takes 30 to 60 seconds. A failed compile
prints nvcc's error in the first check.
"""

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

// TODO 1. One element per thread: dst[i] = src[i] for i < n.
// Build i from blockIdx.x, blockDim.x, and threadIdx.x as a long long, and
// guard the tail with `if (i < n)`.
__global__ void copy_scalar_kernel(const float* __restrict__ src,
                                   float* __restrict__ dst, long long n) {
}

// TODO 2. One float4 (16 bytes) per thread. n4 counts float4 elements.
// Copy src[i] to dst[i] for i < n4. A float4 load is one 128-bit instruction.
__global__ void copy_vec4_kernel(const float4* __restrict__ src,
                                 float4* __restrict__ dst, long long n4) {
}

// TODO 3. A grid-stride loop. Start at the thread's global index and advance
// by the total number of threads in the grid, blockDim.x * gridDim.x, until
// i >= n. Keep the stride a long long.
__global__ void copy_grid_stride_kernel(const float* __restrict__ src,
                                        float* __restrict__ dst, long long n) {
}

// TODO 4. A coalesced write with a strided read: dst[i] = src[(i * stride) % n].
// Do the multiply and the modulo in long long.
__global__ void copy_strided_kernel(const float* __restrict__ src,
                                    float* __restrict__ dst, long long n, long long stride) {
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

    // TODO 5. Choose the path.
    //  - If either pointer is not 16-byte aligned (check the address modulo
    //    16), call launch_scalar for all n elements and return. A view such as
    //    t[1:] starts 4 bytes past an aligned address, and a float4 access
    //    there is a misaligned-address error.
    //  - Otherwise launch copy_vec4_kernel over n / 4 float4 elements (skip the
    //    launch when n / 4 is 0), with reinterpret_cast<const float4*>(s) and
    //    reinterpret_cast<float4*>(d).
    //  - Then copy the last n % 4 floats with launch_scalar on the pointers
    //    offset by 4 * (n / 4).
}

void copy_grid_stride(torch::Tensor src, torch::Tensor dst, int64_t blocks) {
    CHECK_F32_CUDA(src);
    CHECK_F32_CUDA(dst);
    TORCH_CHECK(src.numel() == dst.numel(), "src and dst must have the same size");
    TORCH_CHECK(blocks >= 1, "blocks must be at least 1");
    long long n = src.numel();
    if (n == 0) return;
    // TODO 6. Launch copy_grid_stride_kernel with `blocks` blocks of THREADS
    // threads on the current stream, then check the launch with
    // C10_CUDA_KERNEL_LAUNCH_CHECK().
}

void copy_strided(torch::Tensor src, torch::Tensor dst, int64_t stride) {
    CHECK_F32_CUDA(src);
    CHECK_F32_CUDA(dst);
    TORCH_CHECK(src.numel() == dst.numel(), "src and dst must have the same size");
    long long n = src.numel();
    if (n == 0) return;
    // TODO 7. One thread per element, as in launch_scalar, but launch
    // copy_strided_kernel with the stride.
}
"""

CPP_DECLARATIONS = """
void copy_scalar(torch::Tensor src, torch::Tensor dst);
void copy_vec4(torch::Tensor src, torch::Tensor dst);
void copy_grid_stride(torch::Tensor src, torch::Tensor dst, int64_t blocks);
void copy_strided(torch::Tensor src, torch::Tensor dst, int64_t stride);
"""


def bytes_moved(kind: str, n: int) -> int:
    """Return the bytes a kernel logically moves on `n` float32 elements.

    Count every read and every write, in bytes:

    - "copy": read one array, write one array.
    - "add": read two arrays, write one array.

    Raise ValueError for any other `kind`.
    """
    raise NotImplementedError


def sector_bytes_per_element(stride_floats: int) -> int:
    """Return the bytes the memory system moves per element of a strided copy.

    The kernel reads `src[i * stride_floats]` and writes `dst[i]` with a
    coalesced store. Memory moves in 32-byte sectors. Take these as given:

    - A warp reading at a stride of `s` floats fetches 4 * s bytes per element
      while that is under 32, and a whole 32-byte sector per element from a
      stride of 8 floats on.
    - The coalesced write costs 4 bytes per element.

    So `s = 1` returns 8 and `s = 32` returns 36. Ignore cache reuse.
    """
    raise NotImplementedError


def plateau_gbs(sweep, min_mib: float = 2 * A100.L2_MB) -> float:
    """Return the median GB/s of the sweep entries at least `min_mib` MiB big.

    `sweep` is a list of `(size_mib, gbs)` pairs, one per tensor size. The
    default threshold is twice the L2, so a cache-resident size can't pull the
    plateau up. Raise ValueError if no entry is large enough. The median of an
    even count is the mean of the two middle values.
    """
    raise NotImplementedError


def time_copy_gbs(fn, n_floats: int) -> float:
    """Return the GB/s of `fn(src, dst)` copying `n_floats` float32 values.

    Allocate a random source and an empty destination on the GPU, time
    `fn(src, dst)` with `cuda_time`, and convert with `bandwidth_gbs` using
    `bytes_moved("copy", n_floats)`.
    """
    raise NotImplementedError
