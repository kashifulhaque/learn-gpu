"""Lab 06: Reductions. Reference solution.

Four reductions over float32 data on the A100:

    sum_tree    two-pass sum; each block reduces in shared memory (sequential addressing)
    sum_atomic  one-pass sum; warp shuffles inside the block, one atomicAdd per block
    max_atomic  one-pass max; warp shuffles, one atomic max per block
    row_sum     sum of each row of a 2-D tensor, one warp per row
"""

from lab_common import load_cuda

CPP_SOURCE = """
torch::Tensor sum_tree(torch::Tensor x);
torch::Tensor sum_atomic(torch::Tensor x);
torch::Tensor max_atomic(torch::Tensor x);
torch::Tensor row_sum(torch::Tensor x);
"""

FUNCTIONS = ["sum_tree", "sum_atomic", "max_atomic", "row_sum"]

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

constexpr int THREADS = 256;            // threads per block; a multiple of 32
constexpr int WARPS = THREADS / 32;     // 8 warps per block
constexpr int MAX_BLOCKS = 108 * 8;     // 108 SMs x 8 resident blocks of 256 threads

// ---------------------------------------------------------------------------
// Loading. Provided: each thread walks the input with a grid-stride loop, so
// any n works and a thread keeps many bytes in flight. Four floats per load
// (a float4) when the pointer is 16-byte aligned, scalars for the rest.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float load_sum(const float* __restrict__ x, int n) {
    float acc = 0.0f;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int n4 = (((uintptr_t)x & 15) == 0) ? (n >> 2) : 0;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    for (int i = tid; i < n4; i += stride) {
        float4 v = x4[i];
        acc += (v.x + v.y) + (v.z + v.w);
    }
    for (int i = n4 * 4 + tid; i < n; i += stride) acc += x[i];
    return acc;
}

__device__ __forceinline__ float load_max(const float* __restrict__ x, int n) {
    float acc = -INFINITY;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int n4 = (((uintptr_t)x & 15) == 0) ? (n >> 2) : 0;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    for (int i = tid; i < n4; i += stride) {
        float4 v = x4[i];
        acc = fmaxf(acc, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
    }
    for (int i = n4 * 4 + tid; i < n; i += stride) acc = fmaxf(acc, x[i]);
    return acc;
}

// ---------------------------------------------------------------------------
// Warp and block reductions.
// ---------------------------------------------------------------------------

// Sum across the 32 lanes of a warp. Every lane must call this. Lane 0 returns
// the total; the other lanes return partial sums you should not use.
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    return v;
}

// Sum across the whole block. Every thread of the block must call this exactly
// once per kernel. Thread 0 returns the total.
__device__ __forceinline__ float block_sum(float v) {
    __shared__ float warp_part[WARPS];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) warp_part[warp] = v;
    __syncthreads();
    // The first warp reduces the per-warp sums. Lanes past WARPS add the identity.
    v = (lane < WARPS) ? warp_part[lane] : 0.0f;
    if (warp == 0) v = warp_sum(v);
    return v;
}

__device__ __forceinline__ float block_max(float v) {
    __shared__ float warp_part[WARPS];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    v = warp_max(v);
    if (lane == 0) warp_part[warp] = v;
    __syncthreads();
    v = (lane < WARPS) ? warp_part[lane] : -INFINITY;
    if (warp == 0) v = warp_max(v);
    return v;
}

// Atomic max for floats, which CUDA does not provide. A non-negative float's
// bits order like an int; a negative float's bits order backwards as an
// unsigned int. So: atomicMax on the int bits for v >= 0, atomicMin on the
// unsigned bits for v < 0. The target must start at -infinity.
__device__ __forceinline__ void atomic_max_float(float* addr, float v) {
    int bits = __float_as_int(v);
    if (bits >= 0) atomicMax(reinterpret_cast<int*>(addr), bits);
    else atomicMin(reinterpret_cast<unsigned int*>(addr), (unsigned int)bits);
}

// ---------------------------------------------------------------------------
// Kernels.
// ---------------------------------------------------------------------------

// Pass 1 and pass 2 of the two-pass sum. Each block writes out[blockIdx.x].
// Sequential addressing: in each round the lower half of the threads add the
// upper half, so the active threads stay together in the low warps.
__global__ void sum_tree_kernel(const float* __restrict__ x, float* __restrict__ out, int n) {
    __shared__ float s[THREADS];
    int t = threadIdx.x;
    s[t] = load_sum(x, n);
    __syncthreads();
    for (int offset = THREADS / 2; offset > 0; offset >>= 1) {
        if (t < offset) s[t] += s[t + offset];
        __syncthreads();                 // outside the if: every thread reaches it
    }
    if (t == 0) out[blockIdx.x] = s[0];
}

__global__ void sum_atomic_kernel(const float* __restrict__ x, float* out, int n) {
    float total = block_sum(load_sum(x, n));
    if (threadIdx.x == 0) atomicAdd(out, total);
}

__global__ void max_atomic_kernel(const float* __restrict__ x, float* out, int n) {
    float best = block_max(load_max(x, n));
    if (threadIdx.x == 0) atomic_max_float(out, best);
}

// One warp per row; a block of 256 threads handles 8 rows. A warp that has no
// row exits as a whole, so the full-mask shuffles never see a partial warp.
__global__ void row_sum_kernel(const float* __restrict__ x, float* __restrict__ out,
                               int rows, int cols) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * WARPS + warp;
    if (row >= rows) return;
    const float* p = x + (long long)row * cols;
    float acc = 0.0f;
    bool vec = ((cols & 3) == 0) && (((uintptr_t)x & 15) == 0);
    int done = 0;
    if (vec) {
        const float4* p4 = reinterpret_cast<const float4*>(p);
        for (int i = lane; i < (cols >> 2); i += 32) {
            float4 v = p4[i];
            acc += (v.x + v.y) + (v.z + v.w);
        }
        done = cols;
    }
    for (int i = done + lane; i < cols; i += 32) acc += p[i];
    acc = warp_sum(acc);
    if (lane == 0) out[row] = acc;
}

// ---------------------------------------------------------------------------
// Host wrappers. Provided.
// ---------------------------------------------------------------------------

static void check_input(const torch::Tensor& x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.numel() > 0 && x.numel() < (1LL << 31), "x must hold 1 to 2^31 - 1 elements");
}

static int blocks_for(int64_t n) {
    int64_t b = (n + THREADS - 1) / THREADS;
    return (int)(b < MAX_BLOCKS ? b : MAX_BLOCKS);
}

torch::Tensor sum_tree(torch::Tensor x) {
    check_input(x);
    int n = (int)x.numel();
    int blocks = blocks_for(n);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto partial = torch::empty({blocks}, x.options());
    auto out = torch::empty({}, x.options());
    sum_tree_kernel<<<blocks, THREADS, 0, stream>>>(x.data_ptr<float>(), partial.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // Pass 2: one block reduces the per-block partial sums with the same kernel.
    sum_tree_kernel<<<1, THREADS, 0, stream>>>(partial.data_ptr<float>(), out.data_ptr<float>(), blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor sum_atomic(torch::Tensor x) {
    check_input(x);
    int n = (int)x.numel();
    auto out = torch::zeros({}, x.options());
    sum_atomic_kernel<<<blocks_for(n), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor max_atomic(torch::Tensor x) {
    check_input(x);
    int n = (int)x.numel();
    auto out = torch::full({}, -INFINITY, x.options());
    max_atomic_kernel<<<blocks_for(n), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor row_sum(torch::Tensor x) {
    check_input(x);
    TORCH_CHECK(x.dim() == 2, "row_sum takes a 2-D tensor");
    int rows = (int)x.size(0);
    int cols = (int)x.size(1);
    auto out = torch::empty({rows}, x.options());
    int blocks = (rows + WARPS - 1) / WARPS;
    row_sum_kernel<<<blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), rows, cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""


def build():
    """Compile the CUDA source (cached) and return the extension module."""
    return load_cuda("lab06_reduce", CUDA_SOURCE, CPP_SOURCE, FUNCTIONS)


def sum_tree(x):
    return build().sum_tree(x)


def sum_atomic(x):
    return build().sum_atomic(x)


def max_atomic(x):
    return build().max_atomic(x)


def row_sum(x):
    return build().row_sum(x)
