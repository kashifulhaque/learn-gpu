"""Lab 06: Reductions. Starter.

You fill in the TODOs inside the CUDA source below. Everything else (the
loading loops, the host wrappers, the Python glue) is provided. Four reductions
over float32 data on the A100:

    sum_tree    two-pass sum; each block reduces in shared memory (sequential addressing)
    sum_atomic  one-pass sum; warp shuffles inside the block, one atomicAdd per block
    max_atomic  one-pass max; warp shuffles, one atomic max per block
    row_sum     sum of each row of a 2-D tensor, one warp per row

The tests check each against PyTorch on awkward sizes (1, 31, 33, 257,
1,000,003, and an input that starts 4 bytes past an aligned address), then
report bandwidth: bytes read divided by time, against the 1275 GB/s copy
ceiling. The first compile takes 30 to 60 seconds. The stubs below compile but
return wrong answers, so run the tests to see them fail first.
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

// TODO 1: sum across the 32 lanes of a warp. Every lane must call this. Lane 0
// must return the total; the other lanes may return anything.
// Use __shfl_down_sync(0xffffffffu, v, offset) with offset 16, 8, 4, 2, 1.
__device__ __forceinline__ float warp_sum(float v) {
    return v;   // TODO: replace with five shuffle-and-add rounds
}

// TODO 2: the same for max, with fmaxf in place of +.
__device__ __forceinline__ float warp_max(float v) {
    return v;   // TODO
}

// TODO 3: sum across the whole block. Every thread of the block calls this
// exactly once per kernel; thread 0 must return the total.
// Plan: reduce inside each warp, have lane 0 of each warp write its result to
// warp_part[warp], __syncthreads(), then let warp 0 reduce the WARPS values.
// Lanes of warp 0 beyond WARPS must contribute the identity element (0 for a sum).
__device__ __forceinline__ float block_sum(float v) {
    __shared__ float warp_part[WARPS];
    return v;   // TODO
}

// TODO 4: the same for max. The identity element of max is -INFINITY, not 0:
// an all-negative array must still give a negative answer.
__device__ __forceinline__ float block_max(float v) {
    __shared__ float warp_part[WARPS];
    return v;   // TODO
}

// TODO 5: atomic max for floats, which CUDA does not provide. Hint: a
// non-negative float's bits (__float_as_int) order like an int. A negative
// float's bits order backwards when read as an unsigned int. So atomicMax on the
// int bits works for v >= 0, and atomicMin on the unsigned bits for v < 0. The
// target starts at -infinity (the host wrapper sets that).
__device__ __forceinline__ void atomic_max_float(float* addr, float v) {
    // TODO
}

// ---------------------------------------------------------------------------
// Kernels.
// ---------------------------------------------------------------------------

// TODO 6: one block's share of the two-pass sum. The host launches this kernel
// twice: once over x with many blocks, once over the per-block results with one
// block. Each block must write its total to out[blockIdx.x].
// Use shared memory and sequential addressing: in each round, thread t < offset
// adds s[t + offset] into s[t], for offset = THREADS/2, THREADS/4, ..., 1.
// Put __syncthreads() after the load and after every round, outside any if.
__global__ void sum_tree_kernel(const float* __restrict__ x, float* __restrict__ out, int n) {
    __shared__ float s[THREADS];
    // TODO
}

// TODO 7: one-pass sum. Reduce this thread's loaded sum across the block with
// block_sum, then thread 0 adds the block's total into *out with atomicAdd.
// The host zeroed *out. One atomic per block, not per thread.
__global__ void sum_atomic_kernel(const float* __restrict__ x, float* out, int n) {
    // TODO
}

// TODO 8: one-pass max, the same shape with block_max and atomic_max_float.
__global__ void max_atomic_kernel(const float* __restrict__ x, float* out, int n) {
    // TODO
}

// TODO 9: one warp per row. A block of 256 threads holds 8 warps, so it handles
// 8 consecutive rows: warp w of block b owns row b * 8 + w. Each lane sums
// elements lane, lane + 32, lane + 64, ... of its row, then warp_sum combines the
// lanes and lane 0 writes out[row]. A warp whose row is >= rows must exit as a
// whole (return before any shuffle), so a full-mask shuffle never sees half a warp.
// Optional speedup: when cols % 4 == 0 and x is 16-byte aligned, read float4s.
__global__ void row_sum_kernel(const float* __restrict__ x, float* __restrict__ out,
                               int rows, int cols) {
    // TODO
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
