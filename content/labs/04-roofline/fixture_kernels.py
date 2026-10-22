"""CUDA kernels for lab 04. The reader does not edit this file.

Three kernels, compiled once with `load_cuda`:

- `fma_chain`: reads one float, applies k dependent fused multiply-adds, and
  writes one float. Its arithmetic intensity is 2k FLOP per 8 bytes, so k
  sets where the kernel sits on the roofline.
- `read_sum`: each thread issues `ilp` independent loads per iteration and
  adds them up. The launch shape sets how many bytes are in flight.
- `block_work`: every block does the same fixed chain of FMAs and holds a
  large shared-memory allocation, so only one block fits on an SM. That makes
  the number of waves equal to ceil(blocks / SM count).
"""

import torch
from lab_common import load_cuda

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

__global__ void fma_chain(const float* __restrict__ x, float* __restrict__ y,
                          long n, int k, float a, float b) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  for (; i < n; i += stride) {
    float v = x[i];
#pragma unroll 8
    for (int j = 0; j < k; ++j) v = fmaf(v, a, b);
    y[i] = v;
  }
}

void fma_chain_out(torch::Tensor x, torch::Tensor y, int64_t k, double a, double b) {
  TORCH_CHECK(x.is_cuda() && y.is_cuda(), "tensors must be on the GPU");
  TORCH_CHECK(x.scalar_type() == torch::kFloat && y.scalar_type() == torch::kFloat, "float32 only");
  TORCH_CHECK(x.is_contiguous() && y.is_contiguous() && x.numel() == y.numel(), "contiguous, same size");
  long n = x.numel();
  if (n == 0) return;
  int threads = 256;
  long blocks = (n + threads - 1) / threads;
  if (blocks > 2147483647L) blocks = 2147483647L;
  fma_chain<<<(unsigned)blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), y.data_ptr<float>(), n, (int)k, (float)a, (float)b);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int ILP>
__global__ void read_sum_kernel(const float* __restrict__ x, float* __restrict__ out, long n) {
  long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long nthreads = (long)gridDim.x * blockDim.x;
  float acc = 0.f;
  for (long base = tid; base < n; base += nthreads * ILP) {
    float v[ILP];
#pragma unroll
    for (int j = 0; j < ILP; ++j) {
      long i = base + (long)j * nthreads;
      v[j] = i < n ? x[i] : 0.f;
    }
#pragma unroll
    for (int j = 0; j < ILP; ++j) acc += v[j];
  }
  out[tid] = acc;
}

torch::Tensor read_sum(torch::Tensor x, int64_t blocks, int64_t threads, int64_t ilp) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat && x.is_contiguous(), "contiguous float32 on the GPU");
  TORCH_CHECK(blocks >= 1 && threads >= 1 && threads <= 1024, "bad launch shape");
  auto out = torch::empty({blocks * threads}, x.options());
  long n = x.numel();
  auto stream = at::cuda::getCurrentCUDAStream();
  const float* xp = x.data_ptr<float>();
  float* op = out.data_ptr<float>();
  if (ilp == 1) read_sum_kernel<1><<<(unsigned)blocks, (unsigned)threads, 0, stream>>>(xp, op, n);
  else if (ilp == 2) read_sum_kernel<2><<<(unsigned)blocks, (unsigned)threads, 0, stream>>>(xp, op, n);
  else if (ilp == 4) read_sum_kernel<4><<<(unsigned)blocks, (unsigned)threads, 0, stream>>>(xp, op, n);
  else if (ilp == 8) read_sum_kernel<8><<<(unsigned)blocks, (unsigned)threads, 0, stream>>>(xp, op, n);
  else TORCH_CHECK(false, "ilp must be 1, 2, 4, or 8");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

__global__ void block_work(float* __restrict__ out, int iters, float a, float b) {
  extern __shared__ char smem[];  // never touched: its size is what limits residency
  float v = (float)threadIdx.x * 1e-3f;
  for (int j = 0; j < iters; ++j) v = fmaf(v, a, b);
  out[(long)blockIdx.x * blockDim.x + threadIdx.x] = v;
}

void block_work_out(torch::Tensor out, int64_t num_blocks, int64_t iters, int64_t smem_bytes) {
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat && out.numel() >= num_blocks * 256, "out is too small");
  C10_CUDA_CHECK(cudaFuncSetAttribute(block_work, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
  block_work<<<(unsigned)num_blocks, 256, (size_t)smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
      out.data_ptr<float>(), (int)iters, 0.999f, 0.001f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_CPP = """
void fma_chain_out(torch::Tensor x, torch::Tensor y, int64_t k, double a, double b);
torch::Tensor read_sum(torch::Tensor x, int64_t blocks, int64_t threads, int64_t ilp);
void block_work_out(torch::Tensor out, int64_t num_blocks, int64_t iters, int64_t smem_bytes);
"""

# The constants the kernel uses, so the tests can build a float64 reference.
FMA_A = float(torch.tensor(0.999, dtype=torch.float32).item())
FMA_B = float(torch.tensor(0.001, dtype=torch.float32).item())
# 100 KiB of dynamic shared memory per block. Two blocks would need 2 x 101 KiB,
# more than the 164 KiB an SM has, so exactly one block is resident per SM.
WAVE_SMEM_BYTES = 100 * 1024
BLOCK_THREADS = 256


def _module():
    return load_cuda("lab04_roofline", _CUDA, _CPP, ["fma_chain_out", "read_sum", "block_work_out"])


def fma_chain_into(x: torch.Tensor, y: torch.Tensor, k: int) -> None:
    """y = k fused multiply-adds applied to x. Writes into a preallocated y."""
    _module().fma_chain_out(x, y, int(k), FMA_A, FMA_B)


def fma_chain(x: torch.Tensor, k: int) -> torch.Tensor:
    y = torch.empty_like(x)
    fma_chain_into(x, y, k)
    return y


def fma_chain_reference(x: torch.Tensor, k: int) -> torch.Tensor:
    """The same chain in float64, for checking."""
    v = x.double()
    for _ in range(k):
        v = v * FMA_A + FMA_B
    return v


def read_sum(x: torch.Tensor, blocks: int, threads: int, ilp: int) -> torch.Tensor:
    """Per-thread partial sums of x. Bytes in flight per SM scale with blocks * threads * ilp * 4."""
    return _module().read_sum(x, int(blocks), int(threads), int(ilp))


def block_work_into(out: torch.Tensor, num_blocks: int, iters: int) -> None:
    """Launch `num_blocks` blocks that each do `iters` dependent FMAs, one block per SM."""
    _module().block_work_out(out, int(num_blocks), int(iters), WAVE_SMEM_BYTES)
