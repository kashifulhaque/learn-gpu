"""Lab 07: Scan and histogram.

Three groups of kernels, all in one CUDA source:

1. Block scans, one element per thread, one 256-element tile per block:
   Kogge-Stone and Blelloch, both in shared memory.
2. A device-wide exclusive scan by reduce-then-scan: a tile-reduce kernel, a
   (recursive) scan of the tile sums, and a scan kernel that adds each tile's
   offset. Each tile is 1024 elements: 256 threads, 4 items per thread, a
   serial scan in registers, and a warp-shuffle scan across threads.
3. A 256-bin histogram of bytes: global atomics (given), a shared-memory
   privatized version, and, as a stretch goal, a warp-aggregated version.

Fill in every `TODO` inside CUDA_SRC. The host code, the launches, and the
Python wrappers are written for you. The tests call the six Python functions
at the bottom of this file. The first call compiles CUDA_SRC with nvcc, which
takes 30 to 90 seconds.
"""

import torch
from lab_common import load_cuda

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <cstdint>
#include <vector>

constexpr int BLOCK = 256;               // threads per block, everywhere
constexpr int ITEMS = 4;                 // elements per thread in the device scan
constexpr int TILE = BLOCK * ITEMS;      // 1024 elements per block in the device scan
constexpr unsigned FULL = 0xffffffffu;   // every lane of a warp

// ---------------------------------------------------------------------------
// 1. Block scans: one element per thread, one 256-element tile per block.
//    Both write the exclusive scan of the tile to `out` and the tile's total
//    to `sums[blockIdx.x]`. Elements past `n` count as 0.
// ---------------------------------------------------------------------------

__global__ void block_scan_ks_kernel(const int* in, int* out, int* sums, int64_t n) {
  __shared__ int buf[BLOCK];
  const int tid = threadIdx.x;
  const int64_t i = (int64_t)blockIdx.x * BLOCK + tid;
  buf[tid] = (i < n) ? in[i] : 0;
  __syncthreads();
  // TODO: Kogge-Stone. For d = 1, 2, 4, ..., BLOCK/2, every thread with tid >= d adds buf[tid - d]
  // to buf[tid]. A thread must read its neighbor's old value before anyone overwrites it, so
  // each step needs a barrier between the read and the write, and another before the next read.
  // TODO: write the exclusive scan to out[i] (the inclusive scan shifted right by one, 0 for
  // tid == 0), and have the last thread write the tile's total to sums[blockIdx.x].
}

__global__ void block_scan_blelloch_kernel(const int* in, int* out, int* sums, int64_t n) {
  __shared__ int t[BLOCK];
  const int tid = threadIdx.x;
  const int64_t i = (int64_t)blockIdx.x * BLOCK + tid;
  t[tid] = (i < n) ? in[i] : 0;
  int offset = 1;
  // TODO: up-sweep. For d = BLOCK/2, BLOCK/4, ..., 1: barrier; threads with tid < d add
  // t[a] into t[b], where a = offset*(2*tid+1)-1 and b = offset*(2*tid+2)-1; then offset *= 2.
  // Keep the barrier outside the `if (tid < d)`.
  // TODO: thread 0 saves the root t[BLOCK-1] to sums[blockIdx.x], then sets it to 0.
  // TODO: down-sweep. For d = 1, 2, ..., BLOCK/2: offset /= 2; barrier; threads with tid < d
  // do tmp = t[a]; t[a] = t[b]; t[b] += tmp.
  // TODO: after a final barrier, write t[tid] to out[i] when i < n.
}

// ---------------------------------------------------------------------------
// 2. The device-wide scan, for int and float.
// ---------------------------------------------------------------------------

// Inclusive scan across the 32 lanes of a warp, with shuffles. Every lane of
// the warp must call it.
template <typename T>
__device__ __forceinline__ T warp_inclusive_scan(T v) {
  // TODO: for d = 1, 2, 4, 8, 16: up = __shfl_up_sync(FULL, v, d); if (lane >= d) v += up.
  // The shuffle must run on every lane, so keep it outside the `if`.
  return v;
}

// Exclusive scan of one value per thread across the 256-thread block. Returns
// this thread's exclusive prefix and writes the block total to *total.
// Call it once per kernel: it reuses one static shared array without a final barrier.
template <typename T>
__device__ T block_exclusive_scan(T v, T* total) {
  __shared__ T warp_sums[BLOCK / 32];
  // TODO: 1. inclusive scan within the warp; the lane's exclusive value is the previous lane's
  //          inclusive value (shuffle up by 1, and 0 in lane 0).
  //       2. lane 31 of each warp writes its inclusive value (the warp total) to warp_sums[warp].
  //       3. barrier; warp 0 scans the 8 warp totals (use lanes 0 to 7, and 0 elsewhere); barrier.
  //       4. the result is (the scanned total of the warps before this one) + (the in-warp
  //          exclusive value); *total is the grand total.
  *total = T(0);
  return T(0);
}

// A 16-byte vector type for each element type, so a full tile moves with 128-bit accesses.
template <typename T> struct V4;
template <> struct V4<int> {
  using type = int4;
  __device__ static type make(int a, int b, int c, int d) { return make_int4(a, b, c, d); }
};
template <> struct V4<float> {
  using type = float4;
  __device__ static type make(float a, float b, float c, float d) { return make_float4(a, b, c, d); }
};

// Thread t owns items 4t .. 4t+3 of the tile. A full tile moves as one 128-bit access per thread; the last tile
// falls back to scalar accesses and pads with zeros.
template <typename T>
__device__ __forceinline__ void load_items(const T* in, int64_t base, int64_t n, T (&items)[ITEMS]) {
  if (base + TILE <= n) {
    typename V4<T>::type v = reinterpret_cast<const typename V4<T>::type*>(in + base)[threadIdx.x];
    items[0] = v.x; items[1] = v.y; items[2] = v.z; items[3] = v.w;
  } else {
#pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
      int64_t i = base + (int64_t)threadIdx.x * ITEMS + k;
      items[k] = (i < n) ? in[i] : T(0);
    }
  }
}

template <typename T>
__device__ __forceinline__ void store_items(T* out, int64_t base, int64_t n, const T (&items)[ITEMS]) {
  if (base + TILE <= n) {
    reinterpret_cast<typename V4<T>::type*>(out + base)[threadIdx.x] =
        V4<T>::make(items[0], items[1], items[2], items[3]);
  } else {
#pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
      int64_t i = base + (int64_t)threadIdx.x * ITEMS + k;
      if (i < n) out[i] = items[k];
    }
  }
}

// Kernel 1 of reduce-then-scan: the sum of each tile. It reads every element once and writes one value per tile.
template <typename T>
__global__ void tile_reduce_kernel(const T* in, T* sums, int64_t n) {
  const int64_t base = (int64_t)blockIdx.x * TILE;
  // TODO: each thread adds ITEMS elements, base + k*BLOCK + threadIdx.x for k = 0..ITEMS-1
  // (skip indices >= n), so that neighboring threads read neighboring addresses. Then reduce
  // across the block with __shfl_down_sync and one shared slot per warp (chapter 6), and have
  // thread 0 write the tile's total to sums[blockIdx.x].
}

// Kernel 3 of reduce-then-scan: the exclusive scan of each tile, plus the tile's offset.
// `offsets` may be null, which means every offset is 0 (the one-tile case).
template <typename T>
__global__ void scan_tiles_kernel(const T* in, T* out, const T* offsets, int64_t n) {
  const int64_t base = (int64_t)blockIdx.x * TILE;
  T items[ITEMS];
  load_items<T>(in, base, n, items);
  // TODO phase 1: scan the thread's 4 items serially in registers, as an exclusive scan:
  // items[k] becomes the sum of the items before it, and `run` ends as the thread's total.
  T run = T(0);
  // TODO phase 2: T total; T prefix = block_exclusive_scan<T>(run, &total);
  // TODO phase 3: add prefix, plus offsets[blockIdx.x] when offsets is not null, to every item.
  store_items<T>(out, base, n, items);
}

template <typename T>
static void scan_recursive(const T* in, T* out, int64_t n, const torch::TensorOptions& opts, cudaStream_t stream) {
  if (n == 0) return;
  const int64_t tiles = (n + TILE - 1) / TILE;
  if (tiles == 1) {
    scan_tiles_kernel<T><<<1, BLOCK, 0, stream>>>(in, out, nullptr, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  auto sums = torch::empty({tiles}, opts);
  auto offs = torch::empty({tiles}, opts);
  tile_reduce_kernel<T><<<(unsigned)tiles, BLOCK, 0, stream>>>(in, sums.data_ptr<T>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  scan_recursive<T>(sums.data_ptr<T>(), offs.data_ptr<T>(), tiles, opts, stream);   // scan the tile sums
  scan_tiles_kernel<T><<<(unsigned)tiles, BLOCK, 0, stream>>>(in, out, offs.data_ptr<T>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// 3. Histograms of bytes into 256 bins.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void count_word(int* local, unsigned w) {
  atomicAdd(&local[w & 255u], 1);
  atomicAdd(&local[(w >> 8) & 255u], 1);
  atomicAdd(&local[(w >> 16) & 255u], 1);
  atomicAdd(&local[(w >> 24) & 255u], 1);
}

// Privatized: each block counts into its own 256-bin copy in shared memory, then adds the copy to the global bins.
// `nvec` is the number of 16-byte chunks at the start of `in` (0 when `in` is not 16-byte aligned).
__global__ void hist_shared_kernel(const uint8_t* in, int* bins, int64_t n, int64_t nvec) {
  __shared__ int local[256];
  // TODO: 1. zero local[] with the block's threads, then __syncthreads().
  //       2. grid-stride loop over the nvec 16-byte chunks: load a uint4, and for each of its
  //          four 32-bit words call count_word(local, word).
  //       3. grid-stride loop over the remaining bytes, from index nvec*16 to n:
  //          atomicAdd(&local[in[i]], 1).
  //       4. __syncthreads(), then add each nonzero local[b] to bins[b] with atomicAdd.
}

// Stretch goal: a warp-aggregated histogram kernel. Add it here, for example as
//   __global__ void hist_agg_kernel(const uint8_t* in, int* bins, int64_t n, int64_t nwords)
// and call it from histogram_cuda for variant 2. The chapter sketches the idea:
// __match_any_sync groups the lanes that hold the same byte, and one lane of each group does one atomicAdd.

// The baseline: every thread adds straight to the global bins.
__global__ void hist_global_kernel(const uint8_t* in, int* bins, int64_t n) {
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) atomicAdd(&bins[in[i]], 1);
}

// ---------------------------------------------------------------------------
// Host side.
// ---------------------------------------------------------------------------

std::vector<torch::Tensor> block_scan_cuda(torch::Tensor x, int64_t variant) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kInt && x.dim() == 1 && x.is_contiguous(),
              "block_scan needs a contiguous 1-D int32 CUDA tensor");
  const int64_t n = x.numel();
  const int64_t tiles = (n + BLOCK - 1) / BLOCK;
  auto out = torch::empty_like(x);
  auto sums = torch::empty({tiles}, x.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (tiles > 0) {
    if (variant == 0) {
      block_scan_ks_kernel<<<(unsigned)tiles, BLOCK, 0, stream>>>(x.data_ptr<int>(), out.data_ptr<int>(), sums.data_ptr<int>(), n);
    } else {
      block_scan_blelloch_kernel<<<(unsigned)tiles, BLOCK, 0, stream>>>(x.data_ptr<int>(), out.data_ptr<int>(), sums.data_ptr<int>(), n);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {out, sums};
}

torch::Tensor exclusive_scan_cuda(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 1 && x.is_contiguous(), "exclusive_scan needs a contiguous 1-D CUDA tensor");
  // The tile loads are 16-byte vector accesses, so a view such as x[1:] that starts off a 16-byte boundary is copied first.
  if (reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 != 0) x = x.clone();
  auto out = torch::empty_like(x);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (x.scalar_type() == torch::kInt) {
    scan_recursive<int>(x.data_ptr<int>(), out.data_ptr<int>(), x.numel(), x.options(), stream);
  } else if (x.scalar_type() == torch::kFloat) {
    scan_recursive<float>(x.data_ptr<float>(), out.data_ptr<float>(), x.numel(), x.options(), stream);
  } else {
    TORCH_CHECK(false, "exclusive_scan supports int32 and float32");
  }
  return out;
}

// variant: 0 = global atomics, 1 = shared-memory privatized, 2 = warp-aggregated.
torch::Tensor histogram_cuda(torch::Tensor x, int64_t variant) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kByte && x.dim() == 1 && x.is_contiguous(),
              "histogram needs a contiguous 1-D uint8 CUDA tensor");
  auto bins = torch::zeros({256}, x.options().dtype(torch::kInt));
  const int64_t n = x.numel();
  if (n == 0) return bins;
  const uint8_t* p = x.data_ptr<uint8_t>();
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t max_blocks = (int64_t)sms * 8;   // 8 blocks of 256 threads fill an SM's 2048 threads
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto blocks_for = [&](int64_t per_block) {
    int64_t b = (n + per_block - 1) / per_block;
    return (unsigned)std::max<int64_t>(1, std::min<int64_t>(b, max_blocks));
  };
  if (variant == 0) {
    hist_global_kernel<<<blocks_for(BLOCK), BLOCK, 0, stream>>>(p, bins.data_ptr<int>(), n);
  } else if (variant == 1) {
    const int64_t nvec = (reinterpret_cast<uintptr_t>(p) % 16 == 0) ? n / 16 : 0;
    hist_shared_kernel<<<blocks_for(BLOCK * 16), BLOCK, 0, stream>>>(p, bins.data_ptr<int>(), n, nvec);
  } else {
    TORCH_CHECK(false, "variant 2 (warp-aggregated) is the stretch goal: write hist_agg_kernel first");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return bins;
}
"""

CPP_SRC = """
std::vector<torch::Tensor> block_scan_cuda(torch::Tensor x, int64_t variant);
torch::Tensor exclusive_scan_cuda(torch::Tensor x);
torch::Tensor histogram_cuda(torch::Tensor x, int64_t variant);
"""

_module = None


def _mod():
    global _module
    if _module is None:
        _module = load_cuda(
            "scan_hist",
            CUDA_SRC,
            CPP_SRC,
            ["block_scan_cuda", "exclusive_scan_cuda", "histogram_cuda"],
        )
    return _module


def block_scan_kogge_stone(x: torch.Tensor):
    """Exclusive scan of every 256-element tile of an int32 tensor, Kogge-Stone.

    Returns (scanned, tile_sums): `scanned` has the shape of `x`, and
    `tile_sums[i]` is the total of tile i.
    """
    scanned, sums = _mod().block_scan_cuda(x.contiguous(), 0)
    return scanned, sums


def block_scan_blelloch(x: torch.Tensor):
    """The same contract as `block_scan_kogge_stone`, computed with Blelloch's up-sweep and down-sweep."""
    scanned, sums = _mod().block_scan_cuda(x.contiguous(), 1)
    return scanned, sums


def exclusive_scan(x: torch.Tensor) -> torch.Tensor:
    """Exclusive prefix sum of a 1-D int32 or float32 CUDA tensor of any length."""
    return _mod().exclusive_scan_cuda(x.contiguous())


def histogram256_global(x: torch.Tensor) -> torch.Tensor:
    """256-bin histogram of a uint8 tensor with one global atomic per byte (the baseline)."""
    return _mod().histogram_cuda(x.contiguous(), 0)


def histogram256(x: torch.Tensor) -> torch.Tensor:
    """256-bin histogram of a uint8 tensor, privatized in shared memory. Returns int32[256]."""
    return _mod().histogram_cuda(x.contiguous(), 1)


def histogram256_aggregated(x: torch.Tensor) -> torch.Tensor:
    """Stretch goal: warp-aggregated histogram, for skewed data. Returns int32[256].

    Write `hist_agg_kernel` in CUDA_SRC and launch it for variant 2 in
    `histogram_cuda`, then delete the `raise` below. The tests skip this
    function while it raises NotImplementedError.
    """
    raise NotImplementedError("stretch goal")
