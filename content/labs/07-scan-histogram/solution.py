"""Lab 07: Scan and histogram. Reference solution.

Three groups of kernels, all in one CUDA source:

1. Block scans, one element per thread, one 256-element tile per block:
   Kogge-Stone and Blelloch, both in shared memory.
2. A device-wide exclusive scan by reduce-then-scan: a tile-reduce kernel, a
   (recursive) scan of the tile sums, and a scan kernel that adds each tile's
   offset. Each tile is 1024 elements: 256 threads, 4 items per thread, a
   serial scan in registers, and a warp-shuffle scan across threads.
3. A 256-bin histogram of bytes: global atomics (given), a shared-memory
   privatized version, and a warp-aggregated version for skewed data.
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
  // Kogge-Stone: at distance d, every thread adds the value d places to its left.
  // Read, barrier, write, barrier: a thread must not overwrite a value that a
  // neighbor has yet to read.
  for (int d = 1; d < BLOCK; d <<= 1) {
    int add = (tid >= d) ? buf[tid - d] : 0;
    __syncthreads();
    buf[tid] += add;
    __syncthreads();
  }
  // buf now holds the inclusive scan. The exclusive scan is the same array shifted by one.
  if (i < n) out[i] = (tid > 0) ? buf[tid - 1] : 0;
  if (tid == BLOCK - 1) sums[blockIdx.x] = buf[tid];
}

__global__ void block_scan_blelloch_kernel(const int* in, int* out, int* sums, int64_t n) {
  __shared__ int t[BLOCK];
  const int tid = threadIdx.x;
  const int64_t i = (int64_t)blockIdx.x * BLOCK + tid;
  t[tid] = (i < n) ? in[i] : 0;
  int offset = 1;
  // Up-sweep (reduce): build partial sums in place, up a binary tree.
  for (int d = BLOCK >> 1; d > 0; d >>= 1) {
    __syncthreads();
    if (tid < d) {
      int a = offset * (2 * tid + 1) - 1;
      int b = offset * (2 * tid + 2) - 1;
      t[b] += t[a];
    }
    offset <<= 1;
  }
  __syncthreads();
  if (tid == 0) {
    sums[blockIdx.x] = t[BLOCK - 1];   // the root holds the tile's total
    t[BLOCK - 1] = 0;                  // clear it: the identity for an exclusive scan
  }
  // Down-sweep: pass each node's prefix down to its children.
  for (int d = 1; d < BLOCK; d <<= 1) {
    offset >>= 1;
    __syncthreads();
    if (tid < d) {
      int a = offset * (2 * tid + 1) - 1;
      int b = offset * (2 * tid + 2) - 1;
      int tmp = t[a];
      t[a] = t[b];
      t[b] += tmp;
    }
  }
  __syncthreads();
  if (i < n) out[i] = t[tid];
}

// ---------------------------------------------------------------------------
// 2. The device-wide scan, for int and float.
// ---------------------------------------------------------------------------

// Inclusive scan across the 32 lanes of a warp, with shuffles. Every lane of
// the warp must call it.
template <typename T>
__device__ __forceinline__ T warp_inclusive_scan(T v) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    T up = __shfl_up_sync(FULL, v, d);
    if (lane >= d) v += up;
  }
  return v;
}

// Exclusive scan of one value per thread across the 256-thread block. Returns
// this thread's exclusive prefix and writes the block total to *total.
// Call it once per kernel: it reuses one static shared array without a final barrier.
template <typename T>
__device__ T block_exclusive_scan(T v, T* total) {
  __shared__ T warp_sums[BLOCK / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  T incl = warp_inclusive_scan(v);
  T excl = __shfl_up_sync(FULL, incl, 1);
  if (lane == 0) excl = T(0);
  if (lane == 31) warp_sums[warp] = incl;
  __syncthreads();
  if (warp == 0) {
    T w = (lane < BLOCK / 32) ? warp_sums[lane] : T(0);
    w = warp_inclusive_scan(w);
    if (lane < BLOCK / 32) warp_sums[lane] = w;   // now the inclusive scan of the warp totals
  }
  __syncthreads();
  T warp_prefix = (warp > 0) ? warp_sums[warp - 1] : T(0);
  *total = warp_sums[BLOCK / 32 - 1];
  return warp_prefix + excl;
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
  __shared__ T ws[BLOCK / 32];
  const int64_t base = (int64_t)blockIdx.x * TILE;
  T acc = T(0);
#pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    int64_t i = base + (int64_t)k * BLOCK + threadIdx.x;   // coalesced: neighbors read neighbors
    if (i < n) acc += in[i];
  }
  for (int d = 16; d > 0; d >>= 1) acc += __shfl_down_sync(FULL, acc, d);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) ws[warp] = acc;
  __syncthreads();
  if (warp == 0) {
    T v = (lane < BLOCK / 32) ? ws[lane] : T(0);
    for (int d = 4; d > 0; d >>= 1) v += __shfl_down_sync(FULL, v, d);
    if (lane == 0) sums[blockIdx.x] = v;
  }
}

// Kernel 3 of reduce-then-scan: the exclusive scan of each tile, plus the tile's offset.
// `offsets` may be null, which means every offset is 0 (the one-tile case).
template <typename T>
__global__ void scan_tiles_kernel(const T* in, T* out, const T* offsets, int64_t n) {
  const int64_t base = (int64_t)blockIdx.x * TILE;
  T items[ITEMS];
  load_items<T>(in, base, n, items);
  // Phase 1: each thread scans its own 4 items serially, in registers.
  T run = T(0);
#pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    T x = items[k];
    items[k] = run;
    run += x;
  }
  // Phase 2: scan the 256 thread totals across the block.
  T total;
  T prefix = block_exclusive_scan<T>(run, &total);
  // Phase 3: add the tile's offset and the thread's prefix to every item.
  T add = prefix + ((offsets != nullptr) ? offsets[blockIdx.x] : T(0));
#pragma unroll
  for (int k = 0; k < ITEMS; ++k) items[k] += add;
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
  for (int b = threadIdx.x; b < 256; b += blockDim.x) local[b] = 0;
  __syncthreads();
  const int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  const uint4* in4 = reinterpret_cast<const uint4*>(in);
  for (int64_t v = gid; v < nvec; v += stride) {
    uint4 w = in4[v];
    count_word(local, w.x);
    count_word(local, w.y);
    count_word(local, w.z);
    count_word(local, w.w);
  }
  for (int64_t i = nvec * 16 + gid; i < n; i += stride) atomicAdd(&local[in[i]], 1);
  __syncthreads();
  for (int b = threadIdx.x; b < 256; b += blockDim.x) {
    int c = local[b];
    if (c != 0) atomicAdd(&bins[b], c);
  }
}

// One element per lane. Lanes holding the same key elect a leader, which does one atomic for the whole group.
// key < 0 marks an inactive lane. Every lane of the warp must call it.
__device__ __forceinline__ void aggregate_add(int* local, int key) {
  unsigned peers = __match_any_sync(FULL, key);
  const int lane = threadIdx.x & 31;
  if (key >= 0 && lane == __ffs(peers) - 1) atomicAdd(&local[key], __popc(peers));
}

// Warp-aggregated. The loop bounds depend only on blockIdx, so all lanes of a warp take every trip together.
// `nwords` is the number of 4-byte words at the start of `in` (0 when `in` is not 4-byte aligned).
__global__ void hist_agg_kernel(const uint8_t* in, int* bins, int64_t n, int64_t nwords) {
  __shared__ int local[256];
  for (int b = threadIdx.x; b < 256; b += blockDim.x) local[b] = 0;
  __syncthreads();
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  const unsigned* in32 = reinterpret_cast<const unsigned*>(in);
  for (int64_t base = (int64_t)blockIdx.x * blockDim.x; base < nwords; base += stride) {
    int64_t idx = base + threadIdx.x;
    bool valid = idx < nwords;
    unsigned w = valid ? in32[idx] : 0u;
#pragma unroll
    for (int j = 0; j < 4; ++j) aggregate_add(local, valid ? (int)((w >> (8 * j)) & 255u) : -1);
  }
  for (int64_t base = nwords * 4 + (int64_t)blockIdx.x * blockDim.x; base < n; base += stride) {
    int64_t i = base + threadIdx.x;
    aggregate_add(local, (i < n) ? (int)in[i] : -1);
  }
  __syncthreads();
  for (int b = threadIdx.x; b < 256; b += blockDim.x) {
    int c = local[b];
    if (c != 0) atomicAdd(&bins[b], c);
  }
}

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
    const int64_t nwords = (reinterpret_cast<uintptr_t>(p) % 4 == 0) ? n / 4 : 0;
    hist_agg_kernel<<<blocks_for(BLOCK * 4), BLOCK, 0, stream>>>(p, bins.data_ptr<int>(), n, nwords);
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
    """Stretch goal: warp-aggregated histogram, for skewed data. Returns int32[256]."""
    return _mod().histogram_cuda(x.contiguous(), 2)
