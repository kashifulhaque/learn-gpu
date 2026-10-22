"""Lab 12: a multistage cp.async pipeline for a tensor-core GEMM.

C = A @ B with A [M, K] and B [K, N] in bfloat16 and C [M, N] in float32. One
thread block computes a 128 x 128 tile of C with eight warps, each warp owning
a 64 x 32 piece, and walks down K in steps of 32. Each step needs a 128 x 32
slice of A and a 32 x 128 slice of B in shared memory. `STAGES` is the number
of such slices the block keeps in flight: with one stage the block loads, waits,
and computes in turn; with more, `cp.async` loads for later steps overlap the
tensor-core work on the current one.
"""

import torch
import torch.nn.functional as F
from lab_common import A100, load_cuda

# Tile geometry, shared by the Python helpers and the CUDA source below.
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32
PAD = 8  # extra bf16 elements per shared-memory row: 16 bytes, avoids bank conflicts
BYTES_PER_ELEMENT = 2  # bfloat16
RESERVED_SMEM_PER_BLOCK = 1024  # CUDA reserves 1 KB of shared memory per block (docs/FACTS.md 3.2)
MAX_STAGES = 5  # the CUDA source instantiates 1 to 5 stages

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

constexpr int kBM = 128;
constexpr int kBN = 128;
constexpr int kBK = 32;
constexpr int kPad = 8;
constexpr int kThreads = 256;
constexpr int kALd = kBK + kPad;        // elements per row of the A tile: 40 (80 bytes)
constexpr int kBLd = kBN + kPad;        // elements per row of the B tile: 136 (272 bytes)
constexpr int kAStage = kBM * kALd;     // elements of A tile per stage
constexpr int kBStage = kBK * kBLd;     // elements of B tile per stage

// ---- PTX wrappers -----------------------------------------------------------

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// Copy 16 bytes global -> shared without touching registers. When `valid` is
// false the source size is 0, so nothing is read and the 16 bytes become zero.
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
  const int src_size = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :: "r"(smem_u32(dst)), "l"(src), "r"(src_size) : "memory");
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                                            uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr) : "memory");
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                                  uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr) : "memory");
}

// D = A * B + D for one 16x8x16 tile: bf16 inputs, fp32 accumulators.
__device__ __forceinline__ void mma_bf16_16816(float* d, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ---- The kernel ---------------------------------------------------------------

template <int STAGES>
__global__ void __launch_bounds__(kThreads, 2)
gemm_kernel(const uint16_t* __restrict__ A, const uint16_t* __restrict__ B,
            float* __restrict__ C, int M, int N, int K) {
  extern __shared__ __align__(16) uint16_t smem[];
  uint16_t* As = smem;                     // STAGES tiles of kBM x kALd
  uint16_t* Bs = smem + STAGES * kAStage;  // STAGES tiles of kBK x kBLd

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int warp_m = warp >> 2;  // 0..1: which 64 rows of the block tile
  const int warp_n = warp & 3;   // 0..3: which 32 columns of the block tile
  const int m0 = blockIdx.y * kBM;
  const int n0 = blockIdx.x * kBN;
  const int k_tiles = (K + kBK - 1) / kBK;

  // Issue the copies for k-tile `kt` into ring slot `stage`. Each thread copies
  // two 16-byte chunks of A and two of B. Chunks outside the matrix are
  // zero-filled, which handles M, N, and K that are not multiples of the tile.
  auto load_tile = [&](int stage, int kt) {
    const int k0 = kt * kBK;
    uint16_t* as = As + stage * kAStage;
    uint16_t* bs = Bs + stage * kBStage;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int c = tid + i * kThreads;  // chunk id, 0..511
      const int ar = c >> 2, ac = (c & 3) * 8;     // A tile: 128 rows x 4 chunks
      const int gm = m0 + ar, gk = k0 + ac;
      const bool a_ok = gm < M && gk < K;
      cp_async16(as + ar * kALd + ac, a_ok ? A + (size_t)gm * K + gk : A, a_ok);
      const int br = c >> 4, bc = (c & 15) * 8;    // B tile: 32 rows x 16 chunks
      const int gk2 = k0 + br, gn = n0 + bc;
      const bool b_ok = gk2 < K && gn < N;
      cp_async16(bs + br * kBLd + bc, b_ok ? B + (size_t)gk2 * N + gn : B, b_ok);
    }
  };

  float acc[4][4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.f;

  // Tensor-core work on the k-tile in ring slot `stage`: two k16 steps, each
  // loading 4 A fragments and 4 B fragments with ldmatrix and issuing 16 MMAs.
  auto compute_tile = [&](int stage) {
    const uint16_t* as = As + stage * kAStage;
    const uint16_t* bs = Bs + stage * kBStage;
#pragma unroll
    for (int kk = 0; kk < kBK; kk += 16) {
      uint32_t a[4][4];
      uint32_t b[4][2];
#pragma unroll
      for (int mi = 0; mi < 4; ++mi) {
        const uint16_t* p = as + (warp_m * 64 + mi * 16 + (lane & 15)) * kALd + kk + (lane >> 4) * 8;
        ldmatrix_x4(a[mi][0], a[mi][1], a[mi][2], a[mi][3], smem_u32(p));
      }
#pragma unroll
      for (int nj = 0; nj < 2; ++nj) {
        const uint16_t* p = bs + (kk + (lane & 15)) * kBLd + warp_n * 32 + nj * 16 + (lane >> 4) * 8;
        ldmatrix_x4_trans(b[2 * nj][0], b[2 * nj][1], b[2 * nj + 1][0], b[2 * nj + 1][1], smem_u32(p));
      }
#pragma unroll
      for (int mi = 0; mi < 4; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) mma_bf16_16816(acc[mi][ni], a[mi], b[ni]);
    }
  };

  if constexpr (STAGES == 1) {
    // One stage: load, wait for the copy, compute, and only then reuse the slot.
    for (int kt = 0; kt < k_tiles; ++kt) {
      load_tile(0, kt);
      cp_async_commit();
      cp_async_wait<0>();
      __syncthreads();
      compute_tile(0);
      __syncthreads();
    }
  } else {
    // Prologue: put STAGES - 1 tiles in flight. Commit a group even when the
    // tile is past the end, so every thread's group count stays the same.
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
      if (s < k_tiles) load_tile(s, s);
      cp_async_commit();
    }
    // Steady state: at the top of iteration kt, groups 0 .. kt + STAGES - 2 are
    // committed. Tile kt is group kt, so waiting until at most STAGES - 2 are
    // pending guarantees tile kt has landed.
    for (int kt = 0; kt < k_tiles; ++kt) {
      cp_async_wait<STAGES - 2>();
      __syncthreads();  // tile kt is visible to every warp, and every warp is done with tile kt - 1
      const int next = kt + STAGES - 1;
      if (next < k_tiles) load_tile(next % STAGES, next);  // the slot tile kt - 1 used
      cp_async_commit();
      compute_tile(kt % STAGES);
    }
  }
  cp_async_wait<0>();

  // Epilogue: each thread owns pairs of adjacent columns of the accumulators.
#pragma unroll
  for (int mi = 0; mi < 4; ++mi) {
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      const int row = m0 + warp_m * 64 + mi * 16 + (lane >> 2);
      const int col = n0 + warp_n * 32 + ni * 8 + (lane & 3) * 2;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = row + h * 8;
        if (r < M) {
          if (col < N) C[(size_t)r * N + col] = acc[mi][ni][2 * h];
          if (col + 1 < N) C[(size_t)r * N + col + 1] = acc[mi][ni][2 * h + 1];
        }
      }
    }
  }
}

template <int S>
static void launch(const uint16_t* a, const uint16_t* b, float* c, int M, int N, int K,
                   cudaStream_t stream) {
  constexpr size_t smem = (size_t)S * (kAStage + kBStage) * sizeof(uint16_t);
  cudaFuncSetAttribute(reinterpret_cast<const void*>(gemm_kernel<S>),
                       cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
  dim3 grid((N + kBN - 1) / kBN, (M + kBM - 1) / kBM);
  gemm_kernel<S><<<grid, kThreads, smem, stream>>>(a, b, c, M, N, K);
}

torch::Tensor gemm_cuda(torch::Tensor a, torch::Tensor b, int64_t stages) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "a and b must be CUDA tensors");
  TORCH_CHECK(a.scalar_type() == torch::kBFloat16 && b.scalar_type() == torch::kBFloat16,
              "a and b must be bfloat16");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(0), "shapes must be [M,K] and [K,N]");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "a and b must be contiguous");
  const int M = a.size(0), K = a.size(1), N = b.size(1);
  TORCH_CHECK(K % 8 == 0 && N % 8 == 0, "K and N must be multiples of 8 (16-byte rows)");
  auto c = torch::empty({M, N}, a.options().dtype(torch::kFloat32));
  if (M == 0 || N == 0) return c;
  if (K == 0) return c.zero_();
  const uint16_t* pa = reinterpret_cast<const uint16_t*>(a.data_ptr());
  const uint16_t* pb = reinterpret_cast<const uint16_t*>(b.data_ptr());
  float* pc = c.data_ptr<float>();
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  switch (stages) {
    case 1: launch<1>(pa, pb, pc, M, N, K, st); break;
    case 2: launch<2>(pa, pb, pc, M, N, K, st); break;
    case 3: launch<3>(pa, pb, pc, M, N, K, st); break;
    case 4: launch<4>(pa, pb, pc, M, N, K, st); break;
    case 5: launch<5>(pa, pb, pc, M, N, K, st); break;
    default: TORCH_CHECK(false, "stages must be 1 to 5");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}
"""

CPP_SRC = "torch::Tensor gemm_cuda(torch::Tensor a, torch::Tensor b, int64_t stages);"


def stage_smem_bytes(stages: int) -> int:
    """Shared memory, in bytes, that `stages` pipeline stages need.

    One stage holds a BLOCK_M x (BLOCK_K + PAD) tile of A and a
    BLOCK_K x (BLOCK_N + PAD) tile of B, in bfloat16.
    """
    per_stage = (BLOCK_M * (BLOCK_K + PAD) + BLOCK_K * (BLOCK_N + PAD)) * BYTES_PER_ELEMENT
    return stages * per_stage


def max_stages(
    smem_per_sm_bytes: int = A100.SMEM_PER_SM_BYTES,
    blocks_per_sm: int = 1,
    per_block_cap_bytes: int = A100.SMEM_PER_BLOCK_MAX_BYTES,
) -> int:
    """The most stages that fit when `blocks_per_sm` blocks share one SM.

    Each block gets an equal share of the SM's shared memory, minus the 1 KB
    CUDA reserves per block, and no block may exceed `per_block_cap_bytes`.
    """
    usable = min(smem_per_sm_bytes // blocks_per_sm - RESERVED_SMEM_PER_BLOCK, per_block_cap_bytes)
    return max(0, usable // stage_smem_bytes(1))


def schedule(num_k_tiles: int, stages: int) -> dict:
    """The order of copies, waits, and computes in the kernel's main loop.

    Returns {"prologue": [...], "steps": [...]}.

    - "prologue" has one entry per group committed before the loop: the k-tile
      loaded into slot s, or None when the tile is past the end. For one stage
      it is empty. Otherwise it has stages - 1 entries.
    - "steps" has one dict per loop iteration kt, with the keys
      "kt", "wait_pending", "load_tile", "load_slot", "compute_tile",
      "compute_slot". `wait_pending` is the N in cp.async.wait_group N that
      runs before computing tile kt. `load_tile` is the tile whose copy this
      iteration issues (None past the end, though the group is still
      committed), and `load_slot` the ring slot it goes to.
    """
    steps = []
    if stages == 1:
        for kt in range(num_k_tiles):
            steps.append(
                {"kt": kt, "wait_pending": 0, "load_tile": kt, "load_slot": 0,
                 "compute_tile": kt, "compute_slot": 0}
            )
        return {"prologue": [], "steps": steps}
    prologue = [s if s < num_k_tiles else None for s in range(stages - 1)]
    for kt in range(num_k_tiles):
        nxt = kt + stages - 1
        steps.append(
            {
                "kt": kt,
                "wait_pending": stages - 2,
                "load_tile": nxt if nxt < num_k_tiles else None,
                "load_slot": nxt % stages,
                "compute_tile": kt,
                "compute_slot": kt % stages,
            }
        )
    return {"prologue": prologue, "steps": steps}


def _module():
    return load_cuda("cp_async_gemm", CUDA_SRC, CPP_SRC, ["gemm_cuda"])


def gemm(a: torch.Tensor, b: torch.Tensor, stages: int = 3) -> torch.Tensor:
    """C = a @ b for bfloat16 a [M, K] and b [K, N], accumulated in float32.

    The kernel copies 16-byte chunks, so K and N must be multiples of 8. This
    wrapper pads them with zeros, which leaves the product unchanged, and
    slices the padding off the result.
    """
    if not 1 <= stages <= MAX_STAGES:
        raise ValueError(f"stages must be 1 to {MAX_STAGES}")
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError("inner dimensions differ")
    pad_k = (-k) % 8
    pad_n = (-n) % 8
    a = a.contiguous()
    b = b.contiguous()
    if pad_k:
        a = F.pad(a, (0, pad_k))
        b = F.pad(b, (0, 0, 0, pad_k))
    if pad_n:
        b = F.pad(b, (0, pad_n))
    c = _module().gemm_cuda(a, b, stages)
    return c[:, :n] if pad_n else c
