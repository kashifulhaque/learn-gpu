"""Lab 30: sort, select, and sample, built from a histogram and a scan.

Do the CPU pieces first, in order. Check 1 needs only int32_to_key. They are the order-preserving key transforms, one
stable radix pass, a full LSD radix sort, the merge-path diagonal search,
radix-select top-k, a top-p mask that needs no full sort, and a Philox
counter-based sampler. The last function is a CUDA kernel: the histogram of one
radix digit. A machine without a GPU skips it.

1. int32_to_key
2. float32_to_key, key_to_float32
3. float32_sort_key
4. radix_pass
5. radix_sort
6. merge_path_partition, merge_path_merge
7. radix_select_topk
8. top_p_mask
9. philox4x32_10, sample_categorical
10. radix_histogram (CUDA)

Keys are int64 tensors that hold unsigned 32-bit values (0 to 2**32 - 1), so
that Python arithmetic and torch comparisons work without an unsigned dtype.
"""

from __future__ import annotations

import torch

MASK32 = 0xFFFFFFFF
SIGN32 = 0x80000000


# ---------------------------------------------------------------------------
# 1 to 3. Order-preserving keys.
# ---------------------------------------------------------------------------


def int32_to_key(x: torch.Tensor) -> torch.Tensor:
    """Map int32 values to unsigned 32-bit keys that sort in the same order.

    Flip the sign bit. Return an int64 tensor with values in 0 .. 2**32 - 1.
    Convert to int64 and mask with MASK32 first, so -1 is 0xFFFFFFFF before the flip.
    """
    raise NotImplementedError("int32_to_key")


def float32_to_key(x: torch.Tensor) -> torch.Tensor:
    """Bit-exact, invertible float32 to unsigned key map (a total order).

    A positive float (sign bit 0) sorts correctly as its bits once the sign bit
    is set. A negative float (sign bit 1) sorts in reverse, so invert every bit.
    View the float32 bits as int32, then mask to 32 bits in int64.
    """
    raise NotImplementedError("float32_to_key")


def key_to_float32(key: torch.Tensor) -> torch.Tensor:
    """Inverse of float32_to_key. The result has the same bits as the input.

    A key with the top bit set came from a positive float.
    """
    raise NotImplementedError("key_to_float32")


def float32_sort_key(x: torch.Tensor) -> torch.Tensor:
    """The key a sort uses: float32_to_key, with -0.0 equal to +0.0 and one key for every NaN.

    torch.sort treats -0.0 and +0.0 as equal and puts NaN after +inf. Map both
    zeros to the key of +0.0, and every NaN, whatever its sign or payload, to
    the largest key, 0xFFFFFFFF.
    """
    raise NotImplementedError("float32_sort_key")


# ---------------------------------------------------------------------------
# 4 and 5. Radix sort.
# ---------------------------------------------------------------------------


def radix_pass(keys: torch.Tensor, idx: torch.Tensor, shift: int, bits: int):
    """One stable counting-sort pass on the digit (key >> shift) & (2**bits - 1).

    Histogram the digits, take the exclusive scan, and scatter each element to
    the next free slot of its digit in input order. Return (keys, idx) reordered.
    Walk the input from the first element to the last, and fill each bin from its start.
    """
    raise NotImplementedError("radix_pass")


def radix_sort(x: torch.Tensor, bits: int = 8):
    """Stable LSD radix sort of a 1-D int32 or float32 tensor.

    Return (values, indices), equal to torch.sort(x, stable=True). Use
    int32_to_key for int32 and float32_sort_key for float32, then run
    32 // bits passes from the lowest digit to the highest.
    """
    raise NotImplementedError("radix_sort")


# ---------------------------------------------------------------------------
# 6. Merge path.
# ---------------------------------------------------------------------------


def merge_path_partition(a: list, b: list, diag: int) -> int:
    """How many of the first `diag` merged elements come from `a`.

    Binary search on the diagonal i + j = diag. On a tie, `a` comes first, so the
    merge is stable. The count from `b` is diag - result.
    """
    raise NotImplementedError("merge_path_partition")


def merge_path_merge(a: list, b: list, parts: int):
    """Merge two sorted lists in `parts` independent slices.

    Return (merged, source). source[t] is 0 when merged[t] came from `a` and 1
    when it came from `b`. Cut the output at diagonals (p * total) // parts,
    find each cut with merge_path_partition, and merge each slice on its own.
    """
    raise NotImplementedError("merge_path_merge")


# ---------------------------------------------------------------------------
# 7. Radix-select top-k.
# ---------------------------------------------------------------------------


def radix_select_topk(x: torch.Tensor, k: int, bits: int = 8):
    """The k largest values of a 1-D float32 tensor, by radix select.

    Walk the key from the highest digit down. At each digit, histogram the
    candidates, scan the counts from the largest digit, and keep only the
    candidates in the digit that holds the k-th largest. After 32 // bits
    passes the threshold key is known. Keep every element above it, plus the
    lowest-index elements equal to it until k are kept.

    Return (values, indices) ordered like a stable descending sort: larger
    values first, and the lower index first among equal values.
    """
    raise NotImplementedError("radix_select_topk")


# ---------------------------------------------------------------------------
# 8. Top-p without a full sort.
# ---------------------------------------------------------------------------


def top_p_mask(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Boolean mask of the nucleus of a 1-D float32 probability vector.

    A token stays when the probability mass of the tokens ranked before it is
    less than p. Ties rank by lower index first. Do not sort: binary-search the
    float bit pattern of the threshold probability, using the sum of the
    probabilities at or above each candidate, then fix up the ties.
    Sum in float64.
    """
    raise NotImplementedError("top_p_mask")


# ---------------------------------------------------------------------------
# 9. Philox and seeded sampling.
# ---------------------------------------------------------------------------

PHILOX_M0 = 0xD2511F53
PHILOX_M1 = 0xCD9E8D57
PHILOX_W0 = 0x9E3779B9
PHILOX_W1 = 0xBB67AE85


def philox4x32_10(counter, key):
    """Philox-4x32-10: four 32-bit counter words and two key words in, four words out.

    Each of the ten rounds computes p0 = M0 * c0 and p1 = M1 * c2 as 64-bit products and
    sets (c0, c1, c2, c3) = (hi(p1) ^ c1 ^ k0, lo(p1), hi(p0) ^ c3 ^ k1, lo(p0)).
    The key then adds W0 and W1, mod 2**32.
    """
    raise NotImplementedError("philox4x32_10")


def sample_categorical(probs: torch.Tensor, seed: int, step: int = 0, row_ids=None) -> torch.Tensor:
    """Draw one token per row of a 2-D probability tensor, reproducibly.

    Row r uses the Philox counter (row_id, 0, step, 0) and the key
    (seed low word, seed high word). The first output word becomes
    u = (word >> 8) / 2**24, and the token is the first index whose cumulative
    probability exceeds u times the row total. `row_ids` defaults to 0, 1, 2, ...
    The draw depends on (seed, step, row id), never on the batch position.
    """
    raise NotImplementedError("sample_categorical")


# ---------------------------------------------------------------------------
# 10. CUDA: histogram of one radix digit.
# ---------------------------------------------------------------------------

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <cstdint>

constexpr int BLOCK = 256;

// Count the digit (key >> shift) & (nbins - 1) of every key. Each block counts
// into its own shared-memory copy and merges it into the global bins once.
__global__ void radix_hist_kernel(const uint32_t* keys, int* hist, int64_t n, int shift, int nbins) {
  extern __shared__ int local[];
  // TODO: 1. zero local[0 .. nbins) with the block's threads, then __syncthreads().
  //       2. grid-stride loop over the n keys: atomicAdd(&local[(keys[i] >> shift) & mask], 1),
  //          where mask = nbins - 1. keys is uint32_t, so the shift does not copy the sign bit.
  //       3. __syncthreads(), then add each nonzero local[b] to hist[b] with atomicAdd.
}

torch::Tensor radix_histogram_cuda(torch::Tensor keys, int64_t shift, int64_t bits) {
  TORCH_CHECK(keys.is_cuda() && keys.scalar_type() == torch::kInt && keys.dim() == 1 && keys.is_contiguous(),
              "radix_histogram needs a contiguous 1-D int32 CUDA tensor");
  TORCH_CHECK(bits >= 1 && bits <= 11, "bits must be in 1 .. 11");
  TORCH_CHECK(shift >= 0 && shift < 32, "shift must be in 0 .. 31");
  const int nbins = 1 << bits;
  auto hist = torch::zeros({nbins}, keys.options());
  const int64_t n = keys.numel();
  if (n == 0) return hist;
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t blocks = std::max<int64_t>(1, std::min<int64_t>((n + BLOCK - 1) / BLOCK, (int64_t)sms * 8));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  radix_hist_kernel<<<(unsigned)blocks, BLOCK, nbins * sizeof(int), stream>>>(
      reinterpret_cast<const uint32_t*>(keys.data_ptr<int>()), hist.data_ptr<int>(), n, (int)shift, nbins);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return hist;
}
"""

CPP_SRC = """
torch::Tensor radix_histogram_cuda(torch::Tensor keys, int64_t shift, int64_t bits);
"""

_module = None


def _mod():
    global _module
    if _module is None:
        from lab_common import load_cuda

        _module = load_cuda("radix_hist", CUDA_SRC, CPP_SRC, ["radix_histogram_cuda"])
    return _module


def radix_histogram(keys: torch.Tensor, shift: int, bits: int) -> torch.Tensor:
    """Histogram of the digit (key >> shift) & (2**bits - 1) of an int32 CUDA tensor.

    The int32 bits are read as unsigned, so a negative key has a nonzero top digit.
    Return int32[2**bits].
    """
    return _mod().radix_histogram_cuda(keys.contiguous(), shift, bits)
