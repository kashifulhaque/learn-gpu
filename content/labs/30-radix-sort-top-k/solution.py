"""Lab 30: sort, select, and sample, built from a histogram and a scan.

Do the CPU pieces first. They are the order-preserving key transforms, one
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
    """
    return (x.to(torch.int64) & MASK32) ^ SIGN32


def float32_to_key(x: torch.Tensor) -> torch.Tensor:
    """Bit-exact, invertible float32 to unsigned key map (a total order).

    A positive float (sign bit 0) sorts correctly as its bits once the sign bit
    is set. A negative float (sign bit 1) sorts in reverse, so invert every bit.
    """
    bits = x.contiguous().view(torch.int32).to(torch.int64) & MASK32
    negative = (bits >> 31) == 1
    return torch.where(negative, bits ^ MASK32, bits ^ SIGN32)


def key_to_float32(key: torch.Tensor) -> torch.Tensor:
    """Inverse of float32_to_key. The result has the same bits as the input."""
    key = key.to(torch.int64)
    was_positive = key >= SIGN32
    bits = torch.where(was_positive, key ^ SIGN32, key ^ MASK32)
    signed = torch.where(bits >= SIGN32, bits - (1 << 32), bits)
    return signed.to(torch.int32).view(torch.float32)


def float32_sort_key(x: torch.Tensor) -> torch.Tensor:
    """The key a sort uses: float32_to_key, with -0.0 equal to +0.0 and one key for every NaN.

    torch.sort treats -0.0 and +0.0 as equal and puts NaN after +inf. Map both
    zeros to the key of +0.0, and every NaN, whatever its sign or payload, to
    the largest key, 0xFFFFFFFF.
    """
    key = float32_to_key(x)
    key = torch.where(x == 0, torch.full_like(key, SIGN32), key)
    key = torch.where(torch.isnan(x), torch.full_like(key, MASK32), key)
    return key


# ---------------------------------------------------------------------------
# 4 and 5. Radix sort.
# ---------------------------------------------------------------------------


def radix_pass(keys: torch.Tensor, idx: torch.Tensor, shift: int, bits: int):
    """One stable counting-sort pass on the digit (key >> shift) & (2**bits - 1).

    Histogram the digits, take the exclusive scan, and scatter each element to
    the next free slot of its digit in input order. Return (keys, idx) reordered.
    """
    nbins = 1 << bits
    mask = nbins - 1
    key_list = keys.tolist()
    idx_list = idx.tolist()
    n = len(key_list)
    digits = [(k >> shift) & mask for k in key_list]
    counts = [0] * nbins
    for d in digits:
        counts[d] += 1
    offsets = [0] * nbins
    running = 0
    for d in range(nbins):
        offsets[d] = running
        running += counts[d]
    out_keys = [0] * n
    out_idx = [0] * n
    # Forward order from the start of each bin keeps equal digits in input order.
    # unstable_scatter fills each bin from its end while walking forward.
    for pos in range(n):
        d = digits[pos]
        slot = offsets[d]
        offsets[d] = slot + 1
        out_keys[slot] = key_list[pos]
        out_idx[slot] = idx_list[pos]
    return (
        torch.tensor(out_keys, dtype=torch.int64),
        torch.tensor(out_idx, dtype=torch.int64),
    )


def radix_sort(x: torch.Tensor, bits: int = 8):
    """Stable LSD radix sort of a 1-D int32 or float32 tensor.

    Return (values, indices), equal to torch.sort(x, stable=True). Use
    int32_to_key for int32 and float32_sort_key for float32, then run
    32 // bits passes from the lowest digit to the highest.
    """
    if x.dtype == torch.int32:
        keys = int32_to_key(x)
    elif x.dtype == torch.float32:
        keys = float32_sort_key(x)
    else:
        raise TypeError(f"radix_sort takes int32 or float32, actual {x.dtype}")
    if 32 % bits != 0:
        raise ValueError(f"bits must divide 32, actual {bits}")
    idx = torch.arange(x.numel(), dtype=torch.int64)
    for shift in range(0, 32, bits):
        keys, idx = radix_pass(keys, idx, shift, bits)
    return x[idx], idx


# ---------------------------------------------------------------------------
# 6. Merge path.
# ---------------------------------------------------------------------------


def merge_path_partition(a: list, b: list, diag: int) -> int:
    """How many of the first `diag` merged elements come from `a`.

    Binary search on the diagonal i + j = diag. On a tie, `a` comes first, so the
    merge is stable. The count from `b` is diag - result.
    """
    lo = max(0, diag - len(b))
    hi = min(diag, len(a))
    while lo < hi:
        mid = (lo + hi) // 2
        # a[mid] precedes b[diag - 1 - mid] when it is smaller or equal.
        if a[mid] <= b[diag - 1 - mid]:
            lo = mid + 1
        else:
            hi = mid
    return lo


def merge_path_merge(a: list, b: list, parts: int):
    """Merge two sorted lists in `parts` independent slices.

    Return (merged, source). source[t] is 0 when merged[t] came from `a` and 1
    when it came from `b`. Cut the output at diagonals round(p * total / parts),
    find each cut with merge_path_partition, and merge each slice on its own.
    """
    total = len(a) + len(b)
    cuts = [(p * total) // parts for p in range(parts + 1)]
    merged: list = []
    source: list = []
    for p in range(parts):
        i = merge_path_partition(a, b, cuts[p])
        j = cuts[p] - i
        i_end = merge_path_partition(a, b, cuts[p + 1])
        j_end = cuts[p + 1] - i_end
        while i < i_end or j < j_end:
            take_a = j >= j_end or (i < i_end and a[i] <= b[j])
            if take_a:
                merged.append(a[i])
                source.append(0)
                i += 1
            else:
                merged.append(b[j])
                source.append(1)
                j += 1
    return merged, source


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
    n = x.numel()
    if not 1 <= k <= n:
        raise ValueError(f"k must be in 1 .. {n}, actual {k}")
    keys = float32_sort_key(x).tolist()
    nbins = 1 << bits
    mask = nbins - 1
    candidates = list(range(n))
    remaining = k
    threshold = 0
    for shift in range(32 - bits, -1, -bits):
        counts = [0] * nbins
        for i in candidates:
            counts[(keys[i] >> shift) & mask] += 1
        digit = nbins - 1
        while digit > 0 and counts[digit] < remaining:
            remaining -= counts[digit]
            digit -= 1
        threshold |= digit << shift
        candidates = [i for i in candidates if (keys[i] >> shift) & mask == digit]
    # `candidates` now holds every element whose key equals the threshold, in index order.
    # keeping_all_ties would return all of them, and more than k elements in total.
    chosen = [i for i in range(n) if keys[i] > threshold] + candidates[:remaining]
    chosen.sort(key=lambda i: (-keys[i], i))
    idx = torch.tensor(chosen, dtype=torch.int64)
    return x[idx], idx


# ---------------------------------------------------------------------------
# 8. Top-p without a full sort.
# ---------------------------------------------------------------------------


def top_p_mask(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Boolean mask of the nucleus of a 1-D float32 probability vector.

    A token stays when the probability mass of the tokens ranked before it is
    less than p. Ties rank by lower index first. Do not sort: binary-search the
    float bit pattern of the threshold probability, using the sum of the
    probabilities at or above each candidate, then fix up the ties.
    """
    mass = probs.double()
    bits = probs.contiguous().view(torch.int32).to(torch.int64)  # non-negative, so monotone in value
    lo = 0
    hi = int(bits.max()) + 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if float(mass[bits >= mid].sum()) >= p:
            lo = mid
        else:
            hi = mid
    threshold = lo
    keep = bits > threshold
    taken = float(mass[keep].sum())
    for i in torch.nonzero(bits == threshold).flatten().tolist():
        if taken < p:
            keep[i] = True
            taken += float(mass[i])
    return keep


# ---------------------------------------------------------------------------
# 9. Philox and seeded sampling.
# ---------------------------------------------------------------------------

PHILOX_M0 = 0xD2511F53
PHILOX_M1 = 0xCD9E8D57
PHILOX_W0 = 0x9E3779B9
PHILOX_W1 = 0xBB67AE85


def philox4x32_10(counter, key):
    """Philox-4x32-10: four 32-bit counter words and two key words in, four words out."""
    c0, c1, c2, c3 = counter
    k0, k1 = key
    for _ in range(10):
        p0 = PHILOX_M0 * c0
        p1 = PHILOX_M1 * c2
        hi0, lo0 = p0 >> 32, p0 & MASK32
        hi1, lo1 = p1 >> 32, p1 & MASK32
        c0, c1, c2, c3 = (hi1 ^ c1 ^ k0) & MASK32, lo1, (hi0 ^ c3 ^ k1) & MASK32, lo0
        k0 = (k0 + PHILOX_W0) & MASK32
        k1 = (k1 + PHILOX_W1) & MASK32
    return (c0, c1, c2, c3)


def sample_categorical(probs: torch.Tensor, seed: int, step: int = 0, row_ids=None) -> torch.Tensor:
    """Draw one token per row of a 2-D probability tensor, reproducibly.

    Row r uses the Philox counter (row_id, 0, step, 0) and the key
    (seed low word, seed high word). The first output word becomes
    u = (word >> 8) / 2**24, and the token is the first index whose cumulative
    probability exceeds u times the row total. `row_ids` defaults to 0, 1, 2, ...
    The draw depends on (seed, step, row id), never on the batch position.
    """
    rows, vocab = probs.shape
    if row_ids is None:
        row_ids = list(range(rows))
    key = (seed & MASK32, (seed >> 32) & MASK32)
    cdf = torch.cumsum(probs.double(), dim=1)
    out = []
    for r in range(rows):
        rid = int(row_ids[r])
        word = philox4x32_10((rid & MASK32, (rid >> 32) & MASK32, step & MASK32, 0), key)[0]
        u = (word >> 8) / float(1 << 24)
        target = u * float(cdf[r, -1])
        token = int(torch.searchsorted(cdf[r], torch.tensor(target, dtype=torch.float64), right=True))
        out.append(min(token, vocab - 1))
    return torch.tensor(out, dtype=torch.int64)


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
  for (int b = threadIdx.x; b < nbins; b += blockDim.x) local[b] = 0;
  __syncthreads();
  const uint32_t mask = (uint32_t)(nbins - 1);
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    atomicAdd(&local[(keys[i] >> shift) & mask], 1);
  }
  __syncthreads();
  for (int b = threadIdx.x; b < nbins; b += blockDim.x) {
    if (local[b] != 0) atomicAdd(&hist[b], local[b]);
  }
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
