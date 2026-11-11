"""Lab 41 solution: the grader's arithmetic, and three Triton kernels."""

from __future__ import annotations

import math
import statistics

from lab_common import A100

try:  # Triton exists only on the GPU worker. The CPU checks don't need it.
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

# ---------------------------------------------------------------------------
# Part 1: the grader, on the CPU
# ---------------------------------------------------------------------------


def traffic(task: str, shape: tuple) -> tuple[int, int]:
    if task == "grayscale_blur":
        h, w = shape
        return 22 * h * w, (12 + 4) * h * w
    if task == "topk":
        rows, n, k = shape
        return 0, 4 * rows * n + (4 + 8) * rows * k
    if task == "mlp":
        m, d, h = shape
        elements = 2 * m * d + 2 * d * h + h + d
        return 4 * m * d * h, 2 * elements
    raise ValueError(f"unknown task {task!r}")


def speed_of_light(task: str, shape: tuple) -> float:
    flops, nbytes = traffic(task, shape)
    peak = A100.TENSOR_BF16_TFLOPS if task == "mlp" else A100.FP32_TFLOPS
    compute_s = flops / (peak * 1e12)
    memory_s = nbytes / (A100.COPY_CEILING_GBS * 1e9)
    return max(compute_s, memory_s) * 1e3


def robust_time(times_ms: list[float], warmup: int = 2) -> float:
    kept = list(times_ms)[warmup:]
    if not kept:
        raise ValueError(f"no times left after dropping {warmup} warmup iterations")
    return float(statistics.median(kept))


def score(time_ms: float, floor_ms: float, ref_ms: float) -> float:
    if time_ms <= 0 or floor_ms <= 0:
        raise ValueError("times must be positive")
    if ref_ms <= floor_ms:
        raise ValueError("the reference must be slower than the floor")
    value = math.log(ref_ms / time_ms) / math.log(ref_ms / floor_ms)
    return min(1.0, max(0.0, value))


def geomean(values: list[float]) -> float:
    if not values:
        raise ValueError("geomean of an empty list")
    if any(v == 0 for v in values):
        return 0.0
    return math.exp(sum(math.log(v) for v in values) / len(values))


def topk_matches(x, indices, k: int) -> bool:
    import torch

    _, order = torch.sort(x, dim=-1, descending=True, stable=True)
    want = torch.sort(order[:, :k], dim=-1).values
    got = torch.sort(torch.as_tensor(indices).long(), dim=-1).values
    return tuple(got.shape) == tuple(want.shape) and bool(torch.equal(got, want))


# ---------------------------------------------------------------------------
# Part 2: the kernels
# ---------------------------------------------------------------------------

if HAVE_TRITON:

    @triton.jit
    def _gray_blur_kernel(x_ptr, w_ptr, out_ptr, H, W, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for dr in tl.static_range(-1, 2):
            for dc in tl.static_range(-1, 2):
                r = row + dr
                c = cols + dc
                inside = (r >= 0) & (r < H) & (c >= 0) & (c < W)
                base = (r * W + c) * 3
                red = tl.load(x_ptr + base, mask=inside, other=0.0)
                green = tl.load(x_ptr + base + 1, mask=inside, other=0.0)
                blue = tl.load(x_ptr + base + 2, mask=inside, other=0.0)
                luma = 0.299 * red + 0.587 * green + 0.114 * blue
                weight = tl.load(w_ptr + (dr + 1) * 3 + (dc + 1))
                acc += weight * luma
        tl.store(out_ptr + row * W + cols, acc, mask=cols < W)

    @triton.jit
    def _topk_kernel(x_ptr, val_ptr, idx_ptr, n, K: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK)
        x = tl.load(x_ptr + row * n + offs, mask=offs < n, other=float("-inf"))
        for j in tl.static_range(K):
            best = tl.max(x, axis=0)
            # The lowest index among the tied maxima is the tie rule.
            pick = tl.min(tl.where(x == best, offs, BLOCK), axis=0)
            tl.store(val_ptr + row * K + j, best)
            tl.store(idx_ptr + row * K + j, pick.to(tl.int64))
            x = tl.where(offs == pick, float("-inf"), x)

    @triton.jit
    def _linear_kernel(
        a_ptr, w_ptr, bias_ptr, out_ptr, M, N, K,
        GELU: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        offs_m = tl.program_id(0) * BM + tl.arange(0, BM)
        offs_n = tl.program_id(1) * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
        w_ptrs = w_ptr + offs_k[:, None] * N + offs_n[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BK)):
            left = K - k * BK
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < left), other=0.0)
            w = tl.load(w_ptrs, mask=(offs_k[:, None] < left) & (offs_n[None, :] < N), other=0.0)
            acc = tl.dot(a, w, acc)
            a_ptrs += BK
            w_ptrs += BK * N
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]
        if GELU:
            acc = 0.5 * acc * (1.0 + tl.erf(acc * 0.7071067811865476))
        out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def grayscale_blur(x):
    import torch

    x = x.contiguous()
    h, w, _ = x.shape
    weights = torch.tensor([1, 2, 1, 2, 4, 2, 1, 2, 1], dtype=torch.float32, device=x.device) / 16.0
    out = torch.empty(h, w, dtype=torch.float32, device=x.device)
    block = 256
    _gray_blur_kernel[(h, triton.cdiv(w, block))](x, weights, out, h, w, BLOCK=block)
    return out


def topk_rows(x, k):
    import torch

    rows, n = x.shape
    if n > 32768 or k > n:  # a row that does not fit one block: fall back to the stable sort
        values, indices = torch.sort(x, dim=-1, descending=True, stable=True)
        return values[:, :k], indices[:, :k]
    x = x.contiguous()
    values = torch.empty(rows, k, dtype=x.dtype, device=x.device)
    indices = torch.empty(rows, k, dtype=torch.int64, device=x.device)
    block = max(16, triton.next_power_of_2(n))
    warps = 32 if block >= 16384 else 16 if block >= 4096 else 8 if block >= 1024 else 4
    _topk_kernel[(rows,)](x, values, indices, n, K=k, BLOCK=block, num_warps=warps)
    return values, indices


def _linear(a, w, bias, gelu):
    import torch

    m, k = a.shape
    n = w.shape[1]
    out = torch.empty(m, n, dtype=torch.bfloat16, device=a.device)
    grid = (triton.cdiv(m, 128), triton.cdiv(n, 128))
    _linear_kernel[grid](a, w, bias, out, m, n, k, GELU=gelu, BM=128, BN=128, BK=32, num_warps=8, num_stages=3)
    return out


def mlp_block(x, w1, b1, w2, b2):
    h = _linear(x.contiguous(), w1.contiguous(), b1.contiguous(), True)
    return _linear(h, w2.contiguous(), b2.contiguous(), False)
