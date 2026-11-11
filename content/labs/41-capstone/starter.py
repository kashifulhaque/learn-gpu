"""Lab 41: the capstone. Optimize three kernels under a fixed grader.

Part 1 is the grader's arithmetic, in plain Python. Write six small functions:
`traffic`, `speed_of_light`, `robust_time`, `score`, `geomean`, and
`topk_matches`. They run on any machine.

Part 2 is the three kernels. The starter versions at the bottom are correct and
slow: eager PyTorch, one kernel per step. Replace each with a Triton or CUDA
kernel (CUDA through `load_cuda` from lab_common) and keep it correct. On a GPU
the tests check each kernel against a float64 reference, then time it with the
L2 flushed and report a score between 0 and 1.

The three tasks and their shapes:

    "grayscale_blur"  shape (H, W)        float32 RGB (H, W, 3) -> float32 (H, W)
    "topk"            shape (rows, n, k)  float32 (rows, n) -> values (rows, k), indices (rows, k) int64
    "mlp"             shape (M, d, h)     bf16 y = gelu(x w1 + b1) w2 + b2, fp32 accumulation
"""

from __future__ import annotations

from lab_common import A100

# ---------------------------------------------------------------------------
# Part 1: the grader, on the CPU
# ---------------------------------------------------------------------------


def traffic(task: str, shape: tuple) -> tuple[int, int]:
    """Return (flops, bytes) of the ideal fused kernel for one task.

    Count only the unavoidable traffic: every input read once, every output
    written once, and nothing for intermediates. Conventions:

    grayscale_blur (H, W): the image is float32 (H, W, 3) and the output float32
        (H, W). The luma is 3 multiplies and 2 adds. The 3 by 3 blur is 9
        multiplies and 8 adds. That is 22 FLOP per pixel. Ignore the borders.
    topk (rows, n, k): the logits are float32. The output is k float32 values
        and k int64 indices per row. Comparisons are not FLOPs, so flops is 0.
    mlp (M, d, h): x and y are (M, d), w1 is (d, h), w2 is (h, d), b1 has h
        entries, b2 has d, all bf16 (2 bytes). The hidden layer never reaches
        memory. FLOPs are the two GEMMs only: 2 * M * d * h each. Ignore bias
        and GELU.

    Raise ValueError for an unknown task.
    """
    raise NotImplementedError("write traffic")


def speed_of_light(task: str, shape: tuple) -> float:
    """Return the roofline floor in milliseconds: max(compute time, memory time).

    Memory time uses the measured copy ceiling, A100.COPY_CEILING_GBS. Compute
    time uses A100.TENSOR_BF16_TFLOPS for "mlp" and A100.FP32_TFLOPS for the
    other two tasks. Use `traffic` for the counts.
    """
    raise NotImplementedError("write speed_of_light")


def robust_time(times_ms: list[float], warmup: int = 2) -> float:
    """Drop the first `warmup` recorded times and return the median of the rest.

    Raise ValueError when no time is left after the drop.
    """
    raise NotImplementedError("write robust_time")


def score(time_ms: float, floor_ms: float, ref_ms: float) -> float:
    """Leaderboard score in [0, 1]: how much of the headroom you closed, on a log scale.

        score = log(ref / time) / log(ref / floor), clipped to [0, 1]

    A kernel as slow as the reference scores 0, and one at the floor scores 1.
    Raise ValueError when time_ms or floor_ms is not positive, or when ref_ms
    is not greater than floor_ms (there is no headroom to score).
    """
    raise NotImplementedError("write score")


def geomean(values: list[float]) -> float:
    """Geometric mean, the way a leaderboard combines scores over shapes.

    A zero anywhere gives 0.0. Raise ValueError for an empty list.
    """
    raise NotImplementedError("write geomean")


def topk_matches(x, indices, k: int) -> bool:
    """True when `indices` (rows, k) is the right top-k set in every row of `x` (rows, n).

    "Right" is the chapter 30 tie rule: among tied values, the lowest index
    wins, which is the first k of a stable descending sort. Compare each row as
    a set, because the kernel may return its k indices in any order.
    """
    raise NotImplementedError("write topk_matches")


# ---------------------------------------------------------------------------
# Part 2: the kernels. These baselines are correct and slow. Replace them.
# ---------------------------------------------------------------------------


def grayscale_blur(x):
    """x: float32 (H, W, 3) on the GPU. Return float32 (H, W).

    Luma 0.299 R + 0.587 G + 0.114 B, then the blur [1 2 1] x [1 2 1] / 16 with
    zeros outside the image. Two kernels and an intermediate image: fuse them.
    """
    import torch
    import torch.nn.functional as F

    g = x @ torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype, device=x.device)
    k1 = torch.tensor([1.0, 2.0, 1.0], dtype=x.dtype, device=x.device)
    kernel = (k1[:, None] * k1[None, :] / 16.0)[None, None]
    return F.conv2d(g[None, None], kernel, padding=1)[0, 0]


def topk_rows(x, k):
    """x: float32 (rows, n) on the GPU. Return (values, indices) of the k largest per row.

    Values are sorted descending. Indices are int64. Ties go to the lower index.
    The baseline sorts every row in full: you only need k of n.
    """
    import torch

    values, indices = torch.sort(x, dim=-1, descending=True, stable=True)
    return values[:, :k], indices[:, :k]


def mlp_block(x, w1, b1, w2, b2):
    """bf16 x (M, d), w1 (d, h), b1 (h), w2 (h, d), b2 (d). Return bf16 (M, d).

    y = gelu(x w1 + b1) w2 + b2, with the exact (erf) GELU. The baseline runs
    at least three kernels and writes the hidden layer to memory twice.
    """
    import torch
    import torch.nn.functional as F

    h = F.gelu(torch.addmm(b1, x, w1))
    return torch.addmm(b2, h, w2)
