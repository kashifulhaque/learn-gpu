#!/usr/bin/env python3
"""Numbers for chapter 41's score figure and the three starting points.

The floors use the A100 constants in docs/FACTS.md: the measured 1275 GB/s copy
ceiling, 19.5 TFLOP/s fp32, and 312 TFLOP/s bf16. A floor is max(compute time,
memory time) for the ideal fused kernel. These are predictions, not
measurements. The score is log(ref / time) / log(ref / floor), clipped to [0, 1].
The figure lets the reader pick the reference time as a multiple of the floor,
because the lab, not this script, measures the real reference.
"""

from __future__ import annotations

import math

COPY_GBS = 1275.0
HBM_GBS = 1935.0
FP32_TFLOPS = 19.5
BF16_TFLOPS = 312.0


def traffic(task: str, shape: tuple) -> tuple[int, int]:
    if task == "grayscale_blur":
        h, w = shape
        return 22 * h * w, 16 * h * w
    if task == "topk":
        rows, n, k = shape
        return 0, 4 * rows * n + 12 * rows * k
    m, d, h = shape
    return 4 * m * d * h, 2 * (2 * m * d + 2 * d * h + h + d)


def floor_ms(task: str, shape: tuple, bw_gbs: float = COPY_GBS) -> float:
    flops, nbytes = traffic(task, shape)
    peak = BF16_TFLOPS if task == "mlp" else FP32_TFLOPS
    return max(flops / (peak * 1e12), nbytes / (bw_gbs * 1e9)) * 1e3


def score(t: float, floor: float, ref: float) -> float:
    return min(1.0, max(0.0, math.log(ref / t) / math.log(ref / floor)))


def linear_score(t: float, floor: float, ref: float) -> float:
    return min(1.0, max(0.0, (ref - t) / (ref - floor)))


TASKS = (
    ("grayscale_blur", (4096, 4096)),
    ("topk", (4096, 32000, 8)),
    ("mlp", (4096, 1024, 4096)),
)

if __name__ == "__main__":
    for task, shape in TASKS:
        flops, nbytes = traffic(task, shape)
        print(f"{task} {shape}: flops={flops:,} bytes={nbytes:,} "
              f"floor={floor_ms(task, shape):.4f} ms (rated-bandwidth floor {floor_ms(task, shape, HBM_GBS):.4f} ms)")
    # Unfused traffic for context.
    h, w = 4096, 4096
    print(f"unfused grayscale_blur bytes={24 * h * w:,}, floor {24 * h * w / (COPY_GBS * 1e9) * 1e3:.4f} ms")
    m, d, hh = 4096, 1024, 4096
    print(f"MLP hidden layer write plus read bytes={2 * 2 * m * hh:,}, {2 * 2 * m * hh / (COPY_GBS * 1e9) * 1e3:.4f} ms")
    print(f"MLP intensity {4 * m * d * hh / traffic('mlp', (m, d, hh))[1]:.1f} FLOP/byte, ridge {BF16_TFLOPS * 1e12 / (COPY_GBS * 1e9):.1f}")
    # The figure's initial state: reference 4x the floor, time 2x the floor.
    print("score at 2x floor, ref 4x floor: log", score(2, 1, 4), "linear", round(linear_score(2, 1, 4), 4))
    for t in (1, 1.25, 1.5, 2, 3, 4):
        print(f"time {t}x floor: log {score(t, 1, 4):.3f} linear {linear_score(t, 1, 4):.3f}")
