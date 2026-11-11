#!/usr/bin/env python3
"""Numbers for chapter 31's figures and worked examples.

Run it with torch installed (CPU is enough):

    python3 scripts/figures/31-convolution-and-stencils.py

Five sections print. The first two are pasted into the two figures.

1. Halo overhead for the output-tile strategy and the active-thread fraction
   for the input-tile strategy, for 2D and 3D, as filter size and tile size
   vary. These are exact ratios.
2. The im2col blow-up: elements of the lowered matrix per element of the
   input, for stride 1 and 2 and filter sides 1 to 9 on a 56 by 56 input with
   "same" padding. Each entry is checked against the shape of F.unfold.
3. Arithmetic intensity of direct convolution, from the formulas in the
   chapter: one channel, then a 64-channel layer, against the A100 ridge
   points in docs/FACTS.md.
4. The 7-point stencil: bytes and FLOP per cell, and the byte count at the
   copy ceiling for a 256 cubed grid, and the global reads per output of the
   coarsened kernel.
5. Fusing conv, ReLU, and a 2 by 2 maxpool: bytes of the intermediate
   tensors that a fused kernel does not write.
"""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F

KS = [1, 3, 5, 7, 9, 11, 13, 15]
OUT_TILES = [4, 8, 16, 32]
BLOCKS = [8, 16, 32]


def halo_overhead(out_tile: int, k: int, dims: int) -> float:
    return ((out_tile + k - 1) / out_tile) ** dims


def active_fraction(block: int, k: int, dims: int) -> float:
    return (max(block - k + 1, 0) / block) ** dims


def section1() -> dict:
    out = {"ks": KS, "outTiles": OUT_TILES, "blocks": BLOCKS, "loads": {}, "active": {}}
    for dims in (2, 3):
        out["loads"][str(dims)] = {
            str(t): [round(halo_overhead(t, k, dims), 4) for k in KS] for t in OUT_TILES
        }
        out["active"][str(dims)] = {
            str(b): [round(active_fraction(b, k, dims), 4) for k in KS] for b in BLOCKS
        }
    return out


def section2() -> dict:
    H = W = 56
    out = {"ks": [1, 3, 5, 7, 9], "strides": [1, 2], "blowup": {}}
    x = torch.zeros(1, 1, H, W)
    for stride in (1, 2):
        row = []
        for k in out["ks"]:
            pad = k // 2
            P = (H + 2 * pad - k) // stride + 1
            unfolded = F.unfold(x, (k, k), padding=pad, stride=stride)
            assert unfolded.shape == (1, k * k, P * P)
            row.append(round(unfolded.numel() / x.numel(), 4))
        out["blowup"][str(stride)] = row
    return out


def section3() -> dict:
    ridge_fp32 = 19.5e12 / 1935e9
    ridge_copy = 19.5e12 / 1275e9
    rows = {}
    for k in (3, 5, 7, 9):
        flop = 2 * k * k
        rows[f"1 channel, {k}x{k}, ideal bytes"] = flop / 8
        rows[f"1 channel, {k}x{k}, no reuse bytes"] = flop / ((k * k + 1) * 4)
    # A ResNet-style layer: N=32, C=K=64, 56 by 56, 3x3, same padding, fp32.
    N, C, K, H, R = 32, 64, 64, 56, 3
    flop = 2 * N * H * H * K * C * R * R
    nbytes = 4 * (N * C * H * H + K * C * R * R + N * K * H * H)
    im2col_bytes = 4 * N * C * R * R * H * H
    rows["layer FLOP"] = flop
    rows["layer ideal bytes"] = nbytes
    rows["layer intensity"] = flop / nbytes
    rows["layer im2col bytes"] = im2col_bytes
    rows["layer input bytes"] = 4 * N * C * H * H
    rows["layer time floor at 19.5 TFLOP/s fp32, ms"] = flop / 19.5e12 * 1e3
    rows["layer im2col write+read at copy ceiling, ms"] = 2 * im2col_bytes / 1275e9 * 1e3
    return {"ridge_fp32_hbm": ridge_fp32, "ridge_fp32_copy": ridge_copy, "rows": rows}


def section4() -> dict:
    X = Y = Z = 256
    cells = X * Y * Z
    nbytes = 2 * cells * 4
    return {
        "flop_per_cell": 1 + 5 + 1 + 1,  # c0*centre, five adds for six neighbours, c1*sum, one add
        "ideal_bytes_per_cell": 8,
        "naive_global_reads_per_cell": 7,
        "grid_bytes_read_plus_write": nbytes,
        "floor_ms_at_copy_ceiling": nbytes / 1275e9 * 1e3,
        "floor_ms_at_rated_bandwidth": nbytes / 1935e9 * 1e3,
        "coarse_block_active_fraction": (30 * 6) / (32 * 8),
        "coarse_block_reads_per_output": (32 * 8) / (30 * 6),
        # Planes loaded per block of 16 outputs: z0 - 1, z0, then 16 more.
        "coarse_plane_factor": (16 + 2) / 16,
        "coarse_global_reads_per_output": (32 * 8) / (30 * 6) * (16 + 2) / 16,
        # A 7-point stencil needs the six faces of the halo, not the corners.
        "faces_only_loads_per_output": {str(T): 1 + 6 / T for T in (8, 16, 32)},
        "box_loads_per_output": {str(T): ((T + 2) / T) ** 3 for T in (8, 16, 32)},
    }


def section5() -> dict:
    """Fusing conv + ReLU + 2 by 2 maxpool, N=32, K=64, 56 by 56 output, fp32."""
    N, K, P = 32, 64, 56
    conv_out = 4 * N * K * P * P
    pooled = 4 * N * K * (P // 2) ** 2
    unfused = conv_out + 2 * conv_out + conv_out + pooled  # write, relu r+w, pool read + write
    fused = pooled
    return {
        "conv_out_bytes": conv_out,
        "pooled_bytes": pooled,
        "unfused_extra_bytes": unfused,
        "fused_extra_bytes": fused,
        "unfused_ms_at_copy_ceiling": unfused / 1275e9 * 1e3,
        "fused_ms_at_copy_ceiling": fused / 1275e9 * 1e3,
    }


def main() -> None:
    print("== 1. halo (paste into viz/31/halo-overhead.tsx)")
    print(json.dumps(section1()))
    print("== 2. im2col blow-up (paste into viz/31/im2col-blowup.tsx)")
    print(json.dumps(section2()))
    print("== 3. arithmetic intensity")
    s3 = section3()
    print(json.dumps(s3, indent=1))
    print("== 4. stencil")
    print(json.dumps(section4(), indent=1))
    print("== 5. fusion")
    print(json.dumps(section5(), indent=1))


if __name__ == "__main__":
    main()
