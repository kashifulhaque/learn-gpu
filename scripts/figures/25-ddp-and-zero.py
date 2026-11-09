#!/usr/bin/env python3
"""Bytes moved per step for data parallel and ZeRO-1.

The points are arithmetic on docs/FACTS.md and on the ZeRO paper's
mixed-precision accounting (K = 12 optimizer-state bytes per parameter,
fp16 gradients and parameters at 2 bytes each). Nothing here is a
measurement.

Ring all-reduce, from FACTS section 7.2: each rank sends
2 (p - 1) / p of the array. ZeRO-1 replaces that all-reduce with a
reduce-scatter of the gradients and an all-gather of the parameters.
When the two arrays have the same byte count, the ring sends the same
number of bytes. The paper's large-p volume is 2 Ψ elements either way.

Optimizer state stays 12 Ψ bytes on every data-parallel rank, and
ZeRO-1 stores 12 Ψ / p. Ψ = 7.5e9 is the paper's worked model.
"""

from __future__ import annotations

import math
from pathlib import Path

# FACTS 7.1 and 7.2. Alpha is the assumed 5 microseconds, tagged U there.
NVLINK3_GBS = 300.0
PCIE4_GBS = 31.5
ALPHA_S = 5e-6
GIB = 2**30
MIB = 2**20

# ZeRO paper, Figure 1 and section 5. Ψ parameters, K = 12 for mixed-precision Adam.
PSI = 7.5e9
FP16_BYTES = 2
K_OPT_BYTES = 12
RANKS = (2, 4, 8, 64)


def ring_factor(ranks: int) -> float:
    """Bytes a rank sends, divided by the array size, for a ring all-reduce."""
    return 2.0 * (ranks - 1) / ranks


def bytes_moved(ranks: int, array_bytes: float) -> float:
    """Bytes one rank sends in a ring all-reduce, or in ZeRO-1 when both
    arrays (gradients and parameters) have this size."""
    return ring_factor(ranks) * array_bytes


def optimizer_state_bytes(ranks: int, sharded: bool) -> float:
    """Resident optimizer-state bytes per rank. Data parallel keeps K Ψ.
    ZeRO-1 keeps K Ψ / ranks."""
    full = K_OPT_BYTES * PSI
    return full / ranks if sharded else full


def ring_seconds(ranks: int, nbytes: float, bandwidth_gbs: float, buckets: int = 1) -> float:
    """Alpha-beta time of `buckets` ring all-reduces that together move `nbytes`."""
    latency = buckets * 2 * (ranks - 1) * ALPHA_S
    bandwidth = ring_factor(ranks) * nbytes / (bandwidth_gbs * 1e9)
    return latency + bandwidth


def bucket_rows() -> list[tuple[int, float, float, float]]:
    """(buckets, latency µs, bandwidth µs, total µs) for 16 MiB over 8 ranks."""
    nbytes = 16 * MIB
    rows = []
    for buckets in (1, 8):
        latency = buckets * 2 * (8 - 1) * ALPHA_S
        bandwidth = ring_factor(8) * nbytes / (NVLINK3_GBS * 1e9)
        rows.append((buckets, latency * 1e6, bandwidth * 1e6, (latency + bandwidth) * 1e6))
    return rows


def svg() -> str:
    """Two panels. Left: bytes a rank sends. Right: optimizer-state bytes."""
    array_bytes = PSI * FP16_BYTES  # fp16 gradients, and the same for parameters
    moved = [bytes_moved(p, array_bytes) / 1e9 for p in RANKS]
    state_dp = [optimizer_state_bytes(p, False) / 1e9 for p in RANKS]
    state_zero = [optimizer_state_bytes(p, True) / 1e9 for p in RANKS]

    width, height = 760, 360
    left, right = 56, width - 24
    panel_gap = 36
    panel_w = (right - left - panel_gap) / 2
    top, bottom = 36, 300
    plot_h = bottom - top

    def xticks(x0: float) -> str:
        parts = []
        for i, ranks in enumerate(RANKS):
            x = x0 + (i + 0.5) * panel_w / len(RANKS)
            parts.append(
                f'<text x="{x:.1f}" y="{bottom + 18}" text-anchor="middle" '
                f'font-size="12" fill="#1c1917">{ranks}</text>'
            )
        return "\n".join(parts)

    def yaxis(x0: float, ymax: float, ticks: list[float], unit: str) -> str:
        parts = []
        for tick in ticks:
            y = bottom - (tick / ymax) * plot_h
            parts.append(
                f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x0 + panel_w:.1f}" y2="{y:.1f}" '
                f'stroke="#e7e5e4" stroke-width="1"/>'
            )
            label = f"{tick:g}"
            parts.append(
                f'<text x="{x0 - 8:.1f}" y="{y + 4:.1f}" text-anchor="end" '
                f'font-size="11" fill="#44403c">{label}</text>'
            )
        parts.append(
            f'<text x="{x0 + panel_w / 2:.1f}" y="{bottom + 36}" text-anchor="middle" '
            f'font-size="12" fill="#1c1917">{unit}</text>'
        )
        return "\n".join(parts)

    def bars(x0: float, values: list[float], ymax: float, color: str, shift: float, bar_w: float) -> str:
        parts = []
        slot = panel_w / len(RANKS)
        for i, value in enumerate(values):
            h = (value / ymax) * plot_h
            x = x0 + i * slot + shift
            y = bottom - h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" '
                f'fill="{color}" rx="2"/>'
            )
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 4:.1f}" text-anchor="middle" '
                f'font-size="10" fill="#1c1917">{value:.2f}</text>'
            )
        return "\n".join(parts)

    x1 = left
    x2 = left + panel_w + panel_gap
    slot = panel_w / len(RANKS)
    bar_w = slot * 0.28
    # Data parallel and ZeRO-1 send the same bytes, so the pairs match.
    moved_bars = bars(x1, moved, 32, "#1d4e89", slot * 0.18, bar_w)
    moved_bars += "\n" + bars(x1, moved, 32, "#0f766e", slot * 0.52, bar_w)
    state_bars = bars(x2, state_dp, 100, "#1d4e89", slot * 0.18, bar_w)
    state_bars += "\n" + bars(x2, state_zero, 100, "#0f766e", slot * 0.52, bar_w)

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
<title>Bytes a rank sends per step, and optimizer-state bytes, at 2, 4, 8, and 64 ranks</title>
<desc>Paired bars for data parallel and ZeRO-1. The send bars match at every rank count. The state bars fall as 1 over the rank count for ZeRO-1 and stay at 90 GB for data parallel.</desc>
<rect width="{width}" height="{height}" fill="#ffffff"/>
<text x="{x1 + panel_w / 2:.1f}" y="20" text-anchor="middle" font-size="13" fill="#1c1917">Bytes one rank sends per step (GB)</text>
<text x="{x2 + panel_w / 2:.1f}" y="20" text-anchor="middle" font-size="13" fill="#1c1917">Optimizer state resident per rank (GB)</text>
{yaxis(x1, 32, [0, 8, 16, 24, 32], "ranks")}
{yaxis(x2, 100, [0, 25, 50, 75, 100], "ranks")}
{moved_bars}
{state_bars}
{xticks(x1)}
{xticks(x2)}
<rect x="180" y="332" width="12" height="12" fill="#1d4e89"/>
<text x="196" y="343" font-size="12" fill="#1c1917">Data parallel</text>
<rect x="320" y="332" width="12" height="12" fill="#0f766e"/>
<text x="336" y="343" font-size="12" fill="#1c1917">ZeRO-1</text>
</svg>
'''


def main() -> None:
    array_gb = PSI * FP16_BYTES / 1e9
    print(f"fp16 array = {array_gb:.1f} GB (7.5e9 parameters x 2 bytes)")
    print("rank  dp_and_zero1_GB  dp_state_GB  zero1_state_GB")
    for ranks in RANKS:
        moved_gb = bytes_moved(ranks, PSI * FP16_BYTES) / 1e9
        print(
            f"{ranks:4d}  {moved_gb:16.5f}  "
            f"{optimizer_state_bytes(ranks, False) / 1e9:11.5f}  "
            f"{optimizer_state_bytes(ranks, True) / 1e9:14.5f}"
        )
    print(
        "P_os memory at N=64, GB:",
        4 * PSI / 1e9 + K_OPT_BYTES * PSI / 64 / 1e9,
    )
    print("16 MiB, 8 ranks, NVLink 3, assumed 5 us alpha:")
    for buckets, latency, bandwidth, total in bucket_rows():
        print(
            f"  buckets={buckets}: latency {latency:.3f} us, "
            f"bandwidth {bandwidth:.3f} us, total {total:.3f} us"
        )
    cap = 25 * MIB
    grad_14 = 14e9
    buckets_14 = math.ceil(grad_14 / cap)
    serialized = ring_seconds(8, grad_14, NVLINK3_GBS, buckets_14)
    one = ring_seconds(8, grad_14, NVLINK3_GBS, 1)
    print(
        f"14 GB into 25 MiB buckets: {buckets_14} buckets, "
        f"serialized {serialized * 1e3:.2f} ms, one collective {one * 1e3:.2f} ms, "
        f"latency term {(buckets_14 * 2 * 7 * ALPHA_S) * 1e3:.2f} ms"
    )
    one_gib_nv = ring_seconds(2, GIB, NVLINK3_GBS, 1)
    one_gib_pcie = ring_seconds(2, GIB, PCIE4_GBS, 1)
    print(
        f"1 GiB, 2 ranks: NVLink 3 {one_gib_nv * 1e3:.3f} ms, "
        f"PCIe Gen4 {one_gib_pcie * 1e3:.3f} ms"
    )
    n_eq = ALPHA_S * 8 * NVLINK3_GBS * 1e9
    print(f"latency equals bandwidth at p=8, NVLink 3: {n_eq / 1e6:.1f} MB")
    out = Path(__file__).resolve().parents[2] / "content" / "figures" / "25-ddp-and-zero.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
