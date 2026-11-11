#!/usr/bin/env python3
"""Numbers for chapter 35's two figures: the TMEM map and the pair traffic.

The TMEM map: 128 lanes by 512 columns of 32-bit cells, drawn as 4 lane
groups (one per warp of a warpgroup) by 16 blocks of 32 columns. For each
accumulator width N it prints how many accumulators fit when every one is
its own tcgen05.alloc (rounded to a power of two, at least 32) and when a
kernel carves one 512-column allocation.

The pair traffic: per-CTA operand bytes for one K = 64 bf16 slice of a
tile whose CTAs each own 128 accumulator rows. With cta_group 1 a CTA
stages its own 128 rows of A and all N rows of B. With cta_group 2 it
stages its own 128 rows of A and N / 2 rows of B.

Run `python3 scripts/figures/35-blackwell-tcgen05.py` and paste the two
tables into frontend/src/components/viz/35/. The functions are the lab's
solution, so the figure and the lab agree.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "content" / "labs" / "35-blackwell-tmem"))

import solution as lab  # noqa: E402

WIDTHS = (64, 96, 128, 160, 192, 256)
K = 64
BF16 = 2


def main() -> None:
    print("// width, allocated columns, stages separate, stages carved, columns holding no accumulator (separate), same (carved)")
    for n in WIDTHS:
        _, cols, alloc = lab.accumulator_footprint(128, n, 1)
        sep = lab.accumulator_stages(n, carve=False)
        carved = lab.accumulator_stages(n, carve=True)
        idle_sep = 512 - sep * n
        idle_carved = 512 - carved * n
        print(f"  [{n}, {alloc}, {sep}, {carved}, {idle_sep}, {idle_carved}],")
    print()
    print("// N, group 1 A bytes, group 1 B bytes, group 2 A bytes, group 2 B bytes, FLOP/byte g1, FLOP/byte g2")
    for n in (64, 128, 192, 256):
        a1, b1 = lab.per_cta_operand_bytes(128, n, K, 1, BF16)
        a2, b2 = lab.per_cta_operand_bytes(128, n, K, 2, BF16)
        i1 = lab.operand_intensity(128, n, 1, BF16)
        i2 = lab.operand_intensity(128, n, 2, BF16)
        print(f"  [{n}, {a1}, {b1}, {a2}, {b2}, {i1:.1f}, {i2:.1f}],")
    print()
    a1, b1 = lab.per_cta_operand_bytes(128, 256, K, 1, BF16)
    a2, b2 = lab.per_cta_operand_bytes(128, 256, K, 2, BF16)
    print(f"128 x 256 x 64 bf16 stage, one CTA: A {a1} + B {b1} = {a1 + b1} bytes = {(a1 + b1) / 1024:.0f} KiB")
    print(f"256 x 256 x 64 bf16 stage, per CTA of a pair: A {a2} + B {b2} = {a2 + b2} bytes = {(a2 + b2) / 1024:.0f} KiB")
    print(f"fraction of one-CTA bytes: {(a2 + b2) / (a1 + b1):.4f}; B alone: {b2 / b1:.2f}")
    smem = 227 * 1024
    print(f"stages in 227 KiB: one CTA {smem // (a1 + b1)} ({smem // (a1 + b1) * (a1 + b1) // 1024} KiB), pair {smem // (a2 + b2)} ({smem // (a2 + b2) * (a2 + b2) // 1024} KiB)")
    print(f"FLOP per byte of staged operand: {lab.operand_intensity(128, 256, 1, BF16):.1f} -> {lab.operand_intensity(128, 256, 2, BF16):.1f}")
    # Accumulator register pressure on Hopper: 128 x 256 fp32 over 128 threads.
    acc_bytes = 128 * 256 * 4
    print(f"128 x 256 fp32 accumulator: {acc_bytes} bytes = {acc_bytes // 1024} KiB")
    print(f"Hopper m64n256 fp32, one warpgroup of 128 threads: {64 * 256 // 128} registers per thread; two consumer warpgroups hold {acc_bytes // 1024} KiB")
    print(f"256 KiB register file = 65536 registers; accumulator share = {acc_bytes / (256 * 1024):.2f}")
    # TMEM and MMA time
    print(f"TMEM: {128 * 512 * 4} bytes = {128 * 512 * 4 // 1024} KiB; 32 columns = {128 * 32 * 4 // 1024} KiB")
    flop = 2 * 256 * 256 * 16
    print(f"256x256x16 pair FLOP {flop}; per SM per clock 8192 -> pair 16384 -> {flop // 16384} clocks")
    flop1 = 2 * 128 * 256 * 16
    print(f"128x256x16 one CTA FLOP {flop1}; at 8192 per clock -> {flop1 // 8192} clocks")


if __name__ == "__main__":
    main()
