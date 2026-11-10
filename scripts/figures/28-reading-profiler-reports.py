#!/usr/bin/env python3
"""Numbers for chapter 28's roofline and wave figures.

Run it from the repository root. It reads the recorded `ncu --csv` report in
content/labs/28-profile-reports/fixture_ncu.py with the lab's reference parser,
then prints two TypeScript literals for the figures:

1. ROOFLINE: for each kernel, the DRAM bytes, the FP32 FLOPs, the duration, and
   the numbers derived from them. FLOPs are FADD + FMUL + 2 x FFMA from the
   instruction counters. The A100 80GB PCIe peaks come from docs/FACTS.md:
   19.5 TFLOP/s FP32 and 1935 GB/s HBM2e.
2. WAVES: the wave efficiency, waves / ceil(waves), for a grid swept in steps
   of 54 blocks at 1, 2, 4, and 8 resident blocks per SM on 108 SMs.

The kernels in the report are synthesized. No GPU produced these numbers.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "content" / "labs" / "28-profile-reports"
sys.path.insert(0, str(ROOT / "content" / "labs"))
sys.path.insert(0, str(LAB))

import fixture_ncu  # noqa: E402
import solution  # noqa: E402

SM_COUNT = 108


def roofline_rows() -> list[dict]:
    report = solution.parse_ncu_csv(fixture_ncu.NCU_REPORT)
    rows = []
    for name in fixture_ncu.KERNELS:
        m = report[name]
        r = solution.roofline_from_counters(m)
        rows.append(
            {
                "name": name,
                "label": solution.diagnose(m),
                "bytes": int(r["bytes"]),
                "flops": int(r["flops"]),
                "ms": round(r["seconds"] * 1e3, 4),
                "ai": None if r["bytes"] == 0 else round(r["ai"], 4),
                "gflops": round(r["achieved_gflops"], 1),
                "gbs": round(r["achieved_gbs"], 1),
                "pct": round(r["pct_of_ceiling"], 1),
            }
        )
    return rows


def wave_rows() -> dict[int, list[float]]:
    out = {}
    for per_sm in (1, 2, 4, 8):
        series = []
        for grid in range(54, 54 * 65, 54):
            waves = grid / (SM_COUNT * per_sm)
            series.append(round(waves / math.ceil(waves), 4))
        out[per_sm] = series
    return out


def main() -> None:
    print("// ROOFLINE: peak 19.5 TFLOP/s FP32, 1935 GB/s, ridge", round(19.5e12 / 1935e9, 2), "FLOP/B")
    print("export const ROOFLINE = [")
    for row in roofline_rows():
        fields = ", ".join(f"{k}: {v!r}".replace("None", "null").replace("'", '"') for k, v in row.items())
        print(f"  {{ {fields} }},")
    print("];")
    print("// WAVES: efficiency at grid = 54, 108, ... 3456")
    print("export const WAVES: Record<number, number[]> = {")
    for per_sm, series in wave_rows().items():
        print(f"  {per_sm}: {series},")
    print("};")


if __name__ == "__main__":
    main()
