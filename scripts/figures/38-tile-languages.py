#!/usr/bin/env python3
"""Numbers for chapter 38's two figures.

Both figures are computed with the functions in the lab's worked solution
(content/labs/38-tile-languages/solution.py), so the figure and the lab agree.

1. The pipeline schedule. For 8 tiles and 2 to 5 stages, the (load, mma) pair of
   each step, the number of steps (tiles + stages - 1), the number of steps in
   which a load overlaps an MMA, and the most tiles resident in shared memory.
   A serial loop (load, wait, multiply) takes 2 * tiles steps of one tile each.

2. The blocked layout. Four layouts of a 16 by 16 tile. For each element, the
   (warp, lane, register) that holds it, under the lab's convention (see
   blocked_owner). The registers per thread are 256 / (32 * warps).

Run it to print the TypeScript constant pasted into
frontend/src/components/viz/38/data.ts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[2] / "content" / "labs" / "38-tile-languages"
sys.path.insert(0, str(LAB))

import solution  # noqa: E402

TILES = 8
STAGES = [2, 3, 4, 5]

# (name, size_per_thread, threads_per_warp, warps_per_cta, order)
LAYOUTS = [
    ("1 warp, 4 contiguous columns per thread", (1, 4), (8, 4), (1, 1), (1, 0)),
    ("4 warps, 2 contiguous columns per thread", (1, 2), (8, 4), (2, 2), (1, 0)),
    ("2 warps, column-major order", (2, 1), (8, 4), (1, 2), (0, 1)),
    ("1 warp, one element per thread, column-major", (1, 1), (4, 8), (1, 1), (0, 1)),
]


def pipeline_rows():
    rows = []
    for s in STAGES:
        sched = solution.pipeline_schedule(TILES, s)
        rows.append(
            {
                "stages": s,
                "steps": len(sched),
                "serial_steps": 2 * TILES,
                "overlap_steps": sum(1 for load, mma in sched if load is not None and mma is not None),
                "resident": solution.max_resident(sched),
                "schedule": [[-1 if v is None else v for v in pair] for pair in sched],
            }
        )
    return rows


def layout_rows():
    out = []
    for name, spt, tpw, wpc, order in LAYOUTS:
        grid = [
            [list(solution.blocked_owner(r, c, (16, 16), spt, tpw, wpc, order)) for c in range(16)]
            for r in range(16)
        ]
        warps = wpc[0] * wpc[1]
        out.append(
            {
                "name": name,
                "spt": list(spt),
                "tpw": list(tpw),
                "wpc": list(wpc),
                "order": list(order),
                "warps": warps,
                "regs_per_thread": 256 // (32 * warps),
                "owner": grid,
            }
        )
    return out


if __name__ == "__main__":
    data = {"tiles": TILES, "pipeline": pipeline_rows(), "layouts": layout_rows()}
    for row in data["pipeline"]:
        print(
            f"stages {row['stages']}: steps {row['steps']} (serial {row['serial_steps']}), "
            f"overlap steps {row['overlap_steps']}, resident {row['resident']}",
            file=sys.stderr,
        )
    for lay in data["layouts"]:
        print(f"{lay['name']}: {lay['warps']} warps, {lay['regs_per_thread']} registers per thread", file=sys.stderr)
    print("export const DATA = " + json.dumps(data, separators=(",", ":")) + ";")
