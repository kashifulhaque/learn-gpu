#!/usr/bin/env python3
"""Numbers for chapter 32's two figures: work per warp, and BFS cost per level.

Everything here is computed on the seeded inputs of lab 32 (fixture_irregular.py,
seed 32) with the lab's reference solution. Nothing is a measurement on a GPU.

Figure 1, 32-warp-work: the 4096 by 4096 power-law matrix. Rows are 4096, so
there are 128 warps of 32 rows. For each scheme the script prints the work of
every warp, in lockstep iterations, and the max/mean ratio.

Figure 2, 32-bfs-directions: the 1520-vertex preferential-attachment graph.
For each BFS level, the script prints the frontier size, the edges a top-down
round would examine, and the edges a bottom-up round would examine from the same
state, so the crossover that direction optimization uses is visible.

Run it and paste the printed arrays into the two components under
frontend/src/components/viz/32/.
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[2] / "content" / "labs" / "32-spmv-bfs"
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent))

import fixture_irregular as fx  # noqa: E402
import solution as lab  # noqa: E402


def warp_work_figure() -> None:
    n = 4096
    row_ptr, _, _ = fx.power_law_csr(n, n)
    lengths = [row_ptr[i + 1] - row_ptr[i] for i in range(n)]
    warps = n // 32
    print(f"rows {n}, nnz {row_ptr[-1]}, mean row {row_ptr[-1] / n:.3f}, longest row {max(lengths)}, warps {warps}")
    for scheme in ("thread_per_row", "warp_per_row", "merge_path"):
        work = lab.warp_work(row_ptr, scheme, warps)
        print(f"{scheme}: imbalance {lab.imbalance(work):.4f}, max {max(work)}, mean {sum(work) / len(work):.2f}")
        print(f"  {work}")
    # Lane use of warp-per-row: useful lane-steps over the 32 lanes x iterations it runs.
    iterations = sum(lab.warp_work(row_ptr, "warp_per_row", warps))
    print(f"warp-per-row lane use {row_ptr[-1] / (32 * iterations):.3f}")
    tpr = sum(lab.warp_work(row_ptr, "thread_per_row", warps))
    print(f"thread-per-row lane use {row_ptr[-1] / (32 * tpr):.3f}")
    # Byte costs of the formats on the same matrix.
    print(
        "bytes: coo",
        lab.coo_bytes(row_ptr[-1]),
        "csr",
        lab.csr_bytes(n, row_ptr[-1]),
        "ell",
        lab.ell_bytes(n, max(lengths)),
        "hyb",
        lab.hyb_bytes(lengths),
        "hyb width",
        lab.hyb_width(lengths),
    )


def bfs_figure() -> None:
    row_ptr, col_idx = fx.power_law_graph()
    n = len(row_ptr) - 1
    degree = [row_ptr[v + 1] - row_ptr[v] for v in range(n)]
    level = lab.bfs_levels_frontier(row_ptr, col_idx, 0)
    deepest = max(level)
    print(f"vertices {n}, directed edges {row_ptr[-1]}, deepest level {deepest}, unreached {level.count(-1)}")
    total_degree = sum(degree)
    print("depth, frontier size, top-down edges (m_f), bottom-up edges, unexplored edges (m_u) at the start of the round")
    rows = []
    for depth in range(deepest + 1):
        frontier = [v for v in range(n) if level[v] == depth]
        state = [lv if 0 <= lv <= depth else -1 for lv in level]
        # m_u as bfs_direction_optimizing tracks it: every edge of a vertex that is not yet in a level <= depth.
        unexplored = total_degree - sum(degree[v] for v in range(n) if state[v] != -1)
        top_down = sum(degree[v] for v in frontier)
        _, bottom_up = lab.bottom_up_step(row_ptr, col_idx, state, depth)
        rows.append((depth, len(frontier), top_down, bottom_up, unexplored))
        print(depth, len(frontier), top_down, bottom_up, unexplored)
    _, directions, examined = lab.bfs_direction_optimizing(row_ptr, col_idx, 0)
    reached_edges = sum(degree[v] for v in range(n) if level[v] != -1)
    print(f"direction-optimizing: {directions}, {examined} edges examined against {reached_edges} top-down")
    print("alpha sweep (beta 18): alpha, edges examined, directions")
    for alpha in (1, 2, 4, 8, 15, 30):
        _, dirs, edges = lab.bfs_direction_optimizing(row_ptr, col_idx, 0, alpha=alpha)
        print(alpha, edges, "".join(d[0] for d in dirs))


if __name__ == "__main__":
    warp_work_figure()
    print()
    bfs_figure()
