"""Seeded inputs for lab 32: a power-law sparse matrix and a power-law graph.

Both generators use random.Random, so the same seed gives the same input on
every machine and every Python version. scripts/figures/32-sparse-and-irregular-work.py
imports this file, so the chapter's figure and the lab's checks use one matrix.
"""

from __future__ import annotations

import random

SEED = 32


def power_law_row_lengths(n_rows: int, n_cols: int, seed: int = SEED, cap: int | None = None) -> list[int]:
    """Row lengths from a Pareto tail. About 5% of the rows are empty.

    A row has int(2 * Pareto(1.2)) entries, capped at n_cols and at `cap` when given. The Pareto
    shape 1.2 has a heavy tail: most rows are short and a few hold hundreds
    or thousands of entries.
    """
    limit = n_cols if cap is None else min(n_cols, cap)
    rng = random.Random(seed)
    lengths = []
    for _ in range(n_rows):
        if rng.random() < 0.05:
            lengths.append(0)
        else:
            lengths.append(min(limit, int(2.0 * rng.paretovariate(1.2))))
    return lengths


def power_law_csr(n_rows: int, n_cols: int, seed: int = SEED, cap: int | None = None):
    """(row_ptr, col_idx, values) for the power-law matrix. Columns are distinct and sorted per row."""
    lengths = power_law_row_lengths(n_rows, n_cols, seed, cap)
    rng = random.Random(seed + 1)
    row_ptr = [0]
    col_idx: list[int] = []
    values: list[float] = []
    for length in lengths:
        cols = sorted(rng.sample(range(n_cols), length))
        col_idx.extend(cols)
        values.extend(round(rng.uniform(-1.0, 1.0), 3) for _ in cols)
        row_ptr.append(len(col_idx))
    return row_ptr, col_idx, values


def power_law_graph(n: int = 1500, m: int = 3, extra: int = 20, seed: int = SEED):
    """An undirected preferential-attachment graph in CSR form, plus a detached path.

    Vertex v >= m attaches to m distinct earlier vertices, chosen in proportion
    to degree, so a few hubs collect most edges and the diameter is small.
    `extra` more vertices form a path with no edge to the rest, so BFS from
    vertex 0 leaves them at level -1. Returns (row_ptr, col_idx), symmetric,
    with sorted neighbor lists and no duplicates.
    """
    rng = random.Random(seed)
    edges: set[tuple[int, int]] = set()
    endpoints: list[int] = list(range(m))
    for v in range(m, n):
        chosen: set[int] = set()
        while len(chosen) < m:
            chosen.add(rng.choice(endpoints))
        for u in chosen:
            edges.add((u, v))
            endpoints.extend((u, v))
    for k in range(extra - 1):
        edges.add((n + k, n + k + 1))
    total = n + extra
    adj: list[list[int]] = [[] for _ in range(total)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    row_ptr = [0]
    col_idx: list[int] = []
    for neighbors in adj:
        col_idx.extend(sorted(neighbors))
        row_ptr.append(len(col_idx))
    return row_ptr, col_idx
