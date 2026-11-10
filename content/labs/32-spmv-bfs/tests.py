"""Tests for lab 32: sparse formats, SpMV load balance, and BFS.

The CPU checks run anywhere. They build CSR from COO, multiply, price the
formats, split work by merge path, count load imbalance, and run BFS. The GPU
checks compile three SpMV kernels and compare them with torch.sparse. A
machine without CUDA skips those checks, and a skip is not a failure.

Plausible wrong implementations that these checks catch:

csr_end_minus_one: spmv_csr loops to row_ptr[i + 1] - 1 instead of row_ptr[i + 1].
    The range is half-open, so the last entry of every row is dropped. A row of
    one entry then gives 0. The SpMV check includes single-entry rows.

merge_by_rows_only: merge_path_partition gives each worker an equal number of
    rows, or an equal number of nonzeros, instead of an equal number of rows plus
    nonzeros. On the power-law matrix one worker then holds a row of 1153 entries.
    The partition check compares every boundary with a brute-force walk, and the
    imbalance check asks for a ratio under 1.05.

bottom_up_in_place: bottom_up_step labels a vertex the moment it finds a parent, and
    accepts any labeled neighbor as a parent. A vertex found this round then becomes
    a parent in the same round, so a whole chain of vertices joins level depth + 1.
    The BFS check compares the vertices of levels 1 and 2 with a plain queue BFS.
"""

from __future__ import annotations

import warnings
from collections import deque

from lab_common import Checks

import fixture_irregular as fx

NAMES = (
    "coo_to_csr",
    "spmv_csr",
    "coo_bytes",
    "csr_bytes",
    "ell_bytes",
    "hyb_width",
    "hyb_bytes",
    "merge_path_coordinate",
    "merge_path_partition",
    "spmv_merge_path",
    "warp_work",
    "imbalance",
    "bfs_levels_frontier",
    "bottom_up_step",
    "should_switch",
    "bfs_direction_optimizing",
    "simulate_work_queue",
    "spmv_thread_per_row",
    "spmv_warp_per_row",
    "spmv_merge",
)

N_ROWS = 4096  # the matrix for the CPU checks: 4096 by 4096, seed 32


def _torch():
    warnings.filterwarnings("ignore", message=".*Sparse.*")
    import torch

    return torch


def _fmt(values, limit=8):
    values = list(values)
    shown = ", ".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in values[:limit])
    return f"[{shown}{', ...' if len(values) > limit else ''}]"


def _queue_bfs(row_ptr, col_idx, source):
    """Plain-Python reference: one queue, one dequeue at a time."""
    level = [-1] * (len(row_ptr) - 1)
    level[source] = 0
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for k in range(row_ptr[u], row_ptr[u + 1]):
            v = col_idx[k]
            if level[v] == -1:
                level[v] = level[u] + 1
                queue.append(v)
    return level


def _brute_coordinates(row_ptr):
    """Walk the merge path one step at a time: coordinate after every diagonal."""
    n_rows = len(row_ptr) - 1
    i = j = 0
    path = [(0, 0)]
    while i < n_rows or j < row_ptr[-1]:
        if i < n_rows and row_ptr[i + 1] <= j:
            i += 1
        else:
            j += 1
        path.append((i, j))
    return path


def _small_coo():
    # Row 0 has a duplicate at column 1. Row 1 is empty. Row 3 has a duplicate at column 2.
    rows = [3, 0, 2, 0, 3, 0, 2]
    cols = [2, 1, 0, 1, 2, 3, 4]
    vals = [1.5, 2.0, -1.0, 0.25, 3.0, 4.0, 0.5]
    return 5, 6, rows, cols, vals


def run(submission):
    c = Checks()
    if not c.require(submission, *NAMES):
        return c.finish()
    sub = submission

    row_ptr, col_idx, values = fx.power_law_csr(N_ROWS, N_ROWS)
    lengths = [row_ptr[i + 1] - row_ptr[i] for i in range(N_ROWS)]
    nnz = row_ptr[-1]

    # Check 1 calls only coo_to_csr. Implementing that one function passes it.
    def csr_build():
        torch = _torch()
        n_rows, n_cols, rows, cols, vals = _small_coo()
        cases = [("a 5 by 6 matrix with duplicates and an empty row", n_rows, n_cols, rows, cols, vals)]
        # The power-law matrix as shuffled COO, with every 7th entry repeated.
        import random

        rng = random.Random(1)
        r_all = [i for i in range(N_ROWS) for _ in range(lengths[i])]
        triples = list(zip(r_all, col_idx, values))
        triples += triples[::7]
        rng.shuffle(triples)
        cases.append(
            (
                "the power-law matrix, shuffled, with repeats",
                N_ROWS,
                N_ROWS,
                [t[0] for t in triples],
                [t[1] for t in triples],
                [t[2] for t in triples],
            )
        )
        for label, nr, nc, r, cc, v in cases:
            ref = (
                torch.sparse_coo_tensor(torch.tensor([r, cc]), torch.tensor(v, dtype=torch.float64), (nr, nc))
                .coalesce()
                .to_sparse_csr()
            )
            want_ptr = ref.crow_indices().tolist()
            want_col = ref.col_indices().tolist()
            want_val = ref.values().tolist()
            try:
                got_ptr, got_col, got_val = sub.coo_to_csr(nr, r, cc, v)
            except Exception as exc:  # noqa: BLE001
                return False, f"expected row_ptr {_fmt(want_ptr)} for {label}, actual {type(exc).__name__}: {exc}"
            if list(got_ptr) != want_ptr:
                return False, f"expected row_ptr {_fmt(want_ptr)} for {label}, actual {_fmt(got_ptr)}"
            if list(got_col) != want_col:
                return False, f"expected col_idx {_fmt(want_col)} for {label}, actual {_fmt(got_col)}"
            if len(got_val) != len(want_val) or max(abs(a - b) for a, b in zip(got_val, want_val)) > 1e-12:
                return False, f"expected values {_fmt(want_val)} for {label}, actual {_fmt(got_val)}"
        return True, f"expected torch's coalesced CSR on 2 inputs, actual the same: row_ptr ends at {want_ptr[-1]}"

    c.check("coo_to_csr sorts and sums duplicates like torch", csr_build)

    # csr_end_minus_one fails here: single-entry rows give 0.
    def spmv_reference():
        torch = _torch()
        import random

        rng = random.Random(2)
        cases = []
        # Rows of one entry each, then the power-law matrix.
        cases.append(("rows of one entry", [0, 1, 2, 3, 4], [0, 1, 2, 0, 1], [1.0, 2.0, 3.0, 4.0, 5.0], 4))
        cases.append(("the power-law matrix", row_ptr, col_idx, values, N_ROWS))
        for label, rp, ci, vv, n_cols_ in cases:
            x = [rng.uniform(-1.0, 1.0) for _ in range(n_cols_)]
            csr = torch.sparse_csr_tensor(
                torch.tensor(rp), torch.tensor(ci), torch.tensor(vv, dtype=torch.float64), (len(rp) - 1, n_cols_)
            )
            want = torch.sparse.mm(csr, torch.tensor(x, dtype=torch.float64).reshape(-1, 1)).flatten().tolist()
            try:
                got = sub.spmv_csr(rp, ci, vv, x)
            except Exception as exc:  # noqa: BLE001
                return False, f"expected {_fmt(want)} for {label}, actual {type(exc).__name__}: {exc}"
            err = max(abs(a - b) for a, b in zip(got, want)) if len(got) == len(want) else float("inf")
            if err > 1e-9:
                return False, f"expected {_fmt(want)} for {label}, actual {_fmt(got)} (max error {err:.2e})"
        return True, "expected torch.sparse.mm on 2 matrices, actual a match within 1e-9"

    c.check("spmv_csr equals torch.sparse.mm, including single-entry rows", spmv_reference)

    def format_costs():
        want = {
            "coo": 12 * nnz,
            "csr": 4 * (N_ROWS + 1) + 8 * nnz,
            "ell": 8 * N_ROWS * max(lengths),
        }
        got = {
            "coo": sub.coo_bytes(nnz),
            "csr": sub.csr_bytes(N_ROWS, nnz),
            "ell": sub.ell_bytes(N_ROWS, max(lengths)),
        }
        for key in want:
            if got[key] != want[key]:
                return False, f"expected {key} {want[key]} bytes, actual {got[key]}"
        # The rule from Bell and Garland: add a column while a third of the rows reach it.
        for lens, k in (([3, 3, 3, 0, 0, 0], 3), ([5] + [1] * 8, 1), ([0, 0], 0)):
            if sub.hyb_width(lens) != k:
                return False, f"expected hyb_width {k} for {lens}, actual {sub.hyb_width(lens)}"
        if sub.hyb_bytes([5] + [1] * 8) != 9 * 1 * 8 + 4 * 12:
            return False, f"expected 120 bytes for lengths [5, 1 x 8], actual {sub.hyb_bytes([5] + [1] * 8)}"
        want_k = max(k for k in range(1, max(lengths) + 1) if 3 * sum(1 for x in lengths if x >= k) >= N_ROWS)
        hyb = sub.hyb_bytes(lengths)
        if sub.hyb_width(lengths) != want_k:
            return False, f"expected hyb_width {want_k} on the power-law matrix, actual {sub.hyb_width(lengths)}"
        if not (got["csr"] < hyb < got["ell"]):
            return False, f"expected csr {got['csr']} < hyb {hyb} < ell {got['ell']}"
        return True, (
            f"expected ELL to cost {got['ell'] / got['csr']:.1f}x CSR on this matrix, actual "
            f"{got['ell']} / {got['csr']}; hybrid {hyb} bytes at width {want_k}"
        )

    c.check("format byte costs: COO, CSR, ELL padding, and the hybrid split", format_costs)

    # merge_by_rows_only fails here: its boundaries are not the merge path's.
    def partition():
        brute = _brute_coordinates(row_ptr)
        total = N_ROWS + nnz
        for workers in (1, 7, 64, 128):
            try:
                got = sub.merge_path_partition(row_ptr, workers)
            except Exception as exc:  # noqa: BLE001
                return False, f"expected {workers + 1} boundaries for {workers} workers, actual {type(exc).__name__}: {exc}"
            share = -(-total // workers)
            want = [brute[min(w * share, total)] for w in range(workers + 1)]
            got = [tuple(b) for b in got]
            if got != want:
                at = next(w for w in range(workers + 1) if got[w] != want[w])
                return False, f"expected boundary {at} of {workers} workers at {want[at]}, actual {got[at]}"
        # Spot-check the search itself at every diagonal of a small matrix.
        small_ptr = [0, 3, 3, 4, 9, 9, 10]
        small = _brute_coordinates(small_ptr)
        for d, want in enumerate(small):
            got = tuple(sub.merge_path_coordinate(small_ptr, d))
            if got != want:
                return False, f"expected diagonal {d} at {want}, actual {got}"
        return True, "expected every boundary to match a brute-force walk of the merge path, actual a match"

    c.check("merge-path boundaries match a brute-force walk", partition)

    def merge_spmv():
        import random

        rng = random.Random(3)
        x = [rng.uniform(-1.0, 1.0) for _ in range(N_ROWS)]
        want = [0.0] * N_ROWS
        for i in range(N_ROWS):
            want[i] = sum(values[k] * x[col_idx[k]] for k in range(row_ptr[i], row_ptr[i + 1]))
        for workers in (1, 3, 64, 5000):
            got = sub.spmv_merge_path(row_ptr, col_idx, values, x, workers)
            err = max(abs(a - b) for a, b in zip(got, want))
            if err > 1e-9:
                return False, f"expected the reference SpMV with {workers} workers, actual max error {err:.2e}"
        return True, "expected the CSR result for 1, 3, 64, and 5000 workers, actual a match within 1e-9"

    c.check("spmv_merge_path gives the CSR result for any worker count", merge_spmv)

    def balance():
        warps = -(-N_ROWS // 32)
        ratios = {}
        for scheme in ("thread_per_row", "warp_per_row", "merge_path"):
            work = sub.warp_work(row_ptr, scheme, warps)
            ratios[scheme] = sub.imbalance(work)
        tpr, wpr, mp = ratios["thread_per_row"], ratios["warp_per_row"], ratios["merge_path"]
        shown = f"thread-per-row {tpr:.2f}, warp-per-row {wpr:.2f}, merge-path {mp:.3f}"
        if not (1.0 <= mp < 1.05):
            return False, f"expected merge-path imbalance in [1, 1.05), actual {shown}"
        if not (tpr > 5.0 and tpr > wpr > mp):
            return False, f"expected thread-per-row > 5 and thread-per-row > warp-per-row > merge-path, actual {shown}"
        c.metric("imbalance_thread_per_row", round(tpr, 3))
        c.metric("imbalance_warp_per_row", round(wpr, 3))
        c.metric("imbalance_merge_path", round(mp, 4))
        return True, f"expected merge-path near 1 and thread-per-row worst, actual {shown}"

    c.check("max/mean work per warp: thread-per-row is worst, merge-path is near 1", balance)

    # bottom_up_in_place fails in the second half of this check.
    def bfs_levels():
        g_ptr, g_col = fx.power_law_graph()
        want = _queue_bfs(g_ptr, g_col, 0)
        got = sub.bfs_levels_frontier(g_ptr, g_col, 0)
        if list(got) != want:
            at = next(v for v in range(len(want)) if got[v] != want[v])
            return False, f"expected level {want[at]} at vertex {at}, actual {got[at]}"
        if want.count(-1) != 20:
            return False, f"expected 20 unreached vertices, the reference found {want.count(-1)}"
        # One bottom-up step from level 0: only the neighbors of the source join.
        level = [-1] * len(want)
        level[0] = 0
        frontier, _ = sub.bottom_up_step(g_ptr, g_col, level, 0)
        level2 = [-1] * len(want)
        level2[0] = 0
        # A second round must find vertices at distance 2, not re-label distance 1.
        sub.bottom_up_step(g_ptr, g_col, level2, 0)
        frontier2, _ = sub.bottom_up_step(g_ptr, g_col, level2, 1)
        got_one = sorted(v for v in range(len(level)) if level[v] == 1)
        want_one = sorted(v for v in range(len(want)) if want[v] == 1)
        got_two = sorted(frontier2)
        want_two = sorted(v for v in range(len(want)) if want[v] == 2)
        if got_one != want_one:
            return False, f"expected {len(want_one)} vertices at level 1, actual {len(got_one)}"
        if got_two != want_two:
            return False, f"expected {len(want_two)} vertices at level 2, actual {len(got_two)}"
        return True, f"expected plain-queue levels on {len(want)} vertices, actual a match; deepest level {max(want)}"

    c.check("frontier BFS and one bottom-up step match a plain queue BFS", bfs_levels)

    def direction():
        cases = (
            (("top_down", 300, 1000, 5, 1000), "bottom_up"),
            (("top_down", 200, 1000, 5, 1000), "top_down"),
            (("bottom_up", 10, 1000, 10, 1000), "top_down"),
            (("bottom_up", 10, 1000, 100, 1000), "bottom_up"),
        )
        for args, want in cases:
            got = sub.should_switch(*args)
            if got != want:
                return False, f"expected {want} for {args}, actual {got}"
        g_ptr, g_col = fx.power_law_graph()
        want_level = _queue_bfs(g_ptr, g_col, 0)
        level, dirs, examined = sub.bfs_direction_optimizing(g_ptr, g_col, 0)
        if list(level) != want_level:
            at = next(v for v in range(len(want_level)) if level[v] != want_level[v])
            return False, f"expected level {want_level[at]} at vertex {at}, actual {level[at]}"
        top_down_edges = sum(g_ptr[v + 1] - g_ptr[v] for v in range(len(want_level)) if want_level[v] != -1)
        if dirs[0] != "top_down" or "bottom_up" not in dirs:
            return False, f"expected a top-down start and at least one bottom-up round, actual {dirs}"
        if examined >= top_down_edges:
            return False, f"expected fewer than {top_down_edges} edges examined, actual {examined}"
        c.metric("bfs_edges_top_down", top_down_edges)
        c.metric("bfs_edges_direction_optimizing", examined)
        return True, (
            f"expected fewer than {top_down_edges} edges examined, actual {examined} "
            f"({top_down_edges / examined:.2f}x fewer); directions {','.join(d[0] for d in dirs)}"
        )

    c.check("the direction rule switches, and direction-optimizing BFS examines fewer edges", direction)

    def work_queue():
        costs = [100 if i % 8 == 0 else 1 for i in range(64)]
        workers = 8
        static = sub.simulate_work_queue(costs, workers, False)
        dynamic = sub.simulate_work_queue(costs, workers, True)
        bound = sum(costs) / workers + max(costs) * (1 - 1 / workers)
        if static != 800:
            return False, f"expected a static makespan of 800, actual {static}"
        if dynamic > bound or dynamic >= static / 3:
            return False, f"expected a dynamic makespan at most {bound:.1f} and under {static / 3:.1f}, actual {dynamic}"
        if sub.simulate_work_queue([4, 4, 4], 8, True) != 4:
            return False, f"expected 4 for three tasks of cost 4 on 8 workers, actual {sub.simulate_work_queue([4, 4, 4], 8, True)}"
        return True, f"expected a dynamic makespan at most {bound:.1f}, actual {dynamic} against a static {static}"

    c.check("an atomic work counter beats static assignment on skewed tasks", work_queue)

    _gpu_section(c, sub, fx)
    return c.finish()


def _skip_reason():
    try:
        import torch
    except ImportError:
        return "SKIP: the CUDA SpMV kernels were not run. torch is not installed, so there is no GPU to compile them on."
    if not torch.cuda.is_available():
        return "SKIP: the CUDA SpMV kernels were not run. No CUDA GPU is visible, and they need one."
    return None


def _gpu_section(c, sub, fx):
    reason = _skip_reason()
    metrics = ("tpr_gbs", "wpr_gbs", "merge_gbs", "torch_gbs", "tpr_over_wpr_ms")
    if reason is not None:
        print(reason, flush=True)
        for key in metrics:
            c.metric(key, "skipped (no GPU)")
        return

    import warnings as _w

    import torch

    from lab_common import bandwidth_gbs, cuda_time

    _w.filterwarnings("ignore", message=".*Sparse.*")
    torch.manual_seed(32)
    n = 1 << 19
    print("Building the 524288 by 524288 power-law matrix, rows capped at 65536 entries, and compiling. The first run takes 30 to 60 seconds.", flush=True)
    rp, ci, vv = fx.power_law_csr(n, n, cap=1 << 16)
    nnz = rp[-1]
    max_len = max(rp[i + 1] - rp[i] for i in range(n))
    dev = torch.device("cuda")
    row_ptr = torch.tensor(rp, dtype=torch.int32, device=dev)
    col_idx = torch.tensor(ci, dtype=torch.int32, device=dev)
    val = torch.tensor(vv, dtype=torch.float32, device=dev)
    x = torch.randn(n, device=dev, dtype=torch.float32)

    csr64 = torch.sparse_csr_tensor(torch.tensor(rp), torch.tensor(ci), torch.tensor(vv, dtype=torch.float64), (n, n))
    abs64 = torch.sparse_csr_tensor(
        torch.tensor(rp), torch.tensor(ci), torch.tensor(vv, dtype=torch.float64).abs(), (n, n)
    )
    x64 = x.double().cpu().reshape(-1, 1)
    ref = torch.sparse.mm(csr64, x64).flatten()
    mag = torch.sparse.mm(abs64, x64.abs()).flatten()
    # Each row sums at most max_len products in float32: error under (len + 8) * 2^-24 * sum |a x|, doubled for margin.
    bound = 2.0 * (max_len + 8) * 2.0**-24 * mag + 1e-9

    kernels = (
        ("tpr", "thread-per-row", sub.spmv_thread_per_row),
        ("wpr", "warp-per-row", sub.spmv_warp_per_row),
        ("merge", "merge-based", sub.spmv_merge),
    )
    times = {}
    sparse_bytes = 4 * (n + 1) + 8 * nnz + 4 * n + 4 * n  # CSR, x read once, y written once

    for key, label, fn in kernels:

        def correct(fn=fn, label=label):
            try:
                y = fn(row_ptr, col_idx, val, x)
                torch.cuda.synchronize()
            except Exception as exc:  # noqa: BLE001
                return False, f"expected y with {n} entries from {label}, actual {type(exc).__name__}: {str(exc)[:400]}"
            if tuple(y.shape) != (n,):
                return False, f"expected shape ({n},), actual {tuple(y.shape)}"
            err = (y.double().cpu() - ref).abs()
            bad = (err > bound).nonzero(as_tuple=False)
            if bad.numel():
                r = int(bad[0])
                return False, f"expected {float(ref[r]):.6f} at row {r}, actual {float(y[r]):.6f}"
            return True, f"expected the float64 CSR product within the float32 bound, actual max error {float(err.max()):.2e}"

        ok = c.check(f"{label} SpMV matches torch.sparse on the power-law matrix", correct)
        if ok:
            ms = cuda_time(lambda fn=fn: fn(row_ptr, col_idx, val, x), warmup=5, iters=20, flush_l2=True)
            times[key] = ms
            c.metric(f"{key}_gbs", round(bandwidth_gbs(sparse_bytes, ms), 1))

    csr32 = torch.sparse_csr_tensor(row_ptr, col_idx, val, (n, n))
    xc = x.reshape(-1, 1)
    torch.sparse.mm(csr32, xc)
    ms = cuda_time(lambda: torch.sparse.mm(csr32, xc), warmup=5, iters=20, flush_l2=True)
    c.metric("torch_gbs", round(bandwidth_gbs(sparse_bytes, ms), 1))

    if "tpr" in times and "wpr" in times:

        def skew():
            ratio = times["tpr"] / times["wpr"]
            c.metric("tpr_over_wpr_ms", round(ratio, 3))
            # Loose: the model predicts thread-per-row loses here. Only a clear win for it fails.
            return ratio >= 0.8, (
                f"expected thread-per-row at least 0.8x warp-per-row's time on the skewed matrix, "
                f"actual {ratio:.3f}x ({times['tpr']:.4f} ms against {times['wpr']:.4f} ms)"
            )

        c.check("thread-per-row is not faster than warp-per-row on the skewed matrix", skew)
    else:
        c.metric("tpr_over_wpr_ms", "skipped (a kernel failed)")
