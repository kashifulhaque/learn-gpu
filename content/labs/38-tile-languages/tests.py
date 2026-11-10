"""Tests for the tile languages lab.

Everything except the last check is pure Python or torch-CPU. Randomness is
seeded. Each check message shows the expected value and the actual value.

Wrong implementations these checks catch:

- index_as_offset: block_slices (and a pallas_call built on it) treats the
  index map's result as an element offset, not a block index. Block (1, 2) of
  shape (4, 8) becomes rows 1:5 and columns 2:10, not rows 4:8 and columns 16:20.
- no_tail_mask: tile_gemm loads and stores full tiles without a mask on the last
  partial tile. On a 37 by 50 by 29 problem the fixture raises OutOfBounds,
  where a real kernel would read memory that belongs to something else.
- no_rescale: online_softmax adds exp(x - m_new) to the running sum without
  multiplying the old sum by exp(m_old - m_new). It passes when the maximum sits
  in the first tile and fails when a later tile holds a larger value.
- pad_zero: online_softmax loads the masked tail with other=0 instead of -inf.
  On rows of large negative values the zeros become the maximum.
- blocked_ignores_order: blocked_owner always treats the last dimension as the
  fastest, whatever `order` says.
- pipeline_off_by_one: pipeline_schedule loads tile s + 1 at step s, so one more
  tile is resident than the stages provide buffers for.
"""

from __future__ import annotations

import itertools

import fixture_tiles as fx
from lab_common import Checks

SEED = 38


def invoke(fn, *args, **kwargs):
    """Call `fn` and return `(value, None)`, or `(None, exception)`."""
    try:
        return fn(*args, **kwargs), None
    except Exception as exc:  # noqa: BLE001
        return None, exc


def fail(expected, exc):
    return False, f"expected {expected}, actual {type(exc).__name__}: {exc}"


def run(submission):
    c = Checks()
    names = (
        "grid_shape",
        "block_slices",
        "gemm_block_indices",
        "pallas_call",
        "tile_gemm",
        "online_softmax",
        "blocked_owner",
        "pipeline_schedule",
        "max_resident",
        "softmax_triton",
    )
    if not c.require(submission, *names):
        return c.finish()

    c.check("grid_shape is ceil(array / block)", lambda: _grid_shape(submission))
    c.check("the index map returns block indices, not element offsets", lambda: _index_maps(submission))
    c.check("pallas_call runs a kernel over the grid, with a partial last block", lambda: _pallas(submission))
    c.check("tile_gemm equals torch.matmul on multiples of the tile", lambda: _gemm_exact(submission, c))
    c.check("tile_gemm masks the partial last tile", lambda: _gemm_ragged(submission))
    c.check("online_softmax equals torch.softmax on a ragged width", lambda: _softmax_plain(submission))
    c.check("online_softmax rescales the running sum and masks with -inf", lambda: _softmax_hard(submission))
    c.check("blocked_owner matches a brute-force map of the layout", lambda: _layout(submission))
    c.check("pipeline_schedule overlaps loads with MMAs and fits S buffers", lambda: _pipeline(submission, c))
    _triton_section(c, submission)
    return c.finish()


# ---------------------------------------------------------------------------


def _grid_shape(sub):
    cases = (
        (((10, 20), (4, 8)), (3, 3)),
        (((8, 16), (4, 8)), (2, 2)),
        (((1, 1), (4, 8)), (1, 1)),
    )
    for args, want in cases:
        got, exc = invoke(sub.grid_shape, *args)
        if exc is not None:
            return fail(want, exc)
        if tuple(got) != want:
            return False, f"expected {want} for array {args[0]} and block {args[1]}, actual {tuple(got)}"
    return True, "expected (3, 3), (2, 2), (1, 1), actual the same"


def _index_maps(sub):
    got, exc = invoke(sub.block_slices, (1, 2), (4, 8), (10, 20))
    want = (slice(4, 8), slice(16, 20))
    if exc is not None:
        return fail(want, exc)
    if tuple(got) != want:
        # index_as_offset gives (slice(1, 5), slice(2, 10)).
        return False, f"expected {want} for block (1, 2), actual {tuple(got)}"
    got, exc = invoke(sub.block_slices, (2, 0), (4, 8), (10, 20))
    want = (slice(8, 10), slice(0, 8))
    if exc is not None:
        return fail(want, exc)
    if tuple(got) != want:
        return False, f"expected {want} for the clipped edge block (2, 0), actual {tuple(got)}"
    got, exc = invoke(sub.gemm_block_indices, 1, 2, 3)
    want = {"a": (1, 3), "b": (3, 2), "c": (1, 2)}
    if exc is not None:
        return fail(want, exc)
    if got != want:
        return False, f"expected {want} at grid point (1, 2, 3), actual {got}"
    return True, "expected rows 4:8 and columns 16:20 for block (1, 2), actual the same"


def _pallas(sub):
    import torch

    g = torch.Generator().manual_seed(SEED)
    x = torch.randn(10, 20, generator=g)
    y = torch.randn(10, 20, generator=g)

    def add(xb, yb, ob):
        ob[...] = xb + yb

    spec = fx.BlockSpec((4, 8), lambda i, j: (i, j))
    grid = (3, 3)  # ceil(10 / 4) by ceil(20 / 8)
    call, exc = invoke(sub.pallas_call, add, grid, [spec, spec], spec, (10, 20))
    if exc is not None:
        return fail("a callable", exc)
    got, exc = invoke(call, x, y)
    if exc is not None:
        return fail("x + y", exc)
    if tuple(got.shape) != (10, 20) or not torch.allclose(got, x + y):
        err = float((got - (x + y)).abs().max()) if tuple(got.shape) == (10, 20) else "wrong shape"
        return False, f"expected x + y on a (10, 20) output, actual max error {err}"

    def row_softmax(xb, ob):
        ob[...] = torch.softmax(xb, dim=-1)

    row_spec = fx.BlockSpec((4, 20), lambda i: (i, 0))
    call, exc = invoke(sub.pallas_call, row_softmax, (3,), [row_spec], row_spec, (10, 20))
    if exc is not None:
        return fail("a callable", exc)
    got, exc = invoke(call, x)
    if exc is not None:
        return fail("softmax over rows", exc)
    want = torch.softmax(x, dim=-1)
    if tuple(got.shape) != (10, 20) or not torch.allclose(got, want, atol=1e-6):
        err = float((got - want).abs().max()) if tuple(got.shape) == (10, 20) else "wrong shape"
        return False, f"expected softmax(x) with 3 blocks of 4 rows, actual max error {err}"
    return True, "expected x + y and softmax(x), actual the same"


def _matrices(m, k, n):
    import torch

    g = torch.Generator().manual_seed(SEED)
    return torch.randn(m, k, generator=g), torch.randn(k, n, generator=g)


def _gemm_exact(sub, c):
    import torch

    a, b = _matrices(64, 64, 64)
    fx.reset_stats()
    got, exc = invoke(sub.tile_gemm, a, b, 32, 32, 32)
    if exc is not None:
        return fail("A @ B", exc)
    stats = dict(fx.STATS)
    err = float((got - a @ b).abs().max())
    if err > 1e-4:
        return False, f"expected max error under 1e-4 on 64 x 64 x 64, actual {err:.3g}"
    # 4 programs, 2 K steps each: 16 loads, 8 mma, 4 stores.
    want = {"loads": 16, "stores": 4, "mmas": 8}
    c.metric("tile_gemm_loads", stats["loads"])
    if stats != want:
        return False, f"expected tile calls {want}, actual {stats}"
    return True, f"expected max error under 1e-4 and calls {want}, actual {err:.2g} and {stats}"


def _gemm_ragged(sub):
    import torch

    for m, k, n, tiles in ((37, 50, 29, (16, 16, 16)), (5, 7, 3, (8, 8, 8)), (33, 40, 17, (16, 8, 32))):
        a, b = _matrices(m, k, n)
        got, exc = invoke(sub.tile_gemm, a, b, *tiles)
        if exc is not None:
            return fail(f"A @ B for {m} x {k} x {n}", exc)
        if tuple(got.shape) != (m, n):
            return False, f"expected shape {(m, n)}, actual {tuple(got.shape)}"
        err = float((got - torch.matmul(a, b)).abs().max())
        if err > 1e-4:
            return False, f"expected max error under 1e-4 for {m} x {k} x {n} tiles {tiles}, actual {err:.3g}"
    return True, "expected max error under 1e-4 on three ragged shapes, actual the same"


def _softmax_check(sub, x, block_n, label):
    import torch

    got, exc = invoke(sub.online_softmax, x, block_n)
    if exc is not None:
        return fail(f"softmax for {label}", exc)
    want = torch.softmax(x, dim=-1)
    if tuple(got.shape) != tuple(want.shape):
        return False, f"expected shape {tuple(want.shape)}, actual {tuple(got.shape)}"
    err = float((got - want).abs().max())
    if not err <= 1e-5:
        return False, f"expected max error under 1e-5 for {label}, actual {err}"
    return True, f"expected max error under 1e-5 for {label}, actual {err:.2g}"


def _softmax_plain(sub):
    import torch

    g = torch.Generator().manual_seed(SEED)
    x = torch.randn(5, 70, generator=g)
    return _softmax_check(sub, x, 16, "5 x 70 and block 16")


def _softmax_hard(sub):
    import torch

    g = torch.Generator().manual_seed(SEED + 1)
    ramp = torch.randn(4, 70, generator=g)
    ramp[:, 60:] += 20.0  # the maximum is in the last tile, so the sum must be rescaled
    ok, msg = _softmax_check(sub, ramp, 16, "a maximum in the last tile")
    if not ok:
        return ok, msg
    neg = torch.randn(4, 70, generator=g) - 100.0  # all negative: a zero pad would win the max
    return _softmax_check(sub, neg, 16, "rows near -100 with a masked tail")


def _brute_force(shape, spt, tpw, wpc, order):
    """Walk every (warp, lane, register) and compute the element it holds."""
    cover = [spt[d] * tpw[d] * wpc[d] for d in range(2)]
    reps = [shape[d] // cover[d] for d in range(2)]

    def unlinear(index, extents):
        coords = [0, 0]
        for d in order:
            coords[d] = index % extents[d]
            index //= extents[d]
        return coords

    owner = {}
    for warp in range(wpc[0] * wpc[1]):
        w = unlinear(warp, wpc)
        for lane in range(tpw[0] * tpw[1]):
            t = unlinear(lane, tpw)
            for reg in range(spt[0] * spt[1] * reps[0] * reps[1]):
                loc = unlinear(reg % (spt[0] * spt[1]), spt)
                rep = unlinear(reg // (spt[0] * spt[1]), reps)
                elem = tuple(
                    rep[d] * cover[d] + (w[d] * tpw[d] + t[d]) * spt[d] + loc[d] for d in range(2)
                )
                owner[elem] = (warp, lane, reg)
    return owner


def _layout(sub):
    specs = (
        # (shape, size_per_thread, threads_per_warp, warps_per_cta, order, label)
        ((16, 16), (1, 4), (8, 4), (1, 1), (1, 0), "row-major, one warp"),
        ((16, 16), (1, 2), (8, 4), (2, 2), (1, 0), "row-major, four warps"),
        ((16, 16), (2, 1), (8, 4), (1, 2), (0, 1), "column-major order"),
        ((16, 16), (1, 1), (4, 8), (1, 1), (0, 1), "column-major with wrapped tiles"),
    )
    for shape, spt, tpw, wpc, order, label in specs:
        want = _brute_force(shape, spt, tpw, wpc, order)
        if len(want) != shape[0] * shape[1]:
            return False, f"the fixture's own map for {label} covers {len(want)} elements"
        for r, col in itertools.product(range(shape[0]), range(shape[1])):
            got, exc = invoke(sub.blocked_owner, r, col, shape, spt, tpw, wpc, order)
            if exc is not None:
                return fail(f"{want[(r, col)]} for ({r}, {col}) in {label}", exc)
            if tuple(got) != want[(r, col)]:
                return False, (
                    f"expected (warp, lane, register) {want[(r, col)]} for element ({r}, {col}) "
                    f"in the {label} layout, actual {tuple(got)}"
                )
    return True, "expected every element of four 16 x 16 layouts to match the brute-force map, actual the same"


def _pipeline(sub, c):
    want3 = [(0, None), (1, None), (2, 0), (3, 1), (4, 2), (None, 3), (None, 4)]
    got, exc = invoke(sub.pipeline_schedule, 5, 3)
    if exc is not None:
        return fail(want3, exc)
    if [tuple(s) for s in got] != want3:
        return False, f"expected {want3} for 5 tiles and 3 stages, actual {got}"
    want2 = [(0, None), (1, 0), (2, 1), (3, 2), (None, 3)]
    got, exc = invoke(sub.pipeline_schedule, 4, 2)
    if exc is not None:
        return fail(want2, exc)
    if [tuple(s) for s in got] != want2:
        return False, f"expected {want2} for 4 tiles and 2 stages, actual {got}"
    c.metric("pipeline_steps", len(got))
    for tiles, stages in ((5, 3), (4, 2), (8, 4), (2, 4)):
        sched, exc = invoke(sub.pipeline_schedule, tiles, stages)
        if exc is not None:
            return fail("a schedule", exc)
        resident, exc = invoke(sub.max_resident, sched)
        if exc is not None:
            return fail(min(stages, tiles), exc)
        want = min(stages, tiles)
        if resident != want:
            return False, f"expected {want} resident tiles for {tiles} tiles and {stages} stages, actual {resident}"
        seen_load, seen_mma = {}, {}
        for s, (load, use) in enumerate(sched):
            if load is not None:
                seen_load[load] = s
            if use is not None:
                seen_mma[use] = s
        if sorted(seen_load) != list(range(tiles)) or sorted(seen_mma) != list(range(tiles)):
            return False, f"expected every tile loaded and used once for {tiles} tiles and {stages} stages"
        if any(seen_load[t] >= seen_mma[t] for t in range(tiles)):
            return False, f"expected each tile's load before its MMA for {tiles} tiles and {stages} stages"
    return True, "expected the schedules for (5, 3) and (4, 2) and S resident tiles, actual the same"


def _skip_reason():
    try:
        import torch
    except ImportError:
        return "SKIP: the Triton softmax was not run. torch is not installed."
    if not torch.cuda.is_available():
        return "SKIP: the Triton softmax was not run. No CUDA GPU is visible. It runs on the A100 worker."
    try:
        import triton  # noqa: F401
    except ImportError:
        return "SKIP: the Triton softmax was not run. triton is not installed."
    return None


def _triton_section(c, sub):
    reason = _skip_reason()
    if reason is not None:
        print(reason, flush=True)
        c.metric("triton_softmax_gbs", "skipped (no CUDA GPU)")
        return

    import torch

    from lab_common import bandwidth_gbs, cuda_time

    def correct():
        g = torch.Generator().manual_seed(SEED)
        x = torch.randn(37, 100, generator=g).cuda()
        got = sub.softmax_triton(x)
        want = torch.softmax(x, dim=-1)
        err = float((got - want).abs().max())
        return err <= 1e-5, f"expected max error under 1e-5 on 37 x 100, actual {err:.3g}"

    ok = c.check("the Triton row softmax equals torch.softmax on the GPU", correct)
    if ok:
        x = torch.randn(4096, 4096, device="cuda")
        ms = cuda_time(lambda: sub.softmax_triton(x), flush_l2=True)
        c.metric("triton_softmax_gbs", round(bandwidth_gbs(2 * x.numel() * 4, ms), 1))
