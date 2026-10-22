import math

from lab_common import Checks

# Link rates per direction in GB/s, from docs/FACTS.md section 7.1. They are
# half the both-directions totals that NVIDIA quotes. The tests keep their own
# copy so that a change to the starter's table can't move the targets.
PCIE4 = 31.5
NVLINK3 = 300.0
NVLINK4 = 450.0
IB_NDR = 50.0
ALPHA = 5e-6  # an assumed 5 microseconds per message; FACTS marks it unconfirmed
GIB = 2**30

# (metric key, ranks, link rate in GB/s). Each is a 1 GiB all-reduce.
PREDICTED_CASES = [
    ("pred_ring_8x_nvlink3_ms", 8, NVLINK3),
    ("pred_ring_8x_pcie4_ms", 8, PCIE4),
    ("pred_ring_8x_nvlink4_ms", 8, NVLINK4),
    ("pred_ring_2x_nvlink3_ms", 2, NVLINK3),
    ("pred_ring_16x_ib_ndr_ms", 16, IB_NDR),
]


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-30)


def _model_checks(c, sub):
    t, ring, tree, bus = sub.transfer_time, sub.ring_allreduce_time, sub.tree_allreduce_time, sub.bus_bandwidth_gbs

    def same(got, want, tol=1e-9):
        return _rel(got, want) < tol

    c.check("transfer_time(0 bytes) is alpha", lambda: same(t(0, 5e-6, 300.0), 5e-6))
    c.check("transfer_time(1 GB at 100 GB/s, alpha 0) is 10 ms", lambda: same(t(1e9, 0.0, 100.0), 0.01))

    # FACTS 7.2: 1 GiB over 8 ranks at 300 GB/s per direction is 6.26 ms of bandwidth.
    def bw_nvlink():
        got, want = ring(8, GIB, 0.0, NVLINK3), 1.75 * GIB / 300e9
        return same(got, want), f"got {got * 1e3:.3f} ms, expected {want * 1e3:.3f} ms"

    def bw_pcie():
        got, want = ring(8, GIB, 0.0, PCIE4), 1.75 * GIB / 31.5e9
        return same(got, want), f"got {got * 1e3:.2f} ms, expected {want * 1e3:.2f} ms"

    def lat():
        got = ring(8, 0, 5e-6, NVLINK3)
        return same(got, 14 * 5e-6), f"got {got * 1e6:.1f} us, expected 70.0 us"

    c.check("ring bandwidth term: 8 ranks, 1 GiB, 300 GB/s", bw_nvlink)
    c.check("ring bandwidth term: 8 ranks, 1 GiB, PCIe Gen4", bw_pcie)
    c.check("ring latency term is 2 (p - 1) alpha", lat)
    c.check("p = 2 ring moves n bytes once each way", lambda: same(ring(2, 1e9, 0.0, 100.0), 0.01))
    c.check("one rank needs no communication", lambda: ring(1, GIB, 5e-6, NVLINK3) == 0.0)

    c.check("tree latency term is 2 ceil(log2 p) alpha", lambda: same(tree(8, 0, 5e-6, NVLINK3), 6 * 5e-6))
    c.check("tree bandwidth term is 2 n / B", lambda: same(tree(8, GIB, 0.0, NVLINK3), 2 * GIB / 300e9))

    def cross():
        n_x = sub.crossover_bytes(8, 5e-6, NVLINK3)
        return same(ring(8, n_x, 5e-6, NVLINK3), tree(8, n_x, 5e-6, NVLINK3)), f"n* = {n_x / 2**20:.1f} MiB"

    def sides():
        n_x = sub.crossover_bytes(8, 5e-6, NVLINK3)
        return tree(8, n_x / 4, 5e-6, NVLINK3) < ring(8, n_x / 4, 5e-6, NVLINK3) and ring(
            8, n_x * 4, 5e-6, NVLINK3
        ) < tree(8, n_x * 4, 5e-6, NVLINK3)

    c.check("crossover: ring and tree cost the same at n*", cross)
    c.check("tree wins below the crossover, ring above it", sides)

    # 8 ranks, 1 GB all-reduced in 10 ms: algbw 100 GB/s, busbw 175 GB/s.
    c.check("busbw for all-reduce is algbw x 2(p-1)/p", lambda: same(bus("all_reduce", 8, 1e9, 0.01), 175.0))
    c.check("busbw for all-gather is algbw x (p-1)/p", lambda: same(bus("all_gather", 8, 1e9, 0.01), 87.5))
    c.check("busbw for broadcast is algbw", lambda: same(bus("broadcast", 8, 1e9, 0.01), 100.0))

    def fit():
        sizes = [2**k for k in range(10, 25, 2)]
        alpha, beta = sub.fit_alpha_beta(sizes, [8e-6 + n / 25e9 for n in sizes])
        return (
            same(alpha, 8e-6, 1e-3) and same(beta, 25.0, 1e-3),
            f"got alpha {alpha * 1e6:.2f} us, B {beta:.2f} GB/s; expected 8 us, 25 GB/s",
        )

    c.check("fit_alpha_beta recovers alpha and B from exact data", fit)


def _chunk_checks(c, sub):
    def check():
        for n in (0, 1, 5, 7, 64, 1000, 4097):
            for p in (1, 2, 3, 4, 8):
                bounds = sub.ring_chunks(n, p)
                if len(bounds) != p:
                    return False, f"n={n}, p={p}: {len(bounds)} chunks"
                if bounds[0][0] != 0 or bounds[-1][1] != n:
                    return False, f"n={n}, p={p}: chunks don't cover [0, n)"
                if any(bounds[i][1] != bounds[i + 1][0] for i in range(p - 1)):
                    return False, f"n={n}, p={p}: chunks aren't contiguous"
                sizes = [b - a for a, b in bounds]
                if max(sizes) - min(sizes) > 1:
                    return False, f"n={n}, p={p}: sizes {sizes} differ by more than 1"
        return True, "covers n exactly, sizes differ by at most 1, including n < p"

    c.check("ring_chunks splits n into p near-equal chunks", check)


def _ring_checks(c):
    import fixture_ring

    sizes = [1, 3, 7, 1200, 4097]
    for p in (2, 3, 4):
        try:
            res = fixture_ring.launch(p, {"name": "correctness", "sizes": sizes}, timeout=90)
        except Exception as exc:  # noqa: BLE001
            c.add(f"ring all-reduce runs with {p} processes", False, f"{type(exc).__name__}: {exc}")
            continue
        exact = [r for r in res["cases"] if r["kind"] == "int"]
        noisy = [r for r in res["cases"] if r["kind"] == "normal"]
        bad = [r["n"] for r in exact if r["max_err"] != 0.0]
        c.add(
            f"p = {p}: ring equals dist.all_reduce exactly on integer data",
            not bad,
            f"wrong for n = {bad}" if bad else f"n = {sizes}, including n < p",
        )
        worst = max(r["max_err"] for r in noisy)
        c.add(f"p = {p}: ring matches dist.all_reduce on random float32", worst < 1e-4, f"max abs error {worst:.2e}")
        c.add(f"p = {p}: ring_reduce_scatter leaves rank r the reduced chunk (r + 1) % p", res["reduce_scatter_ok"])
        # Every rank sends 2 (p - 1) / p of the data. Exact when p divides n.
        r = next(r for r in exact if r["n"] == 1200)
        want = 2 * (p - 1) * (1200 // p) * 4
        c.add(
            f"p = {p}: each rank sends 2(p - 1)/p of the array",
            r["min_bytes"] == want and r["max_bytes"] == want,
            f"n = 1200 floats: sent {r['min_bytes']} to {r['max_bytes']} bytes per rank, formula says {want}",
        )


def _measurements(c, sub):
    import fixture_ring

    # 1. Fit alpha and B to point-to-point times between two processes.
    sizes = [2**k for k in range(10, 25, 2)]
    alpha = beta = None
    try:
        res = fixture_ring.launch(2, {"name": "pingpong", "sizes": sizes, "warmup": 3, "iters": 15}, timeout=120)
        xs = [n for n, _ in res["points"]]
        ys = [t for _, t in res["points"]]
        alpha, beta = sub.fit_alpha_beta(xs, ys)
        ok = math.isfinite(alpha) and math.isfinite(beta) and beta > 0
        c.add("alpha and B fit to two-process gloo transfers", ok, f"alpha = {alpha * 1e6:.1f} us, B = {beta:.2f} GB/s")
        if ok:
            c.metric("fit_alpha_us", round(alpha * 1e6, 2))
            c.metric("fit_bandwidth_gbs", round(beta, 2))
    except Exception as exc:  # noqa: BLE001
        c.add("two-process ping-pong runs", False, f"{type(exc).__name__}: {exc}")

    # 2. Time the ring against gloo's own all-reduce.
    p, n = 4, 2**20
    try:
        res = fixture_ring.launch(p, {"name": "timing", "n": n, "warmup": 3, "iters": 9}, timeout=180)
    except Exception as exc:  # noqa: BLE001
        c.add("timing run", False, f"{type(exc).__name__}: {exc}")
        return
    n_bytes = n * 4
    ring_s, built_s = res["ring_s"], res["builtin_s"]
    c.metric("ring_ms", round(ring_s * 1e3, 3))
    c.metric("builtin_ms", round(built_s * 1e3, 3))
    try:
        c.metric("ring_busbw_gbs", round(sub.bus_bandwidth_gbs("all_reduce", p, n_bytes, ring_s), 3))
        c.metric("builtin_busbw_gbs", round(sub.bus_bandwidth_gbs("all_reduce", p, n_bytes, built_s), 3))
    except Exception:  # noqa: BLE001
        pass  # bus_bandwidth_gbs is checked above
    c.metric("ring_over_builtin", round(ring_s / built_s, 2))
    c.add(
        "your ring is within 50x of dist.all_reduce",
        ring_s < 50 * built_s,
        f"{ring_s * 1e3:.2f} ms against {built_s * 1e3:.2f} ms for 4 MiB over {p} processes",
    )
    if alpha is not None and beta is not None and beta > 0:
        try:
            model_s = sub.ring_allreduce_time(p, n_bytes, max(alpha, 0.0), beta)
            c.metric("model_over_measured", round(model_s / ring_s, 2))
        except Exception:  # noqa: BLE001
            pass


def run(submission):
    c = Checks()
    names = [
        "transfer_time",
        "ring_allreduce_time",
        "tree_allreduce_time",
        "crossover_bytes",
        "bus_bandwidth_gbs",
        "fit_alpha_beta",
        "ring_chunks",
        "ring_reduce_scatter",
        "ring_all_gather",
        "ring_allreduce",
    ]
    if not c.require(submission, *names):
        return c.finish()

    _model_checks(c, submission)
    _chunk_checks(c, submission)

    # Predicted times for 1 GiB, with the assumed 5 us alpha. Reported always,
    # so the reader sees the model's numbers next to the measurement below.
    try:
        for key, p, bw in PREDICTED_CASES:
            c.metric(key, round(submission.ring_allreduce_time(p, GIB, ALPHA, bw) * 1e3, 3))
        c.metric("crossover_mib_8x_nvlink3", round(submission.crossover_bytes(8, ALPHA, NVLINK3) / 2**20, 1))
    except Exception as exc:  # noqa: BLE001
        c.add("the model reports its predictions", False, f"{type(exc).__name__}: {exc}")

    _ring_checks(c)
    _measurements(c, submission)
    return c.finish()
