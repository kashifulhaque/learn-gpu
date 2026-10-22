import time

import torch
from lab_common import A100, Checks, bandwidth_gbs, close, cuda_time, require_cuda

# Thresholds. Each one is a fraction of what the arithmetic in chapter 5 predicts.
#
# Kernel 1 (uncoalesced): a warp that touches 32 sectors for 128 useful bytes
# reads 8x too many bytes and, with the coalesced write, moves 4.5x the bytes
# (36 B per 8 B counted). The companion course measured 6.9x on this card for
# a stride of 32 floats. We ask for 2x, under half of the byte prediction.
SCALE_MIN_SPEEDUP = 2.0
# ...and the fixed kernel must be a memory-bound kernel at the copy ceiling:
# at least 50% of A100.COPY_CEILING_GBS.
SCALE_MIN_CEILING_FRACTION = 0.5
# Function 2 (host syncs): NVIDIA's V100 measurement is 9.6 us per kernel with a
# sync after every launch against 3.8 us with launches overlapped, a ratio of
# 2.5. We ask for 60% of it. Batching the tensors beats this by far.
CLIP_MIN_SPEEDUP = 1.5
# Kernel 3 (divergence): a warp that runs two paths with half its lanes active
# each takes 2x the issue slots of one that runs them with all 32 active. We ask
# for 1.3x.
BRANCHY_MIN_SPEEDUP = 1.3


def _wall_ms(fn, warmup=3, iters=20):
    """Median wall-clock milliseconds of fn() followed by a synchronize."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]


def run(submission):
    c = Checks()
    names = [
        "hot_kernel",
        "fast_scale",
        "clip_to_max_abs",
        "fast_branchy",
        "parse_ptxas_info",
        "count_sass",
        "max_load_bits",
    ]
    if not c.require(submission, *names):
        return c.finish()

    import fixture_sass as fs
    import fixture_slow as slow

    # --- Part 3 first: it needs no GPU -------------------------------------
    c.check(
        "parse_ptxas_info: a clean kernel",
        lambda: submission.parse_ptxas_info(fs.PTXAS_CLEAN)
        == {"registers": 16, "smem_bytes": 0, "spill_stores": 0, "spill_loads": 0},
        f"got {submission.parse_ptxas_info(fs.PTXAS_CLEAN)}",
    )
    c.check(
        "parse_ptxas_info: a kernel that spills",
        lambda: submission.parse_ptxas_info(fs.PTXAS_SPILL)
        == {"registers": 255, "smem_bytes": 8192, "spill_stores": 48, "spill_loads": 52},
        f"got {submission.parse_ptxas_info(fs.PTXAS_SPILL)}",
    )
    c.check(
        "count_sass: scalar kernel",
        lambda: submission.count_sass(fs.SASS_SCALAR)
        == {"MOV": 1, "S2R": 2, "IMAD": 3, "ISETP": 1, "EXIT": 2, "LDG": 1, "FFMA": 1, "STG": 1, "BRA": 1},
        f"got {submission.count_sass(fs.SASS_SCALAR)}",
    )
    c.check(
        "count_sass: vector kernel with BAR, HMMA, and a predicated BRA",
        lambda: submission.count_sass(fs.SASS_VECTOR)
        == {
            "MOV": 1,
            "S2R": 2,
            "IMAD": 3,
            "ISETP": 1,
            "EXIT": 2,
            "LDG": 2,
            "FFMA": 4,
            "STG": 1,
            "BAR": 1,
            "HMMA": 1,
            "BRA": 1,
        },
        f"got {submission.count_sass(fs.SASS_VECTOR)}",
    )
    c.check(
        "max_load_bits: 32 for the scalar kernel, 128 for the vector one, 0 with no load",
        lambda: (
            submission.max_load_bits(fs.SASS_SCALAR) == 32
            and submission.max_load_bits(fs.SASS_VECTOR) == 128
            and submission.max_load_bits("        /*0000*/                   EXIT ;\n") == 0
        ),
    )

    device = require_cuda()
    print(f"      device = {torch.cuda.get_device_name(0)}", flush=True)

    # --- Part 1: the profiler finds the hot kernel -------------------------
    x_big = torch.randn(2048, 1024, device=device)

    def scale_then_small():
        y = slow.slow_scale(x_big)
        return y + 1

    name = None
    try:
        name = submission.hot_kernel(scale_then_small)
    except Exception as exc:  # noqa: BLE001
        c.add("hot_kernel names the dominant kernel", False, f"{type(exc).__name__}: {exc}")
    if name is not None:
        c.add(
            "hot_kernel names the dominant kernel",
            isinstance(name, str) and "slow_scale" in name,
            f"got {name!r}; the slow kernel should dominate a 2-D float32 add of the same size",
        )

    # --- Kernel 1: coalescing ----------------------------------------------
    for rows, cols in [(1, 1), (37, 129), (1000, 7), (5, 3000), (4096, 1000)]:
        x = torch.randn(rows, cols, device=device)
        c.check(
            f"fast_scale is correct for {rows}x{cols}",
            lambda x=x: close(submission.fast_scale(x), x * 2 + 1, 1e-6),
        )
    x = torch.randn(16384, 1024, device=device)  # 2^24 floats, 64 MiB in and 64 MiB out
    n_bytes = 2 * x.numel() * 4
    slow_ms = cuda_time(lambda: slow.slow_scale(x), flush_l2=True)
    fast_ms = cuda_time(lambda: submission.fast_scale(x), flush_l2=True)
    c.metric("slow_scale_ms", round(slow_ms, 4))
    c.metric("fast_scale_ms", round(fast_ms, 4))
    speedup = slow_ms / fast_ms
    gbs = bandwidth_gbs(n_bytes, fast_ms)
    c.metric("scale_speedup", round(speedup, 2))
    c.metric("scale_fast_gbs", round(gbs, 1))
    c.add(
        f"fast_scale is at least {SCALE_MIN_SPEEDUP}x faster than the slow kernel",
        speedup >= SCALE_MIN_SPEEDUP,
        f"{speedup:.2f}x. Neighboring threads should read neighboring floats.",
    )
    c.add(
        f"fast_scale reaches {SCALE_MIN_CEILING_FRACTION:.0%} of the {A100.COPY_CEILING_GBS:.0f} GB/s copy ceiling",
        gbs >= SCALE_MIN_CEILING_FRACTION * A100.COPY_CEILING_GBS,
        f"{gbs:.0f} GB/s counting a read and a write per element",
    )

    # --- Function 2: host synchronizations ---------------------------------
    gen = torch.Generator(device="cpu").manual_seed(0)

    def make(sizes, scales):
        return [
            (torch.randn(s, generator=gen) * sc).to(device)
            for s, sc in zip(sizes, scales)
        ]

    cases = {
        "ragged sizes, some over the threshold": make([1, 37, 4096, 1000, 5], [3.0, 0.2, 5.0, 0.5, 9.0]),
        "a tensor of all zeros": [torch.zeros(10, device=device), torch.full((7,), -4.0, device=device)],
        "equal sizes": make([256] * 6, [0.1, 1.0, 2.0, 7.0, 0.5, 30.0]),
    }
    thresh = 1.0
    for label, tensors in cases.items():
        originals = [t.clone() for t in tensors]

        def go(tensors=tensors, label=label):
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")
            try:
                got = submission.clip_to_max_abs(tensors, thresh)
            finally:
                torch.cuda.set_sync_debug_mode(0)
                torch.cuda.synchronize()
            return got

        got = None
        try:
            got = go()
        except Exception as exc:  # noqa: BLE001
            c.add(
                f"clip_to_max_abs makes no host sync: {label}",
                False,
                f"{type(exc).__name__}: {str(exc)[:160]}. Look for .item(), .tolist(), "
                "bool(tensor), or a tensor built from a Python float.",
            )
        if got is None:
            continue
        ok = len(got) == len(tensors)
        detail = "wrong number of tensors"
        if ok:
            ref = slow.slow_clip(originals, thresh)
            worst = 0.0
            for g, r, t in zip(got, ref, originals):
                if g.shape != t.shape or g.dtype != t.dtype:
                    ok, detail = False, f"shape {tuple(g.shape)} dtype {g.dtype}, expected {tuple(t.shape)} {t.dtype}"
                    break
                worst = max(worst, (g.float() - r.float()).abs().max().item())
            else:
                ok, detail = worst <= 1e-5, f"max abs error {worst:.2e} (tolerance 1e-5)"
        c.add(f"clip_to_max_abs is correct and sync-free: {label}", ok, detail)
        c.check(
            f"clip_to_max_abs leaves its inputs alone: {label}",
            lambda t=tensors, o=originals: all(torch.equal(a, b) for a, b in zip(t, o)),
        )

    many = [(torch.randn(4096, generator=gen) * 5).to(device) for _ in range(256)]
    slow_wall = _wall_ms(lambda: slow.slow_clip(many, thresh))
    fast_wall = _wall_ms(lambda: submission.clip_to_max_abs(many, thresh))
    c.metric("slow_clip_ms", round(slow_wall, 3))
    c.metric("fast_clip_ms", round(fast_wall, 3))
    clip_speedup = slow_wall / fast_wall
    c.metric("clip_speedup", round(clip_speedup, 2))
    c.add(
        f"clip_to_max_abs is at least {CLIP_MIN_SPEEDUP}x faster on 256 tensors of 4096",
        clip_speedup >= CLIP_MIN_SPEEDUP,
        f"{clip_speedup:.2f}x. Each .item() waits for the GPU to drain.",
    )

    # --- Kernel 3: divergence ----------------------------------------------
    for n in [1, 2, 37, 1001, 4096, 100000]:
        x = torch.rand(n, device=device)
        c.check(
            f"fast_branchy is correct for n={n}",
            lambda x=x: close(submission.fast_branchy(x), slow.slow_branchy(x), 1e-5),
        )
    x = torch.rand(1 << 22, device=device)
    slow_ms = cuda_time(lambda: slow.slow_branchy(x))
    fast_ms = cuda_time(lambda: submission.fast_branchy(x))
    c.metric("slow_branchy_ms", round(slow_ms, 4))
    c.metric("fast_branchy_ms", round(fast_ms, 4))
    speedup = slow_ms / fast_ms
    c.metric("branchy_speedup", round(speedup, 2))
    c.add(
        f"fast_branchy is at least {BRANCHY_MIN_SPEEDUP}x faster than the slow kernel",
        speedup >= BRANCHY_MIN_SPEEDUP,
        f"{speedup:.2f}x; the issue-slot arithmetic predicts about 2x",
    )
    return c.finish()
