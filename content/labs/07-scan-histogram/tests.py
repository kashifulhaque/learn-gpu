import torch
from lab_common import A100, Checks, bandwidth_gbs, cuda_time, require_cuda

TILE = 256  # the block scans' tile: one element per thread, 256 threads

# Sizes chosen to hit the awkward cases: a single element, sizes that are not
# multiples of the 256-element block-scan tile or the 1024-element device-scan
# tile, exact multiples, and sizes whose tile sums need a second and a third
# level of scan (1025 tiles at 2**20 + 1, more at 2**21 + 17).
BLOCK_SIZES = [1, 2, 31, 255, 256, 257, 1000, 4096, 100_003]
SCAN_SIZES = [1, 2, 31, 255, 1023, 1024, 1025, 4096, 65_537, 1_000_003, 2**20, 2**20 + 1, 2**21 + 17, 5_000_000]
HIST_SIZES = [1, 7, 255, 256, 1000, 123_457, 2**20 + 3]


def ref_block_scan(x: torch.Tensor):
    """Per-tile exclusive scan and tile sums, in int64, then narrowed to int32."""
    n = x.numel()
    tiles = (n + TILE - 1) // TILE
    padded = torch.nn.functional.pad(x.long(), (0, tiles * TILE - n)).view(tiles, TILE)
    incl = padded.cumsum(1)
    excl = (incl - padded).flatten()[:n]
    return excl.int(), incl[:, -1].int()


def ref_exclusive(x: torch.Tensor):
    wide = x.double() if x.is_floating_point() else x.long()
    return wide.cumsum(0) - wide


def run(submission):
    c = Checks()
    needed = ["block_scan_kogge_stone", "block_scan_blelloch", "exclusive_scan", "histogram256", "histogram256_global"]
    if not c.require(submission, *needed):
        return c.finish()
    device = require_cuda()
    gen = torch.Generator(device="cpu").manual_seed(7)

    def rand_int(n, lo, hi):
        return torch.randint(lo, hi, (n,), generator=gen, dtype=torch.int32).to(device)

    # ---- Block scans: correctness ----
    for name, fn in (("Kogge-Stone", submission.block_scan_kogge_stone), ("Blelloch", submission.block_scan_blelloch)):

        def check_block(fn=fn):
            for n in BLOCK_SIZES:
                x = rand_int(n, -3, 4)
                out, sums = fn(x)
                torch.cuda.synchronize()
                want_out, want_sums = ref_block_scan(x)
                if out.shape != x.shape or out.dtype != torch.int32:
                    return False, f"n={n}: out has shape {tuple(out.shape)} and dtype {out.dtype}, expected {tuple(x.shape)} int32"
                if not torch.equal(out.cpu(), want_out.cpu()):
                    bad = (out.cpu() != want_out.cpu()).nonzero()[0].item()
                    return False, f"n={n}: first wrong element is {bad}: got {out[bad].item()}, expected {want_out[bad].item()}"
                if sums.numel() != want_sums.numel() or not torch.equal(sums.cpu(), want_sums.cpu()):
                    return False, f"n={n}: the tile sums are wrong: got {sums.cpu().tolist()[:4]}..., expected {want_sums.cpu().tolist()[:4]}..."
            return True, f"{len(BLOCK_SIZES)} sizes, from 1 to {max(BLOCK_SIZES)}"

        c.check(f"{name} block scan is exact on every tile", check_block)

    # ---- Device-wide scan: correctness ----
    def check_scan_int():
        for n in SCAN_SIZES:
            x = rand_int(n, -3, 4)
            before = x.clone()
            out = submission.exclusive_scan(x)
            torch.cuda.synchronize()
            if out.dtype != torch.int32 or out.shape != x.shape:
                return False, f"n={n}: out has shape {tuple(out.shape)} and dtype {out.dtype}"
            want = ref_exclusive(x).int()
            if not torch.equal(out.cpu(), want.cpu()):
                bad = (out.cpu() != want.cpu()).nonzero()[0].item()
                return False, f"n={n}: first wrong element is {bad}: got {out[bad].item()}, expected {want[bad].item()}"
            if not torch.equal(x, before):
                return False, f"n={n}: exclusive_scan changed its input"
        return True, f"{len(SCAN_SIZES)} sizes, from 1 to {max(SCAN_SIZES):,}"

    c.check("exclusive_scan is exact for int32 at every size", check_scan_int)

    def check_scan_view():
        base = rand_int(5001, -3, 4)
        view = base[1:]  # starts 4 bytes into the allocation, so it is not 16-byte aligned
        out = submission.exclusive_scan(view)
        torch.cuda.synchronize()
        want = ref_exclusive(view).int()
        return torch.equal(out.cpu(), want.cpu()), "an int32 view that starts 4 bytes past an aligned address"

    c.check("exclusive_scan handles an unaligned view", check_scan_view)

    def check_scan_float_exact():
        # Small integers are exact in float32 while the running sum stays below 2**24.
        for n in SCAN_SIZES:
            x = rand_int(n, 0, 4).float()
            out = submission.exclusive_scan(x)
            torch.cuda.synchronize()
            if out.dtype != torch.float32:
                return False, f"n={n}: out has dtype {out.dtype}"
            want = ref_exclusive(x).float()
            if not torch.equal(out.cpu(), want.cpu()):
                bad = (out.cpu() != want.cpu()).nonzero()[0].item()
                return False, f"n={n}: first wrong element is {bad}: got {out[bad].item()}, expected {want[bad].item()}"
        return True, "float32 holding small integers, so every partial sum is exact"

    c.check("exclusive_scan is exact for float32 holding small integers", check_scan_float_exact)

    def check_scan_float_random():
        # Real data: the order of additions differs from the reference, so allow rounding.
        n = 300_001
        x = torch.rand(n, generator=gen).to(device)
        out = submission.exclusive_scan(x).double()
        torch.cuda.synchronize()
        want = ref_exclusive(x)
        ok = bool(((out - want).abs() <= 1e-4 * want.abs() + 1e-3).all())
        worst = ((out - want).abs() / (want.abs() + 1e-3)).max().item()
        return ok, f"worst relative error {worst:.2e} (limit 1e-4)"

    c.check("exclusive_scan matches a float64 reference on random float32", check_scan_float_random)

    # ---- Histogram: correctness ----
    def check_hist(fn):
        def inner():
            for n in HIST_SIZES:
                x = torch.randint(0, 256, (n,), generator=gen, dtype=torch.uint8).to(device)
                got = fn(x)
                torch.cuda.synchronize()
                want = torch.bincount(x.long(), minlength=256)
                if got.numel() != 256:
                    return False, f"n={n}: expected 256 bins, got {got.numel()}"
                if not torch.equal(got.long().cpu(), want.cpu()):
                    bad = (got.long().cpu() != want.cpu()).nonzero()[0].item()
                    return False, f"n={n}: bin {bad} is {got[bad].item()}, expected {want[bad].item()}"
            # Every byte the same: all counts land in one bin.
            x = torch.full((100_000,), 200, dtype=torch.uint8, device=device)
            got = fn(x).long().cpu()
            if got[200].item() != 100_000 or got.sum().item() != 100_000:
                return False, f"all-equal input: bin 200 is {got[200].item()}, expected 100000"
            # A view that starts one byte into an allocation is not 16-byte aligned.
            base = torch.randint(0, 256, (50_001,), generator=gen, dtype=torch.uint8).to(device)
            view = base[1:]
            got = fn(view)
            torch.cuda.synchronize()
            want = torch.bincount(view.long(), minlength=256)
            if not torch.equal(got.long().cpu(), want.cpu()):
                return False, "a view that starts at an odd byte offset gave wrong counts (alignment of the vector loads?)"
            return True, f"{len(HIST_SIZES)} sizes, all-equal input, and an unaligned view"

        return inner

    c.check("histogram256_global (given) is correct", check_hist(submission.histogram256_global))
    c.check("histogram256 (privatized) is correct", check_hist(submission.histogram256))

    has_agg = False
    if hasattr(submission, "histogram256_aggregated"):
        try:
            submission.histogram256_aggregated(torch.zeros(4, dtype=torch.uint8, device=device))
            has_agg = True
        except NotImplementedError:
            print("      histogram256_aggregated is not implemented: skipping this optional stretch goal", flush=True)
    if has_agg:
        c.check("histogram256_aggregated (stretch) is correct", check_hist(submission.histogram256_aggregated))

    # ---- Performance ----
    n = 2**26  # 67,108,864 elements: 256 MiB of int32, far larger than the 40 MiB L2
    x = rand_int(n, 0, 4)

    # Block scans: read n, write n (plus one sum per tile).
    for key, fn in (("kogge_stone_gbs", submission.block_scan_kogge_stone), ("blelloch_gbs", submission.block_scan_blelloch)):
        try:
            ms = cuda_time(lambda fn=fn: fn(x), warmup=3, iters=10)
            c.metric(key, round(bandwidth_gbs(2 * n * 4, ms), 1))
        except Exception as exc:  # noqa: BLE001
            print(f"      {key}: could not be timed ({type(exc).__name__}: {exc})", flush=True)

    # Device scan: reduce-then-scan reads the input twice and writes the output once.
    ms = cuda_time(lambda: submission.exclusive_scan(x), warmup=5, iters=20)
    scan_gbs = bandwidth_gbs(3 * n * 4, ms)
    c.metric("scan_ms", round(ms, 3))
    c.metric("scan_gbs", round(scan_gbs, 1))
    c.metric("scan_fraction", round(scan_gbs / A100.COPY_CEILING_GBS, 3))
    try:
        ref_ms = cuda_time(lambda: torch.cumsum(x, 0, dtype=torch.int32), warmup=5, iters=20)
        c.metric("torch_cumsum_ms", round(ref_ms, 3))
    except Exception as exc:  # noqa: BLE001
        print(f"      torch.cumsum could not be timed ({type(exc).__name__}: {exc})", flush=True)
    floor = 0.40
    c.add(
        f"the scan moves 3n elements at {floor:.0%} or more of the copy ceiling",
        scan_gbs >= floor * A100.COPY_CEILING_GBS,
        f"{scan_gbs:.0f} GB/s counting 3n x 4 bytes; the floor is {floor * A100.COPY_CEILING_GBS:.0f} GB/s "
        f"({floor:.0%} of {A100.COPY_CEILING_GBS:.0f})",
    )

    # Histogram: read n bytes. The 256 bins are too small to count.
    hn = 2**26
    uniform = torch.randint(0, 256, (hn,), generator=gen, dtype=torch.uint8).to(device)
    constant = torch.full((hn,), 7, dtype=torch.uint8, device=device)

    ms = cuda_time(lambda: submission.histogram256(uniform), warmup=5, iters=20, flush_l2=True)
    hist_gbs = bandwidth_gbs(hn, ms)
    c.metric("hist_gbs", round(hist_gbs, 1))
    hist_floor = 0.25
    c.add(
        f"the privatized histogram reads at {hist_floor:.0%} or more of the copy ceiling",
        hist_gbs >= hist_floor * A100.COPY_CEILING_GBS,
        f"{hist_gbs:.0f} GB/s counting n bytes; the floor is {hist_floor * A100.COPY_CEILING_GBS:.0f} GB/s",
    )

    ms = cuda_time(lambda: submission.histogram256_global(uniform), warmup=3, iters=10, flush_l2=True)
    c.metric("hist_global_gbs", round(bandwidth_gbs(hn, ms), 1))

    # Every byte equal is the worst case for contention. The global-atomic version can be slow here, so time it lightly.
    ms = cuda_time(lambda: submission.histogram256(constant), warmup=2, iters=10, flush_l2=True)
    c.metric("hist_const_gbs", round(bandwidth_gbs(hn, ms), 1))
    ms = cuda_time(lambda: submission.histogram256_global(constant), warmup=1, iters=3, flush_l2=True)
    c.metric("hist_global_const_gbs", round(bandwidth_gbs(hn, ms), 1))
    if has_agg:
        ms = cuda_time(lambda: submission.histogram256_aggregated(constant), warmup=2, iters=10, flush_l2=True)
        c.metric("hist_agg_const_gbs", round(bandwidth_gbs(hn, ms), 1))
        ms = cuda_time(lambda: submission.histogram256_aggregated(uniform), warmup=3, iters=10, flush_l2=True)
        c.metric("hist_agg_gbs", round(bandwidth_gbs(hn, ms), 1))

    return c.finish()
