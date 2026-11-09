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

# The chapter's 2-bit digits. Histogram [1, 3, 2, 2], exclusive offsets [0, 1, 4, 6],
# sorted [0, 1, 1, 1, 2, 2, 3, 3]. Slots in input order: 6, 1, 4, 2, 0, 7, 5, 3.
CHAPTER_KEYS = [3, 1, 2, 1, 0, 3, 2, 1]
CHAPTER_SORTED = [0, 1, 1, 1, 2, 2, 3, 3]
CHAPTER_OFFSETS = [0, 1, 4, 6]

# Wrong implementation this catches: an inclusive scan of the histogram
# (torch.cumsum of the bin counts) used as each bin's first output slot.
# On CHAPTER_KEYS those slots are [1, 4, 6, 8], not the exclusive offsets
# [0, 1, 4, 6], and the first key 3 is written at index 8.


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


def _show(expected, actual) -> str:
    return f"expected {expected}, actual {actual}"


def _expect_sort(sorted_keys, offsets) -> str:
    return f"sorted {list(sorted_keys)} and offsets {list(offsets)}"


def _values(tensor: torch.Tensor) -> list[int]:
    return [int(v) for v in tensor.detach().cpu().reshape(-1).tolist()]


def _pair(got):
    if not isinstance(got, tuple) or len(got) != 2:
        raise TypeError(f"got {type(got).__name__}, not a (sorted_keys, offsets) pair")
    got_keys, got_off = got
    if not isinstance(got_keys, torch.Tensor) or not isinstance(got_off, torch.Tensor):
        raise TypeError(
            f"got {type(got_keys).__name__} and {type(got_off).__name__}, not two tensors"
        )
    return got_keys, got_off


def ref_counting_sort(keys: torch.Tensor, num_bins: int):
    """CPU oracle: histogram, then an exclusive scan built by shifting the inclusive totals."""
    work = keys.detach().to(dtype=torch.int64, device="cpu")
    counts = torch.bincount(work, minlength=num_bins)
    offsets = torch.zeros(num_bins, dtype=torch.int64)
    if num_bins > 1:
        offsets[1:] = torch.cumsum(counts, 0)[:-1]
    out = torch.empty(work.numel(), dtype=torch.int64)
    cursor = offsets.clone()
    for index in range(int(work.numel())):
        key = int(work[index])
        slot = int(cursor[key])
        out[slot] = work[index]
        cursor[key] = slot + 1
    return out, offsets


def _grade_counting_sort(c: Checks, submission) -> None:
    def check_chapter():
        for dtype in (torch.int64, torch.int32):
            keys = torch.tensor(CHAPTER_KEYS, dtype=dtype)
            try:
                got_keys, got_off = _pair(submission.counting_sort(keys, 4))
            except Exception as exc:  # noqa: BLE001
                return False, _show(
                    _expect_sort(CHAPTER_SORTED, CHAPTER_OFFSETS),
                    f"{dtype}: {type(exc).__name__}: {exc}",
                )
            actual_keys, actual_off = _values(got_keys), _values(got_off)
            detail = _show(
                _expect_sort(CHAPTER_SORTED, CHAPTER_OFFSETS) + f" dtype {dtype}",
                _expect_sort(actual_keys, actual_off) + f" dtype {got_keys.dtype}",
            )
            if actual_keys != CHAPTER_SORTED or actual_off != CHAPTER_OFFSETS or got_keys.dtype != dtype:
                return False, detail
        return True, _show(
            _expect_sort(CHAPTER_SORTED, CHAPTER_OFFSETS),
            _expect_sort(CHAPTER_SORTED, CHAPTER_OFFSETS),
        )

    c.check("counting_sort matches the chapter's 8 digits", check_chapter)

    def check_empty_bin():
        # Counts [3, 0, 2, 3]. The empty bin must stay in the histogram, or the
        # exclusive scan collapses and the later offsets move.
        keys = torch.tensor([3, 0, 2, 0, 3, 2, 0, 3], dtype=torch.int64)
        want_keys = [0, 0, 0, 2, 2, 3, 3, 3]
        want_off = [0, 3, 3, 5]
        try:
            got_keys, got_off = _pair(submission.counting_sort(keys, 4))
        except Exception as exc:  # noqa: BLE001
            return False, _show(_expect_sort(want_keys, want_off), f"{type(exc).__name__}: {exc}")
        actual = _expect_sort(_values(got_keys), _values(got_off))
        ok = _values(got_keys) == want_keys and _values(got_off) == want_off
        return ok, _show(_expect_sort(want_keys, want_off), actual)

    c.check("counting_sort keeps an empty bin in the exclusive scan", check_empty_bin)

    def check_all_equal():
        keys = torch.tensor([2, 2, 2, 2], dtype=torch.int64)
        want_keys = [2, 2, 2, 2]
        want_off = [0, 0, 0, 4]
        try:
            got_keys, got_off = _pair(submission.counting_sort(keys, 4))
        except Exception as exc:  # noqa: BLE001
            return False, _show(_expect_sort(want_keys, want_off), f"{type(exc).__name__}: {exc}")
        actual = _expect_sort(_values(got_keys), _values(got_off))
        ok = _values(got_keys) == want_keys and _values(got_off) == want_off
        return ok, _show(_expect_sort(want_keys, want_off), actual)

    c.check("counting_sort puts an all-equal input in one bin", check_all_equal)

    def check_seeded():
        gen = torch.Generator(device="cpu").manual_seed(7)
        draws = [(0, 4), (1, 1), (1, 8), (7, 3), (31, 8), (255, 16), (1000, 32), (4096, 64)]
        sample = None
        for n, bins in draws:
            keys = torch.randint(0, bins, (n,), generator=gen, dtype=torch.int64)
            want_keys, want_off = ref_counting_sort(keys, bins)
            want = _expect_sort(_values(want_keys), _values(want_off))
            try:
                got_keys, got_off = _pair(submission.counting_sort(keys, bins))
            except Exception as exc:  # noqa: BLE001
                return False, _show(want, f"n={n}, bins={bins}: {type(exc).__name__}: {exc}")
            actual = _expect_sort(_values(got_keys), _values(got_off))
            if _values(got_keys) != _values(want_keys) or _values(got_off) != _values(want_off):
                return False, _show(want, f"n={n}, bins={bins}: {actual}")
            if n == 1 and bins == 1:
                sample = (want, actual)
        return True, _show(sample[0], sample[1]) + f" on {len(draws)} seeded draws (seed 7)"

    c.check("counting_sort matches a seeded reference", check_seeded)

    def check_unchanged():
        keys = torch.tensor(CHAPTER_KEYS, dtype=torch.int64)
        before = keys.clone()
        try:
            submission.counting_sort(keys, 4)
        except Exception as exc:  # noqa: BLE001
            return False, _show(before.tolist(), f"{type(exc).__name__}: {exc}")
        if not torch.equal(keys, before):
            return False, _show(before.tolist(), keys.tolist())
        return True, _show(before.tolist(), keys.tolist())

    c.check("counting_sort leaves its input unchanged", check_unchanged)


def run(submission):
    c = Checks()
    if not hasattr(submission, "counting_sort"):
        c.add(
            "counting_sort is defined",
            False,
            _show("counting_sort(keys, num_bins)", "the submission has no counting_sort"),
        )
    else:
        _grade_counting_sort(c, submission)

    if not torch.cuda.is_available():
        print("      No CUDA device is visible, so the kernel checks are skipped.", flush=True)
        return c.finish()

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
