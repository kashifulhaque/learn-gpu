import statistics
import time

import torch
from lab_common import A100, Checks, bandwidth_gbs, cuda_time, load_cuda, require_cuda

FUNCTIONS = ["copy_scalar", "copy_vec4", "copy_grid_stride", "copy_strided"]

# Sizes of one tensor, in MiB, for the sweep. The copy moves a source and a
# destination of this size. Below about 20 MiB both fit in the 40 MiB L2.
SWEEP_MIB = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
# The A100 80GB SXM4 is rated at 2039 GB/s (docs/FACTS.md, section 2.1). No
# copy on either card should beat that rating by more than a few percent.
SXM_RATING_GBS = 2039.0
# Loose thresholds, as fractions of the measured copy ceiling in lab_common.
MIN_COALESCED_FRACTION = 0.5
MIN_COALESCING_RATIO = 2.0


def _pure_checks(c, s):
    """Checks that need no GPU: the byte accounting and the plateau."""
    c.check("bytes_moved('copy', n) counts the read and the write",
            lambda: (s.bytes_moved("copy", 1000) == 8000, f"got {s.bytes_moved('copy', 1000)}, expected 8000"))
    c.check("bytes_moved('add', n) counts two reads and a write",
            lambda: (s.bytes_moved("add", 1000) == 12000, f"got {s.bytes_moved('add', 1000)}, expected 12000"))

    def sector_table():
        expected = {1: 8, 2: 12, 4: 20, 8: 36, 32: 36}
        got = {k: s.sector_bytes_per_element(k) for k in expected}
        return got == expected, f"got {got}, expected {expected}"

    c.check("sector_bytes_per_element follows the 32-byte sector model", sector_table)

    def plateau_table():
        sweep = [(4, 2500.0), (32, 1900.0), (128, 1300.0), (256, 1250.0), (512, 1280.0)]
        got = s.plateau_gbs(sweep)
        want = statistics.median([1300.0, 1250.0, 1280.0])
        even = s.plateau_gbs(sweep, min_mib=100)
        return (
            abs(got - want) < 1e-6 and abs(even - want) < 1e-6,
            f"got {got} and {even}, expected {want} for entries at or above 80 MiB",
        )

    c.check("plateau_gbs takes the median of the large sizes", plateau_table)

    def plateau_even():
        got = s.plateau_gbs([(100, 1000.0), (200, 1200.0)], min_mib=50)
        return abs(got - 1100.0) < 1e-6, f"got {got}, expected 1100.0"

    c.check("plateau_gbs averages the middle pair of an even count", plateau_even)

    def plateau_empty():
        try:
            s.plateau_gbs([(4, 1.0)])
        except ValueError:
            return True
        return False, "expected ValueError when no entry is large enough"

    c.check("plateau_gbs raises when nothing is large enough", plateau_empty)


def _copy_ok(fn, n, *, offset=0, **kwargs):
    """Run `fn` on n floats starting `offset` floats into a larger buffer."""
    base = torch.randn(n + offset + 8, device="cuda")
    out = torch.full((n + offset + 8,), 7.0, device="cuda")
    src = base[offset : offset + n]
    dst = out[offset : offset + n]
    fn(src, dst, **kwargs)
    torch.cuda.synchronize()
    if not torch.equal(dst, src):
        bad = (dst != src).nonzero().flatten()
        first = int(bad[0].item()) if bad.numel() else -1
        return False, f"{bad.numel()} of {n} elements differ, the first at index {first}"
    # Everything outside the destination view must be untouched.
    guard = torch.cat([out[:offset], out[offset + n :]])
    if not bool((guard == 7.0).all()):
        return False, "the kernel wrote outside the destination"
    return True, f"n={n:,}"


def _performance(c, submission, module, device):
    # Performance. Every number is a metric, and the thresholds are loose.
    floor = MIN_COALESCED_FRACTION * A100.COPY_CEILING_GBS
    big = 1 << 26  # 256 MiB per tensor, far above the 40 MiB L2
    grid_blocks = A100.SM_COUNT * 8

    scalar = submission.time_copy_gbs(module.copy_scalar, big)
    vec4 = submission.time_copy_gbs(module.copy_vec4, big)
    grid = submission.time_copy_gbs(lambda s, d: module.copy_grid_stride(s, d, grid_blocks), big)
    c.metric("scalar_gbs", round(scalar, 1))
    c.metric("vec4_gbs", round(vec4, 1))
    c.metric("grid_stride_gbs", round(grid, 1))
    for name, value in (("copy_scalar", scalar), ("copy_vec4", vec4), ("copy_grid_stride", grid)):
        c.add(
            f"{name} reaches at least {MIN_COALESCED_FRACTION:.0%} of the {A100.COPY_CEILING_GBS:.0f} GB/s copy ceiling",
            value >= floor,
            f"{value:.0f} GB/s, {value / A100.COPY_CEILING_GBS:.0%} of the ceiling",
        )

    torch_src = torch.randn(big, device=device)
    torch_dst = torch.empty_like(torch_src)
    torch_ms = cuda_time(lambda: torch_dst.copy_(torch_src))
    torch_gbs = bandwidth_gbs(submission.bytes_moved("copy", big), torch_ms)
    print(f"      torch copy_ = {torch_gbs:.1f} GB/s (for comparison)", flush=True)
    del torch_src, torch_dst

    # Strided against coalesced, at the size the companion course measured:
    # 2^24 floats, a stride of 32, and a read footprint of 16.8 MB that fits in the L2.
    n_small = 1 << 24
    coalesced_small = submission.time_copy_gbs(module.copy_scalar, n_small)
    s_src = torch.randn(n_small, device=device)
    s_dst = torch.empty_like(s_src)
    s_ms = cuda_time(lambda: module.copy_strided(s_src, s_dst, 32))
    strided = bandwidth_gbs(submission.bytes_moved("copy", n_small), s_ms)
    ratio = coalesced_small / max(strided, 1e-9)
    c.metric("strided_gbs", round(strided, 1))
    c.metric("coalescing_ratio", round(ratio, 2))
    c.add(
        f"a stride of 32 floats is at least {MIN_COALESCING_RATIO:.0f}x slower than coalesced",
        ratio >= MIN_COALESCING_RATIO,
        f"{ratio:.1f}x ({coalesced_small:.0f} against {strided:.0f} GB/s). "
        "Counted bytes undercount the strided kernel: it moves "
        f"{submission.sector_bytes_per_element(32)} bytes per element but counts 8.",
    )
    del s_src, s_dst

    # The same stride with a read footprint above the L2: 2^26 floats read at a
    # stride of 32 touch 2^21 distinct sectors, 64 MiB.
    b_src = torch.randn(big, device=device)
    b_dst = torch.empty_like(b_src)
    b_ms = cuda_time(lambda: module.copy_strided(b_src, b_dst, 32), iters=20)
    strided_big = bandwidth_gbs(submission.bytes_moved("copy", big), b_ms)
    c.metric("strided_big_gbs", round(strided_big, 1))
    del b_src, b_dst

    # The sweep: one vectorized copy per tensor size.
    print("\nSweep (copy_vec4):", flush=True)
    print(f"  {'MiB':>6}  {'GB/s':>8}  {'of 1935':>8}  {'of 1275':>8}", flush=True)
    sweep = []
    for mib in SWEEP_MIB:
        n = mib * 2**20 // 4
        torch.cuda.empty_cache()
        gbs = submission.time_copy_gbs(module.copy_vec4, n)
        sweep.append((float(mib), gbs))
        print(
            f"  {mib:>6}  {gbs:>8.1f}  {gbs / A100.HBM_BW_GBS:>8.1%}  {gbs / A100.COPY_CEILING_GBS:>8.1%}",
            flush=True,
        )
        torch.cuda.empty_cache()

    plateau = submission.plateau_gbs(sweep)
    c.metric("plateau_gbs", round(plateau, 1))
    c.metric("rating_fraction", round(plateau / A100.HBM_BW_GBS, 3))
    c.metric("l2_resident_gbs", round(dict(sweep)[8.0], 1))
    print(
        f"      the datasheet's 95% DRAM efficiency would be {0.95 * A100.HBM_BW_GBS:.0f} GB/s; "
        f"your plateau is {plateau:.0f} GB/s",
        flush=True,
    )
    c.add(
        f"the plateau is at least {MIN_COALESCED_FRACTION:.0%} of the copy ceiling",
        plateau >= floor,
        f"{plateau:.0f} GB/s, {plateau / A100.COPY_CEILING_GBS:.0%} of {A100.COPY_CEILING_GBS:.0f}",
    )
    c.add(
        "no size beats the rated bandwidth by more than 5% once it exceeds the L2",
        all(gbs <= 1.05 * SXM_RATING_GBS for mib, gbs in sweep if mib >= 2 * A100.L2_MB),
        "a figure above the rating usually means the bytes or the time are wrong",
    )


def run(submission):
    c = Checks()
    needed = (
        "CUDA_SOURCE",
        "CPP_DECLARATIONS",
        "bytes_moved",
        "sector_bytes_per_element",
        "plateau_gbs",
        "time_copy_gbs",
    )
    if not c.require(submission, *needed):
        return c.finish()

    _pure_checks(c, submission)
    device = require_cuda()
    torch.manual_seed(0)
    props = torch.cuda.get_device_properties(0)
    print(f"      device = {props.name}", flush=True)

    print("Compiling. The first run takes 30 to 60 seconds.", flush=True)
    started = time.perf_counter()
    try:
        module = load_cuda("lab03_probe", submission.CUDA_SOURCE, submission.CPP_DECLARATIONS, FUNCTIONS)
    except Exception as exc:  # noqa: BLE001
        c.add("the CUDA source compiles", False, str(exc)[-2000:])
        return c.finish()
    c.add("the CUDA source compiles", True, f"{time.perf_counter() - started:.1f}s")

    # Correctness. Sizes that are not multiples of the block size or of 4.
    sizes = (1, 3, 5, 1023, 1024, 1_000_003, (1 << 22) + 1)
    for n in sizes:
        c.check(f"copy_scalar is correct at n={n:,}", lambda n=n: _copy_ok(module.copy_scalar, n))
    for n in sizes:
        c.check(f"copy_vec4 is correct at n={n:,}", lambda n=n: _copy_ok(module.copy_vec4, n))
    for blocks in (1, 3, A100.SM_COUNT * 8):
        for n in (1, 1_000_003):
            c.check(
                f"copy_grid_stride is correct at n={n:,} with {blocks} blocks",
                lambda n=n, b=blocks: _copy_ok(module.copy_grid_stride, n, blocks=b),
            )

    def strided_ok(n, stride):
        src = torch.randn(n, device=device)
        dst = torch.full_like(src, float("nan"))
        module.copy_strided(src, dst, stride)
        torch.cuda.synchronize()
        index = (torch.arange(n, device=device, dtype=torch.long) * stride) % n
        return torch.equal(dst, src[index]), f"n={n:,}, stride={stride}"

    for n, stride in ((1, 32), (1_000_003, 1), (1_000_003, 7), (1 << 20, 32), (1 << 24, 32)):
        c.check(f"copy_strided is correct at n={n:,}, stride={stride}", lambda n=n, s=stride: strided_ok(n, s))

    try:
        _performance(c, submission, module, device)
    except Exception as exc:  # noqa: BLE001
        c.add(
            "the performance section runs",
            False,
            f"{type(exc).__name__}: {exc}. Finish bytes_moved, sector_bytes_per_element, plateau_gbs, and time_copy_gbs.",
        )

    # Last, because a misaligned float4 access crashes the process.
    for n in (7, 1_000_003):
        c.check(
            f"copy_vec4 handles a pointer that is not 16-byte aligned (n={n:,})",
            lambda n=n: _copy_ok(module.copy_vec4, n, offset=1),
        )
    return c.finish()
