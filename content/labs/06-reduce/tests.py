"""Tests for lab 06: Reductions.

Correctness is a hard check on awkward sizes. Performance is judged against the
measured 1275 GB/s copy ceiling (learn-inference chapter 10). A reduction reads
every input byte once and writes almost nothing, so its roofline floor is
4n bytes / 1275 GB/s. The thresholds sit well below that floor.
"""

import torch
from lab_common import A100, Checks, bandwidth_gbs, cuda_time, require_cuda

# A float32 sum of m terms in some order has an error of at most about (m - 1)
# rounding steps, each at most 2^-24 relative to the running magnitude, so the
# error is bounded by (m - 1) * 2^-24 * sum(|x|). A thread adds ceil(n / 221184)
# terms serially, a block tree adds log2(256) = 8 levels, and the second stage
# adds a few more. This bound covers all of them with margin.
EPS = 2.0**-24
RESIDENT_THREADS = 108 * 8 * 256


def sum_tolerance(n: int, l1: float) -> float:
    terms = -(-n // RESIDENT_THREADS) + 40
    return 2.0 * terms * EPS * l1 + 1e-9


def run(submission):
    c = Checks()
    if not c.require(submission, "build", "sum_tree", "sum_atomic", "max_atomic", "row_sum"):
        return c.finish()
    device = require_cuda()
    torch.manual_seed(0)

    print("Compiling. The first run takes 30 to 60 seconds.", flush=True)
    try:
        submission.build()
    except Exception as exc:  # noqa: BLE001
        c.add("the CUDA source compiles", False, str(exc)[:2000])
        return c.finish()
    c.add("the CUDA source compiles", True)

    # ---- correctness: sum -------------------------------------------------
    sizes = [1, 2, 31, 32, 33, 255, 256, 257, 1000, 1_000_003, (1 << 20) + 7, 1 << 24]
    for name, fn in (("sum_tree", submission.sum_tree), ("sum_atomic", submission.sum_atomic)):
        for n in sizes:
            x = torch.randn(n, device=device)
            ref = x.double().sum().item()
            tol = sum_tolerance(n, x.abs().double().sum().item())
            got = float(fn(x).item())
            torch.cuda.synchronize()
            c.add(
                f"{name} is correct at n={n:,}",
                abs(got - ref) <= tol,
                f"got {got:.6f}, expected {ref:.6f}, error {abs(got - ref):.2e} (tolerance {tol:.2e})",
            )
        # An input that starts 4 bytes past an aligned address: no float4 loads allowed.
        base = torch.randn(1_000_004, device=device)
        x = base[1:]
        ref = x.double().sum().item()
        tol = sum_tolerance(x.numel(), x.abs().double().sum().item())
        got = float(fn(x).item())
        c.add(f"{name} handles a misaligned input", abs(got - ref) <= tol, f"error {abs(got - ref):.2e}")

    # The atomic sum must not accumulate across calls: the output is zeroed each time.
    x = torch.randn(100_000, device=device)
    first = float(submission.sum_atomic(x).item())
    second = float(submission.sum_atomic(x).item())
    tol = sum_tolerance(x.numel(), x.abs().double().sum().item())
    c.add("sum_atomic gives the same answer on a second call", abs(first - second) <= 2 * tol, f"{first} then {second}")

    # ---- correctness: max -------------------------------------------------
    for n in sizes:
        x = torch.randn(n, device=device)
        c.add(
            f"max_atomic is exact at n={n:,}",
            float(submission.max_atomic(x).item()) == x.max().item(),
            f"got {float(submission.max_atomic(x).item())}, expected {x.max().item()}",
        )
    neg = torch.randn(100_003, device=device) - 50.0
    got = float(submission.max_atomic(neg).item())
    c.add("max_atomic works when every value is negative", got == neg.max().item(), f"got {got}, expected {neg.max().item()}")
    last = torch.zeros(1_000_003, device=device)
    last[-1] = 3.5
    got = float(submission.max_atomic(last).item())
    c.add("max_atomic sees the very last element", got == 3.5, f"got {got}")
    mixed = torch.cat([torch.full((50_000,), -3.0, device=device), torch.full((50_000,), 2.0, device=device)])
    mixed = mixed[torch.randperm(mixed.numel(), device=device)]
    got = float(submission.max_atomic(mixed).item())
    c.add("max_atomic orders negatives and positives correctly", got == 2.0, f"got {got}")
    base = torch.randn(1_000_004, device=device)
    x = base[1:]
    c.add("max_atomic handles a misaligned input", float(submission.max_atomic(x).item()) == x.max().item())

    # ---- correctness: row sums ---------------------------------------------
    shapes = [(1, 1), (3, 5), (7, 33), (9, 128), (10, 4097), (1000, 1000), (4096, 4096)]
    for rows, cols in shapes:
        x = torch.randn(rows, cols, device=device)
        ref = x.double().sum(1)
        tol = (cols / 32 + 40) * 2.0 * EPS * x.abs().double().sum(1) + 1e-9
        got = submission.row_sum(x).double()
        torch.cuda.synchronize()
        ok = got.shape == ref.shape and bool(((got - ref).abs() <= tol).all().item())
        c.add(f"row_sum is correct at shape ({rows}, {cols})", ok, f"shape {tuple(got.shape)}")

    # ---- performance ------------------------------------------------------
    # 2^26 floats = 256 MiB, far beyond the 40 MiB L2. Flush it anyway so every
    # call starts cold. A reduction reads 4n bytes and writes 4, so count 4n.
    n = 1 << 26
    x = torch.randn(n, device=device)
    nbytes = 4 * n
    ceiling = A100.COPY_CEILING_GBS
    print(f"      roofline floor at the copy ceiling: {nbytes / (ceiling * 1e9) * 1e3:.4f} ms for {nbytes / 2**20:.0f} MiB", flush=True)

    def measure(fn):
        ms = cuda_time(fn, warmup=10, iters=30, flush_l2=True)
        return bandwidth_gbs(nbytes, ms)

    tree = measure(lambda: submission.sum_tree(x))
    atomic = measure(lambda: submission.sum_atomic(x))
    mx = measure(lambda: submission.max_atomic(x))
    x2 = x.view(16384, 4096)
    rows = measure(lambda: submission.row_sum(x2))
    ref_sum = measure(lambda: x.sum())

    c.metric("sum_tree_gbs", round(tree, 1))
    c.metric("sum_atomic_gbs", round(atomic, 1))
    c.metric("max_atomic_gbs", round(mx, 1))
    c.metric("row_sum_gbs", round(rows, 1))
    c.metric("torch_sum_gbs", round(ref_sum, 1))
    best = max(tree, atomic, mx, rows)
    c.metric("best_ceiling_fraction", round(best / ceiling, 3))

    # Thresholds are fractions of the measured copy ceiling. The rated 1935 GB/s
    # bandwidth is not the target, and the SXM4 card, if you land on it, only
    # raises the achievable figure.
    c.add("sum_tree reaches 30% of the copy ceiling", tree >= 0.30 * ceiling, f"{tree:.0f} GB/s of {0.30 * ceiling:.0f} needed")
    c.add("sum_atomic reaches 40% of the copy ceiling", atomic >= 0.40 * ceiling, f"{atomic:.0f} GB/s of {0.40 * ceiling:.0f} needed")
    c.add("max_atomic reaches 40% of the copy ceiling", mx >= 0.40 * ceiling, f"{mx:.0f} GB/s of {0.40 * ceiling:.0f} needed")
    c.add("row_sum reaches 40% of the copy ceiling", rows >= 0.40 * ceiling, f"{rows:.0f} GB/s of {0.40 * ceiling:.0f} needed")
    return c.finish()
