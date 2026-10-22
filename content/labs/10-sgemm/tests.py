import torch
from lab_common import A100, Checks, cuda_time, load_cuda, require_cuda, tflops

# Shapes include sizes of 1, primes, sizes one past a tile edge (the kernels use
# 32 and 128 tiles, with K steps of 8 and 32), and a size divisible by 4 so the
# vectorized path runs. K = 1 and M = 1 are the degenerate cases.
SHAPES = [
    (1, 1, 1),
    (1, 17, 3),
    (5, 1, 9),
    (31, 33, 35),
    (64, 64, 64),
    (129, 127, 130),
    (128, 128, 8),
    (200, 300, 100),
    (256, 260, 252),
    (513, 515, 517),
    (1023, 1025, 1021),
]
REQUIRED = ["sgemm_naive", "sgemm_tiled", "sgemm_regblock"]

# The performance shape: 4096^3 is 2 * 4096^3 = 137.4 GFLOP, so the 19.5 TFLOP/s
# fp32 peak puts the floor at 7.05 ms.
PERF_M = PERF_N = PERF_K = 4096

# Loose thresholds, as a percentage of cuBLAS on the same card. Boehm's SGEMM
# worklog (RTX A6000) measured 12.8% for shared-memory tiling and 68.7% for 2D
# blocktiling; the thresholds below are less than half of those, because this
# lab runs on a different card and nobody here has run it.
MIN_TILED_PCT = 5.0
MIN_REGBLOCK_PCT = 30.0
MIN_REGBLOCK_OVER_TILED = 1.5


def _tolerance(k):
    # fp32 accumulation over K terms of N(0,1) products against a float64
    # reference. The rounding error grows roughly like K * 6e-8 * |partial sum|;
    # a wrong index or a missing guard is off by order 1.
    return 1e-5 + 4e-6 * k


def _helpers(c, submission):
    if not c.require(submission, "gemm_flops", "min_bytes", "tile_intensity", "smem_bytes_per_fma"):
        return

    def run(name, fn, expected, tol=1e-9):
        try:
            got = fn()
        except NotImplementedError:
            c.add(name, False, "not implemented yet")
            return
        c.add(name, abs(got - expected) <= tol * max(1.0, abs(expected)), f"got {got}, expected {expected}")

    run("gemm_flops(4096, 4096, 4096)", lambda: submission.gemm_flops(4096, 4096, 4096), 2 * 4096**3)
    run("gemm_flops(3, 5, 7)", lambda: submission.gemm_flops(3, 5, 7), 210)
    run("min_bytes(4096, 4096, 4096)", lambda: submission.min_bytes(4096, 4096, 4096), 3 * 4096 * 4096 * 4)
    run("min_bytes(2, 3, 4, dtype_bytes=2)", lambda: submission.min_bytes(2, 3, 4, 2), 2 * (8 + 12 + 6))
    run("tile_intensity(128, 128) is 32 FLOP/byte", lambda: submission.tile_intensity(128, 128), 32.0)
    run("tile_intensity(32, 32) is 8 FLOP/byte", lambda: submission.tile_intensity(32, 32), 8.0)
    run("smem_bytes_per_fma(1, 1) is 8", lambda: submission.smem_bytes_per_fma(1, 1), 8.0)
    run("smem_bytes_per_fma(8, 8) is 1", lambda: submission.smem_bytes_per_fma(8, 8), 1.0)


def _correct(module, name, device):
    """Return (ok, detail): the first shape the kernel gets wrong, or a summary."""
    fn = getattr(module, name)
    gen = torch.Generator(device=device).manual_seed(1234)
    for m, n, k in SHAPES:
        a = torch.randn(m, k, device=device, generator=gen)
        b = torch.randn(k, n, device=device, generator=gen)
        a0, b0 = a.clone(), b.clone()
        out = fn(a, b)
        torch.cuda.synchronize()
        if tuple(out.shape) != (m, n) or out.dtype != torch.float32:
            return False, f"M={m} N={n} K={k}: got shape {tuple(out.shape)} and dtype {out.dtype}"
        reference = (a.double() @ b.double()).float()
        err = (out - reference).abs().max().item()
        tol = _tolerance(k)
        if not (err <= tol):  # also catches NaN
            return False, f"M={m} N={n} K={k}: max abs error {err:.3e}, tolerance {tol:.1e}"
        if not (torch.equal(a, a0) and torch.equal(b, b0)):
            return False, f"M={m} N={n} K={k}: the kernel modified its inputs"
    return True, f"all {len(SHAPES)} shapes match a float64 reference"


def run(submission):
    c = Checks()
    if not c.require(submission, "cuda_source", "CPP_DECLARATIONS", "FUNCTIONS"):
        return c.finish()

    _helpers(c, submission)

    names = list(submission.FUNCTIONS)
    missing = [n for n in REQUIRED if n not in names]
    if missing:
        c.add("FUNCTIONS lists the required kernels", False, f"missing {', '.join(missing)}")
        return c.finish()

    device = require_cuda()
    # cuBLAS must use the CUDA cores, as the lab's kernels do, not TF32 tensor cores.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    print("Compiling. The first run takes 30 to 60 seconds.", flush=True)
    try:
        module = load_cuda("lab10_sgemm", submission.cuda_source(), submission.CPP_DECLARATIONS, names)
    except Exception as exc:  # noqa: BLE001
        c.add("the CUDA source compiles", False, str(exc)[-2000:])
        return c.finish()
    c.add("the CUDA source compiles", True)

    # --- correctness ---------------------------------------------------------
    correct = {}
    for name in names:
        if not hasattr(module, name):
            c.add(f"{name} exists", False, "declared in FUNCTIONS but missing from the module")
            correct[name] = False
            continue
        try:
            ok, detail = _correct(module, name, device)
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"{type(exc).__name__}: {str(exc)[:300]}"
        correct[name] = c.add(f"{name} is correct on awkward shapes", ok, detail)

    # --- performance ---------------------------------------------------------
    gen = torch.Generator(device=device).manual_seed(7)
    a = torch.randn(PERF_M, PERF_K, device=device, generator=gen)
    b = torch.randn(PERF_K, PERF_N, device=device, generator=gen)
    flops = 2 * PERF_M * PERF_N * PERF_K
    print(
        f"      4096^3 SGEMM is {flops / 1e9:.1f} GFLOP; at the {A100.FP32_TFLOPS} TFLOP/s "
        f"fp32 peak the floor is {flops / (A100.FP32_TFLOPS * 1e12) * 1e3:.2f} ms",
        flush=True,
    )

    def gflops(fn, warmup, iters):
        ms = cuda_time(lambda: fn(a, b), warmup=warmup, iters=iters)
        return tflops(flops, ms) * 1000.0, ms

    cublas_gf, cublas_ms = gflops(torch.matmul, 5, 30)
    c.metric("cublas_gflops", round(cublas_gf, 1))
    print(f"      cuBLAS (TF32 off): {cublas_ms:.3f} ms", flush=True)

    reference = torch.matmul(a, b)
    pct = {}
    for name in names:
        if not correct.get(name):
            print(f"      skipping the timing of {name}: it is not correct yet", flush=True)
            continue
        slow = name in ("sgemm_naive", "sgemm_coalesced")
        # The result at full size must also match cuBLAS, which catches bugs
        # (an overflowing index, a race) that the small shapes miss.
        full = getattr(module, name)(a, b)
        torch.cuda.synchronize()
        err = (full - reference).abs().max().item()
        c.add(
            f"{name} matches cuBLAS at 4096^3",
            err <= _tolerance(PERF_K),
            f"max abs difference {err:.3e}",
        )
        gf, ms = gflops(getattr(module, name), 1 if slow else 3, 3 if slow else 20)
        short = name.removeprefix("sgemm_")
        pct[short] = 100.0 * gf / cublas_gf
        c.metric(f"{short}_gflops", round(gf, 1))
        c.metric(f"{short}_pct_cublas", round(pct[short], 1))
        print(f"      {name}: {ms:.3f} ms", flush=True)
        c.add(
            f"{name} is not faster than the fp32 peak",
            gf / 1000.0 <= 1.1 * A100.FP32_TFLOPS,
            f"{gf / 1000.0:.2f} TFLOP/s against a {A100.FP32_TFLOPS} TFLOP/s peak. "
            "A figure above the peak means the timing or the result is wrong.",
        )
    if pct:
        c.metric("best_pct_cublas", round(max(pct.values()), 1))

    if "tiled" in pct:
        c.add(
            f"tiled reaches at least {MIN_TILED_PCT:.0f}% of cuBLAS",
            pct["tiled"] >= MIN_TILED_PCT,
            f"{pct['tiled']:.1f}%",
        )
    if "regblock" in pct:
        c.add(
            f"regblock reaches at least {MIN_REGBLOCK_PCT:.0f}% of cuBLAS",
            pct["regblock"] >= MIN_REGBLOCK_PCT,
            f"{pct['regblock']:.1f}%",
        )
    if "regblock" in pct and "tiled" in pct:
        ratio = pct["regblock"] / max(pct["tiled"], 1e-9)
        c.add(
            f"regblock is at least {MIN_REGBLOCK_OVER_TILED:.0f}x faster than tiled",
            ratio >= MIN_REGBLOCK_OVER_TILED,
            f"{ratio:.1f}x",
        )
    return c.finish()
