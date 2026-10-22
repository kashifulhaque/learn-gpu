"""Tests for lab 01: derive peaks from structure, then check them on the card.

Every formula check runs on a CPU. The matmul measurements need a GPU and are
skipped without one. Rated figures come from docs/FACTS.md, section 2.
"""

from lab_common import A100, Checks

# Rated dense peaks in TFLOP/s, from the NVIDIA datasheets and product pages
# (halved where the page prints sparse figures), and HGX B200 totals divided by 8.
RATED_TFLOPS = {
    ("A100", "bf16"): 312.0,
    ("A100", "fp16"): 312.0,
    ("A100", "tf32"): 156.0,
    ("A100", "fp32"): 19.5,
    ("H100", "bf16"): 989.4,
    ("H100", "tf32"): 494.7,
    ("H100", "fp8"): 1978.9,
    ("H100", "fp32"): 66.9,
    ("B200", "bf16"): 2250.0,
    ("B200", "tf32"): 1125.0,
    ("B200", "fp8"): 4500.0,
    ("B200", "fp4"): 9000.0,
    ("B200", "fp32"): 75.0,
}
RATED_GBS = {"A100": 1935.0, "H100": 3350.0, "B200": 8000.0}
# FLOP per SM per clock, bf16 tensor cores: 2,048 on A100, 4,096 on H100, 8,192 on B200.
BF16_PER_SM_CLOCK = {"A100": 2048, "H100": 4096, "B200": 8192}
# Ridge points in FLOP per byte (docs/FACTS.md, section 2.3).
RATED_RIDGE = {
    ("A100", "bf16"): 161.2,
    ("A100", "tf32"): 80.6,
    ("A100", "fp32"): 10.1,
    ("H100", "bf16"): 295.3,
    ("H100", "fp8"): 590.7,
    ("B200", "bf16"): 281.3,
    ("B200", "fp8"): 562.5,
    ("B200", "fp4"): 1125.0,
}


def within(actual, expected, rel):
    ok = abs(actual - expected) <= rel * abs(expected)
    return ok, f"got {actual:.4g}, expected {expected:.4g} (within {rel:.0%})"


def attempt(fn, *args, **kwargs):
    """Call a reader function, returning None when it raises (for example NotImplementedError)."""
    try:
        return fn(*args, **kwargs)
    except Exception:  # noqa: BLE001
        return None


def run(submission):
    c = Checks()
    names = [
        "SPECS", "flop_per_sm_clock", "peak_tflops", "hbm_bus_bits", "pin_rate_gbps", "ridge_point",
        "roofline_ms", "attainable_tflops", "gemm_intensity", "compare_gpus", "describe_local_device",
        "local_peak_tflops", "measure_matmul_tflops",
    ]
    if not c.require(submission, *names):
        return c.finish()
    specs = submission.SPECS

    # -- FLOP per SM per clock ------------------------------------------------
    def per_clock():
        for gpu, expected in BF16_PER_SM_CLOCK.items():
            got = submission.flop_per_sm_clock(specs[gpu], "bf16")
            if got != expected:
                return False, f"{gpu} bf16: got {got}, expected {expected}"
        a, h, b = specs["A100"], specs["H100"], specs["B200"]
        checks = [
            (a, "tf32", 1024), (a, "fp32", 128), (h, "fp8", 8192), (h, "tf32", 2048),
            (h, "fp32", 256), (b, "fp8", 16384), (b, "fp4", 32768), (b, "fp32", 256),
        ]
        for spec, precision, expected in checks:
            got = submission.flop_per_sm_clock(spec, precision)
            if got != expected:
                return False, f"{spec.name} {precision}: got {got}, expected {expected}"
        return True, "2048, 4096, and 8192 bf16 FLOP per SM per clock"

    c.check("flop_per_sm_clock matches the tensor-core and CUDA-core widths", per_clock)

    def unsupported():
        for gpu, precision in (("A100", "fp8"), ("A100", "fp4"), ("H100", "fp4")):
            try:
                submission.flop_per_sm_clock(specs[gpu], precision)
            except ValueError:
                continue
            return False, f"{gpu} has no {precision} path, but no ValueError was raised"
        return True, ""

    c.check("a precision the GPU lacks raises ValueError", unsupported)

    # -- Peak FLOP/s ----------------------------------------------------------
    def peaks():
        worst = 0.0
        for (gpu, precision), rated in RATED_TFLOPS.items():
            got = submission.peak_tflops(specs[gpu], precision)
            error = abs(got - rated) / rated
            worst = max(worst, error)
            if error > 0.01:
                return False, f"{gpu} {precision}: got {got:.1f} TFLOP/s, rated {rated}"
        return True, f"all 13 peaks within 1% of the rating (worst {worst:.2%})"

    c.check("peak_tflops reproduces the rated peaks of the A100, H100, and B200", peaks)
    c.check(
        "A100 bf16 peak is 108 x 2048 x 1.41 GHz = 311.9 TFLOP/s",
        lambda: within(submission.peak_tflops(specs["A100"], "bf16"), 311.9, 0.002),
    )
    c.check(
        "sm_count overrides the SM count",
        lambda: within(submission.peak_tflops(specs["A100"], "bf16", sm_count=54), 311.9 / 2, 0.002),
    )

    # -- Bandwidth ------------------------------------------------------------
    def bus():
        got = {gpu: submission.hbm_bus_bits(specs[gpu]) for gpu in specs}
        return got == {"A100": 5120, "H100": 5120, "B200": 8192}, f"got {got}"

    c.check("hbm_bus_bits is stacks x 1024 (5120, 5120, 8192)", bus)

    def pins():
        for gpu in specs:
            rate = submission.pin_rate_gbps(specs[gpu])
            rebuilt = submission.hbm_bus_bits(specs[gpu]) * rate / 8
            if abs(rebuilt - RATED_GBS[gpu]) > 1e-6 * RATED_GBS[gpu]:
                return False, f"{gpu}: bus x pin rate / 8 gives {rebuilt:.1f} GB/s, not {RATED_GBS[gpu]}"
        a = submission.pin_rate_gbps(specs["A100"])
        return abs(a - 3.0234) < 0.001, f"A100 pin rate {a:.4f} Gb/s, expected 3.0234"

    c.check("pin_rate_gbps inverts GB/s = bits x Gb/s / 8", pins)

    # -- Ridge points and the roofline ---------------------------------------
    def ridges():
        for (gpu, precision), rated in RATED_RIDGE.items():
            got = submission.ridge_point(specs[gpu], precision)
            if abs(got - rated) > 0.01 * rated:
                return False, f"{gpu} {precision}: got {got:.1f} FLOP/byte, expected {rated}"
        return True, "A100 bf16 161, H100 bf16 295, B200 bf16 281 FLOP/byte"

    c.check("ridge_point matches the table in docs/FACTS.md", ridges)
    c.check(
        "ridge_point against the measured 1275 GB/s copy is 244.7 for the A100",
        lambda: within(submission.ridge_point(specs["A100"], "bf16", bandwidth_gbs=A100.COPY_CEILING_GBS), 244.7, 0.01),
    )

    def vector_add():
        n = 2**24
        ms, limit = submission.roofline_ms(specs["A100"], "fp32", flop=n, nbytes=12 * n)
        expected = 12 * n / 1935e9 * 1e3
        ok = limit == "memory" and abs(ms - expected) < 0.01 * expected
        return ok, f"{ms:.4f} ms, limited by {limit}; expected {expected:.4f} ms, memory"

    c.check("a vector add of 2^24 floats is memory bound at 0.104 ms", vector_add)

    def big_gemm():
        n = 8192
        ms, limit = submission.roofline_ms(specs["A100"], "bf16", flop=2 * n**3, nbytes=3 * n * n * 2)
        expected = 2 * n**3 / 312e12 * 1e3
        ok = limit == "compute" and abs(ms - expected) < 0.01 * expected
        return ok, f"{ms:.3f} ms, limited by {limit}; expected {expected:.3f} ms, compute"

    c.check("an 8192-cubed bf16 GEMM is compute bound at 3.5 ms", big_gemm)

    def attainable():
        a = specs["A100"]
        low = submission.attainable_tflops(a, "bf16", 1.0)
        high = submission.attainable_tflops(a, "bf16", 1000.0)
        ok = abs(low - 1.935) < 0.001 and abs(high - 311.9) < 0.5
        return ok, f"intensity 1 gives {low:.3f}, intensity 1000 gives {high:.1f} TFLOP/s"

    c.check("attainable_tflops follows the roofline (1.935 at 1 FLOP/byte, the peak at 1000)", attainable)

    def intensity():
        big = submission.gemm_intensity(4096, 4096, 4096, 2)
        decode = submission.gemm_intensity(1, 8192, 8192, 2)
        ok = abs(big - 4096 / 3) < 0.01 and abs(decode - 0.999) < 0.002
        return ok, f"4096^3 bf16 gives {big:.1f}, one token against an 8192 x 8192 weight gives {decode:.3f}"

    c.check("gemm_intensity gives n/3 for a cube and about 1 for one token", intensity)

    def table():
        t = submission.compare_gpus("bf16")
        t4 = submission.compare_gpus("fp4")
        ok = set(t) == {"A100", "H100", "B200"} and set(t4) == {"B200"}
        ok = ok and all(set(row) == {"tflops", "gbs", "ridge"} for row in t.values())
        if ok:
            print("      GPU    bf16 TFLOP/s   GB/s   ridge (FLOP/byte)", flush=True)
            for gpu, row in t.items():
                print(f"      {gpu:5s}  {row['tflops']:11.1f}  {row['gbs']:6.0f}  {row['ridge']:8.1f}", flush=True)
        return ok, "A100, H100, B200 for bf16; only B200 for fp4"

    c.check("compare_gpus returns one row per GPU that supports the precision", table)

    # Metrics that need no GPU.
    a100_peak = attempt(submission.peak_tflops, specs["A100"], "bf16")
    if a100_peak is not None:
        c.metric("a100_bf16_peak_tflops", round(a100_peak, 1))
    for gpu in ("A100", "H100", "B200"):
        ridge = attempt(submission.ridge_point, specs[gpu], "bf16")
        if ridge is not None:
            c.metric(f"{gpu.lower()}_bf16_ridge", round(ridge, 1))

    # -- The card in front of you --------------------------------------------
    try:
        import torch

        has_gpu = torch.cuda.is_available()
    except ImportError:  # the formula checks above need no torch
        has_gpu = False
    if not has_gpu:
        print("      No GPU is visible, so the device and matmul checks are skipped.", flush=True)
        return c.finish()

    props = torch.cuda.get_device_properties(0)
    info = attempt(submission.describe_local_device)
    if c.check("describe_local_device returns a dict", lambda: isinstance(info, dict)):
        c.add("name matches the device", info.get("name") == props.name, f"got {info.get('name')!r}")
        c.add(
            "sm_count matches the device",
            info.get("sm_count") == props.multi_processor_count,
            f"got {info.get('sm_count')}, the device has {props.multi_processor_count}",
        )
        c.add(
            "capability is (major, minor)",
            tuple(info.get("capability", ())) == (props.major, props.minor),
            f"got {info.get('capability')}",
        )
        l2 = getattr(props, "L2_cache_size", None)
        if l2 is not None:
            c.check("l2_mib is the L2 size in MiB", lambda: within(info.get("l2_mib") or 0.0, l2 / 2**20, 0.001))
        print(f"      device = {props.name}, spec = {info.get('spec')}", flush=True)

    is_a100 = (props.major, props.minor) == (8, 0)
    on_course_card = is_a100 and props.multi_processor_count == A100.SM_COUNT
    if is_a100:
        c.check(
            "local_peak_tflops(bf16) matches the A100 constant in lab_common",
            lambda: within(submission.local_peak_tflops("bf16") or 0.0, A100.TENSOR_BF16_TFLOPS, 0.01),
        )
    else:
        print(
            f"      note: this is not an sm_80 card, so the matmul checks below only run "
            f"when a peak is known for it.",
            flush=True,
        )

    for precision in ("bf16", "fp32"):
        peak = attempt(submission.local_peak_tflops, precision)
        measured = attempt(submission.measure_matmul_tflops, precision)
        if measured is None:
            c.add(f"measure_matmul_tflops({precision!r}) runs", False, "it raised or is not implemented yet")
            continue
        c.metric(f"matmul_{precision}_tflops", round(measured, 1))
        if peak is None:
            continue
        fraction = measured / peak
        c.metric(f"matmul_{precision}_fraction", round(fraction, 3))
        if on_course_card:
            # The roofline floor says no kernel exceeds the peak, so a figure
            # above it means the timing or the FLOP count is wrong. 30% is a loose
            # floor: real GEMMs reach a large share of the peak, not all of it.
            c.check(
                f"a {precision} matmul stays below the derived peak",
                lambda: (fraction <= 1.10, f"{measured:.1f} of {peak:.1f} TFLOP/s ({fraction:.0%})"),
            )
            c.check(
                f"a {precision} matmul reaches at least 30% of the derived peak",
                lambda: (fraction >= 0.30, f"{measured:.1f} of {peak:.1f} TFLOP/s ({fraction:.0%})"),
            )
    return c.finish()
