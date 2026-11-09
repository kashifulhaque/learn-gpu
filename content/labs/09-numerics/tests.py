import math

import torch
from lab_common import A100, Checks, bandwidth_gbs, cuda_time, require_cuda

# The mantissa bits m of each format. The unit roundoff is 2**-(m + 1).
MANTISSA_BITS = {"fp32": 23, "tf32": 10, "bf16": 7, "fp16": 10, "e4m3": 3, "e5m2": 2}

# Chapter 9's fused-multiply-add example, as a short vector. Length 8 keeps
# the check off the one-element kernel path.
_FMA_N = 8
_FMA_A = 1.0 + 2.0**-12
_FMA_C = -(1.0 + 2.0**-11)
_FMA_FUSED = 2.0**-24


def separate_in_float64(a: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A plausible wrong fma_versus_separate.

    It squares in float64 and then adds, and it returns that total for both
    paths. On the chapter's inputs the product 1 + 2**-11 + 2**-24 is exact
    in float64, so the separate path returns 2**-24. The float32 separate
    path returns 0.
    """
    total = (a.double() * a.double() + c.double()).to(dtype=torch.float32)
    return total, total


def _rel_err(value: float, exact: float, scale: float) -> float:
    """|value - exact| relative to `scale`, here the sum of absolute values."""
    return abs(value - exact) / scale


def run(submission):
    c = Checks()
    names = [
        "format_stats",
        "round_tf32",
        "roundtrip",
        "measured_roundoff",
        "fma_versus_separate",
        "sum_bf16_sequential",
        "sum_bf16_pairwise",
        "seq_sum_cuda",
        "kahan_sum_cuda",
        "tree_sum_cuda",
        "atomic_sum_cuda",
        "expf_max_ulp_error",
        "subnormal_survives",
    ]
    if not c.require(submission, *names):
        return c.finish()

    # ------------------------------------------------------------------
    # Part 1: format round trips
    # ------------------------------------------------------------------
    print("\nPart 1: format round trips", flush=True)

    def stats_match():
        problems = []
        cases = {
            "fp32": ((8, 23, False), torch.float32),
            "bf16": ((8, 7, False), torch.bfloat16),
            "fp16": ((5, 10, False), torch.float16),
            "e4m3": ((4, 3, True), torch.float8_e4m3fn),
            "e5m2": ((5, 2, False), torch.float8_e5m2),
        }
        for name, (spec, dtype) in cases.items():
            got = submission.format_stats(*spec)
            info = torch.finfo(dtype)
            for key, want in (("max", info.max), ("min_normal", info.smallest_normal), ("epsilon", info.eps)):
                if not math.isclose(got[key], want, rel_tol=1e-6):
                    problems.append(f"{name} {key}: got {got[key]:.6g}, want {want:.6g}")
            sub = got["min_subnormal"]
            held = torch.tensor(sub, dtype=torch.float64).to(dtype).double().item()
            half = torch.tensor(sub / 2, dtype=torch.float64).to(dtype).double().item()
            if held != sub:
                problems.append(f"{name} min_subnormal {sub:.6g} does not survive a cast")
            if half != 0.0:
                problems.append(f"{name} half of min_subnormal should round to zero")
        # TF32 has no torch dtype; compare with the published layout (1/8/10).
        tf32 = submission.format_stats(8, 10, False)
        if not math.isclose(tf32["max"], 3.4012e38, rel_tol=1e-4):
            problems.append(f"tf32 max: got {tf32['max']:.6g}, want 3.4012e38")
        if not math.isclose(tf32["min_subnormal"], 2.0**-136, rel_tol=1e-9):
            problems.append("tf32 min_subnormal should be 2**-136")
        return (not problems, "; ".join(problems) or "limits match torch.finfo for fp32, bf16, fp16, e4m3, e5m2")

    c.check("format_stats matches torch.finfo", stats_match)

    def tf32_rounds():
        one = 1.0
        tie = one + 2.0**-11  # exactly halfway between 1 and 1 + 2**-10
        x = torch.tensor([one, tie, -tie, one + 3 * 2.0**-11, 3.14159265], dtype=torch.float32)
        got = submission.round_tf32(x)
        want = torch.tensor([one, one + 2.0**-10, -(one + 2.0**-10), one + 2.0**-9, 3.140625], dtype=torch.float32)
        if got.dtype != torch.float32:
            return False, f"dtype {got.dtype}, expected float32"
        if not torch.equal(got, want):
            return False, f"got {got.tolist()}, want {want.tolist()}; ties go away from zero"
        g = torch.Generator().manual_seed(0)
        r = torch.randn(10000, generator=g) * 100.0
        out = submission.round_tf32(r)
        low_bits = (out.view(torch.int32) & 0x1FFF)
        if low_bits.any():
            return False, "the low 13 mantissa bits should be zero"
        rel = ((out - r).abs() / r.abs()).max().item()
        return rel <= 2.0**-11 * (1 + 1e-6), f"max relative error {rel:.3e} (bound 2^-11 = {2.0**-11:.3e})"

    c.check("round_tf32 keeps 10 mantissa bits and rounds ties away", tf32_rounds)

    def roundtrip_exact():
        g = torch.Generator().manual_seed(1)
        x = torch.randn(4097, generator=g) * 3.0
        problems = []
        if not torch.equal(submission.roundtrip(x, "fp32"), x):
            problems.append("fp32 should be the identity")
        if not torch.equal(submission.roundtrip(x, "bf16"), x.to(torch.bfloat16).float()):
            problems.append("bf16 differs from a bfloat16 cast")
        if not torch.equal(submission.roundtrip(x, "fp16"), x.to(torch.float16).float()):
            problems.append("fp16 differs from a float16 cast")
        if not torch.equal(submission.roundtrip(x, "tf32"), submission.round_tf32(x)):
            problems.append("tf32 should use round_tf32")
        for fmt in ("tf32", "bf16", "fp16", "e4m3", "e5m2"):
            once = submission.roundtrip(x, fmt)
            if not torch.equal(submission.roundtrip(once, fmt), once):
                problems.append(f"{fmt} is not idempotent: a value already on the grid moved")
        return (not problems, "; ".join(problems))

    c.check("roundtrip: exact formats, and every format is idempotent", roundtrip_exact)

    def fp8_saturates():
        x = torch.tensor([1000.0, -1000.0, 448.0, 449.0, 0.5, float("inf")])
        got = submission.roundtrip(x, "e4m3").tolist()
        want = [448.0, -448.0, 448.0, 448.0, 0.5, 448.0]
        if got != want:
            return False, f"e4m3 got {got}, want {want}"
        y = torch.tensor([1.0e5, -1.0e5, 57344.0, 3.0])
        got = submission.roundtrip(y, "e5m2").tolist()
        want = [57344.0, -57344.0, 57344.0, 3.0]
        return got == want, f"e5m2 got {got}, want {want}"

    c.check("roundtrip: fp8 saturates instead of producing NaN or infinity", fp8_saturates)

    def fp8_subnormals():
        # 2**-9 is E4M3's smallest subnormal; 2**-10 is a tie between it and 0.
        x = torch.tensor([2.0**-9, 2.0**-11, 3 * 2.0**-9])
        got = submission.roundtrip(x, "e4m3").tolist()
        want = [2.0**-9, 0.0, 3 * 2.0**-9]
        return got == want, f"got {got}, want {want}"

    c.check("roundtrip: fp8 subnormals are kept and tiny values flush to zero", fp8_subnormals)

    def roundoff_matches():
        problems = []
        for fmt, m in MANTISSA_BITS.items():
            measured = submission.measured_roundoff(fmt)
            bound = 2.0 ** -(m + 1)
            if fmt == "fp32":
                if measured != 0.0:
                    problems.append(f"fp32 should round-trip exactly, got {measured:.3e}")
                continue
            ratio = measured / bound
            if not 0.85 <= ratio <= 1.001:
                problems.append(f"{fmt}: measured {measured:.4e} is {ratio:.3f} x the bound {bound:.4e}")
        return (not problems, "; ".join(problems) or "each format's worst error sits just under 2^-(m+1)")

    c.check("measured_roundoff matches 2^-(m+1) for every format", roundoff_matches)
    for fmt in ("bf16", "e4m3"):
        try:
            c.metric(f"{fmt}_roundoff", round(float(submission.measured_roundoff(fmt)), 6))
        except Exception:  # noqa: BLE001 - reported by the check above
            pass

    # ------------------------------------------------------------------
    # Fused multiply-add. CPU only: the chapter's fp32 example.
    # separate_in_float64 fails the chapter example. Its separate path is
    # 2**-24, and the float32 separate path is 0.
    # ------------------------------------------------------------------
    print("\nFused multiply-add", flush=True)

    def _call_fma(a, c):
        try:
            separate, fused = submission.fma_versus_separate(a, c)
        except Exception as exc:  # noqa: BLE001 - an unfinished function fails the check
            detail = type(exc).__name__
            if str(exc):
                detail = f"{detail}: {exc}"
            return None, None, detail
        return separate, fused, ""

    def fma_chapter_example():
        value = torch.full((_FMA_N,), _FMA_A)
        addend = torch.full((_FMA_N,), _FMA_C)
        want_separate = torch.zeros(_FMA_N)
        want_fused = torch.full((_FMA_N,), _FMA_FUSED)
        wrong_separate, _wrong_fused = separate_in_float64(value, addend)
        separate, fused, error = _call_fma(value, addend)
        if error:
            return False, (
                f"expected separate {want_separate[0].item()} and fused {want_fused[0].item()}, "
                f"got {error}. separate_in_float64 returns separate {wrong_separate[0].item()}."
            )
        if not torch.is_tensor(separate) or not torch.is_tensor(fused):
            return False, (
                f"expected two float32 tensors, got {type(separate).__name__} and {type(fused).__name__}. "
                f"separate_in_float64 returns separate {wrong_separate[0].item()}."
            )
        if separate.dtype != torch.float32 or fused.dtype != torch.float32:
            return False, (
                f"expected dtype float32, got separate {separate.dtype} and fused {fused.dtype}"
            )
        if tuple(separate.shape) != (_FMA_N,) or tuple(fused.shape) != (_FMA_N,):
            return False, (
                f"expected shape ({_FMA_N},), got separate {tuple(separate.shape)} "
                f"and fused {tuple(fused.shape)}"
            )
        c.metric("fma_residual", float(fused[0].item()))
        ok = torch.equal(separate, want_separate) and torch.equal(fused, want_fused)
        return ok, (
            f"separate got {separate[0].item()}, expected {want_separate[0].item()}; "
            f"fused got {fused[0].item()}, expected {want_fused[0].item()}. "
            f"separate_in_float64 returns separate {wrong_separate[0].item()}."
        )

    c.check(
        "fma_versus_separate: the chapter example is 0 one way and 2^-24 the other",
        fma_chapter_example,
    )

    def fma_seeded_residual():
        generator = torch.Generator().manual_seed(9)
        a = (1.0 + torch.rand(4096, generator=generator)).float()
        product = a * a
        addend = -product
        exact = a.double() * a.double()
        want_fused = (exact - product.double()).float()
        want_separate = torch.zeros_like(a)
        bound = (2.0**-24) * exact.abs()
        separate, fused, error = _call_fma(a, addend)
        if error:
            return False, (
                f"expected separate max abs 0 and fused max abs error 0, got {error}"
            )
        if not torch.is_tensor(separate) or not torch.is_tensor(fused):
            return False, (
                f"expected two float32 tensors, got {type(separate).__name__} and {type(fused).__name__}"
            )
        if separate.dtype != torch.float32 or fused.dtype != torch.float32:
            return False, (
                f"expected dtype float32, got separate {separate.dtype} and fused {fused.dtype}"
            )
        if tuple(separate.shape) != tuple(a.shape) or tuple(fused.shape) != tuple(a.shape):
            return False, (
                f"expected shape {tuple(a.shape)}, got separate {tuple(separate.shape)} "
                f"and fused {tuple(fused.shape)}"
            )
        separate_err = (separate - want_separate).abs().max().item()
        fused_err = (fused - want_fused).abs().max().item()
        over = (fused.double().abs() - bound).max().item()
        nonzero = int((want_fused != 0).sum().item())
        ok = separate_err == 0.0 and fused_err == 0.0 and over <= 0.0 and nonzero > 0
        return ok, (
            f"separate max abs {separate_err}, expected 0; "
            f"fused max abs error {fused_err}, expected 0; "
            f"largest residual past 2^-24 times the exact product {over}, expected <= 0; "
            f"nonzero fused values {nonzero}, expected > 0"
        )

    c.check(
        "fma_versus_separate: a seeded cancellation stays inside the unit roundoff",
        fma_seeded_residual,
    )

    # ------------------------------------------------------------------
    # Part 2: accumulation order in bfloat16
    # ------------------------------------------------------------------
    print("\nPart 2: accumulation order in bfloat16", flush=True)

    def stall_exact():
        ones = torch.ones(2, 1024)
        seq = submission.sum_bf16_sequential(ones)
        if seq.dtype != torch.float32 or tuple(seq.shape) != (2,):
            return False, f"expected a float32 tensor of shape (2,), got {seq.dtype} {tuple(seq.shape)}"
        if seq.tolist() != [256.0, 256.0]:
            return False, f"1024 ones summed left to right gave {seq.tolist()}; 256 + 1 rounds back to 256"
        pair = submission.sum_bf16_pairwise(ones)
        return pair.tolist() == [1024.0, 1024.0], f"pairwise gave {pair.tolist()}, expected 1024"

    c.check("1024 ones: sequential stalls at 256, pairwise gets 1024", stall_exact)

    def odd_sizes():
        problems = []
        for n in (1, 2, 3, 7, 33, 1000):
            x = torch.arange(1, n + 1, dtype=torch.float32).repeat(3, 1) % 5 + 1  # small integers
            exact = x.double().sum(1)
            pair = submission.sum_bf16_pairwise(x).double()
            if (pair - exact).abs().max().item() > 0.01 * exact.max().item() + 1e-9:
                problems.append(f"n={n}: pairwise {pair.tolist()} vs exact {exact.tolist()}")
            if n <= 33 and n > 0:
                seq = submission.sum_bf16_sequential(x).double()
                if (seq - exact).abs().max().item() > 1e-9:
                    problems.append(f"n={n}: short sequential sums of small integers should be exact")
        return (not problems, "; ".join(problems))

    c.check("sizes that are not powers of two, including a single column", odd_sizes)

    def accuracy_gap():
        g = torch.Generator().manual_seed(2)
        x = torch.rand(4, 5120, generator=g)
        ref = x.to(torch.bfloat16).double().sum(1)
        seq = submission.sum_bf16_sequential(x).double()
        pair = submission.sum_bf16_pairwise(x).double()
        seq_err = ((seq - ref).abs() / ref).max().item()
        pair_err = ((pair - ref).abs() / ref).max().item()
        c.metric("bf16_seq_rel_err", round(seq_err, 4))
        c.metric("bf16_pairwise_rel_err", round(pair_err, 5))
        bound = math.log2(5120) * 2.0**-8  # the pairwise worst case, about 0.048
        ok = seq_err > 0.5 and pair_err < bound
        return ok, (
            f"sequential error {seq_err:.3f} (the sum stalls near 256 of ~2560), "
            f"pairwise {pair_err:.4f} (bound {bound:.3f})"
        )

    c.check("5120 uniform values: the sequential sum stalls, the pairwise sum doesn't", accuracy_gap)

    # ------------------------------------------------------------------
    # Part 3: summation kernels. Skipped, not failed, when this machine has
    # no CUDA device. The checks above are the CPU numerics.
    # ------------------------------------------------------------------
    if not torch.cuda.is_available():
        print(
            "\nNo CUDA device is visible, so the summation kernels and the "
            "--use_fast_math checks are skipped.",
            flush=True,
        )
        return c.finish()

    print("\nPart 3: summation kernels (the first call compiles with nvcc)", flush=True)
    require_cuda()
    dev = torch.device("cuda")

    def scaled_error(fn, x):
        exact = x.double().sum().item()
        scale = x.double().abs().sum().item()
        return _rel_err(float(fn(x).item()), exact, scale)

    for n in (1, 7, 1000, 1025, 4097, 1_000_003):
        g = torch.Generator().manual_seed(n)
        x = torch.randn(n, generator=g).to(dev)
        for label, fn, tol in (
            ("kahan_sum_cuda", submission.kahan_sum_cuda, 3e-7),
            ("tree_sum_cuda", submission.tree_sum_cuda, 2e-6),
            ("atomic_sum_cuda", submission.atomic_sum_cuda, 2e-6),
        ):
            def one(fn=fn, x=x, tol=tol):
                err = scaled_error(fn, x)
                return err <= tol, f"error / sum|x| = {err:.2e} (tolerance {tol:.0e})"

            c.check(f"{label} is correct for n={n}", one)

    # The accuracy demonstration: positive data, so the sum grows large next
    # to every addend and the left-to-right float32 sum loses low-order bits.
    n = 1 << 20
    g = torch.Generator().manual_seed(0)
    x = torch.rand(n, generator=g).to(dev)
    errors = {}
    for label, fn in (
        ("seq", submission.seq_sum_cuda),
        ("kahan", submission.kahan_sum_cuda),
        ("tree", submission.tree_sum_cuda),
    ):
        try:
            errors[label] = scaled_error(fn, x)
            c.metric(f"{label}_rel_err", float(f"{errors[label]:.3e}"))
        except Exception as exc:  # noqa: BLE001
            c.add(f"{label} sum of 2^20 values", False, f"{type(exc).__name__}: {exc}")
    if "kahan" in errors and "seq" in errors:
        c.add(
            "Kahan is more accurate than the plain sequential sum",
            errors["kahan"] <= errors["seq"],
            f"Kahan {errors['kahan']:.2e}, sequential {errors['seq']:.2e}",
        )
        # A plain sum of 2^20 values in [0, 1) is expected to be off by about
        # 1e-5 (the chapter derives 1.3e-5); Kahan by under 3e-7.
        c.add(
            "Kahan error is below 3e-7 on 2^20 positive values",
            errors["kahan"] <= 3e-7,
            f"{errors['kahan']:.2e}",
        )
    if "tree" in errors:
        c.add("the tree sum error is below 2e-6", errors["tree"] <= 2e-6, f"{errors['tree']:.2e}")

    # Determinism.
    g = torch.Generator().manual_seed(7)
    big = torch.randn(1 << 22, generator=g).to(dev)

    def tree_deterministic():
        values = {float(submission.tree_sum_cuda(big).item()) for _ in range(20)}
        return len(values) == 1, f"{len(values)} distinct result(s) in 20 runs"

    c.check("tree_sum_cuda gives the same bits on every run", tree_deterministic)

    try:
        distinct = {float(submission.atomic_sum_cuda(big).item()) for _ in range(50)}
        exact = big.double().sum().item()
        spread = max(abs(v - exact) for v in distinct)
        c.metric("atomic_distinct", len(distinct))
        c.add(
            "atomic_sum_cuda stays within float32 accuracy of the exact sum",
            spread <= 2e-6 * big.double().abs().sum().item(),
            f"{len(distinct)} distinct result(s) in 50 runs; worst error {spread:.3e}",
        )
        if len(distinct) == 1:
            print(
                "      note: all 50 atomic runs agreed. That is luck, not a guarantee: the "
                "order in which blocks reach atomicAdd is up to the scheduler.",
                flush=True,
            )
    except Exception as exc:  # noqa: BLE001
        c.add("atomic_sum_cuda runs", False, f"{type(exc).__name__}: {exc}")

    # Throughput of the tree sum. It reads 4 bytes per element once, so the
    # floor is a quarter of the measured copy ceiling, a deliberately loose bar.
    try:
        vec = torch.randn(1 << 24, generator=torch.Generator().manual_seed(3)).to(dev)
        ms = cuda_time(lambda: submission.tree_sum_cuda(vec), flush_l2=True)
        gbs = bandwidth_gbs(vec.numel() * 4, ms)
        c.metric("tree_gbs", round(gbs, 1))
        floor = 0.25 * A100.COPY_CEILING_GBS
        c.add(
            "tree_sum_cuda reads at least 25% of the copy ceiling",
            gbs >= floor,
            f"{gbs:.0f} GB/s against a floor of {floor:.0f} GB/s (0.25 x {A100.COPY_CEILING_GBS:.0f})",
        )
    except Exception as exc:  # noqa: BLE001
        c.add("tree_sum_cuda throughput", False, f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # Part 4: compiler flags
    # ------------------------------------------------------------------
    print("\nPart 4: --use_fast_math", flush=True)
    try:
        precise = float(submission.expf_max_ulp_error(False))
        fast = float(submission.expf_max_ulp_error(True))
        c.metric("expf_ulp", round(precise, 2))
        c.metric("fast_expf_ulp", round(fast, 2))
        # CUDA documents expf at 2 ulp. __expf, which --use_fast_math substitutes,
        # is documented at 2 + floor(|1.173 x|): 25 ulp at x = 20.
        c.add("expf is within 3 ulp of the float64 result", precise <= 3.0, f"{precise:.2f} ulp")
        c.add(
            "__expf stays within its documented bound",
            fast <= 2 + math.floor(1.173 * 20) + 1,
            f"{fast:.2f} ulp (bound 2 + floor(1.173 x 20) = 25, plus rounding of the reference)",
        )
    except Exception as exc:  # noqa: BLE001
        c.add("expf_max_ulp_error", False, f"{type(exc).__name__}: {exc}")

    try:
        c.add(
            "without --use_fast_math a subnormal times 1.0 stays nonzero",
            submission.subnormal_survives(False) is True,
        )
        flushed = not submission.subnormal_survives(True)
        c.metric("fast_math_flushes_subnormals", int(flushed))
        print(
            "      --use_fast_math implies -ftz=true, so a subnormal input is "
            + ("flushed to zero here." if flushed else "NOT flushed here; check the build flags."),
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        c.add("subnormal_survives", False, f"{type(exc).__name__}: {exc}")

    return c.finish()

