"""Tests for lab 33.

Every check but the last two runs on a CPU. The Triton block-scaled GEMM
needs CUDA and Triton, and the native `torch._scaled_mm` timing needs
compute capability 8.9 or higher, so the A100 skips it.

Plausible wrong implementations the checks catch:

* round_half_up: `floor(x / gap + 0.5)` instead of ties-to-even. The
  tie 0.75 still lands on 1.0 by luck, but 0.25 becomes 0.5, 1.25 becomes
  1.5, and the fp8 tie checks fail on half of all midpoints.
* ceil_scale: an E8M0 exponent from `ceil(log2(amax))` instead of floor.
  The block [0.3, -1.7, 5.2, 0.05] must have exponent 0, and the ceil gives 1.
* rne_as_stochastic: "stochastic" rounding that rounds to nearest. The mean
  of 20,000 roundings of 0.3 is 0.5, not 0.3.
* no_reset: a limited accumulator that never restarts after a promotion. The
  fp32 total then counts every earlier chunk again, so the result is far
  too large.
"""

from __future__ import annotations

import math

from lab_common import Checks

NAMES = (
    "e2m1_decode",
    "round_minifloat",
    "e2m1_encode",
    "decode_fp8",
    "encode_fp8",
    "e8m0_exponent",
    "mxfp4_quantize",
    "mxfp4_dequantize",
    "nvfp4_quantize",
    "nvfp4_dequantize",
    "stochastic_round_e2m1",
    "fp8_scale",
    "quantize_with_scale",
    "block_shape",
    "fp8_fake_quant",
    "block_scaled_gemm",
    "round_to_bits",
    "gemm_limited_accumulator",
    "triton_block_scaled_gemm",
)

E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

# (input, expected code). Ties go to the even code, and 7 and -6.5 saturate.
E2M1_CASES = (
    (0.25, 0), (0.75, 2), (1.25, 2), (1.75, 4), (2.5, 4), (3.5, 6), (5.0, 6),
    (0.3, 1), (0.8, 2), (7.0, 7), (-0.25, 8), (-0.1, 8), (-0.75, 10), (-6.5, 15),
)


def _fail(want, got):
    return False, f"expected {want}, actual {got}"


def _torch():
    import torch

    return torch


def _skip_reason():
    try:
        import torch
    except ImportError:
        return "SKIP: torch is not installed, so the Triton kernel was not run."
    if not torch.cuda.is_available():
        return "SKIP: no CUDA GPU is visible, so the Triton block-scaled GEMM was not run."
    try:
        import triton  # noqa: F401
    except ImportError:
        return "SKIP: Triton is not installed, so the Triton block-scaled GEMM was not run."
    return None


def _same(a, b):
    torch = _torch()
    return bool(((a == b) | (torch.isnan(a) & torch.isnan(b))).all())


def run(submission):
    c = Checks()
    if not c.require(submission, *NAMES):
        return c.finish()
    torch = _torch()
    S = submission

    # Check 1 calls only e2m1_decode.
    def decode_e2m1():
        got = S.e2m1_decode(torch.arange(16, dtype=torch.uint8)).tolist()
        want = list(E2M1_VALUES) + [-v for v in E2M1_VALUES]
        for code, (g, w) in enumerate(zip(got, want)):
            if g != w or math.copysign(1, g) != math.copysign(1, w):
                return _fail(f"{w} for code {code}", g)
        return True, "expected 0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives, actual the same"

    c.check("E2M1 decodes all 16 codes", decode_e2m1)

    def encode_e2m1():
        # round_half_up fails here: 0.25 becomes code 1 and 1.25 becomes code 3.
        x = torch.tensor([v for v, _ in E2M1_CASES], dtype=torch.float32)
        got = S.e2m1_encode(x).tolist()
        for (v, want), g in zip(E2M1_CASES, got):
            if g != want:
                return _fail(f"code {want} for {v}", f"code {g}")
        every = S.e2m1_encode(S.e2m1_decode(torch.arange(16, dtype=torch.uint8))).tolist()
        if every != list(range(16)):
            return _fail(list(range(16)), every)
        return True, "expected ties to even, saturation at 6, and all 16 codes to round-trip, actual the same"

    c.check("E2M1 rounds ties to even and saturates at 6", encode_e2m1)

    def decode_fp8_all():
        codes = torch.arange(256, dtype=torch.uint8)
        for fmt, dt in (("e4m3", torch.float8_e4m3fn), ("e5m2", torch.float8_e5m2)):
            want = codes.view(dt).to(torch.float32)
            got = S.decode_fp8(codes, fmt).to(torch.float32)
            if not _same(want, got):
                bad = int(((want != got) & ~(torch.isnan(want) & torch.isnan(got))).nonzero()[0])
                return _fail(f"{fmt} code {bad:#04x} -> {want[bad].item()}", got[bad].item())
        return True, "expected all 256 codes of e4m3 and e5m2 to equal torch, actual equal"

    c.check("decode_fp8 equals torch on all 256 codes of E4M3 and E5M2", decode_fp8_all)

    def encode_fp8_exact():
        # round_half_up fails on the midpoints below.
        for fmt, dt in (("e4m3", torch.float8_e4m3fn), ("e5m2", torch.float8_e5m2)):
            codes = torch.arange(256, dtype=torch.uint8)
            vals = codes.view(dt).to(torch.float32)
            finite = vals[torch.isfinite(vals)]
            pos = torch.sort(finite[finite >= 0]).values.unique()
            mids = (pos[1:] + pos[:-1]) / 2  # exact ties, subnormals included
            gen = torch.Generator().manual_seed(33)
            spread = torch.randn(50000, generator=gen) * torch.exp(torch.randn(50000, generator=gen) * 4)
            extra = torch.tensor([float("inf"), -float("inf"), float("nan"), 0.0, -0.0, 464.0, 464.5, 61440.0, 1e30])
            groups = {
                "representable": finite,
                "tie midpoints": torch.cat([mids, -mids]),
                "just off a tie": torch.cat([mids * (1 + 2**-20), mids * (1 - 2**-20)]),
                "random, overflow, inf, NaN": torch.cat([spread, extra]),
            }
            mx_f = float(torch.finfo(dt).max)
            # Overflow is checked against the format rule, not torch: torch 2.11 casts an
            # e4m3 overflow to NaN, and torch 2.15 saturates it to 448.
            edge = 464.0 if fmt == "e4m3" else 61440.0  # the tie above max, which rounds down for e4m3 and up for e5m2
            top = 0x7E if fmt == "e4m3" else 0x7B
            over = 0x7F if fmt == "e4m3" else 0x7C
            for name, x in groups.items():
                big = x.abs() > mx_f
                want = torch.where(big, torch.zeros_like(x), x).to(dt).view(torch.uint8)
                rounds_down = (x.abs() < edge) | ((x.abs() == edge) & (fmt == "e4m3"))
                spec = torch.where(rounds_down, torch.full_like(want, top), torch.full_like(want, over))
                spec = spec | (torch.signbit(x).to(torch.uint8) << 7)
                want = torch.where(big, spec, want)
                got = S.encode_fp8(x, fmt).to(torch.uint8)
                nan = torch.isnan(x)
                bad = ((want != got) & ~nan).nonzero()
                if bad.numel():
                    i = int(bad[0])
                    return _fail(f"{fmt} {name}: {x[i].item()!r} -> {int(want[i]):#04x}", f"{int(got[i]):#04x}")
                if nan.any() and not bool((got[nan] & 0x7F == 0x7F).all()):
                    return _fail(f"{fmt} NaN to a NaN code", got[nan].tolist())
            mx = S.FP8[fmt]["max"] if hasattr(S, "FP8") else torch.finfo(dt).max
            x = torch.cat([spread, extra[:2], extra[3:]])
            ref = torch.where(torch.isnan(x), x, x.clamp(-mx, mx)).to(dt).view(torch.uint8)
            got = S.encode_fp8(x, fmt, saturate=True).to(torch.uint8)
            bad = ((ref != got) & ~torch.isnan(x)).nonzero()
            if bad.numel():
                i = int(bad[0])
                return _fail(f"{fmt} saturate: {x[i].item()!r} -> {int(ref[i]):#04x}", f"{int(got[i]):#04x}")
        return True, "expected every code, tie, and subnormal to equal torch, and overflow to follow the format rule, actual equal"

    c.check("encode_fp8 is bit-exact against torch casts, with and without saturation", encode_fp8_exact)

    def mxfp4_checks():
        # ceil_scale fails here: amax 5.2 has floor(log2) = 2, so the exponent is 0.
        cases = ((5.2, 0), (4.0, 0), (7.9, 0), (8.0, 1), (0.3, -4), (1000.0, 7), (0.0, -127))
        for amax, want in cases:
            got = int(S.e8m0_exponent(torch.tensor(amax), 2))
            if got != want:
                return _fail(f"exponent {want} for amax {amax}", got)
        block = torch.zeros(32)
        block[:4] = torch.tensor([0.3, -1.7, 5.2, 0.05])
        codes, scale = S.mxfp4_quantize(block)
        if scale.dtype != torch.uint8 or int(scale.reshape(-1)[0]) != 127:
            return _fail("E8M0 byte 127 (scale 1)", scale.tolist())
        deq = S.mxfp4_dequantize(codes, scale).reshape(-1)[:4].tolist()
        if deq != [0.5, -1.5, 6.0, 0.0]:
            return _fail([0.5, -1.5, 6.0, 0.0], deq)
        gen = torch.Generator().manual_seed(1)
        x = torch.randn(64, 128, generator=gen) * torch.exp(torch.randn(64, 4, 1, generator=gen).repeat(1, 1, 32).reshape(64, 128) * 2)
        codes, scale = S.mxfp4_quantize(x)
        deq = S.mxfp4_dequantize(codes, scale)
        blocks = x.reshape(64, 4, 32)
        amax = blocks.abs().amax(-1, keepdim=True)
        err = (x - deq).reshape(64, 4, 32).abs().amax(-1, keepdim=True)
        ratio = (err / amax).max().item()
        if ratio > 0.25 + 1e-6:
            return _fail("every block error at most 0.25 of the block amax", f"{ratio:.4f}")
        try:
            S.mxfp4_quantize(torch.zeros(2, 33))
        except ValueError:
            return True, f"expected exponent floor(log2(amax)) - 2 and error at most 0.25 amax, actual worst {ratio:.4f}"
        return _fail("ValueError for a last dimension of 33", "no error")

    c.check("MXFP4 uses a floor exponent and stays within 0.25 of the block amax", mxfp4_checks)

    def nvfp4_checks():
        # Data that sits on the grid after scaling must round-trip exactly.
        grid = torch.tensor(E2M1_VALUES)
        b0 = torch.cat([grid, -grid.flip(0)]) * 1.0
        b1 = torch.cat([grid, -grid.flip(0)]) * 0.5
        x = torch.stack([b0, b1]).reshape(1, 32)
        codes, sc, ts = S.nvfp4_quantize(x)
        deq = S.nvfp4_dequantize(codes, sc, ts)
        if not torch.allclose(deq, x, rtol=1e-6, atol=1e-7):
            return _fail(x.tolist(), deq.tolist())
        if abs(ts - 6.0 / (448 * 6)) > 1e-9:
            return _fail(f"tensor scale {6.0 / 2688:.6g}", f"{ts:.6g}")
        gen = torch.Generator().manual_seed(2)
        x = torch.randn(256, 128, generator=gen)
        codes, sc, ts = S.nvfp4_quantize(x)
        deq = S.nvfp4_dequantize(codes, sc, ts)
        vals = torch.as_tensor(sc).to(torch.uint8)
        e4 = vals.view(torch.float8_e4m3fn).float()
        if not bool(torch.isfinite(e4).all()) or float(e4.max()) > 448:
            return _fail("block scales that are finite E4M3 values at most 448", float(e4.max()))
        amax = x.reshape(256, 8, 16).abs().amax(-1, keepdim=True)
        err = (x - deq).reshape(256, 8, 16).abs().amax(-1, keepdim=True)
        ratio = (err / amax).max().item()
        if ratio > 0.25:
            return _fail("every block error at most 0.25 of the block amax", f"{ratio:.4f}")
        mx_codes, mx_scale = S.mxfp4_quantize(x)
        mx_mse = ((x - S.mxfp4_dequantize(mx_codes, mx_scale)) ** 2).mean().item()
        nv_mse = ((x - deq) ** 2).mean().item()
        c.metric("mxfp4_mse", float(f"{mx_mse:.4g}"))
        c.metric("nvfp4_mse", float(f"{nv_mse:.4g}"))
        if not nv_mse < mx_mse:
            return _fail(f"NVFP4 MSE below MXFP4's {mx_mse:.4g}", f"{nv_mse:.4g}")
        return True, f"expected NVFP4 MSE below MXFP4's, actual {nv_mse:.4g} against {mx_mse:.4g}"

    c.check("NVFP4 round-trips grid data exactly and beats MXFP4 on Gaussian data", nvfp4_checks)

    def stochastic():
        # rne_as_stochastic fails here: the mean of 0.3 would be 0.5.
        n = 20000
        gen = torch.Generator().manual_seed(7)
        grid = list(E2M1_VALUES)
        for v in (0.3, 2.2, 3.7, -1.1, 5.0):
            out = S.stochastic_round_e2m1(torch.full((n,), v), gen).to(torch.float64)
            a = abs(v)
            lo = max(g for g in grid if g <= a)
            hi = min(g for g in grid if g >= a)
            allowed = {math.copysign(lo, v), math.copysign(hi, v)}
            if not set(out.unique().tolist()) <= allowed:
                return _fail(f"only {sorted(allowed)} for {v}", sorted(set(out.unique().tolist())))
            p = 0.0 if hi == lo else (a - lo) / (hi - lo)
            sigma = math.sqrt(p * (1 - p)) * (hi - lo) / math.sqrt(n)
            mean = out.mean().item()
            if abs(mean - v) > 5 * sigma + 1e-12:
                return _fail(f"mean within {5 * sigma:.4f} of {v}", f"{mean:.4f}")
        x = torch.linspace(-7, 7, 101)
        a = S.stochastic_round_e2m1(x, torch.Generator().manual_seed(9))
        b = S.stochastic_round_e2m1(x, torch.Generator().manual_seed(9))
        if not torch.equal(a, b):
            return _fail("the same output for the same seed", "outputs differ")
        if float(a.abs().max()) > 6:
            return _fail("saturation at 6", float(a.abs().max()))
        return True, "expected unbiased means within 5 sigma and only the two neighbours, actual so"

    c.check("stochastic rounding to E2M1 is unbiased and seeded", stochastic)

    def scaling_modes():
        history = [1.0, 0.9, 1.0]
        cur = S.fp8_scale(4.0, history, "current")
        dly = S.fp8_scale(4.0, history, "delayed")
        if abs(cur - 4.0 / 448) > 1e-12 or abs(dly - 1.0 / 448) > 1e-12:
            return _fail(f"current {4.0 / 448:.6g} and delayed {1.0 / 448:.6g}", f"{cur:.6g} and {dly:.6g}")
        if S.fp8_scale(2.0, [], "delayed") != 2.0 / 448 or S.fp8_scale(0.0, [], "current") != 1.0:
            return _fail("an empty history to fall back to the current amax, and a zero amax to give 1.0", "different")
        x = torch.tensor([4.0, 0.5, -3.0])
        now = S.quantize_with_scale(x, cur).tolist()
        late = S.quantize_with_scale(x, dly).tolist()
        if late != [1.0, 0.5, -1.0]:
            return _fail("delayed scale to saturate to [1.0, 0.5, -1.0]", late)
        if max(abs(a - b) for a, b in zip(now, x.tolist())) > 0.25:
            return _fail("current scale to keep every error under 0.25", now)
        return True, "expected the delayed scale to clip 4.0 to 1.0 and the current scale to keep it, actual so"

    c.check("delayed scaling saturates when the tensor grows and current scaling doesn't", scaling_modes)

    def gemm():
        gen = torch.Generator().manual_seed(0)
        m, k, n = 70, 300, 130  # none is a multiple of 128
        a = torch.randn(m, k, generator=gen)
        a[[3, 17]] *= 1e5  # two outlier tokens
        b = torch.randn(k, n, generator=gen)
        b[torch.rand(k, n, generator=gen) < 0.001] *= 100
        ref = a.double() @ b.double()
        good = torch.ones(m, dtype=torch.bool)
        good[[3, 17]] = False

        def rel(y):
            e = (y.double() - ref).norm(dim=1) / ref.norm(dim=1)
            return e[good].mean().item()

        for gran in ("tensor", "row", "col", "tile1x128", "block128x128"):
            shape = tuple(S.block_shape(gran, (m, k)))
            q = S.fp8_fake_quant(a, gran)
            if tuple(q.shape) != (m, k) or q.dtype != torch.float32:
                return _fail(f"shape {(m, k)} and float32 for {gran}", (tuple(q.shape), str(q.dtype), shape))
        rows = S.fp8_fake_quant(torch.tensor([[1.0, 2.0, 3.0], [100.0, 200.0, 300.0]]), "row")
        if not torch.allclose(rows[0], torch.tensor([1.0, 2.0, 3.0]), rtol=0.07) or not torch.allclose(
            rows[1], torch.tensor([100.0, 200.0, 300.0]), rtol=0.07
        ):
            return _fail("each row to keep its own scale within 7%", rows.tolist())
        e_tensor = rel(S.block_scaled_gemm(a, b, "tensor", "tensor"))
        e_block = rel(S.block_scaled_gemm(a, b))
        c.metric("gemm_rel_err_tensor", round(e_tensor, 4))
        c.metric("gemm_rel_err_block", round(e_block, 4))
        if e_block > 0.05:
            return _fail("1x128 and 128x128 relative error at most 0.05", f"{e_block:.4f}")
        if not e_block < e_tensor / 4:
            return _fail(f"block error under a quarter of the per-tensor error {e_tensor:.4f}", f"{e_block:.4f}")
        return True, f"expected block scaling under 0.05 and 4x better than per-tensor, actual {e_block:.4f} against {e_tensor:.4f}"

    c.check("block scaling beats per-tensor scaling on outlier rows, at awkward sizes", gemm)

    def accumulator():
        # no_reset fails here: without the restart the error equals the no-promotion error.
        gen = torch.Generator().manual_seed(2)
        k = 8192
        # Positive values in [0.5, 1.5), already on the E4M3 grid, so no scale is needed.
        a = (torch.rand(16, k, generator=gen) + 0.5).to(torch.float8_e4m3fn).to(torch.float32)
        b = (torch.rand(k, 16, generator=gen) + 0.5).to(torch.float8_e4m3fn).to(torch.float32)
        ref = a.double() @ b.double()
        scale = ref.abs().max()

        def err(nc):
            y = S.gemm_limited_accumulator(a, b, n_c=nc)
            return ((y.double() - ref).abs().max() / scale).item()

        e128, ek = err(128), err(k)
        c.metric("acc_err_nc128", float(f"{e128:.3g}"))
        c.metric("acc_err_nc_k", float(f"{ek:.3g}"))
        if not ek > 2e-4:
            return _fail("a narrow accumulator over K = 8192 to err by more than 2e-4", f"{ek:.3g}")
        if not e128 < ek / 10:
            return _fail(f"promotion every 128 to cut the error 10x below {ek:.3g}", f"{e128:.3g}")
        wide = S.gemm_limited_accumulator(a, b, n_c=128, acc_bits=40)
        if ((wide.double() - ref).abs().max() / scale).item() > 1e-6:
            return _fail("a 40-bit accumulator to be nearly exact", "error above 1e-6")
        return True, f"expected promotion at 128 to cut the error 10x, actual {e128:.3g} against {ek:.3g}"

    c.check("fp32 promotion every 128 elements cuts the narrow-accumulator error", accumulator)

    reason = _skip_reason()
    if reason is not None:
        print(reason, flush=True)
        return c.finish()

    from lab_common import cuda_time, require_cuda

    device = require_cuda()

    def triton_gemm():
        gen = torch.Generator().manual_seed(5)
        m, k, n = 200, 640, 192
        a = torch.randn(m, k, generator=gen)
        b = torch.randn(k, n, generator=gen)
        a[5] *= 1e3
        kb = (k + 127) // 128
        nb = (n + 127) // 128
        a_pad = torch.nn.functional.pad(a, (0, kb * 128 - k)).reshape(m, kb, 128)
        sa = torch.where(a_pad.abs().amax(-1) > 0, a_pad.abs().amax(-1) / 448, torch.ones(m, kb))
        aq = S.quantize_with_scale(a_pad, sa.unsqueeze(-1)).reshape(m, kb * 128)[:, :k]
        b_pad = torch.nn.functional.pad(b, (0, nb * 128 - n, 0, kb * 128 - k)).reshape(kb, 128, nb, 128)
        sb = b_pad.abs().amax(dim=(1, 3))
        sb = torch.where(sb > 0, sb / 448, torch.ones_like(sb))
        bq = S.quantize_with_scale(b_pad, sb[:, None, :, None]).reshape(kb * 128, nb * 128)[:k, :n]
        cpu = S.block_scaled_gemm(a, b)
        got = S.triton_block_scaled_gemm(
            (aq / sa.repeat_interleave(128, 1)[:, :k]).half().to(device).contiguous(),
            sa.to(device),
            (bq / sb.repeat_interleave(128, 0).repeat_interleave(128, 1)[:k, :n]).half().to(device).contiguous(),
            sb.to(device),
        ).cpu()
        err = ((got.double() - cpu.double()).abs().max() / cpu.double().abs().max()).item()
        c.metric("triton_max_rel_err", float(f"{err:.3g}"))
        if not err < 1e-4:
            return _fail("the Triton result within 1e-4 of the CPU emulation", f"{err:.3g}")
        aq16 = (aq / sa.repeat_interleave(128, 1)[:, :k]).half().to(device).contiguous()
        bq16 = (bq / sb.repeat_interleave(128, 0).repeat_interleave(128, 1)[:k, :n]).half().to(device).contiguous()
        sa_d, sb_d = sa.to(device), sb.to(device)
        ms = cuda_time(lambda: S.triton_block_scaled_gemm(aq16, sa_d, bq16, sb_d), warmup=3, iters=10)
        c.metric("triton_ms", round(ms, 4))
        return True, f"expected a relative error under 1e-4, actual {err:.3g}"

    c.check("the Triton block-scaled GEMM matches the CPU emulation on the GPU", triton_gemm)

    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < (8, 9):
        print(
            f"SKIP: torch._scaled_mm needs compute capability 8.9 or higher, and this GPU is sm_{major}{minor}. "
            "The metric scaled_mm_max_rel_err is recorded only on an H100.",
            flush=True,
        )
    else:
        try:
            a = torch.randn(128, 256, device=device).to(torch.float8_e4m3fn)
            bt = torch.randn(64, 256, device=device).to(torch.float8_e4m3fn)
            one = torch.tensor(1.0, device=device)
            out = torch._scaled_mm(a, bt.t(), one, one, out_dtype=torch.bfloat16)
            ref = a.float() @ bt.float().t()
            c.metric("scaled_mm_max_rel_err", float(f"{((out.float() - ref).abs().max() / ref.abs().max()).item():.3g}"))
        except Exception as exc:  # noqa: BLE001
            print(f"SKIP: torch._scaled_mm failed on this build: {type(exc).__name__}: {exc}", flush=True)
    return c.finish()
