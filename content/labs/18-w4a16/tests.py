"""Tests for lab 18: W4A16.

Correctness is a hard check against an fp32 reference built from the dequantized
weights. Performance is reported, and fails only when the kernel is far from
what the byte count predicts.
"""

import torch
from lab_common import A100, Checks, bandwidth_gbs, cuda_time, require_cuda, tflops

GROUP = 128

# (M, N, K): decode, awkward, and prefill-sized shapes. K is always a multiple of
# the group size; M and N are deliberately not multiples of any block size.
SHAPES = [
    (1, 128, 128),
    (1, 33, 384),
    (1, 4096, 4096),
    (3, 257, 256),
    (16, 512, 1024),
    (17, 129, 512),
    (100, 300, 640),
    (256, 1024, 1024),
    (1000, 2048, 512),
]

# fp16 rounding of the output is 2**-11 = 4.9e-4 relative per element. The
# kernel accumulates in fp32, so 2e-3 of the reference's norm leaves a wide margin.
REL_TOL = 2e-3
# Round-to-nearest int4 with 15 levels over a group of 128 Gaussians: the step is
# about 5.3 sigma / 15 = 0.35 sigma, so the RMS error is 0.35 / sqrt(12) = 0.10 sigma.
QUANT_REL_TOL = 0.15
# The kernel must reach this fraction of the byte-count bound on speedup at M = 1:
# 0.25 x 3.82 = 0.95x, so it must be about as fast as fp16. A plain Triton kernel with
# per-group tile loads, a transpose, and a reduction per iteration is often instruction
# bound well below the memory rate, so a tighter bound would risk failing a correct kernel.
SPEEDUP_FRACTION = 0.25


def _rel_err(actual, expected):
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


def run(submission):
    c = Checks()
    names = ["quantize_int4", "pack_int4", "unpack_int4", "dequantize_int4", "int4_to_fp16_magic", "w4a16_matmul"]
    if not c.require(submission, *names):
        return c.finish()
    device = require_cuda()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)

    # -- Part 1: quantize, pack, unpack ---------------------------------------
    w = (torch.randn(64, 512, device=device) * 0.02).half()
    out = {}

    def quantize():
        out["q"], out["s"], out["z"] = submission.quantize_int4(w, GROUP)
        q, s, z = out["q"], out["s"], out["z"]
        ok = (
            tuple(q.shape) == (64, 512)
            and q.dtype == torch.uint8
            and int(q.max()) <= 15
            and tuple(s.shape) == (64, 512 // GROUP)
            and s.dtype == torch.float16
            and tuple(z.shape) == (64, 512 // GROUP)
            and int(z.max()) <= 15
        )
        return ok, f"codes {tuple(q.shape)} {q.dtype}, scales {tuple(s.shape)} {s.dtype}, zeros {tuple(z.shape)} {z.dtype}"

    if c.check("quantize_int4 returns uint8 codes, fp16 scales, and zeros per (row, group)", quantize):
        q, s, z = out["q"], out["s"], out["z"]

        def packing():
            p = submission.pack_int4(q)
            if tuple(p.shape) != (64, 256) or p.dtype != torch.uint8:
                return False, f"packed is {tuple(p.shape)} {p.dtype}, expected (64, 256) uint8"
            layout = bool(((p & 0xF) == q[:, 0::2]).all() and ((p >> 4) == q[:, 1::2]).all())
            if not layout:
                return False, "the low nibble must hold the even-index code and the high nibble the odd-index code"
            return bool((submission.unpack_int4(p) == q).all()), "pack then unpack returns the codes"

        c.check("pack_int4 puts code 2j in the low nibble and 2j + 1 in the high nibble", packing)
        p = submission.pack_int4(q)

        def error_bounds():
            w_hat = submission.dequantize_int4(p, s, z, GROUP)
            step = s.float().repeat_interleave(GROUP, dim=1)
            worst = ((w_hat - w.float()).abs() / step).max().item()
            rel = _rel_err(w_hat, w)
            return (
                worst <= 1.05 and rel <= QUANT_REL_TOL,
                f"largest error {worst:.2f} steps (at most 1), relative error {rel:.3f} (at most {QUANT_REL_TOL})",
            )

        c.check("dequantized weights are within one step and 15% relative error", error_bounds)

        def zero_exact():
            w0 = w.clone()
            w0[:, ::7] = 0
            q0, s0, z0 = submission.quantize_int4(w0, GROUP)
            w_hat = submission.dequantize_int4(submission.pack_int4(q0), s0, z0, GROUP)
            return bool((w_hat[:, ::7] == 0).all()), "0.0 round-trips exactly"

        c.check("zero is exactly representable", zero_exact)

    def magic():
        codes = torch.arange(16, device=device, dtype=torch.uint8).repeat(5)
        got = submission.int4_to_fp16_magic(codes)
        want = codes.to(torch.float16)
        return got.dtype == torch.float16 and bool((got == want).all()), "0..15 convert exactly"

    c.check("int4_to_fp16_magic converts codes 0..15 exactly", magic)

    # -- Part 2: the fused kernel, against an fp32 reference ---------------------
    for m, n, k in SHAPES:
        name = f"w4a16_matmul matches the fp32 reference at M={m}, N={n}, K={k}"

        def one(m=m, n=n, k=k):
            wt = (torch.randn(n, k, device=device) * 0.02).half()
            x = torch.randn(m, k, device=device).half()
            qq, ss, zz = submission.quantize_int4(wt, GROUP)
            pp = submission.pack_int4(qq)
            ref = x.float() @ submission.dequantize_int4(pp, ss, zz, GROUP).t()
            y = submission.w4a16_matmul(x, pp, ss, zz, GROUP)
            torch.cuda.synchronize()
            if tuple(y.shape) != (m, n) or y.dtype != torch.float16:
                return False, f"got {tuple(y.shape)} {y.dtype}, expected ({m}, {n}) float16"
            err = _rel_err(y, ref)
            return err <= REL_TOL, f"relative error {err:.2e} (tolerance {REL_TOL:.0e})"

        c.check(name, one)

    # A non-contiguous X: rows are a slice of a wider matrix.
    def strided_x():
        n, k, m = 64, 256, 5
        wt = (torch.randn(n, k, device=device) * 0.02).half()
        base = torch.randn(m, k + 64, device=device).half()
        x = base[:, :k]
        qq, ss, zz = submission.quantize_int4(wt, GROUP)
        pp = submission.pack_int4(qq)
        ref = x.float() @ submission.dequantize_int4(pp, ss, zz, GROUP).t()
        y = submission.w4a16_matmul(x, pp, ss, zz, GROUP)
        err = _rel_err(y, ref)
        return err <= REL_TOL, f"relative error {err:.2e}"

    c.check("w4a16_matmul accepts a row-strided X", strided_x)

    # -- Part 3: performance ----------------------------------------------------
    # N = K = 8192: 67.1 M weights, 134.2 MB in fp16. The int4 tensors are about
    # 35 MB, so the L2 is flushed before every timed call.
    n = k = 8192
    try:
        wt = (torch.randn(n, k, device=device) * 0.02).half()
        qq, ss, zz = submission.quantize_int4(wt, GROUP)
        pp = submission.pack_int4(qq)
        del qq
        q_bytes = pp.numel() * pp.element_size() + ss.numel() * ss.element_size() + zz.numel() * zz.element_size()
        fp16_bytes = wt.numel() * wt.element_size()
        ratio = fp16_bytes / q_bytes
        c.metric("bytes_ratio", round(ratio, 3))

        times = {}
        for m in (1, 16, 1024):
            x = torch.randn(m, k, device=device).half()
            submission.w4a16_matmul(x, pp, ss, zz, GROUP)  # compile and autotune
            torch.cuda.synchronize()
            t4 = cuda_time(lambda: submission.w4a16_matmul(x, pp, ss, zz, GROUP), flush_l2=True)
            t16 = cuda_time(lambda: torch.matmul(x, wt.t()), flush_l2=True)
            times[m] = (t4, t16)
            print(f"      M={m}: W4A16 {t4:.4f} ms, fp16 torch.matmul {t16:.4f} ms", flush=True)

        t4, t16 = times[1]
        gbs = bandwidth_gbs(q_bytes, t4)
        c.metric("weight_gbs_m1", round(gbs, 1))
        c.metric("speedup_m1", round(t16 / t4, 3))
        c.metric("speedup_m16", round(times[16][1] / times[16][0], 3))
        c.metric("tflops_m1024", round(tflops(2 * 1024 * n * k, times[1024][0]), 1))
        bound = SPEEDUP_FRACTION * ratio
        c.add(
            f"speedup at M=1 is at least {SPEEDUP_FRACTION:.0%} of the byte-count bound",
            t16 / t4 >= bound,
            f"{t16 / t4:.2f}x, bound {bound:.2f}x ({SPEEDUP_FRACTION:.0%} of the {ratio:.2f}x byte ratio). "
            f"Effective weight bandwidth {gbs:.0f} GB/s against the {A100.COPY_CEILING_GBS:.0f} GB/s copy ceiling.",
        )
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        c.add("performance run completes", False, f"{type(exc).__name__}: {exc}")
    return c.finish()
