"""Tests for the FlashAttention lab.

Correctness is a hard check against a float32 PyTorch reference, over shapes that
include sequence lengths that are not multiples of the block size, a length of 1,
grouped-query heads, and both head dimensions. The backward pass is checked
against torch.autograd too. Performance is reported in TFLOP/s next to PyTorch's
scaled_dot_product_attention, with loose thresholds.
"""

import math

import torch
import torch.nn.functional as F

from lab_common import A100, Checks, cuda_time, require_cuda, tflops

# (batch, heads, kv_heads, seq_len, head_dim). Lengths 1, 100, 130, 257, and 333 are
# not multiples of any block size the solution uses (32, 64, or 128).
SHAPES = [
    (1, 2, 2, 128, 64),
    (2, 4, 4, 100, 64),
    (1, 4, 2, 333, 128),
    (1, 1, 1, 1, 64),
    (2, 8, 2, 257, 128),
    (1, 2, 2, 130, 64),
    (1, 2, 2, 64, 128),
]

# Tolerances are fractions of the reference's largest magnitude. The inputs are
# bfloat16 (8 significant bits, so a relative step of 2^-8 = 0.4%), and P and dS
# are cast to bfloat16 before their matmuls, so 3 to 4% of the maximum leaves a
# wide margin while a missing scale factor or a wrong mask misses by far more.
O_TOL = 0.03
GRAD_TOL = 0.04
LSE_TOL = 2e-3  # absolute; LSE is float32 and the scores accumulate in float32

# Performance thresholds. learn-inference's lab asks the forward pass for 0.33 of
# the vendor kernel. The backward pass here recomputes P twice (7 matmuls against
# the vendor's 5), so its ceiling relative to the vendor is about 5/7 of the
# forward's, and 0.33 * 5/7 = 0.24, which this lab rounds down to 0.2.
FWD_VS_SDPA_MIN = 0.33
BWD_VS_SDPA_MIN = 0.2
PEAK_SANITY = A100.TENSOR_BF16_TFLOPS * 1.05


def reference(q, k, v, causal):
    """Float32 attention that materializes the scores. Returns (O, LSE)."""
    group = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)
    s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if causal:
        n = q.shape[2]
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=q.device), diagonal=1)
        s = s.masked_fill(mask, float("-inf"))
    return torch.softmax(s, dim=-1) @ v, torch.logsumexp(s, dim=-1)


def make_inputs(shape, device, scale=1.0):
    b, h, hk, n, d = shape
    g = torch.Generator(device=device).manual_seed(1234 + n + d)
    q = (torch.randn(b, h, n, d, device=device, generator=g) * scale).to(torch.bfloat16)
    k = torch.randn(b, hk, n, d, device=device, generator=g).to(torch.bfloat16)
    v = torch.randn(b, hk, n, d, device=device, generator=g).to(torch.bfloat16)
    do = torch.randn(b, h, n, d, device=device, generator=g).to(torch.bfloat16)
    return q, k, v, do


def rel_err(actual, expected):
    """Largest absolute error as a fraction of the reference's largest magnitude."""
    denom = max(expected.abs().max().item(), 1e-2)
    return (actual.float() - expected.float()).abs().max().item() / denom


def reference_grads(q, k, v, do, causal):
    qf, kf, vf = (t.detach().float().requires_grad_(True) for t in (q, k, v))
    o, lse = reference(qf, kf, vf, causal)
    dq, dk, dv = torch.autograd.grad(o, (qf, kf, vf), do.float())
    return o.detach(), lse.detach(), dq, dk, dv


def tag(shape, causal):
    b, h, hk, n, d = shape
    return f"B={b} H={h}/{hk} N={n} D={d} {'causal' if causal else 'full'}"


def run(submission):
    c = Checks()
    if not c.require(submission, "flash_forward", "flash_backward", "flash_attention"):
        return c.finish()
    device = require_cuda()
    props = torch.cuda.get_device_properties(0)
    print(f"      device = {props.name}", flush=True)

    # ---- Correctness: forward and backward, every shape, causal and not ----
    for shape in SHAPES:
        for causal in (False, True):
            q, k, v, do = make_inputs(shape, device)
            o_ref, lse_ref, dq_ref, dk_ref, dv_ref = reference_grads(q, k, v, do, causal)
            label = tag(shape, causal)

            def fwd():
                o, lse = submission.flash_forward(q, k, v, causal)
                assert o.shape == q.shape and o.dtype == q.dtype, f"O is {tuple(o.shape)} {o.dtype}"
                assert lse.shape == q.shape[:3], f"LSE is {tuple(lse.shape)}, expected {tuple(q.shape[:3])}"
                assert torch.isfinite(o.float()).all(), "O has NaN or inf"
                eo = rel_err(o, o_ref)
                el = (lse.float() - lse_ref).abs().max().item()
                return (
                    eo <= O_TOL and el <= LSE_TOL,
                    f"O error {eo:.2e} of max (limit {O_TOL}), LSE max abs error {el:.2e} (limit {LSE_TOL})",
                )

            def bwd():
                # Feed the reference O and LSE, so a forward bug cannot hide a backward bug.
                dq, dk, dv = submission.flash_backward(
                    q, k, v, o_ref.to(torch.bfloat16), lse_ref.float(), do, causal
                )
                assert dq.shape == q.shape and dk.shape == k.shape and dv.shape == v.shape, "gradient shapes"
                errs = [rel_err(dq, dq_ref), rel_err(dk, dk_ref), rel_err(dv, dv_ref)]
                return (
                    max(errs) <= GRAD_TOL,
                    f"dQ {errs[0]:.2e}, dK {errs[1]:.2e}, dV {errs[2]:.2e} of max (limit {GRAD_TOL})",
                )

            c.check(f"forward matches float32 reference: {label}", fwd)
            c.check(f"backward matches torch.autograd: {label}", bwd)

    # ---- The autograd.Function path, end to end ----
    for shape, causal in [((1, 4, 2, 200, 128), True), ((2, 2, 2, 90, 64), False)]:
        q, k, v, do = make_inputs(shape, device)
        o_ref, _, dq_ref, dk_ref, dv_ref = reference_grads(q, k, v, do, causal)

        def end_to_end():
            q1, k1, v1 = (t.clone().requires_grad_(True) for t in (q, k, v))
            out = submission.flash_attention(q1, k1, v1, causal)
            out.backward(do)
            assert q1.grad is not None and k1.grad is not None and v1.grad is not None, "no gradients reached the inputs"
            errs = [
                rel_err(out, o_ref),
                rel_err(q1.grad, dq_ref),
                rel_err(k1.grad, dk_ref),
                rel_err(v1.grad, dv_ref),
            ]
            return (
                max(errs) <= GRAD_TOL,
                f"O {errs[0]:.2e}, dQ {errs[1]:.2e}, dK {errs[2]:.2e}, dV {errs[3]:.2e} of max (limit {GRAD_TOL})",
            )

        c.check(f"flash_attention(...).backward() matches torch.autograd: {tag(shape, causal)}", end_to_end)

    # ---- Numerical range: large scores must not overflow ----
    def large_scores():
        shape = (1, 2, 2, 150, 64)
        q, k, v, _ = make_inputs(shape, device, scale=8.0)
        o_ref, lse_ref = reference(q.float(), k.float(), v.float(), True)
        o, lse = submission.flash_forward(q, k, v, True)
        finite = torch.isfinite(o.float()).all().item() and torch.isfinite(lse).all().item()
        e = rel_err(o, o_ref)
        return finite and e <= O_TOL, f"finite={finite}, O error {e:.2e} of max (limit {O_TOL})"

    c.check("queries 8x larger (scores near 64) stay finite and correct", large_scores)

    # ---- Determinism of the backward pass ----
    def deterministic():
        shape = (1, 4, 2, 300, 64)
        q, k, v, do = make_inputs(shape, device)
        o, lse = submission.flash_forward(q, k, v, True)
        first = submission.flash_backward(q, k, v, o, lse, do, True)
        second = submission.flash_backward(q, k, v, o, lse, do, True)
        same = all(torch.equal(a, b) for a, b in zip(first, second))
        return same, "two backward calls return bit-identical gradients" if same else "gradients differ run to run"

    c.check("backward is deterministic (no atomics in the lab's design)", deterministic)

    # ---- Memory: the score matrix must not exist ----
    def no_score_matrix():
        b, h, n, d = 1, 8, 8192, 64
        q = torch.randn(b, h, n, d, device=device, dtype=torch.bfloat16)
        k = torch.randn(b, h, n, d, device=device, dtype=torch.bfloat16)
        v = torch.randn(b, h, n, d, device=device, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        o, lse = submission.flash_forward(q, k, v, True)
        torch.cuda.synchronize()
        extra_mib = (torch.cuda.max_memory_allocated() - base) / 2**20
        scores_mib = b * h * n * n * 2 / 2**20  # the score matrix in bfloat16
        c.metric("fwd_peak_extra_mib", round(extra_mib, 1))
        del o, lse
        return (
            extra_mib < 128,
            f"{extra_mib:.1f} MiB above the inputs; the bfloat16 score matrix alone would be {scores_mib:.0f} MiB",
        )

    c.check("forward peak memory excludes the N x N scores", no_score_matrix)

    # ---- Performance ----
    def measure(d, causal, with_backward, n=4096, b=4, h=16):
        g = torch.Generator(device=device).manual_seed(7)
        q, k, v, do = (
            torch.randn(b, h, n, d, device=device, generator=g).to(torch.bfloat16) for _ in range(4)
        )
        fwd_flops = 4.0 * b * h * n * n * d * (0.5 if causal else 1.0)
        out = {}
        out["fwd_ms"] = cuda_time(lambda: submission.flash_forward(q, k, v, causal), warmup=5, iters=20)
        out["sdpa_fwd_ms"] = cuda_time(
            lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal), warmup=5, iters=20
        )
        out["fwd_tflops"] = tflops(fwd_flops, out["fwd_ms"])
        out["sdpa_fwd_tflops"] = tflops(fwd_flops, out["sdpa_fwd_ms"])
        if with_backward:
            o, lse = submission.flash_forward(q, k, v, causal)
            # The papers count the backward pass as 2.5x the forward FLOPs (5 matmuls against 2).
            bwd_flops = 2.5 * fwd_flops
            out["bwd_ms"] = cuda_time(
                lambda: submission.flash_backward(q, k, v, o, lse, do, causal), warmup=3, iters=10
            )
            qs, ks, vs = (t.clone().requires_grad_(True) for t in (q, k, v))
            res = F.scaled_dot_product_attention(qs, ks, vs, is_causal=causal)
            out["sdpa_bwd_ms"] = cuda_time(lambda: res.backward(do, retain_graph=True), warmup=3, iters=10)
            out["bwd_tflops"] = tflops(bwd_flops, out["bwd_ms"])
            out["sdpa_bwd_tflops"] = tflops(bwd_flops, out["sdpa_bwd_ms"])
        return out

    perf = measure(128, True, True)
    c.metric("fwd_tflops", round(perf["fwd_tflops"], 1))
    c.metric("bwd_tflops", round(perf["bwd_tflops"], 1))
    c.metric("fwd_vs_sdpa", round(perf["fwd_tflops"] / perf["sdpa_fwd_tflops"], 3))
    c.metric("bwd_vs_sdpa", round(perf["bwd_tflops"] / perf["sdpa_bwd_tflops"], 3))
    print(
        f"      SDPA at the same shape: forward {perf['sdpa_fwd_tflops']:.1f} TFLOP/s, "
        f"backward {perf['sdpa_bwd_tflops']:.1f} TFLOP/s (B=4 H=16 N=4096 D=128 causal)",
        flush=True,
    )

    perf64 = measure(64, True, True)
    c.metric("fwd_tflops_d64", round(perf64["fwd_tflops"], 1))
    c.metric("bwd_tflops_d64", round(perf64["bwd_tflops"], 1))
    nc = measure(128, False, False)
    c.metric("fwd_tflops_noncausal", round(nc["fwd_tflops"], 1))

    c.add(
        "reported TFLOP/s are below the 312 TFLOP/s bf16 tensor-core peak",
        max(perf["fwd_tflops"], perf["bwd_tflops"], perf64["fwd_tflops"], perf64["bwd_tflops"], nc["fwd_tflops"])
        <= PEAK_SANITY,
        "a figure above the peak means the FLOP count or the timing is wrong",
    )
    c.add(
        f"forward is at least {FWD_VS_SDPA_MIN:.2f}x of scaled_dot_product_attention (D=128, causal, N=4096)",
        perf["fwd_tflops"] >= FWD_VS_SDPA_MIN * perf["sdpa_fwd_tflops"],
        f"{perf['fwd_tflops'] / perf['sdpa_fwd_tflops']:.2f}x",
    )
    c.add(
        f"backward is at least {BWD_VS_SDPA_MIN:.2f}x of scaled_dot_product_attention (D=128, causal, N=4096)",
        perf["bwd_tflops"] >= BWD_VS_SDPA_MIN * perf["sdpa_bwd_tflops"],
        f"{perf['bwd_tflops'] / perf['sdpa_bwd_tflops']:.2f}x",
    )
    return c.finish()
