"""Tests for lab 37: chunked scans for linear attention.

Checks 1 to 9 run in torch float64 on a CPU. Check 1 calls only `combine`:
implementing that one function passes it. The Triton kernel needs an A100 and
Triton. A machine without them skips the kernel check and reports a metric of
"skipped".

Plausible wrong implementations, and the check that catches each:

swapped_operator is a plausible wrong `combine`. It returns
(a1 * a2, a1 * b2 + b1), which applies the second step first. Check 1 fails
on combine((2, 3), (4, 5)): the right answer is (8, 17), the wrong one is
(8, 13). The wrong operator is still associative, so check 2 alone misses it.

exp_then_mask is a plausible wrong `decay_matrix`. It computes
exp(b_i - b_j) for every pair and multiplies by the lower-triangular mask.
Above the diagonal the exponent is large and positive, exp overflows to inf,
and inf * 0 is NaN. Check 5 uses a decay of exp(-6) per step over 200 tokens,
so the exponent reaches 1194. Dividing cumulative products fails the same
way, and the tests show it in check 5.

wrong_boundary_decay is a plausible wrong chunked algorithm. It decays the
entering state by exp(b_{i-1}), the cumulative decay before position i, when
position i's output needs exp(b_i). That is off by one position. It also
builds a chunk's state update with exp(b_last - b_{j-1}). Checks 6 and 7
fail, and the failure shows how far the output moves.
"""

from __future__ import annotations

import math

from lab_common import A100, Checks

NAMES = (
    "combine",
    "scan_sequential",
    "scan_parallel",
    "linear_attention_sequential",
    "decay_matrix",
    "chunk_states",
    "entering_states",
    "chunked_linear_attention",
    "chunk_flops",
    "chunk_bytes",
    "chunk_time",
    "best_chunk",
    "softmax_flops",
    "crossover_length",
    "chunked_forward_triton",
)


def _data(t, dk, dv, seed, decay=(-0.5, -0.01)):
    """Seeded float64 inputs. `decay` is the range of the per-step log decay."""
    import torch

    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(t, dk, generator=gen, dtype=torch.float64)
    k = torch.randn(t, dk, generator=gen, dtype=torch.float64) / math.sqrt(dk)
    v = torch.randn(t, dv, generator=gen, dtype=torch.float64)
    lo, hi = decay
    logg = lo + (hi - lo) * torch.rand(t, generator=gen, dtype=torch.float64)
    return q, k, v, logg


def _ref(q, k, v, logg):
    """The tests' own sequential reference, independent of the reader's."""
    import torch

    s = torch.zeros(k.shape[1], v.shape[1], dtype=torch.float64)
    out = []
    for t in range(q.shape[0]):
        s = math.exp(float(logg[t])) * s + torch.outer(k[t], v[t])
        out.append(q[t] @ s)
    return torch.stack(out), s


def _err(actual, expected):
    import torch

    actual = torch.as_tensor(actual, dtype=torch.float64)
    expected = torch.as_tensor(expected, dtype=torch.float64)
    if actual.shape != expected.shape:
        return None, f"expected shape {tuple(expected.shape)}, actual {tuple(actual.shape)}"
    if not torch.isfinite(actual).all():
        return None, "expected finite values, actual a NaN or inf"
    return float((actual - expected).abs().max()), ""


def _compare(actual, expected, tol, what):
    err, why = _err(actual, expected)
    if err is None:
        return False, why
    if err > tol:
        return False, f"expected max error <= {tol:g} for {what}, actual {err:.3e}"
    return True, f"expected max error <= {tol:g} for {what}, actual {err:.2e}"


def _skip_reason():
    try:
        import torch
    except ImportError:
        return "SKIP: the Triton kernel was not run. torch is not installed."
    if not torch.cuda.is_available():
        return "SKIP: the Triton kernel was not run. No CUDA GPU is visible."
    try:
        import triton  # noqa: F401
    except ImportError:
        return "SKIP: the Triton kernel was not run. Triton is not installed."
    return None


def run(submission):
    c = Checks()
    if not c.require(submission, *NAMES):
        return c.finish()
    r = submission

    import torch

    # ---- 1. The operator -------------------------------------------------
    def operator():
        # swapped_operator fails here: the answer is (8, 17), not (8, 13).
        x = (torch.tensor(2.0, dtype=torch.float64), torch.tensor(3.0, dtype=torch.float64))
        y = (torch.tensor(4.0, dtype=torch.float64), torch.tensor(5.0, dtype=torch.float64))
        a, b = r.combine(x, y)
        got = (float(a), float(b))
        if got != (8.0, 17.0):
            return False, f"expected (8.0, 17.0), actual {got}"
        return True, "expected (8.0, 17.0), actual (8.0, 17.0)"

    c.check("combine applies the first step, then the second", operator)

    # ---- 2. Associativity ------------------------------------------------
    def associativity():
        gen = torch.Generator().manual_seed(37)
        worst = 0.0
        for _ in range(200):
            p = [(torch.randn(1, generator=gen, dtype=torch.float64)[0],
                  torch.randn(1, generator=gen, dtype=torch.float64)[0]) for _ in range(3)]
            left = r.combine(r.combine(p[0], p[1]), p[2])
            right = r.combine(p[0], r.combine(p[1], p[2]))
            worst = max(worst, abs(float(left[0] - right[0])), abs(float(left[1] - right[1])))
        swapped = r.combine(p[0], p[1])
        other = r.combine(p[1], p[0])
        commutes = abs(float(swapped[1] - other[1])) < 1e-9
        if worst > 1e-12:
            return False, f"expected (x o y) o z = x o (y o z) to 1e-12, actual gap {worst:.2e}"
        if commutes:
            return False, "expected a non-commutative operator, actual combine(x, y) = combine(y, x)"
        return True, f"expected gap <= 1e-12 over 200 triples, actual {worst:.2e}; the operator is not commutative"

    c.check("combine is associative on random triples and not commutative", associativity)

    # ---- 3. The scans ----------------------------------------------------
    def scans():
        gen = torch.Generator().manual_seed(3)
        for t in (1, 2, 13, 64, 100):
            a = torch.rand(t, generator=gen, dtype=torch.float64) * 1.2
            b = torch.randn(t, generator=gen, dtype=torch.float64)
            want = []
            h = 0.0
            for i in range(t):
                h = float(a[i]) * h + float(b[i])
                want.append(h)
            want = torch.tensor(want, dtype=torch.float64)
            ok, msg = _compare(r.scan_sequential(a, b), want, 1e-10, f"scan_sequential, T={t}")
            if not ok:
                return False, msg
            ok, msg = _compare(r.scan_parallel(a, b), want, 1e-10, f"scan_parallel, T={t}")
            if not ok:
                return False, msg
        # With a = 1 the recurrence is a cumulative sum.
        b = torch.arange(1, 10, dtype=torch.float64)
        ok, msg = _compare(r.scan_parallel(torch.ones(9, dtype=torch.float64), b), torch.cumsum(b, 0), 1e-12, "a = 1")
        if not ok:
            return False, msg
        # Matrix-valued b with a scalar a per step, the shape the chunk scan uses.
        a = torch.rand(7, generator=gen, dtype=torch.float64)
        b = torch.randn(7, 3, 2, generator=gen, dtype=torch.float64)
        return _compare(r.scan_parallel(a, b), r.scan_sequential(a, b), 1e-10, "matrix b")

    c.check("the parallel scan over combine equals the sequential loop", scans)

    # ---- 4. The sequential reference ------------------------------------
    def reference():
        q, k, v, _ = _data(9, 4, 3, 4)
        zero = torch.zeros(9, dtype=torch.float64)
        o, s = r.linear_attention_sequential(q, k, v, zero)
        want = torch.tril(q @ k.T) @ v
        ok, msg = _compare(o, want, 1e-10, "no decay equals tril(Q K^T) V")
        if not ok:
            return False, msg
        # One hand-checked step: S_1 = 0.5 * outer(k_0, v_0) + outer(k_1, v_1).
        q2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
        k2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
        v2 = torch.tensor([[2.0], [3.0]], dtype=torch.float64)
        lg = torch.log(torch.tensor([0.9, 0.5], dtype=torch.float64))
        o2, _ = r.linear_attention_sequential(q2, k2, v2, lg)
        want2 = torch.tensor([[2.0], [3.0]], dtype=torch.float64)
        # o_0 = q_0 S_0 = 2. S_1 = 0.5 * [[2],[0]] + [[0],[3]] = [[1],[3]]. o_1 = 3.
        return _compare(o2, want2, 1e-12, "the two-token example")

    c.check("the sequential reference matches a closed form", reference)

    # ---- 5. Log-space decay ---------------------------------------------
    def log_space():
        # exp_then_mask fails here: the exponent above the diagonal reaches 1194.
        t = 200
        logg = torch.full((t,), -6.0, dtype=torch.float64)
        d = r.decay_matrix(logg)
        if d.shape != (t, t):
            return False, f"expected shape {(t, t)}, actual {tuple(d.shape)}"
        if not torch.isfinite(d).all():
            return False, "expected a finite matrix for exp(-6) decay over 200 tokens, actual a NaN or inf"
        idx = torch.arange(t)
        rows, cols = idx[:, None], idx[None, :]
        want = torch.where(rows >= cols, torch.exp(-6.0 * (rows - cols).to(torch.float64)), torch.zeros(()))
        if float((d - want).abs().max()) > 1e-12:
            return False, f"expected D[i, j] = exp(-6 (i - j)), actual max error {float((d - want).abs().max()):.2e}"
        if float(torch.triu(d, 1).abs().max()) != 0.0:
            return False, "expected zeros above the diagonal, actual nonzero entries"
        # The naive cumulative-product ratio on the same input, to show what the log form avoids.
        g = torch.exp(logg)
        cp = torch.cumprod(g, 0)
        naive = torch.tril(cp[:, None] / cp[None, :])
        bad = int((~torch.isfinite(naive)).sum())
        if bad == 0:
            return False, "expected the cumprod ratio to produce NaN here, actual all finite (the test is broken)"
        # A moderate decay: the log form must also match the plain product.
        lg = torch.tensor([-0.1, -0.2, -0.3, -0.4], dtype=torch.float64)
        d4 = r.decay_matrix(lg)
        if abs(float(d4[3, 0]) - math.exp(-0.9)) > 1e-12:
            return False, f"expected D[3, 0] = exp(-0.9) = {math.exp(-0.9):.6f}, actual {float(d4[3, 0]):.6f}"
        return True, f"expected a finite D, actual finite; the cumprod ratio gives {bad} NaN entries on the same input"

    c.check("decay_matrix stays finite where cumprod ratios give NaN", log_space)

    # ---- 6. Chunk states and the entering state -------------------------
    def boundary():
        # wrong_boundary_decay fails here: S entering chunk c is the state after token c*C - 1.
        t, chunk = 11, 4
        q, k, v, logg = _data(t, 3, 2, 6)
        a, u = r.chunk_states(k, v, logg, chunk)
        n = math.ceil(t / chunk)
        if tuple(a.shape) != (n,) or tuple(u.shape) != (n, 3, 2):
            return False, f"expected shapes {(n,)} and {(n, 3, 2)}, actual {tuple(a.shape)} and {tuple(u.shape)}"
        want_a = torch.stack([torch.exp(logg[s:min(s + chunk, t)].sum()) for s in range(0, t, chunk)])
        ok, msg = _compare(a, want_a, 1e-12, "the chunk decay A_c")
        if not ok:
            return False, msg
        s_in, s_final = r.entering_states(a, u)
        for cidx in range(n):
            start = cidx * chunk
            if start == 0:
                want = torch.zeros(3, 2, dtype=torch.float64)
            else:
                _, want = _ref(q[:start], k[:start], v[:start], logg[:start])
            ok, msg = _compare(s_in[cidx], want, 1e-10, f"the state entering chunk {cidx}")
            if not ok:
                return False, msg
        _, want_final = _ref(q, k, v, logg)
        return _compare(s_final, want_final, 1e-10, "the final state")

    c.check("the state entering each chunk is the sequential state at its boundary", boundary)

    # ---- 7. Chunked equals sequential -----------------------------------
    def chunked():
        # wrong_boundary_decay fails here as well.
        t = 50
        q, k, v, logg = _data(t, 5, 4, 7)
        want, want_s = _ref(q, k, v, logg)
        for chunk in (1, 7, 16, 25, 32, 50, 64):
            o, s = r.chunked_linear_attention(q, k, v, logg, chunk)
            ok, msg = _compare(o, want, 1e-9, f"the output at chunk size {chunk} (T=50)")
            if not ok:
                return False, msg
            ok, msg = _compare(s, want_s, 1e-9, f"the final state at chunk size {chunk}")
            if not ok:
                return False, msg
        return True, "expected max error <= 1e-9 at chunk sizes 1, 7, 16, 25, 32, 50, 64 for T=50; actual within it (7 does not divide 50)"

    c.check("the chunked algorithm equals the sequential reference at every chunk size", chunked)

    # ---- 8. Strong decay -------------------------------------------------
    def strong_decay():
        t = 256
        q, k, v, logg = _data(t, 4, 4, 8, decay=(-6.5, -5.5))
        want, want_s = _ref(q, k, v, logg)
        for chunk in (64, 128):
            o, s = r.chunked_linear_attention(q, k, v, logg, chunk)
            ok, msg = _compare(o, want, 1e-9, f"the output at chunk {chunk} with decay near exp(-6)")
            if not ok:
                return False, msg
        return True, "expected finite output matching the reference, actual finite and within 1e-9 at chunk sizes 64 and 128"

    c.check("the chunked algorithm survives decays that overflow a cumprod", strong_decay)

    # ---- 9. The cost model ----------------------------------------------
    def cost():
        t, dk, dv = 4096, 128, 128
        n = t // 64
        want_f = n * (2 * 64 * 64 * 256 + 4 * 64 * 128 * 128)
        got_f = r.chunk_flops(t, 64, dk, dv)
        if got_f != want_f:
            return False, f"expected {want_f} FLOPs for T=4096, C=64, actual {got_f}"
        want_b = t * 512 * 2 + 2 * n * 128 * 128 * 4
        got_b = r.chunk_bytes(t, 64, dk, dv)
        if got_b != want_b:
            return False, f"expected {want_b} bytes for T=4096, C=64, actual {got_b}"
        # The minimum of the modelled time is inside the candidate range.
        peak = A100.TENSOR_BF16_TFLOPS * 1e12
        bw = A100.COPY_CEILING_GBS * 1e9
        t_long = 16384
        cands = (16, 32, 64, 128, 256, 512, 1024)
        times = [r.chunk_time(t_long, ch, dk, dv, peak, bw) for ch in cands]
        want_best = cands[min(range(len(cands)), key=lambda i: (times[i], cands[i]))]
        got_best = r.best_chunk(t_long, dk, dv, peak, bw)
        if got_best != want_best:
            return False, f"expected best chunk {want_best}, actual {got_best}"
        if got_best in (cands[0], cands[-1]):
            return False, f"expected an interior minimum, actual {got_best}, the end of the range"
        for ch in (cands[0], cands[-1]):
            if r.chunk_time(t_long, ch, dk, dv, peak, bw) <= r.chunk_time(t_long, got_best, dk, dv, peak, bw):
                return False, f"expected C={got_best} to beat C={ch}, actual it does not"
        # Closed form: chunked FLOPs equal causal softmax FLOPs at T = 2C + 4 dk dv / (dk + dv).
        x = r.crossover_length(64, dk, dv)
        if abs(x - 384.0) > 1e-9:
            return False, f"expected a crossover of 384 tokens for C=64, dk=dv=128, actual {x}"
        if r.softmax_flops(4096, dk, dv) != 4096 * 4096 * 256:
            return False, f"expected {4096 * 4096 * 256} softmax FLOPs at T=4096, actual {r.softmax_flops(4096, dk, dv)}"
        return True, f"expected an interior minimum, actual C={got_best} at T=16384; chunked FLOPs drop below softmax's above T=384 for C=64"

    c.check("the cost model has an interior minimum and a crossover length", cost)

    _gpu_section(c, r)
    return c.finish()


def _gpu_section(c: Checks, r) -> None:
    """The Triton kernel. Skipped, not failed, when this machine has no A100 or no Triton."""
    reason = _skip_reason()
    if reason is not None:
        print(reason, flush=True)
        c.metric("best_chunk", "skipped (no GPU)")
        return

    import torch

    from lab_common import cuda_time

    torch.manual_seed(37)
    heads, t, dk, dv = 8, 1000, 32, 32
    gen = torch.Generator().manual_seed(371)
    q = torch.randn(heads, t, dk, generator=gen)
    k = torch.randn(heads, t, dk, generator=gen) / math.sqrt(dk)
    v = torch.randn(heads, t, dv, generator=gen)
    logg = -0.5 * torch.rand(heads, t, generator=gen) - 0.005
    want = torch.stack([_ref(q[h].double(), k[h].double(), v[h].double(), logg[h].double())[0] for h in range(heads)])
    qc, kc, vc, lc = (x.cuda() for x in (q, k, v, logg))

    def kernel():
        # T = 1000 is not a multiple of any chunk size, so the last chunk is ragged.
        scale = float(want.abs().max())
        for chunk in (16, 32, 64, 128):
            try:
                got = r.chunked_forward_triton(qc, kc, vc, lc, chunk).cpu().double()
            except Exception as exc:  # noqa: BLE001
                return False, f"expected an output at chunk size {chunk}, actual {type(exc).__name__}: {exc}"
            ok, msg = _compare(got, want, 2e-3 * max(scale, 1.0), f"the Triton output at chunk size {chunk}")
            if not ok:
                return False, msg
        return True, "expected the Triton output within 2e-3 of the float64 reference at chunk sizes 16, 32, 64, 128; actual within it"

    if not c.check("the Triton chunked forward matches the float64 reference at every chunk size", kernel):
        return

    # The sweep is a metric, not a pass/fail threshold.
    heads, t, dk, dv = 16, 8192, 64, 64
    gen = torch.Generator().manual_seed(372)
    q = torch.randn(heads, t, dk, generator=gen).cuda()
    k = (torch.randn(heads, t, dk, generator=gen) / math.sqrt(dk)).cuda()
    v = torch.randn(heads, t, dv, generator=gen).cuda()
    logg = (-0.1 * torch.rand(heads, t, generator=gen) - 0.001).cuda()
    best_ms, best_c = None, None
    for chunk in (16, 32, 64, 128):
        try:
            r.chunked_forward_triton(q, k, v, logg, chunk)
            ms = cuda_time(lambda ch=chunk: r.chunked_forward_triton(q, k, v, logg, ch), warmup=3, iters=10)
        except Exception as exc:  # noqa: BLE001
            print(f"      chunk {chunk}: skipped ({type(exc).__name__})", flush=True)
            continue
        c.metric(f"ms_c{chunk}", round(ms, 4))
        if best_ms is None or ms < best_ms:
            best_ms, best_c = ms, chunk
    c.metric("best_chunk", best_c)
