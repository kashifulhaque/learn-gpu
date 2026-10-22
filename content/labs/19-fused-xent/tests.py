import torch
import torch.nn.functional as F

from lab_common import Checks, cuda_time, pick_device, tflops

IGNORE = -100

# (tokens, hidden, vocab, chunk) including sizes that are not multiples of the
# Triton block (1024), a vocab smaller than one block, a single token, and a
# chunk larger than the batch.
SHAPES = [
    (1, 8, 10, 4),
    (7, 16, 100, 3),
    (64, 32, 1000, 17),
    (130, 48, 2049, 64),
    (257, 64, 5000, 256),
    (33, 16, 1024, 1024),
]

# Benchmark geometry. N * V * 2 bytes = 1.07 GB is the size of the bf16 logits
# alone, which the reference must keep for the backward pass.
BENCH_N, BENCH_D, BENCH_V, BENCH_CHUNK = 8192, 512, 65536, 512


def make_case(n, d, v, dtype, device, seed=0):
    g = torch.Generator().manual_seed(seed + n + v)
    h = torch.randn(n, d, generator=g, dtype=torch.float64)
    w = torch.randn(v, d, generator=g, dtype=torch.float64) * (2.0 / d**0.5)
    t = torch.randint(0, v, (n,), generator=g)
    if n >= 7:
        t[::5] = IGNORE  # some ignored tokens, at least one valid
    return h.to(dtype).to(device), w.to(dtype).to(device), t.to(device)


def reference64(h, w, t):
    """Loss and gradients in float64, from the same (rounded) inputs."""
    h64 = h.detach().double().requires_grad_()
    w64 = w.detach().double().requires_grad_()
    loss = F.cross_entropy(h64 @ w64.t(), t, ignore_index=IGNORE)
    loss.backward()
    return loss.detach(), h64.grad, w64.grad


def rel_err(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def run(submission):
    c = Checks()
    if not c.require(
        submission,
        "softmax_xent_grad_torch_",
        "softmax_xent_grad_triton_",
        "fused_linear_xent",
        "reference_linear_xent",
    ):
        return c.finish()

    device = pick_device()
    on_cuda = device.type == "cuda"
    print(f"      device = {device}", flush=True)

    # --- Stage 1: the in-place softmax-gradient of one chunk of logits -------
    def stage1(fn, dtype, tol):
        worst = 0.0
        for rows, vocab in [(1, 10), (7, 100), (33, 2049), (5, 3000)]:
            g = torch.Generator().manual_seed(rows + vocab)
            x64 = torch.randn(rows, vocab, generator=g, dtype=torch.float64) * 3
            t = torch.randint(0, vocab, (rows,), generator=g)
            if rows >= 5:
                t[0] = IGNORE
            n_valid = max(int((t != IGNORE).sum()), 1)
            scale = 1.0 / n_valid
            x = x64.to(dtype).to(device).contiguous()
            tt = t.to(device)
            # float64 reference from the rounded logits
            xr = x.double().cpu().requires_grad_()
            loss_rows = F.cross_entropy(xr, t, ignore_index=IGNORE, reduction="none")
            (loss_rows.sum() * scale).backward()
            row_loss = fn(x, tt, scale, IGNORE)
            worst = max(worst, (row_loss.double().cpu() - loss_rows.detach()).abs().max().item() / max(1.0, loss_rows.abs().max().item()))
            err = (x.double().cpu() - xr.grad).abs().max().item() / scale
            worst = max(worst, err)
        return worst <= tol, f"worst normalized error {worst:.2e} (tolerance {tol:.0e})"

    c.check(
        "softmax_xent_grad_torch_ matches float64 (float32 logits)",
        lambda: stage1(submission.softmax_xent_grad_torch_, torch.float32, 1e-5),
    )
    if on_cuda:
        c.check(
            "softmax_xent_grad_torch_ matches float64 (bfloat16 logits)",
            lambda: stage1(submission.softmax_xent_grad_torch_, torch.bfloat16, 1.2e-2),
        )
        c.check(
            "softmax_xent_grad_triton_ matches float64 (float32 logits)",
            lambda: stage1(submission.softmax_xent_grad_triton_, torch.float32, 1e-5),
        )
        c.check(
            "softmax_xent_grad_triton_ matches float64 (bfloat16 logits)",
            lambda: stage1(submission.softmax_xent_grad_triton_, torch.bfloat16, 1.2e-2),
        )
    else:
        print("      note: no GPU, so the Triton kernel and the benchmarks are skipped.", flush=True)

    # --- Stage 2: the Function, against float64 ------------------------------
    def fused_case(shape, dtype, use_triton, loss_tol, grad_tol):
        n, d, v, chunk = shape
        h, w, t = make_case(n, d, v, dtype, device)
        h.requires_grad_()
        w.requires_grad_()
        loss = submission.fused_linear_xent(h, w, t, chunk_size=chunk, use_triton=use_triton)
        loss.backward()
        ref_loss, ref_gh, ref_gw = reference64(h, w, t)
        e_loss = abs(loss.item() - ref_loss.item()) / max(1.0, abs(ref_loss.item()))
        e_h, e_w = rel_err(h.grad, ref_gh), rel_err(w.grad, ref_gw)
        ok = e_loss <= loss_tol and e_h <= grad_tol and e_w <= grad_tol
        return ok, f"loss err {e_loss:.1e}, grad_h rel err {e_h:.1e}, grad_w rel err {e_w:.1e}"

    for shape in SHAPES:
        c.check(
            f"fused loss and gradients, float32, N={shape[0]} V={shape[2]} chunk={shape[3]}",
            lambda shape=shape: fused_case(shape, torch.float32, False, 1e-5, 1e-4),
        )
    if on_cuda:
        for shape in SHAPES:
            c.check(
                f"fused loss and gradients, float32 with Triton, N={shape[0]} V={shape[2]} chunk={shape[3]}",
                lambda shape=shape: fused_case(shape, torch.float32, True, 1e-5, 1e-4),
            )
        for shape in [SHAPES[2], SHAPES[3], SHAPES[4]]:
            for use_triton in (False, True):
                c.check(
                    f"fused bfloat16 close to float64, N={shape[0]} V={shape[2]}, triton={use_triton}",
                    lambda shape=shape, use_triton=use_triton: fused_case(shape, torch.bfloat16, use_triton, 3e-2, 5e-2),
                )

    def chunk_invariance():
        n, d, v, _ = SHAPES[3]
        h, w, t = make_case(n, d, v, torch.float64, device)
        losses = [submission.fused_linear_xent(h, w, t, chunk_size=k).item() for k in (1, 7, 64, n, n + 5)]
        spread = max(losses) - min(losses)
        return spread < 1e-10, f"losses differ by {spread:.1e} across chunk sizes 1, 7, 64, N, N+5"

    c.check("the loss does not depend on the chunk size", chunk_invariance)

    def gradcheck():
        torch.manual_seed(0)
        h = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
        w = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
        t = torch.tensor([1, 6, IGNORE, 0, 3])
        fn = lambda a, b: submission.fused_linear_xent(a, b, t, chunk_size=2)  # noqa: E731
        return torch.autograd.gradcheck(fn, (h, w), eps=1e-6, atol=1e-6, rtol=1e-5), "torch.autograd.gradcheck in float64"

    c.check("gradcheck passes in float64", gradcheck)

    def double_backward_scale():
        h, w, t = make_case(16, 8, 50, torch.float64, device)
        h.requires_grad_()
        w.requires_grad_()
        (3.0 * submission.fused_linear_xent(h, w, t, chunk_size=5)).backward()
        _, ref_gh, ref_gw = reference64(h, w, t)
        return (
            rel_err(h.grad, 3 * ref_gh) < 1e-9 and rel_err(w.grad, 3 * ref_gw) < 1e-9,
            "backward scales the saved gradients by the incoming gradient",
        )

    c.check("backward multiplies by the incoming gradient", double_backward_scale)

    if not on_cuda:
        return c.finish()

    # --- Stage 3: memory and time on the GPU ---------------------------------
    n, d, v, chunk = BENCH_N, BENCH_D, BENCH_V, BENCH_CHUNK
    h, w, t = make_case(n, d, v, torch.bfloat16, device)
    h.requires_grad_()
    w.requires_grad_()

    def step(fn):
        h.grad = None
        w.grad = None
        fn(h, w, t).backward()

    def peak_extra_bytes(fn):
        step(fn)  # warm up: Triton compile, cuBLAS workspaces, allocator pools
        h.grad = None
        w.grad = None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        step(fn)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - base

    ref_fn = submission.reference_linear_xent
    fused_triton = lambda a, b, tt: submission.fused_linear_xent(a, b, tt, chunk_size=chunk, use_triton=True)  # noqa: E731
    fused_torch = lambda a, b, tt: submission.fused_linear_xent(a, b, tt, chunk_size=chunk, use_triton=False)  # noqa: E731

    ref_peak = peak_extra_bytes(ref_fn)
    fused_peak = peak_extra_bytes(fused_triton)
    logits_bf16 = n * v * 2
    c.metric("ref_peak_mb", round(ref_peak / 2**20, 1))
    c.metric("fused_peak_mb", round(fused_peak / 2**20, 1))
    c.metric("memory_ratio", round(ref_peak / max(fused_peak, 1), 2))
    c.add(
        "the fused path never holds the full logits",
        fused_peak < logits_bf16,
        f"peak extra {fused_peak / 2**20:.0f} MiB; the full bf16 logits alone would be {logits_bf16 / 2**20:.0f} MiB",
    )
    c.add(
        "the fused path peaks far below the unfused one",
        fused_peak < 0.5 * ref_peak,
        f"fused {fused_peak / 2**20:.0f} MiB, unfused {ref_peak / 2**20:.0f} MiB",
    )

    ref_ms = cuda_time(lambda: step(ref_fn), warmup=3, iters=10)
    fused_ms = cuda_time(lambda: step(fused_triton), warmup=3, iters=10)
    torch_ms = cuda_time(lambda: step(fused_torch), warmup=3, iters=10)
    flops = 6.0 * n * d * v  # three GEMMs of 2*N*d*V: logits, grad_h, grad_w
    c.metric("ref_ms", round(ref_ms, 2))
    c.metric("fused_ms", round(fused_ms, 2))
    c.metric("fused_torch_path_ms", round(torch_ms, 2))
    c.metric("speed_ratio", round(ref_ms / fused_ms, 2))
    c.metric("fused_tflops", round(tflops(flops, fused_ms), 1))
    c.add(
        "the fused path is within 2x of the unfused time",
        fused_ms <= 2.0 * ref_ms,
        f"fused {fused_ms:.2f} ms, unfused {ref_ms:.2f} ms. It does the same three GEMMs, so a large gap points at the chunk loop.",
    )

    # --- Optional: PyTorch's own chunked op, when this build has it ----------
    lce = getattr(F, "linear_cross_entropy", None)
    options_cls = getattr(torch.nn, "LinearCrossEntropyOptions", None)
    if lce is None or options_cls is None:
        print("      note: this PyTorch has no linear_cross_entropy, so the comparison is skipped.", flush=True)
    else:
        try:
            # batch_chunk_size and a chunking_method heuristic can't both be set, so turn the heuristic off.
            opts = options_cls(batch_chunk_size=chunk, chunking_method=None)
            torch_lce = lambda a, b, tt: lce(a, b, tt, options=opts)  # noqa: E731
            lce_peak = peak_extra_bytes(torch_lce)
            lce_ms = cuda_time(lambda: step(torch_lce), warmup=3, iters=10)
            c.metric("torch_lce_peak_mb", round(lce_peak / 2**20, 1))
            c.metric("torch_lce_ms", round(lce_ms, 2))
        except Exception as exc:  # noqa: BLE001
            print(f"      note: linear_cross_entropy did not run here ({type(exc).__name__}: {exc}); skipped.", flush=True)

    return c.finish()
