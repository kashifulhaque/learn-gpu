"""Tests for lab 15: online softmax and LayerNorm forward and backward."""

import torch
import torch.nn.functional as F
from lab_common import A100, Checks, bandwidth_gbs, cuda_time, require_cuda

# Thresholds are fractions of the measured 1275 GB/s copy ceiling
# (learn-inference chapter 10). They are loose on purpose: they catch a kernel
# that is far from the roofline, such as one that makes an extra pass over HBM.
SOFTMAX_FRACTION = 0.40
FWD_FRACTION = 0.40
BWD_FRACTION = 0.30


def rel_close(actual, expected, rtol, floor=1e-6):
    """Pass when the largest absolute error is at most rtol times the largest |expected|.

    `floor` keeps the limit from collapsing when every expected value is near zero.
    """
    actual = actual.double().cpu()
    expected = expected.double().cpu()
    if actual.shape != expected.shape:
        return False, f"shape {tuple(actual.shape)}, expected {tuple(expected.shape)}"
    scale = max(expected.abs().max().item(), floor)
    err = (actual - expected).abs().max().item()
    return err <= rtol * scale, f"max abs error {err:.3e}, {err / scale:.2e} of the largest value (limit {rtol:.0e})"


class _LayerNorm(torch.autograd.Function):
    """The submission's kernels wrapped as an autograd function."""

    submission = None

    @staticmethod
    def forward(ctx, x, w, b, eps):
        y, mean, rstd = _LayerNorm.submission.layernorm_forward(x, w, b, eps)
        ctx.save_for_backward(x, w, mean, rstd)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, mean, rstd = ctx.saved_tensors
        dx, dw, db = _LayerNorm.submission.layernorm_backward(dy, x, w, mean, rstd)
        return dx, dw, db, None


def make(shape, dtype, device, seed, offset=0.0, scale=1.0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(shape, generator=g) * scale + offset).to(device=device, dtype=dtype)


def run(submission):
    c = Checks()
    if not c.require(submission, "softmax_online", "layernorm_forward", "layernorm_backward"):
        return c.finish()
    try:
        _run(submission, c)
    except Exception as exc:  # noqa: BLE001 - an unfinished kernel must fail the lab, not crash the harness
        c.add("tests ran to the end", False, f"{type(exc).__name__}: {exc}; later checks were skipped")
    return c.finish()


def _run(submission, c):
    device = require_cuda()
    _LayerNorm.submission = submission

    # ---- Softmax ----------------------------------------------------------
    # Includes size 1, sizes that are not multiples of the block (4096 is the
    # largest block, so 4097 and 10000 need several chunks plus a ragged tail),
    # and a row of 70000 columns that streams through many chunks.
    for rows, cols in [(1, 1), (1, 7), (3, 1000), (5, 4097), (4, 10000), (2, 70000)]:
        x = make((rows, cols), torch.float32, device, seed=cols)
        got = submission.softmax_online(x)
        want = torch.softmax(x.double(), dim=-1)
        c.check(f"softmax fp32, {rows} x {cols}", lambda: rel_close(got, want, 1e-4))

    x = make((4, 5000), torch.bfloat16, device, seed=11, scale=3.0)
    got = submission.softmax_online(x)
    c.add("softmax bf16 keeps the input dtype", got.dtype == torch.bfloat16, f"got {got.dtype}")
    want = torch.softmax(x.double(), dim=-1)
    c.check("softmax bf16, 4 x 5000", lambda: rel_close(got, want, 2e-2))

    # Large logits overflow exp() unless the maximum is subtracted. This also
    # catches a missing rescale when a later chunk holds the maximum.
    x = make((3, 9000), torch.float32, device, seed=5, scale=30.0, offset=1000.0)
    x[0, 8999] = 1200.0  # the maximum sits in the last, ragged chunk
    got = submission.softmax_online(x)
    want = torch.softmax(x.double(), dim=-1)
    c.check("softmax survives logits near 1000 and a late maximum", lambda: rel_close(got, want, 1e-4))
    c.check(
        "softmax rows sum to 1",
        lambda: (
            (got.double().sum(-1) - 1).abs().max().item() < 1e-4,
            f"largest |sum - 1| = {(got.double().sum(-1) - 1).abs().max().item():.2e}",
        ),
    )

    # ---- LayerNorm forward ------------------------------------------------
    eps = 1e-5
    for rows, cols in [(1, 1), (1, 7), (3, 1000), (5, 4097), (2, 20000)]:
        x = make((rows, cols), torch.float32, device, seed=cols)
        w = make((cols,), torch.float32, device, seed=cols + 1)
        b = make((cols,), torch.float32, device, seed=cols + 2)
        y, mean, rstd = submission.layernorm_forward(x, w, b, eps)
        want = F.layer_norm(x.double(), (cols,), w.double(), b.double(), eps)
        c.check(f"layernorm forward y, fp32, {rows} x {cols}", lambda: rel_close(y, want, 1e-4))
        c.check(
            f"layernorm forward mean and rstd, {rows} x {cols}",
            lambda: (
                rel_close(mean, x.double().mean(-1), 1e-4, floor=1.0)[0]
                and rel_close(rstd, 1.0 / torch.sqrt(x.double().var(-1, unbiased=False) + eps), 1e-4)[0],
                "",
            ),
        )

    # A large mean exposes the E[x^2] - E[x]^2 shortcut: the variance is 1 but
    # x^2 is about 1e6, which float32 cannot subtract accurately.
    cols = 4096
    x = make((4, cols), torch.float32, device, seed=3, offset=1000.0)
    w = torch.ones(cols, device=device)
    b = torch.zeros(cols, device=device)
    y, _, _ = submission.layernorm_forward(x, w, b, eps)
    want = F.layer_norm(x.double(), (cols,), None, None, eps)
    c.check("layernorm forward with a mean of 1000 (Welford)", lambda: rel_close(y, want, 1e-3))

    x = make((6, 3000), torch.bfloat16, device, seed=8, scale=2.0)
    w = make((3000,), torch.bfloat16, device, seed=9)
    b = make((3000,), torch.bfloat16, device, seed=10)
    y, mean, rstd = submission.layernorm_forward(x, w, b, eps)
    c.add("layernorm bf16 y keeps the input dtype", y.dtype == torch.bfloat16, f"got {y.dtype}")
    c.add("mean and rstd are float32", mean.dtype == torch.float32 and rstd.dtype == torch.float32)
    want = F.layer_norm(x.double(), (3000,), w.double(), b.double(), eps)
    c.check("layernorm forward bf16, 6 x 3000", lambda: rel_close(y, want, 3e-2))

    # ---- LayerNorm backward, against torch.autograd ------------------------
    # Row counts above 432 (4 programs per SM) make each program loop over
    # several rows; the small ones make some programs own a single row.
    for dtype, rtol in [(torch.float32, 1e-4), (torch.bfloat16, 3e-2)]:
        for rows, cols in [(1, 1), (3, 7), (37, 1000), (600, 4097), (1000, 513)]:
            x = make((rows, cols), dtype, device, seed=cols, scale=2.0).requires_grad_(True)
            w = make((cols,), dtype, device, seed=cols + 1).requires_grad_(True)
            b = make((cols,), dtype, device, seed=cols + 2).requires_grad_(True)
            dy = make((rows, cols), dtype, device, seed=cols + 3)

            x64 = x.detach().double().requires_grad_(True)
            w64 = w.detach().double().requires_grad_(True)
            b64 = b.detach().double().requires_grad_(True)
            F.layer_norm(x64, (cols,), w64, b64, eps).backward(dy.double())

            _LayerNorm.apply(x, w, b, eps).backward(dy)
            name = str(dtype).replace("torch.", "")
            c.check(f"layernorm dx, {name}, {rows} x {cols}", lambda: rel_close(x.grad, x64.grad, rtol, floor=1e-2))
            c.check(f"layernorm dgamma, {name}, {rows} x {cols}", lambda: rel_close(w.grad, w64.grad, rtol, floor=1e-2))
            c.check(f"layernorm dbeta, {name}, {rows} x {cols}", lambda: rel_close(b.grad, b64.grad, rtol, floor=1e-2))

    # ---- Performance --------------------------------------------------------
    # bfloat16 tensors of 268 MB (softmax) and 134 MB (LayerNorm), far larger
    # than the 40 MiB L2, so the timing reflects HBM and needs no cache flush.
    ceiling = A100.COPY_CEILING_GBS

    rows, cols = 8192, 16384
    x = torch.randn(rows, cols, device=device, dtype=torch.bfloat16)
    tensor_bytes = x.numel() * x.element_size()
    ms = cuda_time(lambda: submission.softmax_online(x), warmup=5, iters=20)
    gbs = bandwidth_gbs(2 * tensor_bytes, ms)  # one read and one write of the tensor
    ref_ms = cuda_time(lambda: torch.softmax(x, dim=-1), warmup=5, iters=20)
    c.metric("softmax_gbs", round(gbs, 1))
    c.metric("softmax_vs_torch", round(ref_ms / ms, 2))
    print(f"      softmax floor at the copy ceiling: {2 * tensor_bytes / (ceiling * 1e9) * 1e3:.3f} ms; yours {ms:.3f} ms", flush=True)
    c.add(
        f"softmax reaches {SOFTMAX_FRACTION:.0%} of the copy ceiling",
        gbs >= SOFTMAX_FRACTION * ceiling,
        f"{gbs:.0f} GB/s against {SOFTMAX_FRACTION * ceiling:.0f} GB/s",
    )
    del x

    rows, cols = 16384, 4096
    x = torch.randn(rows, cols, device=device, dtype=torch.bfloat16)
    w = torch.randn(cols, device=device, dtype=torch.bfloat16)
    b = torch.randn(cols, device=device, dtype=torch.bfloat16)
    dy = torch.randn(rows, cols, device=device, dtype=torch.bfloat16)
    tensor_bytes = x.numel() * x.element_size()

    ms = cuda_time(lambda: submission.layernorm_forward(x, w, b, eps), warmup=5, iters=20)
    gbs = bandwidth_gbs(2 * tensor_bytes, ms)  # read x, write y
    c.metric("layernorm_fwd_gbs", round(gbs, 1))
    c.add(
        f"LayerNorm forward reaches {FWD_FRACTION:.0%} of the copy ceiling",
        gbs >= FWD_FRACTION * ceiling,
        f"{gbs:.0f} GB/s against {FWD_FRACTION * ceiling:.0f} GB/s",
    )

    _, mean, rstd = submission.layernorm_forward(x, w, b, eps)
    ms = cuda_time(lambda: submission.layernorm_backward(dy, x, w, mean, rstd), warmup=5, iters=20)
    gbs = bandwidth_gbs(3 * tensor_bytes, ms)  # read x and dy, write dx
    c.metric("layernorm_bwd_gbs", round(gbs, 1))
    print(f"      backward floor at the copy ceiling: {3 * tensor_bytes / (ceiling * 1e9) * 1e3:.3f} ms; yours {ms:.3f} ms", flush=True)
    c.add(
        f"LayerNorm backward reaches {BWD_FRACTION:.0%} of the copy ceiling",
        gbs >= BWD_FRACTION * ceiling,
        f"{gbs:.0f} GB/s against {BWD_FRACTION * ceiling:.0f} GB/s",
    )
