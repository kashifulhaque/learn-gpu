"""Chapter 19 lab, worked solution: a fused, chunked linear plus cross-entropy.

The unfused loss builds the full logits tensor, `tokens x vocab`, and autograd
keeps it (and the softmax derived from it) until the backward pass. This file
builds what `torch.nn.functional.linear_cross_entropy` does with chunking on:
it walks the tokens in chunks, computes each chunk's logits, turns them into a
loss and a gradient in place, uses that gradient at once to form the chunk's
share of the gradients for the hidden states and the weight, and throws the
logits away. Only one chunk of logits exists at a time.
"""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

try:  # Triton is in the lab image; a laptop without it still runs the torch path.
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

IGNORE_INDEX = -100
BLOCK_V = 1024


# ---------------------------------------------------------------------------
# Stage 1: loss and gradient of one chunk of logits, in place
# ---------------------------------------------------------------------------


def softmax_xent_grad_torch_(
    logits: torch.Tensor, targets: torch.Tensor, scale: float, ignore_index: int = IGNORE_INDEX
) -> torch.Tensor:
    """Turn a chunk of logits into its gradient, in place, and return per-row losses.

    `logits` is `[rows, vocab]`, contiguous. On return it holds
    `scale * (softmax(logits) - onehot(targets))`, with rows whose target is
    `ignore_index` set to zero. The return value is a float32 (or float64)
    vector of per-row losses `logsumexp(row) - row[target]`, zero for ignored rows.
    """
    work = torch.float64 if logits.dtype == torch.float64 else torch.float32
    x = logits.to(work)  # a copy for half precision; the same tensor for float32/64
    valid = targets != ignore_index
    safe = torch.where(valid, targets, torch.zeros_like(targets))
    lse = torch.logsumexp(x, dim=1)
    target_logit = x.gather(1, safe[:, None]).squeeze(1)
    row_loss = torch.where(valid, lse - target_logit, torch.zeros_like(lse))
    x.sub_(lse[:, None]).exp_()  # softmax, reusing x's storage
    x[torch.arange(x.shape[0], device=x.device), safe] -= 1.0
    x.mul_(scale)
    x.mul_(valid[:, None].to(work))
    logits.copy_(x)
    return row_loss


if HAVE_TRITON:

    @triton.jit
    def _xent_grad_kernel(
        logits_ptr,
        targets_ptr,
        loss_ptr,
        stride,
        V,
        scale,
        ignore_index,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        base = logits_ptr + row.to(tl.int64) * stride
        target = tl.load(targets_ptr + row)
        offs = tl.arange(0, BLOCK)

        # Pass 1: online logsumexp. m is the running maximum, s the running sum
        # of exp(x - m). When m grows, s is rescaled by exp(m_old - m_new).
        m = tl.max(tl.full([BLOCK], float("-inf"), tl.float32), axis=0)
        s = tl.sum(tl.zeros([BLOCK], tl.float32), axis=0)
        for start in range(0, V, BLOCK):
            cols = start + offs
            x = tl.load(base + cols, mask=cols < V, other=float("-inf")).to(tl.float32)
            m_new = tl.maximum(m, tl.max(x, axis=0))
            s = s * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
            m = m_new
        lse = m + tl.log(s)

        # The target's logit must be read before pass 2 overwrites it.
        valid = target != ignore_index
        safe = tl.where(valid, target, 0)
        target_logit = tl.load(base + safe).to(tl.float32)
        tl.store(loss_ptr + row, tl.where(valid, lse - target_logit, 0.0))

        # Pass 2: write scale * (softmax - onehot) over the logits.
        for start in range(0, V, BLOCK):
            cols = start + offs
            mask = cols < V
            x = tl.load(base + cols, mask=mask, other=float("-inf")).to(tl.float32)
            p = tl.exp(x - lse)
            p = tl.where(cols == safe, p - 1.0, p) * scale
            p = tl.where(valid, p, 0.0)
            tl.store(base + cols, p.to(logits_ptr.dtype.element_ty), mask=mask)


def softmax_xent_grad_triton_(
    logits: torch.Tensor, targets: torch.Tensor, scale: float, ignore_index: int = IGNORE_INDEX
) -> torch.Tensor:
    """The same contract as `softmax_xent_grad_torch_`, as one Triton program per row.

    The kernel works on the logits in their own dtype, so a bfloat16 chunk is
    overwritten in place with no float32 copy.
    """
    assert HAVE_TRITON and logits.is_cuda and logits.is_contiguous()
    rows, vocab = logits.shape
    row_loss = torch.empty(rows, dtype=torch.float32, device=logits.device)
    if rows == 0:
        return row_loss
    _xent_grad_kernel[(rows,)](
        logits, targets, row_loss, logits.stride(0), vocab, scale, ignore_index, BLOCK=BLOCK_V, num_warps=8
    )
    return row_loss


# ---------------------------------------------------------------------------
# Stage 2: the autograd Function
# ---------------------------------------------------------------------------


def _accumulate_weight_grad(grad_w: torch.Tensor, dlogits: torch.Tensor, h_c: torch.Tensor) -> torch.Tensor:
    """Return `grad_w + dlogits.T @ h_c`, accumulated in grad_w's (float32) dtype.

    On CUDA, `torch.addmm(..., out_dtype=torch.float32)` multiplies bfloat16
    operands and adds into a float32 matrix without rounding the product to
    bfloat16 first. Where that is unavailable, the product is rounded to the
    operand dtype and then added in float32: one rounding per chunk, which the
    tests' tolerances allow.
    """
    if dlogits.dtype != grad_w.dtype and dlogits.is_cuda:
        try:
            return torch.addmm(grad_w, dlogits.t(), h_c, out_dtype=grad_w.dtype)
        except (TypeError, RuntimeError, NotImplementedError):
            pass
    return grad_w + (dlogits.t() @ h_c).to(grad_w.dtype)


class FusedLinearXent(torch.autograd.Function):
    """`mean cross_entropy(h @ w.T, targets)` without the full logits tensor."""

    @staticmethod
    def forward(ctx, h, w, targets, chunk_size, ignore_index, use_triton):
        n, d = h.shape
        vocab = w.shape[0]
        acc = torch.float64 if h.dtype == torch.float64 else torch.float32
        n_valid = int((targets != ignore_index).sum())
        scale = 1.0 / max(n_valid, 1)
        grad_row_fn = softmax_xent_grad_triton_ if use_triton else softmax_xent_grad_torch_

        grad_h = torch.empty_like(h)
        grad_w = torch.zeros(vocab, d, dtype=acc, device=h.device)
        loss = torch.zeros((), dtype=acc, device=h.device)
        for start in range(0, n, chunk_size):
            h_c = h[start : start + chunk_size]
            t_c = targets[start : start + chunk_size]
            logits = h_c @ w.t()  # [chunk, vocab], the only logits that exist
            loss += grad_row_fn(logits, t_c, scale, ignore_index).sum(dtype=acc)
            # `logits` now holds d(loss)/d(logits) for this chunk.
            grad_h[start : start + chunk_size] = logits @ w
            grad_w = _accumulate_weight_grad(grad_w, logits, h_c)
        loss *= scale  # the mean over valid tokens; the gradients already carry this scale
        ctx.save_for_backward(grad_h, grad_w)
        ctx.w_dtype = w.dtype
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss):
        grad_h, grad_w = ctx.saved_tensors
        g = grad_loss.to(grad_h.dtype)
        return grad_h * g, (grad_w * grad_loss.to(grad_w.dtype)).to(ctx.w_dtype), None, None, None, None


def fused_linear_xent(
    h: torch.Tensor,
    w: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 512,
    ignore_index: int = IGNORE_INDEX,
    use_triton: bool = False,
) -> torch.Tensor:
    """Mean cross-entropy of `h @ w.T` against `targets`, chunked over tokens."""
    return FusedLinearXent.apply(h, w, targets, chunk_size, ignore_index, use_triton)


def reference_linear_xent(
    h: torch.Tensor, w: torch.Tensor, targets: torch.Tensor, ignore_index: int = IGNORE_INDEX
) -> torch.Tensor:
    """The unfused loss: materialize every logit, in float32, and let autograd keep it."""
    logits = (h @ w.t()).float()
    return torch.nn.functional.cross_entropy(logits, targets, ignore_index=ignore_index)
