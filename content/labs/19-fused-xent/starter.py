"""Chapter 19 lab: a fused, chunked linear plus cross-entropy.

The unfused loss builds the full logits tensor, `tokens x vocab`, and autograd
keeps it until the backward pass. You write what
`torch.nn.functional.linear_cross_entropy` does with chunking on: walk the
tokens in chunks, turn each chunk's logits into a loss and a gradient in place,
use that gradient at once for the chunk's share of the gradients of the hidden
states and the weight, and discard the logits. Only one chunk of logits exists
at a time.

You fill in three pieces:

  1. `softmax_xent_grad_torch_`: one chunk of logits to its gradient, in PyTorch.
  2. `softmax_xent_grad_triton_` and its kernel: the same, one Triton program
     per row, with an online logsumexp so the row is read in blocks.
  3. `FusedLinearXent`: the autograd Function that loops over chunks.

Run the tests to see which checks pass. Work in that order; each check gives a
clue to the next.
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
BLOCK_V = 1024  # a power of two: tl.arange needs one


# ---------------------------------------------------------------------------
# Stage 1: loss and gradient of one chunk of logits, in place
# ---------------------------------------------------------------------------


def softmax_xent_grad_torch_(
    logits: torch.Tensor, targets: torch.Tensor, scale: float, ignore_index: int = IGNORE_INDEX
) -> torch.Tensor:
    """Turn a chunk of logits into its gradient, in place, and return per-row losses.

    `logits` is `[rows, vocab]` and contiguous. On return it must hold
    `scale * (softmax(logits) - onehot(targets))`, and rows whose target is
    `ignore_index` must be all zero. The return value is a float32 (float64 for
    float64 input) vector of per-row losses, `logsumexp(row) - row[target]`,
    zero for ignored rows.

    Do the arithmetic in float32 even when `logits` is bfloat16, then write the
    result back into `logits` with `copy_`.

    TODO: implement it. Use `torch.logsumexp`, `gather`, and in-place ops so the
    float32 copy is the only extra [rows, vocab] tensor.
    """
    raise NotImplementedError("softmax_xent_grad_torch_")


# ---------------------------------------------------------------------------
# Stage 2: the same, in Triton
# ---------------------------------------------------------------------------

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
        base = logits_ptr + row.to(tl.int64) * stride  # int64: row * stride can pass 2**31
        target = tl.load(targets_ptr + row)
        offs = tl.arange(0, BLOCK)

        # Pass 1: online logsumexp over the row, BLOCK columns at a time.
        # Keep a running maximum m and a running sum s of exp(x - m), both
        # scalars. For each block: m_new = max(m, block max);
        # s = s * exp(m - m_new) + sum(exp(x - m_new)); m = m_new.
        # TODO: write the loop, then set lse = m + log(s).
        lse = 0.0

        # Read the target's logit now: pass 2 overwrites the row.
        valid = target != ignore_index
        safe = tl.where(valid, target, 0)
        target_logit = tl.load(base + safe).to(tl.float32)
        tl.store(loss_ptr + row, tl.where(valid, lse - target_logit, 0.0))

        # Pass 2: write scale * (softmax - onehot) over the row, in the logits'
        # own dtype, zero if the row is ignored.
        # TODO: loop over blocks again. p = exp(x - lse); subtract 1 where
        # cols == safe; multiply by scale; zero if not valid; store with a mask.


def softmax_xent_grad_triton_(
    logits: torch.Tensor, targets: torch.Tensor, scale: float, ignore_index: int = IGNORE_INDEX
) -> torch.Tensor:
    """The same contract as `softmax_xent_grad_torch_`, as one Triton program per row.

    TODO: allocate a float32 `row_loss` of length `rows` on the logits' device
    and launch `_xent_grad_kernel[(rows,)]` with `BLOCK=BLOCK_V` and
    `num_warps=8`. Pass `logits.stride(0)` for the row stride. Return `row_loss`.
    """
    raise NotImplementedError("softmax_xent_grad_triton_")


# ---------------------------------------------------------------------------
# Stage 3: the autograd Function
# ---------------------------------------------------------------------------


class FusedLinearXent(torch.autograd.Function):
    """`mean cross_entropy(h @ w.T, targets)` without the full logits tensor."""

    @staticmethod
    def forward(ctx, h, w, targets, chunk_size, ignore_index, use_triton):
        """h: [N, d], w: [V, d], targets: [N] long. Returns a scalar loss.

        TODO:
          * acc = float64 if h is float64, else float32. The loss and the weight
            gradient accumulate in `acc`.
          * n_valid = number of targets != ignore_index; scale = 1 / max(n_valid, 1).
          * grad_h = torch.empty_like(h); grad_w = zeros [V, d] in `acc`;
            loss = zero scalar in `acc`.
          * For each chunk of rows: logits = h_c @ w.T; call the row function
            (the Triton one if `use_triton`) to get row losses and overwrite
            `logits` with its gradient; add the row losses to `loss`;
            grad_h[chunk] = logits @ w; grad_w += (logits.T @ h_c) in `acc`.
          * Multiply `loss` by `scale` (the mean), save grad_h and grad_w with
            ctx.save_for_backward, remember w.dtype on ctx, return `loss`.
        """
        raise NotImplementedError("FusedLinearXent.forward")

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss):
        """Scale the saved gradients by the incoming gradient of the loss.

        Return one value per argument of `forward`: grad for h (in h's dtype),
        grad for w (in w's dtype), then None for targets, chunk_size,
        ignore_index, and use_triton.

        TODO: implement it.
        """
        raise NotImplementedError("FusedLinearXent.backward")


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
    """The unfused loss, given: materialize every logit in float32 and let autograd keep it.

    The tests time and measure this as the baseline. Don't change it.
    """
    logits = (h @ w.t()).float()
    return torch.nn.functional.cross_entropy(logits, targets, ignore_index=ignore_index)
