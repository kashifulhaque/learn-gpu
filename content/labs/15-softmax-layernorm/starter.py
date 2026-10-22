"""Lab 15: Softmax and LayerNorm.

You write three functions for 2D, contiguous CUDA tensors of shape (rows, cols):

    softmax_online(x)                          -> y
    layernorm_forward(x, w, b, eps)            -> y, mean, rstd
    layernorm_backward(dy, x, w, mean, rstd)   -> dx, dw, db

The Python wrappers are written for you. Fill in the four Triton kernels marked
TODO. Every kernel handles a row of any length, so every load and store needs a
mask, and `other=` must be the identity of the reduction that follows.

Run the lab from the app, or call your functions on a GPU yourself and compare
them with torch.softmax and torch.nn.functional.layer_norm.
"""

import torch
import triton
import triton.language as tl


def _block_size(n_cols: int, cap: int) -> int:
    """A power of two that covers the row, or `cap` if the row is longer."""
    return min(cap, max(16, triton.next_power_of_2(n_cols)))


def _num_warps(block: int) -> int:
    """One warp per 256 elements, from 1 to 8 warps."""
    return min(max(block // 256, 1), 8)


# ---------------------------------------------------------------------------
# Part 1: online softmax for rows that don't fit one block
# ---------------------------------------------------------------------------


@triton.jit
def _softmax_online_kernel(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    """One program computes one row.

    Pass 1 loops over the row in chunks of BLOCK columns and keeps two scalars:
      m  the running maximum of everything seen so far
      d  the running denominator, sum(exp(x - m)) over everything seen so far
    When a chunk raises the maximum from m to m_new, the old d is a sum of
    exp(x - m) terms and must be multiplied by exp(m - m_new) first.

    Pass 2 loops again, writes exp(x - m) / d, and casts to y's dtype.
    Do the arithmetic in float32, whatever the input dtype is.
    """
    row = tl.program_id(0).to(tl.int64)
    x_row = x_ptr + row * n_cols
    y_row = y_ptr + row * n_cols
    # TODO: pass 1, the running (m, d). Masked lanes load -inf.
    # TODO: pass 2, normalize and store.
    pass


def softmax_online(x: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax of a (rows, cols) tensor, in the input's dtype."""
    assert x.is_cuda and x.ndim == 2
    x = x.contiguous()
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    block = _block_size(n_cols, cap=4096)
    _softmax_online_kernel[(n_rows,)](x, y, n_cols, BLOCK=block, num_warps=_num_warps(block))
    return y


# ---------------------------------------------------------------------------
# Part 2: LayerNorm forward, with Welford statistics by chunks
# ---------------------------------------------------------------------------


@triton.jit
def _layernorm_fwd_kernel(
    x_ptr, y_ptr, w_ptr, b_ptr, mean_ptr, rstd_ptr, n_cols, eps, BLOCK: tl.constexpr
):
    """One program normalizes one row: y = (x - mean) * rstd * w + b.

    Pass 1 loops over the row in chunks and merges each chunk into a running
    (count, mean, M2), where M2 is the sum of squared deviations from the mean.
    For a chunk with n_b valid elements, mean mean_b and M2 value m2_b, and a
    running state (n_a, mean, m2):
        count = n_a + n_b
        delta = mean_b - mean
        mean  = mean + delta * n_b / count
        m2    = m2 + m2_b + delta * delta * n_a * n_b / count
    Compute m2_b from the deviations from mean_b, not from raw squares, and
    zero the masked lanes after you subtract. Then var = m2 / n_cols and
    rstd = 1 / sqrt(var + eps). Store mean and rstd for the backward pass.

    Pass 2 loops again and writes y, casting to y's dtype.
    """
    row = tl.program_id(0).to(tl.int64)
    x_row = x_ptr + row * n_cols
    y_row = y_ptr + row * n_cols
    # TODO: pass 1, the merge.
    # TODO: store mean and rstd for this row.
    # TODO: pass 2, normalize, scale, shift, and store.
    pass


def layernorm_forward(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float = 1e-5):
    """LayerNorm over the last dimension. Returns y (in x's dtype), mean, and rstd (float32)."""
    assert x.is_cuda and x.ndim == 2
    x = x.contiguous()
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    mean = torch.empty(n_rows, dtype=torch.float32, device=x.device)
    rstd = torch.empty(n_rows, dtype=torch.float32, device=x.device)
    block = _block_size(n_cols, cap=4096)
    _layernorm_fwd_kernel[(n_rows,)](
        x, y, w, b, mean, rstd, n_cols, eps, BLOCK=block, num_warps=_num_warps(block)
    )
    return y, mean, rstd


# ---------------------------------------------------------------------------
# Part 3: LayerNorm backward, with a two-stage dw and db reduction
# ---------------------------------------------------------------------------


@triton.jit
def _layernorm_bwd_dx_kernel(
    dy_ptr, x_ptr, w_ptr, mean_ptr, rstd_ptr, dx_ptr, dw_part_ptr, db_part_ptr,
    n_rows, n_cols, BLOCK: tl.constexpr,
):
    """Stage 1. Program `pid` handles rows pid, pid + G, pid + 2G, ..., where G is the grid size.

    The whole row sits in one block (BLOCK >= n_cols). For each row:
        xhat = (x - mean) * rstd          (zero in masked lanes)
        g    = dy * w
        c1   = sum(xhat * g) / n_cols
        c2   = sum(g) / n_cols
        dx   = (g - xhat * c1 - c2) * rstd
    Keep running sums of dy * xhat and dy in float32 registers across the rows
    the program owns, and store them at the end as row `pid` of dw_part and
    db_part, which have shape (G, n_cols). No atomics, no locks.
    """
    pid = tl.program_id(0)
    groups = tl.num_programs(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    # TODO: load w, zero the accumulators, loop over this program's rows,
    # store dx, and store the two partial rows.
    pass


@triton.jit
def _layernorm_bwd_reduce_kernel(
    dw_part_ptr, db_part_ptr, dw_ptr, db_ptr, groups, n_cols,
    BLOCK_G: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Stage 2. Program `pid` sums the G partial rows over BLOCK_N columns.

    Loop over the partial rows in chunks of BLOCK_G with a 2D load of shape
    (BLOCK_G, BLOCK_N), sum over axis 0, accumulate in float32, and store
    dw and db in their own dtype.
    """
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < n_cols
    # TODO: the reduction over the partial rows.
    pass


def layernorm_backward(dy, x, w, mean, rstd):
    """Gradients of LayerNorm. Returns dx (x's dtype), dw and db (w's dtype)."""
    assert x.is_cuda and x.ndim == 2
    dy = dy.contiguous()
    x = x.contiguous()
    n_rows, n_cols = x.shape
    block = triton.next_power_of_2(n_cols)
    assert block <= 16384, "this backward kernel keeps a whole row in one block"
    block = max(16, block)
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    groups = min(n_rows, 4 * sm_count)

    dx = torch.empty_like(x)
    dw_part = torch.empty((groups, n_cols), dtype=torch.float32, device=x.device)
    db_part = torch.empty((groups, n_cols), dtype=torch.float32, device=x.device)
    dw = torch.empty_like(w)
    db = torch.empty_like(w)

    _layernorm_bwd_dx_kernel[(groups,)](
        dy, x, w, mean, rstd, dx, dw_part, db_part, n_rows, n_cols,
        BLOCK=block, num_warps=_num_warps(block),
    )
    block_n = 128
    _layernorm_bwd_reduce_kernel[(triton.cdiv(n_cols, block_n),)](
        dw_part, db_part, dw, db, groups, n_cols, BLOCK_G=64, BLOCK_N=block_n
    )
    return dx, dw, db
