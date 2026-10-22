"""Lab 15: Softmax and LayerNorm. Reference solution.

Three public functions, all for 2D contiguous CUDA tensors of shape (rows, cols):

    softmax_online(x)                      -> y
    layernorm_forward(x, w, b, eps)        -> y, mean, rstd
    layernorm_backward(dy, x, w, mean, rstd) -> dx, dw, db
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
# Online softmax
# ---------------------------------------------------------------------------


@triton.jit
def _softmax_online_kernel(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    x_row = x_ptr + row * n_cols
    y_row = y_ptr + row * n_cols

    # Pass 1: stream the row once, keeping a running maximum m and a running
    # denominator d = sum(exp(x - m)) for the columns seen so far.
    m = float("-inf")
    d = 0.0
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(x_row + cols, mask=cols < n_cols, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, axis=0))
        # exp(m - m_new) rescales the old denominator to the new maximum.
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new

    # Pass 2: read the row again (the second read often hits L2) and write the result.
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=float("-inf")).to(tl.float32)
        y = tl.exp(x - m) / d
        tl.store(y_row + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


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
# LayerNorm forward: Welford statistics by chunks
# ---------------------------------------------------------------------------


@triton.jit
def _layernorm_fwd_kernel(
    x_ptr, y_ptr, w_ptr, b_ptr, mean_ptr, rstd_ptr, n_cols, eps, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    x_row = x_ptr + row * n_cols
    y_row = y_ptr + row * n_cols

    # Pass 1: merge each chunk's (count, mean, M2) into the running statistics
    # with Chan's parallel update, which is Welford's algorithm for blocks.
    count = 0.0
    mean = 0.0
    m2 = 0.0
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        n_b = tl.minimum(n_cols - start, BLOCK).to(tl.float32)
        mean_b = tl.sum(x, axis=0) / n_b
        centered = tl.where(mask, x - mean_b, 0.0)
        m2_b = tl.sum(centered * centered, axis=0)
        n_a = count
        count = n_a + n_b
        delta = mean_b - mean
        mean = mean + delta * (n_b / count)
        m2 = m2 + m2_b + delta * delta * (n_a * n_b / count)

    var = m2 / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)

    # Pass 2: normalize, scale, and shift.
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(y_row + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


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
# LayerNorm backward: dx per row, dw and db by a two-stage reduction
# ---------------------------------------------------------------------------


@triton.jit
def _layernorm_bwd_dx_kernel(
    dy_ptr, x_ptr, w_ptr, mean_ptr, rstd_ptr, dx_ptr, dw_part_ptr, db_part_ptr,
    n_rows, n_cols, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    groups = tl.num_programs(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    dw_acc = tl.zeros((BLOCK,), dtype=tl.float32)
    db_acc = tl.zeros((BLOCK,), dtype=tl.float32)
    # This program owns rows pid, pid + groups, pid + 2 * groups, ...
    for row in range(pid, n_rows, groups):
        offs = row.to(tl.int64) * n_cols + cols
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + row)
        rstd = tl.load(rstd_ptr + row)

        xhat = tl.where(mask, (x - mean) * rstd, 0.0)
        g = dy * w  # zero in masked lanes, because dy and w are
        c1 = tl.sum(xhat * g, axis=0) / n_cols
        c2 = tl.sum(g, axis=0) / n_cols
        dx = (g - xhat * c1 - c2) * rstd
        tl.store(dx_ptr + offs, dx.to(dx_ptr.dtype.element_ty), mask=mask)

        dw_acc += dy * xhat
        db_acc += dy

    # No atomics and no locks: each program owns one row of the partial buffers.
    tl.store(dw_part_ptr + pid * n_cols + cols, dw_acc, mask=mask)
    tl.store(db_part_ptr + pid * n_cols + cols, db_acc, mask=mask)


@triton.jit
def _layernorm_bwd_reduce_kernel(
    dw_part_ptr, db_part_ptr, dw_ptr, db_ptr, groups, n_cols,
    BLOCK_G: tl.constexpr, BLOCK_N: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < n_cols
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for g0 in range(0, groups, BLOCK_G):
        g = g0 + tl.arange(0, BLOCK_G)
        mask = (g[:, None] < groups) & col_mask[None, :]
        offs = g[:, None] * n_cols + cols[None, :]
        dw += tl.sum(tl.load(dw_part_ptr + offs, mask=mask, other=0.0), axis=0)
        db += tl.sum(tl.load(db_part_ptr + offs, mask=mask, other=0.0), axis=0)
    tl.store(dw_ptr + cols, dw.to(dw_ptr.dtype.element_ty), mask=col_mask)
    tl.store(db_ptr + cols, db.to(db_ptr.dtype.element_ty), mask=col_mask)


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
