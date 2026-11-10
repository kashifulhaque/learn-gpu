"""Lab 38: Tile languages. Worked solution.

Every function except `softmax_triton` is pure Python or torch-CPU. The tile
programs run against the fixture's tile API, which raises `OutOfBounds` when a
load or a store touches memory outside the array without a mask.
"""

from __future__ import annotations

import itertools
import math

from fixture_tiles import BlockSpec, arange, load_tile, mma, store_tile  # noqa: F401

try:  # Triton is installed on the A100 worker image, not on a laptop.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


# ---------------------------------------------------------------------------
# Part 1: the grid and the block index map (Pallas)
# ---------------------------------------------------------------------------


def grid_shape(array_shape, block_shape):
    """Blocks per dimension: ceil(array / block)."""
    return tuple(-(-a // b) for a, b in zip(array_shape, block_shape))


def block_slices(block_index, block_shape, array_shape):
    """The slice of the array that one block index names, clipped at the edge.

    The index is a *block* index: block 2 of width 8 starts at element 16.
    """
    out = []
    for i, b, n in zip(block_index, block_shape, array_shape):
        start = i * b
        out.append(slice(start, min(start + b, n)))
    return tuple(out)


def gemm_block_indices(i, j, k):
    """The block index each GEMM operand uses at grid point (i, j, k)."""
    return {"a": (i, k), "b": (k, j), "c": (i, j)}


def pallas_call(kernel, grid, in_specs, out_specs, out_shape):
    """A small `pl.pallas_call`: returns a function of the input arrays.

    At every grid point the kernel gets the input blocks and one output block.
    Input blocks that hang over the edge are padded with zeros to the full
    block shape. The kernel writes the output block in place, and only the part
    inside the array is copied back.
    """

    def call(*inputs):
        import torch

        out = torch.zeros(out_shape)
        for point in itertools.product(*(range(g) for g in grid)):
            blocks = []
            for x, spec in zip(inputs, in_specs):
                sl = block_slices(spec.index_map(*point), spec.block_shape, x.shape)
                block = torch.zeros(spec.block_shape, dtype=x.dtype)
                part = x[sl]
                block[tuple(slice(0, s) for s in part.shape)] = part
                blocks.append(block)
            osl = block_slices(out_specs.index_map(*point), out_specs.block_shape, out.shape)
            oblock = torch.zeros(out_specs.block_shape)
            kernel(*blocks, oblock)
            part_shape = out[osl].shape
            out[osl] = oblock[tuple(slice(0, s) for s in part_shape)]
        return out

    return call


# ---------------------------------------------------------------------------
# Part 2: a tile-loop GEMM with masks
# ---------------------------------------------------------------------------


def tile_gemm(a, b, bm=16, bn=16, bk=16):
    """C = A @ B, one (bm, bn) output tile per program, a loop over K tiles."""
    import torch

    m, k_dim = a.shape
    _, n = b.shape
    c = torch.zeros(m, n)
    for pm in range(-(-m // bm)):
        for pn in range(-(-n // bn)):
            rows = pm * bm + arange(bm)
            cols = pn * bn + arange(bn)
            acc = torch.zeros(bm, bn)
            for pk in range(-(-k_dim // bk)):
                ks = pk * bk + arange(bk)
                a_tile = load_tile(a, rows, ks, mask=(rows[:, None] < m) & (ks[None, :] < k_dim))
                b_tile = load_tile(b, ks, cols, mask=(ks[:, None] < k_dim) & (cols[None, :] < n))
                acc = mma(a_tile, b_tile, acc)
            store_tile(c, rows, cols, acc, mask=(rows[:, None] < m) & (cols[None, :] < n))
    return c


# ---------------------------------------------------------------------------
# Part 3: an online softmax tile program
# ---------------------------------------------------------------------------


def online_softmax(x, block_n=16):
    """Row softmax, one program per row, one pass of (max, sum) then one write."""
    import torch

    rows_n, n = x.shape
    y = torch.zeros_like(x)
    for r in range(rows_n):
        rows = torch.tensor([r])
        m = float("-inf")
        l = 0.0
        for start in range(0, n, block_n):
            cols = start + arange(block_n)
            mask = (cols < n)[None, :]
            tile = load_tile(x, rows, cols, mask=mask, other=float("-inf"))
            m_new = max(m, float(tile.max()))
            l = l * math.exp(m - m_new) + float(torch.exp(tile - m_new).sum())
            m = m_new
        for start in range(0, n, block_n):
            cols = start + arange(block_n)
            mask = (cols < n)[None, :]
            tile = load_tile(x, rows, cols, mask=mask, other=float("-inf"))
            store_tile(y, rows, cols, torch.exp(tile - m) / l, mask=mask)
    return y


# ---------------------------------------------------------------------------
# Part 4: which thread holds which element
# ---------------------------------------------------------------------------


def _linear(coords, extents, order):
    """Linear index of `coords`, where dimension order[0] varies fastest."""
    index = 0
    stride = 1
    for d in order:
        index += coords[d] * stride
        stride *= extents[d]
    return index


def blocked_owner(row, col, shape, size_per_thread, threads_per_warp, warps_per_cta, order):
    """(warp, lane, register) that holds element (row, col) of a 2-D tile."""
    e = (row, col)
    loc, thr, wrp, rep, reps = [], [], [], [], []
    for d in range(2):
        spt, tpw, wpc = size_per_thread[d], threads_per_warp[d], warps_per_cta[d]
        cover = spt * tpw * wpc
        loc.append(e[d] % spt)
        thr.append((e[d] // spt) % tpw)
        wrp.append((e[d] // (spt * tpw)) % wpc)
        rep.append(e[d] // cover)
        reps.append(shape[d] // cover)
    lane = _linear(thr, threads_per_warp, order)
    warp = _linear(wrp, warps_per_cta, order)
    local = _linear(loc, size_per_thread, order)
    reg = local + size_per_thread[0] * size_per_thread[1] * _linear(rep, reps, order)
    return warp, lane, reg


# ---------------------------------------------------------------------------
# Part 5: the software pipeline
# ---------------------------------------------------------------------------


def pipeline_schedule(num_tiles, stages):
    """One (load, mma) pair per step. A tile index, or None.

    Step s loads tile s and runs the MMA of tile s - (stages - 1). The loads of
    the first stages - 1 tiles are the prologue, and the last stages - 1 steps
    are the drain.
    """
    if stages < 2:
        raise ValueError("a pipeline needs at least 2 stages")
    steps = []
    for s in range(num_tiles + stages - 1):
        load = s if s < num_tiles else None
        t = s - (stages - 1)
        steps.append((load, t if 0 <= t < num_tiles else None))
    return steps


def max_resident(schedule):
    """The most tiles that sit in shared memory at once.

    A tile is resident from the step that issues its load through the step that
    runs its MMA, inclusive.
    """
    loaded_at, used_at = {}, {}
    for s, (load, use) in enumerate(schedule):
        if load is not None:
            loaded_at[load] = s
        if use is not None:
            used_at[use] = s
    best = 0
    for s in range(len(schedule)):
        live = sum(1 for t in loaded_at if loaded_at[t] <= s <= used_at[t])
        best = max(best, live)
    return best


# ---------------------------------------------------------------------------
# Part 6: the same softmax in Triton (A100 only)
# ---------------------------------------------------------------------------

if triton is not None:

    @triton.jit
    def _softmax_kernel(y_ptr, x_ptr, n_cols, stride, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < n_cols
        x = tl.load(x_ptr + row * stride + offs, mask=mask, other=-float("inf"))
        x = x - tl.max(x, axis=0)
        num = tl.exp(x)
        y = num / tl.sum(num, axis=0)
        tl.store(y_ptr + row * stride + offs, y, mask=mask)


def softmax_triton(x):
    """Row softmax of a CUDA float32 matrix, one Triton program per row."""
    x = x.contiguous()
    y = x.new_empty(x.shape)
    rows, cols = x.shape
    _softmax_kernel[(rows,)](y, x, cols, x.stride(0), BLOCK=triton.next_power_of_2(cols))
    return y
