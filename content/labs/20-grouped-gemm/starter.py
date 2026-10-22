"""Lab 20: token permutation, a Triton grouped GEMM, and 2:4 sparsity.

You fill in the functions marked TODO. The given functions (`moe_forward`,
`moe_loop`, `unpermute_combine`, and the launch code in `grouped_gemm`) show how
the parts fit together. Use plain pointer arithmetic and masks; don't use
`tl.make_block_ptr`, which Triton deprecated in 3.7.

The tests check every piece against a float32 PyTorch reference on awkward
shapes, then report TFLOP/s and the speedup over a per-expert loop. The 2:4
part is optional and the tests never fail on it. The first call compiles the
kernel, which takes several seconds.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Part 1: routing and permutation, in PyTorch
# ---------------------------------------------------------------------------


def route_topk(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick the top-k experts per token and the gate weights for them.

    `logits` has shape (T, E). Return `(ids, gates)`: `ids` is an int64 tensor
    of shape (T, k) holding each token's k highest-scoring experts, best first,
    and `gates` is a float32 tensor of shape (T, k) holding the softmax of those
    k logits (Mixtral's rule: softmax over the chosen k, not over all E).
    """
    # TODO: torch.topk on the float32 logits, then torch.softmax over the
    # selected values along dim=-1.
    raise NotImplementedError


def expert_offsets(expert_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Histogram, then exclusive scan: int32 offsets of length num_experts + 1.

    `expert_ids` holds one expert id per (token, slot) assignment, any shape.
    `offsets[e]` is the first row of expert e in the sorted order and
    `offsets[e + 1] - offsets[e]` is its count, so `offsets[-1]` is the number
    of assignments. Experts that receive nothing have an empty range.
    """
    # TODO: count with torch.bincount(..., minlength=num_experts), then take a
    # cumulative sum into a zero-initialised int32 tensor of length E + 1, on
    # the same device as `expert_ids`. Chapter 7's histogram and scan, in two calls.
    raise NotImplementedError


def sort_by_expert(flat_expert_ids: torch.Tensor) -> torch.Tensor:
    """Stable permutation that groups assignments by expert id.

    Return an int64 tensor `order` of the same length so that
    `flat_expert_ids[order]` is non-decreasing, and assignments of one expert
    keep their original relative order.
    """
    # TODO: one call to torch.argsort with stable=True.
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 2: the tile schedule, in plain Python
# ---------------------------------------------------------------------------


def tile_schedule(counts: list[int], block_m: int) -> list[tuple[int, int, int]]:
    """List the (expert, first_row, num_rows) of every row tile, expert by expert.

    `counts[e]` is how many rows expert e owns. Rows are numbered in the sorted
    order, so expert e starts at `sum(counts[:e])`. Each expert's rows are cut
    into tiles of `block_m` rows and the last tile can be shorter. An expert with
    no rows contributes no tile. For counts [0, 5, 64, 1, 130] and block_m = 64
    the result is [(1, 0, 5), (2, 5, 64), (3, 69, 1), (4, 70, 64), (4, 134, 64),
    (4, 198, 2)].
    """
    # TODO: loop over experts, keep a running start row, and emit one tuple per
    # block_m rows.
    raise NotImplementedError


def max_m_tiles(num_rows: int, num_experts: int, block_m: int) -> int:
    """An upper bound on the number of row tiles, computable without the counts.

    The launch grid has to be sized before the kernel reads the counts, so use a
    bound that holds for every split of `num_rows` rows across `num_experts`
    experts: ceil(num_rows / block_m) + num_experts.
    """
    # TODO: the ceiling division plus the number of experts.
    raise NotImplementedError


def tile_waste(counts: list[int], block_m: int) -> float:
    """The fraction of tile rows that are padding.

    Each non-empty expert uses ceil(count / block_m) tiles of block_m rows. The
    waste is 1 - (real rows) / (tile rows), or 0.0 when there are no tiles.
    For counts [0, 5, 64, 1, 130] and block_m = 64 that's 6 tiles, 384 tile
    rows, 200 real rows, and a waste of 0.479.
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 3: the grouped GEMM kernel
# ---------------------------------------------------------------------------


@triton.jit
def grouped_gemm_kernel(
    x_ptr, w_ptr, out_ptr, gate_ptr, offsets_ptr,
    N, K, E,  # N: output columns, K: reduction size, E: number of experts
    stride_xm, stride_xk,
    stride_we, stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    HAS_GATE: tl.constexpr,
):
    # TODO 1. The scheduler. pid_m is a row-tile number counted across all
    #   experts. Find the expert whose tiles contain it: walk e = 0 .. E - 1,
    #   load offsets[e] and offsets[e + 1], compute that expert's tile count with
    #   tl.cdiv, and keep a running total of tiles. Keep three scalars, initialised
    #   as int32 tensors (for example `pid_m * 0 - 1`, not the Python int -1, so
    #   the loop's types agree): `expert` (-1 when pid_m is past the last tile),
    #   `first_tile` (the running total when this expert started), and the
    #   running total. Use tl.where, not `if`, inside the loop.
    #
    # TODO 2. Programs past the last tile have expert == -1. Don't return early:
    #   make `valid = expert >= 0`, clamp the expert to 0, and fold `valid` into
    #   the row mask and into the K loop's trip count so such a program loads
    #   nothing, runs no iterations, and stores nothing.
    #
    # TODO 3. Row, column and K offsets. The first row of this tile is
    #   offsets[expert] + (pid_m - first_tile) * BLOCK_M. A row is real when it is
    #   below offsets[expert + 1]. Pointer tiles: x_ptrs (BLOCK_M, BLOCK_K) from
    #   the row stride, and w_ptrs (BLOCK_K, BLOCK_N) starting at
    #   w_ptr + expert * stride_we; cast expert to int64 before the multiply.
    #
    # TODO 4. The K loop: masked loads with other=0.0 (mask K with
    #   K - k * BLOCK_K), `acc = tl.dot(a, b, acc)` into a float32 accumulator,
    #   and advance both pointer tiles.
    #
    # TODO 5. The epilogue. When HAS_GATE, load gate_ptr[offs_m] (masked) and
    #   scale each row of acc. Cast to out_ptr.dtype.element_ty and store with a
    #   mask on both the rows and the columns.
    raise NotImplementedError


def grouped_gemm(
    x_sorted: torch.Tensor,
    w: torch.Tensor,
    offsets: torch.Tensor,
    gates_sorted: torch.Tensor | None = None,
    *,
    block_m: int = 64,
    block_n: int = 128,
    block_k: int = 32,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Multiply each expert's rows of x_sorted by that expert's weight matrix."""
    rows, k_dim = x_sorted.shape
    num_experts, _, n_dim = w.shape
    out = torch.empty(rows, n_dim, dtype=x_sorted.dtype, device=x_sorted.device)
    if rows == 0:
        return out
    grid = (max_m_tiles(rows, num_experts, block_m), triton.cdiv(n_dim, block_n))
    grouped_gemm_kernel[grid](
        x_sorted, w, out, gates_sorted if gates_sorted is not None else x_sorted, offsets,
        n_dim, k_dim, num_experts,
        x_sorted.stride(0), x_sorted.stride(1),
        w.stride(0), w.stride(1), w.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        HAS_GATE=gates_sorted is not None,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


# ---------------------------------------------------------------------------
# Part 4: the whole layer
# ---------------------------------------------------------------------------


def unpermute_combine(out_sorted: torch.Tensor, order: torch.Tensor, num_tokens: int, k: int) -> torch.Tensor:
    """Undo the sort and add each token's k (already gate-scaled) rows."""
    out_flat = torch.empty_like(out_sorted)
    out_flat[order] = out_sorted
    return out_flat.view(num_tokens, k, -1).float().sum(dim=1).to(out_sorted.dtype)


def moe_forward(x: torch.Tensor, w: torch.Tensor, logits: torch.Tensor, k: int, **config) -> torch.Tensor:
    """Route, permute, grouped GEMM with the gates fused, unpermute and combine."""
    num_tokens = x.shape[0]
    num_experts = w.shape[0]
    ids, gates = route_topk(logits, k)
    flat = ids.reshape(-1)
    offsets = expert_offsets(flat, num_experts)
    order = sort_by_expert(flat)
    x_sorted = x.index_select(0, order // k)
    gates_sorted = gates.reshape(-1).index_select(0, order).contiguous()
    out_sorted = grouped_gemm(x_sorted, w, offsets, gates_sorted, **config)
    return unpermute_combine(out_sorted, order, num_tokens, k)


def moe_loop(x: torch.Tensor, w: torch.Tensor, logits: torch.Tensor, k: int) -> torch.Tensor:
    """The per-expert Python loop that the grouped kernel replaces (given)."""
    ids, gates = route_topk(logits, k)
    out = torch.zeros(x.shape[0], w.shape[2], dtype=x.dtype, device=x.device)
    for e in range(w.shape[0]):
        rows, slot = torch.where(ids == e)
        if rows.numel() == 0:
            continue
        y = (x[rows] @ w[e]) * gates[rows, slot, None].to(x.dtype)
        out.index_add_(0, rows, y)
    return out


# ---------------------------------------------------------------------------
# Part 5 (optional): 2:4 structured sparsity
# ---------------------------------------------------------------------------


def prune_2_4(w: torch.Tensor) -> torch.Tensor:
    """Zero the two smallest magnitudes in every group of four along the last dim.

    `w` is 2D with a last dimension that is a multiple of 4. Return a tensor of
    the same shape, dtype and device in which each group of four consecutive
    values keeps its two largest-magnitude entries and has the other two set to
    zero.
    """
    # TODO: reshape to (rows, cols // 4, 4), find the indices of the two largest
    # |values| per group with topk, build a boolean mask with scatter_, multiply,
    # and reshape back.
    raise NotImplementedError


def sparse_24_weight(w: torch.Tensor):
    """Prune w to 2:4 and compress it; return None when the GPU or PyTorch can't.

    Optional. Call `prune_2_4`, then `torch.sparse.to_sparse_semi_structured`
    (fp16 or bf16, CUDA, 2D, sizes that are multiples of 64). Wrap it in
    try/except and print a note on failure: older GPUs and some builds don't
    support it, and the tests must not fail because of that.
    """
    # TODO (optional)
    return None
