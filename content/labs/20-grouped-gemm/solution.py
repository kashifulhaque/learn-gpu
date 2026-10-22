"""Lab 20: token permutation, a Triton grouped GEMM, and 2:4 sparsity (solution)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Part 1: routing and permutation, in PyTorch
# ---------------------------------------------------------------------------


def route_topk(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick the top-k experts per token and the gate weights for them."""
    values, ids = torch.topk(logits.float(), k, dim=-1)
    gates = torch.softmax(values, dim=-1)
    return ids, gates


def expert_offsets(expert_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Histogram, then exclusive scan: int32 offsets of length num_experts + 1."""
    counts = torch.bincount(expert_ids.reshape(-1), minlength=num_experts)
    offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=expert_ids.device)
    offsets[1:] = torch.cumsum(counts, dim=0).to(torch.int32)
    return offsets


def sort_by_expert(flat_expert_ids: torch.Tensor) -> torch.Tensor:
    """Stable permutation that groups assignments by expert id."""
    return torch.argsort(flat_expert_ids, stable=True)


# ---------------------------------------------------------------------------
# Part 2: the tile schedule, in plain Python
# ---------------------------------------------------------------------------


def tile_schedule(counts: list[int], block_m: int) -> list[tuple[int, int, int]]:
    """List the (expert, first_row, num_rows) of every row tile, expert by expert."""
    tiles = []
    start = 0
    for expert, count in enumerate(counts):
        for first in range(0, count, block_m):
            tiles.append((expert, start + first, min(block_m, count - first)))
        start += count
    return tiles


def max_m_tiles(num_rows: int, num_experts: int, block_m: int) -> int:
    """An upper bound on the number of row tiles, computable without the counts."""
    return -(-num_rows // block_m) + num_experts


def tile_waste(counts: list[int], block_m: int) -> float:
    """The fraction of tile rows that are padding."""
    tiles = sum(-(-c // block_m) for c in counts)
    if tiles == 0:
        return 0.0
    return 1.0 - sum(counts) / (tiles * block_m)


# ---------------------------------------------------------------------------
# Part 3: the grouped GEMM kernel
# ---------------------------------------------------------------------------


@triton.jit
def grouped_gemm_kernel(
    x_ptr, w_ptr, out_ptr, gate_ptr, offsets_ptr,
    N, K, E,
    stride_xm, stride_xk,
    stride_we, stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    HAS_GATE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Scheduler: find the expert whose range of row tiles holds pid_m.
    expert = pid_m * 0 - 1
    first_tile = pid_m * 0
    running = pid_m * 0
    for e in range(0, E):
        seg_start = tl.load(offsets_ptr + e)
        seg_end = tl.load(offsets_ptr + e + 1)
        tiles = tl.cdiv(seg_end - seg_start, BLOCK_M)
        mine = (pid_m >= running) & (pid_m < running + tiles)
        expert = tl.where(mine, e, expert)
        first_tile = tl.where(mine, running, first_tile)
        running += tiles
    valid = expert >= 0
    expert = tl.maximum(expert, 0)

    row_begin = tl.load(offsets_ptr + expert) + (pid_m - first_tile) * BLOCK_M
    row_end = tl.load(offsets_ptr + expert + 1)
    offs_m = row_begin + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = (offs_m < row_end) & valid
    mask_n = offs_n < N

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = (
        w_ptr
        + expert.to(tl.int64) * stride_we
        + offs_k[:, None] * stride_wk
        + offs_n[None, :] * stride_wn
    )

    k_iters = tl.where(valid, tl.cdiv(K, BLOCK_K), 0)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, k_iters):
        k_left = K - k * BLOCK_K
        a = tl.load(x_ptrs, mask=mask_m[:, None] & (offs_k[None, :] < k_left), other=0.0)
        b = tl.load(w_ptrs, mask=(offs_k[:, None] < k_left) & mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    if HAS_GATE:
        gate = tl.load(gate_ptr + offs_m, mask=mask_m, other=0.0)
        acc = acc * gate[:, None]

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


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
    """Zero the two smallest magnitudes in every group of four along the last dim."""
    rows, cols = w.shape
    if cols % 4:
        raise ValueError("the last dimension must be a multiple of 4")
    groups = w.reshape(rows, cols // 4, 4)
    keep = groups.abs().topk(2, dim=-1).indices
    mask = torch.zeros_like(groups, dtype=torch.bool).scatter_(-1, keep, True)
    return (groups * mask).reshape(rows, cols)


def sparse_24_weight(w: torch.Tensor):
    """Prune w to 2:4 and compress it; return None when the GPU or PyTorch can't."""
    try:
        from torch.sparse import to_sparse_semi_structured

        return to_sparse_semi_structured(prune_2_4(w))
    except Exception as exc:  # noqa: BLE001
        print(f"      note: to_sparse_semi_structured is unavailable here ({type(exc).__name__}: {exc})", flush=True)
        return None
