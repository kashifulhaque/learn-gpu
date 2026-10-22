"""Lab 14: a Triton matmul with grouped ordering and autotuning (worked solution).

The kernel uses plain pointer arithmetic and masks. It does not use
`tl.make_block_ptr`, which Triton deprecated in 3.7.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Part 1: the program order, in plain Python
# ---------------------------------------------------------------------------


def grouped_tile(pid: int, num_pid_m: int, num_pid_n: int, group_m: int) -> tuple[int, int]:
    """Map a linear program id to the (pid_m, pid_n) output tile it computes.

    Tiles are visited in vertical groups of `group_m` rows: inside a group the
    row index changes fastest, then the column, and only then does the next
    group of rows start. The last group can be shorter than `group_m`.
    """
    num_pid_in_group = group_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_m
    group_size_m = min(num_pid_m - first_pid_m, group_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


def unique_tile_loads(
    num_pid_m: int, num_pid_n: int, num_k_tiles: int, group_m: int, n_programs: int
) -> int:
    """Count the distinct input tiles that the first `n_programs` programs read.

    A program that computes output tile (pid_m, pid_n) reads a row of A tiles
    (`num_k_tiles` of them, one per K step) and a column of B tiles. Programs
    that share a pid_m share the A row, and programs that share a pid_n share
    the B column. Return (distinct pid_m + distinct pid_n) * num_k_tiles.
    """
    total = num_pid_m * num_pid_n
    tiles = [grouped_tile(p, num_pid_m, num_pid_n, group_m) for p in range(min(n_programs, total))]
    rows = {m for m, _ in tiles}
    cols = {n for _, n in tiles}
    return (len(rows) + len(cols)) * num_k_tiles


# ---------------------------------------------------------------------------
# Part 2: the kernel
# ---------------------------------------------------------------------------


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    # Grouped program ordering: the same arithmetic as grouped_tile().
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulate in fp32 whatever the input dtype is.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_left = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_left), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_left) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Optional epilogue, applied to the fp32 accumulator before the downcast.
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc += bias[None, :]
    if ACTIVATION == "relu":
        acc = tl.maximum(acc, 0.0)
    elif ACTIVATION == "silu":
        acc = acc * tl.sigmoid(acc)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


# ---------------------------------------------------------------------------
# Part 3: the autotune list
# ---------------------------------------------------------------------------

SUPPORTS_EPILOGUE = True


def autotune_configs() -> list[triton.Config]:
    """Return the tile shapes the autotuner tries."""

    def cfg(bm, bn, bk, stages, warps, group_m=8):
        return triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": group_m},
            num_stages=stages,
            num_warps=warps,
        )

    # Shared memory per config = (BM*BK + BK*BN) * 2 bytes * stages, at most 163 KiB.
    return [
        cfg(128, 256, 64, 3, 8),  # 144 KiB
        cfg(64, 256, 32, 4, 4),   # 80 KiB
        cfg(128, 128, 32, 4, 4),  # 64 KiB
        cfg(128, 64, 32, 4, 4),   # 48 KiB
        cfg(64, 128, 32, 4, 4),   # 48 KiB
        cfg(128, 32, 32, 4, 4),   # 40 KiB
        cfg(64, 32, 32, 5, 2),    # 30 KiB
        cfg(32, 64, 32, 5, 2),    # 30 KiB
    ]


# ---------------------------------------------------------------------------
# Plumbing: launching the kernel
# ---------------------------------------------------------------------------

_AUTOTUNED = None


def _autotuned_kernel():
    """The same kernel, wrapped so Triton benchmarks every config per (M, N, K)."""
    global _AUTOTUNED
    if _AUTOTUNED is None:
        _AUTOTUNED = triton.autotune(configs=autotune_configs(), key=["M", "N", "K"])(matmul_kernel)
    return _AUTOTUNED


def _prepare(a, b, bias):
    assert a.is_cuda and b.is_cuda, "inputs must be CUDA tensors"
    assert a.dtype == b.dtype and a.dtype in (torch.float16, torch.bfloat16), a.dtype
    assert a.dim() == 2 and b.dim() == 2 and a.shape[1] == b.shape[0], (a.shape, b.shape)
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    if bias is not None:
        assert bias.shape == (N,) and bias.is_cuda
    # With no bias the pointer is never read; pass c as a placeholder.
    bias_arg = bias if bias is not None else c
    return M, N, K, c, bias_arg


def matmul(a: torch.Tensor, b: torch.Tensor, bias=None, activation=None) -> torch.Tensor:
    """C = activation(A @ B + bias), autotuned. `activation` is None, "relu", or "silu"."""
    M, N, K, c, bias_arg = _prepare(a, b, bias)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    _autotuned_kernel()[grid](
        a, b, c, bias_arg, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        HAS_BIAS=bias is not None, ACTIVATION=activation or "none",
    )
    return c


def matmul_fixed(
    a: torch.Tensor, b: torch.Tensor, block_m=128, block_n=128, block_k=32,
    group_m=8, num_warps=4, num_stages=4,
) -> torch.Tensor:
    """C = A @ B with one explicit configuration and no autotuning."""
    M, N, K, c, bias_arg = _prepare(a, b, None)
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    matmul_kernel[grid](
        a, b, c, bias_arg, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=group_m,
        HAS_BIAS=False, ACTIVATION="none",
        num_warps=num_warps, num_stages=num_stages,
    )
    return c
