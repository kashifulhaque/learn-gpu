"""Lab 14: a Triton matmul with grouped ordering and autotuning.

You fill in four things: `grouped_tile`, `unique_tile_loads`, `matmul_kernel`,
and `autotune_configs`. The launch code at the bottom is given. Use plain
pointer arithmetic and masks. Don't use `tl.make_block_ptr`: Triton deprecated
it in 3.7.

The tests check correctness against a float32 reference on awkward shapes,
then report TFLOP/s and the fraction of torch.matmul. The first call for each
new (M, N, K) autotunes, which takes seconds. That's expected.
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
    # TODO: work out which group of `group_m` rows `pid` falls in, how tall
    # that group really is (the last one can be shorter), and the row and
    # column of `pid` inside it. With group_m = 1 you must get plain row-major
    # order. With 9 x 9 tiles and group_m = 3, pids 0, 1, 2 are (0, 0), (1, 0),
    # (2, 0) and pid 3 is (0, 1).
    raise NotImplementedError


def unique_tile_loads(
    num_pid_m: int, num_pid_n: int, num_k_tiles: int, group_m: int, n_programs: int
) -> int:
    """Count the distinct input tiles that the first `n_programs` programs read.

    A program that computes output tile (pid_m, pid_n) reads a row of A tiles
    (`num_k_tiles` of them, one per K step) and a column of B tiles. Programs
    that share a pid_m share the A row, and programs that share a pid_n share
    the B column. Return (distinct pid_m + distinct pid_n) * num_k_tiles.
    """
    # TODO: collect the tiles of programs 0 .. n_programs - 1 (clamp n_programs
    # to the number of tiles) and count their distinct rows and columns. For a
    # 9 x 9 grid with 9 K tiles and 9 programs, group_m = 1 gives 90 and
    # group_m = 3 gives 54.
    raise NotImplementedError


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
    # TODO 1. Grouped program ordering: the same arithmetic as grouped_tile(),
    #   written with tl.program_id(0), tl.cdiv and tl.minimum. Get pid_m, pid_n.
    #
    # TODO 2. Build the row offsets offs_m (BLOCK_M), column offsets offs_n
    #   (BLOCK_N), and K offsets offs_k (BLOCK_K) with tl.arange, then the
    #   pointer tiles a_ptrs (BLOCK_M, BLOCK_K) and b_ptrs (BLOCK_K, BLOCK_N)
    #   from the strides. Don't use tl.make_block_ptr.
    #
    # TODO 3. Loop over K in steps of BLOCK_K. Load an A tile and a B tile with
    #   masks that cover the ragged edges of M, N, and K (other=0.0, the
    #   identity for a dot product), accumulate with
    #   acc = tl.dot(a, b, acc) into a float32 accumulator, and advance both
    #   pointer tiles by BLOCK_K along K.
    #
    # TODO 4. (Optional: set SUPPORTS_EPILOGUE = True below when done.) After the
    #   loop, if HAS_BIAS add bias_ptr[offs_n] (load it with a mask, convert
    #   to float32). Then apply ACTIVATION: "relu" is max(x, 0) and "silu" is
    #   x * sigmoid(x). Do this on the float32 accumulator, before the cast.
    #
    # TODO 5. Cast the accumulator to c_ptr.dtype.element_ty and store it with
    #   a mask on both M and N.
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 3: the autotune list
# ---------------------------------------------------------------------------

# Set to True once your kernel handles bias and activation. The tests then check them.
SUPPORTS_EPILOGUE = False


def autotune_configs() -> list[triton.Config]:
    """Return the tile shapes the autotuner tries."""
    # TODO: return at least four triton.Config objects, each built as
    #   triton.Config({"BLOCK_M": ..., "BLOCK_N": ..., "BLOCK_K": ..., "GROUP_M": ...},
    #                 num_stages=..., num_warps=...)
    # Every block size is a power of two, at least 32 for BLOCK_M and BLOCK_N and
    # at least 16 for BLOCK_K. Each config's shared memory,
    #   (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * 2 bytes * num_stages,
    # must fit in 163 KiB, and its accumulator, BLOCK_M * BLOCK_N / (32 * num_warps)
    # float32 registers per thread, must stay under 255. Mix big tiles with
    # many warps and small tiles with few, so small matrices have a good choice too.
    raise NotImplementedError


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
