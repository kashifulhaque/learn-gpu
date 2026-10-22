"""Lab 18: W4A16, a fused dequantize-GEMM in Triton.

Quantize an fp16 weight matrix W of shape (N, K) to 4 bits, and write a kernel
that computes Y = X @ dequant(W)^T for X of shape (M, K) in fp16, reading the
packed weights directly.

The format, which every function below must follow:

- Asymmetric, unsigned 4-bit codes 0..15.
- One scale and one zero point per output row and per group of 128 consecutive
  weights along K. The weight is (code - zero) * scale.
- scale = (max - min) / 15 per group, stored as fp16. The range always includes 0,
  so min is at most 0 and max is at least 0. zero = round(-min / scale), an
  integer 0..15 stored as uint8. An all-zero group uses scale 1.
- Two codes per byte, packed along K: byte j of a row holds code 2j in the low
  nibble and code 2j + 1 in the high nibble.

K must be a multiple of 128. N and M can be anything.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

GROUP_SIZE = 128

# ---------------------------------------------------------------------------
# Part 1: quantize, pack, unpack, dequantize (plain torch)
# ---------------------------------------------------------------------------


def quantize_int4(w: torch.Tensor, group_size: int = GROUP_SIZE):
    """Quantize `w` (N, K) to int4 codes.

    Return a tuple (codes, scales, zeros):
    - codes: (N, K) uint8 with values 0..15
    - scales: (N, K // group_size) float16
    - zeros: (N, K // group_size) uint8 with values 0..15

    Reshape to (N, K // group_size, group_size) and reduce over the last axis.
    Compute in float32. Round the scale to fp16 BEFORE computing the codes, so
    the codes match the scale the kernel will use. Round to nearest, then clamp
    to 0..15.
    """
    # TODO: implement.
    raise NotImplementedError


def pack_int4(codes: torch.Tensor) -> torch.Tensor:
    """Pack (N, K) uint8 codes into (N, K // 2) uint8: even index low, odd index high.

    Do the shift in int32 and convert back to uint8 at the end.
    """
    # TODO: implement.
    raise NotImplementedError


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of pack_int4: (N, K // 2) uint8 to (N, K) uint8 codes 0..15."""
    # TODO: implement. torch.stack((lo, hi), dim=-1).reshape(...) interleaves two tensors.
    raise NotImplementedError


def dequantize_int4(packed, scales, zeros, group_size: int = GROUP_SIZE) -> torch.Tensor:
    """Return the float32 weights (code - zero) * scale, shape (N, K).

    This is the reference the tests compare your kernel with, so compute it in
    float32 from the packed bytes, the fp16 scales, and the uint8 zeros.
    """
    # TODO: implement.
    raise NotImplementedError


def int4_to_fp16_magic(codes: torch.Tensor) -> torch.Tensor:
    """Convert uint8 codes 0..15 to float16 without an integer-to-float convert.

    The trick: an fp16 whose 16 bits are 0x6400 | q (q below 1024) has the value
    1024 + q, because 0x6400 is the exponent for 1024 and the unit in the last
    place is 1 there. OR the code into 0x6400 as an int16, reinterpret the bits
    with `.view(torch.float16)`, and subtract 1024.0. A GPU does the OR for two
    values at once with one lop3 instruction.
    """
    # TODO: implement.
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 2: the fused kernel
# ---------------------------------------------------------------------------

# Tile shapes to try. BLOCK_M = 16 is the smallest M that tl.dot accepts, and it
# is the right choice for decode. The tests report how well the autotuner does.
_CONFIGS = [
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 16}, num_warps=2, num_stages=4),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 16}, num_warps=4, num_stages=5),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 32}, num_warps=2, num_stages=4),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 32}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=_CONFIGS, key=["M_CLASS", "N", "K"])
@triton.jit
def w4a16_kernel(
    x_ptr, wq_ptr, s_ptr, z_ptr, y_ptr,
    M, N, K, M_CLASS,
    stride_xm, stride_xk,
    stride_wn,
    stride_gn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP: tl.constexpr,
):
    """One program computes a (BLOCK_M, BLOCK_N) tile of Y.

    The loop runs once per group of 128 weights along K. In iteration g:
    - x_e and x_o are the (BLOCK_M, 64) tiles of X at even and odd columns of
      the group, because byte j holds codes 2j and 2j + 1.
    - `packed` is the (BLOCK_N, 64) tile of bytes. `packed & 0xF` holds the codes
      for the even columns, `packed >> 4` for the odd columns.
    - scale and zero are the (BLOCK_N,) values for this group.
    """
    HALF: tl.constexpr = GROUP // 2  # packed bytes per group
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_j = tl.arange(0, HALF)
    m_mask = offs_m < M
    n_mask = offs_n < N

    x_even_ptrs = x_ptr + offs_m[:, None] * stride_xm + (2 * offs_j)[None, :] * stride_xk
    x_odd_ptrs = x_even_ptrs + stride_xk
    w_ptrs = wq_ptr + offs_n[:, None] * stride_wn + offs_j[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for g in range(0, K // GROUP):
        # TODO: load x_e and x_o at offset g * GROUP * stride_xk, the packed bytes
        # at offset g * HALF (convert to int32), and this group's scale and zero
        # at offs_n * stride_gn + g. Mask every load.
        #
        # Then either
        #   (a) dequantize: w = (code - zero) * scale as fp16, and acc += tl.dot(x, w), or
        #   (b) the factored form, which needs far less work per weight:
        #       part = dot(x_e, lo^T) + dot(x_o, hi^T)        with lo, hi converted to fp16
        #       acc += (part - sum_k(x) * zero) * scale
        # tl.trans transposes a tile, and tl.dot(a, b, acc) adds to an accumulator.
        pass

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(tl.float16), mask=m_mask[:, None] & n_mask[None, :])


# Decode, M = 1: a GEMV. A tl.dot tile is at least 16 rows of X, so at M = 1 it
# wastes 15/16 of every tensor-core instruction, and the 64 bytes per row it reads
# per step are too short a run for the memory system. This kernel reads GPB groups
# (GPB x 64 contiguous bytes) of each weight row per step on CUDA cores, sums each
# group's products first, and applies the group's scale and zero point once:
#     y_n = sum_g s_ng * (sum_k q_nk x_k  -  z_ng * sum_k x_k)
_GEMV_CONFIGS = [
    triton.Config({"BLOCK_N": bn, "GPB": gpb}, num_warps=w, num_stages=st)
    for bn in (4, 8, 16) for gpb in (2, 4, 8) for w in (4, 8) for st in (2, 3)
]


@triton.autotune(configs=_GEMV_CONFIGS, key=["N", "K"])
@triton.jit
def w4a16_gemv_kernel(
    x_ptr, wq_ptr, s_ptr, z_ptr, y_ptr,
    N, K,
    stride_xm, stride_xk, stride_wn, stride_gn, stride_ym, stride_yn,
    BLOCK_N: tl.constexpr, GPB: tl.constexpr, GROUP: tl.constexpr,
):
    HALF: tl.constexpr = GROUP // 2  # packed bytes per group
    pid_n = tl.program_id(0)
    row = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_g = tl.arange(0, GPB)
    offs_j = tl.arange(0, HALF)
    n_mask = offs_n < N
    n_groups = K // GROUP

    acc = tl.zeros((BLOCK_N, GPB), dtype=tl.float32)
    for g0 in range(0, n_groups, GPB):
        gg = g0 + offs_g                               # (GPB,) group indices
        g_mask = gg < n_groups
        kbyte = gg[:, None] * HALF + offs_j[None, :]   # (GPB, HALF) byte offsets in a row
        # TODO: load this step's slice of X at even and odd columns (2 * kbyte and
        # 2 * kbyte + 1, shape (GPB, HALF)), the packed bytes of BLOCK_N rows as a
        # (BLOCK_N, GPB, HALF) tile (convert to int32), and the (BLOCK_N, GPB)
        # scales and zero points at offs_n * stride_gn + gg. Mask every load.
        # Then, in fp32:
        #   part = sum over the last axis of lo * x_e + hi * x_o     -> (BLOCK_N, GPB)
        #   acc += (part - zero * sum_k(x)) * scale
        pass

    y = tl.sum(acc, axis=1)
    tl.store(y_ptr + row * stride_ym + offs_n * stride_yn, y.to(tl.float16), mask=n_mask)


def w4a16_matmul(x, packed, scales, zeros, group_size: int = GROUP_SIZE) -> torch.Tensor:
    """Y = X @ dequant(W)^T for X (M, K) fp16, returning (M, N) fp16."""
    assert x.is_cuda and x.dtype == torch.float16 and x.dim() == 2
    assert group_size == GROUP_SIZE, "the kernel's tile is one group of 128"
    m, k = x.shape
    n = packed.shape[0]
    assert packed.shape == (n, k // 2) and k % group_size == 0
    y = torch.empty((m, n), dtype=torch.float16, device=x.device)
    if m == 1:
        def gemv_grid(meta):
            return (triton.cdiv(n, meta["BLOCK_N"]), m)

        w4a16_gemv_kernel[gemv_grid](
            x, packed, scales, zeros, y,
            n, k,
            x.stride(0), x.stride(1), packed.stride(0), scales.stride(0),
            y.stride(0), y.stride(1),
            GROUP=group_size,
        )
        return y

    m_class = 1 if m <= 16 else (2 if m <= 64 else 3)

    def grid(meta):
        return (triton.cdiv(n, meta["BLOCK_N"]), triton.cdiv(m, meta["BLOCK_M"]))

    w4a16_kernel[grid](
        x, packed, scales, zeros, y,
        m, n, k, m_class,
        x.stride(0), x.stride(1),
        packed.stride(0),
        scales.stride(0),
        y.stride(0), y.stride(1),
        GROUP=group_size,
    )
    return y
