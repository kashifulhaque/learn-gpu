"""Lab 18: W4A16, a fused dequantize-GEMM in Triton (worked solution).

Weights are quantized per output row and per group of 128 along K to
asymmetric uint4: w ~ (code - zero) * scale, with code in 0..15. Two codes share
a byte, packed along K. The kernel reads the packed bytes, never builds an fp16
copy of the weights, and uses the factored form

    sum_k x_k (q_k - z) s  =  s * (sum_k x_k q_k  -  z * sum_k x_k)

so the inner loop only converts 4-bit integers to fp16 and feeds the tensor
cores. The kernel uses plain pointer arithmetic and masks, not
`tl.make_block_ptr`, which Triton deprecated in 3.7.
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
    """Asymmetric group quantization of a (N, K) matrix to 4-bit codes."""
    n, k = w.shape
    if k % group_size != 0:
        raise ValueError(f"K = {k} must be a multiple of the group size {group_size}")
    g = w.float().reshape(n, k // group_size, group_size)
    # Include 0 in the range, so that 0.0 is exactly representable.
    wmin = g.amin(dim=-1, keepdim=True).clamp(max=0.0)
    wmax = g.amax(dim=-1, keepdim=True).clamp(min=0.0)
    scale = ((wmax - wmin) / 15.0).to(torch.float16)
    # An all-zero group has no range; any positive scale works.
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    s32 = scale.float()
    zero = torch.round(-wmin / s32).clamp(0, 15)
    codes = torch.round(g / s32 + zero).clamp(0, 15)
    return (
        codes.reshape(n, k).to(torch.uint8),
        scale.squeeze(-1),
        zero.squeeze(-1).to(torch.uint8),
    )


def pack_int4(codes: torch.Tensor) -> torch.Tensor:
    """Pack two 4-bit codes per byte along K: even index low, odd index high."""
    c = codes.to(torch.int32)
    return (c[:, 0::2] | (c[:, 1::2] << 4)).to(torch.uint8)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of pack_int4: (N, K // 2) uint8 to (N, K) uint8 codes."""
    p = packed.to(torch.int32)
    lo = p & 0xF
    hi = p >> 4
    n, half = packed.shape
    return torch.stack((lo, hi), dim=-1).reshape(n, 2 * half).to(torch.uint8)


def dequantize_int4(packed, scales, zeros, group_size: int = GROUP_SIZE) -> torch.Tensor:
    """fp32 weights (code - zero) * scale. This is the reference the kernel must match."""
    codes = unpack_int4(packed).float()
    n, k = codes.shape
    g = codes.reshape(n, k // group_size, group_size)
    w = (g - zeros.float().unsqueeze(-1)) * scales.float().unsqueeze(-1)
    return w.reshape(n, k)


def int4_to_fp16_magic(codes: torch.Tensor) -> torch.Tensor:
    """Convert codes 0..15 to fp16 with the magic-number trick, with no int-to-float convert.

    An fp16 with exponent bits 0x6400 has the value 1024 and a unit in the last
    place of 1, so OR-ing a code q below 1024 into the mantissa gives 1024 + q.
    """
    bits = codes.to(torch.int16) | 0x6400
    return bits.view(torch.float16) - 1024.0


# ---------------------------------------------------------------------------
# Part 2: the fused kernel
# ---------------------------------------------------------------------------

_CONFIGS = [
    # Decode shapes: BLOCK_M = 16 is the smallest tile tl.dot accepts.
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 16}, num_warps=2, num_stages=4),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 16}, num_warps=4, num_stages=5),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 32}, num_warps=2, num_stages=4),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 32}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_warps=4, num_stages=3),
    # Larger M.
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
    HALF: tl.constexpr = GROUP // 2  # packed bytes per group
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_j = tl.arange(0, HALF)  # byte index inside a group
    m_mask = offs_m < M
    n_mask = offs_n < N

    # X at even and at odd columns of the group: byte j holds codes 2j and 2j + 1.
    x_even_ptrs = x_ptr + offs_m[:, None] * stride_xm + (2 * offs_j)[None, :] * stride_xk
    x_odd_ptrs = x_even_ptrs + stride_xk
    w_ptrs = wq_ptr + offs_n[:, None] * stride_wn + offs_j[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for g in range(0, K // GROUP):
        x_e = tl.load(x_even_ptrs + g * GROUP * stride_xk, mask=m_mask[:, None], other=0.0)
        x_o = tl.load(x_odd_ptrs + g * GROUP * stride_xk, mask=m_mask[:, None], other=0.0)
        packed = tl.load(w_ptrs + g * HALF, mask=n_mask[:, None], other=0).to(tl.int32)
        scale = tl.load(s_ptr + offs_n * stride_gn + g, mask=n_mask, other=0.0).to(tl.float32)
        zero = tl.load(z_ptr + offs_n * stride_gn + g, mask=n_mask, other=0).to(tl.float32)

        q_lo = tl.trans((packed & 0xF).to(tl.float16))  # (HALF, BLOCK_N)
        q_hi = tl.trans((packed >> 4).to(tl.float16))

        part = tl.dot(x_e, q_lo)
        part = tl.dot(x_o, q_hi, part)  # sum over the group of x_k * q_k
        x_sum = tl.sum(x_e.to(tl.float32), axis=1) + tl.sum(x_o.to(tl.float32), axis=1)
        acc += (part - x_sum[:, None] * zero[None, :]) * scale[None, :]

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(tl.float16), mask=m_mask[:, None] & n_mask[None, :])


def w4a16_matmul(x, packed, scales, zeros, group_size: int = GROUP_SIZE) -> torch.Tensor:
    """Y = X @ dequant(W)^T for X of shape (M, K) fp16, returning (M, N) fp16."""
    assert x.is_cuda and x.dtype == torch.float16 and x.dim() == 2
    assert group_size == GROUP_SIZE, "the kernel's tile is one group of 128"
    m, k = x.shape
    n = packed.shape[0]
    assert packed.shape == (n, k // 2) and k % group_size == 0
    y = torch.empty((m, n), dtype=torch.float16, device=x.device)
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
