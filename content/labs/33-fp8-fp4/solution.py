"""Lab 33: FP8, MXFP4, and NVFP4 numerics, bit by bit.

Everything here runs on a CPU. The only kernel is `triton_block_scaled_gemm`,
which needs a CUDA GPU and Triton. Every other function is plain torch math.

Conventions used throughout the lab:

* Minifloat rounding is round-to-nearest, ties to even (RNE).
* "Saturate" means an overflow becomes the largest finite value instead of
  infinity or NaN.
* An fp8 `fmt` is the string "e4m3" (the fn variant: no infinity, one NaN
  pattern, largest value 448) or "e5m2" (IEEE-style, largest value 57344).
"""

from __future__ import annotations

# (mantissa bits, smallest normal exponent, largest finite value)
FP8 = {
    "e4m3": {"m": 3, "emin": -6, "bias": 7, "max": 448.0, "ebits": 4},
    "e5m2": {"m": 2, "emin": -14, "bias": 15, "max": 57344.0, "ebits": 5},
}

# Positive values of E2M1, indexed by the 3 low bits of the code.
E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_EMAX = 2  # exponent of the largest E2M1 value, 6 = 1.5 * 2**2
E4M3_MAX = 448.0
MX_BLOCK = 32
NV_BLOCK = 16


# ---------------------------------------------------------------- E2M1 ----


def e2m1_decode(codes):
    """Decode 4-bit E2M1 codes (0 to 15) to float32.

    Bit 3 is the sign. Bits 2-0 index E2M1_GRID. Code 8 is negative zero.
    """
    import torch

    codes = torch.as_tensor(codes).to(torch.int64)
    grid = torch.tensor(E2M1_GRID, dtype=torch.float32)
    mag = grid[codes & 7]
    return torch.where((codes & 8) != 0, -mag, mag)


def round_minifloat(x, m_bits, emin, max_finite, saturate=True):
    """Round to the nearest minifloat value, ties to even.

    `m_bits` is the number of mantissa bits, `emin` the exponent of the
    smallest normal number, and `max_finite` the largest finite value. A
    value that rounds above `max_finite` becomes `max_finite` when
    `saturate` is true and infinity when it is false. NaN stays NaN and the
    sign of zero is kept. Returns float64.
    """
    import torch

    x = torch.as_tensor(x).to(torch.float64)
    a = x.abs()
    _, ex = torch.frexp(a)  # a = mant * 2**ex with mant in [0.5, 1)
    e = torch.clamp(ex.to(torch.int64) - 1, min=emin)  # floor(log2 a), at least emin
    step = torch.ldexp(torch.ones_like(a), e - m_bits)  # the gap in this binade
    q = torch.round(a / step) * step  # torch.round rounds ties to even
    over = q > max_finite
    q = torch.where(over, torch.full_like(q, max_finite if saturate else float("inf")), q)
    q = torch.where(torch.isnan(x), torch.full_like(q, float("nan")), q)
    return torch.copysign(q, x)


def e2m1_encode(x):
    """Encode to 4-bit E2M1 codes (uint8, values 0 to 15), saturating at 6."""
    import torch

    x = torch.as_tensor(x).to(torch.float64)
    q = round_minifloat(x, 1, 0, 6.0, saturate=True)
    grid = torch.tensor(E2M1_GRID, dtype=torch.float64)
    idx = (q.abs().unsqueeze(-1) == grid).to(torch.int64).argmax(-1)
    sign = torch.signbit(q).to(torch.int64)
    return (sign * 8 + idx).to(torch.uint8)


# ----------------------------------------------------------------- fp8 ----


def decode_fp8(codes, fmt):
    """Decode uint8 fp8 codes to float32 by hand: sign, exponent, mantissa."""
    import torch

    p = FP8[fmt]
    m, bias, ebits = p["m"], p["bias"], p["ebits"]
    c = torch.as_tensor(codes).to(torch.int64)
    sign = (c >> 7) & 1
    E = (c >> m) & ((1 << ebits) - 1)
    M = c & ((1 << m) - 1)
    normal = torch.ldexp(1.0 + M.to(torch.float64) / (1 << m), (E - bias))
    subnormal = torch.ldexp(M.to(torch.float64) / (1 << m), torch.full_like(E, 1 - bias))
    val = torch.where(E == 0, subnormal, normal)
    all_ones = (1 << ebits) - 1
    if fmt == "e4m3":
        val = torch.where((E == all_ones) & (M == (1 << m) - 1), torch.full_like(val, float("nan")), val)
    else:
        special = torch.where(M == 0, torch.full_like(val, float("inf")), torch.full_like(val, float("nan")))
        val = torch.where(E == all_ones, special, val)
    val = torch.where(sign == 1, -val, val)
    return val.to(torch.float32)


def encode_fp8(x, fmt, saturate=False):
    """Encode float values to fp8 codes (uint8), equal to a torch cast in range.

    With `saturate=False` an overflow becomes NaN for e4m3 and infinity for
    e5m2, the non-saturating rule of the format. Torch releases disagree here
    (2.11 gives NaN, 2.15 saturates), so the tests check the rule, not torch.
    With `saturate=True` it becomes the largest finite value.
    """
    import torch

    p = FP8[fmt]
    m, bias = p["m"], p["bias"]
    x = torch.as_tensor(x).to(torch.float64)
    q = round_minifloat(x, m, p["emin"], p["max"], saturate=saturate)
    sign = torch.signbit(x).to(torch.int64)
    a = q.abs()
    finite = torch.isfinite(a)
    a0 = torch.where(finite, a, torch.zeros_like(a))
    _, ex = torch.frexp(a0)
    e = ex.to(torch.int64) - 1  # floor(log2 a)
    sub = e < p["emin"]
    E = torch.where(sub, torch.zeros_like(e), e + bias)
    M_sub = torch.ldexp(a0, torch.full_like(e, m - p["emin"]))
    M_norm = torch.ldexp(a0, m - e) - (1 << m)
    M = torch.where(sub, M_sub, M_norm).round().to(torch.int64)
    zero = a0 == 0
    M = torch.where(zero, torch.zeros_like(M), M)
    E = torch.where(zero, torch.zeros_like(E), E)
    code = (sign << 7) | (E << m) | M
    inf_code = 0x7F if fmt == "e4m3" else 0x7C  # e4m3fn has no infinity: NaN
    code = torch.where(torch.isinf(a), (sign << 7) | inf_code, code)
    code = torch.where(torch.isnan(x), torch.full_like(code, 0x7F), code)
    return code.to(torch.uint8)


# ------------------------------------------------------- MXFP4 and NVFP4 --


def e8m0_exponent(amax, emax_elem):
    """Shared exponent of an MX block: floor(log2(amax)) - emax_elem.

    Returns an int64 tensor clamped to [-127, 127], the range of E8M0.
    `amax == 0` returns -127.
    """
    import torch

    amax = torch.as_tensor(amax).to(torch.float64)
    _, ex = torch.frexp(amax)
    s = ex.to(torch.int64) - 1 - emax_elem  # floor(log2(amax)) is ex - 1
    s = torch.where(amax == 0, torch.full_like(s, -127), s)
    return torch.clamp(s, -127, 127)


def mxfp4_quantize(x):
    """Quantize the last dimension in blocks of 32 to E2M1 with E8M0 scales.

    Returns `(codes, scale_codes)`. `codes` is uint8 with the shape of `x`.
    `scale_codes` is uint8 with shape `(..., K // 32)`: the E8M0 byte, which
    is the exponent plus 127.
    """
    import torch

    x = torch.as_tensor(x).to(torch.float64)
    k = x.shape[-1]
    if k % MX_BLOCK:
        raise ValueError(f"last dimension {k} is not a multiple of {MX_BLOCK}")
    blocks = x.reshape(*x.shape[:-1], k // MX_BLOCK, MX_BLOCK)
    s = e8m0_exponent(blocks.abs().amax(-1), E2M1_EMAX)
    scale = torch.ldexp(torch.ones_like(s, dtype=torch.float64), s).unsqueeze(-1)
    codes = e2m1_encode(blocks / scale)
    return codes.reshape(x.shape), (s + 127).to(torch.uint8)


def mxfp4_dequantize(codes, scale_codes):
    """Inverse of `mxfp4_quantize`. Returns float32. Scale byte 255 is NaN."""
    import torch

    vals = e2m1_decode(codes).to(torch.float64)
    k = vals.shape[-1]
    blocks = vals.reshape(*vals.shape[:-1], k // MX_BLOCK, MX_BLOCK)
    sc = scale_codes.to(torch.int64)
    scale = torch.ldexp(torch.ones(sc.shape, dtype=torch.float64), sc - 127)
    scale = torch.where(sc == 255, torch.full_like(scale, float("nan")), scale)
    return (blocks * scale.unsqueeze(-1)).reshape(vals.shape).to(torch.float32)


def nvfp4_quantize(x):
    """Quantize the last dimension in blocks of 16 to E2M1 with E4M3 scales.

    Two-level scaling:
      tensor_scale = amax(x) / (448 * 6)                         (fp32)
      block_scale  = E4M3(amax_block / 6 / tensor_scale)         (saturating)
      element      = E2M1(x / (block_scale * tensor_scale))
    Returns `(codes, block_scale_codes, tensor_scale)`: uint8 E2M1 codes with
    the shape of `x`, uint8 E4M3 bytes with shape `(..., K // 16)`, and a
    Python float. An all-zero tensor uses a tensor scale of 1.
    """
    import torch

    x = torch.as_tensor(x).to(torch.float64)
    k = x.shape[-1]
    if k % NV_BLOCK:
        raise ValueError(f"last dimension {k} is not a multiple of {NV_BLOCK}")
    amax_t = x.abs().max().item()
    ts = float(torch.tensor(amax_t / (E4M3_MAX * 6.0), dtype=torch.float32)) if amax_t > 0 else 1.0
    blocks = x.reshape(*x.shape[:-1], k // NV_BLOCK, NV_BLOCK)
    amax_b = blocks.abs().amax(-1)
    s_codes = encode_fp8(amax_b / 6.0 / ts, "e4m3", saturate=True)
    s_b = decode_fp8(s_codes, "e4m3").to(torch.float64)
    eff = (s_b * ts).unsqueeze(-1)
    safe = torch.where(eff == 0, torch.ones_like(eff), eff)
    scaled = torch.where(eff == 0, torch.zeros_like(blocks), blocks / safe)
    return e2m1_encode(scaled).reshape(x.shape), s_codes, ts


def nvfp4_dequantize(codes, block_scale_codes, tensor_scale):
    """Inverse of `nvfp4_quantize`. Returns float32."""
    import torch

    vals = e2m1_decode(codes).to(torch.float64)
    k = vals.shape[-1]
    blocks = vals.reshape(*vals.shape[:-1], k // NV_BLOCK, NV_BLOCK)
    s_b = decode_fp8(block_scale_codes, "e4m3").to(torch.float64)
    return (blocks * (s_b * tensor_scale).unsqueeze(-1)).reshape(vals.shape).to(torch.float32)


# ----------------------------------------------------------- rounding -----


def stochastic_round_e2m1(x, generator):
    """Round to the E2M1 grid, up with probability equal to the fraction.

    For a value between neighbours `lo` and `hi`, return `hi` with
    probability `(x - lo) / (hi - lo)` and `lo` otherwise, so the mean of
    many roundings is `x`. Clamp to [-6, 6]. Draw one uniform number per
    element from `generator` (a seeded `torch.Generator`). Returns float32.
    """
    import torch

    x = torch.as_tensor(x).to(torch.float64).clamp(-6.0, 6.0)
    a = x.abs()
    grid = torch.tensor(E2M1_GRID, dtype=torch.float64)
    hi_idx = torch.searchsorted(grid, a, right=False)
    hi = grid[hi_idx]
    lo = grid[torch.clamp(hi_idx - 1, min=0)]
    gap = hi - lo
    p = torch.where(gap > 0, (a - lo) / torch.where(gap > 0, gap, torch.ones_like(gap)), torch.zeros_like(a))
    u = torch.rand(a.shape, generator=generator, dtype=torch.float64)
    mag = torch.where(u < p, hi, lo)
    return torch.copysign(mag, x).to(torch.float32)


# ------------------------------------------------------------ scaling -----


def fp8_scale(amax_now, history, mode, fmt_max=E4M3_MAX):
    """Scale that maps a tensor into the fp8 range: amax / fmt_max.

    `mode == "current"` uses `amax_now`, the amax of the tensor about to be
    quantized. `mode == "delayed"` uses the largest amax in `history`, the
    amaxes of earlier steps, and falls back to `amax_now` when the history
    is empty. A scale of zero becomes 1.0.
    """
    if mode == "current":
        amax = amax_now
    elif mode == "delayed":
        amax = max(history) if len(history) else amax_now
    else:
        raise ValueError(f"unknown mode {mode!r}")
    scale = float(amax) / fmt_max
    return scale if scale > 0 else 1.0


def quantize_with_scale(x, scale, fmt="e4m3"):
    """Divide by `scale`, encode to fp8 with saturation, decode, multiply back."""
    import torch

    x = torch.as_tensor(x).to(torch.float32)
    codes = encode_fp8(x / scale, fmt, saturate=True)
    return (decode_fp8(codes, fmt) * scale).to(torch.float32)


def block_shape(granularity, shape):
    """Rows and columns that share one scale, for a 2-D tensor of `shape`.

    "tensor" -> the whole tensor, "row" -> one row, "col" -> one column,
    "tile1x128" -> 1 row by 128 columns, "block128x128" -> 128 by 128.
    """
    rows, cols = shape
    table = {
        "tensor": (rows, cols),
        "row": (1, cols),
        "col": (rows, 1),
        "tile1x128": (1, 128),
        "block128x128": (128, 128),
    }
    return table[granularity]


def fp8_fake_quant(x, granularity, fmt="e4m3"):
    """Quantize a 2-D tensor to fp8 with one scale per block, then dequantize.

    The scale of a block is `amax / 448` (current scaling), and an all-zero
    block uses 1. Edge blocks hold fewer elements when a dimension is not a
    multiple of the block size. Returns float32 with the shape of `x`.
    """
    import torch

    x = torch.as_tensor(x).to(torch.float32)
    rows, cols = x.shape
    br, bc = block_shape(granularity, x.shape)
    pr, pc = -rows % br, -cols % bc
    xp = torch.nn.functional.pad(x, (0, pc, 0, pr))
    rb, cb = xp.shape[0] // br, xp.shape[1] // bc
    blocks = xp.reshape(rb, br, cb, bc)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True)
    scale = torch.where(amax > 0, amax / FP8[fmt]["max"], torch.ones_like(amax))
    out = quantize_with_scale(blocks, scale, fmt)
    return out.reshape(xp.shape)[:rows, :cols].contiguous()


def block_scaled_gemm(a, b, a_granularity="tile1x128", b_granularity="block128x128"):
    """Emulated FP8 GEMM: fake-quantize both operands, then multiply in fp32.

    `a` is (M, K) and `b` is (K, N). The defaults are the DeepSeek-V3 layout.
    """
    import torch

    aq = fp8_fake_quant(a, a_granularity)
    bq = fp8_fake_quant(b, b_granularity)
    return aq.to(torch.float32) @ bq.to(torch.float32)


def round_to_bits(x, bits):
    """Round a float64 tensor to `bits` significant bits, ties to even."""
    import torch

    mant, ex = torch.frexp(x)
    return torch.ldexp(torch.round(mant * (1 << bits)) / (1 << bits), ex)


def gemm_limited_accumulator(a, b, n_c=128, acc_bits=14, k_inst=32):
    """GEMM with a narrow tensor-core accumulator and fp32 promotion.

    Model: the tensor core adds `k_inst` products at a time into an
    accumulator that keeps `acc_bits` significant bits, so every step ends
    with `round_to_bits`. Every `n_c` elements of K, the partial result
    is added to an fp32 total and the accumulator restarts from zero.
    `n_c >= K` means no promotion. `a` is (M, K) and `b` is (K, N).
    Returns float32 (M, N).
    """
    import torch

    a = torch.as_tensor(a).to(torch.float64)
    b = torch.as_tensor(b).to(torch.float64)
    k = a.shape[1]
    total = torch.zeros(a.shape[0], b.shape[1], dtype=torch.float32)
    for k0 in range(0, k, n_c):
        acc = torch.zeros(a.shape[0], b.shape[1], dtype=torch.float64)
        for k1 in range(k0, min(k0 + n_c, k), k_inst):
            k2 = min(k1 + k_inst, k0 + n_c, k)
            acc = round_to_bits(acc + a[:, k1:k2] @ b[k1:k2], acc_bits)
        total = total + acc.to(torch.float32)
    return total


# ------------------------------------------------------------- Triton -----


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _block_scaled_kernel(
        a_ptr, b_ptr, sa_ptr, sb_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        stride_sam, stride_sak, stride_sbk, stride_sbn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        total = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for kb in range(0, tl.cdiv(K, BLOCK_K)):
            k = kb * BLOCK_K + rk
            a = tl.load(
                a_ptr + rm[:, None] * stride_am + k[None, :] * stride_ak,
                mask=(rm[:, None] < M) & (k[None, :] < K), other=0.0,
            )
            b = tl.load(
                b_ptr + k[:, None] * stride_bk + rn[None, :] * stride_bn,
                mask=(k[:, None] < K) & (rn[None, :] < N), other=0.0,
            )
            partial = tl.dot(a, b, out_dtype=tl.float32)  # fp32 accumulator
            sa = tl.load(sa_ptr + rm * stride_sam + kb * stride_sak, mask=rm < M, other=0.0)
            sb = tl.load(sb_ptr + kb * stride_sbk + (rn // 128) * stride_sbn, mask=rn < N, other=0.0)
            total += partial * sa[:, None] * sb[None, :]  # promote with the scales
        tl.store(
            c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
            total, mask=(rm[:, None] < M) & (rn[None, :] < N),
        )

except ImportError:  # no Triton on this machine
    _block_scaled_kernel = None


def triton_block_scaled_gemm(aq, a_scale, bq, b_scale):
    """Block-scaled GEMM on the GPU. Needs CUDA and Triton.

    `aq` is (M, K) fp16 holding fp8 grid values, `a_scale` is (M, ceil(K/128))
    fp32. `bq` is (K, N) fp16, `b_scale` is (ceil(K/128), ceil(N/128)) fp32.
    Each K step of 128 does `dot(aq_tile, bq_tile)` with an fp32 accumulator,
    then promotes `partial * a_scale * b_scale` into an fp32 total. Returns
    float32 (M, N).
    """
    import torch
    import triton

    m, k = aq.shape
    n = bq.shape[1]
    out = torch.empty(m, n, device=aq.device, dtype=torch.float32)
    grid = (triton.cdiv(m, 64), triton.cdiv(n, 64))
    _block_scaled_kernel[grid](
        aq, bq, a_scale, b_scale, out, m, n, k,
        aq.stride(0), aq.stride(1), bq.stride(0), bq.stride(1), out.stride(0), out.stride(1),
        a_scale.stride(0), a_scale.stride(1), b_scale.stride(0), b_scale.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=128,
    )
    return out
