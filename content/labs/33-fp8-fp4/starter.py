"""Lab 33: FP8, MXFP4, and NVFP4 numerics, bit by bit.

Write each function that ends in `raise NotImplementedError`, in file order.
The first check needs only `e2m1_decode`.

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
    # Index E2M1_GRID with the low 3 bits and apply the sign bit.
    raise NotImplementedError


def round_minifloat(x, m_bits, emin, max_finite, saturate=True):
    """Round to the nearest minifloat value, ties to even.

    `m_bits` is the number of mantissa bits, `emin` the exponent of the
    smallest normal number, and `max_finite` the largest finite value. A
    value that rounds above `max_finite` becomes `max_finite` when
    `saturate` is true and infinity when it is false. NaN stays NaN and the
    sign of zero is kept. Returns float64.
    """
    # TODO: Find the binade with torch.frexp, then the gap 2**(e - m_bits),
    # clamped at the subnormal exponent emin. Round a / gap with torch.round (ties
    # to even), multiply back, and handle overflow and NaN.
    raise NotImplementedError


def e2m1_encode(x):
    """Encode to 4-bit E2M1 codes (uint8, values 0 to 15), saturating at 6."""
    # TODO: Round with round_minifloat, then map each magnitude to its index in
    # E2M1_GRID. Keep the sign of a negative zero.
    raise NotImplementedError


# ----------------------------------------------------------------- fp8 ----


def decode_fp8(codes, fmt):
    """Decode uint8 fp8 codes to float32 by hand: sign, exponent, mantissa."""
    # TODO: sign = bit 7, E = exponent field, M = mantissa field. E == 0 is
    # subnormal. e4m3 has one NaN code, and e5m2 has infinity and NaN.
    raise NotImplementedError


def encode_fp8(x, fmt, saturate=False):
    """Encode float values to fp8 codes (uint8), equal to a torch cast in range.

    With `saturate=False` an overflow becomes NaN for e4m3 and infinity for
    e5m2, the non-saturating rule of the format. Torch releases disagree here
    (2.11 gives NaN, 2.15 saturates), so the tests check the rule, not torch.
    With `saturate=True` it becomes the largest finite value.
    """
    # TODO: Round with round_minifloat, then pack sign, exponent field, and
    # mantissa field. An overflow is NaN (e4m3) or infinity (e5m2) unless saturate
    # is true.
    raise NotImplementedError


# ------------------------------------------------------- MXFP4 and NVFP4 --


def e8m0_exponent(amax, emax_elem):
    """Shared exponent of an MX block: floor(log2(amax)) - emax_elem.

    Returns an int64 tensor clamped to [-127, 127], the range of E8M0.
    `amax == 0` returns -127.
    """
    # TODO: floor(log2(amax)) is torch.frexp's exponent minus 1. Use floor, not ceil.
    raise NotImplementedError


def mxfp4_quantize(x):
    """Quantize the last dimension in blocks of 32 to E2M1 with E8M0 scales.

    Returns `(codes, scale_codes)`. `codes` is uint8 with the shape of `x`.
    `scale_codes` is uint8 with shape `(..., K // 32)`: the E8M0 byte, which
    is the exponent plus 127.
    """
    # TODO: Reshape to blocks of 32, take the block amax, get the exponent, divide
    # by 2**exponent, and encode with e2m1_encode.
    raise NotImplementedError


def mxfp4_dequantize(codes, scale_codes):
    """Inverse of `mxfp4_quantize`. Returns float32. Scale byte 255 is NaN."""
    # TODO: Decode, then multiply each block by 2**(scale_code - 127).
    raise NotImplementedError


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
    # TODO: Follow the three formulas in the docstring. Encode the block scale
    # with encode_fp8(..., 'e4m3', saturate=True).
    raise NotImplementedError


def nvfp4_dequantize(codes, block_scale_codes, tensor_scale):
    """Inverse of `nvfp4_quantize`. Returns float32."""
    # TODO: Decode the elements and the block scales, then multiply by the tensor scale.
    raise NotImplementedError


# ----------------------------------------------------------- rounding -----


def stochastic_round_e2m1(x, generator):
    """Round to the E2M1 grid, up with probability equal to the fraction.

    For a value between neighbours `lo` and `hi`, return `hi` with
    probability `(x - lo) / (hi - lo)` and `lo` otherwise, so the mean of
    many roundings is `x`. Clamp to [-6, 6]. Draw one uniform number per
    element from `generator` (a seeded `torch.Generator`). Returns float32.
    """
    # TODO: Find lo and hi on the grid with torch.searchsorted, then compare
    # torch.rand(..., generator=generator) with the fraction.
    raise NotImplementedError


# ------------------------------------------------------------ scaling -----


def fp8_scale(amax_now, history, mode, fmt_max=E4M3_MAX):
    """Scale that maps a tensor into the fp8 range: amax / fmt_max.

    `mode == "current"` uses `amax_now`, the amax of the tensor about to be
    quantized. `mode == "delayed"` uses the largest amax in `history`, the
    amaxes of earlier steps, and falls back to `amax_now` when the history
    is empty. A scale of zero becomes 1.0.
    """
    # TODO: amax / fmt_max, where amax comes from `amax_now` or from the history.
    raise NotImplementedError


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
    # TODO: Pad to whole blocks, reshape to (row blocks, br, column blocks, bc),
    # take the amax over dims (1, 3), and call quantize_with_scale.
    raise NotImplementedError


def block_scaled_gemm(a, b, a_granularity="tile1x128", b_granularity="block128x128"):
    """Emulated FP8 GEMM: fake-quantize both operands, then multiply in fp32.

    `a` is (M, K) and `b` is (K, N). The defaults are the DeepSeek-V3 layout.
    """
    # TODO: Fake-quantize both operands, then multiply in float32.
    raise NotImplementedError


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
    # TODO: Loop over chunks of n_c along K. Inside a chunk, start the accumulator
    # at zero and step by k_inst with round_to_bits. Add each finished chunk to an
    # fp32 total.
    raise NotImplementedError


# ------------------------------------------------------------- Triton -----


def triton_block_scaled_gemm(aq, a_scale, bq, b_scale):
    """Block-scaled GEMM on the GPU. Needs CUDA and Triton.

    `aq` is (M, K) fp16 holding fp8 grid values, `a_scale` is (M, ceil(K/128))
    fp32. `bq` is (K, N) fp16, `b_scale` is (ceil(K/128), ceil(N/128)) fp32.
    Each K step of 128 does `dot(aq_tile, bq_tile)` with an fp32 accumulator,
    then promotes `partial * a_scale * b_scale` into an fp32 total. Returns
    float32 (M, N).
    """
    # TODO (GPU only): One program per 64 by 64 output tile. Loop over K in steps
    # of 128, take tl.dot of the fp16 tiles into an fp32 partial, then add partial
    # * a_scale * b_scale to a fp32 total.
    raise NotImplementedError
