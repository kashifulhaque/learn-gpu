"""Reference solution for the wavefront reduction lab.

Pure Python. Nothing here compiles CUDA, HIP, Triton, or Pallas.
"""

from __future__ import annotations


def reduce_simd(values, width):
    """Sum `values` with a tree of `width` lanes."""
    width = int(width)
    if width < 1 or len(values) < width:
        raise ValueError("values must hold `width` lanes")
    acc = list(values[:width])
    offset = width // 2
    while offset > 0:
        for lane in range(offset):
            acc[lane] = acc[lane] + acc[lane + offset]
        offset //= 2
    return acc[0]


def ballot_mask(active, width):
    """Return an int with bit `i` set when `active[i]` is true."""
    width = int(width)
    if len(active) < width:
        raise ValueError("active must hold `width` lanes")
    mask = 0
    for lane in range(width):
        if active[lane]:
            mask |= 1 << lane
    return mask


def ridge_flops_per_byte(peak_tflops, bandwidth_gbs):
    """Return FLOP per byte from a TFLOP/s peak and a GB/s bandwidth."""
    if bandwidth_gbs == 0:
        raise ZeroDivisionError("bandwidth_gbs must be non-zero")
    return peak_tflops * 1000 / bandwidth_gbs


def scratchpad_bytes(tile_m, tile_n, elem_bytes, stages):
    """Return the bytes in `stages` copies of one tile."""
    return tile_m * tile_n * elem_bytes * stages


def scratchpad_fits(nbytes, budget_bytes):
    """Return whether `nbytes` is within `budget_bytes`."""
    return nbytes <= budget_bytes


def reduction_source(width):
    """Return a HIP-style reduction whose offset starts at `width / 2`."""
    if int(width) < 1:
        raise ValueError("width must be positive")
    return (
        "__device__ float wave_sum(float v, int width) {\n"
        "    for (int offset = width / 2; offset > 0; offset >>= 1)\n"
        "        v += __shfl_down(v, offset);\n"
        "    return v;\n"
        "}\n"
    )
