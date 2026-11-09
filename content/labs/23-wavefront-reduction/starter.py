"""Wavefront reduction, in pure Python.

The checks never call CUDA, HIP, Triton, or Pallas. Implement the functions
below. `reduce_simd` already sums a width of 4. It still folds only 32 lanes,
so a 64-lane wavefront is wrong until you change the limit.

1. `reduce_simd(values, width)` sums `values` with a tree of `width` lanes.
   `len(values)` is `width`, and `width` is a positive power of two. The
   offset starts at `width / 2`. `[1, 2, 3, 4]` at width 4 returns 10. The
   64-lane vector of 32 ones followed by 32 threes returns 128.
2. `ballot_mask(active, width)` returns an int with bit `i` set when
   `active[i]` is true, for `i` in `0 .. width - 1`. A width of 64 keeps
   bit 40. `mask & 0xFFFFFFFF` drops that bit.
3. `ridge_flops_per_byte(peak_tflops, bandwidth_gbs)` returns
   `peak_tflops * 1000 / bandwidth_gbs`. For 312 and 1935 that is about
   161.24, not 312 / 1935.
4. `scratchpad_bytes(tile_m, tile_n, elem_bytes, stages)` returns
   `tile_m * tile_n * elem_bytes * stages`.
   `scratchpad_fits(nbytes, budget_bytes)` returns `nbytes <= budget_bytes`.
   A 128 by 128 fp16 tile with 3 stages is 98304 bytes. It fits in
   166912 bytes and doesn't fit in 65536 bytes.
5. `reduction_source(width)` returns source text for a HIP-style reduction.
   The offset loop uses `width`, for example `width / 2`. A loop that
   starts at 16 fails the check. The check reads the string. It doesn't
   compile it.
"""

from __future__ import annotations


def reduce_simd(values, width):
    """Sum `values` with a tree of `width` lanes.

    This copy stops at 32 lanes. A width of 4 passes. A width of 64 returns
    the first-half sum, 32, on the lab vector.
    """
    limit = min(int(width), 32)
    if limit < 1 or len(values) < limit:
        raise ValueError("values must hold at least `limit` lanes")
    acc = list(values[:limit])
    offset = limit // 2
    while offset > 0:
        for lane in range(offset):
            acc[lane] = acc[lane] + acc[lane + offset]
        offset //= 2
    return acc[0]


def ballot_mask(active, width):
    """Return an int with bit `i` set when `active[i]` is true."""
    raise NotImplementedError("ballot_mask")


def ridge_flops_per_byte(peak_tflops, bandwidth_gbs):
    """Return peak TFLOP/s times 1000, divided by bandwidth in GB/s."""
    raise NotImplementedError("ridge_flops_per_byte")


def scratchpad_bytes(tile_m, tile_n, elem_bytes, stages):
    """Return tile_m * tile_n * elem_bytes * stages."""
    raise NotImplementedError("scratchpad_bytes")


def scratchpad_fits(nbytes, budget_bytes):
    """Return whether `nbytes` fits in `budget_bytes`."""
    raise NotImplementedError("scratchpad_fits")


def reduction_source(width):
    """Return HIP-style reduction source whose loop bound uses `width`."""
    raise NotImplementedError("reduction_source")
