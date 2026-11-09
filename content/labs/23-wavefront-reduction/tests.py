"""Tests for the wavefront reduction lab.

Pure Python. No randomness and no gradient. Each check message shows the
expected value and the actual value.

Wrong implementations these checks catch:

- fold_first_32: reduce_simd sums only values[:32]. On
  [1] * 32 + [3] * 32 it returns 32. The full sum is 128.
- mask_low_32: ballot_mask returns mask & 0xFFFFFFFF. Lane 40 at width 64
  becomes 0, because bit 40 sits above bit 31.
- ridge_ratio: ridge_flops_per_byte returns peak_tflops / bandwidth_gbs.
  For 312 and 1935 that is about 0.161, not about 161.24.
- bytes_drop_stages: scratchpad_bytes ignores stages. 128 * 128 * 2 * 3
  is 98304. Dropping stages yields 32768.
- fits_a100_budget: scratchpad_fits compares against 163 * 1024 and ignores
  budget_bytes. 98304 fits that cap, so a budget of 65536 incorrectly passes.
- hip_offset_16: reduction_source returns only
  `for (int offset = 16; ...)` and doesn't use width in the loop bound.
"""

from __future__ import annotations

import re

from lab_common import Checks

# 32 ones (sum 32) then 32 threes (sum 96). The full sum is 128.
WAVE64 = [1] * 32 + [3] * 32
A100_BYTES = 163 * 1024  # 166912, the A100 per-block shared-memory cap
LDS_BYTES = 64 * 1024  # 65536, the MI300X LDS
TILE_BYTES = 128 * 128 * 2 * 3  # 98304
# Absolute tolerance: two-decimal rounding of the quotient passes.
# 312 / 1935 (about 0.161) and the table's integer 161 do not.
RIDGE_TOL = 1e-3
OFFSET_LOOP = re.compile(r"for\s*\(\s*int\s+offset\s*=\s*([^;]+);")


def invoke(fn, *args):
    """Call `fn` and return `(value, None)`, or `(None, exception)`."""
    try:
        return fn(*args), None
    except Exception as exc:  # noqa: BLE001
        return None, exc


def show(expected, actual, exc):
    """Build a pass/fail pair whose message shows expected and actual."""
    if exc is not None:
        return False, f"expected {expected}, actual {type(exc).__name__}: {exc}"
    return actual == expected, f"expected {expected}, actual {actual}"


def run(submission):
    c = Checks()
    names = (
        "reduce_simd",
        "ballot_mask",
        "ridge_flops_per_byte",
        "scratchpad_bytes",
        "scratchpad_fits",
        "reduction_source",
    )
    if not c.require(submission, *names):
        return c.finish()

    # fold_first_32 still returns 10 here, because 4 is under 32.
    c.check(
        "reduce_simd sums [1, 2, 3, 4] at width 4",
        lambda: show(*_pack(invoke(submission.reduce_simd, [1, 2, 3, 4], 4), 10)),
    )

    # fold_first_32 returns 32 on WAVE64. The full tree returns 128.
    c.check(
        "reduce_simd sums all 64 lanes",
        lambda: show(*_pack(invoke(submission.reduce_simd, WAVE64, 64), 128)),
    )

    # Smallest ballot: bits 0 and 2 set is 5. mask_low_32 still passes this.
    active4 = [True, False, True, False]
    c.check(
        "ballot_mask sets bits for a width of 4",
        lambda: show(*_pack(invoke(submission.ballot_mask, active4, 4), 0b0101)),
    )

    # mask_low_32 returns 0. Bit 40 must stay set.
    active64 = [False] * 64
    active64[40] = True
    c.check(
        "ballot_mask keeps lane 40 at width 64",
        lambda: _ballot_lane_40(submission, active64),
    )

    # ridge_ratio returns about 0.161 and about 0.247. The factor of 1000
    # is the difference between TFLOP/s and FLOP per byte.
    c.check(
        "ridge_flops_per_byte scales TFLOP/s over GB/s by 1000",
        lambda: _ridge(submission),
    )

    # bytes_drop_stages returns 32768 for the 128 by 128 fp16 tile.
    c.check(
        "scratchpad_bytes counts the stages",
        lambda: show(*_pack(invoke(submission.scratchpad_bytes, 128, 128, 2, 3), TILE_BYTES)),
    )

    # fits_a100_budget returns True for 98304 against a 65536 budget.
    c.check(
        "scratchpad_fits uses the budget you pass",
        lambda: _fits(submission),
    )

    # hip_offset_16 has `for (int offset = 16;` and never mentions width
    # in that initializer.
    c.check(
        "reduction_source bounds the loop with width",
        lambda: _source(submission),
    )

    wave, wave_exc = invoke(submission.reduce_simd, WAVE64, 64)
    ridge, ridge_exc = invoke(submission.ridge_flops_per_byte, 312, 1935)
    tile, tile_exc = invoke(submission.scratchpad_bytes, 128, 128, 2, 3)
    c.metric("wave64_sum", None if wave_exc else wave)
    c.metric("a100_ridge_flop_per_byte", None if ridge_exc else round(ridge, 4))
    c.metric("tile_scratchpad_bytes", None if tile_exc else tile)
    return c.finish()


def _pack(called, expected):
    actual, exc = called
    return expected, actual, exc


def _ballot_lane_40(submission, active64):
    expected = 1 << 40
    actual, exc = invoke(submission.ballot_mask, active64, 64)
    if exc is not None:
        return False, f"expected {expected}, actual {type(exc).__name__}: {exc}"
    # A 32-bit truncation is 0. Say so when that is what came back.
    if actual == (expected & 0xFFFFFFFF):
        return False, f"expected {expected}, actual {actual}"
    return actual == expected, f"expected {expected}, actual {actual}"


def _ridge(submission):
    cases = ((312, 1935), (1307.4, 5300))
    for peak, bandwidth in cases:
        expected = peak * 1000 / bandwidth
        actual, exc = invoke(submission.ridge_flops_per_byte, peak, bandwidth)
        if exc is not None:
            return False, f"expected {expected:.6f}, actual {type(exc).__name__}: {exc}"
        try:
            ok = abs(float(actual) - expected) <= RIDGE_TOL
        except (TypeError, ValueError):
            ok = False
        if not ok:
            return False, f"expected {expected:.6f}, actual {actual}"
    return True, f"expected {312 * 1000 / 1935:.6f} and {1307.4 * 1000 / 5300:.6f}, actual matches"


def _fits(submission):
    cases = (
        (TILE_BYTES, A100_BYTES, True),
        (TILE_BYTES, LDS_BYTES, False),
        (256 * 256 * 2 * 3, A100_BYTES, False),
    )
    details = []
    for nbytes, budget, expected in cases:
        actual, exc = invoke(submission.scratchpad_fits, nbytes, budget)
        if exc is not None:
            return False, (
                f"expected {expected} for {nbytes} against {budget}, "
                f"actual {type(exc).__name__}: {exc}"
            )
        if actual != expected:
            return False, f"expected {expected} for {nbytes} against {budget}, actual {actual}"
        details.append(f"{nbytes} against {budget} is {actual}")
    return True, "expected fits, and actual " + "; ".join(details)


def _source(submission):
    actual, exc = invoke(submission.reduction_source, 64)
    if exc is not None:
        return False, f"expected a loop bound that uses width, actual {type(exc).__name__}: {exc}"
    if not isinstance(actual, str):
        return False, f"expected a string that mentions width, actual {type(actual).__name__}"
    loops = OFFSET_LOOP.findall(actual)
    if not loops:
        return False, f"expected a for-loop that sets offset from width, actual {actual!r}"
    for init in loops:
        if "width" not in init:
            return False, f"expected the offset loop to use width, actual {init.strip()!r}"
    return True, f"expected a width-based offset, actual {loops[0].strip()!r}"
