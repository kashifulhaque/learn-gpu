"""Lab 04: Predict, then measure.

Write the predictors below. The tests then launch kernels with tunable
arithmetic intensity, a fixed amount of work per block, and a tunable number of
loads in flight, and compare each prediction with the measured time. Each
function is plain Python arithmetic. The tests print a table of predicted and
measured times and report the measured ridge point.

Units: TFLOP/s is 10**12 FLOP/s, GB/s is 10**9 byte/s, and a latency is in
clock cycles unless the name says otherwise.
"""

import math  # noqa: F401  (you will want math.ceil)


def ridge_point(peak_tflops: float, bw_gbs: float) -> float:
    """Return the arithmetic intensity, in FLOP per byte, where the roofline bends.

    It is the peak arithmetic rate divided by the memory bandwidth. For 312
    TFLOP/s and 1935 GB/s the answer is 161.2.
    """
    raise NotImplementedError


def roofline_ms(flops: float, nbytes: float, peak_tflops: float, bw_gbs: float) -> float:
    """Return the roofline's lower bound on the kernel's time, in milliseconds.

    The kernel can't finish before its bytes arrive or before its arithmetic is
    done, so the bound is the larger of the two times.
    """
    raise NotImplementedError


def fma_kernel_counts(n: int, k: int) -> tuple[float, float]:
    """Return (flops, nbytes) for a kernel that processes n float32 elements.

    Each element is read once (4 bytes), passed through k fused multiply-adds
    (2 FLOP each), and written once (4 bytes).
    """
    raise NotImplementedError


def num_waves(num_blocks: int, blocks_per_sm: int, sm_count: int) -> int:
    """Return how many waves a grid of `num_blocks` blocks takes.

    A wave is one full set of resident blocks: blocks_per_sm * sm_count of
    them. Return 0 for an empty grid.
    """
    raise NotImplementedError


def wave_efficiency(num_blocks: int, blocks_per_sm: int, sm_count: int) -> float:
    """Return the fraction of block slots the grid fills across all its waves.

    That is num_blocks divided by (waves * blocks_per_sm * sm_count). Return 1.0
    for an empty grid.
    """
    raise NotImplementedError


def gemm_tiles(m: int, n: int, tile_m: int, tile_n: int) -> int:
    """Return the number of output tiles in an m x n GEMM, rounding each side up."""
    raise NotImplementedError


def gemm_wave_efficiency(
    m: int, n: int, tile_m: int, tile_n: int, sm_count: int, blocks_per_sm: int = 1
) -> float:
    """Return the wave efficiency of a GEMM that runs one block per output tile."""
    raise NotImplementedError


def bytes_in_flight(bw_gbs: float, latency_cycles: float, clock_mhz: float) -> float:
    """Little's law: return the bytes that must be in flight, across the whole
    GPU, to sustain `bw_gbs` when each request takes `latency_cycles` cycles at
    `clock_mhz` MHz.
    """
    raise NotImplementedError


def predicted_read_gbs(
    inflight_bytes_per_sm: float,
    latency_cycles: float,
    clock_mhz: float,
    sm_count: int,
    ceiling_gbs: float,
) -> float:
    """Return the read bandwidth Little's law allows, in GB/s.

    Bandwidth is the bytes in flight on the whole GPU divided by the latency in
    seconds, and it can't exceed `ceiling_gbs`.
    """
    raise NotImplementedError


def measured_ridge(ks, times_ms, n: int) -> tuple[float, float, float]:
    """Return (peak_tflops, peak_gbs, ridge) from a sweep of the FMA kernel.

    `ks[i]` is the number of FMAs per element and `times_ms[i]` the measured
    time of the kernel on n elements. Use `fma_kernel_counts` to turn each run
    into a FLOP rate and a byte rate. `peak_tflops` is the highest FLOP rate any
    run reached, `peak_gbs` the highest byte rate, and `ridge` their ratio in
    FLOP per byte.
    """
    raise NotImplementedError
