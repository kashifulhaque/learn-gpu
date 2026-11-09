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


def partial_wave(num_blocks: int, sm_count: int) -> int:
    """Return how many blocks sit in the last, partial wave.

    A wave holds `sm_count` blocks when one block is resident per SM, so the
    tail is num_blocks % sm_count. 117 blocks on 108 SMs leave 9. A grid that
    fills every wave, such as 108 blocks, leaves 0.
    """
    # floor_partial returns num_blocks // sm_count and gives 1 for 117 blocks
    # on 108 SMs. The tail is the remainder, not the number of full waves.
    return num_blocks % sm_count


def split_k_blocks(tiles: int, splits: int) -> int:
    """Return the number of blocks a split-K GEMM launches.

    Each output tile becomes `splits` blocks, one per slice of K, so the grid
    is tiles * splits. Two splits of 117 tiles is 234 blocks, and the partial
    wave of that grid on 108 SMs is 18.
    """
    # split_ignores_k returns tiles and fails the 234-block check.
    raise NotImplementedError


def stream_k_bounds(total_iters: int, sm_count: int) -> tuple[int, int]:
    """Return (lo, hi), the fewest and most K-iterations any SM receives.

    The iterations are dealt out one at a time. lo is total_iters // sm_count.
    hi is lo + 1 when total_iters % sm_count is nonzero, and lo otherwise.
    117 tiles with 4 K-iterations each is 468 iterations. Across 108 SMs the
    bounds are (4, 5): 36 SMs receive 5 and 72 receive 4.
    """
    # even_tiles returns (total_iters // sm_count, total_iters // sm_count)
    # and gives (4, 4) for 468 iterations on 108 SMs.
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
