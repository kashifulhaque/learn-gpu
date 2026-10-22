"""Lab 04: Predict, then measure. Reference solution."""

import math


def ridge_point(peak_tflops: float, bw_gbs: float) -> float:
    # TFLOP/s * 1e12 over GB/s * 1e9 is 1e3 * the ratio of the two numbers.
    return peak_tflops * 1e12 / (bw_gbs * 1e9)


def roofline_ms(flops: float, nbytes: float, peak_tflops: float, bw_gbs: float) -> float:
    compute_s = flops / (peak_tflops * 1e12)
    memory_s = nbytes / (bw_gbs * 1e9)
    return max(compute_s, memory_s) * 1e3


def fma_kernel_counts(n: int, k: int) -> tuple[float, float]:
    # k FMAs per element, 2 FLOP each. One float32 read and one write: 8 bytes.
    return 2.0 * k * n, 8.0 * n


def num_waves(num_blocks: int, blocks_per_sm: int, sm_count: int) -> int:
    capacity = blocks_per_sm * sm_count
    return math.ceil(num_blocks / capacity) if num_blocks > 0 else 0


def wave_efficiency(num_blocks: int, blocks_per_sm: int, sm_count: int) -> float:
    waves = num_waves(num_blocks, blocks_per_sm, sm_count)
    if waves == 0:
        return 1.0
    return num_blocks / (waves * blocks_per_sm * sm_count)


def gemm_tiles(m: int, n: int, tile_m: int, tile_n: int) -> int:
    return math.ceil(m / tile_m) * math.ceil(n / tile_n)


def gemm_wave_efficiency(
    m: int, n: int, tile_m: int, tile_n: int, sm_count: int, blocks_per_sm: int = 1
) -> float:
    return wave_efficiency(gemm_tiles(m, n, tile_m, tile_n), blocks_per_sm, sm_count)


def bytes_in_flight(bw_gbs: float, latency_cycles: float, clock_mhz: float) -> float:
    latency_s = latency_cycles / (clock_mhz * 1e6)
    return bw_gbs * 1e9 * latency_s


def predicted_read_gbs(
    inflight_bytes_per_sm: float,
    latency_cycles: float,
    clock_mhz: float,
    sm_count: int,
    ceiling_gbs: float,
) -> float:
    latency_s = latency_cycles / (clock_mhz * 1e6)
    littles = inflight_bytes_per_sm * sm_count / latency_s / 1e9
    return min(littles, ceiling_gbs)


def measured_ridge(ks, times_ms, n: int) -> tuple[float, float, float]:
    best_tflops = 0.0
    best_gbs = 0.0
    for k, ms in zip(ks, times_ms):
        flops, nbytes = fma_kernel_counts(n, k)
        seconds = ms * 1e-3
        best_tflops = max(best_tflops, flops / seconds / 1e12)
        best_gbs = max(best_gbs, nbytes / seconds / 1e9)
    return best_tflops, best_gbs, ridge_point(best_tflops, best_gbs)
