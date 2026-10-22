import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from lab_common import A100, Checks, bandwidth_gbs, cuda_time, require_cuda  # noqa: E402

# Latency of a global-memory load on an A100 PCIe 40GB, 466.3 cycles at 1410 MHz,
# measured by Luo et al. (arXiv 2402.13499), docs/FACTS.md section 3.3. It is an
# idle-memory latency, so Little's law with it gives a lower bound on bytes in flight.
HBM_LATENCY_CYCLES = 466.3
# The SXM4 module's rated bandwidth. A kernel can't beat the roofline built from
# the faster of the two cards, and the lab can land on either.
SXM4_BW_GBS = 2039.0

N = 2**26  # 256 MiB of float32, far larger than the 40 MiB L2


def rel(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * abs(b)


def check_predictors(c: Checks, s) -> None:
    """The pure-arithmetic part. It needs no GPU."""
    c.check(
        "ridge_point: 312 TFLOP/s over 1935 GB/s is 161.2 FLOP/byte",
        lambda: (rel(s.ridge_point(312, 1935), 161.24, 1e-3), f"got {s.ridge_point(312, 1935)}"),
    )
    c.check(
        "ridge_point: 19.5 TFLOP/s over 1275 GB/s is 15.29 FLOP/byte",
        lambda: (rel(s.ridge_point(19.5, 1275), 15.294, 1e-3), f"got {s.ridge_point(19.5, 1275)}"),
    )

    def vector_add():
        n = 2**24
        got = s.roofline_ms(n, 12 * n, 19.5, 1275)
        return rel(got, 0.15790, 2e-3), f"got {got:.5f} ms, expected 0.15790 ms (memory bound)"

    c.check("roofline_ms: a float32 vector add is memory bound", vector_add)

    def compute_bound():
        got = s.roofline_ms(1e12, 1e6, 312, 1935)
        return rel(got, 3.2051, 2e-3), f"got {got:.4f} ms, expected 3.2051 ms (compute bound)"

    c.check("roofline_ms: 1 TFLOP on 1 MB is compute bound", compute_bound)

    def counts():
        flops, nbytes = s.fma_kernel_counts(1000, 10)
        return (flops, nbytes) == (20000, 8000), f"got ({flops}, {nbytes}), expected (20000, 8000)"

    c.check("fma_kernel_counts: 2k FLOP and 8 bytes per element", counts)

    def waves():
        cases = [(108, 1, 1), (109, 1, 2), (216, 1, 2), (216, 2, 1), (217, 2, 2), (1, 1, 1), (0, 1, 0)]
        for blocks, per_sm, want in cases:
            got = s.num_waves(blocks, per_sm, 108)
            if got != want:
                return False, f"num_waves({blocks}, {per_sm}, 108) = {got}, expected {want}"
        return True, "ok"

    c.check("num_waves: a ceiling division by blocks_per_sm * sm_count", waves)

    def efficiency():
        cases = [(108, 1, 1.0), (109, 1, 109 / 216), (217, 2, 217 / 432), (1729, 1, 1729 / 1836), (0, 1, 1.0)]
        for blocks, per_sm, want in cases:
            got = s.wave_efficiency(blocks, per_sm, 108)
            if abs(got - want) > 1e-9:
                return False, f"wave_efficiency({blocks}, {per_sm}, 108) = {got}, expected {want}"
        return True, "ok"

    c.check("wave_efficiency: blocks over waves times capacity", efficiency)

    def gemm():
        # NVIDIA's worked example: M = 2304 with 256 x 128 tiles.
        # N = 1536 gives 9 x 12 = 108 tiles, one full wave; N = 1544 gives 9 x 13 = 117.
        tiles = [(2304, 1536, 108), (2304, 1544, 117), (130, 130, 4), (128, 128, 1), (1, 1, 1)]
        for m, n, want in tiles:
            tm, tn = (256, 128) if m == 2304 else (128, 128)
            got = s.gemm_tiles(m, n, tm, tn)
            if got != want:
                return False, f"gemm_tiles({m}, {n}, {tm}, {tn}) = {got}, expected {want}"
        full = s.gemm_wave_efficiency(2304, 1536, 256, 128, 108)
        spill = s.gemm_wave_efficiency(2304, 1544, 256, 128, 108)
        ok = abs(full - 1.0) < 1e-9 and abs(spill - 117 / 216) < 1e-9
        return ok, f"N=1536: {full:.4f} (expected 1.0), N=1544: {spill:.4f} (expected {117 / 216:.4f})"

    c.check("gemm_tiles and gemm_wave_efficiency match NVIDIA's 108-tile example", gemm)

    def little():
        rated = s.bytes_in_flight(1935, HBM_LATENCY_CYCLES, 1410)
        measured = s.bytes_in_flight(1275, HBM_LATENCY_CYCLES, 1410)
        ok = rel(rated, 639.9e3, 2e-3) and rel(measured, 421.6e3, 2e-3)
        return ok, f"got {rated / 1e3:.1f} KB and {measured / 1e3:.1f} KB, expected 639.9 KB and 421.6 KB"

    c.check("bytes_in_flight: Little's law, bandwidth times latency", little)

    def read_bw():
        low = s.predicted_read_gbs(512, HBM_LATENCY_CYCLES, 1410, 108, 1275)
        high = s.predicted_read_gbs(65536, HBM_LATENCY_CYCLES, 1410, 108, 1275)
        ok = rel(low, 167.2, 3e-3) and abs(high - 1275) < 1e-6
        return ok, f"got {low:.1f} and {high:.1f} GB/s, expected 167.2 and 1275.0"

    c.check("predicted_read_gbs: in flight over latency, capped at the ceiling", read_bw)


def sweep_fma(fx, ks):
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    times = []
    for k in ks:
        times.append(cuda_time(lambda k=k: fx.fma_chain_into(x, y, k), warmup=3, iters=10))
    del x, y
    return times


def run(submission):
    c = Checks()
    needed = (
        "ridge_point", "roofline_ms", "fma_kernel_counts", "num_waves", "wave_efficiency",
        "gemm_tiles", "gemm_wave_efficiency", "bytes_in_flight", "predicted_read_gbs", "measured_ridge",
    )
    if not c.require(submission, *needed):
        return c.finish()

    check_predictors(c, submission)
    if not all(r["passed"] for r in c.results):
        print("      Fix the predictors first: the measurements below use them.", flush=True)
        return c.finish()

    device = require_cuda()
    import fixture_kernels as fx

    props = torch.cuda.get_device_properties(0)
    sms = props.multi_processor_count
    print(f"      device = {props.name}, {sms} SMs", flush=True)
    if sms != A100.SM_COUNT:
        print(f"      note: this is not the course's A100 ({A100.SM_COUNT} SMs); wave counts use {sms}.", flush=True)

    # ---- Kernel correctness, including awkward sizes -------------------------------------
    def correct_fma():
        for n in (1, 7, 1000, 2**20 + 13):
            for k in (0, 1, 5, 33):
                x = torch.randn(n, device=device)
                got = fx.fma_chain(x, k).double()
                want = fx.fma_chain_reference(x, k)
                err = (got - want).abs().max().item()
                if err > 1e-4:
                    return False, f"n={n}, k={k}: max abs error {err:.2e}"
        return True, "n = 1, 7, 1000, and 2**20 + 13 match a float64 reference"

    c.check("the FMA kernel matches a float64 reference", correct_fma)

    def correct_read():
        for n in (1, 7, 1000, 12345):
            x = torch.randint(0, 4, (n,), device=device).float()
            for ilp in (1, 2, 4, 8):
                got = fx.read_sum(x, blocks=3, threads=128, ilp=ilp).double().sum().item()
                if got != x.double().sum().item():
                    return False, f"n={n}, ilp={ilp}: sum {got}, expected {x.double().sum().item()}"
        return True, "exact integer sums for n = 1, 7, 1000, 12345 and ilp = 1, 2, 4, 8"

    c.check("the read kernel sums every element exactly once", correct_read)

    # ---- Part 1: the roofline against an intensity sweep ---------------------------------
    ks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    times = sweep_fma(fx, ks)
    ceiling_floor_bw = max(A100.HBM_BW_GBS, SXM4_BW_GBS)
    print("      k   FLOP/B  predicted ms  roofline floor ms  measured ms  measured/predicted", flush=True)
    beats_roof = []
    for k, ms in zip(ks, times):
        flops, nbytes = submission.fma_kernel_counts(N, k)
        pred = submission.roofline_ms(flops, nbytes, A100.FP32_TFLOPS, A100.COPY_CEILING_GBS)
        floor = submission.roofline_ms(flops, nbytes, A100.FP32_TFLOPS, ceiling_floor_bw)
        print(f"    {k:4d}  {flops / nbytes:6.2f}  {pred:12.4f}  {floor:17.4f}  {ms:11.4f}  {ms / pred:18.2f}", flush=True)
        if ms < 0.9 * floor:
            beats_roof.append((k, ms, floor))

    c.add(
        "no kernel beats the roofline built from the rated peaks",
        not beats_roof,
        "a time below the floor means the bytes, the FLOPs, or the timing are wrong"
        if beats_roof
        else "every measured time is at or above its floor",
    )
    flops1, bytes1 = submission.fma_kernel_counts(N, 1)
    k1_bw = bandwidth_gbs(bytes1, times[0])
    c.metric("k1_bw_gbs", round(k1_bw, 1))
    c.add(
        "at k = 1 the kernel reaches at least half the copy ceiling",
        k1_bw >= 0.5 * A100.COPY_CEILING_GBS,
        f"{k1_bw:.0f} GB/s against {0.5 * A100.COPY_CEILING_GBS:.0f}",
    )
    c.add(
        "time is flat while memory bound: k = 2 costs under 1.3x k = 1",
        times[1] < 1.3 * times[0],
        f"{times[1] / times[0]:.2f}x",
    )
    c.add(
        "time grows once compute bound: k = 512 costs over 3x k = 1",
        times[-1] > 3.0 * times[0],
        f"{times[-1] / times[0]:.2f}x",
    )

    peak_tflops, peak_gbs, ridge = submission.measured_ridge(ks, times, N)
    pred_ridge = submission.ridge_point(A100.FP32_TFLOPS, A100.COPY_CEILING_GBS)
    c.metric("fp32_tflops", round(peak_tflops, 2))
    c.metric("ridge_measured", round(ridge, 2))
    print(
        f"      predicted ridge {pred_ridge:.1f} FLOP/B (19.5 TFLOP/s over the 1275 GB/s copy ceiling); "
        f"measured {ridge:.1f} FLOP/B from {peak_tflops:.2f} TFLOP/s and {peak_gbs:.0f} GB/s",
        flush=True,
    )
    c.add(
        "the FP32 rate reaches at least 40% of the 19.5 TFLOP/s rating",
        peak_tflops >= 0.4 * A100.FP32_TFLOPS,
        f"{peak_tflops:.2f} TFLOP/s",
    )
    c.add(
        "the measured ridge is within a factor of 4 of the prediction",
        pred_ridge / 4 <= ridge <= pred_ridge * 4,
        f"measured {ridge:.1f}, predicted {pred_ridge:.1f}",
    )

    # ---- Part 2: launch floor ------------------------------------------------------------
    one = torch.zeros(1, device=device)
    one_out = torch.empty_like(one)
    floor_us = cuda_time(lambda: fx.fma_chain_into(one, one_out, 0), warmup=50, iters=200) * 1e3
    c.metric("launch_floor_us", round(floor_us, 2))
    c.add(
        "an empty kernel launches in under 500 microseconds",
        0.0 < floor_us < 500.0,
        f"{floor_us:.1f} microseconds from event to event, host launch included",
    )

    # ---- Part 3: wave quantization -------------------------------------------------------
    out = torch.empty(2 * sms * fx.BLOCK_THREADS + 8 * fx.BLOCK_THREADS, device=device)
    iters = 100_000
    probe = {}
    for blocks in (sms // 2, sms, sms + 1, 2 * sms, 2 * sms + 1):
        probe[blocks] = cuda_time(lambda b=blocks: fx.block_work_into(out, b, iters), warmup=3, iters=10)
    base = probe[sms]
    print("      blocks  predicted waves  predicted ratio  measured ms  measured ratio", flush=True)
    bad = []
    for blocks, ms in probe.items():
        waves = submission.num_waves(blocks, 1, sms)
        pred_ratio = waves / submission.num_waves(sms, 1, sms)
        ratio = ms / base
        print(f"    {blocks:6d}  {waves:15d}  {pred_ratio:15.2f}  {ms:11.4f}  {ratio:14.2f}", flush=True)
        if not (0.7 * pred_ratio <= ratio <= 1.3 * pred_ratio):
            bad.append((blocks, round(ratio, 2), pred_ratio))
    c.add(
        "measured time tracks the predicted number of waves within 30%",
        not bad,
        f"off at (blocks, measured ratio, predicted ratio): {bad}" if bad else "all five grid sizes agree",
    )
    step = probe[sms + 1] / probe[sms]
    c.metric("wave_step_ratio", round(step, 2))
    c.add(
        "one block past a full wave costs at least 1.6x",
        step >= 1.6,
        f"{sms + 1} blocks take {step:.2f}x as long as {sms}",
    )

    # ---- Part 4: Little's law ------------------------------------------------------------
    x = torch.ones(N, device=device)
    threads = 128
    configs = [(1, 1), (2, 1), (4, 1), (8, 1), (16, 1), (16, 2), (16, 4), (16, 8)]
    results = []
    print("      blocks/SM  ilp  KB in flight/SM  predicted GB/s  measured GB/s", flush=True)
    for blocks_per_sm, ilp in configs:
        ms = cuda_time(
            lambda b=blocks_per_sm, i=ilp: fx.read_sum(x, b * sms, threads, i), warmup=3, iters=10
        )
        gbs = bandwidth_gbs(4 * N, ms)
        inflight = blocks_per_sm * threads * ilp * 4
        pred = submission.predicted_read_gbs(inflight, HBM_LATENCY_CYCLES, A100.BOOST_MHZ, sms, A100.COPY_CEILING_GBS)
        results.append((inflight, gbs))
        print(f"    {blocks_per_sm:9d}  {ilp:3d}  {inflight / 1000:15.1f}  {pred:14.0f}  {gbs:13.0f}", flush=True)
    peak = max(g for _, g in results)
    c.metric("read_peak_gbs", round(peak, 1))
    saturating = [i for i, g in results if g >= 0.9 * peak]
    c.metric("inflight_kb_per_sm_90pct", round(min(saturating) / 1000, 2))
    needed_kb = submission.bytes_in_flight(A100.COPY_CEILING_GBS, HBM_LATENCY_CYCLES, A100.BOOST_MHZ) / sms / 1000
    print(
        f"      Little's law at the copy ceiling asks for {needed_kb:.1f} KB per SM; "
        f"your sweep reached 90% of its peak at {min(saturating) / 1000:.1f} KB per SM",
        flush=True,
    )
    c.add(
        "the best read configuration reaches at least half the copy ceiling",
        peak >= 0.5 * A100.COPY_CEILING_GBS,
        f"{peak:.0f} GB/s",
    )
    c.add(
        "starved of loads in flight, the kernel is at most half as fast as the best",
        results[0][1] <= 0.5 * peak,
        f"{results[0][1]:.0f} GB/s with {results[0][0]} bytes in flight per SM, against {peak:.0f}",
    )
    return c.finish()
