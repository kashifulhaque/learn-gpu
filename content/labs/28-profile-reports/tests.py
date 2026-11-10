"""Tests for lab 28: read profiler reports.

Pure Python. No randomness. Every check message shows the expected value and
the actual value. The fixtures `fixture_ncu.py` and `fixture_nsys.py` are
recorded reports that the reader sees. The held-out kernels and traces that the
classifiers also face are built in this file and are not shown to the reader.

Wrong implementations these checks catch:

- split_on_comma: parse_ncu_csv splits each line on `,`. The rows are quoted,
  so the cells shift, and a value such as "805,306,368" splits in two.
- ffma_counts_one: flops counts an FFMA as one FLOP. sgemm_tiled then reports
  half its work, 68.7 GFLOP instead of 137.5, and half its arithmetic intensity.
- occupancy_from_registers: theoretical_occupancy uses the register limit only.
  It reports 12.5 percent for reduce_tail, which is limited by shared memory
  to one block, and it ignores that the block holds 8 warps.
- sectors_over_eight: diagnose calls a kernel uncoalesced when it fetches more
  than 8 sectors per request. A coalesced float4 copy fetches 16 sectors per
  request, so the held-out copy_float4 kernel comes back uncoalesced.
- any_conflict: diagnose calls any nonzero bank-conflict count a conflict.
  sgemm_tiled has 1 percent conflicted wavefronts and is compute-bound.
- speed_of_light_first: diagnose reads only the two Speed of Light percentages.
  transpose_naive_smem, 30 percent DRAM and 21 percent SM, comes back
  latency-bound, and copy_strided comes back memory-bound, though it has an
  8x read amplification.
- fractional_waves: diagnose calls any kernel with a fractional wave count a
  tail effect. vec_add runs 303.4 waves, so its tail costs 0.2 percent.
- sum_of_durations: busy_and_gaps adds the kernel durations instead of taking
  the union. On two overlapped streams the busy time is 1.89 times the span.
- mean_gap: diagnose_timeline averages the gaps. One 20 ms gap at the end of a
  trace of tiny kernels makes the mean look like a host stall.
"""

from __future__ import annotations

import io
import csv
import shutil
import subprocess
import sys

from lab_common import A100, Checks, ncu_profile_blocked

import fixture_ncu
import fixture_nsys

DURATION = "gpu__time_duration.sum"
HEADER = [
    "ID", "Process ID", "Process Name", "Host Name", "Kernel Name", "Context", "Stream",
    "Block Size", "Grid Size", "Device", "CC", "Section Name", "Metric Name", "Metric Unit",
    "Metric Value",
]

STALLS = ("long_scoreboard", "short_scoreboard", "barrier", "mio_throttle",
          "math_pipe_throttle", "lg_throttle", "not_selected", "selected", "wait")


def invoke(fn, *args, **kwargs):
    """Call `fn` and return `(value, None)`, or `(None, exception)`."""
    try:
        return fn(*args, **kwargs), None
    except Exception as exc:  # noqa: BLE001
        return None, exc


def _about(actual, expected, tol):
    return abs(actual - expected) <= tol


def _fail_exc(label, exc):
    return False, f"{label}: expected a value, actual {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Held-out kernels, built from a neutral base and rendered as ncu CSV text
# ---------------------------------------------------------------------------


def _base() -> dict:
    """A neutral kernel: coalesced fp32 traffic, 100 whole waves, nothing saturated."""
    m = {
        DURATION: 1_000_000.0,
        "dram__bytes_read.sum": 10_000_000.0,
        "dram__bytes_write.sum": 10_000_000.0,
        "dram__throughput.avg.pct_of_peak_sustained_elapsed": 1.0,
        "sm__throughput.avg.pct_of_peak_sustained_elapsed": 5.0,
        "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum": 0.0,
        "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum": 0.0,
        "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": 0.0,
        "sm__cycles_elapsed.avg": 1_410_000.0,
        "sm__cycles_active.avg": 1_400_000.0,
        "launch__grid_size": 86_400.0,
        "launch__block_size": 256.0,
        "launch__registers_per_thread": 32.0,
        "launch__occupancy_limit_registers": 8.0,
        "launch__occupancy_limit_shared_mem": 32.0,
        "launch__occupancy_limit_warps": 8.0,
        "launch__occupancy_limit_blocks": 32.0,
        "launch__waves_per_multiprocessor": 100.0,
        "sm__maximum_warps_per_active_cycle_pct": 100.0,
        "sm__warps_active.avg.pct_of_peak_sustained_active": 80.0,
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum": 0.0,
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum": 0.0,
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": 0.0,
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum": 0.0,
    }
    for op in ("ld", "st"):
        m[f"l1tex__t_requests_pipe_lsu_mem_global_op_{op}.sum"] = 1_000_000.0
        m[f"l1tex__t_sectors_pipe_lsu_mem_global_op_{op}.sum"] = 4_000_000.0
        m[f"smsp__sass_data_bytes_mem_global_op_{op}.sum"] = 128_000_000.0
    shares = dict.fromkeys(STALLS, 5.0)
    shares["long_scoreboard"] = 55.0
    for name, value in shares.items():
        m[f"smsp__pcsamp_warps_issue_stalled_{name}.avg"] = value
    return m


def _csv(kernels: dict) -> str:
    """`ncu --csv` text for `{kernel name: {metric: value}}`, quoted, with banner lines."""
    out = io.StringIO()
    writer = csv.writer(out, quoting=csv.QUOTE_ALL, lineterminator="\n")
    out.write("==PROF== Connected to process 99 (/usr/bin/python3)\n")
    writer.writerow(HEADER)
    for index, (name, metrics) in enumerate(kernels.items()):
        for metric, value in metrics.items():
            text = f"{value:,.0f}" if float(value).is_integer() else f"{value:.3f}"
            writer.writerow([index, 99, "python3", "host", name, 1, 7, "(256, 1, 1)", "(1, 1, 1)",
                             "A100", "8.0", "", metric, "", text])
    out.write("==PROF== Disconnected from process 99\n")
    return out.getvalue()


def _held_out() -> dict:
    """name -> (metrics, expected label)."""
    def mk(over: dict) -> dict:
        m = _base()
        m.update(over)
        return m

    dram = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
    sm = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    shared_ld = "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"
    shared_ld_conf = "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"
    sectors_ld = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
    sectors_st = "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"
    return {
        # A coalesced float4 copy: 512 bytes per request is 16 sectors, and every byte is used.
        "copy_float4": (mk({dram: 66.0, sm: 7.0, sectors_ld: 16_000_000.0, sectors_st: 16_000_000.0,
                            "smsp__sass_data_bytes_mem_global_op_ld.sum": 512_000_000.0,
                            "smsp__sass_data_bytes_mem_global_op_st.sum": 512_000_000.0}), "memory-bound"),
        # 1000.5 waves: the last wave is half full, which wastes 0.05 percent of the time.
        "triad_big_grid": (mk({dram: 64.0, sm: 11.0, "launch__grid_size": 1000.5 * 864.0,
                               "launch__waves_per_multiprocessor": 1000.5}), "memory-bound"),
        # A padded transpose with a trace of conflicts, 0.4 percent of its wavefronts.
        "transpose_padded": (mk({dram: 63.0, sm: 18.0, shared_ld: 2_100_000.0, shared_ld_conf: 8_400.0,
                                 "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum": 2_100_000.0}),
                             "memory-bound"),
        # LDS.128 needs 4 wavefronts per request. None of them is a conflict.
        "gemm_lds128": (mk({dram: 6.0, sm: 78.0, shared_ld: 8_400_000.0,
                            "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum": 2_100_000.0,
                            "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": 4.0e10}), "compute-bound"),
        # Fewer blocks than the machine holds: 54 blocks on 108 SMs is half a wave, not a tail.
        "small_grid": (mk({dram: 2.0, sm: 4.0, "launch__grid_size": 54.0,
                           "launch__waves_per_multiprocessor": 0.0625}), "latency-bound"),
        # 2.04 waves on 108 SMs at 8 blocks per SM: the third wave is 4 percent full.
        "tail_two_waves": (mk({dram: 20.0, sm: 35.0, "launch__grid_size": 2.04 * 864.0,
                               "launch__waves_per_multiprocessor": 2.04}), "tail-effect"),
        # A two-way conflict on half of the shared wavefronts: 50 percent conflicted.
        "two_way_conflicts": (mk({dram: 35.0, sm: 22.0, shared_ld: 4_000_000.0, shared_ld_conf: 2_000_000.0}),
                              "bank-conflicts"),
        # Reads are fine. Writes touch 32 sectors per request for 128 useful bytes.
        "scatter_store": (mk({dram: 55.0, sm: 6.0, sectors_st: 32_000_000.0}), "uncoalesced"),
        # A tensor-core-like kernel: SM busy, DRAM nearly idle.
        "mma_loop": (mk({dram: 8.0, sm: 91.0, "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": 1.0e10}),
                     "compute-bound"),
        # No global traffic at all: zero requests and zero sectors must not divide by zero.
        "register_loop": (mk({dram: 0.0, sm: 74.0,
                              "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum": 0.0,
                              sectors_ld: 0.0, sectors_st: 0.0,
                              "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum": 0.0,
                              "smsp__sass_data_bytes_mem_global_op_ld.sum": 0.0,
                              "smsp__sass_data_bytes_mem_global_op_st.sum": 0.0,
                              "dram__bytes_read.sum": 0.0, "dram__bytes_write.sum": 0.0}), "compute-bound"),
    }


def _events(rows: list) -> str:
    """cuda_gpu_trace CSV text for (start, duration, stream, name) rows."""
    lines = ["Processing [held.sqlite] with [cuda_gpu_trace.py]... ", "",
             "Start (ns),Duration (ns),CorrId,Device,Strm,Name"]
    for i, (start, dur, stream, name) in enumerate(rows):
        lines.append(f'{start},{dur},{i},"GPU (0)",{stream},"{name}"')
    return "\n".join(lines)


def _api(launch_avg: float, calls: int) -> str:
    return ("Processing [held.sqlite] with [cuda_api_sum.py]... \n\n"
            "Time (%),Total Time (ns),Num Calls,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name\n"
            f"100.0,{launch_avg * calls:.0f},{calls},{launch_avg:.1f},{launch_avg:.1f},1,2,3,cudaLaunchKernel\n")


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------


def _check_parse(submission):
    kernels, exc = invoke(submission.parse_ncu_csv, fixture_ncu.NCU_REPORT)
    if exc:
        return _fail_exc("parse_ncu_csv", exc)
    names = list(kernels)
    if names != fixture_ncu.KERNELS:
        return False, f"expected kernels {fixture_ncu.KERNELS}, actual {names}"
    vec = kernels["vec_add"]
    want = {DURATION: 629146.0, "dram__bytes_read.sum": 536870912.0, "launch__grid_size": 262144.0}
    for key, value in want.items():
        if vec.get(key) != value:
            return False, f"vec_add {key}: expected {value}, actual {vec.get(key)}"
    # A value that is not a number is dropped, not turned into 0.
    text = _csv({"k": {DURATION: 5.0}}).replace('"5"', '"n/a"')
    got, exc = invoke(submission.parse_ncu_csv, text)
    if exc or got.get("k", {}).get(DURATION) is not None:
        return False, f"expected an n/a value to be left out, actual {got if not exc else exc}"
    return True, f"expected 6 kernels, actual {names}; vec_add has {len(vec)} counters"


def _check_roofline(c, submission, report):
    vec, sgemm, copy = report["vec_add"], report["sgemm_tiled"], report["copy_strided"]
    flops_vec, e1 = invoke(submission.flops, vec)
    flops_gemm, e2 = invoke(submission.flops, sgemm)
    if e1 or e2:
        return _fail_exc("flops", e1 or e2)
    want_gemm = 2 * 4096**3 + 4096**2  # FFMA counts twice, plus the alpha multiply
    if flops_vec != 2**26 or flops_gemm != want_gemm:  # ffma_counts_one fails here
        return False, f"expected flops vec_add {2**26} and sgemm_tiled {want_gemm}, actual {flops_vec} and {flops_gemm}"
    rv, e1 = invoke(submission.roofline_from_counters, vec)
    rg, e2 = invoke(submission.roofline_from_counters, sgemm)
    rc, e3 = invoke(submission.roofline_from_counters, copy)
    if e1 or e2 or e3:
        return _fail_exc("roofline_from_counters", e1 or e2 or e3)
    ridge = A100.FP32_TFLOPS * 1e12 / (A100.HBM_BW_GBS * 1e9)  # 10.08 FLOP per byte
    checks = [
        ("vec_add arithmetic intensity", rv["ai"], 1 / 12, 0.002),
        ("vec_add achieved GB/s", rv["achieved_gbs"], 1280.0, 1.0),
        ("vec_add percent of its roof", rv["pct_of_ceiling"], 66.15, 0.2),
        ("fp32 ridge point", rv["ridge"], ridge, 0.01),
        ("sgemm_tiled arithmetic intensity", rg["ai"], 682.75, 0.1),
        ("sgemm_tiled achieved GFLOP/s", rg["achieved_gflops"], 16200.0, 20.0),
        ("sgemm_tiled percent of its roof", rg["pct_of_ceiling"], 83.08, 0.2),
    ]
    for label, actual, expected, tol in checks:
        if not _about(actual, expected, tol):
            return False, f"{label}: expected {expected:.4g}, actual {actual:.4g}"
    if (rv["bound"], rg["bound"], rc["bound"]) != ("memory", "compute", "memory"):
        return False, (f"expected bounds memory, compute, memory for vec_add, sgemm_tiled, copy_strided; "
                       f"actual {rv['bound']}, {rg['bound']}, {rc['bound']}")
    tensor, exc = invoke(submission.roofline_from_counters, vec, peak_tflops=312.0)
    if exc or not _about(tensor["ridge"], 312e12 / 1935e9, 0.01):
        return False, f"expected ridge 161.2 with peak_tflops=312, actual {tensor if not exc else exc}"
    c.metric("sgemm_achieved_gflops", round(rg["achieved_gflops"], 1))
    c.metric("sgemm_pct_of_roof", round(rg["pct_of_ceiling"], 1))
    return True, f"vec_add {rv['achieved_gbs']:.0f} GB/s at AI {rv['ai']:.3f}; sgemm_tiled {rg['achieved_gflops']:.0f} GFLOP/s at AI {rg['ai']:.1f}"


def _check_occupancy(submission, report):
    want = {"vec_add": (100.0, "warps"), "sgemm_tiled": (25.0, "registers"),
            "reduce_tail": (12.5, "shared_mem"), "scan_lookback": (18.75, "registers")}
    for name, (occupancy, limiter) in want.items():
        got, e1 = invoke(submission.theoretical_occupancy, report[name])
        lim, e2 = invoke(submission.occupancy_limiter, report[name])
        if e1 or e2:
            return _fail_exc(name, e1 or e2)
        if not _about(got, occupancy, 0.01) or lim != limiter:
            return False, f"{name}: expected {occupancy} percent limited by {limiter}, actual {got} percent limited by {lim}"
    return True, "expected and actual match for 4 kernels"


def _check_counters(submission, report):
    cs = submission.sectors_per_request
    eff = submission.sector_efficiency
    frac = submission.shared_conflict_fraction
    pairs = [
        ("vec_add ld sectors per request", cs(report["vec_add"], "ld"), 4.0, 1e-6),
        ("copy_strided ld sectors per request", cs(report["copy_strided"], "ld"), 32.0, 1e-6),
        ("copy_strided st sectors per request", cs(report["copy_strided"], "st"), 4.0, 1e-6),
        ("copy_strided ld efficiency", eff(report["copy_strided"], "ld"), 0.125, 1e-6),
        ("vec_add ld efficiency", eff(report["vec_add"], "ld"), 1.0, 1e-6),
        ("transpose_naive_smem conflict fraction", frac(report["transpose_naive_smem"]), 0.9393, 0.001),
        ("sgemm_tiled conflict fraction", frac(report["sgemm_tiled"]), 0.0089, 0.001),
        ("vec_add conflict fraction (no shared memory)", frac(report["vec_add"]), 0.0, 1e-9),
    ]
    held = _held_out()["copy_float4"][0]
    parsed, exc = invoke(submission.parse_ncu_csv, _csv({"copy_float4": held}))
    if exc:
        return _fail_exc("parse_ncu_csv on a held-out report", exc)
    pairs.append(("float4 copy ld sectors per request", cs(parsed["copy_float4"], "ld"), 16.0, 1e-6))
    pairs.append(("float4 copy ld efficiency", eff(parsed["copy_float4"], "ld"), 1.0, 1e-6))
    for label, actual, expected, tol in pairs:
        if not _about(actual, expected, tol):
            return False, f"{label}: expected {expected}, actual {actual:.4f}"
    return True, f"expected and actual match for {len(pairs)} counters"


def _check_waves_stalls(submission, report):
    pairs = [
        ("reduce_tail wave efficiency", submission.wave_efficiency(report["reduce_tail"]), 1.111 / 2, 0.002),
        ("sgemm_tiled wave efficiency", submission.wave_efficiency(report["sgemm_tiled"]), 4.741 / 5, 0.002),
        ("vec_add wave efficiency", submission.wave_efficiency(report["vec_add"]), 303.407 / 304, 0.0005),
    ]
    for label, actual, expected, tol in pairs:
        if not _about(actual, expected, tol):
            return False, f"{label}: expected {expected:.4f}, actual {actual:.4f}"
    want = {"vec_add": "long_scoreboard", "sgemm_tiled": "math_pipe_throttle",
            "transpose_naive_smem": "mio_throttle", "scan_lookback": "long_scoreboard",
            "reduce_tail": "math_pipe_throttle"}
    for name, stall in want.items():
        got = submission.dominant_stall(report[name])
        if got != stall:
            return False, f"{name} dominant stall: expected {stall}, actual {got}"
    # A kernel whose stalls are mostly `not_selected` is waiting on nothing.
    m = _base()
    m["smsp__pcsamp_warps_issue_stalled_not_selected.avg"] = 70.0
    got = submission.dominant_stall(m)
    if got != "long_scoreboard":
        return False, f"expected long_scoreboard when not_selected is largest, actual {got}"
    return True, "expected and actual match for 3 wave counts and 6 stall tables"


EXPECTED = {
    "vec_add": "memory-bound",
    "sgemm_tiled": "compute-bound",
    "copy_strided": "uncoalesced",
    "transpose_naive_smem": "bank-conflicts",
    "reduce_tail": "tail-effect",
    "scan_lookback": "latency-bound",
}


def _check_diagnose_visible(c, submission, report):
    wrong = []
    for name, label in EXPECTED.items():
        got, exc = invoke(submission.diagnose, report[name])
        if got != label:
            wrong.append(f"{name}: expected {label}, actual {got if not exc else exc!r}")
    if wrong:
        return False, "; ".join(wrong)
    c.metric("kernels_diagnosed", len(EXPECTED))
    return True, f"expected and actual match for {len(EXPECTED)} kernels"


def _check_diagnose_held_out(submission):
    cases = _held_out()
    parsed, exc = invoke(submission.parse_ncu_csv, _csv({name: m for name, (m, _) in cases.items()}))
    if exc:
        return _fail_exc("parse_ncu_csv on held-out reports", exc)
    wrong = []
    for name, (_, label) in cases.items():
        got, exc = invoke(submission.diagnose, parsed[name])
        if got != label:
            wrong.append(f"{name}: expected {label}, actual {got if not exc else repr(exc)}")
    if wrong:
        return False, f"{len(wrong)} of {len(cases)} held-out kernels differ. " + "; ".join(wrong)
    return True, f"expected and actual match for {len(cases)} held-out kernels"


def _check_nsys(c, submission):
    rows, exc = invoke(submission.parse_nsys_csv, fixture_nsys.LAUNCH_BOUND_TRACE)
    if exc:
        return _fail_exc("parse_nsys_csv", exc)
    events, exc = invoke(submission.gpu_events, rows)
    if exc:
        return _fail_exc("gpu_events", exc)
    if len(events) != 61:
        return False, f"expected 61 events in LAUNCH_BOUND_TRACE (60 kernels and 1 copy), actual {len(events)}"
    kernels = [e for e in events if e["is_kernel"]]
    if len(kernels) != 60 or "AddFunctor<float>" not in kernels[0]["name"] or "," not in kernels[0]["name"]:
        return False, f"expected 60 kernels whose quoted names keep their commas, actual {len(kernels)} with name {kernels[0]['name'] if kernels else None!r}"
    over_rows = submission.parse_nsys_csv(fixture_nsys.OVERLAP_TRACE)
    over = submission.busy_and_gaps(submission.gpu_events(over_rows))
    total = sum(e["dur_ns"] for e in submission.gpu_events(over_rows))
    if not (over["busy_ns"] <= over["span_ns"] and 0.98 <= over["duty"] <= 1.0):  # sum_of_durations fails here
        return False, (f"expected the overlapped streams to be busy 98 to 100 percent of a span of {over['span_ns']} ns, "
                       f"actual busy {over['busy_ns']} ns, duty {over['duty']:.3f} (the durations add to {total} ns)")
    launch = submission.busy_and_gaps(events)
    if len(launch["gaps"]) != 60 or not _about(launch["duty"], 0.424, 0.01):
        return False, f"expected 60 gaps and duty 0.424 for LAUNCH_BOUND_TRACE, actual {len(launch['gaps'])} gaps and duty {launch['duty']:.3f}"
    avg, exc = invoke(submission.launch_avg_ns, submission.parse_nsys_csv(fixture_nsys.LAUNCH_BOUND_API))
    if exc or not _about(avg, 5120.0, 1.0):
        return False, f"expected the cudaLaunchKernel average 5120 ns, actual {avg if not exc else exc!r}"
    c.metric("overlap_duty", round(over["duty"], 3))
    c.metric("launch_bound_duty", round(launch["duty"], 3))
    return True, f"expected and actual match: overlap duty {over['duty']:.3f}, launch-bound duty {launch['duty']:.3f}"


def _check_timeline(submission):
    def label(trace, api):
        events = submission.gpu_events(submission.parse_nsys_csv(trace))
        return submission.diagnose_timeline(events, submission.parse_nsys_csv(api))

    cases = [
        ("LAUNCH_BOUND", fixture_nsys.LAUNCH_BOUND_TRACE, fixture_nsys.LAUNCH_BOUND_API, "launch-bound"),
        ("SYNC_BOUND", fixture_nsys.SYNC_BOUND_TRACE, fixture_nsys.SYNC_BOUND_API, "sync-bound"),
        ("OVERLAP", fixture_nsys.OVERLAP_TRACE, fixture_nsys.OVERLAP_API, "kernel-bound"),
    ]
    # Held out. 12 us kernels with 4 us gaps: the launch is hidden, the kernels are the work.
    rows = [(i * 16_000, 12_000, 7, "k") for i in range(40)]
    cases.append(("12 us kernels", _events(rows), _api(5000.0, 40), "kernel-bound"))
    # Held out. 3 us kernels 7.5 us apart, then one 20 ms gap.
    rows = [(i * 7_500, 3_000, 7, "k") for i in range(100)] + [(100 * 7_500 + 20_000_000, 3_000, 7, "k")]
    cases.append(("tiny kernels, one 20 ms gap", _events(rows), _api(5000.0, 101), "launch-bound"))
    # Held out. 400 us kernels with 600 us pauses between them.
    rows = [(i * 1_000_000, 400_000, 7, "k") for i in range(10)]
    cases.append(("host pauses", _events(rows), _api(5000.0, 10), "sync-bound"))
    wrong = []
    for name, trace, api, want in cases:
        got, exc = invoke(label, trace, api)
        if got != want:
            wrong.append(f"{name}: expected {want}, actual {got if not exc else repr(exc)}")
    if wrong:
        return False, "; ".join(wrong)
    return True, f"expected and actual match for {len(cases)} timelines"


def _live_ncu(c, submission):
    """Profile one PyTorch elementwise kernel with ncu. Skips when it cannot run."""
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA device")
    except Exception as exc:  # noqa: BLE001
        return _skip(c, f"no GPU ({exc})")
    if shutil.which("ncu") is None:
        return _skip(c, "ncu is not installed")
    needed = ["dram__bytes_read.sum", "dram__bytes_write.sum", DURATION,
              "dram__throughput.avg.pct_of_peak_sustained_elapsed",
              "sm__throughput.avg.pct_of_peak_sustained_elapsed",
              "launch__grid_size", "launch__block_size", "launch__occupancy_limit_registers",
              "launch__occupancy_limit_shared_mem", "launch__occupancy_limit_warps",
              "launch__occupancy_limit_blocks",
              "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
              "smsp__sass_data_bytes_mem_global_op_ld.sum",
              "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum", "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
              "smsp__sass_data_bytes_mem_global_op_st.sum",
              "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum", "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
              "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"]
    program = ("import torch; x = torch.empty(2**26, device='cuda'); y = x + 1; torch.cuda.synchronize()")
    cmd = ["ncu", "--csv", "--metrics", ",".join(needed), "-k", "regex:elementwise", "-c", "1",
           sys.executable, "-c", program]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    except Exception as exc:  # noqa: BLE001
        return _skip(c, f"ncu did not finish ({type(exc).__name__})")
    output = done.stdout + done.stderr
    if done.returncode != 0:
        if ncu_profile_blocked(done.returncode, output):
            return _skip(c, "ncu cannot read GPU performance counters here (ERR_NVGPUCTRPERM or similar)")
        return _skip(c, f"ncu exited with status {done.returncode}: {output.strip().splitlines()[-1] if output.strip() else ''}")
    kernels = submission.parse_ncu_csv(done.stdout)
    if not kernels:
        return _skip(c, "ncu wrote no rows that parse_ncu_csv could read")
    name, metrics = next(iter(kernels.items()))
    missing = [m for m in needed if m not in metrics]
    if missing:
        return _skip(c, f"the live report lacks {missing[0]} and {len(missing) - 1} more")
    label = submission.diagnose(metrics)
    c.metric("live_ncu", f"{name[:40]}: {label}")
    return c.add("live ncu: a PyTorch add is memory-bound", label == "memory-bound",
                 f"expected memory-bound, actual {label}")


def _skip(c, reason):
    print(f"SKIP live ncu check: {reason}", flush=True)
    c.metric("live_ncu", f"skipped ({reason})")
    return None


def run(submission):
    c = Checks()
    names = (
        "parse_ncu_csv", "flops", "roofline_from_counters", "theoretical_occupancy", "occupancy_limiter",
        "sectors_per_request", "sector_efficiency", "shared_conflict_fraction", "wave_efficiency",
        "dominant_stall", "diagnose", "parse_nsys_csv", "gpu_events", "busy_and_gaps", "launch_avg_ns",
        "diagnose_timeline",
    )
    if not c.require(submission, *names):
        return c.finish()

    c.check("parse_ncu_csv reads the six recorded kernels", lambda: _check_parse(submission))
    report, exc = invoke(submission.parse_ncu_csv, fixture_ncu.NCU_REPORT)
    if exc or not isinstance(report, dict) or set(report) != set(fixture_ncu.KERNELS):
        c.add("the remaining ncu checks need parse_ncu_csv to work", False,
              "expected a dict with the six recorded kernels; fix parse_ncu_csv first")
        report = None
    if report is not None:
        c.check("flops and roofline_from_counters place a kernel on the roofline",
                lambda: _check_roofline(c, submission, report))
        c.check("theoretical_occupancy and occupancy_limiter", lambda: _check_occupancy(submission, report))
        c.check("sector and bank-conflict counters", lambda: _check_counters(submission, report))
        c.check("wave_efficiency and dominant_stall", lambda: _check_waves_stalls(submission, report))
        c.check("diagnose names the six recorded kernels", lambda: _check_diagnose_visible(c, submission, report))
        c.check("diagnose holds on held-out kernels", lambda: _check_diagnose_held_out(submission))
    c.check("nsys: parse the trace, then find busy time and gaps", lambda: _check_nsys(c, submission))
    c.check("diagnose_timeline names launch-, sync-, and kernel-bound traces", lambda: _check_timeline(submission))
    _live_ncu(c, submission)
    return c.finish()
