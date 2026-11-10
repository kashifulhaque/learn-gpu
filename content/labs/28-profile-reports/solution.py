"""Lab 28: Read profiler reports. Worked solution.

Every function here is pure Python and runs without a GPU. The tests feed them
recorded `ncu --csv` and `nsys stats` exports.
"""

import csv
import io
import math
import statistics

from lab_common import A100

# Thresholds the classifier uses. They are the rules from chapter 28.
UNCOALESCED_EFFICIENCY = 0.5  # useful bytes over fetched bytes, below this is uncoalesced
BANK_CONFLICT_FRACTION = 0.25  # conflict wavefronts over all shared wavefronts
TAIL_EFFICIENCY = 0.8  # waves over ceil(waves), below this the last wave is mostly empty
MEMORY_BOUND_DRAM_PCT = 50.0
COMPUTE_BOUND_SM_PCT = 60.0

DURATION = "gpu__time_duration.sum"
DRAM_READ = "dram__bytes_read.sum"
DRAM_WRITE = "dram__bytes_write.sum"
DRAM_PCT = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
SM_PCT = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
STALL_PREFIX = "smsp__pcsamp_warps_issue_stalled_"
LIMITS = ("registers", "shared_mem", "warps", "blocks")


# ---------------------------------------------------------------------------
# Part 1: parse an ncu report
# ---------------------------------------------------------------------------


def _number(text):
    """A float from `1,234.5`, or None for `n/a` and empty text."""
    raw = text.strip().replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def parse_ncu_csv(text: str) -> dict:
    """`{kernel name: {metric name: value}}` from `ncu --csv` text."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('"ID"') or line.startswith("ID,"))
    kernels: dict = {}
    for row in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
        if not row.get("Metric Name"):
            continue  # a banner line such as ==PROF== Disconnected
        value = _number(row["Metric Value"])
        if value is None:
            continue
        kernels.setdefault(row["Kernel Name"], {})[row["Metric Name"]] = value
    return kernels


# ---------------------------------------------------------------------------
# Part 2: the roofline from raw counters
# ---------------------------------------------------------------------------


def flops(m: dict) -> float:
    """FP32 FLOPs: one per FADD and FMUL, two per FFMA."""
    return (
        m.get("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum", 0.0)
        + m.get("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum", 0.0)
        + 2 * m.get("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum", 0.0)
    )


def roofline_from_counters(m: dict, peak_tflops: float = A100.FP32_TFLOPS, bandwidth_gbs: float = A100.HBM_BW_GBS) -> dict:
    seconds = m[DURATION] * 1e-9
    nbytes = m[DRAM_READ] + m[DRAM_WRITE]
    work = flops(m)
    ai = work / nbytes if nbytes else math.inf
    ridge = peak_tflops * 1e12 / (bandwidth_gbs * 1e9)
    ceiling = min(peak_tflops * 1e3, ai * bandwidth_gbs) if nbytes else peak_tflops * 1e3
    achieved = work / seconds / 1e9
    gbs = nbytes / seconds / 1e9
    bound = "memory" if ai < ridge else "compute"
    # Against the sloped roof the share is the bandwidth share, which also works when FLOPs are zero.
    pct = 100.0 * (gbs / bandwidth_gbs if bound == "memory" else achieved / (peak_tflops * 1e3))
    return {
        "flops": work,
        "bytes": nbytes,
        "seconds": seconds,
        "ai": ai,
        "achieved_gflops": achieved,
        "achieved_gbs": gbs,
        "ridge": ridge,
        "ceiling_gflops": ceiling,
        "bound": bound,
        "pct_of_ceiling": pct,
    }


# ---------------------------------------------------------------------------
# Part 3: occupancy, coalescing, bank conflicts, waves, stalls
# ---------------------------------------------------------------------------


def _limit(m: dict, name: str) -> float:
    return m[f"launch__occupancy_limit_{name}"]


def occupancy_limiter(m: dict) -> str:
    """The launch limit that allows the fewest resident blocks per SM."""
    return min(LIMITS, key=lambda name: _limit(m, name))  # min keeps the first of a tie


def theoretical_occupancy(m: dict) -> float:
    """Percent of the 64 warps an SM holds that the launch shape allows."""
    blocks = min(_limit(m, name) for name in LIMITS)
    warps = math.ceil(m["launch__block_size"] / A100.WARP_SIZE)
    return 100.0 * blocks * warps / (A100.MAX_THREADS_PER_SM // A100.WARP_SIZE)


def sectors_per_request(m: dict, op: str) -> float:
    requests = m[f"l1tex__t_requests_pipe_lsu_mem_global_op_{op}.sum"]
    return m[f"l1tex__t_sectors_pipe_lsu_mem_global_op_{op}.sum"] / requests if requests else 0.0


def sector_efficiency(m: dict, op: str) -> float:
    """Useful bytes over fetched bytes for global `ld` or `st`. 1.0 when there is no traffic."""
    sectors = m[f"l1tex__t_sectors_pipe_lsu_mem_global_op_{op}.sum"]
    if not sectors:
        return 1.0
    return m[f"smsp__sass_data_bytes_mem_global_op_{op}.sum"] / (32.0 * sectors)


def shared_conflict_fraction(m: dict) -> float:
    """Wavefronts caused by bank conflicts, as a fraction of all shared wavefronts."""
    wavefronts = (
        m["l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"]
        + m["l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum"]
    )
    conflicts = (
        m["l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"]
        + m["l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"]
    )
    return conflicts / wavefronts if wavefronts else 0.0


def waves(m: dict, sm_count: int = A100.SM_COUNT) -> float:
    blocks_per_sm = min(_limit(m, name) for name in LIMITS)
    return m["launch__grid_size"] / (sm_count * blocks_per_sm)


def wave_efficiency(m: dict, sm_count: int = A100.SM_COUNT) -> float:
    """Waves over the whole waves the kernel occupies: 1.0 when the last wave is full."""
    w = waves(m, sm_count)
    return w / math.ceil(w)


def dominant_stall(m: dict) -> str:
    """The stall reason with the largest share, not counting `selected` or `not_selected`."""
    stalls = {
        name[len(STALL_PREFIX) : -len(".avg")]: value
        for name, value in m.items()
        if name.startswith(STALL_PREFIX) and name.endswith(".avg")
    }
    for healthy in ("selected", "not_selected"):
        stalls.pop(healthy, None)
    return max(stalls, key=stalls.get)


# ---------------------------------------------------------------------------
# Part 4: the diagnosis
# ---------------------------------------------------------------------------


def diagnose(m: dict) -> str:
    """One of: uncoalesced, bank-conflicts, tail-effect, memory-bound,
    compute-bound, latency-bound. The first rule that fires wins."""
    if min(sector_efficiency(m, "ld"), sector_efficiency(m, "st")) < UNCOALESCED_EFFICIENCY:
        return "uncoalesced"
    if shared_conflict_fraction(m) >= BANK_CONFLICT_FRACTION:
        return "bank-conflicts"
    if waves(m) > 1.0 and wave_efficiency(m) < TAIL_EFFICIENCY:
        return "tail-effect"
    if m[DRAM_PCT] >= MEMORY_BOUND_DRAM_PCT:
        return "memory-bound"
    if m[SM_PCT] >= COMPUTE_BOUND_SM_PCT:
        return "compute-bound"
    return "latency-bound"


# ---------------------------------------------------------------------------
# Part 5: read an nsys stats export
# ---------------------------------------------------------------------------


def parse_nsys_csv(text: str) -> list:
    """A list of rows, each a dict from header name to text. Lines before the
    header, such as `Processing [...]`, are skipped."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if "," in line)
    return list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))


def gpu_events(rows: list) -> list:
    """`{start_ns, dur_ns, stream, name, is_kernel}` per row of a cuda_gpu_trace export."""
    return [
        {
            "start_ns": int(row["Start (ns)"]),
            "dur_ns": int(row["Duration (ns)"]),
            "stream": int(row["Strm"]),
            "name": row["Name"],
            "is_kernel": not row["Name"].startswith("[CUDA"),
        }
        for row in rows
    ]


def busy_and_gaps(events: list) -> dict:
    """The union of busy time across all streams, and the idle gaps between."""
    spans = sorted((e["start_ns"], e["start_ns"] + e["dur_ns"]) for e in events)
    merged: list = []
    for begin, end in spans:
        if merged and begin <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([begin, end])
    busy = sum(end - begin for begin, end in merged)
    gaps = [merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1)]
    span = merged[-1][1] - merged[0][0]
    return {"span_ns": span, "busy_ns": busy, "duty": busy / span, "gaps": gaps}


def launch_avg_ns(api_rows: list) -> float:
    """Average time of one kernel launch call, from an cuda_api_sum export."""
    total = calls = 0.0
    for row in api_rows:
        if row["Name"].startswith(("cudaLaunchKernel", "cuLaunchKernel")):
            total += float(row["Total Time (ns)"])
            calls += float(row["Num Calls"])
    return total / calls


def diagnose_timeline(events: list, api_rows: list) -> str:
    """One of: kernel-bound, sync-bound, launch-bound."""
    # Kernels only: a 4-byte copy inside a host pause is part of the pause.
    kernels = [e for e in events if e["is_kernel"]]
    timeline = busy_and_gaps(kernels)
    launch = launch_avg_ns(api_rows)
    kernel_durations = [e["dur_ns"] for e in kernels]
    if timeline["duty"] >= 0.85:
        return "kernel-bound"
    if timeline["gaps"] and statistics.median(timeline["gaps"]) >= 10 * launch:
        return "sync-bound"
    if statistics.median(kernel_durations) < 2 * launch:
        return "launch-bound"
    return "kernel-bound"
