"""Lab 28: Read profiler reports.

`fixture_ncu.py` and `fixture_nsys.py`, next to this file, hold recorded
reports: an `ncu --csv` export of six kernels and three `nsys stats` exports.
Do not edit them. They are synthesized to agree with one another on an A100
80GB PCIe, and chapter 28 says how. You write the code that reads them, with no
GPU. The tests also run your functions on kernels and traces that you don't
see, so write rules, not a lookup of the six names.

Work in this order:

1. `parse_ncu_csv`: the report as `{kernel: {metric: value}}`.
2. `flops` and `roofline_from_counters`: place a kernel on the roofline.
3. `theoretical_occupancy` and `occupancy_limiter`: what the launch shape allows.
4. `sectors_per_request`, `sector_efficiency`, `shared_conflict_fraction`.
5. `wave_efficiency` and `dominant_stall`.
6. `diagnose`: one label per kernel.
7. `parse_nsys_csv`, `gpu_events`, `busy_and_gaps`, `launch_avg_ns`.
8. `diagnose_timeline`: one label per trace.

A `m` argument is one kernel's dict from `parse_ncu_csv`: a metric name to a
float. The metric names are constants below.
"""

import csv  # noqa: F401
import io  # noqa: F401
import math  # noqa: F401
import statistics  # noqa: F401

from lab_common import A100

# The thresholds `diagnose` and `diagnose_timeline` use. Chapter 28 derives them.
UNCOALESCED_EFFICIENCY = 0.5  # useful bytes over fetched bytes, below this is uncoalesced
BANK_CONFLICT_FRACTION = 0.25  # conflict wavefronts over all shared wavefronts, at or above this
TAIL_EFFICIENCY = 0.8  # waves over ceil(waves), below this the last wave is mostly empty
MEMORY_BOUND_DRAM_PCT = 50.0  # DRAM throughput, percent of the 1935 GB/s peak, at or above this
COMPUTE_BOUND_SM_PCT = 60.0  # SM throughput, percent of peak, at or above this

DURATION = "gpu__time_duration.sum"  # nanoseconds
DRAM_READ = "dram__bytes_read.sum"  # bytes
DRAM_WRITE = "dram__bytes_write.sum"  # bytes
DRAM_PCT = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
SM_PCT = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
STALL_PREFIX = "smsp__pcsamp_warps_issue_stalled_"  # then a reason, then ".avg"
LIMITS = ("registers", "shared_mem", "warps", "blocks")  # launch__occupancy_limit_<name>, in blocks per SM


# ---------------------------------------------------------------------------
# Part 1: parse an ncu report
# ---------------------------------------------------------------------------


def parse_ncu_csv(text: str) -> dict:
    """`{kernel name: {metric name: value}}` from `ncu --csv` text.

    The text starts with `==PROF==` banner lines and ends with one. The header
    row starts with "ID". Every cell is quoted, and a large value carries
    thousands separators: "805,306,368". Take the kernel from the `Kernel Name`
    column, the metric from `Metric Name`, and the number from `Metric Value`.
    A value that is not a number, such as `n/a`, is left out.

    Use the `csv` module: splitting a line on commas breaks on the quoted cells.
    """
    raise NotImplementedError("TODO: parse the report")


# ---------------------------------------------------------------------------
# Part 2: the roofline from raw counters
# ---------------------------------------------------------------------------


def flops(m: dict) -> float:
    """FP32 FLOPs from the instruction counters.

    `smsp__sass_thread_inst_executed_op_fadd_pred_on.sum` and the `fmul` one
    count thread-level instructions, one FLOP each. The `ffma` one counts
    fused multiply-adds. Count what an FFMA does.
    """
    raise NotImplementedError("TODO: add up the FLOPs")


def roofline_from_counters(m: dict, peak_tflops: float = A100.FP32_TFLOPS, bandwidth_gbs: float = A100.HBM_BW_GBS) -> dict:
    """Place one kernel on the roofline. Return a dict with these keys:

    - `flops`: from `flops(m)`.
    - `bytes`: DRAM bytes read plus written.
    - `seconds`: the duration, from nanoseconds.
    - `ai`: arithmetic intensity, FLOP per DRAM byte (infinity when no bytes moved).
    - `achieved_gflops` and `achieved_gbs`: FLOPs and bytes over the duration, in 10^9 per second.
    - `ridge`: the roofline's ridge point, FLOP per byte, from the two peaks.
    - `ceiling_gflops`: the roof at this intensity, min(peak, ai x bandwidth).
    - `bound`: "memory" when `ai` is below the ridge, otherwise "compute".
    - `pct_of_ceiling`: percent of the relevant roof. For a memory-bound kernel that
      is achieved GB/s over the bandwidth, which still works when the FLOP count is zero.
      For a compute-bound kernel it is achieved GFLOP/s over the peak.
    """
    raise NotImplementedError("TODO: compute the roofline placement")


# ---------------------------------------------------------------------------
# Part 3: occupancy, coalescing, bank conflicts, waves, stalls
# ---------------------------------------------------------------------------


def occupancy_limiter(m: dict) -> str:
    """The name in `LIMITS` whose `launch__occupancy_limit_<name>` is smallest.
    On a tie, the first name in `LIMITS` wins."""
    raise NotImplementedError("TODO: find the limiter")


def theoretical_occupancy(m: dict) -> float:
    """Percent of an SM's 64 warps that the launch shape allows.

    The smallest of the four limits is the blocks per SM. Each block has
    ceil(`launch__block_size` / 32) warps. A100 holds 2048 threads, 64 warps, per SM.
    """
    raise NotImplementedError("TODO: compute theoretical occupancy")


def sectors_per_request(m: dict, op: str) -> float:
    """`l1tex__t_sectors_pipe_lsu_mem_global_op_<op>.sum` over
    `l1tex__t_requests_pipe_lsu_mem_global_op_<op>.sum`, for `op` of "ld" or "st".
    0.0 when there are no requests."""
    raise NotImplementedError("TODO: sectors per request")


def sector_efficiency(m: dict, op: str) -> float:
    """Useful bytes over fetched bytes for global `ld` or `st`.

    Useful bytes are `smsp__sass_data_bytes_mem_global_op_<op>.sum`. Fetched
    bytes are 32 times the sectors. Return 1.0 when there are no sectors.

    The ratio of sectors to requests is not enough: a coalesced float4 load
    fetches 16 sectors per request and wastes nothing.
    """
    raise NotImplementedError("TODO: sector efficiency")


def shared_conflict_fraction(m: dict) -> float:
    """Shared-memory wavefronts that bank conflicts caused, over all shared wavefronts.

    Add the `ld` and `st` counters: `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_<op>.sum`
    over `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_<op>.sum`. 0.0 when there are none.
    """
    raise NotImplementedError("TODO: conflict fraction")


def wave_efficiency(m: dict, sm_count: int = A100.SM_COUNT) -> float:
    """How full the last wave is, as waves over the whole waves the kernel occupies.

    waves = `launch__grid_size` / (sm_count x blocks per SM), where blocks per
    SM is the smallest occupancy limit. Return waves / ceil(waves).
    """
    raise NotImplementedError("TODO: wave efficiency")


def dominant_stall(m: dict) -> str:
    """The stall reason with the largest share, as the bare reason, such as
    "long_scoreboard". Look at every `smsp__pcsamp_warps_issue_stalled_<reason>.avg`.
    Leave out `selected` and `not_selected`: those warps are issuing, not waiting."""
    raise NotImplementedError("TODO: find the dominant stall")


# ---------------------------------------------------------------------------
# Part 4: the diagnosis
# ---------------------------------------------------------------------------


def diagnose(m: dict) -> str:
    """One of: "uncoalesced", "bank-conflicts", "tail-effect", "memory-bound",
    "compute-bound", "latency-bound". Test the rules in this order and return at
    the first that fires:

    1. uncoalesced: the smaller of the `ld` and `st` sector efficiencies is below UNCOALESCED_EFFICIENCY.
    2. bank-conflicts: `shared_conflict_fraction` is at least BANK_CONFLICT_FRACTION.
    3. tail-effect: more than one wave, and `wave_efficiency` below TAIL_EFFICIENCY.
    4. memory-bound: DRAM percent at least MEMORY_BOUND_DRAM_PCT.
    5. compute-bound: SM percent at least COMPUTE_BOUND_SM_PCT.
    6. latency-bound: everything else.
    """
    raise NotImplementedError("TODO: classify the kernel")


# ---------------------------------------------------------------------------
# Part 5: read an nsys stats export
# ---------------------------------------------------------------------------


def parse_nsys_csv(text: str) -> list:
    """Rows of an `nsys stats --format csv` export, each a dict from header
    name to text. Lines before the header, such as `Processing [...]`, are
    skipped: the header is the first line that holds a comma. Kernel names hold
    commas inside quotes, so use the `csv` module."""
    raise NotImplementedError("TODO: parse the nsys export")


def gpu_events(rows: list) -> list:
    """From `cuda_gpu_trace` rows, one dict per row with `start_ns` (int, from
    `Start (ns)`), `dur_ns` (int, from `Duration (ns)`), `stream` (int, from
    `Strm`), `name` (from `Name`), and `is_kernel`: False for a memory copy,
    whose name starts with "[CUDA", otherwise True."""
    raise NotImplementedError("TODO: build the events")


def busy_and_gaps(events: list) -> dict:
    """The time the GPU is busy, and the idle gaps, across all streams.

    Return `span_ns` (end of the last event minus start of the first),
    `busy_ns` (the length of the union of the events' intervals), `duty`
    (`busy_ns` / `span_ns`), and `gaps` (the idle times between the merged
    intervals, in time order).

    Two streams that overlap are busy once, not twice: take the union.
    """
    raise NotImplementedError("TODO: busy time and gaps")


def launch_avg_ns(api_rows: list) -> float:
    """Average time of one kernel launch call, in nanoseconds, from
    `cuda_api_sum` rows: total time over calls for the rows whose `Name`
    starts with "cudaLaunchKernel" or "cuLaunchKernel". Columns are
    `Total Time (ns)` and `Num Calls`."""
    raise NotImplementedError("TODO: average launch time")


def diagnose_timeline(events: list, api_rows: list) -> str:
    """One of "kernel-bound", "sync-bound", "launch-bound". Use only the kernel
    events (`is_kernel`): a 4-byte copy inside a host pause is part of the pause.

    1. The kernels keep the GPU busy at least 85 percent of the span: "kernel-bound".
    2. The median gap is at least 10 launch times: the host is blocked or busy elsewhere, "sync-bound".
    3. The median kernel is shorter than 2 launch times: "launch-bound".
    4. Otherwise "kernel-bound".

    Use the median: one long gap at the end of a trace should not decide.
    """
    raise NotImplementedError("TODO: classify the timeline")
