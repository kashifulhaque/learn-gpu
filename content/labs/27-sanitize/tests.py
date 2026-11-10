"""Tests for lab 27: debugging GPU kernels with compute-sanitizer.

Seven checks need no GPU. They grade the reader's report parser against the
recorded reports in fixture_reports.py, and two pieces of pure-Python
reasoning, one about indices and one about shared-memory hazards. Two checks
need an NVIDIA GPU, nvcc, and compute-sanitizer. A machine without them skips
those two with a message, the way lab 26 skips its Hopper kernels. A skip is
not a failure.

Wrong implementations the CPU checks catch:

- count_blocks is a plausible wrong error_count. It counts the findings the
  tool printed. The tool prints only the first few, so the OOB report shows 2
  findings and the summary says 48 errors.
- keeps_signature is a plausible wrong kernel name. It keeps the template and
  parameter list, "void tail_pair_sum_kernel<128>(const float *, float *, int)",
  where the lab wants "tail_pair_sum_kernel".
- trusts_summary is a plausible wrong is_clean. It returns True when the
  summary says 0 errors. The warp-level race prints a Warning, the summary
  stays at 0 errors, and the kernel is still racy.
- assumes_lockstep is a plausible wrong shared_hazards. It treats every warp
  as ordered, so 8 threads with no barrier report no hazard, where the lab
  expects 7. A GPU since Volta does not promise lockstep.
- any_barrier_clears is another wrong shared_hazards. One barrier anywhere
  clears every hazard, so a barrier after the load alone hides the 3 hazards
  between the tree steps.

The GPU checks run each kernel under its tool in a subprocess. They set
PYTORCH_NO_CUDA_MEMORY_CACHING=1: PyTorch's caching allocator hands out slices
of one large cudaMalloc, and memcheck can only see an access that leaves the
whole allocation. Without that variable an out-of-bounds write to a small
tensor lands inside the pool and goes unreported.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import fixture_driver
import fixture_reports as R
from lab_common import Checks

NAMES = (
    "OOB_CUDA", "RACE_CUDA", "WARP_CUDA", "SYNC_CUDA", "INIT_CUDA",
    "min_grid", "oob_indices", "barrier_arrivals", "shared_hazards",
    "error_count", "parse_report", "classify", "is_clean",
)

SANITIZER_OPTIONS = {
    "memcheck": ["--tool", "memcheck", "--report-api-errors", "no"],
    "racecheck": ["--tool", "racecheck", "--racecheck-report", "all"],
    "synccheck": ["--tool", "synccheck"],
    "initcheck": ["--tool", "initcheck", "--initcheck-address-space", "shared"],
}
PER_RUN_TIMEOUT_S = 420


# ---------------------------------------------------------------------------
# A brute-force reference for shared_hazards, from the access trace itself.
# ---------------------------------------------------------------------------


def reference_hazards(block, barriers, lockstep=False, warp=32):
    steps = block.bit_length() - 1
    if isinstance(barriers, bool):
        barriers = [barriers] * steps
    writes = [(0, t, t) for t in range(block)]  # (phase, thread, address)
    reads = []
    for k in range(1, steps + 1):
        d = block >> k
        for t in range(d):
            reads += [(k, t, t), (k, t, t + d)]
            writes.append((k, t, t))
    found = set()
    for phase, reader, address in reads:
        earlier = [(p, t) for p, t, a in writes if a == address and p < phase]
        if not earlier:
            continue
        w_phase, writer = max(earlier)
        if writer == reader or any(barriers[w_phase:phase]):
            continue
        if lockstep and writer // warp == reader // warp:
            continue
        found.add((address, writer, reader))
    return sorted(found)


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _fields(finding, expected):
    """Return a mismatch message for the first field that differs, else None."""
    for key, want in expected.items():
        got = finding.get(key)
        if got != want:
            return f"{key}: expected {want!r}, actual {got!r}"
    return None


def _compare(reader, text, expected_list):
    got, error = _call(reader.parse_report, text)
    if error:
        return False, f"expected {len(expected_list)} findings, actual {error}"
    if not isinstance(got, list) or len(got) != len(expected_list):
        count = len(got) if isinstance(got, list) else got
        return False, f"expected {len(expected_list)} findings, actual {count}"
    for index, (finding, expected) in enumerate(zip(got, expected_list)):
        problem = _fields(finding, expected)
        if problem:
            return False, f"finding {index}, {problem}"
    return True, f"expected {len(expected_list)} findings, actual {len(got)}, fields match"


# ---------------------------------------------------------------------------
# The GPU part
# ---------------------------------------------------------------------------


def _sanitizer_path():
    found = shutil.which("compute-sanitizer")
    if found:
        return found
    for root in (os.environ.get("CUDA_HOME"), "/usr/local/cuda"):
        if root and (Path(root) / "bin" / "compute-sanitizer").is_file():
            return str(Path(root) / "bin" / "compute-sanitizer")
    return None


def _skip_reason():
    try:
        import torch
    except ImportError:
        return "torch is not installed, so there is no GPU to run the kernels on."
    if not torch.cuda.is_available():
        return "no CUDA GPU is visible."
    if shutil.which("nvcc") is None and not os.environ.get("CUDA_HOME"):
        return "nvcc was not found, so the kernels cannot compile."
    if _sanitizer_path() is None:
        return "compute-sanitizer was not found. It ships with the CUDA toolkit."
    return None


def _env():
    env = dict(os.environ)
    env["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _driver_command(submission, which):
    import lab_common

    return [
        sys.executable, fixture_driver.__file__, submission.__file__,
        str(Path(lab_common.__file__).resolve().parent), which,
    ]


def _results(stdout):
    found = {}
    for line in stdout.splitlines():
        if line.startswith(fixture_driver.RESULT_PREFIX):
            item = json.loads(line[len(fixture_driver.RESULT_PREFIX):])
            found[item["key"]] = item
    return found


def _run(cmd, timeout):
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=_env())
    except subprocess.TimeoutExpired:
        return None, f"timed out after {timeout} s. A divergent __syncthreads can hang the kernel."
    return done, None


def correctness(submission):
    done, error = _run(_driver_command(submission, "all"), 1200)
    if error:
        return False, f"expected five correct kernels, actual {error}"
    got = _results(done.stdout)
    bad = []
    for key in fixture_driver.CASES:
        item = got.get(key)
        if item is None:
            tail = (done.stderr or done.stdout)[-400:].strip()
            bad.append(f"{key}: no result (exit {done.returncode}). {tail}")
        elif not item["ok"]:
            bad.append(f"{key}: {item.get('error') or 'max abs error ' + str(item.get('max_err'))}")
    if bad:
        return False, "expected max abs error 0 for all five, actual " + " | ".join(bad)
    return True, "expected max abs error 0 for all five, actual 0 for all five"


def _clean(stdout, returncode):
    """Strict reading of a sanitizer run: only a 0-error summary and no finding."""
    summary = re.search(r"ERROR SUMMARY:\s*(\d+)\s+errors?", stdout)
    if summary is None:
        return False, "no ERROR SUMMARY line"
    if int(summary.group(1)) != 0:
        return False, f"ERROR SUMMARY: {summary.group(1)} errors"
    hazards = re.search(r"(\d+) hazards? displayed", stdout)
    if hazards and int(hazards.group(1)) != 0:
        return False, f"RACECHECK SUMMARY: {hazards.group(1)} hazards displayed"
    for line in stdout.splitlines():
        body = re.sub(r"^=+ ?", "", line).strip() if line.startswith("=") else ""
        if re.match(r"(Error:|Warning:|Invalid |Uninitialized |Barrier error)", body):
            return False, body[:160]
    if returncode != 0:
        return False, f"exit code {returncode}"
    return True, "ERROR SUMMARY: 0 errors"


def sanitized(submission):
    sanitizer = _sanitizer_path()
    problems = []
    for key, case in fixture_driver.CASES.items():
        cmd = [sanitizer, *SANITIZER_OPTIONS[case["tool"]], "--error-exitcode", "99"]
        cmd += _driver_command(submission, key)
        print(f"      running {case['tool']} on {key} ...", flush=True)
        done, error = _run(cmd, PER_RUN_TIMEOUT_S)
        if error:
            problems.append(f"{key} ({case['tool']}): {error}")
            continue
        ok, why = _clean(done.stdout, done.returncode)
        if not ok:
            problems.append(f"{key} ({case['tool']}): {why}")
    if problems:
        return False, "expected ERROR SUMMARY: 0 errors for all five, actual " + " | ".join(problems)
    return True, "expected ERROR SUMMARY: 0 errors for all five, actual 0 for all five"


# ---------------------------------------------------------------------------


def run(submission):
    c = Checks()
    if not c.require(submission, *NAMES):
        return c.finish()
    reader = submission

    # Check 1 calls only error_count. Writing that one function passes it.
    # count_blocks fails here: the OOB report prints 2 findings and says 48 errors.
    def summary_line():
        cases = (
            (R.OOB, 48), (R.BLOCK_RACE, 2), (R.WARP_RACE, 0), (R.TAIL_SYNC, 1),
            (R.HALO_INIT, 16), (R.CLEAN, 0), ("========= COMPUTE-SANITIZER\n", None),
        )
        for text, want in cases:
            got, error = _call(reader.error_count, text)
            if error or got != want:
                return False, f"expected {want!r} for a report ending {text.strip().splitlines()[-1]!r}, actual {error or repr(got)}"
        return True, "expected 48, 2, 0, 1, 16, 0, None, actual 48, 2, 0, 1, 16, 0, None"

    c.check("the summary line gives the error count", summary_line)

    # keeps_signature fails here: the kernel must be scale_shift_kernel alone.
    c.check(
        "a memcheck report names the tool, the access, the kernel, the line, and the thread",
        lambda: _compare(
            reader, R.OOB,
            [
                {"tool": "memcheck", "kind": "invalid-global-read", "severity": "error", "kernel": "scale_shift_kernel",
                 "file": "oob.cu", "line": 11, "thread": (232, 0, 0), "block": (3, 0, 0), "size": 4, "bytes_after": 0},
                {"tool": "memcheck", "kind": "invalid-global-write", "severity": "error", "kernel": "scale_shift_kernel",
                 "file": "oob.cu", "line": 11, "thread": (232, 0, 0), "block": (3, 0, 0), "size": 4, "bytes_after": 0},
            ],
        ),
    )

    def races():
        ok, detail = _compare(
            reader, R.BLOCK_RACE,
            [
                {"tool": "racecheck", "kind": "raw-hazard", "severity": "error", "warp_level": False,
                 "kernel": "block_sum_kernel", "file": "race.cu", "line": 17, "thread": (64, 0, 0),
                 "block": (0, 0, 0), "other_line": 17, "other_thread": (0, 0, 0)},
                {"tool": "racecheck", "kind": "raw-hazard", "severity": "error", "warp_level": False,
                 "kernel": "block_sum_kernel", "file": "race.cu", "line": 17, "thread": (65, 0, 0),
                 "block": (0, 0, 0), "other_line": 17, "other_thread": (1, 0, 0)},
            ],
        )
        if not ok:
            return ok, detail
        return _compare(
            reader, R.WARP_RACE,
            [
                {"tool": "racecheck", "kind": "raw-hazard", "severity": "warning", "warp_level": True,
                 "kernel": "warp_block_sum_kernel", "file": "warp.cu", "line": 21, "thread": (16, 0, 0),
                 "block": (0, 0, 0), "other_line": 22, "other_thread": (0, 0, 0)},
            ],
        )

    c.check("a racecheck hazard names both threads and both lines, and its severity", races)

    # keeps_signature also fails here: the kernel is a template.
    def sync_and_init():
        ok, detail = _compare(
            reader, R.TAIL_SYNC,
            [{"tool": "synccheck", "kind": "divergent-barrier", "severity": "error",
              "kernel": "tail_pair_sum_kernel", "file": "sync.cu", "line": 14,
              "thread": (96, 0, 0), "block": (7, 0, 0)}],
        )
        if not ok:
            return ok, detail
        return _compare(
            reader, R.HALO_INIT,
            [
                {"tool": "initcheck", "kind": "uninitialized-shared-read", "kernel": "stencil3_kernel",
                 "file": "init.cu", "line": 15, "thread": (0, 0, 0), "block": (0, 0, 0), "size": 4},
                {"tool": "initcheck", "kind": "uninitialized-shared-read", "kernel": "stencil3_kernel",
                 "file": "init.cu", "line": 15, "thread": (127, 0, 0), "block": (0, 0, 0), "size": 4},
            ],
        )

    c.check("synccheck and initcheck reports parse, including a templated kernel", sync_and_init)

    # trusts_summary fails here: WARP_RACE has 0 errors in its summary and one warning.
    def verdicts():
        want = {
            "out-of-bounds read": R.OOB, "shared-memory race": R.BLOCK_RACE,
            "missing __syncwarp": R.WARP_RACE, "divergent __syncthreads": R.TAIL_SYNC,
            "uninitialized shared read": R.HALO_INIT,
        }
        for name, text in want.items():
            found, error = _call(reader.parse_report, text)
            if error or not found:
                return False, f"expected {name!r}, actual {error or 'no findings'}"
            got, error = _call(reader.classify, found[0])
            if error or got != name:
                return False, f"expected {name!r}, actual {error or repr(got)}"
        second, error = _call(reader.classify, reader.parse_report(R.OOB)[1])
        if error or second != "out-of-bounds write":
            return False, f"expected 'out-of-bounds write' for the second OOB finding, actual {error or repr(second)}"
        for name, text, expected in (
            ("CLEAN", R.CLEAN, True), ("OOB", R.OOB, False), ("WARP_RACE", R.WARP_RACE, False),
            ("TAIL_SYNC", R.TAIL_SYNC, False),
        ):
            got, error = _call(reader.is_clean, text)
            if error or got is not expected:
                return False, f"expected is_clean({name}) to be {expected}, actual {error or repr(got)}"
        return True, "expected five bug names and four verdicts, actual the same"

    c.check("classify names the bug, and is_clean counts warnings", verdicts)

    def index_reasoning():
        checks = (
            (lambda: reader.min_grid(1000, 256), 4),
            (lambda: reader.min_grid(1024, 256), 4),
            (lambda: reader.min_grid(1025, 256), 5),
            (lambda: reader.oob_indices(1000, 256, 4, False), list(range(1000, 1024))),
            (lambda: reader.oob_indices(1000, 256, 4, True), []),
            (lambda: reader.oob_indices(1024, 256, 4, False), []),
            (lambda: tuple(reader.barrier_arrivals(1000, 128, 7, True)), (104, True)),
            (lambda: tuple(reader.barrier_arrivals(1000, 128, 6, True)), (128, False)),
            (lambda: tuple(reader.barrier_arrivals(1024, 128, 7, True)), (128, False)),
            (lambda: tuple(reader.barrier_arrivals(900, 128, 8, True)), (0, False)),
            (lambda: tuple(reader.barrier_arrivals(1000, 128, 7, False)), (128, False)),
        )
        for index, (call, want) in enumerate(checks):
            got, error = _call(call)
            if error or got != want:
                return False, f"case {index}: expected {want!r}, actual {error or repr(got)}"
        return True, "expected a grid of 4 and 1000..1023 out of bounds and 104 arrivals in block 7, actual the same"

    c.check("index reasoning: the grid, the threads past the end, and a divergent barrier", index_reasoning)

    # assumes_lockstep fails the first case. any_barrier_clears fails the third.
    def hazards():
        cases = (
            ((8, False), {}, 7),
            ((8, True), {}, 0),
            ((8, [True, False, False]), {}, 3),
            ((8, [False, True, True]), {}, 4),
            ((16, False), {}, 15),
            ((8, False), {"lockstep": True}, 0),
            ((32, False), {"lockstep": True}, 0),
            ((64, False), {"lockstep": True}, 32),
            ((64, [True] + [False] * 5), {}, 31),
        )
        for args, kwargs, count in cases:
            want = reference_hazards(*args, **kwargs)
            assert len(want) == count, (args, kwargs, len(want), count)
            got, error = _call(reader.shared_hazards, *args, **kwargs)
            if error:
                return False, f"expected {count} hazards for {args} {kwargs}, actual {error}"
            got = [tuple(h) for h in got]
            if got != want:
                return False, f"expected {count} hazards for {args} {kwargs}, actual {len(got)}: {got[:3]}"
        return True, "expected 7, 0, 3, 4, 15, 0, 0, 32, 31 hazards, actual the same"

    c.check("shared_hazards finds the reads that race in a tree reduction", hazards)

    reason = _skip_reason()
    if reason is not None:
        for name in ("the five fixed kernels give the right answers", "the five fixed kernels are clean under their sanitizers"):
            c.add(name, True, f"SKIP: {reason} This check needs an NVIDIA GPU with compute-sanitizer.")
        return c.finish()

    print("Compiling five kernels. The first run takes a few minutes.", flush=True)
    c.check("the five fixed kernels give the right answers", lambda: correctness(submission))
    c.check("the five fixed kernels are clean under their sanitizers", lambda: sanitized(submission))
    return c.finish()
