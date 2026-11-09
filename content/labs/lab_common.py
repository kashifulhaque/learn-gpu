"""Helpers shared by every lab's test harness.

Two groups live here. `Checks` and `close` collect results and compare
tensors; they run anywhere. The GPU helpers (`require_cuda`, `cuda_time`,
`bandwidth_gbs`, `tflops`, `load_cuda`, and the `A100` constants) are for labs
that measure or compile kernels on the course's target card.

The worker puts this directory on `PYTHONPATH`, so a lab's `tests.py` and a
reader's `starter.py` can both write `from lab_common import ...`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


class Checks:
    """Collects pass/fail results and the metrics a lab reports."""

    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []
        self.metrics: dict[str, Any] = {}

    def add(self, name: str, passed: bool, detail: str = "") -> bool:
        self.results.append({"name": name, "passed": bool(passed), "detail": detail})
        mark = "PASS" if passed else "FAIL"
        print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""), flush=True)
        return bool(passed)

    def check(self, name: str, fn: Callable[[], Any], detail: str = "") -> bool:
        """Run `fn`; a truthy return passes, an exception fails with its message."""
        try:
            value = fn()
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            return self.add(name, False, f"{type(exc).__name__}: {exc}")
        if isinstance(value, tuple) and len(value) == 2:
            passed, extra = value
            return self.add(name, passed, extra or detail)
        return self.add(name, bool(value), detail)

    def metric(self, key: str, value: Any) -> None:
        self.metrics[key] = value
        print(f"      {key} = {value}", flush=True)

    def require(self, submission: Any, *names: str) -> bool:
        """Fail early when the submission is missing a required function."""
        missing = [n for n in names if not hasattr(submission, n)]
        if missing:
            self.add(
                "required functions",
                False,
                f"Your file must define: {', '.join(missing)}",
            )
            return False
        return True

    def finish(self) -> dict[str, Any]:
        passed = bool(self.results) and all(r["passed"] for r in self.results)
        total = len(self.results)
        won = sum(1 for r in self.results if r["passed"])
        print(f"\n{won}/{total} checks passed", flush=True)
        return {"passed": passed, "checks": self.results, "metrics": self.metrics}


def close(actual, expected, tol: float = 1e-5) -> tuple[bool, str]:
    """Compare two tensors and report the largest absolute difference."""
    import torch

    actual = torch.as_tensor(actual).float()
    expected = torch.as_tensor(expected).float()
    if actual.shape != expected.shape:
        return False, f"shape {tuple(actual.shape)}, expected {tuple(expected.shape)}"
    err = (actual - expected).abs().max().item()
    return err <= tol, f"max abs error {err:.3e} (tolerance {tol:.1e})"


def pick_device():
    """Use the GPU when there is one, otherwise fall back to the CPU.

    Labs where the GPU is the subject call `require_cuda` instead. Labs that
    only check mathematics use this, so the same tests run anywhere.
    """
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# The target GPU
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _A100:
    """The A100 80GB PCIe, the card the labs run on (compute capability 8.0).

    Every value is either rated by NVIDIA or measured on the lab card. A rating
    is a ceiling no kernel reaches; the measured copy ceiling is the number a
    memory-bound kernel is judged against. Ratings come from the NVIDIA A100
    datasheet, https://www.nvidia.com/en-us/data-center/a100/, and the NVIDIA
    Ampere GA100 whitepaper,
    https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf.
    docs/FACTS.md cites these and the other hardware numbers the chapters use.
    """

    # Hardware. Whitepaper: the full GA100 die has 128 SMs, and the A100 enables 108.
    SM_COUNT: int = 108
    # Whitepaper, "Streaming multiprocessor": 64 FP32 cores per SM, so 6,912 in all.
    FP32_CORES_PER_SM: int = 64
    # Whitepaper: 40 MB of L2. cudaDeviceProp.l2CacheSize reports 40 MiB.
    L2_BYTES: int = 40 * 1024 * 1024
    # Whitepaper: up to 164 KB of shared memory per SM, out of a 192 KB L1 and
    # shared-memory array. One block may opt in to 163 KB.
    SMEM_PER_SM_BYTES: int = 164 * 1024
    SMEM_PER_BLOCK_MAX_BYTES: int = 163 * 1024
    # CUDA Programming Guide, compute capability 8.0 table.
    REGISTERS_PER_SM: int = 65536
    MAX_THREADS_PER_SM: int = 2048
    MAX_BLOCKS_PER_SM: int = 32
    MAX_THREADS_PER_BLOCK: int = 1024
    MAX_REGISTERS_PER_THREAD: int = 255
    WARP_SIZE: int = 32
    # Datasheet: boost clock, which the peak figures below assume.
    BOOST_MHZ: int = 1410

    # Memory bandwidth, in GB/s (decimal: 1 GB is 10^9 bytes).
    # Datasheet: the 80GB PCIe card's HBM2e is rated at 1935 GB/s. The SXM4
    # module is rated at 2039 GB/s, and RunPod may hand you either.
    HBM_BW_GBS: float = 1935.0
    # Measured on the lab A100 80GB PCIe by a plain device-to-device copy, 2N
    # bytes moved per N copied: learn-inference chapter 10, quoted in chapter
    # 12 and again in its lab 12 tests. Judge a coalesced memory-bound kernel
    # against this, not against the rating.
    COPY_CEILING_GBS: float = 1275.0

    # Peak arithmetic, in TFLOP/s, dense (no 2:4 sparsity), at the boost clock.
    # Datasheet: 312 for FP16 and BF16 on tensor cores (624 with sparsity),
    # 156 for TF32 on tensor cores (312 with sparsity), 19.5 for FP32 on the
    # CUDA cores. They check against the clock: 108 SMs x 1410 MHz x
    # 2048 FLOP/clock/SM = 311.9 for BF16, and x 128 = 19.5 for FP32.
    TENSOR_BF16_TFLOPS: float = 312.0
    TENSOR_FP16_TFLOPS: float = 312.0
    TENSOR_TF32_TFLOPS: float = 156.0
    FP32_TFLOPS: float = 19.5

    @property
    def L2_MB(self) -> float:
        return self.L2_BYTES / 2**20

    def ridge_flops_per_byte(self, tflops: float, bandwidth_gbs: float | None = None) -> float:
        """The arithmetic intensity where a roofline turns, in FLOP per byte."""
        bw = self.HBM_BW_GBS if bandwidth_gbs is None else bandwidth_gbs
        return tflops * 1e12 / (bw * 1e9)


A100 = _A100()


def require_cuda() -> "torch.device":  # type: ignore[name-defined]  # noqa: F821
    """Return the CUDA device, or fail with a message a reader can act on."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("This lab needs a GPU but none is visible.")
    return torch.device("cuda")


def cuda_time(
    fn: Callable[[], Any], warmup: int = 10, iters: int = 50, *, flush_l2: bool = False
) -> float:
    """Median time of one call to `fn`, in milliseconds, measured with CUDA events.

    The events are recorded on the current stream around each call, so the
    figure is the GPU's own time, not the host's launch time. `warmup` calls
    run first, which absorbs compilation, autotuning, and cold caches.

    Set `flush_l2` to overwrite twice the L2 before every timed call, so each
    call starts with a cold cache. Without it, an input smaller than the 40 MB
    L2 is read from cache after the first iteration and the bandwidth looks
    impossibly good.
    """
    import torch

    require_cuda()
    scratch = torch.empty(2 * A100.L2_BYTES // 4, dtype=torch.float32, device="cuda") if flush_l2 else None
    for _ in range(max(0, warmup)):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends):
        if scratch is not None:
            scratch.zero_()
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return float(statistics.median(s.elapsed_time(e) for s, e in zip(starts, ends)))


def bandwidth_gbs(num_bytes: float, ms: float) -> float:
    """Bytes moved in `ms` milliseconds, as GB/s (decimal, 10^9 bytes)."""
    return num_bytes / (ms * 1e-3) / 1e9


def tflops(flops: float, ms: float) -> float:
    """Floating-point operations done in `ms` milliseconds, as TFLOP/s (10^12)."""
    return flops / (ms * 1e-3) / 1e12


_CUDA_MODULES: dict[str, Any] = {}


def _arch_from_meta() -> str | None:
    """The `arch` field of the current lab's meta.json, when the lab sets one.

    The worker sets `LG_LAB_ID`. A lab that omits `arch` returns None, and
    `load_cuda` keeps `-arch=sm_80`. Hopper labs set `"arch": "sm_90a"`.
    """
    lab_id = os.environ.get("LG_LAB_ID", "").strip()
    if not lab_id:
        return None
    roots: list[Path] = []
    override = os.environ.get("LG_LABS_DIR", "").strip()
    if override:
        roots.append(Path(override))
    roots.append(Path(__file__).resolve().parent)
    for root in roots:
        path = root / lab_id / "meta.json"
        if not path.is_file():
            continue
        try:
            meta = json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
        arch = meta.get("arch")
        if isinstance(arch, str) and arch.strip():
            return arch.strip()
        return None
    return None


def _torch_arch_list(arch: str) -> str:
    """`sm_80` becomes `8.0`, and `sm_90a` becomes `9.0a`."""
    body = arch[3:] if arch.startswith("sm_") else arch
    suffix = ""
    if body.endswith("a"):
        suffix = "a"
        body = body[:-1]
    if not body.isdigit() or len(body) < 2:
        return "8.0"
    return f"{int(body[:-1])}.{body[-1]}{suffix}"


def load_cuda(
    name: str,
    cuda_src: str,
    cpp_src: str | Sequence[str],
    functions: Sequence[str],
    *,
    arch: str | None = None,
    fast_math: bool = False,
    extra_cuda_cflags: Sequence[str] = (),
    verbose: bool = False,
) -> Any:
    """Compile CUDA C++ with `torch.utils.cpp_extension.load_inline`, once.

    `cpp_src` declares each function in `functions`, for example
    `"torch::Tensor add(torch::Tensor a, torch::Tensor b);"`; `cuda_src` defines
    them. The flags are `-O3 -arch=sm_80` unless you pass `arch` or the lab's
    `meta.json` sets `arch`. A Hopper lab passes `arch="sm_90a"`. Callers that
    omit `arch`, and labs that omit the field, keep `-arch=sm_80`.
    `--use_fast_math` is off unless you ask for it with `fast_math=True`,
    because it changes results: it swaps division, `expf`, and `rsqrtf` for
    faster, less accurate versions, and a test that checks numerics should
    not pass or fail on it by accident.

    Results are cached in the process by a hash of the sources, the function
    names, and the flags, so a test that asks for the same kernel twice
    compiles once. torch also caches the built library on disk, so a worker
    that has run the lab before skips the 30 to 60 seconds of nvcc.
    """
    chosen = (arch or "").strip() or _arch_from_meta() or "sm_80"
    flags = ["-O3", f"-arch={chosen}"]
    if fast_math:
        flags.append("--use_fast_math")
    flags += list(extra_cuda_cflags)
    cpp = cpp_src if isinstance(cpp_src, str) else "\n".join(cpp_src)

    key = hashlib.sha256(
        "\0".join([name, cuda_src, cpp, ",".join(functions), " ".join(flags)]).encode()
    ).hexdigest()
    cached = _CUDA_MODULES.get(key)
    if cached is not None:
        return cached

    # The Python-side arch list must not pick a different arch than nvcc gets.
    # The default leaves a worker's TORCH_CUDA_ARCH_LIST=8.0 in place. A
    # Hopper compilation sets 9.0a for this call and then restores the previous
    # value, so a later sm_80 compile is unchanged.
    previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if chosen == "sm_80":
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _torch_arch_list(chosen))
    else:
        os.environ["TORCH_CUDA_ARCH_LIST"] = _torch_arch_list(chosen)
    from torch.utils.cpp_extension import load_inline

    try:
        # The name is part of the build directory, so two sources under one name
        # would collide; the hash keeps edits from reusing a stale library.
        module = load_inline(
            name=f"{name}_{key[:10]}",
            cpp_sources=cpp,
            cuda_sources=cuda_src,
            functions=list(functions),
            extra_cuda_cflags=flags,
            verbose=verbose,
        )
    finally:
        if chosen != "sm_80":
            if previous_arch is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch
    _CUDA_MODULES[key] = module
    return module


# ---------------------------------------------------------------------------
# Recorded Nsight Compute reports
# ---------------------------------------------------------------------------

# ncu --csv writes one row per metric. These are the columns a lab grades.
_NCU_METRIC = "Metric Name"
_NCU_VALUE = "Metric Value"
_NCU_UNIT = "Metric Unit"
_NCU_KERNEL = "Kernel Name"


def _metric_number(text: str) -> float | str:
    raw = text.strip()
    if not raw:
        return raw
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return raw


def read_ncu_csv(source: str | Path) -> dict[str, Any]:
    """Parse an `ncu --csv` report into metrics a lab can grade.

    Pass a path or the CSV text. `ncu` writes one row per metric, with
    `Metric Name`, `Metric Value`, and `Kernel Name`. A number comes back as
    a float. Anything else, such as `n/a`, stays a string.

    The result has `kernels`, `metrics` (kernel, then metric name, then
    value), and `units`.
    """
    if isinstance(source, Path) or (isinstance(source, str) and "\n" not in source and Path(source).is_file()):
        path = Path(source)
        text = path.read_text()
        origin = path.name
    else:
        text = str(source)
        origin = ""

    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        return {"kernels": [], "metrics": {}, "units": {}, "source": origin}

    metrics: dict[str, dict[str, float | str]] = {}
    units: dict[str, dict[str, str]] = {}
    kernels: list[str] = []
    for row in rows:
        # DictReader keeps the header text, including a stray space.
        cells = {key.strip(): (value or "") for key, value in row.items() if key}
        kernel = cells.get(_NCU_KERNEL, "").strip() or "kernel"
        if kernel not in metrics:
            metrics[kernel] = {}
            units[kernel] = {}
            kernels.append(kernel)
        if _NCU_METRIC in cells:
            metric = cells[_NCU_METRIC].strip()
            if not metric:
                continue
            metrics[kernel][metric] = _metric_number(cells.get(_NCU_VALUE, ""))
            unit = cells.get(_NCU_UNIT, "").strip()
            if unit:
                units[kernel][metric] = unit
            continue
        # A wide CSV has one column per metric and one row per kernel.
        for metric, value in cells.items():
            if metric in {_NCU_KERNEL, "ID", "Process ID", "Process Name", "Host Name"}:
                continue
            metrics[kernel][metric] = _metric_number(value)
    return {"kernels": kernels, "metrics": metrics, "units": units, "source": origin}


def _fixture_dirs(directory: str | Path | None) -> list[Path]:
    found: list[Path] = []
    if directory is not None:
        found.append(Path(directory))
    found.append(Path.cwd())
    lab_id = os.environ.get("LG_LAB_ID", "").strip()
    if lab_id:
        override = os.environ.get("LG_LABS_DIR", "").strip()
        if override:
            found.append(Path(override) / lab_id)
        found.append(Path(__file__).resolve().parent / lab_id)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in found:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def read_ncu_fixture(
    directory: str | Path | None = None, name: str | None = None
) -> dict[str, Any]:
    """Read a `fixture_ncu_*.csv` report so a lab can grade without `ncu`.

    Search `directory` when you pass one, then the working directory, then
    the lab directory. `name` selects one file when a lab ships more than
    one. The worker copies `fixture_*.csv` next to the submission, which is
    the working directory of the tests.

    Raises FileNotFoundError when the lab shipped no matching fixture.
    """
    pattern = name or "fixture_ncu_*.csv"
    matches: list[Path] = []
    for folder in _fixture_dirs(directory):
        if not folder.is_dir():
            continue
        matches = sorted(folder.glob(pattern))
        if matches:
            break
    if not matches:
        raise FileNotFoundError(
            "No fixture_ncu_*.csv report was found. Profile with ncu, or ship "
            "a recorded report named fixture_ncu_*.csv."
        )
    report = read_ncu_csv(matches[0])
    report["source"] = matches[0].name
    report["from_fixture"] = True
    return report


def ncu_profile_blocked(returncode: int, output: str = "") -> bool:
    """True when `ncu` failed because it cannot read performance counters.

    A container without the profiling capability prints `ERR_NVGPUCTRPERM`
    or a permission error. A missing `ncu` binary is the same outcome for a
    lab: grade the recorded fixture instead of a live report. Any other
    failure, such as an unknown metric name, is not a permission failure.
    """
    if returncode == 0:
        return False
    text = output.lower()
    markers = (
        "err_nvgpuctrperm",
        "permission",
        "not permitted",
        "insufficient privilege",
        "operation not permitted",
        "no such file or directory",
        "ncu: not found",
        "command not found",
    )
    return returncode == 127 or any(marker in text for marker in markers)


def ncu_metrics(
    *,
    directory: str | Path | None = None,
    name: str | None = None,
    live_csv: str | Path | None = None,
    ncu_returncode: int | None = None,
    ncu_output: str = "",
) -> dict[str, Any]:
    """The report a profiling lab grades.

    Pass `live_csv` when `ncu` wrote a report. When `ncu` is missing, or
    `ncu_profile_blocked` is true, read `fixture_ncu_*.csv` instead. A live
    failure that is not a permission error raises RuntimeError and does not
    grade the fixture.
    """
    blocked = (
        ncu_returncode is not None and ncu_profile_blocked(ncu_returncode, ncu_output)
    )
    if live_csv is not None and not blocked:
        report = read_ncu_csv(live_csv)
        report["from_fixture"] = False
        if not report.get("source") and isinstance(live_csv, Path):
            report["source"] = live_csv.name
        return report
    if ncu_returncode not in (None, 0) and not blocked:
        detail = ncu_output.strip() or f"ncu exited with status {ncu_returncode}"
        raise RuntimeError(detail)
    return read_ncu_fixture(directory, name)
