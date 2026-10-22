"""Lab execution for the RunPod worker.

Writes the submission to a scratch directory, runs harness.py as a subprocess,
and yields one event per line of output.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterator

from harness import RESULT_PREFIX

GPU_DIR = Path(__file__).resolve().parent
REPO_ROOT = GPU_DIR.parent
LABS_DIR = Path(os.environ.get("LG_LABS_DIR", REPO_ROOT / "content" / "labs"))
MAX_CODE_BYTES = 512 * 1024


def _event(kind: str, **fields: Any) -> dict[str, Any]:
    return {"type": kind, **fields}


def run_lab_stream(
    lab_id: str, code: str, timeout: int = 900
) -> Iterator[dict[str, Any]]:
    """Execute one submission, yielding log, result, and error events."""
    if len(code.encode()) > MAX_CODE_BYTES:
        yield _event("error", message="Submission is too large.")
        return

    lab_dir = LABS_DIR / lab_id
    tests_path = lab_dir / "tests.py"
    if not tests_path.is_file():
        yield _event("error", message=f"Lab '{lab_id}' has no tests.py")
        return

    work = Path(tempfile.mkdtemp(prefix=f"lab-{lab_id}-"))
    try:
        (work / "submission.py").write_text(code)
        for extra in lab_dir.glob("fixture_*.py"):
            shutil.copy(extra, work / extra.name)

        env = {
            **os.environ,
            "LG_WORK_DIR": str(work),
            "LG_TESTS_PATH": str(tests_path),
            "LG_LAB_ID": lab_id,
            "PYTHONUNBUFFERED": "1",
            # Compiled kernels (load_inline) are cached here, so a warm worker
            # skips nvcc when a lab's source has not changed.
            "TORCH_EXTENSIONS_DIR": os.environ.get(
                "TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions"
            ),
            "PYTHONPATH": os.pathsep.join(
                [
                    str(work),
                    str(GPU_DIR),
                    str(LABS_DIR),
                    os.environ.get("PYTHONPATH", ""),
                ]
            ),
        }

        yield _event("log", line=f"$ python harness.py   (lab: {lab_id})")

        process = subprocess.Popen(
            [sys.executable, "-u", str(GPU_DIR / "harness.py")],
            cwd=work,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # A lab that keeps printing never reaches process.wait, so the
        # deadline is enforced by a timer that kills the process instead.
        timed_out = threading.Event()

        def _kill() -> None:
            timed_out.set()
            process.kill()

        watchdog = threading.Timer(timeout, _kill)
        watchdog.daemon = True
        watchdog.start()

        result: dict[str, Any] | None = None
        assert process.stdout is not None
        try:
            for raw in process.stdout:
                line = raw.rstrip("\n")
                if line.startswith(RESULT_PREFIX):
                    try:
                        result = json.loads(line[len(RESULT_PREFIX):])
                    except json.JSONDecodeError:
                        yield _event("log", line=line)
                    continue
                yield _event("log", line=line)
            process.wait()
        finally:
            watchdog.cancel()

        if timed_out.is_set():
            yield _event("error", message=f"Lab timed out after {timeout}s.")
            return

        if result is None:
            yield _event(
                "error",
                message=(
                    f"The lab process exited with code {process.returncode} without "
                    "reporting a result. A crash (out of memory, or a bad CUDA "
                    "kernel) usually looks like this."
                ),
            )
            return

        yield _event(
            "result",
            passed=result.get("passed", False),
            checks=result.get("checks", []),
            metrics=result.get("metrics", {}),
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)
