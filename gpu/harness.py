"""Runs one lab submission inside the GPU container.

Invoked as a subprocess so a segfault in a CUDA kernel kills only this process.
Everything printed on stdout is streamed back to the browser. The final verdict
is emitted on a single line prefixed with RESULT_PREFIX so the parent can pick
it out of the log.
"""

import faulthandler
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

RESULT_PREFIX = "@@LAB_RESULT@@ "


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def emit(result: dict[str, Any]) -> None:
    sys.stdout.flush()
    print(RESULT_PREFIX + json.dumps(result), flush=True)


def main() -> int:
    faulthandler.enable()
    work = Path(os.environ["LG_WORK_DIR"])
    tests_path = Path(os.environ["LG_TESTS_PATH"])
    submission_path = work / "submission.py"

    started = time.time()
    try:
        submission = _load_module("submission", submission_path)
    except Exception:
        traceback.print_exc()
        emit(
            {
                "passed": False,
                "checks": [
                    {
                        "name": "import",
                        "passed": False,
                        "detail": "Your file raised while being imported. "
                        "See the traceback above.",
                    }
                ],
                "metrics": {},
            }
        )
        return 0

    try:
        tests = _load_module("lab_tests", tests_path)
        outcome = tests.run(submission)
    except Exception:
        traceback.print_exc()
        emit(
            {
                "passed": False,
                "checks": [
                    {
                        "name": "tests",
                        "passed": False,
                        "detail": "The test harness raised. See the traceback above.",
                    }
                ],
                "metrics": {},
            }
        )
        return 0

    outcome.setdefault("metrics", {})
    outcome["metrics"]["wall_seconds"] = round(time.time() - started, 3)
    checks = outcome.get("checks", [])
    outcome["passed"] = bool(outcome.get("passed", all(c["passed"] for c in checks)))
    emit(outcome)
    return 0


if __name__ == "__main__":
    sys.exit(main())
