#!/usr/bin/env python3
"""Runs every lab's reference solution against its own tests.

A lab whose solution fails its tests is broken, so this is the guard that runs
before anything is deployed. Labs that need a GPU are skipped on a CPU-only
machine and reported as skipped, not passed.

Every lab also gets a static check, with or without a GPU: meta.json has the
fields the app reads, starter.py, tests.py, and solution.py exist and parse,
tests.py defines run(), and every name a file imports from lab_common exists.
A static failure fails the lab even when its tests are skipped.

Usage:
    python3 scripts/check_labs.py [LAB_ID ...]
"""

import ast
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABS = ROOT / "content" / "labs"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LABS))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


REQUIRED_META = {
    "id": str,
    "title": str,
    "brief": str,
    "gpu": str,
    "needs_gpu": bool,
    "timeout": int,
    "metrics": list,
    "hints": list,
}


def static_check(lab_dir: Path, meta: dict) -> list[str]:
    """Problems that need no GPU to find."""
    problems: list[str] = []
    for key, kind in REQUIRED_META.items():
        if key not in meta:
            problems.append(f"meta.json has no {key}")
        elif not isinstance(meta[key], kind):
            problems.append(f"meta.json {key} must be a {kind.__name__}")
    if meta.get("id") != lab_dir.name:
        problems.append(f"meta.json id {meta.get('id')!r} must equal the directory name {lab_dir.name!r}")
    if "needs_weights" in meta:
        problems.append("meta.json needs_weights: this course has no model weights")

    known = {name for name in dir(load("lab_common_static", LABS / "lab_common.py")) if not name.startswith("__")}
    trees = {}
    for name in ("starter.py", "tests.py", "solution.py"):
        path = lab_dir / name
        if not path.is_file():
            problems.append(f"missing {name}")
            continue
        try:
            trees[name] = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as exc:
            problems.append(f"{name} does not parse: {exc}")
            continue
        for node in ast.walk(trees[name]):
            if isinstance(node, ast.ImportFrom) and node.module == "lab_common":
                for alias in node.names:
                    if alias.name not in known:
                        problems.append(f"{name} imports {alias.name} from lab_common, which has no such name")
    tests = trees.get("tests.py")
    if tests is not None and not any(isinstance(n, ast.FunctionDef) and n.name == "run" for n in tests.body):
        problems.append("tests.py defines no run(submission)")
    if (lab_dir / "starter.py").is_file() and (lab_dir / "solution.py").is_file():
        if (lab_dir / "starter.py").read_text() == (lab_dir / "solution.py").read_text():
            problems.append("starter.py and solution.py are identical")
    return problems


def main(argv):
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    wanted = set(argv)
    passed, failed, skipped = [], [], []

    for lab_dir in sorted(p for p in LABS.iterdir() if (p / "meta.json").is_file()):
        meta = json.loads((lab_dir / "meta.json").read_text())
        lab_id = meta.get("id", lab_dir.name)
        if wanted and lab_id not in wanted:
            continue
        static = static_check(lab_dir, meta)
        if static:
            print(f"\n=== {lab_id} ===")
            for problem in static:
                print(f"[FAIL] static: {problem}")
            failed.append(lab_id)
            continue
        # Some GPU labs check mathematics that runs anywhere; their tests call
        # pick_device and are marked device_agnostic so they still run on CPU.
        if meta.get("needs_gpu") and not has_cuda and not meta.get("device_agnostic"):
            skipped.append(lab_id)
            print(f"SKIP  {lab_id}  (needs a GPU)")
            continue

        os.environ["LG_LAB_ID"] = lab_id
        print(f"\n=== {lab_id} ===")
        # The worker copies a lab's fixture_*.py beside the submission, so a
        # lab's own directory must be importable here too.
        sys.path.insert(0, str(lab_dir))
        try:
            solution = load(f"sol_{lab_id.replace('-', '_')}", lab_dir / "solution.py")
            tests = load(f"tst_{lab_id.replace('-', '_')}", lab_dir / "tests.py")
            outcome = tests.run(solution)
        except Exception:
            traceback.print_exc()
            failed.append(lab_id)
            continue
        finally:
            sys.path.remove(str(lab_dir))

        (passed if outcome.get("passed") else failed).append(lab_id)

    print("\n" + "=" * 60)
    print(f"passed  {len(passed)}: {', '.join(passed) or '-'}")
    print(f"failed  {len(failed)}: {', '.join(failed) or '-'}")
    print(f"skipped {len(skipped)}: {', '.join(skipped) or '-'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
