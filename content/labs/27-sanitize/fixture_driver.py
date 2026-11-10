"""Runs the reader's kernels for lab 27, in a process of their own.

tests.py starts this file as a subprocess, once without a sanitizer and once
under each `compute-sanitizer --tool <X>`. A bad kernel can then crash this
process without ending the tests. The torch import is inside the functions, so
the static checker can import this file without torch.

    python fixture_driver.py SUBMISSION.py LAB_COMMON_DIR KEY|all

Each kernel prints one line that starts with RESULT_PREFIX and holds JSON.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path

RESULT_PREFIX = "@@KERNEL@@ "

# key: the attribute of the submission that holds the CUDA source, the host
# function, the sanitizer tool, and the number of elements. The declaration
# is what load_inline needs to bind the function.
CASES = {
    "oob": {
        "attr": "OOB_CUDA", "fn": "scale_shift", "tool": "memcheck", "n": 1000,
        "decl": "torch::Tensor scale_shift(torch::Tensor x);",
    },
    "race": {
        "attr": "RACE_CUDA", "fn": "block_sums", "tool": "racecheck", "n": 4096,
        "decl": "torch::Tensor block_sums(torch::Tensor x);",
    },
    "warp": {
        "attr": "WARP_CUDA", "fn": "warp_block_sums", "tool": "racecheck", "n": 4096,
        "decl": "torch::Tensor warp_block_sums(torch::Tensor x);",
    },
    "sync": {
        "attr": "SYNC_CUDA", "fn": "tail_pair_sum", "tool": "synccheck", "n": 1000,
        "decl": "torch::Tensor tail_pair_sum(torch::Tensor x);",
    },
    "init": {
        "attr": "INIT_CUDA", "fn": "stencil3", "tool": "initcheck", "n": 1000,
        "decl": "torch::Tensor stencil3(torch::Tensor x);",
    },
}


def make_input(n: int):
    """Small integers stored as floats, so every sum is exact in float32."""
    import torch

    return ((torch.arange(n) * 7 + 3) % 17 - 8).float()


def reference(key: str, x):
    """The expected output, computed on the CPU in float64 and cast back."""
    import torch
    import torch.nn.functional as F

    x64 = x.double()
    if key == "oob":
        return (2 * x64 + 1).float()
    if key == "race":
        return x64.view(-1, 256).sum(1).float()
    if key == "warp":
        return x64.view(-1, 128).sum(1).float()
    if key == "sync":
        n = x64.numel()
        padded = F.pad(x64, (0, -n % 128)).view(-1, 128)
        right = torch.cat([padded[:, 1:], torch.zeros(padded.shape[0], 1, dtype=padded.dtype)], dim=1)
        return (padded + right).flatten()[:n].float()
    if key == "init":
        p = F.pad(x64, (1, 1))
        return (p[:-2] + p[1:-1] + p[2:]).float()
    raise KeyError(key)


def run_one(key: str, module, lab_common) -> dict:
    import torch

    case = CASES[key]
    try:
        source = getattr(module, case["attr"])
    except AttributeError:
        return {"key": key, "ok": False, "error": f"the submission has no {case['attr']}"}
    try:
        built = lab_common.load_cuda(
            f"lab27_{key}", source, case["decl"], [case["fn"]], extra_cuda_cflags=("-lineinfo",)
        )
    except Exception as exc:  # noqa: BLE001
        return {"key": key, "ok": False, "error": f"compile failed: {str(exc)[-1500:]}"}
    x = make_input(case["n"])
    try:
        got = getattr(built, case["fn"])(x.cuda())
        torch.cuda.synchronize()
        got = got.cpu()
    except Exception as exc:  # noqa: BLE001
        return {"key": key, "ok": False, "error": f"{type(exc).__name__}: {str(exc)[-800:]}"}
    want = reference(key, x)
    if got.shape != want.shape:
        return {"key": key, "ok": False, "error": f"shape {tuple(got.shape)}, expected {tuple(want.shape)}"}
    err = (got.double() - want.double()).abs().max().item()
    return {"key": key, "ok": err == 0.0, "max_err": err}


def main(argv: list[str]) -> int:
    submission_path, common_dir, which = argv
    sys.path.insert(0, common_dir)
    sys.path.insert(0, str(Path(submission_path).parent))
    import lab_common

    spec = importlib.util.spec_from_file_location("submission", submission_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    keys = list(CASES) if which == "all" else [which]
    for key in keys:
        try:
            result = run_one(key, module, lab_common)
        except Exception:  # noqa: BLE001
            result = {"key": key, "ok": False, "error": traceback.format_exc()[-1500:]}
        print(RESULT_PREFIX + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
