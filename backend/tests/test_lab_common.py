"""The shared lab helpers that need no GPU: the constants, the arithmetic, and
the compile cache."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "content" / "labs"))

import lab_common  # noqa: E402
from lab_common import A100, bandwidth_gbs, tflops  # noqa: E402


def test_a100_constants_agree_with_each_other():
    # Peak rates follow from the SM count and the boost clock.
    clock_hz = A100.BOOST_MHZ * 1e6
    assert A100.SM_COUNT * clock_hz * 2048 / 1e12 == pytest.approx(A100.TENSOR_BF16_TFLOPS, rel=0.005)
    assert A100.SM_COUNT * clock_hz * 128 / 1e12 == pytest.approx(A100.FP32_TFLOPS, rel=0.005)
    assert A100.TENSOR_TF32_TFLOPS == A100.TENSOR_BF16_TFLOPS / 2
    # The copy ceiling is the measured 1275 GB/s of learn-inference chapter 10.
    assert A100.COPY_CEILING_GBS == 1275.0 and A100.HBM_BW_GBS == 1935.0
    assert A100.COPY_CEILING_GBS / A100.HBM_BW_GBS == pytest.approx(0.659, abs=0.001)
    assert A100.L2_MB == 40 and A100.SMEM_PER_SM_BYTES == 164 * 1024
    assert A100.ridge_flops_per_byte(A100.TENSOR_BF16_TFLOPS) == pytest.approx(161.2, abs=0.05)
    assert A100.ridge_flops_per_byte(A100.TENSOR_BF16_TFLOPS, 2039) == pytest.approx(153.0, abs=0.05)


def test_the_constants_are_frozen():
    with pytest.raises(Exception):
        A100.SM_COUNT = 1  # type: ignore[misc]


def test_rate_helpers():
    # 2 GB in 1 ms is 2000 GB/s; 1 TFLOP in 1 ms is 1000 TFLOP/s.
    assert bandwidth_gbs(2e9, 1.0) == pytest.approx(2000.0)
    assert tflops(1e12, 1.0) == pytest.approx(1000.0)


def test_load_cuda_compiles_once_and_defaults_to_safe_flags(monkeypatch):
    torch = pytest.importorskip("torch")
    import torch.utils.cpp_extension as ext

    calls = []

    def fake_load_inline(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(ext, "load_inline", fake_load_inline)
    monkeypatch.delenv("LG_LAB_ID", raising=False)
    assert torch  # keep the import used
    lab_common._CUDA_MODULES.clear()

    first = lab_common.load_cuda("k", "src", "decl", ["f"])
    again = lab_common.load_cuda("k", "src", "decl", ["f"])
    assert first is again and len(calls) == 1
    flags = calls[0]["extra_cuda_cflags"]
    assert "-arch=sm_80" in flags and "-O3" in flags and "--use_fast_math" not in flags

    lab_common.load_cuda("k", "src", "decl", ["f"], fast_math=True)
    assert len(calls) == 2 and "--use_fast_math" in calls[1]["extra_cuda_cflags"]
    lab_common.load_cuda("k", "other source", "decl", ["f"])
    assert len(calls) == 3


def test_load_cuda_arch_comes_from_the_argument_or_from_meta(monkeypatch, tmp_path):
    import os
    import sys
    import types

    seen = []

    def fake_load_inline(**kwargs):
        seen.append((kwargs["extra_cuda_cflags"], os.environ.get("TORCH_CUDA_ARCH_LIST")))
        return object()

    extension = types.ModuleType("torch.utils.cpp_extension")
    extension.load_inline = fake_load_inline
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", extension)
    if "torch" not in sys.modules:
        monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    if "torch.utils" not in sys.modules:
        monkeypatch.setitem(sys.modules, "torch.utils", types.ModuleType("torch.utils"))
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.0")
    lab_common._CUDA_MODULES.clear()

    lab_common.load_cuda("arch", "src", "decl", ["f"], arch="sm_90a")
    assert "-arch=sm_90a" in seen[-1][0] and "-arch=sm_80" not in seen[-1][0]
    assert seen[-1][1] == "9.0a"
    # A Hopper compile restores the worker's A100 arch list.
    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "8.0"

    lab = tmp_path / "hopper"
    lab.mkdir()
    (lab / "meta.json").write_text('{"id": "hopper", "arch": "sm_90a", "gpu": "H100-80GB"}')
    monkeypatch.setenv("LG_LABS_DIR", str(tmp_path))
    monkeypatch.setenv("LG_LAB_ID", "hopper")
    lab_common.load_cuda("from-meta", "src", "decl", ["f"])
    assert "-arch=sm_90a" in seen[-1][0]

    # A lab that names an H100 but omits arch keeps the sm_80 flags.
    plain = tmp_path / "plain-h100"
    plain.mkdir()
    (plain / "meta.json").write_text('{"id": "plain-h100", "gpu": "H100-80GB"}')
    monkeypatch.setenv("LG_LAB_ID", "plain-h100")
    lab_common.load_cuda("plain", "src", "decl", ["f"])
    assert "-arch=sm_80" in seen[-1][0] and "-arch=sm_90a" not in seen[-1][0]
    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "8.0"


NCU_CSV = """\
ID,Process ID,Process Name,Host Name,Kernel Name,Context,Stream,Block Size,Grid Size,Device,CC,Section Name,Metric Name,Metric Unit,Metric Value
0,1,python,labhost,copy,1,7,"(256, 1, 1)","(1, 1, 1)",A100,8.0,Memory,dram__bytes_read.sum,byte,"4,096"
0,1,python,labhost,copy,1,7,"(256, 1, 1)","(1, 1, 1)",A100,8.0,Memory,dram__bytes_write.sum,byte,4096
0,1,python,labhost,copy,1,7,"(256, 1, 1)","(1, 1, 1)",A100,8.0,Speed,sm__throughput.avg.pct_of_peak_sustained_elapsed,%,72.5
"""


def test_read_ncu_fixture_grades_a_recorded_report(tmp_path):
    report_path = tmp_path / "fixture_ncu_copy.csv"
    report_path.write_text(NCU_CSV)
    report = lab_common.read_ncu_fixture(tmp_path)
    assert report["from_fixture"] is True
    assert report["source"] == "fixture_ncu_copy.csv"
    metrics = report["metrics"]["copy"]
    assert metrics["dram__bytes_read.sum"] == 4096.0
    assert metrics["dram__bytes_write.sum"] == 4096.0
    assert metrics["sm__throughput.avg.pct_of_peak_sustained_elapsed"] == pytest.approx(72.5)
    assert report["units"]["copy"]["dram__bytes_read.sum"] == "byte"


def test_ncu_permission_failure_uses_the_fixture_and_other_failures_do_not(tmp_path):
    (tmp_path / "fixture_ncu_copy.csv").write_text(NCU_CSV)
    denied = (
        "ERR_NVGPUCTRPERM - The user does not have permission to access "
        "NVIDIA GPU Performance Counters on the target device."
    )
    report = lab_common.ncu_metrics(
        directory=tmp_path,
        ncu_returncode=1,
        ncu_output=denied,
    )
    assert report["from_fixture"] is True
    assert report["metrics"]["copy"]["dram__bytes_read.sum"] == 4096.0

    with pytest.raises(RuntimeError, match="unknown metric"):
        lab_common.ncu_metrics(
            directory=tmp_path,
            ncu_returncode=1,
            ncu_output="==ERROR== unknown metric 'not_a_metric'",
        )

    live = lab_common.ncu_metrics(live_csv=NCU_CSV, ncu_returncode=0)
    assert live["from_fixture"] is False
    assert live["metrics"]["copy"]["dram__bytes_write.sum"] == 4096.0
