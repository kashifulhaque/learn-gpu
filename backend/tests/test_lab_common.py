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
