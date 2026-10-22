"""Lab 01: Device peaks.

Derive each GPU's peak FLOP/s, bandwidth, and ridge point from its structure:
how many SMs, how wide each SM's tensor cores are, how fast the clock runs, and
how many HBM stacks feed it. Then compare your A100 numbers with a real matmul.

The `Spec` table below is given. You write the functions after it. The tests
import them, check each against NVIDIA's rated figures within 1%, and, on a
GPU, time a bf16 and an fp32 matmul against your peaks.

The formula functions need no torch, so the tests run them on a CPU too. Import
torch inside the two functions that use it, as the solution does.
"""

from __future__ import annotations

from dataclasses import dataclass

from lab_common import cuda_time, tflops


@dataclass(frozen=True)
class Spec:
    """The published structure of one GPU, enough to derive its peaks."""

    name: str
    capability: tuple[int, int]
    sms: int
    tensor_clock_mhz: float  # clock implied by the rated tensor-core throughput
    vector_clock_mhz: float  # clock implied by the rated FP32 CUDA-core throughput
    tensor_cores_per_sm: int
    bf16_fma_per_tensor_core_clock: int
    fp32_lanes_per_sm: int  # FP32 CUDA cores per SM
    hbm_stacks: int
    bits_per_stack: int
    rated_gbs: float
    tensor_precisions: tuple[str, ...]


# Each tensor precision as a multiple of the bf16 rate. fp16 equals bf16.
TENSOR_RATIO = {"fp16": 1.0, "bf16": 1.0, "tf32": 0.5, "fp8": 2.0, "fp4": 4.0}

SPECS = {
    # A100 80GB PCIe or SXM4: GA100 whitepaper and the A100 datasheet.
    # 4 tensor cores per SM x 256 FMA per clock; 64 FP32 cores per SM; 5 HBM2e stacks.
    "A100": Spec("A100", (8, 0), 108, 1410, 1410, 4, 256, 64, 5, 1024, 1935.0, ("fp16", "bf16", "tf32")),
    # H100 SXM5: 132 SMs, 128 FP32 cores per SM, 5 HBM3 stacks. The clocks are
    # implied by the rated throughput (docs/FACTS.md, section 2.2); NVIDIA does
    # not publish them in a document the course reads.
    "H100": Spec("H100", (9, 0), 132, 1830, 1980, 4, 512, 128, 5, 1024, 3350.0, ("fp16", "bf16", "tf32", "fp8")),
    # B200 in HGX B200 (about 1,000 W): 148 SMs on two dies, 8 HBM3e stacks.
    # The tensor clock is implied by 2,250 TFLOP/s and is unconfirmed; the FP32
    # lane count is derived from the 75 TFLOP/s per-GPU rate.
    "B200": Spec(
        "B200", (10, 0), 148, 1856, 1980, 4, 1024, 128, 8, 1024, 8000.0, ("fp16", "bf16", "tf32", "fp8", "fp4")
    ),
}

CAPABILITY_TO_SPEC = {spec.capability: key for key, spec in SPECS.items()}


def flop_per_sm_clock(spec: Spec, precision: str) -> float:
    """FLOP one SM completes per clock at `precision`.

    - "fp32": the CUDA cores. Each of `spec.fp32_lanes_per_sm` lanes does one
      fused multiply-add (2 FLOP) per clock.
    - A tensor precision ("fp16", "bf16", "tf32", "fp8", "fp4"): start from the
      bf16 rate, tensor_cores_per_sm x bf16_fma_per_tensor_core_clock x 2 FLOP
      per FMA, then scale by TENSOR_RATIO[precision].

    Raise ValueError when `precision` isn't in `spec.tensor_precisions`, for
    example "fp8" on an A100.
    """
    raise NotImplementedError


def peak_tflops(spec: Spec, precision: str, sm_count: int | None = None) -> float:
    """Peak dense TFLOP/s: SMs x FLOP per SM per clock x clock.

    Use `spec.vector_clock_mhz` for "fp32" and `spec.tensor_clock_mhz` for the
    tensor precisions. `sm_count` overrides `spec.sms` when given. Convert MHz
    to Hz and FLOP/s to TFLOP/s.
    """
    raise NotImplementedError


def hbm_bus_bits(spec: Spec) -> int:
    """Width of the HBM interface in bits: stacks x bits per stack."""
    raise NotImplementedError


def pin_rate_gbps(spec: Spec) -> float:
    """Gb/s per pin that makes the bus deliver `spec.rated_gbs`.

    Bandwidth in GB/s is bus_bits x pin_rate_gbps / 8. Solve it for the pin rate.
    """
    raise NotImplementedError


def ridge_point(spec: Spec, precision: str, bandwidth_gbs: float | None = None) -> float:
    """FLOP per byte where the roofline turns: peak FLOP/s over bandwidth.

    `bandwidth_gbs` defaults to `spec.rated_gbs`. Pass the measured copy
    bandwidth to get the ridge point you can reach in practice.
    """
    raise NotImplementedError


def roofline_ms(
    spec: Spec, precision: str, flop: float, nbytes: float, bandwidth_gbs: float | None = None
) -> tuple[float, str]:
    """The floor under a kernel's time, in ms, and which side sets it.

    The compute time is flop over peak FLOP/s, and the memory time is nbytes
    over bandwidth. The floor is the larger. Return (ms, "compute") when
    compute sets it and (ms, "memory") otherwise.
    """
    raise NotImplementedError


def attainable_tflops(spec: Spec, precision: str, intensity: float, bandwidth_gbs: float | None = None) -> float:
    """The roofline: min(peak, intensity x bandwidth), in TFLOP/s.

    `intensity` is in FLOP per byte and the bandwidth in GB/s, so
    intensity x bandwidth is GFLOP/s.
    """
    raise NotImplementedError


def gemm_intensity(m: int, n: int, k: int, bytes_per_element: float) -> float:
    """FLOP per byte of an (m,k) x (k,n) product that reads A and B once and writes C once.

    The FLOP count is 2 m n k. This is the best case: real kernels re-read
    tiles, so their intensity is lower.
    """
    raise NotImplementedError


def compare_gpus(precision: str = "bf16") -> dict[str, dict[str, float]]:
    """Peak TFLOP/s, bandwidth in GB/s, and ridge point for each GPU in SPECS.

    Return {"A100": {"tflops": ..., "gbs": ..., "ridge": ...}, ...}. Leave out a
    GPU that has no `precision` path.
    """
    raise NotImplementedError


def describe_local_device() -> dict:
    """Facts about GPU 0 from `torch.cuda.get_device_properties(0)`.

    Return a dict with:
    - "name": the device name
    - "sm_count": `multi_processor_count`
    - "capability": (major, minor)
    - "l2_mib": `L2_cache_size` in MiB, or None if the attribute is missing
    - "registers_per_sm": `regs_per_multiprocessor`, or None if missing
    - "spec": the key in SPECS whose capability matches, or None. CAPABILITY_TO_SPEC maps one to the other.

    Use getattr(props, name, None) for the attributes that older torch builds lack.
    """
    raise NotImplementedError


def local_peak_tflops(precision: str) -> float | None:
    """Peak TFLOP/s of GPU 0, using its real SM count, or None for an unknown architecture."""
    raise NotImplementedError


def measure_matmul_tflops(precision: str, n: int = 4096) -> float:
    """TFLOP/s of an n x n x n matmul on GPU 0, for "bf16" or "fp32".

    Allocate two random n x n CUDA tensors in the right dtype and time `a @ b`
    with `cuda_time`. For "fp32", set `torch.backends.cuda.matmul.allow_tf32 =
    False` while you time, and restore it afterwards: with it on, an fp32
    matmul uses the tensor cores and the 19.5 TFLOP/s CUDA-core peak no longer
    applies. Convert with `tflops(2 * n**3, ms)`.
    """
    raise NotImplementedError
