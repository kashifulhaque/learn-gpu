"""Lab 01: Device peaks. Reference solution."""

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
    """FLOP one SM completes per clock at `precision`."""
    if precision == "fp32":
        return float(spec.fp32_lanes_per_sm * 2)
    if precision not in spec.tensor_precisions:
        raise ValueError(f"{spec.name} has no {precision} tensor-core path")
    bf16 = spec.tensor_cores_per_sm * spec.bf16_fma_per_tensor_core_clock * 2
    return bf16 * TENSOR_RATIO[precision]


def peak_tflops(spec: Spec, precision: str, sm_count: int | None = None) -> float:
    """Peak dense TFLOP/s: SMs x FLOP per SM per clock x clock."""
    sms = spec.sms if sm_count is None else sm_count
    clock_mhz = spec.vector_clock_mhz if precision == "fp32" else spec.tensor_clock_mhz
    return sms * flop_per_sm_clock(spec, precision) * clock_mhz * 1e6 / 1e12


def hbm_bus_bits(spec: Spec) -> int:
    """Width of the HBM interface in bits."""
    return spec.hbm_stacks * spec.bits_per_stack


def pin_rate_gbps(spec: Spec) -> float:
    """Gb/s per pin that makes the bus deliver the rated bandwidth."""
    return spec.rated_gbs * 8 / hbm_bus_bits(spec)


def ridge_point(spec: Spec, precision: str, bandwidth_gbs: float | None = None) -> float:
    """FLOP per byte where the roofline turns: peak FLOP/s over bandwidth."""
    bw = spec.rated_gbs if bandwidth_gbs is None else bandwidth_gbs
    return peak_tflops(spec, precision) * 1e12 / (bw * 1e9)


def roofline_ms(
    spec: Spec, precision: str, flop: float, nbytes: float, bandwidth_gbs: float | None = None
) -> tuple[float, str]:
    """The floor under a kernel's time, in ms, and which side sets it."""
    bw = spec.rated_gbs if bandwidth_gbs is None else bandwidth_gbs
    compute_s = flop / (peak_tflops(spec, precision) * 1e12)
    memory_s = nbytes / (bw * 1e9)
    return max(compute_s, memory_s) * 1e3, "compute" if compute_s >= memory_s else "memory"


def attainable_tflops(spec: Spec, precision: str, intensity: float, bandwidth_gbs: float | None = None) -> float:
    """The roofline: the smaller of the peak and intensity x bandwidth, in TFLOP/s."""
    bw = spec.rated_gbs if bandwidth_gbs is None else bandwidth_gbs
    return min(peak_tflops(spec, precision), intensity * bw / 1e3)


def gemm_intensity(m: int, n: int, k: int, bytes_per_element: float) -> float:
    """FLOP per byte of an (m,k) x (k,n) product that reads A and B once and writes C once."""
    flop = 2 * m * n * k
    nbytes = (m * k + k * n + m * n) * bytes_per_element
    return flop / nbytes


def compare_gpus(precision: str = "bf16") -> dict[str, dict[str, float]]:
    """Peak TFLOP/s, bandwidth in GB/s, and ridge point for each GPU that supports `precision`."""
    table = {}
    for key, spec in SPECS.items():
        try:
            tflops_ = peak_tflops(spec, precision)
        except ValueError:
            continue
        table[key] = {
            "tflops": tflops_,
            "gbs": spec.rated_gbs,
            "ridge": ridge_point(spec, precision),
        }
    return table


def describe_local_device() -> dict:
    """Facts about GPU 0 from torch, plus the key of the matching entry in SPECS."""
    import torch

    props = torch.cuda.get_device_properties(0)
    capability = (props.major, props.minor)
    l2 = getattr(props, "L2_cache_size", None)
    return {
        "name": props.name,
        "sm_count": props.multi_processor_count,
        "capability": capability,
        "l2_mib": None if l2 is None else l2 / 2**20,
        "registers_per_sm": getattr(props, "regs_per_multiprocessor", None),
        "spec": CAPABILITY_TO_SPEC.get(capability),
    }


def local_peak_tflops(precision: str) -> float | None:
    """Peak TFLOP/s of GPU 0: its real SM count with the spec for its architecture, or None."""
    info = describe_local_device()
    if info["spec"] is None:
        return None
    return peak_tflops(SPECS[info["spec"]], precision, sm_count=info["sm_count"])


def measure_matmul_tflops(precision: str, n: int = 4096) -> float:
    """TFLOP/s of an n x n x n matmul, "bf16" or "fp32" (with TF32 switched off)."""
    import torch

    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        ms = cuda_time(lambda: a @ b, warmup=10, iters=30)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    return tflops(2 * n**3, ms)
