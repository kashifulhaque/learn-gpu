"""Lab 00: Hello, GPU. Reference solution."""

import torch
from lab_common import A100, bandwidth_gbs, cuda_time


def describe_device() -> dict:
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "sm_count": props.multi_processor_count,
        "memory_gb": props.total_memory / 1e9,
        "capability": (props.major, props.minor),
    }


def copy_bandwidth_gbs(n_mib: int = 512) -> float:
    n_bytes = n_mib * 2**20
    src = torch.empty(n_bytes // 4, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)
    src.normal_()
    ms = cuda_time(lambda: dst.copy_(src), warmup=10, iters=50)
    # One read and one write of every byte.
    return bandwidth_gbs(2 * n_bytes, ms)


def ceiling_fraction(measured_gbs: float) -> float:
    return measured_gbs / A100.COPY_CEILING_GBS
