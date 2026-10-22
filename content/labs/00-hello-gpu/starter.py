"""Lab 00: Hello, GPU.

Ask the device what it is, then measure how fast it copies memory. The tests
import the three functions below.
"""

import torch
from lab_common import A100, bandwidth_gbs, cuda_time


def describe_device() -> dict:
    """Return facts about GPU 0 as a dict with these keys:

    - "name": the device name, a str
    - "sm_count": the number of streaming multiprocessors, an int
    - "memory_gb": total memory in GB (10**9 bytes), a float
    - "capability": the compute capability as a (major, minor) tuple
    """
    raise NotImplementedError


def copy_bandwidth_gbs(n_mib: int = 512) -> float:
    """Return the GB/s of a device-to-device copy of `n_mib` MiB of float32.

    Allocate a source tensor and a destination tensor on the GPU, time
    `dst.copy_(src)` with `cuda_time`, and convert with `bandwidth_gbs`. A copy
    reads the source and writes the destination, so it moves twice the tensor's
    size in bytes.
    """
    raise NotImplementedError


def ceiling_fraction(measured_gbs: float) -> float:
    """Return `measured_gbs` as a fraction of the A100's measured copy ceiling."""
    raise NotImplementedError
