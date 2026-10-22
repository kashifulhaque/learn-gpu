"""Lab 00: Hello, GPU.

Ask the device what it is, write a timer that measures GPU work correctly, and
use it to measure how fast the GPU copies memory. The tests import the four
functions below.
"""

import torch
from lab_common import A100, bandwidth_gbs


def describe_device() -> dict:
    """Return facts about GPU 0 as a dict with these keys:

    - "name": the device name, a str
    - "sm_count": the number of streaming multiprocessors, an int
    - "memory_gb": total memory in GB (10**9 bytes), a float
    - "capability": the compute capability as a (major, minor) tuple

    Hint: torch.cuda.get_device_properties(0).
    """
    # TODO: read the properties and build the dict.
    raise NotImplementedError


def time_ms(fn, warmup: int = 10, iters: int = 50) -> float:
    """Return the median milliseconds one call to `fn` takes on the GPU.

    Time with CUDA events, not with time.perf_counter. A kernel launch returns
    to Python as soon as the work is queued, so a host timer around `fn()`
    measures the queueing.

    Steps:
    1. Call `fn()` `warmup` times, then torch.cuda.synchronize().
    2. For each of `iters` iterations, make two torch.cuda.Event objects with
       enable_timing=True, record `start`, call `fn()`, record `end`, and
       synchronize on `end` before reading start.elapsed_time(end).
    3. Return the median of the `iters` readings.
    """
    # TODO: implement the event timer described above.
    raise NotImplementedError


def copy_bandwidth_gbs(n_mib: int = 512) -> float:
    """Return the GB/s of a device-to-device copy of `n_mib` MiB of float32.

    Allocate a source tensor and a destination tensor on the GPU, time
    `dst.copy_(src)` with your `time_ms`, and convert with `bandwidth_gbs`.
    A copy reads the source and writes the destination, so it moves twice the
    tensor's size in bytes.
    """
    # TODO: allocate, time, and convert. Count the read and the write.
    raise NotImplementedError


def ceiling_fraction(measured_gbs: float) -> float:
    """Return `measured_gbs` as a fraction of the A100's measured copy ceiling.

    Divide by A100.COPY_CEILING_GBS (1275 GB/s), not by the 1935 GB/s rating.
    """
    # TODO: one division.
    raise NotImplementedError
