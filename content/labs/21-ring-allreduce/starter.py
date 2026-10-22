"""Lab 21: ring all-reduce and the alpha-beta cost model.

You write two things. Part 1 is the cost model: small functions that predict
how long a collective takes from a latency alpha and a link bandwidth B. Part 2
is the ring itself: a reduce-scatter and an all-gather built from point-to-point
`isend` and `irecv`, which you check against `dist.all_reduce`.

The tests start p processes with torch.multiprocessing, connect them with the
gloo backend on CPU tensors, and call your functions inside each rank. This lab
runs on the same machine as your GPU job, but it doesn't use the GPU: the ring
logic is identical for CUDA tensors under NCCL, and a CPU run is enough to
check it.

Bandwidths are per direction, in decimal GB/s. NVIDIA quotes NVLink and PCIe as
the sum of both directions, so the table below holds half of the quoted totals.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

# Link bandwidths per direction, in GB/s. Provided for you; don't change them.
LINKS_GBS = {
    "pcie4": 31.5,  # PCIe Gen4 x16: 16 GT/s x 128/130 x 16 lanes / 8
    "nvlink3": 300.0,  # A100: 600 GB/s total
    "nvlink4": 450.0,  # H100: 900 GB/s total
    "nvlink5": 900.0,  # B200: 1800 GB/s total
    "ib_ndr": 50.0,  # InfiniBand NDR port: 400 Gb/s / 8
}
# An assumed per-message latency, not a measurement. Your fit_alpha_beta
# measures the real value for the processes on this machine.
DEFAULT_ALPHA_S = 5e-6


# --------------------------------------------------------------------------
# Part 1: the alpha-beta model
# --------------------------------------------------------------------------


def transfer_time(n_bytes: float, alpha_s: float, bandwidth_gbs: float) -> float:
    """Seconds to send one message of `n_bytes` over one link: alpha + n / B.

    `bandwidth_gbs` is in decimal GB/s, so convert with 1e9.
    """
    # TODO
    raise NotImplementedError


def ring_allreduce_time(p: int, n_bytes: float, alpha_s: float, bandwidth_gbs: float) -> float:
    """Seconds for a ring all-reduce of `n_bytes` over `p` ranks.

    The ring runs a reduce-scatter and an all-gather. Each has p - 1 steps, and
    each step sends one chunk of n / p bytes. Return 0.0 for p = 1.
    """
    # TODO
    raise NotImplementedError


def tree_allreduce_time(p: int, n_bytes: float, alpha_s: float, bandwidth_gbs: float) -> float:
    """Seconds for an idealized pipelined tree all-reduce.

    Model it as a latency term of 2 * ceil(log2 p) messages of alpha each, plus
    a bandwidth term of 2 n / B. Return 0.0 for p = 1.
    """
    # TODO
    raise NotImplementedError


def crossover_bytes(p: int, alpha_s: float, bandwidth_gbs: float) -> float:
    """The message size below which the tree beats the ring in the model above.

    Set ring_allreduce_time equal to tree_allreduce_time and solve for n. Use
    ceil(log2 p) for the tree's depth. The answer has the form
    alpha * B * p * (p - 1 - depth), in bytes.
    """
    # TODO
    raise NotImplementedError


def bus_bandwidth_gbs(collective: str, p: int, n_bytes: float, seconds: float) -> float:
    """The nccl-tests bus bandwidth, in GB/s: algbw times a correction factor.

    algbw = n_bytes / seconds. The factor for p ranks is:
      all_reduce 2(p-1)/p; reduce_scatter, all_gather, all_to_all (p-1)/p;
      broadcast and reduce 1.
    """
    # TODO
    raise NotImplementedError


def fit_alpha_beta(sizes_bytes, times_s) -> tuple[float, float]:
    """Fit t = alpha + n / B to measured one-way transfer times.

    Returns (alpha in seconds, B in decimal GB/s). A straight-line fit of time
    against size has slope 1 / B and intercept alpha. The ordinary least-squares slope is
    sum((x - mx) * (y - my)) / sum((x - mx) ** 2), and the intercept is my - slope * mx.
    """
    # TODO
    raise NotImplementedError


# --------------------------------------------------------------------------
# Part 2: the ring itself
# --------------------------------------------------------------------------


def ring_chunks(n: int, p: int) -> list[tuple[int, int]]:
    """Split n elements into p contiguous chunks whose sizes differ by at most 1.

    Return [(start, end), ...] with p entries covering [0, n). A chunk is empty
    when n < p, and the ring has to cope with that. divmod(n, p) gives the base
    size and how many chunks get one extra element.
    """
    # TODO
    raise NotImplementedError


def ring_reduce_scatter(x: torch.Tensor, rank: int, world_size: int) -> int:
    """Sum the 1-D tensor `x` across ranks, in place, in p - 1 steps.

    Rank r sends to (r + 1) % p and receives from (r - 1) % p. In step s
    (s = 0 .. p - 2), rank r sends chunk (r - s) % p and receives chunk
    (r - s - 1) % p into a scratch buffer, then adds it into its own copy of
    that chunk. After the last step, rank r holds the fully reduced chunk
    (r + 1) % p. Return that chunk's index.

    Post the send and the receive together with dist.isend and dist.irecv, and
    wait on both. If every rank calls a blocking dist.send first, the ring
    deadlocks. Skip a send or receive whose chunk is empty; both ends can see
    that from the chunk sizes.
    """
    # TODO
    raise NotImplementedError


def ring_all_gather(x: torch.Tensor, rank: int, world_size: int) -> None:
    """Circulate the reduced chunks, in place, in p - 1 steps.

    Rank r starts owning the reduced chunk (r + 1) % p. In step s it sends
    chunk (r + 1 - s) % p and receives chunk (r - s) % p, this time straight
    into x with no addition.
    """
    # TODO
    raise NotImplementedError


def ring_allreduce(x: torch.Tensor, rank: int, world_size: int) -> None:
    """Sum the 1-D tensor `x` across all ranks, in place: reduce-scatter, then all-gather."""
    # TODO
    raise NotImplementedError
