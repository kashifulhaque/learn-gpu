"""Ring all-reduce and the alpha-beta cost model: the worked solution."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

# Link bandwidths per direction, in GB/s (decimal). NVIDIA quotes NVLink and
# PCIe as the sum of both directions; these are half of that, because the ring
# formula needs the rate at which one rank can send.
LINKS_GBS = {
    "pcie4": 31.5,  # PCIe Gen4 x16: 16 GT/s x 128/130 x 16 lanes / 8
    "nvlink3": 300.0,  # A100: 600 GB/s total
    "nvlink4": 450.0,  # H100: 900 GB/s total
    "nvlink5": 900.0,  # B200: 1800 GB/s total
    "ib_ndr": 50.0,  # InfiniBand NDR port: 400 Gb/s / 8
}
# An assumed per-message latency, not a measured one. Your fit below measures
# the real value for the processes on this machine.
DEFAULT_ALPHA_S = 5e-6


# --------------------------------------------------------------------------
# Part 1: the alpha-beta model
# --------------------------------------------------------------------------


def transfer_time(n_bytes: float, alpha_s: float, bandwidth_gbs: float) -> float:
    """Seconds to send one message of `n_bytes` over one link: alpha + n / B."""
    return alpha_s + n_bytes / (bandwidth_gbs * 1e9)


def ring_allreduce_time(p: int, n_bytes: float, alpha_s: float, bandwidth_gbs: float) -> float:
    """Seconds for a ring all-reduce of `n_bytes` over `p` ranks.

    2 (p - 1) steps, each moving n / p bytes: reduce-scatter, then all-gather.
    """
    if p <= 1:
        return 0.0
    steps = 2 * (p - 1)
    return steps * transfer_time(n_bytes / p, alpha_s, bandwidth_gbs)


def tree_allreduce_time(p: int, n_bytes: float, alpha_s: float, bandwidth_gbs: float) -> float:
    """Seconds for an idealized pipelined tree all-reduce.

    The latency term is 2 ceil(log2 p) alpha, and the bandwidth term is 2 n / B.
    """
    if p <= 1:
        return 0.0
    depth = math.ceil(math.log2(p))
    return 2 * depth * alpha_s + 2 * n_bytes / (bandwidth_gbs * 1e9)


def crossover_bytes(p: int, alpha_s: float, bandwidth_gbs: float) -> float:
    """The size below which the tree beats the ring in the model above.

    Set ring == tree and solve for n. The result is alpha * B * p * (p - 1 - log2 p)
    when p is a power of two; use the ceiling of log2 p in general.
    """
    depth = math.ceil(math.log2(p))
    return alpha_s * bandwidth_gbs * 1e9 * p * (p - 1 - depth)


_BUS_FACTOR = {
    "all_reduce": lambda p: 2 * (p - 1) / p,
    "reduce_scatter": lambda p: (p - 1) / p,
    "all_gather": lambda p: (p - 1) / p,
    "all_to_all": lambda p: (p - 1) / p,
    "broadcast": lambda p: 1.0,
    "reduce": lambda p: 1.0,
}


def bus_bandwidth_gbs(collective: str, p: int, n_bytes: float, seconds: float) -> float:
    """nccl-tests' busbw: algbw (n / t) times the collective's correction factor."""
    algbw = n_bytes / seconds / 1e9
    return algbw * _BUS_FACTOR[collective](p)


def fit_alpha_beta(sizes_bytes, times_s) -> tuple[float, float]:
    """Least-squares fit of t = alpha + n / B. Returns (alpha in seconds, B in GB/s)."""
    xs = [float(v) for v in sizes_bytes]
    ys = [float(v) for v in times_s]
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sum((x - mean_x) ** 2 for x in xs)
    intercept = mean_y - slope * mean_x
    return float(intercept), float(1.0 / slope / 1e9)


# --------------------------------------------------------------------------
# Part 2: the ring itself
# --------------------------------------------------------------------------


def ring_chunks(n: int, p: int) -> list[tuple[int, int]]:
    """Split n elements into p contiguous chunks whose sizes differ by at most 1.

    Returns [(start, end), ...]. A chunk can be empty when n < p.
    """
    base, extra = divmod(n, p)
    bounds, start = [], 0
    for i in range(p):
        size = base + (1 if i < extra else 0)
        bounds.append((start, start + size))
        start += size
    return bounds


def ring_reduce_scatter(x: torch.Tensor, rank: int, world_size: int) -> int:
    """Sum `x` across ranks, in place, leaving each rank one fully reduced chunk.

    Returns the index of the chunk this rank owns: (rank + 1) % world_size.
    """
    p = world_size
    bounds = ring_chunks(x.numel(), p)
    nxt, prv = (rank + 1) % p, (rank - 1) % p
    for step in range(p - 1):
        s_lo, s_hi = bounds[(rank - step) % p]
        r_lo, r_hi = bounds[(rank - step - 1) % p]
        reqs, tmp = [], None
        if s_hi > s_lo:
            reqs.append(dist.isend(x[s_lo:s_hi], nxt))
        if r_hi > r_lo:
            tmp = torch.empty(r_hi - r_lo, dtype=x.dtype, device=x.device)
            reqs.append(dist.irecv(tmp, prv))
        for req in reqs:
            req.wait()
        if tmp is not None:
            x[r_lo:r_hi] += tmp
    return (rank + 1) % p


def ring_all_gather(x: torch.Tensor, rank: int, world_size: int) -> None:
    """Circulate the reduced chunks, in place, until every rank holds all of them.

    Assumes rank r starts by owning the reduced chunk (r + 1) % world_size.
    """
    p = world_size
    bounds = ring_chunks(x.numel(), p)
    nxt, prv = (rank + 1) % p, (rank - 1) % p
    for step in range(p - 1):
        s_lo, s_hi = bounds[(rank + 1 - step) % p]
        r_lo, r_hi = bounds[(rank - step) % p]
        reqs = []
        if s_hi > s_lo:
            reqs.append(dist.isend(x[s_lo:s_hi], nxt))
        if r_hi > r_lo:
            reqs.append(dist.irecv(x[r_lo:r_hi], prv))
        for req in reqs:
            req.wait()


def ring_allreduce(x: torch.Tensor, rank: int, world_size: int) -> None:
    """Sum the 1-D tensor `x` across all ranks, in place."""
    if world_size == 1:
        return
    ring_reduce_scatter(x, rank, world_size)
    ring_all_gather(x, rank, world_size)
