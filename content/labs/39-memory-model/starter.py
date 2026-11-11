"""Lab 39: work and span, a litmus-test enumerator, and a grid barrier.

Everything but the last two checks runs on a CPU. Do the pieces in order.

1. amdahl_speedup
2. brent_bounds, reduction_work_span, scan_work_span
3. must_stay_ordered
4. allowed_outcomes
5. generation_barrier
6. fits_cooperative_launch
7. mp_orders

The last two checks compile CUDA on a GPU and run a message-passing kernel
and two grid barriers. A machine without a GPU skips them.
"""

from __future__ import annotations

import math  # noqa: F401  (you need log2 and ceil)

# ---------------------------------------------------------------------------
# 1. Amdahl and Brent
# ---------------------------------------------------------------------------


def amdahl_speedup(parallel_fraction: float, p: float) -> float:
    """Speedup of a program whose fraction f can use p processors.

    The serial part (1 - f) takes the same time on any p. The parallel part
    takes f / p of the one-processor time.
    """
    raise NotImplementedError("TODO: 1 / ((1 - f) + f / p)")


def brent_bounds(work: float, span: float, p: int) -> tuple[float, float]:
    """(lower, upper) bound on the time of a greedy schedule on p processors.

    No schedule beats max(work / p, span). A greedy schedule takes at most
    work / p + span.
    """
    raise NotImplementedError("TODO")


def reduction_work_span(n: int) -> tuple[int, int]:
    """(work, span) of a tree reduction of n values, counted in additions."""
    raise NotImplementedError("TODO")


def scan_work_span(n: int, network: str) -> tuple[int, int]:
    """(work, span) in additions of a scan of n values.

    network is "serial", "kogge_stone", or "blelloch". Kogge-Stone does n - 2**k
    additions at step k. Blelloch does n - 1 additions up the tree and n - 1
    back down, in 2 * log2(n) steps, and needs n to be a power of two.
    """
    raise NotImplementedError("TODO")


# ---------------------------------------------------------------------------
# 2. A litmus-test enumerator
# ---------------------------------------------------------------------------
# An instruction is one of
#   ("st", addr, value, order)   order is "relaxed" or "release"
#   ("ld", addr, reg, order)     order is "relaxed" or "acquire"
#   ("fence", "sc")


def must_stay_ordered(earlier: tuple, later: tuple, model: str) -> bool:
    """True when `later` may not execute before `earlier` in the same thread.

    model "sc": every pair stays in order.
    model "weak": a pair stays in order when either is a fence, when both touch
    the same address, when `earlier` is an acquire load, or when `later` is a
    release store. Every other pair may swap.
    """
    raise NotImplementedError("TODO")


def allowed_outcomes(threads: list[list[tuple]], model: str) -> set[tuple[int, ...]]:
    """Final register values over every execution the model allows.

    `threads` is a list of instruction lists. Memory starts at zero. An
    execution runs the instructions of all threads one at a time, and each
    thread's own order must respect must_stay_ordered. Return a set of tuples,
    each listing the registers in sorted name order.

    Search the states (which instructions are done, memory, registers) and
    keep a set of states you have seen.
    """
    raise NotImplementedError("TODO")


# ---------------------------------------------------------------------------
# 3. A generation-counter grid barrier
# ---------------------------------------------------------------------------
# A barrier is a generator. Each `yield` is one atomic operation on shared
# memory, and the scheduler sends back its result:
#   yield ("load", key)        -> the value
#   yield ("store", key, v)    -> None
#   yield ("add", key, d)      -> the value before the add
# Memory starts at zero. Use the keys "count" and "gen".


def generation_barrier(num_blocks: int):
    """Wait until all `num_blocks` blocks have arrived, and be reusable.

    Read the generation, then arrive with an add on "count". The block that
    sees num_blocks - 1 is last: reset the counter, then add 1 to "gen". Every
    other block spins on "gen" until it differs from the value it read.
    """
    raise NotImplementedError("TODO")
    yield  # keeps this a generator


def fits_cooperative_launch(blocks: int, blocks_per_sm: int, sm_count: int) -> bool:
    """True when every block of the grid can be resident at once."""
    raise NotImplementedError("TODO")


# ---------------------------------------------------------------------------
# 4. Choose the memory orders for the CUDA message-passing kernel
# ---------------------------------------------------------------------------


def mp_orders() -> tuple[str, str]:
    """(flag store order, flag load order) that make the data visible.

    The producer writes the data and then the flag. The consumer reads the
    flag and then the data. A store order is "relaxed" or "release". A load
    order is "relaxed" or "acquire". The tests run your choice through your own
    allowed_outcomes, and then through a CUDA kernel on a GPU.
    """
    return "relaxed", "relaxed"  # TODO
