"""Lab 39 solution: work and span, a litmus-test enumerator, and a grid barrier.

Everything here runs on a CPU. The two GPU checks in tests.py use only
mp_orders from this file.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# 1. Amdahl and Brent
# ---------------------------------------------------------------------------


def amdahl_speedup(parallel_fraction: float, p: float) -> float:
    """Speedup of a program whose fraction f can use p processors."""
    f = parallel_fraction
    return 1.0 / ((1.0 - f) + f / p)


def brent_bounds(work: float, span: float, p: int) -> tuple[float, float]:
    """(lower, upper) bound on the time of a greedy schedule on p processors."""
    return max(work / p, span), work / p + span


def reduction_work_span(n: int) -> tuple[int, int]:
    """(work, span) of a tree reduction of n values, in additions."""
    return n - 1, math.ceil(math.log2(n)) if n > 1 else 0


def scan_work_span(n: int, network: str) -> tuple[int, int]:
    """(work, span) in additions of a scan of n values.

    network is "serial", "kogge_stone", or "blelloch". Blelloch needs a power
    of two.
    """
    if network == "serial":
        return n - 1, n - 1
    levels = math.ceil(math.log2(n)) if n > 1 else 0
    if network == "kogge_stone":
        return sum(n - 2**k for k in range(levels)), levels
    if network == "blelloch":
        if n < 1 or n & (n - 1):
            raise ValueError("blelloch needs a power of two")
        return 2 * (n - 1), 2 * levels
    raise ValueError(f"unknown network {network!r}")


# ---------------------------------------------------------------------------
# 2. A litmus-test enumerator
# ---------------------------------------------------------------------------
# An instruction is one of
#   ("st", addr, value, order)   order is "relaxed" or "release"
#   ("ld", addr, reg, order)     order is "relaxed" or "acquire"
#   ("fence", "sc")


def must_stay_ordered(earlier: tuple, later: tuple, model: str) -> bool:
    """True when `later` may not execute before `earlier` in the same thread."""
    if model == "sc":
        return True
    if earlier[0] == "fence" or later[0] == "fence":
        return True
    if earlier[1] == later[1]:  # same address: per-location coherence
        return True
    if earlier[0] == "ld" and earlier[3] == "acquire":
        return True
    if later[0] == "st" and later[3] == "release":
        return True
    return False


def allowed_outcomes(threads: list[list[tuple]], model: str) -> set[tuple[int, ...]]:
    """Final register values over every execution the model allows.

    The result is a set of tuples. Each tuple lists the registers in sorted
    name order. Memory starts at zero.
    """
    regs = sorted(ins[2] for t in threads for ins in t if ins[0] == "ld")
    addrs = sorted({ins[1] for t in threads for ins in t if ins[0] in ("ld", "st")})
    reg_at = {r: i for i, r in enumerate(regs)}
    addr_at = {a: i for i, a in enumerate(addrs)}
    # need[t][i] is the bitmask of instructions that must run before t[i].
    need = []
    for t in threads:
        masks = []
        for i, later in enumerate(t):
            mask = 0
            for j in range(i):
                if must_stay_ordered(t[j], later, model):
                    mask |= 1 << j
            masks.append(mask)
        need.append(masks)

    results: set[tuple[int, ...]] = set()
    seen: set[tuple] = set()

    def explore(done: tuple[int, ...], mem: tuple[int, ...], vals: tuple[int, ...]) -> None:
        key = (done, mem, vals)
        if key in seen:
            return
        seen.add(key)
        finished = True
        for ti, t in enumerate(threads):
            for i, ins in enumerate(t):
                if done[ti] >> i & 1:
                    continue
                finished = False
                if need[ti][i] & ~done[ti]:
                    continue
                new_done = done[:ti] + (done[ti] | 1 << i,) + done[ti + 1 :]
                if ins[0] == "st":
                    m = list(mem)
                    m[addr_at[ins[1]]] = ins[2]
                    explore(new_done, tuple(m), vals)
                elif ins[0] == "ld":
                    v = list(vals)
                    v[reg_at[ins[2]]] = mem[addr_at[ins[1]]]
                    explore(new_done, mem, tuple(v))
                else:
                    explore(new_done, mem, vals)
        if finished:
            results.add(vals)

    explore(tuple(0 for _ in threads), tuple(0 for _ in addrs), tuple(0 for _ in regs))
    return results


# ---------------------------------------------------------------------------
# 3. A generation-counter grid barrier
# ---------------------------------------------------------------------------
# A barrier is a generator. Each `yield` is one atomic operation on shared
# memory, and the scheduler sends back its result:
#   yield ("load", key)        -> the value
#   yield ("store", key, v)    -> None
#   yield ("add", key, d)      -> the value before the add


def generation_barrier(num_blocks: int):
    """Read the generation, arrive, and let the last block reset and release."""
    my_gen = yield ("load", "gen")  # before arriving, so it cannot see this round's release
    arrived = yield ("add", "count", 1)
    if arrived == num_blocks - 1:
        yield ("store", "count", 0)  # nobody can re-arrive until gen moves
        yield ("add", "gen", 1)  # the release
    else:
        while True:
            now = yield ("load", "gen")
            if now != my_gen:
                break


def fits_cooperative_launch(blocks: int, blocks_per_sm: int, sm_count: int) -> bool:
    """True when every block of the grid can be resident at once."""
    return blocks <= blocks_per_sm * sm_count


# ---------------------------------------------------------------------------
# 4. Choose the memory orders for the CUDA message-passing kernel
# ---------------------------------------------------------------------------


def mp_orders() -> tuple[str, str]:
    """(flag store order, flag load order) that make the data visible."""
    return "release", "acquire"
