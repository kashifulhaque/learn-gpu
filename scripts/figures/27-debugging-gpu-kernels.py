#!/usr/bin/env python3
"""Shared-memory hazards in a tree reduction, for chapter 27's figure and lab.

Model. A block of B threads sums B values held in shared memory `s`:

    phase 0:       every thread t writes s[t]
    phase k >= 1:  with d = B >> k, thread t < d reads s[t] and s[t + d],
                   then writes s[t]

A barrier sits between two phases. barriers[k] is the barrier between phase
k and phase k + 1, so a block with L = log2(B) steps has L boundaries.

A hazard is a read-after-write pair (address, writer, reader). The reader
thread reads an address whose latest earlier write came from a different
thread, and no barrier lies between the two phases. In the lockstep model,
which describes GPUs before Volta, threads in one warp are also ordered. The
independent-lane model is what an sm_70 or newer GPU guarantees: lanes of a
warp are ordered only by __syncwarp or a barrier.

Nothing here is a measurement. It is the trace the lab's shared_hazards
function must reproduce. The figure's data in
frontend/src/components/viz/27/race-trace.tsx is this script's output.

Run:  python3 scripts/figures/27-debugging-gpu-kernels.py
"""

from __future__ import annotations

import json
import math

WARP = 32
MODES = ("none", "load", "every")


def trace(block: int) -> list[tuple[int, int, int, str]]:
    """Every shared access as (phase, thread, address, 'R' or 'W')."""
    log = [(0, t, t, "W") for t in range(block)]
    steps = int(math.log2(block))
    for k in range(1, steps + 1):
        d = block >> k
        for t in range(d):
            log += [(k, t, t, "R"), (k, t, t + d, "R"), (k, t, t, "W")]
    return log


def hazards(block: int, barriers: list[bool], lockstep: bool = False) -> list[tuple[int, int, int, int]]:
    """RAW hazards as (phase, address, writer, reader), sorted."""
    log = trace(block)
    found = set()
    for phase, reader, addr, kind in log:
        if kind != "R":
            continue
        earlier = [(p, t) for p, t, a, k in log if k == "W" and a == addr and p < phase]
        if not earlier:
            continue
        w_phase, writer = max(earlier)
        if writer == reader:
            continue
        if any(barriers[w_phase:phase]):
            continue
        if lockstep and writer // WARP == reader // WARP:
            continue
        found.add((phase, addr, writer, reader))
    return sorted(found)


def barrier_plan(block: int, mode: str) -> list[bool]:
    steps = int(math.log2(block))
    if mode == "none":
        return [False] * steps
    if mode == "load":
        return [True] + [False] * (steps - 1)
    return [True] * steps


def main() -> None:
    data: dict[str, dict[str, dict[str, list]]] = {}
    print("block  barriers  independent-lane  lockstep")
    for block in (8, 16, 32, 64):
        data[str(block)] = {}
        for mode in MODES:
            plan = barrier_plan(block, mode)
            free = hazards(block, plan, lockstep=False)
            lock = hazards(block, plan, lockstep=True)
            data[str(block)][mode] = {"free": [list(h) for h in free], "lock": [list(h) for h in lock]}
            print(f"{block:5d}  {mode:8s}  {len(free):16d}  {len(lock):8d}")
    print()
    print("B=8, no barriers:", hazards(8, [False] * 3))
    print("B=8, barrier after load only:", hazards(8, [True, False, False]))
    print("B=8, barrier after steps 1 and 2 only:", hazards(8, [False, True, True]))
    print("B=64, no barriers, lockstep:", len(hazards(64, [False] * 6, lockstep=True)))
    print()
    print(json.dumps(data, separators=(",", ":")))


if __name__ == "__main__":
    main()
