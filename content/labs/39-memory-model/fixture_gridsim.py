"""A scheduler for grid-barrier experiments. tests.py imports it.

Each simulated block is a generator that yields one atomic operation at a
time. The scheduler runs one operation of one randomly chosen resident block
per step, so the seed picks the interleaving. A block that finishes frees a
slot, and the next block in launch order takes it. A block is never paused to
make room, which is what the hardware does.
"""

from __future__ import annotations

import random


def _block_program(block, num_blocks, rounds, barrier, violations):
    for r in range(rounds):
        yield ("store", ("flag", block, r), 1)  # the work before the barrier
        yield from barrier(num_blocks)
        for other in range(num_blocks):
            seen = yield ("load", ("flag", other, r))
            if seen != 1:
                violations.append((block, r, other))


def run_grid(barrier, num_blocks, resident, rounds, seed, max_idle=3000):
    """Run `num_blocks` blocks through `rounds` barriers, `resident` at a time.

    Returns a dict with `deadlocked`, `violations` (a list of (block, round,
    other_block) where a block passed the barrier before `other_block` had
    written its flag), `finished`, and `steps`.
    """
    rng = random.Random(seed)
    memory: dict = {}
    violations: list = []
    waiting = list(range(num_blocks))
    active: dict[int, tuple] = {}  # block -> (generator, pending op)
    finished = 0
    steps = 0
    idle = 0

    def start_next():
        block = waiting.pop(0)
        gen = _block_program(block, num_blocks, rounds, barrier, violations)
        active[block] = (gen, next(gen))

    while waiting and len(active) < resident:
        start_next()

    while active:
        if idle > max_idle:
            return {"deadlocked": True, "violations": violations, "finished": finished, "steps": steps}
        block = rng.choice(sorted(active))
        gen, op = active[block]
        steps += 1
        kind = op[0]
        if kind == "load":
            result = memory.get(op[1], 0)
            idle += 1
        elif kind == "store":
            memory[op[1]] = op[2]
            result = None
            idle = 0
        else:  # "add"
            result = memory.get(op[1], 0)
            memory[op[1]] = result + op[2]
            idle = 0
        try:
            active[block] = (gen, gen.send(result))
        except StopIteration:
            del active[block]
            finished += 1
            idle = 0
            if waiting:
                start_next()
    return {"deadlocked": False, "violations": violations, "finished": finished, "steps": steps}
