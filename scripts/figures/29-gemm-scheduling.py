#!/usr/bin/env python3
"""Numbers for chapter 29's schedule figure and worked examples.

Two shapes on the 108 SMs of the A100, one CTA per SM:

  big    M = 2304, N = 1544, K = 4096, tile 256 x 128 x 64. NVIDIA's worked
         example from docs/FACTS.md section 4.4, which chapter 4 counts as
         117 tiles. K is 64 iterations of 64.
  decode M = 64, N = 4096, K = 4096, tile 64 x 128 x 64. A skinny GEMM: 32 tiles
         of 64 iterations each.

For each schedule the script deals the work to SMs the way equal-length CTAs
reach SMs, CTA i to SM i mod 108 in wave order, and prints the load per SM, the
critical path in K-iterations, the utilisation (total work over 108 times the
critical path), the number of tiles that need a fix-up, and the cost-model time.
The cost model is the one in content/labs/29-gemm-scheduling/solution.py:
312 TFLOP/s per GPU split over 108 SMs, a 1275 GB/s copy ceiling, a 5 us launch
for the split-K reduction kernel.

Run it to print the TypeScript constants pasted into
frontend/src/components/viz/29/schedule-lanes.tsx.
"""

from __future__ import annotations

import json
import math

SMS = 108
PEAK_TFLOPS = 312.0
BW_GBS = 1275.0
LAUNCH_US = 5.0

SHAPES = {
    "big": dict(m=2304, n=1544, k=4096, tile=(256, 128, 64)),
    "decode": dict(m=64, n=4096, k=4096, tile=(64, 128, 64)),
}
SCHEDULES = {
    "big": [("data_parallel", 1), ("split_k", 2), ("split_k", 4), ("stream_k", 1)],
    "decode": [("data_parallel", 1), ("split_k", 3), ("split_k", 4), ("stream_k", 1)],
}


def split_ranges(k_iters: int, splits: int):
    base, rem = divmod(k_iters, splits)
    return [base + (1 if s < rem else 0) for s in range(splits)]


def stream_k_plan(tiles: int, ipt: int, ctas: int):
    total = tiles * ipt
    base, rem = divmod(total, ctas)
    plan = []
    for c in range(ctas):
        start = c * base + min(c, rem)
        end = start + base + (1 if c < rem else 0)
        segs, i = [], start
        while i < end:
            tile = i // ipt
            stop = min(end, (tile + 1) * ipt)
            segs.append((tile, i - tile * ipt, stop - tile * ipt))
            i = stop
        plan.append(segs)
    return plan


def fixups(plan):
    touched = {}
    for c, segs in enumerate(plan):
        for t, _, _ in segs:
            touched.setdefault(t, set()).add(c)
    return [t for t, cs in touched.items() if len(cs) > 1]


def cta_sizes(tiles: int, ipt: int, schedule: str, splits: int):
    """Iterations of each CTA, in launch order."""
    if schedule == "data_parallel":
        return [ipt] * tiles
    if schedule == "split_k":
        return [s for _ in range(tiles) for s in split_ranges(ipt, splits)]
    plan = stream_k_plan(tiles, ipt, min(SMS, tiles * ipt))
    return [sum(e - b for _, b, e in segs) for segs in plan]


def loads_per_sm(sizes):
    load = [0] * SMS
    for i, size in enumerate(sizes):
        load[i % SMS] += size
    return load


def cost_us(m, n, k, tile, schedule, splits):
    tm, tn, tk = tile
    tiles = math.ceil(m / tm) * math.ceil(n / tn)
    ipt = math.ceil(k / tk)
    t_iter = 2 * tm * tn * tk / (PEAK_TFLOPS * 1e12 / SMS) * 1e6
    extra = 0.0
    if schedule == "data_parallel":
        crit, active = math.ceil(tiles / SMS) * ipt, min(tiles, SMS)
    elif schedule == "split_k":
        ctas = tiles * splits
        crit, active = math.ceil(ctas / SMS) * math.ceil(ipt / splits), min(ctas, SMS)
        extra = LAUNCH_US + 2 * splits * m * n * 4 / (BW_GBS * 1e3)
    else:
        total = tiles * ipt
        g = min(SMS, total)
        crit, active = math.ceil(total / g), g
        extra = len(fixups(stream_k_plan(tiles, ipt, g))) * 2 * tm * tn * 4 / (BW_GBS * 1e3)
    memory = 2 * (m * k + k * n + m * n) / (BW_GBS * 1e3) * SMS / active
    return max(crit * t_iter, memory) + extra, crit * t_iter, memory, extra


def main():
    out = {}
    for name, shape in SHAPES.items():
        m, n, k, tile = shape["m"], shape["n"], shape["k"], shape["tile"]
        tiles = math.ceil(m / tile[0]) * math.ceil(n / tile[1])
        ipt = math.ceil(k / tile[2])
        total = tiles * ipt
        rows = []
        for schedule, splits in SCHEDULES[name]:
            sizes = cta_sizes(tiles, ipt, schedule, splits)
            load = loads_per_sm(sizes)
            crit = max(load)
            if schedule == "stream_k":
                nfix = len(fixups(stream_k_plan(tiles, ipt, min(SMS, total))))
            elif schedule == "split_k":
                nfix = tiles  # every tile is reduced from `splits` partials
            else:
                nfix = 0
            t, comp, mem, extra = cost_us(m, n, k, tile, schedule, splits)
            rows.append(
                dict(
                    schedule=schedule, splits=splits, ctas=len(sizes), critical_iters=crit,
                    utilisation=round(total / (SMS * crit), 4), reduced_tiles=nfix,
                    cost_us=round(t, 2), compute_us=round(comp, 2), memory_us=round(mem, 2),
                    extra_us=round(extra, 2), load=load,
                )
            )
        out[name] = dict(tiles=tiles, iters_per_tile=ipt, total_iters=total, rows=rows)
    for name, block in out.items():
        print(f"// {name}: {block['tiles']} tiles x {block['iters_per_tile']} iterations = {block['total_iters']}")
        for r in block["rows"]:
            tail = {k: v for k, v in r.items() if k != "load"}
            print("//  ", tail)
    # Extra numbers quoted in the chapter.
    print("// hybrid on the big shape: 108 data-parallel tiles, then stream-K over the last 9")
    rest = stream_k_plan(9, 64, SMS)
    sizes = [sum(e - b for _, b, e in segs) for segs in rest]
    print("//  second phase iterations per CTA:", min(sizes), "to", max(sizes), "; fix-up tiles:", len(fixups(rest)),
          "; critical path:", 64 + max(sizes), "iterations")
    print("// big stream-K plan: CTA sizes", sorted(set(cta_sizes(117, 64, "stream_k", 1))),
          "; CTAs with the larger size:", sum(1 for s in cta_sizes(117, 64, "stream_k", 1) if s == 70))
    print("const DATA = " + json.dumps(out, separators=(",", ":")) + ";")


if __name__ == "__main__":
    main()
