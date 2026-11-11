#!/usr/bin/env python3
"""Numbers for chapter 40's figures and worked examples.

Everything here is computed, nothing is measured.

1. The allocator figure steps the lab's recorded allocation trace through the
   lab solution's caching-allocator simulator. For each event it prints the
   segments, each as a list of [size in MiB, state] blocks, and the allocated
   and reserved bytes.
2. The budget figure and the worked example use the byte formulas of the
   chapter: 16 bytes per parameter of state, 34*s*b*h bytes of saved
   activations per layer with fused attention, and 2*s*b*h bytes of boundary.
3. The chooser example runs the lab solution's choose_checkpoints on a
   six-layer model with unequal layers.

Run `python3 scripts/figures/40-memory-management-and-oom.py` and paste the
printed arrays into frontend/src/components/viz/40/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[2] / "content" / "labs" / "40-memory-budget"
sys.path.insert(0, str(LAB))

import fixture_snapshot as F  # noqa: E402
import solution as S  # noqa: E402

MiB = 1 << 20
GiB = 1 << 30


def allocator_steps():
    alloc = S.CachingAllocator()
    handles = {}
    steps = [{"label": "start", "segments": [], "allocated": 0, "reserved": 0}]
    for ev in F.ALLOC_TRACE:
        if ev[0] == "alloc":
            handles[ev[1]] = (alloc.malloc(ev[2]), ev[1])
            label = f"alloc {ev[1]} {ev[2] // MiB} MiB"
        else:
            alloc.free(handles.pop(ev[1])[0])
            label = f"free {ev[1]}"
        names = {h: n for h, n in handles.values()}
        segs = []
        for seg in alloc.segments:
            blocks = []
            for blk in seg["blocks"]:
                tag = names.get(blk["handle"], "") if not blk["free"] else ""
                blocks.append([blk["size"] // MiB, tag])
            segs.append(blocks)
        steps.append({"label": label, "segments": segs,
                      "allocated": alloc.allocated // MiB, "reserved": alloc.reserved // MiB})
    return steps


def main():
    print("== allocator steps (MiB; a block is [size, name], empty name = free) ==")
    print(json.dumps(allocator_steps()))

    print("\n== state bytes ==")
    for n, label in ((1_000_000_000, "1B"), (7_000_000_000, "7B")):
        for mode in ("fp32", "mixed"):
            t = S.state_bytes(n, mode)
            print(label, mode, t["total"], f"{t['total'] / 1e9:.0f} GB", f"{t['total'] / GiB:.2f} GiB")

    print("\n== worked example: h=2048, a=16, 24 layers, s=4096, fused attention ==")
    h, a, layers, s = 2048, 16, 24, 4096
    params = 12 * h * h * layers
    print("params (12 h^2 per layer)", params)
    st = S.state_bytes(params, "mixed")
    print("state", st["total"], f"{st['total'] / GiB:.2f} GiB")
    for b in (4, 8, 12):
        full = S.activation_layer_bytes(s, b, h, a, flash=True)
        bnd = 2 * s * b * h
        plain = full * layers
        print(f"b={b}: layer {full} B = {full / MiB:.0f} MiB, boundary {bnd / MiB:.0f} MiB, "
              f"all layers {plain / GiB:.2f} GiB, all checkpointed {(bnd * layers + full) / GiB:.2f} GiB")
    b = 12
    full = S.activation_layer_bytes(s, b, h, a, flash=True)
    bnd = 2 * s * b * h
    card = 80 * GiB
    headroom = 4 * GiB  # an assumption of the chapter: context, workspace, logits slack
    room = card - st["total"] - headroom
    print("card", card, "headroom", headroom, "room for activations", room, f"{room / GiB:.2f} GiB")
    for k in range(layers + 1):
        peak = S.peak_activation_bytes([full] * layers, [bnd] * layers, set(range(k)))
        print(f"  checkpoint layers 0..{k - 1}: peak {peak / GiB:.2f} GiB, fits {peak <= room}")
        if peak <= room:
            break
    print("recompute cost as a share of one forward pass:", f"{k}/{layers} = {k / layers:.3f}")

    print("\n== chooser: six unequal layers (MiB) ==")
    full6 = [900, 900, 600, 600, 1200, 1200]
    bnd6 = [100, 100, 100, 100, 100, 100]
    cost6 = [9, 9, 6, 6, 12, 12]
    for budget in (5400, 4200, 3000, 2200, 1500):
        got = S.choose_checkpoints(full6, bnd6, cost6, budget)
        print(budget, got)
    print("no checkpoint peak", S.peak_activation_bytes(full6, bnd6, set()),
          "all", S.peak_activation_bytes(full6, bnd6, set(range(6))))


if __name__ == "__main__":
    main()
