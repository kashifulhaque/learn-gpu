#!/usr/bin/env python3
"""Numbers for chapter 39's two figures. Run it and read the output.

Figure 39-brent-bound. A scan or a reduction of n = 2**20 values has a work W
(additions) and a span S (steps). Brent's bound says a greedy schedule on p
processors takes at most W / p + S steps, and no schedule beats
max(W / p, S). The script prints the upper bound for p = 2**0 .. 2**20 for a
tree reduction, a Kogge-Stone scan, and a Blelloch scan. The work and span
come from the lab's solution, the same functions the lab's tests check.

Figure 39-litmus. The script prints the outcomes (r0, r1) that the lab's
enumerator allows for message passing (MP), store buffering (SB), and load
buffering (LB) under sequential consistency, under the weak model with
relaxed accesses, with release and acquire, and with a fence on each side.

Nothing here is a measurement. The bounds are arithmetic, and the outcome
sets come from the enumerator in content/labs/39-memory-model/solution.py,
which is a model of the PTX rules, not a run on a GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "content" / "labs" / "39-memory-model"))

import solution as lab  # noqa: E402

N = 1 << 20
R = "relaxed"


def brent_rows() -> list[dict]:
    nets = {
        "reduction": lab.reduction_work_span(N),
        "kogge_stone": lab.scan_work_span(N, "kogge_stone"),
        "blelloch": lab.scan_work_span(N, "blelloch"),
    }
    rows = []
    for e in range(21):
        p = 1 << e
        row = {"p": p}
        for name, (w, s) in nets.items():
            lo, hi = lab.brent_bounds(w, s, p)
            row[name] = (round(lo, 2), round(hi, 2))
        rows.append(row)
    return rows, nets


def tests(store: str, load: str, fenced: bool):
    fence = [("fence", "sc")] if fenced else []
    mp = [
        [("st", "x", 1, R)] + fence + [("st", "y", 1, store)],
        [("ld", "y", "r0", load)] + fence + [("ld", "x", "r1", R)],
    ]
    sb = [
        [("st", "x", 1, store)] + fence + [("ld", "y", "r0", load)],
        [("st", "y", 1, store)] + fence + [("ld", "x", "r1", load)],
    ]
    lb = [
        [("ld", "x", "r0", load)] + fence + [("st", "y", 1, store)],
        [("ld", "y", "r1", load)] + fence + [("st", "x", 1, store)],
    ]
    return {"MP": mp, "SB": sb, "LB": lb}


def main() -> None:
    rows, nets = brent_rows()
    print(f"n = {N}")
    for name, (w, s) in nets.items():
        print(f"{name}: work {w}, span {s}")
    ks_w, _ = nets["kogge_stone"]
    bl_w, _ = nets["blelloch"]
    cross = (ks_w - bl_w) / (nets["blelloch"][1] - nets["kogge_stone"][1])
    print(f"Kogge-Stone's bound beats Blelloch's above p = {cross:.2f}")
    print("\nconst BOUNDS = [")
    for row in rows:
        print(
            f"  {{ p: {row['p']}, reduction: {row['reduction'][1]}, kogge: {row['kogge_stone'][1]}, "
            f"blelloch: {row['blelloch'][1]} }},"
        )
    print("];")

    print("\n// Allowed (r0, r1) outcomes")
    variants = {
        "sc": ("sc", R, R, False),
        "relaxed": ("weak", R, R, False),
        "ra": ("weak", "release", "acquire", False),
        "fence": ("weak", R, R, True),
    }
    for test in ("MP", "SB", "LB"):
        for key, (model, store, load, fenced) in variants.items():
            out = lab.allowed_outcomes(tests(store, load, fenced)[test], model)
            print(f"{test} {key}: {sorted(out)}")


if __name__ == "__main__":
    main()
