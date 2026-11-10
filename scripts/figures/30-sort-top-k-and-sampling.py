#!/usr/bin/env python3
"""Numbers for chapter 30's figures and worked examples.

Everything here is computed, nothing is measured.

1. The radix-pass figure steps an LSD radix sort of eight 4-bit keys through
   two passes of a 2-bit digit. The keys include the pair 6a and 6b, which are
   equal, so a stable pass keeps 6a before 6b. The script prints the
   histogram, the exclusive scan, the destination slot of every element, and
   the output of each pass.
2. The bitonic figure runs the 8-element bitonic network and prints the
   array after each of its six stages.
3. The key table prints the order-preserving float32 key of six probe values.
4. The traffic model is arithmetic on a 4-pass, 8-bit-digit radix sort and a
   4-pass radix select, against the 1275 GB/s copy ceiling in docs/FACTS.md.
5. The merge-path and Philox examples print the diagonal cuts of a small merge
   and the uniform that the first Philox known-answer word produces.

Run `python3 scripts/figures/30-sort-top-k-and-sampling.py` and paste the
printed arrays into frontend/src/components/viz/30/.
"""

from __future__ import annotations

import json
import struct

KEYS = [13, 6, 9, 5, 3, 14, 6, 1]
TAGS = ["a", "b", "c", "d", "e", "f", "g", "h"]  # position tags, to show stability
BITS = 2


def radix_pass(items, shift, bits):
    """items is a list of (key, tag). Return the pass's trace."""
    nbins = 1 << bits
    digits = [(k >> shift) & (nbins - 1) for k, _ in items]
    counts = [digits.count(d) for d in range(nbins)]
    offsets, running = [], 0
    for c in counts:
        offsets.append(running)
        running += c
    cursor = offsets[:]
    dest = []
    out = [None] * len(items)
    for pos, d in enumerate(digits):
        dest.append(cursor[d])
        out[cursor[d]] = items[pos]
        cursor[d] += 1
    return {
        "digits": digits,
        "counts": counts,
        "offsets": offsets,
        "dest": dest,
        "out": out,
    }


def radix_trace():
    items = [(k, t) for k, t in zip(KEYS, TAGS)]
    passes = []
    for shift in (0, 2):
        trace = radix_pass(items, shift, BITS)
        trace["shift"] = shift
        trace["in"] = items
        passes.append(trace)
        items = trace["out"]
    return passes


def bitonic_stages(values):
    """The 8-element bitonic network. Return (compare pairs, array after) per stage."""
    n = len(values)
    arr = values[:]
    stages = []
    k = 2
    while k <= n:
        j = k // 2
        while j >= 1:
            pairs = []
            for i in range(n):
                partner = i ^ j
                if partner > i:
                    ascending = (i & k) == 0
                    pairs.append((i, partner, ascending))
                    if (arr[i] > arr[partner]) == ascending:
                        arr[i], arr[partner] = arr[partner], arr[i]
            stages.append({"k": k, "j": j, "pairs": pairs, "after": arr[:]})
            j //= 2
        k *= 2
    return stages


def float_bits(x: float) -> int:
    return struct.unpack(">I", struct.pack(">f", x))[0]


def float_key(x: float) -> int:
    b = float_bits(x)
    return b ^ 0xFFFFFFFF if b >> 31 else b ^ 0x80000000


def main() -> None:
    print("== radix passes (keys", KEYS, ") ==")
    for p in radix_trace():
        print(
            f"shift {p['shift']}: digits {p['digits']} counts {p['counts']} "
            f"offsets {p['offsets']} dest {p['dest']}"
        )
        print("   out", [f"{k}{t}" for k, t in p["out"]])
    print("json:", json.dumps([{k: v for k, v in p.items() if k != "in"} for p in radix_trace()]))

    print("\n== bitonic network ==")
    start = [7, 3, 6, 1, 5, 8, 2, 4]
    print("start", start)
    for s in bitonic_stages(start):
        print(f"k={s['k']} j={s['j']} pairs={[(i, q, 'up' if a else 'down') for i, q, a in s['pairs']]} after={s['after']}")
    n = 8
    stages = len(bitonic_stages(start))
    print("stages", stages, "comparators", stages * n // 2)
    for n in (1024, 1 << 20):
        lg = n.bit_length() - 1
        print(f"n={n}: stages {lg * (lg + 1) // 2}, comparators {lg * (lg + 1) // 4 * n}")

    print("\n== float keys ==")
    for x in (float("-inf"), -1.0, -0.0, 0.0, 1.0, float("inf")):
        print(f"{x!r:>6}  bits {float_bits(x):#010x}  key {float_key(x):#010x}")

    print("\n== traffic model ==")
    copy = 1275e9
    n = 2**26
    # One up-front read of every key builds all four digit histograms. Then four
    # passes each read and write every 4-byte key.
    sort_bytes = (4 + 4 * (4 + 4)) * n
    sort_3kernel_bytes = 4 * (4 + 4 + 4) * n  # count, read, and write per pass
    sort_kv_bytes = (4 + 4 * (8 + 8)) * n  # keys and 4-byte indices
    select_bytes = 4 * 4 * n  # four passes that each read every 4-byte key
    for name, b in (("sort keys, 3 phases", sort_3kernel_bytes), ("sort keys", sort_bytes), ("sort keys+indices", sort_kv_bytes), ("select", select_bytes)):
        print(f"{name}: {b / n:.0f} bytes per key, {b / 1e9:.3f} GB, floor {b / copy * 1e3:.3f} ms at 1275 GB/s")
    print("2**26 keys =", n)

    print("\n== merge path ==")
    a, b = [1, 3, 5, 7, 9], [2, 3, 4, 8]
    for diag in (0, 3, 6, 9):
        lo, hi = max(0, diag - len(b)), min(diag, len(a))
        while lo < hi:
            mid = (lo + hi) // 2
            if a[mid] <= b[diag - 1 - mid]:
                lo = mid + 1
            else:
                hi = mid
        print(f"diagonal {diag}: take {lo} from a and {diag - lo} from b")

    print("\n== philox ==")
    word = 0x6627E8D5  # first output word for counter 0 and key 0
    print(f"u = ({word:#x} >> 8) / 2**24 = {(word >> 8) / 2**24:.6f}")


if __name__ == "__main__":
    main()
