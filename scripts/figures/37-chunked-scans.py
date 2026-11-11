#!/usr/bin/env python3
"""Numbers for chapter 37's figures and worked examples.

Everything here is computed, nothing is measured.

1. The chunk-size figure prints the FLOPs, the bytes, and the roofline time of
   the chunked forward pass for dk = dv = 128 at three sequence lengths, with
   the A100 80GB PCIe tensor-core rate (312 TFLOP/s dense bf16) and the
   learn-inference copy ceiling (1275 GB/s) from docs/FACTS.md. The cost model
   is the one in content/labs/37-chunked-scan/solution.py.
2. The decay figure runs torch.cumprod on constant decays g in float32 and
   float64, and reports the first position at which the cumulative product
   underflows to zero and the first chunk length at which the cumprod-ratio
   decay matrix holds a NaN.
3. The worked example prints a tiny chunked run: T = 6, C = 3, scalar state.

Run `python3 scripts/figures/37-chunked-scans.py` and paste the printed data
into frontend/src/components/viz/37/.
"""

from __future__ import annotations

import json
import math

import torch

PEAK = 312e12  # dense bf16 tensor-core FLOP/s, A100 80GB PCIe (docs/FACTS.md)
BW = 1275e9  # measured device-to-device copy, learn-inference chapter 10
DK = DV = 128
CHUNKS = [16, 32, 64, 128, 256, 512, 1024]


def flops(t, c):
    n = math.ceil(t / c)
    return n * (2 * c * c * (DK + DV) + 4 * c * DK * DV)


def nbytes(t, c):
    n = math.ceil(t / c)
    return t * (2 * DK + 2 * DV) * 2 + 2 * n * DK * DV * 4


def cost_table():
    out = {}
    for t in (4096, 16384, 65536):
        rows = []
        for c in CHUNKS:
            f, b = flops(t, c), nbytes(t, c)
            rows.append(
                {
                    "c": c,
                    "gflop": round(f / 1e9, 3),
                    "mbytes": round(b / 1e6, 3),
                    "flop_us": round(f / PEAK * 1e6, 3),
                    "byte_us": round(b / BW * 1e6, 3),
                }
            )
        best = min(rows, key=lambda r: (max(r["flop_us"], r["byte_us"]), r["c"]))["c"]
        out[t] = {"rows": rows, "best": best, "softmax_gflop": round(t * t * (DK + DV) / 1e9, 3)}
    return out


def underflow_table():
    rows = []
    for g in (0.1, 0.5, 0.9, 0.99):
        entry = {"g": g}
        for name, dtype in (("f32", torch.float32), ("f64", torch.float64)):
            cp = torch.cumprod(torch.full((4096,), g, dtype=dtype), 0)
            zero = (cp == 0).nonzero()
            entry[f"{name}_zero_at"] = int(zero[0]) + 1 if len(zero) else None
            bad = None
            for c in range(2, 4097):
                ratio = torch.tril(cp[:c, None] / cp[None, :c])
                if not torch.isfinite(ratio).all():
                    bad = c
                    break
            entry[f"{name}_nan_chunk"] = bad
        rows.append(entry)
    return rows


def decay_curve():
    # log10 of the cumulative product of a constant decay, for the figure's lines.
    out = {}
    for g in (0.1, 0.5, 0.9, 0.99):
        out[g] = [round((i + 1) * math.log10(g), 4) for i in range(512)][::8]
    return out


def tiny_example():
    # Scalar state: S_t = g_t S_{t-1} + k_t v_t, o_t = q_t S_t.
    q = [1.0, 2.0, 1.0, 1.0, 3.0, 1.0]
    k = [1.0, 1.0, 2.0, 1.0, 1.0, 2.0]
    v = [1.0, 2.0, 1.0, 3.0, 1.0, 1.0]
    g = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    s, seq = 0.0, []
    for t in range(6):
        s = g[t] * s + k[t] * v[t]
        seq.append(q[t] * s)
    c = 3
    chunks = []
    s_in = 0.0
    for start in range(0, 6, c):
        b = 0.0
        outs = []
        bs = []
        for i in range(c):
            b += math.log(g[start + i])
            bs.append(b)
        for i in range(c):
            intra = sum(
                q[start + i] * k[start + j] * math.exp(bs[i] - bs[j]) * v[start + j] for j in range(i + 1)
            )
            inter = math.exp(bs[i]) * q[start + i] * s_in
            outs.append(intra + inter)
        a_c = math.exp(bs[-1])
        u_c = sum(k[start + j] * math.exp(bs[-1] - bs[j]) * v[start + j] for j in range(c))
        s_in = a_c * s_in + u_c
        chunks.append({"a": round(a_c, 6), "u": round(u_c, 6), "out": [round(x, 6) for x in outs]})
    return {"sequential": [round(x, 6) for x in seq], "chunks": chunks, "final_state": round(s_in, 6)}


if __name__ == "__main__":
    print("COST", json.dumps(cost_table()))
    print("UNDERFLOW", json.dumps(underflow_table()))
    print("CURVE", json.dumps(decay_curve()))
    print("TINY", json.dumps(tiny_example()))
    print("crossover C=64:", 2 * 64 + 4 * DK * DV / (DK + DV))
    print("state bytes dk=dv=128 fp32:", DK * DV * 4)
