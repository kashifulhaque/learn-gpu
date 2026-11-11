#!/usr/bin/env python3
"""Numbers for chapter 33's two figures.

Run from the repository root: python3 scripts/figures/33-fp8-and-fp4.py

Figure 33-rounding-grid draws the E2M1 grid and, for a value x, the
round-to-nearest-even result and the stochastic-rounding probability.
Figure 33-scale-outliers draws the relative error of four block-scaling
schemes as the outlier rows of an activation matrix grow. Both use the
functions in content/labs/33-fp8-fp4/solution.py, so the figure and the lab
agree. The input is seeded: a 64 by 512 Gaussian matrix with rows 3 and 17
multiplied by 10**e. The error is the mean over the other rows of
||row - dequantized row|| / ||row||.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("sol33", ROOT / "content/labs/33-fp8-fp4/solution.py")
S = importlib.util.module_from_spec(spec)
sys.modules["sol33"] = S
spec.loader.exec_module(S)


def rounding_rows():
    grid = S.E2M1_GRID
    print("E2M1 grid:", list(grid))
    print("x, lo, hi, RNE, P(up)")
    for x in (0.3, 0.75, 1.25, 2.2, 3.7, 5.0, 5.5):
        a = x
        lo = max(g for g in grid if g <= a)
        hi = min(g for g in grid if g >= a)
        rne = float(S.e2m1_decode(S.e2m1_encode(torch.tensor(x))))
        p = 0.0 if hi == lo else (a - lo) / (hi - lo)
        print(f"{x}, {lo}, {hi}, {rne}, {p:.4f}")


def relerr(x, deq, good):
    e = (x - deq).double().norm(dim=1) / x.double().norm(dim=1)
    return e[good].mean().item()


def outlier_rows():
    gen = torch.Generator().manual_seed(33)
    base = torch.randn(64, 512, generator=gen)
    good = torch.ones(64, dtype=torch.bool)
    good[[3, 17]] = False
    print("e, per-tensor E4M3, 1x128 E4M3, MXFP4, NVFP4")
    rows = []
    for e in range(0, 7):
        x = base.clone()
        x[[3, 17]] *= 10.0**e
        t = S.fp8_fake_quant(x, "tensor")
        r = S.fp8_fake_quant(x, "tile1x128")
        mc, ms = S.mxfp4_quantize(x)
        m = S.mxfp4_dequantize(mc, ms)
        nc, ns, nt = S.nvfp4_quantize(x)
        n = S.nvfp4_dequantize(nc, ns, nt)
        row = [relerr(x, v, good) for v in (t, r, m, n)]
        rows.append(row)
        print(e, ", ".join(f"{v:.4f}" for v in row))
    return rows


if __name__ == "__main__":
    rounding_rows()
    print()
    outlier_rows()
