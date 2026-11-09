#!/usr/bin/env python3
"""Numbers for chapter 24's fragment figure and epilogue example.

The accumulator thread-value layout is the one chapter 13 reads off the
Ampere m16n8k16 fragment table:

    ((4, 8), (2, 2)):((32, 1), (16, 8))

An integer lane and value are read colexicographically (first mode fastest).
The offset lands in a column-major 16 by 8 tile, offset = m + 16 * n, so

    m = floor(lane / 4) + 8 * floor(value / 2)
    n = 2 * (lane mod 4) + (value mod 2)

Epilogue bytes are for one 128 by 128 fp16 output tile and an fp32 bias of
length 128. The GEMM floor uses the A100 80GB PCIe ratings in docs/FACTS.md:
312 TFLOP/s dense fp16 tensor cores, 1935 GB/s HBM.
"""

from __future__ import annotations


def accumulator_mn(lane: int, value: int) -> tuple[int, int]:
    m = (lane // 4) + 8 * (value // 2)
    n = 2 * (lane % 4) + (value % 2)
    return m, n


def owners() -> list[list[tuple[int, int]]]:
    """owners[m][n] = (lane, value) for the 16 by 8 accumulator tile."""
    grid: list[list[tuple[int, int] | None]] = [[None] * 8 for _ in range(16)]
    for lane in range(32):
        for value in range(4):
            m, n = accumulator_mn(lane, value)
            grid[m][n] = (lane, value)
    if any(cell is None for row in grid for cell in row):
        raise SystemExit("the TV layout does not cover the 16 by 8 tile")
    return grid  # type: ignore[return-value]


def epilogue_bytes(m: int, n: int) -> tuple[int, int, int, int]:
    """Return (c_bytes, bias_bytes, unfused_epilogue_bytes, fused_extra_bytes).

    C is fp16 (2 bytes). Bias is fp32 (4 bytes). The unfused schedule is a
    bias kernel (read C, read bias, write C) plus a GELU kernel (read C,
    write C). The fused schedule keeps the accumulator in registers and
    only adds the bias read on top of the GEMM's own store of C.
    """
    c_bytes = m * n * 2
    bias_bytes = n * 4
    bias_kernel = c_bytes + bias_bytes + c_bytes
    gelu_kernel = c_bytes + c_bytes
    return c_bytes, bias_bytes, bias_kernel + gelu_kernel, bias_bytes


def gemm_floor_ms(m: int, n: int, k: int) -> tuple[float, float, float]:
    """Return (milliseconds, intensity in FLOP/byte, ridge in FLOP/byte)."""
    flops = 2 * m * n * k
    nbytes = 2 * (m * k + n * k + m * n)
    peak = 312e12
    bandwidth = 1935e9
    intensity = flops / nbytes
    ridge = peak / bandwidth
    seconds = flops / peak if intensity >= ridge else nbytes / bandwidth
    return seconds * 1e3, intensity, ridge


def main() -> None:
    grid = owners()
    print("lane 1's four accumulator cells (m, n, offset):")
    for value in range(4):
        m, n = accumulator_mn(1, value)
        print(f"  value {value}: m={m} n={n} offset={m + 16 * n}")
    print("owner lane of each cell, rows m=0..15, columns n=0..7:")
    for m, row in enumerate(grid):
        lanes = " ".join(f"{lane:2d}" for lane, _value in row)
        print(f"  m={m:2d}  {lanes}")
    c_bytes, bias_bytes, unfused, fused = epilogue_bytes(128, 128)
    print(
        f"epilogue 128x128 fp16: C={c_bytes} B, bias={bias_bytes} B, "
        f"unfused={unfused} B, fused extra={fused} B, ratio={unfused / fused:.0f}"
    )
    ms, intensity, ridge = gemm_floor_ms(4096, 4096, 4096)
    print(
        f"4096^3 fp16 floor: {ms:.4f} ms, intensity {intensity:.1f} FLOP/byte, "
        f"ridge {ridge:.1f} FLOP/byte"
    )


if __name__ == "__main__":
    main()
