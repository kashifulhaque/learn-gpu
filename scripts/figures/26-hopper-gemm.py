#!/usr/bin/env python3
"""Numbers for chapter 26's swizzle map, tensor-map example, and H100 floor.

The 8 by 8 map is the 128-byte swizzle from the PTX ISA (section 5.5.7,
16-byte atomicity). Each cell is one 16-byte chunk. The physical chunk
index is the logical chunk XOR the row, and the row is taken mod 8.
Row 1 of that table is 1, 0, 3, 2, 5, 4, 7, 6.

The byte offset of a chunk is row * 128 + physical_chunk * 16. Row 1,
chunk 1 therefore lands at byte 128, not at the unswizzled byte 144.

The tensor-map example is a bf16 matrix with a 20-byte row stride. The
box origin is column 4, row 2. Element (0, 0) of the box is byte 48.
Element (3, 1) is byte 74.

The floor uses the H100 SXM5 ratings in docs/FACTS.md: 989.4 TFLOP/s
dense bf16 and 3350 GB/s. The lab measures the kernel. These numbers are
the prediction.
"""

from __future__ import annotations


def swizzle_128b_byte(row: int, chunk: int) -> int:
    """Physical byte offset of one 16-byte chunk inside the 1024-byte atom."""
    return row * 128 + (chunk ^ (row & 7)) * 16


def physical_chunks(row: int) -> list[int]:
    return [chunk ^ (row & 7) for chunk in range(8)]


def box_byte_offset(
    origin_col: int,
    origin_row: int,
    local_col: int,
    local_row: int,
    elem_bytes: int,
    stride_bytes: int,
) -> int:
    return (origin_col + local_col) * elem_bytes + (origin_row + local_row) * stride_bytes


def wgmma_descriptor(start: int, leading: int, stride: int, layout_type: int) -> int:
    """Pack a wgmma shared-memory descriptor. layout_type 1 is 128-byte swizzle."""

    def enc(value: int) -> int:
        return (value & 0x3FFFF) >> 4

    return enc(start) | (enc(leading) << 16) | (enc(stride) << 32) | (layout_type << 62)


def predicted_ms(m: int, n: int, k: int) -> float:
    flops = 2 * m * n * k
    nbytes = 2 * (m * k + n * k + m * n)
    peak = 989.4e12
    bandwidth = 3350e9
    if flops / nbytes >= peak / bandwidth:
        return flops / peak * 1e3
    return nbytes / bandwidth * 1e3


# PTX ISA section 5.5.7, the 128-byte mode, 16-byte atomicity, eight rows.
PTX_ROWS = (
    (0, 1, 2, 3, 4, 5, 6, 7),
    (1, 0, 3, 2, 5, 4, 7, 6),
    (2, 3, 0, 1, 6, 7, 4, 5),
    (3, 2, 1, 0, 7, 6, 5, 4),
    (4, 5, 6, 7, 0, 1, 2, 3),
    (5, 4, 7, 6, 1, 0, 3, 2),
    (6, 7, 4, 5, 2, 3, 0, 1),
    (7, 6, 5, 4, 3, 2, 1, 0),
)


def main() -> None:
    for row, expected in enumerate(PTX_ROWS):
        got = tuple(physical_chunks(row))
        if got != expected:
            raise SystemExit(f"row {row}: expected {expected}, actual {got}")
    print("128B swizzle, physical 16-byte chunk by row (PTX table):")
    for row in range(8):
        chunks = " ".join(f"{chunk:d}" for chunk in physical_chunks(row))
        print(f"  row {row}: {chunks}")
    print(f"row 1 chunk 1 byte offset: {swizzle_128b_byte(1, 1)}")
    print(f"unswizzled byte at row 1 chunk 1: {1 * 128 + 1 * 16}")
    print(
        "box origin (4, 2), local (0, 0):",
        box_byte_offset(4, 2, 0, 0, 2, 20),
    )
    print(
        "box origin (4, 2), local (3, 1):",
        box_byte_offset(4, 2, 3, 1, 2, 20),
    )
    desc = wgmma_descriptor(0, 128, 1024, 1)
    print(f"descriptor start 0, leading 128, stride 1024, 128B: {desc:#x}")
    big = predicted_ms(4096, 4096, 4096)
    small = predicted_ms(128, 128, 64)
    print(f"H100 SXM floor, 4096 cubed bf16: {big:.6f} ms")
    print(f"H100 SXM floor, 128x128x64 bf16: {small:.8e} ms")
    print(f"expect_tx, 128x64 and 256x64 bf16: {128 * 64 * 2 + 256 * 64 * 2}")


if __name__ == "__main__":
    main()
