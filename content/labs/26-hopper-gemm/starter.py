"""Lab 26: the address math and the swizzle behind a Hopper GEMM.

Do the CPU pieces first. They are the tensor-map address, the 128-byte
swizzle, and the wgmma descriptor. The last two functions are real
sm_90a kernels. A machine without an H100 skips them.

1. box_byte_offset
2. box_load
3. swizzle_128b_byte
4. choose_swizzle
5. expect_tx_bytes
6. wgmma_descriptor
7. predicted_ms
8. tma_copy
9. wgmma_64x8x16
"""

from __future__ import annotations


def box_byte_offset(
    origin_col: int,
    origin_row: int,
    local_col: int,
    local_row: int,
    elem_bytes: int,
    stride_bytes: int,
) -> int:
    """Byte offset of one element of a 2D box, from the base of the tensor.

    Dimension 0 is the contiguous column. `stride_bytes` is the byte stride
    of a row, which is `globalStrides[0]` in `cuTensorMapEncodeTiled`.
    Add the box origin to the local coordinate, then scale.
    """
    col = origin_col + local_col
    row = origin_row + local_row
    return col * elem_bytes + row * stride_bytes


def box_load(
    matrix: list[list[int]],
    origin_col: int,
    origin_row: int,
    box_cols: int,
    box_rows: int,
) -> list[list[int]]:
    """Values TMA writes for one box. Out-of-bounds elements are 0.

    `matrix` is row-major. The element at local `(local_row, local_col)`
    is `matrix[origin_row + local_row][origin_col + local_col]`, or 0 when
    that coordinate falls outside the matrix. This is the zero fill of
    `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`.
    """
    # forgot_box_origin: the local index is used as the global index.
    out: list[list[int]] = []
    height = len(matrix)
    width = len(matrix[0]) if height else 0
    for local_row in range(box_rows):
        line: list[int] = []
        for local_col in range(box_cols):
            row = local_row
            col = local_col
            if row < 0 or col < 0 or row >= height or col >= width:
                line.append(0)
            else:
                line.append(matrix[row][col])
        out.append(line)
    return out


def swizzle_128b_byte(row: int, chunk: int) -> int:
    """Physical byte offset of a 16-byte chunk in one 1024-byte atom.

    `row` and `chunk` are both in 0..7. The 128-byte mode XORs `row` into
    the chunk index. The byte offset is `row * 128 + physical_chunk * 16`.
    Row 1, chunk 1 lands at byte 128.
    """
    # xor_row_into_low_byte: XOR the row into bits 0-2, not bits 4-6.
    logical = row * 128 + chunk * 16
    return logical ^ (row & 7)


def choose_swizzle(elem_bytes: int, box_inner: int) -> str:
    """Swizzle mode whose span equals the box's inner width, in bytes.

    The inner width is `elem_bytes * box_inner`. Return `"128B"`, `"64B"`,
    or `"32B"` when that product equals the span. Return `"none"` for any
    other multiple of 16. Return `"illegal"` when the product is not a
    multiple of 16.
    """
    raise NotImplementedError


def expect_tx_bytes(tiles: list[tuple[int, int, int]]) -> int:
    """Bytes an mbarrier should expect for these tiles.

    Each tile is `(rows, cols, elem_bytes)`. A 128 by 64 bf16 tile plus a
    256 by 64 bf16 tile is 49,152 bytes.
    """
    raise NotImplementedError


def wgmma_descriptor(start: int, leading: int, stride: int, layout_type: int) -> int:
    """Pack a 64-bit wgmma shared-memory descriptor.

    `start`, `leading`, and `stride` are byte counts and multiples of 16.
    Each is stored as `(value & 0x3FFFF) >> 4`, in bits 0-13, 16-29, and
    32-45. `layout_type` is the 2-bit swizzle field in bits 62-63:
    0 none, 1 the 128-byte mode, 2 the 64-byte mode, 3 the 32-byte mode.
    The base offset stays 0.
    """
    raise NotImplementedError


def predicted_ms(m: int, n: int, k: int) -> float:
    """Roofline floor in milliseconds for a bf16 GEMM on an H100 SXM5.

    Use 989.4 TFLOP/s and 3350 GB/s. Count `2*m*n*k` FLOP and
    `2*(m*k + n*k + m*n)` bytes. Return the compute floor when the
    intensity is at least the ridge, and the memory floor otherwise.
    The lab measures the kernel. This function is the prediction.
    """
    raise NotImplementedError


def tma_copy(src, origin_col: int, origin_row: int, box_cols: int, box_rows: int, swizzle: str):
    """Copy one box with TMA on an H100. Return a bf16 tensor of the box.

    `src` is a contiguous bf16 matrix, row-major. `swizzle` is `"none"` or
    `"128B"`. Out-of-bounds elements are 0. On a machine without an H100,
    print a skip message and return None. Do not compile for sm_80.
    """
    raise NotImplementedError


def wgmma_64x8x16(a, b):
    """One `wgmma.mma_async.m64n8k16` on an H100. Return a fp32 tensor.

    `a` is bf16, shape (64, 16), row-major. `b` is bf16, shape (16, 8),
    row-major. The result is fp32, shape (64, 8). On a machine without an
    H100, print a skip message and return None. Do not compile for sm_80.
    """
    raise NotImplementedError
