"""Lab 35: tensor memory, its addresses, and the descriptors behind tcgen05.

Do the CPU pieces first. They are the TMEM allocator, the address
format, the warp-to-lane rule, the accumulator footprint, the two
descriptors, and the per-CTA operand traffic. The last function is a
real sm_100a kernel stub that is written for you. A machine without a
B200 skips it.

1. tmem_alloc_columns   (the first check needs only this one)
2. TmemAllocator
3. tmem_address, decode_tmem_address
4. warp_lane_range, can_access
5. accumulator_footprint, accumulator_stages
6. instr_descriptor
7. smem_descriptor
8. per_cta_operand_bytes, operand_intensity
9. tmem_roundtrip
"""

from __future__ import annotations

TMEM_LANES = 128
TMEM_COLUMNS = 512
MIN_ALLOC_COLUMNS = 32

# kind::f16 operand formats and accumulator formats, as stored in the
# instruction descriptor.
AB_FORMAT = {"f16": 0, "bf16": 1, "tf32": 2}
C_FORMAT = {"f16": 0, "f32": 1, "s32": 2}
MAJOR = {"K": 0, "MN": 1}

# Shared-memory descriptor layout types on Blackwell. The field is 3 bits
# wide, at bits 61-63. These are not the Hopper values.
LAYOUT_TYPE = {"none": 0, "128B_BASE32B": 1, "128B": 2, "64B": 4, "32B": 6}


def tmem_alloc_columns(requested: int) -> int:
    """Columns that `tcgen05.alloc` actually reserves for a request.

    The count is a power of two, at least 32 and at most 512. Round a
    request up to the next such value: 33 becomes 64, and 129 becomes 256.
    Raise ValueError when the request is below 1 or above 512.
    """
    raise NotImplementedError


class TmemAllocator:
    """A model of one SM's tensor memory: 512 columns, all 128 lanes of each.

    `alloc(n)` reserves `tmem_alloc_columns(n)` contiguous columns at the
    lowest free column and returns that base column. It raises MemoryError
    when no run of free columns is long enough. `free(base)` releases the
    allocation that starts at `base` and raises ValueError for any other
    column. `free_columns()` is the number of unreserved columns.
    """

    def __init__(self) -> None:
        raise NotImplementedError

    def alloc(self, requested: int) -> int:
        raise NotImplementedError

    def free(self, base: int) -> None:
        raise NotImplementedError

    def free_columns(self) -> int:
        raise NotImplementedError


def tmem_address(lane: int, column: int) -> int:
    """32-bit tensor-memory address: lane in bits 31:16, column in bits 15:0.

    Raise ValueError for a lane outside 0 to 127 or a column outside 0 to 511.
    """
    raise NotImplementedError


def decode_tmem_address(address: int) -> tuple[int, int]:
    """Split an address into `(lane, column)`. Raise ValueError outside the array."""
    raise NotImplementedError


def warp_lane_range(warp_id: int) -> tuple[int, int]:
    """First and last lane that warp `warp_id` of the CTA can reach.

    Warp w reaches lanes 32*(w % 4) to 32*(w % 4) + 31. Warps 0 and 4 share
    a lane group, and so do warps 1 and 5.
    """
    raise NotImplementedError


def can_access(warp_id: int, lane: int) -> bool:
    """True when warp `warp_id` may touch `lane` with tcgen05.ld or tcgen05.st."""
    raise NotImplementedError


def accumulator_footprint(m: int, n: int, cta_group: int) -> tuple[int, int, int]:
    """Per-CTA tensor memory of an M by N fp32 accumulator.

    Returns `(lanes, columns, allocated_columns)`. A row of the tile is a
    lane and a column of the tile is a 32-bit column. With `cta_group` 2 the
    pair splits M, so each CTA holds `m // 2` lanes. `allocated_columns` is
    `columns` rounded as `tcgen05.alloc` rounds it.

    Valid shapes: `cta_group` 1 has M of 64 or 128, `cta_group` 2 has M of
    128 or 256, and N is a multiple of 8 from 8 to 256. Raise ValueError
    for any other shape.
    """
    raise NotImplementedError


def accumulator_stages(n: int, carve: bool = False) -> int:
    """How many N-column accumulators fit in one SM's 512 columns.

    With `carve` false, every accumulator is its own `tcgen05.alloc`, so it
    takes `tmem_alloc_columns(n)` columns. With `carve` true, a kernel
    allocates all 512 columns once and places accumulators at column
    offsets, so each one takes exactly `n`.
    """
    raise NotImplementedError


def instr_descriptor(
    m: int,
    n: int,
    a_format: str = "bf16",
    b_format: str = "bf16",
    c_format: str = "f32",
    a_major: str = "K",
    b_major: str = "K",
    a_negate: int = 0,
    b_negate: int = 0,
    saturate: int = 0,
) -> int:
    """32-bit instruction descriptor for `tcgen05.mma.kind::f16`, dense.

    Bits: saturate 3, C format 4-5, A format 7-9, B format 10-12, A negate
    13, B negate 14, A major 15, B major 16, N >> 3 in bits 17-22, and
    M >> 4 in bits 24-28. Sparsity bits, the K-size bit, and the shift field
    stay 0. Use the AB_FORMAT, C_FORMAT, and MAJOR tables above.
    """
    raise NotImplementedError


def smem_descriptor(start: int, leading: int, stride: int, layout: str, base_offset: int = 0) -> int:
    """64-bit shared-memory descriptor for tcgen05.

    `start`, `leading`, and `stride` are byte counts. Each is stored as
    `(value & 0x3FFFF) >> 4`, in bits 0-13, 16-29, and 32-45. Bits 46-47
    hold the version, which is 1 on Blackwell. `base_offset` is in bits
    49-51. The layout type is 3 bits wide, at bits 61-63. `layout` is a key
    of LAYOUT_TYPE.
    """
    raise NotImplementedError


def per_cta_operand_bytes(m_cta: int, n: int, k: int, cta_group: int, elem_bytes: int) -> tuple[int, int]:
    """Bytes of A and of B that one CTA stages in shared memory per K slice.

    `m_cta` is the number of accumulator rows each CTA owns, 128 in the
    standard tile. With `cta_group` 1 the CTA stages its own `m_cta` rows of
    A and all `n` rows of B. With `cta_group` 2 the pair computes
    `2 * m_cta` by `n`: each CTA stages its own `m_cta` rows of A and half of
    the `n` rows of B. Return `(a_bytes, b_bytes)`.
    """
    raise NotImplementedError


def operand_intensity(m_cta: int, n: int, cta_group: int, elem_bytes: int) -> float:
    """FLOP per shared-memory byte staged, per CTA. Independent of K.

    One CTA's share of the MMA is `2 * m_cta * n * k` FLOP.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# The B200-only kernel. Written for you. It runs only when a compute
# capability 10.0 GPU is visible, and it is never compiled for sm_80.
# ---------------------------------------------------------------------------


def _skip_reason() -> str | None:
    try:
        import torch
    except ImportError:
        return "SKIP: the tcgen05 kernel was not run. torch is not installed."
    if not torch.cuda.is_available():
        return (
            "SKIP: the tcgen05 kernel was not run. "
            "No CUDA GPU is visible. It needs a B200 (sm_100a) and is not sent to the A100 endpoint."
        )
    if tuple(torch.cuda.get_device_capability()) != (10, 0):
        return "SKIP: the tcgen05 kernel was not run. The visible GPU is not sm_100."
    return None


def tmem_roundtrip(x):
    """Store a (128, 32) fp32 tensor into TMEM with tcgen05.st, read it back
    with tcgen05.ld, and return `(out, base_address)`.

    On a machine without a B200, print a skip message and return None. When
    you have a B200, port the kernel from the chapter: warp 0 allocates 32
    columns, each warp adds (32 * (warp % 4)) << 16 to the base, and warp 0
    frees the columns. Compile with `load_cuda(..., arch="sm_100a")`.
    """
    reason = _skip_reason()
    if reason is not None:
        print(reason, flush=True)
        return None
    raise NotImplementedError
