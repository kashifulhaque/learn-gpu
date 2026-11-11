"""Lab 35: tensor memory, its addresses, and the descriptors behind tcgen05.

Do the CPU pieces first. They are the TMEM allocator, the address
format, the warp-to-lane rule, the accumulator footprint, the two
descriptors, and the per-CTA operand traffic. The last function is a
real sm_100a kernel stub. A machine without a B200 skips it.

1. tmem_alloc_columns
2. TmemAllocator
3. tmem_address, decode_tmem_address
4. warp_lane_range, can_access
5. accumulator_footprint, accumulator_stages
6. instr_descriptor
7. smem_descriptor
8. per_cta_operand_bytes, operand_intensity
9. tmem_roundtrip (B200 only)
"""

from __future__ import annotations

TMEM_LANES = 128
TMEM_COLUMNS = 512
MIN_ALLOC_COLUMNS = 32

# kind::f16 operand formats and accumulator formats, as stored in the
# instruction descriptor. Source: CUTLASS include/cute/arch/mma_sm100_desc.hpp.
AB_FORMAT = {"f16": 0, "bf16": 1, "tf32": 2}
C_FORMAT = {"f16": 0, "f32": 1, "s32": 2}
MAJOR = {"K": 0, "MN": 1}

# Shared-memory descriptor layout types on Blackwell. These are not the
# Hopper values: the field is 3 bits wide, at bits 61-63.
LAYOUT_TYPE = {"none": 0, "128B_BASE32B": 1, "128B": 2, "64B": 4, "32B": 6}


def tmem_alloc_columns(requested: int) -> int:
    """Columns that `tcgen05.alloc` actually reserves for a request.

    The count is a power of two, at least 32 and at most 512. Round a
    request up to the next such value. Raise ValueError when the request
    is below 1 or above 512.
    """
    if requested < 1 or requested > TMEM_COLUMNS:
        raise ValueError(f"requested {requested} columns, but a request is 1 to {TMEM_COLUMNS}")
    # multiple_of_32_not_power_of_two: rounding 129 up to 160 is a legal-looking
    # answer. The hardware rule is a power of two, so 129 becomes 256.
    columns = 1 << (requested - 1).bit_length()
    return max(columns, MIN_ALLOC_COLUMNS)


class TmemAllocator:
    """A model of one SM's tensor memory: 512 columns, all 128 lanes of each.

    `alloc(n)` reserves `tmem_alloc_columns(n)` contiguous columns at the
    lowest free column and returns that base column. It raises MemoryError
    when no run of free columns is long enough. `free(base)` releases the
    allocation that starts at `base` and raises ValueError for any other
    column. `free_columns()` is the number of unreserved columns.

    The placement policy is a model. On a B200 the hardware writes the base
    address to shared memory, and a kernel reads it from there.
    """

    def __init__(self) -> None:
        self._used: dict[int, int] = {}

    def alloc(self, requested: int) -> int:
        columns = tmem_alloc_columns(requested)
        base = 0
        for start in sorted(self._used):
            if start - base >= columns:
                break
            base = max(base, start + self._used[start])
        if base + columns > TMEM_COLUMNS:
            raise MemoryError(f"need {columns} columns, {self.free_columns()} free in {TMEM_COLUMNS}")
        self._used[base] = columns
        return base

    def free(self, base: int) -> None:
        if base not in self._used:
            raise ValueError(f"column {base} is not the base of a live allocation")
        del self._used[base]

    def free_columns(self) -> int:
        return TMEM_COLUMNS - sum(self._used.values())


def tmem_address(lane: int, column: int) -> int:
    """32-bit tensor-memory address: lane in bits 31:16, column in bits 15:0."""
    if not 0 <= lane < TMEM_LANES:
        raise ValueError(f"lane {lane} is outside 0 to {TMEM_LANES - 1}")
    if not 0 <= column < TMEM_COLUMNS:
        raise ValueError(f"column {column} is outside 0 to {TMEM_COLUMNS - 1}")
    # swapped_lane_column: (column << 16) | lane puts the fields the wrong way round.
    return (lane << 16) | column


def decode_tmem_address(address: int) -> tuple[int, int]:
    """Split an address into `(lane, column)`. Raise ValueError outside the array."""
    lane, column = address >> 16, address & 0xFFFF
    if lane >= TMEM_LANES or column >= TMEM_COLUMNS:
        raise ValueError(f"address {address:#x} decodes to lane {lane}, column {column}, outside the array")
    return lane, column


def warp_lane_range(warp_id: int) -> tuple[int, int]:
    """First and last lane that warp `warp_id` of the CTA can reach.

    Warp w reaches lanes 32*(w % 4) to 32*(w % 4) + 31. Warps 0 and 4 share
    a lane group, and so do warps 1 and 5.
    """
    # warp_id_without_mod_4: 32 * warp_id sends warp 4 to lanes 128-159.
    first = 32 * (warp_id % 4)
    return first, first + 31


def can_access(warp_id: int, lane: int) -> bool:
    """True when warp `warp_id` may touch `lane` with tcgen05.ld or tcgen05.st."""
    first, last = warp_lane_range(warp_id)
    return first <= lane <= last


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
    allowed = {1: (64, 128), 2: (128, 256)}
    if cta_group not in allowed:
        raise ValueError(f"cta_group is 1 or 2, not {cta_group}")
    if m not in allowed[cta_group]:
        raise ValueError(f"M of {m} is not valid for cta_group {cta_group}: use {allowed[cta_group]}")
    if not (8 <= n <= 256 and n % 8 == 0):
        raise ValueError(f"N of {n} is not a multiple of 8 from 8 to 256")
    return m // cta_group, n, tmem_alloc_columns(n)


def accumulator_stages(n: int, carve: bool = False) -> int:
    """How many N-column accumulators fit in one SM's 512 columns.

    With `carve` false, every accumulator is its own `tcgen05.alloc`, so it
    takes `tmem_alloc_columns(n)` columns. With `carve` true, a kernel
    allocates all 512 columns once and places accumulators at column
    offsets, so each one takes exactly `n`.
    """
    per = n if carve else tmem_alloc_columns(n)
    return TMEM_COLUMNS // per


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
    stay 0.
    """
    # swapped_m_n_fields: (m >> 3) << 17 | (n >> 4) << 24 reads the fields the wrong way round.
    word = saturate << 3
    word |= C_FORMAT[c_format] << 4
    word |= AB_FORMAT[a_format] << 7
    word |= AB_FORMAT[b_format] << 10
    word |= a_negate << 13
    word |= b_negate << 14
    word |= MAJOR[a_major] << 15
    word |= MAJOR[b_major] << 16
    word |= (n >> 3) << 17
    word |= (m >> 4) << 24
    return word


def smem_descriptor(start: int, leading: int, stride: int, layout: str, base_offset: int = 0) -> int:
    """64-bit shared-memory descriptor for tcgen05.

    `start`, `leading`, and `stride` are byte counts. Each is stored as
    `(value & 0x3FFFF) >> 4`, in bits 0-13, 16-29, and 32-45. Bits 46-47
    hold the version, which is 1 on Blackwell. `base_offset` is in bits
    49-51. The layout type is 3 bits wide, at bits 61-63. `layout` is a key
    of LAYOUT_TYPE.
    """

    def enc(value: int) -> int:
        return (value & 0x3FFFF) >> 4

    # forgot_version_bit: a Hopper packer leaves bits 46-47 at 0.
    # layout_type_at_bit_62: a Hopper packer shifts the layout type by 62, so the
    # 128-byte mode (2) lands at bit 63, which is the 64-byte mode.
    return (
        enc(start)
        | (enc(leading) << 16)
        | (enc(stride) << 32)
        | (1 << 46)
        | (base_offset << 49)
        | (LAYOUT_TYPE[layout] << 61)
    )


def per_cta_operand_bytes(m_cta: int, n: int, k: int, cta_group: int, elem_bytes: int) -> tuple[int, int]:
    """Bytes of A and of B that one CTA stages in shared memory per K slice.

    `m_cta` is the number of accumulator rows each CTA owns, 128 in the
    standard tile. With `cta_group` 1 the CTA stages its own `m_cta` rows of
    A and all `n` rows of B. With `cta_group` 2 the pair computes
    `2 * m_cta` by `n`: each CTA stages its own `m_cta` rows of A and half of
    the `n` rows of B. Return `(a_bytes, b_bytes)`.
    """
    a_bytes = m_cta * k * elem_bytes
    b_rows = n if cta_group == 1 else n // 2
    return a_bytes, b_rows * k * elem_bytes


def operand_intensity(m_cta: int, n: int, cta_group: int, elem_bytes: int) -> float:
    """FLOP per shared-memory byte staged, per CTA. Independent of K.

    One CTA's share of the MMA is `2 * m_cta * n * k` FLOP.
    """
    a_bytes, b_bytes = per_cta_operand_bytes(m_cta, n, 1, cta_group, elem_bytes)
    return 2 * m_cta * n / (a_bytes + b_bytes)


# ---------------------------------------------------------------------------
# The B200-only kernel. Everything below runs only when a compute capability
# 10.0 GPU is visible. Nothing here is compiled for sm_80.
# ---------------------------------------------------------------------------


def _skip_reason() -> str | None:
    try:
        import torch
    except ImportError:
        return (
            "SKIP: the tcgen05 kernel was not run. "
            "torch is not installed, so there is no B200 to compile it on."
        )
    if not torch.cuda.is_available():
        return (
            "SKIP: the tcgen05 kernel was not run. "
            "No CUDA GPU is visible. It needs a B200 (sm_100a) and is not sent to the A100 endpoint."
        )
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (10, 0):
        return (
            "SKIP: the tcgen05 kernel was not run. "
            f"The visible GPU is sm_{major}{minor}, not sm_100. "
            "It is not sent to the A100 endpoint."
        )
    return None


_CPP = """
std::vector<torch::Tensor> tmem_roundtrip_cuda(torch::Tensor x);
"""

# One CTA of four warps. Warp 0 allocates 32 columns, and all four warps
# write a value into every lane they own, read it back, and release it.
# Warp w reaches lanes 32*(w % 4) to 32*(w % 4) + 31, so the address of a
# warp adds (32 * (w % 4)) << 16 to the base. Not run on this Mac. It
# compiles with arch="sm_100a".
_CUDA = r"""
#include <cstdint>
#include <vector>

constexpr int kCols = 32;

__global__ void tmem_roundtrip_kernel(const float* in, float* out, uint32_t* base_out) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
  __shared__ uint32_t tmem_slot;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;

  if (warp == 0) {
    uint32_t slot = static_cast<uint32_t>(__cvta_generic_to_shared(&tmem_slot));
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(slot), "r"(kCols) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

  const uint32_t base = tmem_slot;
  if (threadIdx.x == 0) base_out[0] = base;

  const uint32_t lane_base = static_cast<uint32_t>((warp % 4) * 32) << 16;
  const int row = warp * 32 + lane;
  for (int c = 0; c < kCols; ++c) {
    const uint32_t taddr = base + lane_base + c;
    const uint32_t v = __float_as_uint(in[row * kCols + c]);
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};"
                 :: "r"(taddr), "r"(v) : "memory");
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    uint32_t r;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
                 : "=r"(r) : "r"(taddr) : "memory");
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    out[row * kCols + c] = __uint_as_float(r);
  }

  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  if (warp == 0) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(base), "r"(kCols) : "memory");
  }
#else
  __trap();
#endif
}

std::vector<torch::Tensor> tmem_roundtrip_cuda(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32, "x must be fp32 on CUDA");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 128 && x.size(1) == kCols, "x must be (128, 32)");
  x = x.contiguous();
  auto out = torch::empty_like(x);
  auto base = torch::zeros({1}, x.options().dtype(torch::kInt32));
  tmem_roundtrip_kernel<<<1, 128>>>(x.data_ptr<float>(), out.data_ptr<float>(),
                                    reinterpret_cast<uint32_t*>(base.data_ptr<int32_t>()));
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  return {out, base};
}
"""

_MODULE = None


def _module():
    global _MODULE
    if _MODULE is None:
        from lab_common import load_cuda

        _MODULE = load_cuda(
            "blackwell_tmem_35",
            _CUDA,
            _CPP,
            ["tmem_roundtrip_cuda"],
            arch="sm_100a",
        )
    return _MODULE


def tmem_roundtrip(x):
    """Store a (128, 32) fp32 tensor into TMEM with tcgen05.st, read it back
    with tcgen05.ld, and return `(out, base_address)`.

    On a machine without a B200, print a skip message and return None.
    """
    reason = _skip_reason()
    if reason is not None:
        print(reason, flush=True)
        return None
    out, base = _module().tmem_roundtrip_cuda(x)
    return out, int(base.item()) & 0xFFFFFFFF
