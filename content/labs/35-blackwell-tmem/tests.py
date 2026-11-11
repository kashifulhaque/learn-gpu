"""Tests for lab 35.

The CPU checks recompute the tensor-memory allocation rule, the address
format, the warp-to-lane rule, the accumulator footprint, both descriptors,
and the per-CTA operand traffic. They need no GPU. The tcgen05 kernel needs
a B200 (sm_100a). A machine without one skips it. A skipped kernel is not a
failure, and the job is not sent to the A100 endpoint.

Plausible wrong implementations, each named in a comment below:

multiple_of_32_not_power_of_two rounds an allocation request up to a
multiple of 32. 129 columns becomes 160, but tcgen05.alloc takes a power of
two, so it is 256.

swapped_lane_column packs the address as (column << 16) | lane. The lane
is in bits 31:16 and the column is in bits 15:0.

warp_id_without_mod_4 sends warp w to lanes 32*w to 32*w + 31. Warp 4 of an
eight-warp CTA reaches lanes 0 to 31, the same group as warp 0.

swapped_m_n_fields reads M >> 3 and N >> 4 instead of N >> 3 and M >> 4.

forgot_version_bit and layout_type_at_bit_62 are the two mistakes of a
packer copied from Hopper's wgmma descriptor. It leaves the version at 0
and shifts the layout type by 62 instead of 61.
"""

from __future__ import annotations

from lab_common import Checks

NAMES = (
    "tmem_alloc_columns",
    "TmemAllocator",
    "tmem_address",
    "decode_tmem_address",
    "warp_lane_range",
    "can_access",
    "accumulator_footprint",
    "accumulator_stages",
    "instr_descriptor",
    "smem_descriptor",
    "per_cta_operand_bytes",
    "operand_intensity",
    "tmem_roundtrip",
)


def _pow2_columns(requested: int) -> int:
    columns = 32
    while columns < requested:
        columns *= 2
    return columns


def _enc(value: int) -> int:
    return (value & 0x3FFFF) >> 4


def _smem_descriptor(start: int, leading: int, stride: int, layout_type: int) -> int:
    return (
        _enc(start)
        | (_enc(leading) << 16)
        | (_enc(stride) << 32)
        | (1 << 46)
        | (layout_type << 61)
    )


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


def _fail_exc(want, exc):
    return False, f"expected {want}, actual {type(exc).__name__}: {exc}"


def run(submission):
    c = Checks()
    if not c.require(submission, *NAMES):
        return c.finish()
    reader = submission

    # Check 1 calls only tmem_alloc_columns. Implementing that one function passes it.
    def rounding():
        # multiple_of_32_not_power_of_two fails here: 129 is 256, not 160.
        for requested in (1, 32, 33, 64, 100, 129, 256, 257, 512):
            want = _pow2_columns(requested)
            try:
                got = reader.tmem_alloc_columns(requested)
            except Exception as exc:  # noqa: BLE001
                return _fail_exc(f"{want} for {requested}", exc)
            if got != want:
                return False, f"expected {want} columns for a request of {requested}, actual {got}"
        for bad in (0, 513):
            try:
                reader.tmem_alloc_columns(bad)
            except ValueError:
                continue
            except Exception as exc:  # noqa: BLE001
                return _fail_exc(f"ValueError for {bad}", exc)
            return False, f"expected ValueError for a request of {bad}, actual no error"
        return True, "expected 256 for a request of 129, actual 256"

    c.check("an allocation is a power of two, at least 32 columns", rounding)

    def allocator():
        try:
            a = reader.TmemAllocator()
            first = a.alloc(160)
            second = a.alloc(160)
        except Exception as exc:  # noqa: BLE001
            return _fail_exc("two 256-column allocations", exc)
        spans = sorted([(first, first + 256), (second, second + 256)])
        if spans[0][0] < 0 or spans[1][1] > 512 or spans[0][1] > spans[1][0]:
            return False, f"expected two disjoint 256-column runs inside 512, actual bases {first} and {second}"
        if a.free_columns() != 0:
            return False, f"expected 0 free columns after 2 x 256, actual {a.free_columns()}"
        try:
            a.alloc(32)
        except MemoryError:
            pass
        except Exception as exc:  # noqa: BLE001
            return _fail_exc("MemoryError on a full allocator", exc)
        else:
            return False, "expected MemoryError for a 32-column request on a full TMEM, actual no error"
        a.free(first)
        if a.free_columns() != 256:
            return False, f"expected 256 free columns after one free, actual {a.free_columns()}"
        try:
            reuse = a.alloc(33)
        except Exception as exc:  # noqa: BLE001
            return _fail_exc("a 64-column allocation after a free", exc)
        if not (spans[0][0] == first and first <= reuse and reuse + 64 <= first + 256):
            return False, f"expected a base inside the freed run starting at {first}, actual {reuse}"
        try:
            a.free(reuse + 1)
        except ValueError:
            pass
        else:
            return False, "expected ValueError when freeing a column that is not a base, actual no error"
        b = reader.TmemAllocator()
        try:
            b.alloc(512)
            ok = b.free_columns() == 0
        except Exception as exc:  # noqa: BLE001
            return _fail_exc("one 512-column allocation", exc)
        return ok, "expected 0 free columns after 512, actual " + str(b.free_columns())

    c.check("the allocator tracks capacity and fails when TMEM is full", allocator)

    def addresses():
        # swapped_lane_column fails here: lane 32, column 5 is 0x200005, not 0x50020.
        cases = ((0, 0, 0x0), (32, 5, 0x200005), (127, 511, 0x7F01FF), (96, 256, 0x600100))
        for lane, column, want in cases:
            try:
                got = reader.tmem_address(lane, column)
            except Exception as exc:  # noqa: BLE001
                return _fail_exc(f"{want:#x} for lane {lane} column {column}", exc)
            if got != want:
                return False, f"expected {want:#x} for lane {lane} column {column}, actual {got:#x}"
            try:
                back = tuple(reader.decode_tmem_address(want))
            except Exception as exc:  # noqa: BLE001
                return _fail_exc(f"{(lane, column)} for {want:#x}", exc)
            if back != (lane, column):
                return False, f"expected {(lane, column)} for {want:#x}, actual {back}"
        for lane, column in ((128, 0), (0, 512)):
            try:
                reader.tmem_address(lane, column)
            except ValueError:
                continue
            except Exception as exc:  # noqa: BLE001
                return _fail_exc(f"ValueError for lane {lane} column {column}", exc)
            return False, f"expected ValueError for lane {lane} column {column}, actual no error"
        return True, "expected 0x200005 for lane 32, column 5, actual 0x200005"

    c.check("the lane is in bits 31:16 and the column in bits 15:0", addresses)

    def warps():
        # warp_id_without_mod_4 fails here: warp 4 reaches lanes 0-31, not 128-159.
        cases = ((0, (0, 31)), (1, (32, 63)), (3, (96, 127)), (4, (0, 31)), (5, (32, 63)), (7, (96, 127)))
        for warp, want in cases:
            try:
                got = tuple(reader.warp_lane_range(warp))
            except Exception as exc:  # noqa: BLE001
                return _fail_exc(f"{want} for warp {warp}", exc)
            if got != want:
                return False, f"expected lanes {want} for warp {warp}, actual {got}"
        for warp, lane, want in ((5, 40, True), (5, 8, False), (4, 8, True), (2, 96, False)):
            try:
                got = reader.can_access(warp, lane)
            except Exception as exc:  # noqa: BLE001
                return _fail_exc(f"{want} for warp {warp} lane {lane}", exc)
            if bool(got) != want:
                return False, f"expected can_access({warp}, {lane}) to be {want}, actual {got}"
        return True, "expected lanes 32 to 63 for warp 5, actual (32, 63)"

    c.check("warp w reaches lanes 32*(w % 4) to 32*(w % 4) + 31", warps)

    def footprint():
        cases = (
            ((64, 256, 1), (64, 256, 256)),
            ((128, 256, 1), (128, 256, 256)),
            ((128, 160, 1), (128, 160, 256)),
            ((128, 256, 2), (64, 256, 256)),
            ((256, 256, 2), (128, 256, 256)),
            ((256, 64, 2), (128, 64, 64)),
        )
        for args, want in cases:
            try:
                got = tuple(reader.accumulator_footprint(*args))
            except Exception as exc:  # noqa: BLE001
                return _fail_exc(f"{want} for M, N, group {args}", exc)
            if got != want:
                return False, f"expected (lanes, columns, allocated) {want} for M, N, group {args}, actual {got}"
        for args in ((256, 128, 1), (64, 128, 2), (128, 260, 1), (128, 100, 1)):
            try:
                reader.accumulator_footprint(*args)
            except ValueError:
                continue
            except Exception as exc:  # noqa: BLE001
                return _fail_exc(f"ValueError for {args}", exc)
            return False, f"expected ValueError for M, N, group {args}, actual no error"
        return True, "expected (128, 256, 256) for M=256, N=256, cta_group 2, actual (128, 256, 256)"

    c.check("an accumulator takes N columns and M / cta_group lanes in each CTA", footprint)

    def stages():
        # Rounding each allocation to a power of two loses a stage: 160 columns
        # becomes 256, so two fit. Carving one 512-column allocation fits three.
        cases = (((256, False), 2), ((256, True), 2), ((160, False), 2), ((160, True), 3),
                 ((96, False), 4), ((96, True), 5), ((64, False), 8))
        for (n, carve), want in cases:
            try:
                got = reader.accumulator_stages(n, carve)
            except Exception as exc:  # noqa: BLE001
                return _fail_exc(f"{want} for N={n}, carve={carve}", exc)
            if got != want:
                return False, f"expected {want} stages for N={n}, carve={carve}, actual {got}"
        return True, "expected 2 stages for N=160 allocated separately and 3 when carved, actual 2 and 3"

    c.check("the number of accumulator stages follows the allocation rule", stages)

    def instruction():
        # swapped_m_n_fields fails here: M=128, N=256 is 0x8400490.
        got_args = (128, 256, "bf16", "bf16", "f32", "K", "K")
        cases = (
            (got_args, 0x8400490),
            ((64, 128, "f16", "f16", "f16", "K", "K"), (16 << 17) | (4 << 24)),
            ((128, 256, "bf16", "bf16", "f32", "MN", "K"), 0x8400490 | (1 << 15)),
            ((128, 256, "bf16", "bf16", "f32", "K", "MN"), 0x8400490 | (1 << 16)),
        )
        for args, want in cases:
            try:
                got = reader.instr_descriptor(*args)
            except Exception as exc:  # noqa: BLE001
                return _fail_exc(f"{want:#x} for {args}", exc)
            if got != want:
                return False, f"expected {want:#x} for {args}, actual {got:#x}"
        return True, "expected 0x8400490 for bf16 in, fp32 out, M=128, N=256, actual 0x8400490"

    c.check("the instruction descriptor packs N >> 3 at bit 17 and M >> 4 at bit 24", instruction)

    def smem():
        # forgot_version_bit and layout_type_at_bit_62 fail here.
        cases = (
            ((0, 16, 1024, "128B"), _smem_descriptor(0, 16, 1024, 2)),
            ((0x4000, 16, 1024, "64B"), _smem_descriptor(0x4000, 16, 1024, 4)),
            ((0, 1, 512, "32B"), _smem_descriptor(0, 1, 512, 6)),
            ((1024, 0, 256, "none"), _smem_descriptor(1024, 0, 256, 0)),
            ((0, 16, 1024, "128B_BASE32B"), _smem_descriptor(0, 16, 1024, 1)),
        )
        for args, want in cases:
            try:
                got = reader.smem_descriptor(*args)
            except Exception as exc:  # noqa: BLE001
                return _fail_exc(f"{want:#x} for {args}", exc)
            if got != want:
                return False, f"expected {want:#x} for {args}, actual {got:#x}"
        try:
            got = reader.smem_descriptor(0, 16, 1024, "128B", 3)
        except Exception as exc:  # noqa: BLE001
            return _fail_exc("a descriptor with base offset 3", exc)
        want = _smem_descriptor(0, 16, 1024, 2) | (3 << 49)
        if got != want:
            return False, f"expected {want:#x} with base offset 3, actual {got:#x}"
        return True, "expected 0x4000404000010000 for 128B, actual 0x4000404000010000"

    c.check("the shared-memory descriptor has a version bit and a 3-bit layout type at bit 61", smem)

    def traffic():
        cases = (
            ((128, 256, 64, 1, 2), (16384, 32768)),
            ((128, 256, 64, 2, 2), (16384, 16384)),
            ((128, 128, 64, 2, 1), (8192, 4096)),
        )
        for args, want in cases:
            try:
                got = tuple(reader.per_cta_operand_bytes(*args))
            except Exception as exc:  # noqa: BLE001
                return _fail_exc(f"{want} for {args}", exc)
            if got != want:
                return False, f"expected (A, B) bytes {want} for {args}, actual {got}"
        for args, want in (((128, 256, 1, 2), 256 / 3), ((128, 256, 2, 2), 128.0)):
            try:
                got = float(reader.operand_intensity(*args))
            except Exception as exc:  # noqa: BLE001
                return _fail_exc(f"{want:.2f} FLOP per byte for {args}", exc)
            if abs(got - want) > 1e-9:
                return False, f"expected {want:.2f} FLOP per byte for M_cta, N, group, bytes {args}, actual {got:.2f}"
        return True, "expected B bytes to fall from 32768 to 16384 per K=64 slice, actual 16384"

    c.check("a CTA pair halves each CTA's B traffic and leaves A alone", traffic)

    _blackwell_section(c, reader)
    return c.finish()


def _blackwell_section(c: Checks, reader) -> None:
    """Kernel check. Skipped, not failed, when this machine has no B200."""
    reason = _skip_reason()
    if reason is not None:
        print(reason, flush=True)
        c.metric("tmem_values_roundtripped", "skipped (no B200)")
        return

    import torch

    torch.manual_seed(35)

    def roundtrip():
        x = torch.randn(128, 32, generator=torch.Generator().manual_seed(35)).cuda()
        try:
            result = reader.tmem_roundtrip(x)
        except Exception as exc:  # noqa: BLE001
            return _fail_exc("a (128, 32) tensor read back from TMEM", exc)
        if result is None:
            return False, "expected (out, base_address), actual None"
        out, base = result
        lane, column = base >> 16, base & 0xFFFF
        if lane != 0 or column + 32 > 512:
            return False, f"expected an allocation at lane 0 with 32 columns inside 512, actual lane {lane}, column {column}"
        if tuple(out.shape) != (128, 32):
            return False, f"expected shape (128, 32), actual {tuple(out.shape)}"
        bad = (out != x).nonzero(as_tuple=False)
        if bad.numel():
            r, col = int(bad[0][0]), int(bad[0][1])
            return False, f"expected {float(x[r, col]):.6f} at lane {r} column {col}, actual {float(out[r, col]):.6f}"
        c.metric("tmem_values_roundtripped", int(x.numel()))
        return True, f"expected 4096 values back unchanged, actual 4096 (allocated at column {column})"

    c.check("tcgen05.st and tcgen05.ld round-trip 128 lanes by 32 columns", roundtrip)
