"""Tests for lab 26.

The CPU checks recompute the tensor-map address and the 128-byte swizzle.
They do not need a GPU. The TMA copy and the wgmma kernel need an H100
(sm_90a). A machine without one skips those checks. A skipped kernel is
not a failure, and the job is not sent to the A100 endpoint.

forgot_box_origin is a plausible wrong box_load. It uses the local index
as the global index. On the 3 by 5 matrix below, the box at column 3, row
1 should start at 9. The wrong load returns 1.

xor_row_into_low_byte is a plausible wrong swizzle_128b_byte. It XORs
(row & 7) into bits 0-2 of the byte offset. The 128-byte mode XORs that
field into bits 4-6, the 16-byte chunk index. Row 1, chunk 1 is byte 128.
The wrong XOR returns 145.

driver_enum_as_descriptor is a plausible wrong wgmma_descriptor. It stores
3 in bits 62-63 because CU_TENSOR_MAP_SWIZZLE_128B is the fourth driver
enumerator. The wgmma layout field stores the 128-byte mode as 1.
"""

from __future__ import annotations

from lab_common import Checks

NAMES = (
    "box_byte_offset",
    "box_load",
    "swizzle_128b_byte",
    "choose_swizzle",
    "expect_tx_bytes",
    "wgmma_descriptor",
    "predicted_ms",
    "tma_copy",
    "wgmma_64x8x16",
)

# A 3 by 5 matrix. The box at column 3, row 1 hangs off the right edge.
MATRIX = (
    (1, 2, 3, 4, 5),
    (6, 7, 8, 9, 10),
    (11, 12, 13, 14, 15),
)

# PTX ISA 128-byte swizzle, 16-byte atomicity, rows 0 through 7.
PTX_ROW1 = (1, 0, 3, 2, 5, 4, 7, 6)


def _enc(value: int) -> int:
    return (value & 0x3FFFF) >> 4


def _descriptor(start: int, leading: int, stride: int, layout_type: int) -> int:
    return _enc(start) | (_enc(leading) << 16) | (_enc(stride) << 32) | (layout_type << 62)


def _floor_ms(m: int, n: int, k: int) -> float:
    flops = 2 * m * n * k
    nbytes = 2 * (m * k + n * k + m * n)
    peak = 989.4e12
    bandwidth = 3350e9
    if flops / nbytes >= peak / bandwidth:
        return flops / peak * 1e3
    return nbytes / bandwidth * 1e3


def _skip_reason() -> str | None:
    try:
        import torch
    except ImportError:
        return (
            "SKIP: the Hopper TMA copy and the wgmma kernel were not run. "
            "torch is not installed, so there is no H100 to compile them on."
        )
    if not torch.cuda.is_available():
        return (
            "SKIP: the Hopper TMA copy and the wgmma kernel were not run. "
            "No CUDA GPU is visible. They need an H100 (sm_90a) and are not sent to the A100 endpoint."
        )
    major, minor = torch.cuda.get_device_capability()
    if major < 9:
        return (
            "SKIP: the Hopper TMA copy and the wgmma kernel were not run. "
            f"The visible GPU is sm_{major}{minor}, not sm_90. "
            "They are not sent to the A100 endpoint."
        )
    return None


def run(submission):
    c = Checks()
    if not c.require(submission, *NAMES):
        return c.finish()
    reader = submission

    # Check 1 calls only box_byte_offset. Implementing that one function passes it.
    def one_box():
        cases = (
            ((4, 2, 0, 0, 2, 20), 48),
            ((4, 2, 3, 1, 2, 20), 74),
        )
        for args, want in cases:
            try:
                got = reader.box_byte_offset(*args)
            except Exception as exc:  # noqa: BLE001
                return False, f"expected {want}, actual {type(exc).__name__}: {exc}"
            if got != want:
                return False, f"expected {want}, actual {got}"
        return True, "expected 48 and 74, actual 48 and 74"

    c.check("the box origin is part of the byte offset", one_box)

    def load_box():
        # forgot_box_origin fails here: local (0, 0) is 9, not 1.
        want = ((9, 10, 0), (14, 15, 0))
        try:
            got = reader.box_load(MATRIX, 3, 1, 3, 2)
        except Exception as exc:  # noqa: BLE001
            return False, f"expected {want[0][0]} at local row 0 col 0, actual {type(exc).__name__}: {exc}"
        try:
            rows = [tuple(line) for line in got]
        except TypeError as exc:
            return False, f"expected {want}, actual {type(exc).__name__}: {exc}"
        if tuple(rows) != want:
            shown = rows[0][0] if rows and rows[0] else rows
            return False, f"expected 9 at local row 0 col 0, actual {shown}"
        return True, "expected 9 at local row 0 col 0, actual 9"

    c.check("a box that hangs off the matrix is filled with zeros", load_box)

    def swizzle_map():
        # xor_row_into_low_byte fails here: row 1 chunk 1 is 128, not 145.
        order = [(1, 1)] + [(row, chunk) for row in range(8) for chunk in range(8)]
        for row, chunk in order:
            want = row * 128 + (chunk ^ row) * 16
            try:
                got = reader.swizzle_128b_byte(row, chunk)
            except Exception as exc:  # noqa: BLE001
                return False, f"expected {want} at row {row} chunk {chunk}, actual {type(exc).__name__}: {exc}"
            if got != want:
                return False, f"expected {want} at row {row} chunk {chunk}, actual {got}"
        row1 = tuple(reader.swizzle_128b_byte(1, chunk) // 16 % 8 for chunk in range(8))
        # The byte's chunk field, printed so the PTX row is visible in the pass line.
        if row1 != PTX_ROW1:
            return False, f"expected row 1 chunks {PTX_ROW1}, actual {row1}"
        return True, "expected 128 at row 1 chunk 1, actual 128"

    c.check("the 128-byte swizzle matches the PTX 16-byte-atom table", swizzle_map)

    def modes():
        cases = (
            ((2, 64), "128B"),
            ((2, 32), "64B"),
            ((2, 16), "32B"),
            ((4, 8), "32B"),
            ((2, 48), "none"),
            ((2, 80), "none"),
            ((2, 3), "illegal"),
        )
        for args, want in cases:
            try:
                got = reader.choose_swizzle(*args)
            except Exception as exc:  # noqa: BLE001
                return False, f"expected {want} for {args}, actual {type(exc).__name__}: {exc}"
            if got != want:
                return False, f"expected {want} for elem {args[0]} inner {args[1]}, actual {got}"
        return True, "expected 128B for a 64-element bf16 box, actual 128B"

    c.check("the swizzle span equals the box's inner width", modes)

    def tx_count():
        want = 49152
        try:
            got = reader.expect_tx_bytes(((128, 64, 2), (256, 64, 2)))
        except Exception as exc:  # noqa: BLE001
            return False, f"expected {want}, actual {type(exc).__name__}: {exc}"
        if got != want:
            return False, f"expected {want}, actual {got}"
        return True, "expected 49152, actual 49152"

    c.check("expect_tx counts every byte of both bf16 tiles", tx_count)

    def descriptor():
        # driver_enum_as_descriptor fails here: layout_type 3 is not the 128-byte mode.
        cases = (
            ((0, 128, 1024, 1), _descriptor(0, 128, 1024, 1)),
            ((1024, 16, 256, 0), _descriptor(1024, 16, 256, 0)),
        )
        for args, want in cases:
            try:
                got = reader.wgmma_descriptor(*args)
            except Exception as exc:  # noqa: BLE001
                return False, f"expected {want:#x}, actual {type(exc).__name__}: {exc}"
            if got != want:
                return False, f"expected {want:#x}, actual {got:#x}"
        return True, "expected 0x4000004000080000 for the 128-byte mode, actual 0x4000004000080000"

    c.check("the descriptor stores the 128-byte mode as layout_type 1", descriptor)

    def floor():
        cases = ((4096, 4096, 4096), (128, 128, 64))
        for shape in cases:
            want = _floor_ms(*shape)
            try:
                got = float(reader.predicted_ms(*shape))
            except Exception as exc:  # noqa: BLE001
                return False, f"expected {want:.6e} ms for {shape}, actual {type(exc).__name__}: {exc}"
            if abs(got - want) > 1e-9:
                return False, f"expected {want:.6e} ms for {shape}, actual {got:.6e} ms"
        big = _floor_ms(4096, 4096, 4096)
        return True, f"expected {big:.4f} ms for 4096 cubed, actual {big:.4f} ms"

    c.check("the H100 SXM floor uses 989.4 TFLOP/s above the ridge and 3350 GB/s below it", floor)

    _hopper_section(c, reader)
    return c.finish()


def _hopper_section(c: Checks, reader) -> None:
    """Kernel checks. Skipped, not failed, when this machine has no H100."""
    reason = _skip_reason()
    if reason is not None:
        print(reason, flush=True)
        c.metric("tflops", "skipped (no H100)")
        return

    import torch

    from lab_common import cuda_time, tflops

    torch.manual_seed(26)

    def tma_cases():
        # Shifted 128B box: a wrong origin reads a different tile.
        rows, cols = 16, 128
        src = torch.zeros(rows, cols, dtype=torch.bfloat16, device="cuda")
        values = torch.tensor(
            [[(r * 3 + c) % 64 for c in range(cols)] for r in range(rows)],
            dtype=torch.float32,
            device="cuda",
        )
        src.copy_(values.to(torch.bfloat16))
        try:
            got = reader.tma_copy(src, 64, 8, 64, 8, "128B")
        except Exception as exc:  # noqa: BLE001
            return False, f"expected an 8 by 64 bf16 box, actual {type(exc).__name__}: {exc}"
        if got is None or tuple(got.shape) != (8, 64):
            shown = None if got is None else tuple(got.shape)
            return False, f"expected shape (8, 64), actual {shown}"
        want = values[8:16, 64:128].to(torch.bfloat16)
        mismatch = (got != want).nonzero(as_tuple=False)
        if mismatch.numel():
            at = mismatch[0]
            r, col = int(at[0]), int(at[1])
            return False, (
                f"expected {int(want[r, col])} at box row {r} col {col}, "
                f"actual {int(got[r, col])}"
            )
        # OOB box: columns 12..19 and rows 6..9 on an 8 by 16 matrix.
        small = torch.arange(8 * 16, device="cuda", dtype=torch.float32).reshape(8, 16) % 64
        small_bf = small.to(torch.bfloat16)
        try:
            edge = reader.tma_copy(small_bf, 12, 6, 8, 4, "none")
        except Exception as exc:  # noqa: BLE001
            return False, f"expected a 4 by 8 box with zeros, actual {type(exc).__name__}: {exc}"
        if edge is None or tuple(edge.shape) != (4, 8):
            shown = None if edge is None else tuple(edge.shape)
            return False, f"expected shape (4, 8), actual {shown}"
        for r in range(4):
            for col in range(8):
                gr, gc = 6 + r, 12 + col
                expected = 0 if gr >= 8 or gc >= 16 else int(small[gr, gc])
                actual = int(edge[r, col])
                if actual != expected:
                    return False, f"expected {expected} at box row {r} col {col}, actual {actual}"
        return True, "expected 24 at the shifted box origin, actual 24"

    c.check("the TMA copy matches the box, including zeros past the edge", tma_cases)

    def wgmma_case():
        a = torch.zeros(64, 16, dtype=torch.bfloat16, device="cuda")
        b = torch.zeros(16, 8, dtype=torch.bfloat16, device="cuda")
        rows = torch.arange(64, device="cuda", dtype=torch.float32)
        cols = torch.arange(8, device="cuda", dtype=torch.float32)
        a[:, 0] = rows.to(torch.bfloat16)
        a[:, 1] = torch.ones(64, device="cuda", dtype=torch.bfloat16)
        b[0, :] = torch.ones(8, device="cuda", dtype=torch.bfloat16)
        b[1, :] = cols.to(torch.bfloat16)
        try:
            got = reader.wgmma_64x8x16(a, b)
        except Exception as exc:  # noqa: BLE001
            return False, f"expected a 64 by 8 fp32 tensor, actual {type(exc).__name__}: {exc}"
        if got is None or tuple(got.shape) != (64, 8) or got.dtype != torch.float32:
            shown = None if got is None else (tuple(got.shape), str(got.dtype))
            return False, f"expected shape (64, 8) and fp32, actual {shown}"
        want = rows[:, None] + cols[None, :]
        err = (got - want).abs()
        flat = int(err.argmax())
        r, col = divmod(flat, 8)
        max_err = float(err[r, col])
        if max_err > 0:
            return False, (
                f"expected {float(want[r, col]):.1f} at row {r} col {col}, "
                f"actual {float(got[r, col]):.1f}"
            )
        return True, "expected 1.0 at row 1 col 0, actual 1.0"

    c.check("one m64n8k16 wgmma matches the fp32 reference", wgmma_case)

    def time_kernel():
        a = torch.ones(64, 16, dtype=torch.bfloat16, device="cuda")
        b = torch.ones(16, 8, dtype=torch.bfloat16, device="cuda")
        reader.wgmma_64x8x16(a, b)
        ms = cuda_time(lambda: reader.wgmma_64x8x16(a, b), warmup=5, iters=20)
        rate = tflops(2 * 64 * 8 * 16, ms)
        floor = _floor_ms(4096, 4096, 4096)
        c.metric("tflops", round(rate, 4))
        return True, (
            f"measured {rate:.4f} TFLOP/s for one m64n8k16 launch in {ms:.4f} ms. "
            f"The H100 SXM roofline floor for a 4096 cubed bf16 GEMM is {floor:.4f} ms. "
            "No pass/fail threshold: this run records the metric."
        )

    c.check("the lab records TFLOP/s for one wgmma and does not grade it", time_kernel)
