"""Tests for lab 11.

The CPU checks grade the m16n8k16 fragment map, the warp-tile register
count, and the bank conflicts of the ldmatrix addresses. They need no GPU.
The kernel checks are the original WMMA and mma.sync GEMM tests. They are
skipped, not failed, when this machine has no CUDA GPU.

swapped_rows is a plausible wrong fragment_cell. It swaps the two rows a
lane owns. For operand C, elements 0 and 1 use row g + 8, and elements 2
and 3 use row g. Lane 5, operand C, element 0 is (row, col) = (1, 2).
swapped_rows returns (9, 2).

unswizzled_ldmatrix is a plausible wrong ldmatrix_byte. It ignores the
8-row swizzle and keeps the logical column. Lane 1, stride 64, with the
swizzle on, addresses byte 144. unswizzled_ldmatrix returns 128. The eight
rows of matrix 0 then share banks 0 to 3, so the conflict degree is 8
where the swizzled addresses are 1.
"""

from lab_common import Checks

# Odd sizes exercise the padding; the multiples of 128 exercise full tiles.
SHAPES = [(1, 1, 1), (16, 16, 16), (17, 33, 19), (100, 200, 300), (128, 128, 32), (257, 129, 65), (512, 512, 512), (1000, 1000, 1000)]
PERF_SHAPE = (4096, 4096, 4096)


def _reference(a, b):
    # fp32 matmul of the bf16 values. TF32 is off by default in PyTorch, so this
    # is an fp32 GEMM on the CUDA cores: every bf16 product is exact in fp32.
    return a.float() @ b.float()


def _tolerance(ref):
    # Tensor-core accumulation order differs from the reference's, so allow a
    # small relative error. A wrong fragment index is off by O(1) of the scale.
    return 1e-3 * max(1.0, ref.abs().max().item())


# Lane 5 on A, from the chapter: g = 1, t = 1, so rows 1 and 9, columns
# 2 and 3, then columns 10 and 11. Element order is a0, a1, a2, a3.
_LANE5_A = ((1, 2), (1, 3), (9, 2), (9, 3), (1, 10), (1, 11), (9, 10), (9, 11))

# Register triples the chapter derives: (accumulators, A fragments, B fragments).
_REGISTERS = {
    (16, 8): (4, 4, 2),
    (64, 32): (64, 16, 8),
    (128, 64): (256, 32, 16),
}

NAMES = (
    "fragment_cell",
    "warp_tile_registers",
    "ldmatrix_byte",
    "wmma_gemm",
    "mma_gemm",
    "percent_of_torch",
)


def _fragment_cell(operand: str, lane: int, index: int) -> tuple[int, int]:
    """Independent copy of the m16n8k16 layout in the chapter."""
    group = lane // 4
    thread = lane % 4
    if operand == "A":
        row = group if index in (0, 1, 4, 5) else group + 8
        col = 2 * thread + (index % 2) + (8 if index >= 4 else 0)
        return row, col
    if operand == "B":
        row = 2 * thread + (index % 2) + (8 if index >= 2 else 0)
        return row, group
    row = group if index < 2 else group + 8
    return row, 2 * thread + (index % 2)


def _conflict_degree(byte_addresses: list[int]) -> int:
    """Largest number of distinct 4-byte words in any one of 32 banks.

    Each ldmatrix row is 16 bytes, four words. Bank = word offset mod 32.
    Two threads that read the same word broadcast, so a word is counted once.
    """
    per_bank: dict[int, set[int]] = {}
    for address in byte_addresses:
        base = address // 4
        for word in range(base, base + 4):
            per_bank.setdefault(word % 32, set()).add(word)
    return max(len(words) for words in per_bank.values())


def _show(expected, actual) -> str:
    return f"expected {expected}, actual {actual}"


def run(submission):
    c = Checks()
    if not c.require(submission, *NAMES):
        return c.finish()
    reader = submission
    _cpu_checks(c, reader)
    _gpu_checks(c, reader)
    return c.finish()


def _cpu_checks(c: Checks, reader) -> None:
    # Check 1 calls only fragment_cell. Implementing that one function passes it.
    def lane_five():
        for index, want in enumerate(_LANE5_A):
            try:
                got = tuple(reader.fragment_cell("A", 5, index))
            except Exception as exc:  # noqa: BLE001
                return False, _show(want, f"{type(exc).__name__}: {exc}") + f" for A lane 5 element {index}"
            if got != want:
                return False, _show(want, got) + f" for A lane 5 element {index}"
        return True, _show(_LANE5_A, _LANE5_A) + " for A lane 5"

    c.check("lane 5 holds rows 1 and 9 of the A fragment", lane_five)

    def every_cell():
        # swapped_rows fails here: C lane 5 element 0 is (1, 2), not (9, 2).
        cells = [("C", 5, 0)]
        for operand, count in (("A", 8), ("B", 4), ("C", 4)):
            for lane in range(32):
                for index in range(count):
                    if (operand, lane, index) != ("C", 5, 0):
                        cells.append((operand, lane, index))
        for operand, lane, index in cells:
            want = _fragment_cell(operand, lane, index)
            try:
                got = tuple(reader.fragment_cell(operand, lane, index))
            except Exception as exc:  # noqa: BLE001
                return False, _show(want, f"{type(exc).__name__}: {exc}") + (
                    f" for {operand} lane {lane} element {index}"
                )
            if got != want:
                return False, _show(want, got) + f" for {operand} lane {lane} element {index}"
        return True, _show((1, 2), (1, 2)) + " for C lane 5 element 0"

    c.check("every lane's A, B, and C elements match the fragment table", every_cell)

    def registers():
        for shape, want in _REGISTERS.items():
            try:
                got = tuple(reader.warp_tile_registers(*shape))
            except Exception as exc:  # noqa: BLE001
                return False, _show(want, f"{type(exc).__name__}: {exc}") + f" for a {shape[0]} by {shape[1]} tile"
            if got != want:
                return False, _show(want, got) + f" for a {shape[0]} by {shape[1]} tile"
        return True, _show((64, 16, 8), (64, 16, 8)) + " for a 64 by 32 tile"

    c.check("a warp tile's accumulator and fragment registers match the count", registers)

    def lane_one_byte():
        # unswizzled_ldmatrix fails the swizzled case: byte 128, not 144.
        cases = (
            (1, 64, False, 128),
            (1, 64, True, 144),
            (0, 40, False, 0),
            (9, 40, False, 720),
        )
        for lane, stride, swizzle, want in cases:
            try:
                got = reader.ldmatrix_byte(lane, stride, swizzle)
            except Exception as exc:  # noqa: BLE001
                return False, _show(want, f"{type(exc).__name__}: {exc}") + (
                    f" for lane {lane}, stride {stride}, swizzle {swizzle}"
                )
            if got != want:
                return False, _show(want, got) + f" for lane {lane}, stride {stride}, swizzle {swizzle}"
        return True, _show(144, 144) + " for lane 1, stride 64, swizzle on"

    c.check("ldmatrix addresses follow the lane rule and the 8-row swizzle", lane_one_byte)

    def banks():
        # unswizzled_ldmatrix makes the swizzled read 8-way instead of 1-way.
        cases = (
            (64, False, 8),
            (32, False, 4),
            (40, False, 1),
            (136, False, 1),
            (64, True, 1),
        )
        degrees = {}
        for stride, swizzle, want in cases:
            label = f"stride {stride}, swizzle {swizzle}"
            try:
                addresses = [reader.ldmatrix_byte(lane, stride, swizzle) for lane in range(8)]
                got = _conflict_degree(addresses)
            except Exception as exc:  # noqa: BLE001
                c.metric("conflicts_plain", "n/a")
                c.metric("conflicts_swizzled", "n/a")
                return False, _show(want, f"{type(exc).__name__}: {exc}") + f" for {label}"
            degrees[(stride, swizzle)] = got
            if got != want:
                c.metric("conflicts_plain", degrees.get((64, False), "n/a"))
                c.metric("conflicts_swizzled", degrees.get((64, True), "n/a"))
                return False, _show(want, got) + f" for {label}"
        plain = degrees[(64, False)]
        swizzled = degrees[(64, True)]
        c.metric("conflicts_plain", plain)
        c.metric("conflicts_swizzled", swizzled)
        return True, _show(f"plain {8}, swizzled {1}", f"plain {plain}, swizzled {swizzled}")

    c.check("ldmatrix rows of a 64-element stride are 8-way until the swizzle", banks)

    c.check(
        "percent_of_torch divides and scales by 100",
        lambda: (
            abs(reader.percent_of_torch(150.0, 200.0) - 75.0) < 1e-6,
            _show(75.0, reader.percent_of_torch(150.0, 200.0)),
        ),
    )


def _gpu_checks(c: Checks, reader) -> None:
    """Kernel checks. Skipped, not failed, when this machine cannot run them."""
    skipped = ("wmma_tflops", "mma_tflops", "torch_tflops", "wmma_pct_torch", "mma_pct_torch")
    try:
        import torch
    except ImportError:
        print(
            "SKIP: the WMMA and mma.sync kernels were not run. "
            "torch is not installed, so there is no CUDA GPU to compile them on.",
            flush=True,
        )
        for key in skipped:
            c.metric(key, "skipped (no torch)")
        return
    if not torch.cuda.is_available():
        print(
            "SKIP: the WMMA and mma.sync kernels were not run. No CUDA GPU is visible. "
            "The kernel checks need an A100 (sm_80).",
            flush=True,
        )
        for key in skipped:
            c.metric(key, "skipped (no GPU)")
        return

    from lab_common import A100, close, cuda_time, require_cuda, tflops

    device = require_cuda()
    torch.manual_seed(0)

    for name in ("wmma_gemm", "mma_gemm"):
        fn = getattr(reader, name)
        for m, n, k in SHAPES:
            a = torch.randn(m, k, device=device).to(torch.bfloat16)
            b = torch.randn(k, n, device=device).to(torch.bfloat16)
            ref = _reference(a, b)

            def check(fn=fn, a=a, b=b, ref=ref, m=m, n=n):
                out = fn(a, b)
                if out.dtype != torch.float32:
                    return False, f"output dtype {out.dtype}, expected torch.float32"
                if tuple(out.shape) != (m, n):
                    return False, f"output shape {tuple(out.shape)}, expected {(m, n)}"
                return close(out, ref, _tolerance(ref))

            c.check(f"{name} is correct for M={m}, N={n}, K={k}", check)

    # Structured input catches a transposed or swapped fragment that random data can hide:
    # A is a row index ramp, B is the identity, so C must equal A.
    def identity_check(fn):
        a = (torch.arange(64, device=device).float().view(64, 1) + torch.arange(64, device=device).float().view(1, 64) / 64).to(torch.bfloat16)
        eye = torch.eye(64, device=device, dtype=torch.bfloat16)
        return close(fn(a, eye), a.float(), 0.0)

    c.check("wmma_gemm times the identity returns A", lambda: identity_check(reader.wmma_gemm))
    c.check("mma_gemm times the identity returns A", lambda: identity_check(reader.mma_gemm))

    # Performance. The ceiling is the dense bf16 peak; the floor for "using the
    # tensor cores at all" is the fp32 CUDA-core peak, 19.5 TFLOP/s.
    m, n, k = PERF_SHAPE
    a = torch.randn(m, k, device=device).to(torch.bfloat16)
    b = torch.randn(k, n, device=device).to(torch.bfloat16)
    flops = 2.0 * m * n * k

    torch_ms = cuda_time(lambda: torch.matmul(a, b), warmup=5, iters=20)
    torch_tf = tflops(flops, torch_ms)
    c.metric("torch_tflops", round(torch_tf, 1))

    for name, key in (("wmma_gemm", "wmma"), ("mma_gemm", "mma")):
        fn = getattr(reader, name)
        try:
            ms = cuda_time(lambda fn=fn: fn(a, b), warmup=5, iters=20)
        except Exception as exc:  # noqa: BLE001
            c.add(f"{name} runs at 4096 x 4096 x 4096", False, f"{type(exc).__name__}: {exc}")
            continue
        tf = tflops(flops, ms)
        c.metric(f"{key}_tflops", round(tf, 1))
        c.metric(f"{key}_pct_torch", round(reader.percent_of_torch(tf, torch_tf), 1))
        c.add(
            f"{name} beats the {A100.FP32_TFLOPS} TFLOP/s fp32 CUDA-core peak",
            tf > A100.FP32_TFLOPS,
            f"{tf:.1f} TFLOP/s; the dense bf16 tensor-core peak is {A100.TENSOR_BF16_TFLOPS:.0f}. "
            "A figure below 19.5 means the tensor cores are not doing the work.",
        )
