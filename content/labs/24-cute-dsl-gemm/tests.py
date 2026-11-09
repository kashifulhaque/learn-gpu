"""Tests for lab 24.

The CPU checks reimplement the layout algebra from lab 13 and compare the
reader's TiledCopy partition to that algebra. They do not import the
reader's lab 13 solution. The GEMM checks need an A100 and
nvidia-cutlass-dsl, and they are skipped with a message when either is
missing. A skipped kernel check is not a failure.

striped_thread_partition is a plausible wrong implementation of
partition_tiled_copy. It assigns element `thread + value * num_threads`,
the grid-stride pattern from a scalar copy. For the 8 by 32 half tile with
thread-value layout ((4, 8), 8):((8, 32), 1), thread 1 value 0 owns offset
8. The striped map returns 1, so the partition check fails it.

row_major_fragment is a plausible wrong accumulator_mn. It returns
(lane mod 16, lane // 16) and ignores the value index. Lane 1 value 0 is
(m, n) = (0, 2). That guess returns (1, 0).

gelu_then_bias is a plausible wrong bias_gelu. It applies GELU, then adds
the bias. On value 1.0 and bias 0.5 the erf GELU of the sum is
1.399789198096713, and gelu_then_bias returns 1.341344746068543.
"""

from __future__ import annotations

import math
import random

from lab_common import A100, Checks

NAMES = (
    "tensor_offsets",
    "partition_tiled_copy",
    "accumulator_mn",
    "bias_gelu",
    "epilogue_bytes",
    "predicted_ms",
    "cute_gemm_bias_gelu",
)

# The 8 by 32 half-tile copy from chapter 13, and the m16n8k16 accumulator.
COPY_TV = (((4, 8), 8), ((8, 32), 1))
ACC_TV = (((4, 8), (2, 2)), ((32, 1), (16, 8)))
_SQRT2_INV = 1.0 / math.sqrt(2.0)


def _product(x) -> int:
    if isinstance(x, tuple):
        total = 1
        for item in x:
            total *= _product(item)
        return total
    return x


def _layout_at(shape, stride, coord) -> int:
    """Independent copy of the lab 13 coordinate rule, used as the reference."""
    if isinstance(shape, tuple):
        if isinstance(coord, tuple):
            return sum(_layout_at(s, d, c) for s, d, c in zip(shape, stride, coord))
        total = 0
        rest = coord
        last = len(shape) - 1
        for index, (mode, step) in enumerate(zip(shape, stride)):
            if index == last:
                total += _layout_at(mode, step, rest)
            else:
                count = _product(mode)
                total += _layout_at(mode, step, rest % count)
                rest //= count
        return total
    return coord * stride


def _reference_partition(tv_shape, tv_stride) -> list[list[int]]:
    threads = _product(tv_shape[0])
    values = _product(tv_shape[1])
    return [
        [_layout_at(tv_shape, tv_stride, (thread, value)) for value in range(values)]
        for thread in range(threads)
    ]


def _reference_mn(lane: int, value: int) -> tuple[int, int]:
    return ((lane // 4) + 8 * (value // 2), 2 * (lane % 4) + (value % 2))


def _gelu(x: float) -> float:
    return 0.5 * x * (1.0 + math.erf(x * _SQRT2_INV))


def _floor_ms(m: int, n: int, k: int) -> float:
    flops = 2 * m * n * k
    nbytes = 2 * (m * k + n * k + m * n)
    peak = A100.TENSOR_FP16_TFLOPS * 1e12
    bandwidth = A100.HBM_BW_GBS * 1e9
    if flops / nbytes >= peak / bandwidth:
        return flops / peak * 1e3
    return nbytes / bandwidth * 1e3


def _mismatch(got, want, what: str):
    if got == want:
        return True, f"expected {want}, actual {got}"
    return False, f"expected {want}, actual {got} ({what})"


def run(submission):
    c = Checks()
    if not c.require(submission, *NAMES):
        return c.finish()
    reader = submission

    # Check 1 calls only tensor_offsets. Implementing that one function passes it.
    def column_major():
        got = reader.tensor_offsets((4, 3), (1, 4))
        want = list(range(12))
        return _mismatch(got, want, "column-major (4,3):(1,4)")

    c.check("a column-major 4 by 3 layout lists offsets 0 through 11", column_major)

    def more_layouts():
        row = reader.tensor_offsets((4, 3), (3, 1))
        row_want = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
        if row != row_want:
            return False, f"expected {row_want}, actual {row} (row-major (4,3):(3,1))"
        hier = reader.tensor_offsets(((2, 2), 3), ((1, 6), 2))
        hier_want = [0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11]
        return _mismatch(hier, hier_want, "((2,2),3):((1,6),2)")

    c.check("row-major and hierarchical layouts match the colexicographic walk", more_layouts)

    def copy_partition():
        # striped_thread_partition fails here: thread 1 value 0 is 8, not 1.
        try:
            got = reader.partition_tiled_copy(*COPY_TV)
        except Exception as exc:  # noqa: BLE001
            return False, f"expected offset 8 at thread 1 value 0, actual {type(exc).__name__}: {exc}"
        want = _reference_partition(*COPY_TV)
        if got == want:
            return True, "expected offset 8 at thread 1 value 0, actual 8"
        threads = _product(COPY_TV[0][0])
        values = _product(COPY_TV[0][1])
        for thread in range(threads):
            for value in range(values):
                try:
                    actual = got[thread][value]
                except (IndexError, TypeError, KeyError) as exc:
                    return False, (
                        f"expected {want[thread][value]} at thread {thread} value {value}, "
                        f"actual {type(exc).__name__}: {exc}"
                    )
                if actual != want[thread][value]:
                    return False, (
                        f"expected {want[thread][value]} at thread {thread} value {value}, "
                        f"actual {actual}"
                    )
        return False, f"expected {len(want)} threads, actual {got!r}"

    c.check("the 8 by 32 TiledCopy partition matches the thread-value layout", copy_partition)

    def acc_partition():
        try:
            got = reader.partition_tiled_copy(*ACC_TV)
        except Exception as exc:  # noqa: BLE001
            return False, f"expected offset 32 at lane 1 value 0, actual {type(exc).__name__}: {exc}"
        want = _reference_partition(*ACC_TV)
        for lane in range(32):
            for value in range(4):
                try:
                    actual = got[lane][value]
                except (IndexError, TypeError, KeyError) as exc:
                    return False, (
                        f"expected {want[lane][value]} at lane {lane} value {value}, "
                        f"actual {type(exc).__name__}: {exc}"
                    )
                if actual != want[lane][value]:
                    return False, (
                        f"expected {want[lane][value]} at lane {lane} value {value}, actual {actual}"
                    )
        return True, "expected offset 32 at lane 1 value 0, actual 32"

    c.check("the m16n8k16 accumulator partition matches the thread-value layout", acc_partition)

    def lanes():
        # row_major_fragment fails here: lane 1 value 0 is (0, 2), not (1, 0).
        for lane in range(32):
            for value in range(4):
                want = _reference_mn(lane, value)
                try:
                    got = reader.accumulator_mn(lane, value)
                except Exception as exc:  # noqa: BLE001
                    return False, f"expected {want} at lane {lane} value {value}, actual {type(exc).__name__}: {exc}"
                if tuple(got) != want:
                    return False, f"expected {want} at lane {lane} value {value}, actual {tuple(got)}"
        return True, "expected (0, 2) at lane 1 value 0, actual (0, 2)"

    c.check("each lane's four accumulator cells match the fragment table", lanes)

    def gelu_values():
        # gelu_then_bias fails the hand case: expected about 1.3998, actual about 1.3413.
        hand_want = _gelu(1.5)
        try:
            hand = reader.bias_gelu([1.0], [0.5])
        except Exception as exc:  # noqa: BLE001
            return False, f"expected [{hand_want:.6f}] for input 1.0 bias 0.5, actual {type(exc).__name__}: {exc}"
        if len(hand) != 1 or abs(hand[0] - hand_want) > 1e-9:
            shown = hand[0] if hand else None
            return False, f"expected {hand_want:.12f} for input 1.0 bias 0.5, actual {shown}"
        rng = random.Random(24)
        values = [rng.uniform(-2.0, 2.0) for _ in range(16)]
        bias = [rng.uniform(-0.5, 0.5) for _ in range(16)]
        try:
            got = reader.bias_gelu(values, bias)
        except Exception as exc:  # noqa: BLE001
            return False, f"expected a length-16 GELU, actual {type(exc).__name__}: {exc}"
        want = [_gelu(v + b) for v, b in zip(values, bias)]
        if len(got) != len(want):
            return False, f"expected length {len(want)}, actual length {len(got)}"
        for index, (actual, expected) in enumerate(zip(got, want)):
            if abs(actual - expected) > 1e-9:
                return False, f"expected {expected:.6f} at index {index}, actual {actual:.6f}"
        return True, f"expected {hand_want:.6f} for input 1.0 bias 0.5, actual {hand[0]:.6f}"

    c.check("bias plus erf GELU matches the formula, including a seeded vector", gelu_values)

    def bytes_check():
        want = (131584, 512)
        try:
            got = tuple(reader.epilogue_bytes(128, 128))
        except Exception as exc:  # noqa: BLE001
            return False, f"expected {want}, actual {type(exc).__name__}: {exc}"
        return _mismatch(got, want, "128 by 128 fp16 tile, fp32 bias")

    c.check("a fused epilogue reads 512 bytes where the unfused pair moves 131584", bytes_check)

    def floor_check():
        cases = ((4096, 4096, 4096), (64, 64, 64))
        for shape in cases:
            want = _floor_ms(*shape)
            try:
                got = float(reader.predicted_ms(*shape))
            except Exception as exc:  # noqa: BLE001
                return False, f"expected {want:.6f} ms for {shape}, actual {type(exc).__name__}: {exc}"
            if abs(got - want) > 1e-6:
                return False, f"expected {want:.6f} ms for {shape}, actual {got:.6f} ms"
        big = _floor_ms(4096, 4096, 4096)
        return True, f"expected {big:.4f} ms for 4096 cubed, actual {big:.4f} ms"

    c.check("the roofline floor uses 312 TFLOP/s above the ridge and 1935 GB/s below it", floor_check)

    _gpu_section(c, reader)
    return c.finish()


def _gpu_section(c: Checks, reader) -> None:
    """Kernel checks. Skipped, not failed, when this machine cannot run them."""
    try:
        import torch
    except ImportError:
        print(
            "SKIP: the CuTe DSL GEMM was not run. torch is not installed, so there is no CUDA GPU to compile it on.",
            flush=True,
        )
        c.metric("tflops", "skipped (no torch)")
        return
    if not torch.cuda.is_available():
        print(
            "SKIP: the CuTe DSL GEMM was not run. No CUDA GPU is visible. "
            "The kernel checks need an A100 (sm_80) and nvidia-cutlass-dsl 4.8.1.",
            flush=True,
        )
        c.metric("tflops", "skipped (no GPU)")
        return
    try:
        import cutlass.cute  # noqa: F401
    except ImportError:
        print(
            "SKIP: the CuTe DSL GEMM was not run. nvidia-cutlass-dsl is not installed on this machine.",
            flush=True,
        )
        c.metric("tflops", "skipped (no cutlass dsl)")
        return

    from lab_common import cuda_time, tflops

    torch.manual_seed(24)

    def one_shape(m: int, n: int, k: int, atol: float):
        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(k, n, device="cuda", dtype=torch.float16)
        bias = torch.randn(n, device="cuda", dtype=torch.float32)
        try:
            got = reader.cute_gemm_bias_gelu(a, b, bias)
        except Exception as exc:  # noqa: BLE001
            return False, f"expected a ({m}, {n}) fp16 tensor, actual {type(exc).__name__}: {exc}"
        if got.shape != (m, n):
            return False, f"expected shape {(m, n)}, actual {tuple(got.shape)}"
        if got.dtype != torch.float16:
            return False, f"expected dtype torch.float16, actual {got.dtype}"
        ref = torch.nn.functional.gelu(a.float() @ b.float() + bias)
        err = (got.float() - ref).abs()
        max_err = float(err.max())
        flat = int(err.argmax())
        return (
            max_err <= atol,
            f"expected max abs error <= {atol:.1e}, actual {max_err:.3e} "
            f"(worst element expected {float(ref.flatten()[flat]):.5f}, "
            f"actual {float(got.flatten()[flat]):.5f}) at {m}x{n}x{k}",
        )

    def correctness():
        # 192 is not a multiple of the 128-wide tile, so padding has to be right.
        shapes = ((128, 128, 64), (256, 128, 128), (128, 192, 64), (17, 19, 23))
        for shape in shapes:
            ok, detail = one_shape(*shape, atol=2e-2)
            if not ok:
                return False, detail
        return True, "expected max abs error <= 2.0e-2 on four shapes, actual within 2.0e-2"

    c.check("the fused GEMM matches erf GELU of a fp32 matmul plus bias", correctness)

    def time_kernel():
        m = n = k = 2048
        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(k, n, device="cuda", dtype=torch.float16)
        bias = torch.randn(n, device="cuda", dtype=torch.float32)
        reader.cute_gemm_bias_gelu(a, b, bias)
        ms = cuda_time(lambda: reader.cute_gemm_bias_gelu(a, b, bias), warmup=5, iters=20)
        rate = tflops(2 * m * n * k, ms)
        floor = _floor_ms(m, n, k)
        c.metric("tflops", round(rate, 2))
        return True, (
            f"measured {rate:.2f} TFLOP/s in {ms:.4f} ms. "
            f"The roofline floor for this shape is {floor:.4f} ms on the A100 80GB PCIe ratings. "
            f"No pass/fail threshold: this run records the metric."
        )

    c.check("the lab records TFLOP/s for a 2048 cubed GEMM", time_kernel)
