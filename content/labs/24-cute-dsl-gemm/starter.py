"""Lab 24: a CuTe DSL GEMM with a fused bias and GELU epilogue.

Do the CPU pieces first. They are the layout algebra from lab 13, applied to
a thread-value layout. The last function compiles a kernel and runs only on
the A100, where the worker image already has nvidia-cutlass-dsl 4.8.1.

1. tensor_offsets
2. partition_tiled_copy
3. accumulator_mn
4. bias_gelu
5. epilogue_bytes
6. predicted_ms
7. cute_gemm_bias_gelu
"""

from __future__ import annotations


def _product(x) -> int:
    if isinstance(x, tuple):
        total = 1
        for item in x:
            total *= _product(item)
        return total
    return x


def _layout_at(shape, stride, coord) -> int:
    """Offset of `coord` in a nested shape:stride pair.

    A tuple coordinate matches the nesting. An int coordinate into a tuple
    shape is colexicographic: the first mode is fastest.
    """
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


def tensor_offsets(shape, stride) -> list[int]:
    """Offsets of integer coordinates 0 .. size-1, in colexicographic order.

    `shape` and `stride` are congruent nested tuples, or a pair of ints.
    This is the tensor chapter 13 builds from a layout: the pointer is
    implied, and the layout is the whole address map.
    """
    return [_layout_at(shape, stride, index) for index in range(_product(shape))]


def partition_tiled_copy(tv_shape, tv_stride) -> list[list[int]]:
    """Return offsets[thread][value] for a thread-value layout.

    `tv_shape` is `(thread_shape, value_shape)` and `tv_stride` matches it.
    Evaluate the layout at the coordinate `(thread, value)`. Thread `t` is an
    integer into `thread_shape`, and value `v` is an integer into
    `value_shape`. Do not use `thread + value * num_threads`.
    """
    raise NotImplementedError


def accumulator_mn(lane: int, value: int) -> tuple[int, int]:
    """Row and column that one lane holds for one value of m16n8k16.

    The accumulator thread-value layout is `((4, 8), (2, 2)):((32, 1), (16, 8))`
    over a column-major 16 by 8 tile. Return `(m, n)` with `m` in 0..15 and
    `n` in 0..7. `value` is in 0..3.
    """
    raise NotImplementedError


def bias_gelu(values: list[float], bias: list[float]) -> list[float]:
    """Element `i` is GELU(values[i] + bias[i]), erf form.

    GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2))). Add the bias before the GELU.
    `math.erf` is enough. The lists have the same length.
    """
    raise NotImplementedError


def epilogue_bytes(m: int, n: int) -> tuple[int, int]:
    """Bytes of an unfused epilogue, and the extra bytes a fused one reads.

    C is fp16 (2 bytes per element). Bias is fp32 (4 bytes per element) and
    has length `n`. The unfused schedule is two kernels after the GEMM: a
    bias kernel that reads C and the bias and writes C, then a GELU kernel
    that reads C and writes C. Return `(unfused_bytes, fused_extra_bytes)`.
    The fused kernel only adds the bias read. The GEMM's own store of C is
    not part of either number.
    """
    raise NotImplementedError


def predicted_ms(m: int, n: int, k: int) -> float:
    """Roofline floor in milliseconds for an fp16 tensor-core GEMM.

    Use the A100 80GB PCIe ratings: 312 TFLOP/s dense fp16 tensor cores and
    1935 GB/s. Count 2*m*n*k FLOP and 2*(m*k + n*k + m*n) bytes. Compare the
    intensity with 312e12/1935e9 FLOP per byte. Return the compute floor when
    the intensity is at least the ridge, and the memory floor otherwise.
    """
    raise NotImplementedError


def cute_gemm_bias_gelu(a, b, bias):
    """GELU(a @ b + bias) on the GPU, fused into one Ampere CuTe DSL kernel.

    `a` is fp16, shape (M, K), row-major. `b` is fp16, shape (K, N),
    row-major. `bias` is fp32, shape (N,). Return an fp16 tensor of shape
    (M, N).

    Build a TiledMMA from `cute.nvgpu.warp.MmaF16BF16Op` with shape
    (16, 8, 16) and fp32 accumulation. Build a TiledCopy with
    `cute.make_tiled_copy_tv` over a 128-bit `cp.async` atom. Call
    `cute.copy` for the global-to-shared load and the `ldmatrix` load, and
    `cute.gemm` for the product. Add the bias and apply the erf GELU to the
    accumulator before the store, so the raw product never lands in global
    memory. Pad shapes that are not multiples of the 128 by 128 by 64 tile
    with zeros.

    The worker image provides `nvidia-cutlass-dsl` 4.8.1. Do not install it.
    """
    raise NotImplementedError
