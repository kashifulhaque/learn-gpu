"""Worked solution for lab 24.

The CPU half is the layout algebra from lab 13. The GPU half is one Ampere
kernel: a TiledCopy, a TiledMMA, cute.copy, cute.gemm, and an epilogue that
adds a bias and applies the erf GELU before the store. Shapes that miss the
tile are padded with zeros on the host.

The kernel follows the CuTe DSL calls in the CUTLASS warp-level MMA guide
(MmaF16BF16Op, make_tiled_mma, cute.copy, cute.gemm) and the Ampere examples'
copy atoms (CopyG2SOp, LdMatrix8x8x16bOp, make_tiled_copy_tv). It is one
stage on purpose. Chapter 12 already pipelines cp.async. This lab's question
is the DSL, not a second pipeline.
"""

from __future__ import annotations

import math

_SQRT2_INV = 1.0 / math.sqrt(2.0)
_COMPILED: dict = {}


def _product(x) -> int:
    if isinstance(x, tuple):
        total = 1
        for item in x:
            total *= _product(item)
        return total
    return x


def _layout_at(shape, stride, coord) -> int:
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
    return [_layout_at(shape, stride, index) for index in range(_product(shape))]


def partition_tiled_copy(tv_shape, tv_stride) -> list[list[int]]:
    threads = _product(tv_shape[0])
    values = _product(tv_shape[1])
    return [
        [_layout_at(tv_shape, tv_stride, (thread, value)) for value in range(values)]
        for thread in range(threads)
    ]


def accumulator_mn(lane: int, value: int) -> tuple[int, int]:
    return ((lane // 4) + 8 * (value // 2), 2 * (lane % 4) + (value % 2))


def bias_gelu(values: list[float], bias: list[float]) -> list[float]:
    return [0.5 * (v + b) * (1.0 + math.erf((v + b) * _SQRT2_INV)) for v, b in zip(values, bias)]


def epilogue_bytes(m: int, n: int) -> tuple[int, int]:
    c_bytes = m * n * 2
    bias_bytes = n * 4
    bias_kernel = c_bytes + bias_bytes + c_bytes
    gelu_kernel = c_bytes + c_bytes
    return bias_kernel + gelu_kernel, bias_bytes


def predicted_ms(m: int, n: int, k: int) -> float:
    flops = 2 * m * n * k
    nbytes = 2 * (m * k + n * k + m * n)
    peak = 312e12
    bandwidth = 1935e9
    if flops / nbytes >= peak / bandwidth:
        return flops / peak * 1e3
    return nbytes / bandwidth * 1e3


def _compile_gemm():
    """JIT-compile the kernel once. Imports of the DSL stay inside this function."""
    if "fn" in _COMPILED:
        return _COMPILED["fn"]

    import cutlass
    import cutlass.cute as cute
    from cutlass.cute import math as cute_math
    import cuda.bindings.driver as cuda
    import torch

    bm, bn, bk = 128, 128, 64
    threads = 128

    @cute.kernel
    def kernel(
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mBias: cute.Tensor,
        sA_layout: cute.Layout,
        sB_layout: cute.Layout,
        sC_layout: cute.Layout,
        tiled_copy_ab: cute.TiledCopy,
        tiled_copy_c: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        s2r_a: cute.TiledCopy,
        s2r_b: cute.TiledCopy,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(cutlass.Float16, sA_layout, byte_alignment=16)
        sB = smem.allocate_tensor(cutlass.Float16, sB_layout, byte_alignment=16)
        sC = smem.allocate_tensor(cutlass.Float16, sC_layout, byte_alignment=16)

        # (tile along M or N, tile along K, k-tile rest). proj drops the mode
        # the operand does not have: A has no N, B has no M, C has no K.
        gA = cute.local_tile(mA, tiler=(bm, bn, bk), coord=(bidx, bidy, None), proj=(1, None, 1))
        gB = cute.local_tile(mB, tiler=(bm, bn, bk), coord=(bidx, bidy, None), proj=(None, 1, 1))
        gC = cute.local_tile(mC, tiler=(bm, bn, bk), coord=(bidx, bidy, None), proj=(1, 1, None))
        gBias = cute.local_tile(mBias, tiler=(bm, bn), coord=(bidx, bidy))

        thr_ab = tiled_copy_ab.get_slice(tid)
        tAgA = thr_ab.partition_S(gA)
        tAsA = thr_ab.partition_D(sA)
        tBgB = thr_ab.partition_S(gB)
        tBsB = thr_ab.partition_D(sB)

        thr_mma = tiled_mma.get_slice(tid)
        tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA))
        tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(sB))
        tCrC = thr_mma.make_fragment_C(thr_mma.partition_C(gC))
        tCrC.fill(0.0)

        thr_s2r_a = s2r_a.get_slice(tid)
        thr_s2r_b = s2r_b.get_slice(tid)
        tCsA = thr_s2r_a.partition_S(sA)
        tCrA_view = thr_s2r_a.retile(tCrA)
        tCsB = thr_s2r_b.partition_S(sB)
        tCrB_view = thr_s2r_b.retile(tCrB)

        k_tiles = cute.size(gA, mode=[2])
        k_blocks = cute.size(tCrA, mode=[2])
        for k_tile in cutlass.range(k_tiles):
            cute.copy(tiled_copy_ab, tAgA[None, None, None, k_tile], tAsA)
            cute.copy(tiled_copy_ab, tBgB[None, None, None, k_tile], tBsB)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            for k_block in cutlass.range(k_blocks, unroll_full=True):
                cute.copy(s2r_a, tCsA[None, None, k_block], tCrA_view[None, None, k_block])
                cute.copy(s2r_b, tCsB[None, None, k_block], tCrB_view[None, None, k_block])
                cute.gemm(
                    tiled_mma,
                    tCrC,
                    tCrA[None, None, k_block],
                    tCrB[None, None, k_block],
                    tCrC,
                )
            cute.arch.sync_threads()

        # Epilogue. The raw product stays in registers and shared memory.
        # Global memory receives GELU(accumulator + bias) only.
        stored = cute.make_fragment_like(tCrC, cutlass.Float16)
        stored.store(tCrC.load().to(cutlass.Float16))
        cute.autovec_copy(stored, thr_mma.partition_C(sC))
        cute.arch.sync_threads()

        thr_c = tiled_copy_c.get_slice(tid)
        tCrE = cute.make_fragment_like(thr_c.partition_S(sC), cutlass.Float16)
        cute.autovec_copy(thr_c.partition_S(sC), tCrE)
        tCrBias = cute.make_fragment_like(thr_c.partition_S(gBias))
        cute.autovec_copy(thr_c.partition_S(gBias), tCrBias)
        acc = tCrE.load().to(cutlass.Float32) + tCrBias.load().to(cutlass.Float32)
        # 1/sqrt(2). The erf GELU is 0.5 * x * (1 + erf(x / sqrt(2))).
        activated = acc * 0.5 * (1.0 + cute_math.erf(acc * 0.7071067811865476))
        tCrE.store(activated.to(cutlass.Float16))
        cute.copy(tiled_copy_c, tCrE, thr_c.partition_D(gC))

    @cute.jit
    def launch(
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        bias_ptr: cute.Pointer,
        m: cutlass.Int32,
        n: cutlass.Int32,
        k: cutlass.Int32,
        stream,
    ):
        m = cute.assume(m, divby=8)
        n = cute.assume(n, divby=8)
        k = cute.assume(k, divby=8)
        mA = cute.make_tensor(a_ptr, cute.make_layout((m, k), stride=(k, 1)))
        # B is (N, K), K-major, which is the layout mma.sync wants for B.
        mB = cute.make_tensor(b_ptr, cute.make_layout((n, k), stride=(k, 1)))
        mC = cute.make_tensor(c_ptr, cute.make_layout((m, n), stride=(n, 1)))
        # Stride 0 along M broadcasts the bias vector across rows.
        mBias = cute.make_tensor(bias_ptr, cute.make_layout((m, n), stride=(0, 1)))

        # Swizzle<3,3,3> on an 8 by 64 half tile, the same atom as chapter 13,
        # repeated to the 128 by 64 CTA tile.
        major = 64
        swizzle = cute.make_swizzle(3, 3, 3)
        atom = cute.make_composed_layout(swizzle, 0, cute.make_layout((8, major), stride=(major, 1)))
        sA_layout = cute.tile_to_shape(atom, (bm, bk), order=(0, 1))
        sB_layout = cute.tile_to_shape(atom, (bn, bk), order=(0, 1))
        sC_layout = cute.make_layout((bm, bn), stride=(bn, 1))

        copy_ab = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            cutlass.Float16,
            num_bits_per_copy=128,
        )
        # 128 threads, 8 halves each. Along K=64 that is 8 threads per row,
        # and 16 rows of threads.
        thread_ab = cute.make_layout((16, 8), stride=(8, 1))
        value_ab = cute.make_layout((1, 8))
        tiled_copy_ab = cute.make_tiled_copy_tv(copy_ab, thread_ab, value_ab)

        copy_c = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Float16,
            num_bits_per_copy=128,
        )
        thread_c = cute.make_layout((16, 8), stride=(8, 1))
        value_c = cute.make_layout((1, 8))
        tiled_copy_c = cute.make_tiled_copy_tv(copy_c, thread_c, value_c)

        mma_op = cute.nvgpu.warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float32, (16, 8, 16))
        # 2 by 2 warps. permutation_mnk is the tile one warp iterates:
        # 32 along M, 32 along N, 16 along K. The CTA tile is larger, so the
        # partition grows a rest mode that the K loop walks.
        tiled_mma = cute.make_tiled_mma(mma_op, (2, 2, 1), permutation_mnk=(32, 32, 16))

        ldmatrix = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            cutlass.Float16,
        )
        s2r_a = cute.make_tiled_copy_A(ldmatrix, tiled_mma)
        s2r_b = cute.make_tiled_copy_B(ldmatrix, tiled_mma)

        grid = (cute.ceil_div(m, bm), cute.ceil_div(n, bn), 1)
        kernel(
            mA,
            mB,
            mC,
            mBias,
            sA_layout,
            sB_layout,
            sC_layout,
            tiled_copy_ab,
            tiled_copy_c,
            tiled_mma,
            s2r_a,
            s2r_b,
        ).launch(grid=grid, block=[threads, 1, 1], smem=128 * 1024, stream=stream)

    # A dummy allocation initializes the CUDA context the DSL compiler expects.
    torch.empty(1, device="cuda")
    dummy16 = torch.empty(16, device="cuda", dtype=torch.float16)
    dummy32 = torch.empty(16, device="cuda", dtype=torch.float32)

    def _ptr(dtype, tensor):
        return cute.runtime.make_ptr(
            dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(
        launch,
        _ptr(cutlass.Float16, dummy16),
        _ptr(cutlass.Float16, dummy16),
        _ptr(cutlass.Float16, dummy16),
        _ptr(cutlass.Float32, dummy32),
        cutlass.Int32(128),
        cutlass.Int32(128),
        cutlass.Int32(64),
        stream,
    )
    _COMPILED["fn"] = compiled
    return compiled


def _pad(n: int, multiple: int) -> int:
    return (n + multiple - 1) // multiple * multiple


def cute_gemm_bias_gelu(a, b, bias):
    import torch

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"expected matrices, got {tuple(a.shape)} and {tuple(b.shape)}")
    m, k = a.shape
    k_b, n = b.shape
    if k != k_b or bias.shape != (n,):
        raise ValueError(f"shapes a {tuple(a.shape)}, b {tuple(b.shape)}, bias {tuple(bias.shape)} do not match")

    m_pad, n_pad, k_pad = _pad(m, 128), _pad(n, 128), _pad(k, 64)
    a_pad = torch.zeros(m_pad, k_pad, device=a.device, dtype=torch.float16)
    b_kn = torch.zeros(k_pad, n_pad, device=b.device, dtype=torch.float16)
    a_pad[:m, :k] = a.to(torch.float16)
    b_kn[:k, :n] = b.to(torch.float16)
    # (N, K) contiguous is K-major, which the tensor core reads as B.
    b_nk = b_kn.transpose(0, 1).contiguous()
    bias_pad = torch.zeros(n_pad, device=bias.device, dtype=torch.float32)
    bias_pad[:n] = bias.to(torch.float32)
    c_pad = torch.empty(m_pad, n_pad, device=a.device, dtype=torch.float16)

    import cutlass
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda

    def _ptr(dtype, tensor):
        return cute.runtime.make_ptr(
            dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    _compile_gemm()(
        _ptr(cutlass.Float16, a_pad),
        _ptr(cutlass.Float16, b_nk),
        _ptr(cutlass.Float16, c_pad),
        _ptr(cutlass.Float32, bias_pad),
        cutlass.Int32(m_pad),
        cutlass.Int32(n_pad),
        cutlass.Int32(k_pad),
        stream,
    )
    torch.cuda.current_stream().synchronize()
    return c_pad[:m, :n]
