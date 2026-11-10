"""Lab 29 solution: GEMM scheduling, from wave counts to Triton kernels.

The first eleven functions are plain Python or torch on the CPU. The Triton
kernels at the bottom run on a CUDA GPU. A machine without Triton imports this
file anyway, because the kernels are defined only when Triton is present.
"""

from __future__ import annotations

import math

try:  # Triton is not installed on a CPU-only machine.
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    HAVE_TRITON = False

# Constants of the cost model. The first three are the A100 80GB PCIe figures in
# docs/FACTS.md. MIN_GAIN is a modelling choice, not a hardware number.
PEAK_TFLOPS = 312.0  # dense bf16 tensor cores
BW_GBS = 1275.0  # measured copy ceiling, learn-inference chapter 10
LAUNCH_US = 5.0  # host launch cost of the second (reduction) kernel
MIN_GAIN = 0.05  # a schedule must beat data-parallel by 5% to be chosen

EPILOGUES = ("bias", "bias_relu", "bias_gelu", "bias_silu", "bias_residual", "scale_residual")


# ---------------------------------------------------------------------------
# 1. Waves
# ---------------------------------------------------------------------------


def wave_stats(
    m: int, n: int, tile_m: int, tile_n: int, sm_count: int = 108, ctas_per_sm: int = 1
) -> tuple[int, int, float]:
    """Return (tiles, waves, efficiency) for one CTA per output tile.

    Each dimension rounds up to whole tiles on its own. Efficiency is
    tiles / (waves * ctas_per_sm * sm_count).
    """
    tiles = math.ceil(m / tile_m) * math.ceil(n / tile_n)
    capacity = ctas_per_sm * sm_count
    waves = math.ceil(tiles / capacity)
    return tiles, waves, tiles / (waves * capacity)


# ---------------------------------------------------------------------------
# 2. Split-K
# ---------------------------------------------------------------------------


def split_k_ranges(k_iters: int, splits: int) -> list[tuple[int, int]]:
    """Cut `k_iters` K-iterations into `splits` half-open ranges.

    Sizes differ by at most one, and the first `k_iters % splits` ranges are the
    longer ones. When splits > k_iters the trailing ranges are empty.
    """
    base, rem = divmod(k_iters, splits)
    out = []
    start = 0
    for s in range(splits):
        size = base + (1 if s < rem else 0)
        out.append((start, start + size))
        start += size
    return out


def split_k_matmul(a, b, splits: int, block_k: int):
    """A @ B as `splits` fp32 partial products, reduced in fp32 in split order.

    `a` is (M, K) and `b` is (K, N), on the CPU, in any floating dtype. K is cut
    in units of `block_k`, so the last K-iteration can be shorter than block_k.
    """
    import torch

    k = a.shape[1]
    k_iters = math.ceil(k / block_k)
    out = torch.zeros(a.shape[0], b.shape[1], dtype=torch.float32)
    for lo, hi in split_k_ranges(k_iters, splits):
        k0, k1 = lo * block_k, min(hi * block_k, k)
        if k1 > k0:
            partial = a[:, k0:k1].float() @ b[k0:k1].float()  # stays fp32
            out = out + partial
    return out


def reduce_partials(partials, order) -> float:
    """Sum scalar partials in fp32, one at a time, in the given order.

    This is what the atomic variant does, with the order chosen by the hardware.
    """
    import torch

    acc = torch.tensor(0.0, dtype=torch.float32)
    for i in order:
        acc = acc + torch.tensor(float(partials[i]), dtype=torch.float32)
    return float(acc)


# ---------------------------------------------------------------------------
# 3. Stream-K
# ---------------------------------------------------------------------------


def stream_k_plan(tiles: int, iters_per_tile: int, ctas: int) -> list[list[tuple[int, int, int]]]:
    """Deal the tiles * iters_per_tile MAC-iterations to `ctas` CTAs.

    Iteration i belongs to tile i // iters_per_tile. CTA c gets one contiguous
    range of the global iteration order. Sizes differ by at most one, and the
    first `total % ctas` CTAs take the extra one. Return, for each CTA, its
    segments as (tile, k_begin, k_end), where k is local to the tile and a
    segment never crosses a tile boundary.
    """
    total = tiles * iters_per_tile
    base, rem = divmod(total, ctas)
    plan = []
    for c in range(ctas):
        start = c * base + min(c, rem)
        end = start + base + (1 if c < rem else 0)
        segments = []
        i = start
        while i < end:
            tile = i // iters_per_tile
            stop = min(end, (tile + 1) * iters_per_tile)
            segments.append((tile, i - tile * iters_per_tile, stop - tile * iters_per_tile))
            i = stop
        plan.append(segments)
    return plan


def fixup_tiles(plan) -> dict[int, list[int]]:
    """Map each tile that two or more CTAs touch to its sorted list of CTAs."""
    owners: dict[int, list[int]] = {}
    for cta, segments in enumerate(plan):
        for tile, _, _ in segments:
            owners.setdefault(tile, []).append(cta)
    return {t: sorted(set(c)) for t, c in sorted(owners.items()) if len(set(c)) > 1}


# ---------------------------------------------------------------------------
# 4. Persistent kernels
# ---------------------------------------------------------------------------


def persistent_tile_order(
    cta: int, ctas: int, tiles_m: int, tiles_n: int, group_m: int
) -> list[tuple[int, int]]:
    """The (pid_m, pid_n) tiles one persistent CTA computes, in order.

    CTA `cta` takes linear tile ids cta, cta + ctas, cta + 2 * ctas, ... below
    tiles_m * tiles_n. Each id maps to a tile with grouped ordering: vertical
    strips of `group_m` rows, the row index changing fastest inside a strip,
    and a shorter last strip when tiles_m is not a multiple of group_m.
    """
    out = []
    per_group = group_m * tiles_n
    for tile_id in range(cta, tiles_m * tiles_n, ctas):
        group = tile_id // per_group
        first_m = group * group_m
        size_m = min(tiles_m - first_m, group_m)
        within = tile_id % per_group
        out.append((first_m + within % size_m, within // size_m))
    return out


def simulate_tile_counter(costs, ctas: int) -> tuple[float, list[list[int]]]:
    """Dynamic scheduling with an atomic tile counter.

    All CTAs start at time 0. Whenever a CTA is free it takes the next tile id
    and works for costs[id]. When several CTAs are free at once, the lowest CTA
    index goes first. Return (makespan, tile ids per CTA, in the order taken).
    """
    free = [0.0] * ctas
    taken: list[list[int]] = [[] for _ in range(ctas)]
    for tile, cost in enumerate(costs):
        c = min(range(ctas), key=lambda i: (free[i], i))
        free[c] += cost
        taken[c].append(tile)
    return max(free) if costs else 0.0, taken


def static_makespan(costs, ctas: int) -> float:
    """Static round-robin: tile i goes to CTA i % ctas. Return the longest CTA time."""
    load = [0.0] * ctas
    for i, cost in enumerate(costs):
        load[i % ctas] += cost
    return max(load) if costs else 0.0


# ---------------------------------------------------------------------------
# 5. Epilogues
# ---------------------------------------------------------------------------


def apply_epilogue(acc, kind: str, bias=None, residual=None, alpha: float = 1.0, beta: float = 1.0):
    """Apply one epilogue to an fp32 accumulator tile. All inputs are fp32.

    bias:           acc + bias
    bias_relu:      relu(acc + bias)
    bias_gelu:      gelu_tanh(acc + bias), the tanh approximation
    bias_silu:      silu(acc + bias)
    bias_residual:  acc + bias + residual
    scale_residual: alpha * acc + beta * residual
    """
    import torch

    if kind == "scale_residual":
        return alpha * acc + beta * residual
    x = acc + bias
    if kind == "bias":
        return x
    if kind == "bias_relu":
        return torch.relu(x)
    if kind == "bias_gelu":
        u = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)
        return 0.5 * x * (1.0 + torch.tanh(u))
    if kind == "bias_silu":
        return x * torch.sigmoid(x)
    if kind == "bias_residual":
        return x + residual
    raise ValueError(f"unknown epilogue {kind!r}")


_STAGES = {"bias": 1, "bias_relu": 2, "bias_gelu": 2, "bias_silu": 2, "bias_residual": 2, "scale_residual": 2}


def epilogue_bytes(m: int, n: int, kind: str, fused: bool, elem_bytes: int = 2) -> int:
    """Bytes of C-sized traffic that the output stage moves, in HBM.

    Fused: the GEMM writes C once, and reads the epilogue's operands, which are
    a bias row of n elements and, for the two residual kinds, an m x n residual.
    Unfused: the GEMM writes C, then every stage is its own kernel that reads
    its m x n input, reads its operand, and writes m x n. _STAGES lists the
    stages per kind: bias -> 1, bias_residual -> 2 (bias, then the residual).
    """
    tile = m * n * elem_bytes
    has_bias = kind != "scale_residual"
    has_residual = kind in ("bias_residual", "scale_residual")
    operands = (n * elem_bytes if has_bias else 0) + (tile if has_residual else 0)
    if fused:
        return tile + operands
    return tile + _STAGES[kind] * 2 * tile + operands


# ---------------------------------------------------------------------------
# 6. The schedule chooser
# ---------------------------------------------------------------------------


def schedule_cost_us(
    m: int,
    n: int,
    k: int,
    schedule: str,
    splits: int = 1,
    tile: tuple[int, int, int] = (128, 128, 64),
    sm_count: int = 108,
) -> float:
    """Predicted microseconds for one bf16 GEMM, one CTA per SM.

    time = max(compute, memory) + extra, where
      compute = critical-path K-iterations * t_iter, and t_iter is one
                tile_m x tile_n x tile_k step at the per-SM peak, 312e12 / sm_count;
      memory  = compulsory bytes 2 * (mk + kn + mn) at 1275 GB/s, scaled by
                sm_count / active CTAs, because a grid that fills fewer SMs
                pulls proportionally less bandwidth;
      extra   = split-K: a 5 us launch plus writing and reading `splits` fp32
                m x n partials at 1275 GB/s (none for splits = 1);
                stream-K: writing and reading one fp32 tile for each tile that
                needs a fix-up.
    Schedules: "data_parallel", "split_k", "stream_k".
    """
    tm, tn, tk = tile
    tiles = math.ceil(m / tm) * math.ceil(n / tn)
    iters_per_tile = math.ceil(k / tk)
    t_iter_us = 2 * tm * tn * tk / (PEAK_TFLOPS * 1e12 / sm_count) * 1e6
    extra = 0.0
    if schedule == "data_parallel":
        crit = math.ceil(tiles / sm_count) * iters_per_tile
        active = min(tiles, sm_count)
    elif schedule == "split_k":
        ctas = tiles * splits
        crit = math.ceil(ctas / sm_count) * math.ceil(iters_per_tile / splits)
        active = min(ctas, sm_count)
        if splits > 1:
            extra = LAUNCH_US + 2 * splits * m * n * 4 / (BW_GBS * 1e3)
    elif schedule == "stream_k":
        total = tiles * iters_per_tile
        g = min(sm_count, total)
        crit = math.ceil(total / g)
        active = g
        n_fix = len(fixup_tiles(stream_k_plan(tiles, iters_per_tile, g)))
        extra = n_fix * 2 * tm * tn * 4 / (BW_GBS * 1e3)
    else:
        raise ValueError(f"unknown schedule {schedule!r}")
    compute = crit * t_iter_us
    memory = 2 * (m * k + k * n + m * n) / (BW_GBS * 1e3) * sm_count / active
    return max(compute, memory) + extra


def choose_schedule(
    m: int,
    n: int,
    k: int,
    tile: tuple[int, int, int] = (128, 128, 64),
    sm_count: int = 108,
    allow_stream_k: bool = True,
) -> tuple[str, int]:
    """Pick the cheapest schedule, as (name, splits).

    Candidates: data_parallel; split_k with 2 <= splits <= min(16, k-iterations);
    and stream_k when allowed. A candidate other than data_parallel wins only if
    its cost is at most (1 - MIN_GAIN) times data_parallel's. Among those, the
    cheapest wins, and the earlier candidate in the list above wins a tie.
    """
    base = schedule_cost_us(m, n, k, "data_parallel", 1, tile, sm_count)
    iters = math.ceil(k / tile[2])
    candidates = [("split_k", s) for s in range(2, min(16, iters) + 1)]
    if allow_stream_k:
        candidates.append(("stream_k", 1))
    best = ("data_parallel", 1)
    best_cost = base
    limit = (1 - MIN_GAIN) * base
    for name, s in candidates:
        cost = schedule_cost_us(m, n, k, name, s, tile, sm_count)
        if cost <= limit and (best[0] == "data_parallel" or cost < best_cost * (1 - 1e-9)):
            best, best_cost = (name, s), cost
    return best


# ---------------------------------------------------------------------------
# 7. Triton kernels (GPU)
# ---------------------------------------------------------------------------

if HAVE_TRITON:

    @triton.jit
    def _grouped_pid(tile_id, num_pid_m, num_pid_n, GROUP_M: tl.constexpr):
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        within = tile_id % num_pid_in_group
        pid_m = first_pid_m + within % group_size_m
        pid_n = within // group_size_m
        return pid_m, pid_n

    @triton.jit
    def _k_loop(
        a_ptr, b_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
        offs_m, offs_n, k_begin, k_end, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for kk in range(k_begin, k_end):
            k_off = kk * BLOCK_K
            ks = k_off + offs_k
            a = tl.load(
                a_ptr + offs_m[:, None] * stride_am + ks[None, :] * stride_ak,
                mask=(offs_m[:, None] < M) & (ks[None, :] < K), other=0.0,
            )
            b = tl.load(
                b_ptr + ks[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                mask=(ks[:, None] < K) & (offs_n[None, :] < N), other=0.0,
            )
            acc = tl.dot(a, b, acc)
        return acc

    @triton.jit
    def splitk_partial_kernel(
        a_ptr, b_ptr, ws_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr, SPLITS: tl.constexpr,
    ):
        # Grid: (tiles, SPLITS). Program (t, s) writes the fp32 partial of tile t over split s.
        tile_id = tl.program_id(0)
        split = tl.program_id(1)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        pid_m, pid_n = _grouped_pid(tile_id, num_pid_m, num_pid_n, GROUP_M)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        k_iters = tl.cdiv(K, BLOCK_K)
        base = k_iters // SPLITS
        rem = k_iters % SPLITS
        k_begin = split * base + tl.minimum(split, rem)
        k_end = k_begin + base + (split < rem).to(tl.int32)
        acc = _k_loop(
            a_ptr, b_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
            offs_m, offs_n, k_begin, k_end, BLOCK_M, BLOCK_N, BLOCK_K,
        )
        ws = ws_ptr + split.to(tl.int64) * M * N
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(ws + offs_m[:, None] * N + offs_n[None, :], acc, mask=mask)

    @triton.jit
    def splitk_reduce_kernel(ws_ptr, c_ptr, MN, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < MN
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for s in tl.static_range(SPLITS):  # a fixed order, so the sum is reproducible
            acc += tl.load(ws_ptr + s * MN + offs, mask=mask, other=0.0)
        tl.store(c_ptr + offs, acc.to(c_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def persistent_kernel(
        a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr, SPLITS: tl.constexpr, NUM_SMS: tl.constexpr,
    ):
        # One program per SM. A work item is one (tile, split) pair, and a program
        # takes items start, start + NUM_SMS, ... (a static tile counter).
        start = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        tiles = num_pid_m * num_pid_n
        k_iters = tl.cdiv(K, BLOCK_K)
        base = k_iters // SPLITS
        rem = k_iters % SPLITS
        for work in range(start, tiles * SPLITS, NUM_SMS):
            tile_id = work // SPLITS
            split = work % SPLITS
            pid_m, pid_n = _grouped_pid(tile_id, num_pid_m, num_pid_n, GROUP_M)
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            k_begin = split * base + tl.minimum(split, rem)
            k_end = k_begin + base + (split < rem).to(tl.int32)
            acc = _k_loop(
                a_ptr, b_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
                offs_m, offs_n, k_begin, k_end, BLOCK_M, BLOCK_N, BLOCK_K,
            )
            c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
            mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
            if SPLITS == 1:
                tl.store(c_ptrs, acc, mask=mask)
            else:
                tl.atomic_add(c_ptrs, acc, mask=mask)  # fp32 atomics: the order varies

    @triton.jit
    def fused_kernel(
        a_ptr, b_ptr, c_ptr, bias_ptr, res_ptr, M, N, K, alpha, beta,
        stride_am, stride_ak, stride_bk, stride_bn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr, EPILOGUE: tl.constexpr,
    ):
        tile_id = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        pid_m, pid_n = _grouped_pid(tile_id, num_pid_m, num_pid_n, GROUP_M)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = _k_loop(
            a_ptr, b_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
            offs_m, offs_n, 0, tl.cdiv(K, BLOCK_K), BLOCK_M, BLOCK_N, BLOCK_K,
        )
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        res_ptrs = res_ptr + offs_m[:, None] * N + offs_n[None, :]
        if EPILOGUE == "scale_residual":
            res = tl.load(res_ptrs, mask=mask, other=0.0).to(tl.float32)
            out = alpha * acc + beta * res
        else:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            x = acc + bias[None, :]
            if EPILOGUE == "bias":
                out = x
            elif EPILOGUE == "bias_relu":
                out = tl.maximum(x, 0.0)
            elif EPILOGUE == "bias_gelu":
                # 0.5 * x * (1 + tanh(u)) equals x * sigmoid(2u).
                u = 0.7978845608028654 * (x + 0.044715 * x * x * x)
                out = x * tl.sigmoid(2.0 * u)
            elif EPILOGUE == "bias_silu":
                out = x * tl.sigmoid(x)
            else:  # bias_residual
                res = tl.load(res_ptrs, mask=mask, other=0.0).to(tl.float32)
                out = x + res
        c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
        tl.store(c_ptrs, out.to(c_ptr.dtype.element_ty), mask=mask)


def _block_m(m: int) -> int:
    """Smallest tile height of 16, 32, 64, or 128 that covers m."""
    for size in (16, 32, 64, 128):
        if m <= size:
            return size
    return 128


def _check_inputs(a, b):
    import torch

    assert a.is_cuda and b.is_cuda and a.dtype == b.dtype and a.dtype in (torch.float16, torch.bfloat16)
    assert a.shape[1] == b.shape[0]
    return a.shape[0], b.shape[1], a.shape[1]


def splitk_matmul(a, b, splits: int):
    """C = A @ B with split-K: fp32 partials in a workspace, then a reduction kernel."""
    import torch

    M, N, K = _check_inputs(a, b)
    bm, bn, bk = _block_m(M), 128, 64
    ws = torch.empty((splits, M, N), device=a.device, dtype=torch.float32)
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    tiles = triton.cdiv(M, bm) * triton.cdiv(N, bn)
    splitk_partial_kernel[(tiles, splits)](
        a, b, ws, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8, SPLITS=splits, num_warps=4, num_stages=3,
    )
    splitk_reduce_kernel[(triton.cdiv(M * N, 1024),)](ws, c, M * N, SPLITS=splits, BLOCK=1024)
    return c


def persistent_matmul(a, b, splits: int = 1):
    """C = A @ B with one program per SM looping over (tile, split) work items.

    splits = 1 stores each tile directly. splits > 1 adds fp32 partials with
    atomics into a zeroed buffer, so the result can differ in the last bit
    from run to run.
    """
    import torch

    M, N, K = _check_inputs(a, b)
    bm, bn, bk = _block_m(M), 128, 64
    sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    c32 = torch.zeros((M, N), device=a.device, dtype=torch.float32)
    items = triton.cdiv(M, bm) * triton.cdiv(N, bn) * splits
    persistent_kernel[(min(items, sms),)](
        a, b, c32, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8, SPLITS=splits, NUM_SMS=sms,
        num_warps=4, num_stages=3,
    )
    return c32.to(a.dtype)


def fused_matmul(a, b, kind: str, bias=None, residual=None, alpha: float = 1.0, beta: float = 1.0):
    """C = epilogue(A @ B), with the epilogue applied to the fp32 accumulator."""
    import torch

    M, N, K = _check_inputs(a, b)
    bm, bn, bk = _block_m(min(M, 128)), 128, 64
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    bias_arg = bias if bias is not None else c
    res_arg = residual if residual is not None else c
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    fused_kernel[grid](
        a, b, c, bias_arg, res_arg, M, N, K, alpha, beta,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8, EPILOGUE=kind, num_warps=4, num_stages=3,
    )
    return c
