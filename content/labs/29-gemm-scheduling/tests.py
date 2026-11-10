"""Tests for lab 29: GEMM scheduling.

Checks 1 to 7 run on any machine: plain Python and torch on the CPU. Checks 8
to 10 need a CUDA GPU with Triton, and a machine without one skips them with a
message. Correctness against a float32 reference is the hard check. The
speedup over a lab-14-style plain kernel is a metric with a loose threshold.

Plausible wrong implementations that these checks catch:

- wave_stats_flat_area: ceil(m * n / (tile_m * tile_n)) instead of rounding each
  dimension up. A 130 x 130 problem on 128 x 128 tiles needs 4 tiles, not 2.
- split_k_floor_ranges: k_iters // splits iterations per split, which drops the
  remainder. 16 iterations in 5 splits would cover only 15.
- split_k_low_precision_reduce: rounds each partial to the input dtype (fp16)
  before the reduction. The fp32 partials of fp16 inputs are exact enough; the
  rounded ones are off by about 5e-4 relative.
- reduce_in_double: sums the partials in float64. It returns 2.0 for every
  order, and the fp32 atomic variant does not.
- stream_k_crosses_tiles: one segment per CTA that runs past the end of a tile.
- persistent_row_major: ignores group_m and visits tiles in row-major order.
- fused_skips_residual_read: counts a fused bias_residual epilogue as if the
  residual were free.
- chooser_ignores_reduction: scores split-K without the workspace traffic, so
  it picks a large split for shapes where the reduction costs more than it saves.
"""

from __future__ import annotations

import heapq
import math

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

CPU_NAMES = (
    "wave_stats",
    "split_k_ranges",
    "split_k_matmul",
    "reduce_partials",
    "stream_k_plan",
    "fixup_tiles",
    "persistent_tile_order",
    "simulate_tile_counter",
    "static_makespan",
    "apply_epilogue",
    "epilogue_bytes",
    "schedule_cost_us",
    "choose_schedule",
)
GPU_NAMES = ("splitk_matmul", "persistent_matmul", "fused_matmul")

# The cost model, restated here so the test does not trust the reader's copy.
PEAK_TFLOPS = 312.0
BW_GBS = 1275.0
LAUNCH_US = 5.0
MIN_GAIN = 0.05

EPILOGUES = ("bias", "bias_relu", "bias_gelu", "bias_silu", "bias_residual", "scale_residual")


# ---------------------------------------------------------------------------
# Reference implementations (independent of the reader's code)
# ---------------------------------------------------------------------------


def _ref_stream_k(tiles: int, ipt: int, ctas: int):
    """Brute force: walk the iterations one at a time."""
    total = tiles * ipt
    base, rem = divmod(total, ctas)
    owner = []
    for c in range(ctas):
        owner += [c] * (base + (1 if c < rem else 0))
    plan = [[] for _ in range(ctas)]
    for i, c in enumerate(owner):
        tile, k = divmod(i, ipt)
        segs = plan[c]
        if segs and segs[-1][0] == tile and segs[-1][2] == k:
            segs[-1] = (tile, segs[-1][1], k + 1)
        else:
            segs.append((tile, k, k + 1))
    return plan


def _ref_fixups(plan):
    touched: dict[int, set] = {}
    for c, segs in enumerate(plan):
        for tile, _, _ in segs:
            touched.setdefault(tile, set()).add(c)
    return {t: sorted(cs) for t, cs in sorted(touched.items()) if len(cs) > 1}


def _ref_grouped_order(tiles_m: int, tiles_n: int, group_m: int):
    """Enumerate the grouped order with nested loops, not the closed-form map."""
    order = []
    for first in range(0, tiles_m, group_m):
        rows = range(first, min(first + group_m, tiles_m))
        for col in range(tiles_n):
            for row in rows:
                order.append((row, col))
    return order


def _ref_counter(costs, ctas):
    heap = [(0.0, c) for c in range(ctas)]
    heapq.heapify(heap)
    taken = [[] for _ in range(ctas)]
    for tile, cost in enumerate(costs):
        t, c = heapq.heappop(heap)
        taken[c].append(tile)
        heapq.heappush(heap, (t + cost, c))
    return max(t for t, _ in heap), taken


def _ref_cost(m, n, k, schedule, splits=1, tile=(128, 128, 64), sm=108):
    tm, tn, tk = tile
    tiles = -(-m // tm) * -(-n // tn)
    ipt = -(-k // tk)
    t_iter = 2 * tm * tn * tk / (PEAK_TFLOPS * 1e12 / sm) * 1e6
    extra = 0.0
    if schedule == "data_parallel":
        crit, active = -(-tiles // sm) * ipt, min(tiles, sm)
    elif schedule == "split_k":
        ctas = tiles * splits
        crit, active = -(-ctas // sm) * -(-ipt // splits), min(ctas, sm)
        if splits > 1:
            extra = LAUNCH_US + 2 * splits * m * n * 4 / (BW_GBS * 1e3)
    else:
        total = tiles * ipt
        g = min(sm, total)
        crit, active = -(-total // g), g
        base, rem = divmod(total, g)
        split_tiles = set()
        for c in range(1, g):
            q = c * base + min(c, rem)
            if q % ipt:
                split_tiles.add(q // ipt)
        extra = len(split_tiles) * 2 * tm * tn * 4 / (BW_GBS * 1e3)
    memory = 2 * (m * k + k * n + m * n) / (BW_GBS * 1e3) * sm / active
    return max(crit * t_iter, memory) + extra


def _ref_best_cost(m, n, k, tile, allow_stream_k):
    base = _ref_cost(m, n, k, "data_parallel", 1, tile)
    best = base
    for s in range(2, min(16, -(-k // tile[2])) + 1):
        best = min(best, _ref_cost(m, n, k, "split_k", s, tile)) if _ref_cost(m, n, k, "split_k", s, tile) <= (1 - MIN_GAIN) * base else best
    if allow_stream_k:
        c = _ref_cost(m, n, k, "stream_k", 1, tile)
        if c <= (1 - MIN_GAIN) * base:
            best = min(best, c)
    return best


def _ref_epilogue(acc, kind, bias, residual, alpha, beta):
    import torch
    import torch.nn.functional as F

    if kind == "scale_residual":
        return alpha * acc + beta * residual
    x = acc + bias
    return {
        "bias": x,
        "bias_relu": F.relu(x),
        "bias_gelu": F.gelu(x, approximate="tanh"),
        "bias_silu": F.silu(x),
        "bias_residual": x + residual,
    }[kind]


# ---------------------------------------------------------------------------
# CPU checks
# ---------------------------------------------------------------------------


def _check_cpu(c, s):
    import torch

    def waves():
        # wave_stats_flat_area fails on the 130 x 130 case.
        cases = [
            ((2304, 1536, 256, 128, 108, 1), (108, 1, 1.0)),
            ((2304, 1544, 256, 128, 108, 1), (117, 2, 117 / 216)),
            ((130, 130, 128, 128, 108, 1), (4, 1, 4 / 108)),
            ((64, 4096, 64, 128, 108, 1), (32, 1, 32 / 108)),
            ((4096, 4096, 128, 256, 108, 2), (512, 3, 512 / 648)),
        ]
        for args, want in cases:
            got = s.wave_stats(*args)
            if got[:2] != want[:2] or abs(got[2] - want[2]) > 1e-9:
                return False, f"wave_stats{args}: expected {want}, actual {tuple(got)}"
        return True, "expected (117, 2, 0.5417) for 2304 x 1544, actual the same"

    c.check("wave_stats rounds each dimension up, then counts waves", waves)

    def split_k():
        want = [(0, 4), (4, 7), (7, 10), (10, 13), (13, 16)]
        got = [tuple(r) for r in s.split_k_ranges(16, 5)]
        if got != want:
            return False, f"split_k_ranges(16, 5): expected {want}, actual {got}"
        got = [tuple(r) for r in s.split_k_ranges(3, 5)]
        want = [(0, 1), (1, 2), (2, 3), (3, 3), (3, 3)]
        if got != want:
            return False, f"split_k_ranges(3, 5): expected {want}, actual {got}"
        for k_iters in range(1, 40):
            for splits in range(1, 12):
                r = [tuple(x) for x in s.split_k_ranges(k_iters, splits)]
                sizes = [e - b for b, e in r]
                if len(r) != splits or r[0][0] != 0 or r[-1][1] != k_iters:
                    return False, f"split_k_ranges({k_iters}, {splits}) does not cover [0, {k_iters}): {r}"
                if any(r[i][1] != r[i + 1][0] for i in range(splits - 1)) or max(sizes) - min(sizes) > 1:
                    return False, f"split_k_ranges({k_iters}, {splits}): ranges are not contiguous and balanced: {r}"
        torch.manual_seed(29)
        for (m, k, n, splits) in [(37, 1000, 29, 5), (16, 1000, 20, 16), (5, 300, 7, 20), (8, 64, 8, 1)]:
            a = torch.randn(m, k).half()
            b = torch.randn(k, n).half()
            ref = a.double() @ b.double()
            out = s.split_k_matmul(a, b, splits, 64)
            if out.dtype != torch.float32:
                return False, f"expected a float32 result, actual {out.dtype}"
            err = (out.double() - ref).abs().max().item()
            if err > 1e-3:
                return False, f"split_k_matmul M={m} K={k} N={n} splits={splits}: expected max error under 1e-3, actual {err:.2e}"
        return True, "expected max error under 1e-3 against float64, actual under 1e-3 for 4 shapes"

    c.check("split-K covers K exactly once, and the fp32 reduction equals the reference", split_k)

    def order():
        parts = [1e8, 1.0, -1e8, 1.0]
        cases = [((0, 1, 2, 3), 1.0), ((0, 2, 1, 3), 2.0), ((1, 3, 0, 2), 0.0)]
        for o, want in cases:
            got = s.reduce_partials(parts, o)
            if got != want:
                return False, f"order {o}: expected {want}, actual {got}"
        again = s.reduce_partials(parts, (0, 1, 2, 3))
        if again != 1.0:
            return False, f"the same order twice: expected 1.0, actual {again}"
        return True, "expected 1.0, 2.0, 0.0 for three orders of one set, actual the same"

    c.check("fp32 addition is not associative: the order of the partials changes the sum", order)

    def stream_k():
        # stream_k_crosses_tiles fails the segment comparison on every case below.
        cases = [(5, 4, 3), (117, 4, 108), (7, 1, 3), (3, 8, 8), (4, 5, 16), (117, 64, 108), (1, 6, 4)]
        for tiles, ipt, ctas in cases:
            plan = [[tuple(seg) for seg in segs] for segs in s.stream_k_plan(tiles, ipt, ctas)]
            want = _ref_stream_k(tiles, ipt, ctas)
            if len(plan) != ctas:
                return False, f"({tiles}, {ipt}, {ctas}): expected {ctas} CTAs, actual {len(plan)}"
            for cta in range(ctas):
                if plan[cta] != want[cta]:
                    return False, f"({tiles}, {ipt}, {ctas}) CTA {cta}: expected {want[cta]}, actual {plan[cta]}"
            got_fix = {t: list(v) for t, v in s.fixup_tiles(plan).items()}
            want_fix = _ref_fixups(want)
            if got_fix != want_fix:
                missing = sorted(set(want_fix) ^ set(got_fix))[:5]
                return False, f"({tiles}, {ipt}, {ctas}) fix-up tiles: expected {len(want_fix)}, actual {len(got_fix)}, first differences {missing}"
        sizes = {sum(e - b for _, b, e in seg) for seg in s.stream_k_plan(117, 4, 108)}
        if sizes != {4, 5}:
            return False, f"117 tiles of 4 iterations on 108 CTAs: expected sizes {{4, 5}}, actual {sizes}"
        n_fix = len(s.fixup_tiles(s.stream_k_plan(117, 64, 108)))
        return True, f"expected 105 fix-up tiles for 117 x 64 on 108 CTAs, actual {n_fix}"

    c.check("stream-K partition matches a one-iteration-at-a-time simulation", stream_k)

    def persistent():
        # persistent_row_major fails the group_m=4 case.
        cases = [(5, 7, 4, 3), (8, 8, 16, 4), (4, 6, 5, 1), (7, 3, 108, 2), (9, 9, 9, 3)]
        for tm, tn, ctas, g in cases:
            order = _ref_grouped_order(tm, tn, g)
            for cta in range(ctas):
                got = [tuple(t) for t in s.persistent_tile_order(cta, ctas, tm, tn, g)]
                want = order[cta::ctas]
                if got != want:
                    return False, f"tiles {tm} x {tn}, {ctas} CTAs, group_m={g}, CTA {cta}: expected {want}, actual {got}"
        costs = [5, 1, 1, 1, 1, 1]
        got = s.simulate_tile_counter(costs, 2)
        if float(got[0]) != 5.0 or [list(x) for x in got[1]] != [[0], [1, 2, 3, 4, 5]]:
            return False, f"simulate_tile_counter([5, 1, 1, 1, 1, 1], 2): expected (5.0, [[0], [1, 2, 3, 4, 5]]), actual {got}"
        if float(s.static_makespan(costs, 2)) != 7.0:
            return False, f"static_makespan([5, 1, 1, 1, 1, 1], 2): expected 7.0, actual {s.static_makespan(costs, 2)}"
        rng = torch.Generator().manual_seed(29)
        for ctas in (3, 8, 108):
            costs = (torch.rand(300, generator=rng) * 9 + 1).round().tolist()
            want_t, want_taken = _ref_counter(costs, ctas)
            t, taken = s.simulate_tile_counter(costs, ctas)
            if abs(t - want_t) > 1e-9 or [list(x) for x in taken] != want_taken:
                return False, f"{ctas} CTAs, 300 random tiles: expected makespan {want_t}, actual {t}"
        return True, "expected grouped order for 5 grids, dynamic 5.0 against static 7.0, actual the same"

    c.check("a persistent CTA walks grouped tile ids, and a tile counter balances uneven tiles", persistent)

    def epilogue():
        torch.manual_seed(29)
        acc = torch.randn(6, 5) * 3
        bias = torch.randn(5)
        res = torch.randn(6, 5)
        for kind in EPILOGUES:
            got = s.apply_epilogue(acc, kind, bias=bias, residual=res, alpha=0.5, beta=2.0)
            want = _ref_epilogue(acc, kind, bias, res, 0.5, 2.0)
            err = (got - want).abs().max().item()
            if err > 1e-5:
                return False, f"{kind}: expected max error under 1e-5, actual {err:.2e}"
        tile = 4096 * 4096 * 2
        cases = [
            ((4096, 4096, "bias_gelu", True), tile + 4096 * 2),
            ((4096, 4096, "bias_gelu", False), tile + 2 * 2 * tile + 4096 * 2),
            ((4096, 4096, "bias_residual", True), 2 * tile + 4096 * 2),
            ((4096, 4096, "bias", False), tile + 2 * tile + 4096 * 2),
            ((4096, 4096, "scale_residual", True), 2 * tile),
            ((4096, 4096, "scale_residual", False), 6 * tile),
        ]
        for args, want in cases:
            got = s.epilogue_bytes(*args)
            if got != want:
                return False, f"epilogue_bytes{args}: expected {want}, actual {got}"
        saved = s.epilogue_bytes(4096, 4096, "bias_gelu", False) - s.epilogue_bytes(4096, 4096, "bias_gelu", True)
        return True, f"expected 134,217,728 bytes saved by fusing bias_gelu at 4096 x 4096, actual {saved:,}"

    c.check("six epilogues match torch, and fusing removes the intermediate round trips", epilogue)

    def chooser():
        anchors = [
            ((64, 4096, 4096, "data_parallel", 1, (64, 128, 64)), 91.596),
            ((64, 4096, 4096, "split_k", 4, (64, 128, 64)), 38.717),
            ((64, 4096, 4096, "stream_k", 1, (64, 128, 64)), 28.784),
            ((2304, 1544, 4096, "data_parallel", 1, (256, 128, 64)), 185.84),
            ((2304, 1544, 4096, "split_k", 2, (256, 128, 64)), 189.02),
            ((2304, 1544, 4096, "stream_k", 1, (256, 128, 64)), 123.22),
        ]
        for args, want in anchors:
            got = s.schedule_cost_us(*args)
            if abs(got - want) > 2e-3 * want:
                return False, f"schedule_cost_us{args}: expected {want} us, actual {got:.3f} us"
        shapes = [
            ((64, 4096, 4096), (64, 128, 64)),
            ((16, 4096, 4096), (64, 128, 64)),
            ((128, 4096, 4096), (128, 128, 64)),
            ((256, 4096, 4096), (128, 128, 64)),
            ((2304, 1544, 4096), (256, 128, 64)),
            ((2304, 1536, 4096), (256, 128, 64)),
            ((4096, 4096, 4096), (128, 128, 64)),
            ((1024, 1024, 8192), (128, 128, 64)),
            ((8192, 8192, 8192), (128, 128, 64)),
        ]
        for (m, n, k), tile in shapes:
            for allow in (True, False):
                name, sp = s.choose_schedule(m, n, k, tile=tile, allow_stream_k=allow)
                if name == "stream_k" and not allow:
                    return False, f"({m}, {n}, {k}): stream_k chosen with allow_stream_k=False"
                got = s.schedule_cost_us(m, n, k, name, sp, tile)
                best = _ref_best_cost(m, n, k, tile, allow)
                if got > best * (1 + 1e-6):
                    return False, f"({m}, {n}, {k}) allow_stream_k={allow}: chose {name} with splits {sp} at {got:.2f} us, the cheapest allowed is {best:.2f} us"
        picks = [
            ((64, 4096, 4096, (64, 128, 64)), True, ("stream_k", 1)),
            ((64, 4096, 4096, (64, 128, 64)), False, ("split_k", 4)),
            ((2304, 1544, 4096, (256, 128, 64)), False, ("data_parallel", 1)),
            ((4096, 4096, 4096, (128, 128, 64)), True, ("data_parallel", 1)),
        ]
        for (m, n, k, tile), allow, want in picks:
            got = tuple(s.choose_schedule(m, n, k, tile=tile, allow_stream_k=allow))
            if got != want:
                return False, f"({m}, {n}, {k}) allow_stream_k={allow}: expected {want}, actual {got}"
        return True, "expected split_k with 4 splits at M=64 without stream-K, data_parallel at 4096 cubed, actual the same"

    c.check("the schedule chooser agrees with the cost model", chooser)


# ---------------------------------------------------------------------------
# GPU checks
# ---------------------------------------------------------------------------

if HAVE_TRITON:

    @triton.jit
    def _plain_kernel(
        a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
    ):
        # The lab 14 kernel at its fixed configuration: one program per tile, grouped order.
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_left = K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_left), other=0.0)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_left) & (offs_n[None, :] < N), other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _plain_matmul(a, b):
    import torch

    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, 128) * triton.cdiv(N, 128),)
    _plain_kernel[grid](
        a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, GROUP_M=8, num_warps=4, num_stages=4,
    )
    return c


def _skip_reason() -> str | None:
    try:
        import torch
    except ImportError:
        return "SKIP: the Triton kernels were not run. torch is not installed."
    if not torch.cuda.is_available():
        return "SKIP: the Triton kernels were not run. No CUDA GPU is visible. They run on the A100."
    if not HAVE_TRITON:
        return "SKIP: the Triton kernels were not run. Triton is not installed."
    return None


# The tolerance is relative to the largest reference value. fp16 rounds a result to
# within 2^-11 (4.9e-4) of its value, so 4e-3 leaves a factor of 8 for summation order.
REL_TOL = 4e-3


def _gpu_compare(got, ref):
    tol = REL_TOL * max(ref.abs().max().item(), 1.0)
    err = (got.float() - ref).abs().max().item()
    return err <= tol, err, tol


def _check_gpu(c, s):
    reason = _skip_reason()
    if reason is not None:
        print(reason, flush=True)
        for key in ("splitk_speedup_m64", "persistent_speedup_m64"):
            c.metric(key, "skipped (no GPU)")
        return

    import torch

    from lab_common import cuda_time

    torch.manual_seed(29)
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"
    shapes = [(16, 4096, 4096), (32, 4096, 4096), (64, 4096, 4096), (128, 4096, 4096), (100, 1000, 1000), (7, 130, 70)]

    def tall_skinny():
        for (m, n, k) in shapes:
            a = torch.randn(m, k, device=dev, dtype=torch.float16)
            b = torch.randn(k, n, device=dev, dtype=torch.float16)
            ref = a.float() @ b.float()
            runs = [("splitk", sp, lambda sp=sp: s.splitk_matmul(a, b, sp)) for sp in (1, 2, 3, 4, 8)]
            runs += [("persistent", sp, lambda sp=sp: s.persistent_matmul(a, b, sp)) for sp in (1, 4)]
            for name, sp, fn in runs:
                ok, err, tol = _gpu_compare(fn(), ref)
                if not ok:
                    return False, f"{name} splits={sp} at M={m} N={n} K={k}: expected max error under {tol:.3g}, actual {err:.3g}"
        a = torch.randn(64, 4096, device=dev, dtype=torch.float16)
        b = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
        first = s.splitk_matmul(a, b, 4)
        for _ in range(3):
            if not torch.equal(first, s.splitk_matmul(a, b, 4)):
                return False, "the workspace split-K result changed between runs: expected bit-identical output"
        return True, "expected max error under 4e-3 of the largest value for 7 schedules on 6 shapes, actual under it"

    c.check("split-K and the persistent kernel match torch on tall-skinny shapes", tall_skinny)

    def fused():
        for (m, n, k) in [(64, 1024, 512), (130, 257, 100), (16, 4096, 4096)]:
            a = torch.randn(m, k, device=dev, dtype=torch.float16)
            b = torch.randn(k, n, device=dev, dtype=torch.float16)
            bias = torch.randn(n, device=dev, dtype=torch.float16)
            res = torch.randn(m, n, device=dev, dtype=torch.float16)
            acc = a.float() @ b.float()
            for kind in EPILOGUES:
                ref = _ref_epilogue(acc, kind, bias.float(), res.float(), 0.5, 2.0)
                got = s.fused_matmul(a, b, kind, bias=bias, residual=res, alpha=0.5, beta=2.0)
                ok, err, tol = _gpu_compare(got, ref)
                if not ok:
                    return False, f"{kind} at M={m} N={n} K={k}: expected max error under {tol:.3g}, actual {err:.3g}"
        return True, "expected max error under 4e-3 of the largest value for 6 epilogues on 3 shapes, actual under it"

    c.check("the fused epilogue matches torch for six epilogues", fused)

    def speed():
        m, n, k = 64, 4096, 4096
        a = torch.randn(m, k, device=dev, dtype=torch.float16)
        b = torch.randn(k, n, device=dev, dtype=torch.float16)
        base = cuda_time(lambda: _plain_matmul(a, b), warmup=10, iters=50, flush_l2=True)
        best_split, best_ms = 1, float("inf")
        for sp in (1, 2, 3, 4, 6, 8):
            ms = cuda_time(lambda sp=sp: s.splitk_matmul(a, b, sp), warmup=10, iters=50, flush_l2=True)
            if ms < best_ms:
                best_split, best_ms = sp, ms
        pers = min(
            cuda_time(lambda sp=sp: s.persistent_matmul(a, b, sp), warmup=10, iters=50, flush_l2=True)
            for sp in (1, 2, 4)
        )
        c.metric("plain_ms_m64", round(base, 4))
        c.metric("splitk_speedup_m64", round(base / best_ms, 3))
        c.metric("splitk_best_splits", best_split)
        c.metric("persistent_speedup_m64", round(base / pers, 3))
        best = max(base / best_ms, base / pers)
        # Loose: fail only when the best schedule is far slower than the plain kernel.
        ok = best >= 0.7
        return ok, f"expected the best of split-K and persistent at least 0.7x the plain kernel at M=64, actual {best:.2f}x"

    c.check("the best schedule is not far slower than the plain kernel at M = 64", speed)


def run(submission):
    from lab_common import Checks

    c = Checks()
    if not c.require(submission, *CPU_NAMES, *GPU_NAMES):
        return c.finish()
    _check_cpu(c, submission)
    _check_gpu(c, submission)
    return c.finish()
