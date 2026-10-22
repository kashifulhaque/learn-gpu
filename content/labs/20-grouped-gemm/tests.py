"""Tests for lab 20: token permutation, a Triton grouped GEMM, and 2:4 sparsity."""

import torch
from lab_common import A100, Checks, close, cuda_time, require_cuda, tflops

# (tokens, experts, k, D, N): from one token to ragged sizes no block size divides.
SHAPES = [
    (1, 4, 2, 32, 16),
    (1, 8, 2, 64, 64),
    (7, 5, 2, 33, 17),
    (64, 2, 1, 48, 48),
    (50, 3, 3, 16, 24),
    (129, 8, 2, 100, 130),
    (1000, 16, 2, 257, 200),
]

# Tolerances are relative to the largest reference value. A correctly rounded
# fp16 value is within 2^-11 (4.9e-4) and a bf16 value within 2^-9 (2e-3); the
# output is rounded per assignment and again after the two-way sum, so these
# leave a wide margin for summation order.
REL_TOL = {torch.float16: 1e-2, torch.bfloat16: 3e-2}


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


def _reference(x, w, logits, k):
    """The float32 answer from the same rounded inputs, written as a loop."""
    values, ids = torch.topk(logits.float(), k, dim=-1)
    gates = torch.softmax(values, dim=-1)
    out = torch.zeros(x.shape[0], w.shape[2], dtype=torch.float32, device=x.device)
    for e in range(w.shape[0]):
        rows, slot = torch.where(ids == e)
        if rows.numel():
            out.index_add_(0, rows, gates[rows, slot, None] * (x[rows].float() @ w[e].float()))
    return out


def _compare(actual, expected, dtype):
    tol = REL_TOL[dtype] * max(expected.abs().max().item(), 1.0)
    return close(actual.float(), expected, tol)


def _make(T, E, D, N, dtype, device, skew=0.0, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(T, D, generator=g).to(device=device, dtype=dtype)
    w = (torch.randn(E, D, N, generator=g) / D**0.5).to(device=device, dtype=dtype)
    logits = torch.randn(T, E, generator=g)
    if skew:
        logits = logits + skew * torch.linspace(1.0, 0.0, E)  # low-numbered experts are popular
    return x, w, logits.to(device)


# ---------------------------------------------------------------------------
# The CPU-only parts
# ---------------------------------------------------------------------------


def _check_python_parts(c, submission):
    torch.manual_seed(0)

    def routing():
        logits = torch.randn(37, 9)
        ids, gates = submission.route_topk(logits, 3)
        want_ids = torch.topk(logits, 3, dim=-1).indices
        if tuple(ids.shape) != (37, 3) or tuple(gates.shape) != (37, 3):
            return False, f"shapes {tuple(ids.shape)} and {tuple(gates.shape)}, expected (37, 3)"
        if not torch.equal(ids, want_ids):
            return False, "ids differ from torch.topk, best expert first"
        if not torch.allclose(gates.sum(-1), torch.ones(37), atol=1e-5):
            return False, "each token's gates must sum to 1"
        want = torch.softmax(torch.gather(logits, 1, want_ids), dim=-1)
        return close(gates.float(), want, 1e-5)

    c.check("route_topk returns the top-k ids and softmax gates over the chosen k", routing)

    def offsets():
        ids = torch.tensor([[2, 0], [2, 3], [0, 3], [3, 2], [3, 0]])
        got = submission.expert_offsets(ids, 6).tolist()
        want = [0, 3, 3, 6, 10, 10, 10]
        if got != want:
            return False, f"got {got}, expected {want}"
        got = submission.expert_offsets(torch.tensor([1, 1, 1]), 3).tolist()
        return got == [0, 0, 3, 3], f"all assignments to expert 1: got {got}, expected [0, 0, 3, 3]"

    c.check("expert_offsets is a histogram followed by an exclusive scan", offsets)

    def offsets_random():
        ids = torch.randint(0, 11, (4, 50))
        counts = torch.bincount(ids.reshape(-1), minlength=11)
        want = [0] + torch.cumsum(counts, 0).tolist()
        got = submission.expert_offsets(ids, 11)
        return got.tolist() == want and got.dtype == torch.int32, f"dtype {got.dtype}, values match: {got.tolist() == want}"

    c.check("expert_offsets returns int32 and handles a 2D id tensor", offsets_random)

    def sorting():
        flat = torch.randint(0, 7, (200,))
        order = submission.sort_by_expert(flat)
        if sorted(order.tolist()) != list(range(200)):
            return False, "the result is not a permutation of 0 .. n - 1"
        s = flat[order]
        if not bool((s[1:] >= s[:-1]).all()):
            return False, "the ids are not grouped in non-decreasing order"
        for e in range(7):
            idx = order[s == e]
            if not bool((idx[1:] > idx[:-1]).all()):
                return False, f"expert {e}'s assignments are not in their original order"
        return True, "a stable grouping"

    c.check("sort_by_expert is a stable permutation that groups by expert", sorting)

    def schedule():
        got = submission.tile_schedule([0, 5, 64, 1, 130], 64)
        want = [(1, 0, 5), (2, 5, 64), (3, 69, 1), (4, 70, 64), (4, 134, 64), (4, 198, 2)]
        if [tuple(t) for t in got] != want:
            return False, f"got {got}"
        return submission.tile_schedule([0, 0], 16) == [], "no rows means no tiles"

    c.check("tile_schedule cuts each expert's rows into tiles and skips empty experts", schedule)

    def bound():
        import random

        rng = random.Random(1)
        for _ in range(200):
            e = rng.randint(1, 40)
            counts = [rng.choice([0, 0, 1, rng.randint(0, 500)]) for _ in range(e)]
            for bm in (16, 32, 64, 128):
                tiles = len(submission.tile_schedule(counts, bm))
                cap = submission.max_m_tiles(sum(counts), e, bm)
                if tiles > cap:
                    return False, f"counts {counts}, block_m {bm}: {tiles} tiles exceed the bound {cap}"
        return submission.max_m_tiles(200, 5, 64) == 9, "max_m_tiles(200, 5, 64) must be ceil(200 / 64) + 5 = 9"

    c.check("max_m_tiles is an upper bound on the real tile count", bound)

    def waste():
        got = submission.tile_waste([0, 5, 64, 1, 130], 64)
        ok = abs(got - (1 - 200 / 384)) < 1e-9
        return ok and submission.tile_waste([0, 0], 64) == 0.0, f"got {got:.4f}, expected 0.4792"

    c.check("tile_waste counts the padded rows", waste)

    def pruning():
        w = torch.randn(12, 64)
        p = submission.prune_2_4(w)
        if p.shape != w.shape or p.dtype != w.dtype:
            return False, "the shape and dtype must be unchanged"
        g = p.reshape(12, 16, 4)
        if not bool(((g != 0).sum(-1) <= 2).all()):
            return False, "a group of four has more than two non-zero values"
        kept = g.abs().sum(-1)
        top2 = w.reshape(12, 16, 4).abs().topk(2, dim=-1).values.sum(-1)
        if not torch.allclose(kept, top2, atol=1e-6):
            return False, "the kept values are not the two largest magnitudes"
        same = torch.where(g != 0, g == w.reshape(12, 16, 4), torch.ones_like(g, dtype=torch.bool))
        return bool(same.all()), "kept values must be the original values"

    c.check("prune_2_4 keeps the two largest magnitudes of every group of four", pruning)


# ---------------------------------------------------------------------------
# The GPU parts
# ---------------------------------------------------------------------------


def _segment_loop(x_sorted, w, counts, out):
    start = 0
    for e, n in enumerate(counts):
        if n:
            torch.mm(x_sorted[start:start + n], w[e], out=out[start:start + n])
        start += n


def _naive_loop(x, w, ids, gates):
    """What a straightforward model does: a where() and an index_add_ per expert."""
    out = torch.zeros(x.shape[0], w.shape[2], dtype=x.dtype, device=x.device)
    for e in range(w.shape[0]):
        rows, slot = torch.where(ids == e)
        if rows.numel():
            out.index_add_(0, rows, (x[rows] @ w[e]) * gates[rows, slot, None].to(x.dtype))
    return out


def _best_grouped_time(submission, x_sorted, w, offsets, gates_sorted):
    """The faster of two reasonable tile shapes, so a tile choice isn't what fails the lab."""
    best = float("inf")
    for kw in (
        dict(block_m=64, block_n=128, block_k=32, num_warps=4, num_stages=3),
        dict(block_m=128, block_n=128, block_k=32, num_warps=8, num_stages=3),
    ):
        ms = cuda_time(lambda kw=kw: submission.grouped_gemm(x_sorted, w, offsets, gates_sorted, **kw), warmup=5, iters=20)
        best = min(best, ms)
    return best


def _perf_case(c, submission, name, T, E, k, D, N, skew, device, floor):
    x, w, logits = _make(T, E, D, N, torch.bfloat16, device, skew=skew, seed=3)
    ids, gates = submission.route_topk(logits, k)
    flat = ids.reshape(-1)
    offsets = submission.expert_offsets(flat, E)
    order = submission.sort_by_expert(flat)
    x_sorted = x.index_select(0, order // k).contiguous()
    gates_sorted = gates.reshape(-1).index_select(0, order).contiguous()
    counts = (offsets[1:] - offsets[:-1]).tolist()
    rows = sum(counts)
    flops = 2 * rows * D * N

    ms_grouped = _best_grouped_time(submission, x_sorted, w, offsets, gates_sorted)
    out = torch.empty(rows, N, dtype=x.dtype, device=device)
    ms_loop = cuda_time(lambda: _segment_loop(x_sorted, w, counts, out), warmup=3, iters=10)
    t_grouped, t_loop = tflops(flops, ms_grouped), tflops(flops, ms_loop)
    speedup = ms_loop / ms_grouped
    mean = rows / E
    c.metric(f"grouped_tflops_{name}", round(t_grouped, 1))
    if name == "big":
        c.metric("loop_tflops_big", round(t_loop, 1))
    if name == "skewed":
        c.metric("max_over_mean_load_skewed", round(max(counts) / mean, 2))
    c.metric(f"speedup_vs_loop_{name}", round(speedup, 2))
    c.add(
        f"{name}: the grouped rate is below the tensor-core peak",
        t_grouped <= 1.1 * A100.TENSOR_BF16_TFLOPS,
        f"{t_grouped:.1f} against a {A100.TENSOR_BF16_TFLOPS:.0f} TFLOP/s peak; a higher figure means the timing is wrong",
    )
    c.add(
        f"{name}: the grouped kernel is at least {floor}x of the per-expert loop",
        speedup >= floor,
        f"{ms_grouped:.3f} ms grouped, {ms_loop:.3f} ms loop of {E} matmuls ({speedup:.2f}x)",
    )
    return t_grouped


def _sparse_24(c, submission, device):
    """Optional: never fails the lab."""
    for key in ("sparse24_dense_ms", "sparse24_sparse_ms", "sparse24_speedup"):
        c.metric(key, 0.0)
    try:
        if not hasattr(submission, "sparse_24_weight"):
            print("      note: no sparse_24_weight, so 2:4 timing is skipped.", flush=True)
            return
        if torch.cuda.get_device_capability(0) < (8, 0):
            print("      note: 2:4 sparse tensor cores need compute capability 8.0 or later.", flush=True)
            return
        M, K, N = 4096, 4096, 4096
        g = torch.Generator(device="cpu").manual_seed(5)
        weight = torch.randn(N, K, generator=g).to(device=device, dtype=torch.float16)
        x = torch.randn(M, K, generator=g).to(device=device, dtype=torch.float16)
        sparse = submission.sparse_24_weight(weight)
        if sparse is None:
            print("      note: sparse_24_weight returned None, so 2:4 timing is skipped.", flush=True)
            return
        dense_pruned = submission.prune_2_4(weight)
        want = torch.nn.functional.linear(x, dense_pruned)
        got = torch.nn.functional.linear(x, sparse)
        err = (got.float() - want.float()).abs().max().item()
        print(f"      sparse24 max abs error against the dense pruned matmul = {err:.3g}", flush=True)
        ms_dense = cuda_time(lambda: torch.nn.functional.linear(x, dense_pruned), warmup=5, iters=20)
        ms_sparse = cuda_time(lambda: torch.nn.functional.linear(x, sparse), warmup=5, iters=20)
        c.metric("sparse24_dense_ms", round(ms_dense, 4))
        c.metric("sparse24_sparse_ms", round(ms_sparse, 4))
        c.metric("sparse24_speedup", round(ms_dense / ms_sparse, 3))
        print("      The A100's sparse tensor cores are rated at 2x dense (624 versus 312 TFLOP/s). "
              "Published speedups for this API are 1.3x to 1.6x, so a value near that is normal.", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"      note: the 2:4 timing was skipped ({type(exc).__name__}: {exc})", flush=True)


def run(submission):
    c = Checks()
    if not c.require(
        submission, "route_topk", "expert_offsets", "sort_by_expert", "tile_schedule", "max_m_tiles",
        "tile_waste", "grouped_gemm", "moe_forward", "prune_2_4",
    ):
        return c.finish()

    _check_python_parts(c, submission)

    device = require_cuda()
    props = torch.cuda.get_device_properties(0)
    print(f"      device = {props.name}", flush=True)
    if props.multi_processor_count != A100.SM_COUNT:
        print(f"      note: this is not the course's A100 ({A100.SM_COUNT} SMs); the ratios below still hold.", flush=True)

    # Correctness of the whole layer over awkward shapes, in both 16-bit formats.
    for dtype, label in ((torch.float16, "fp16"), (torch.bfloat16, "bf16")):
        shapes = SHAPES if dtype == torch.float16 else [SHAPES[0], SHAPES[2], SHAPES[5]]
        for T, E, k, D, N in shapes:
            x, w, logits = _make(T, E, D, N, dtype, device, seed=T)
            c.check(
                f"{label} moe_forward: {T} tokens, {E} experts, top-{k}, D={D}, N={N}",
                lambda x=x, w=w, logits=logits, k=k, dtype=dtype: _compare(
                    submission.moe_forward(x, w, logits, k), _reference(x, w, logits, k), dtype
                ),
            )

    def one_hot_routing():
        # Every token prefers experts 0 and 1: experts 2 .. 7 are empty, and expert 0 or 1 holds every row.
        x, w, logits = _make(333, 8, 64, 50, torch.float16, device, seed=9)
        logits = logits.clone()
        logits[:, 0] += 50.0
        logits[:, 1] += 40.0
        return _compare(submission.moe_forward(x, w, logits, 2), _reference(x, w, logits, 2), torch.float16)

    c.check("six of eight experts are empty and two hold every row", one_hot_routing)

    def kernel_alone():
        # The kernel on a hand-made split with empty experts, a one-row expert and a ragged tile.
        counts = [0, 5, 64, 1, 130, 0, 17]
        offsets = torch.tensor([0] + torch.cumsum(torch.tensor(counts), 0).tolist(), dtype=torch.int32, device=device)
        rows = sum(counts)
        g = torch.Generator(device="cpu").manual_seed(2)
        xs = torch.randn(rows, 70, generator=g).to(device=device, dtype=torch.float16)
        w = torch.randn(len(counts), 70, 45, generator=g).to(device=device, dtype=torch.float16) / 8
        gate = torch.rand(rows, generator=g).to(device)
        want = torch.empty(rows, 45, device=device)
        start = 0
        for e, n in enumerate(counts):
            want[start:start + n] = (xs[start:start + n].float() @ w[e].float()) * gate[start:start + n, None]
            start += n
        for bm in (16, 64):
            got = submission.grouped_gemm(xs, w, offsets, gate, block_m=bm, block_n=32, block_k=32, num_warps=4, num_stages=2)
            ok, detail = _compare(got, want, torch.float16)
            if not ok:
                return False, f"block_m={bm}: {detail}"
        ungated = submission.grouped_gemm(xs, w, offsets, None, block_m=32, block_n=64, block_k=32)
        want_ungated = torch.empty_like(want)
        start = 0
        for e, n in enumerate(counts):
            want_ungated[start:start + n] = xs[start:start + n].float() @ w[e].float()
            start += n
        ok, detail = _compare(ungated, want_ungated, torch.float16)
        return ok, f"block_m = 16 and 64 with gates, and 32 without. {detail}"

    c.check("grouped_gemm alone: empty experts, a one-row expert, ragged tiles, gates optional", kernel_alone)

    def matches_loop():
        x, w, logits = _make(512, 8, 128, 96, torch.bfloat16, device, seed=4)
        got = submission.moe_forward(x, w, logits, 2).float()
        ids, gates = submission.route_topk(logits, 2)
        want = _naive_loop(x, w, ids, gates).float()
        return _compare(got, want, torch.bfloat16)

    c.check("moe_forward agrees with a per-expert PyTorch loop on the same routing", matches_loop)

    def deterministic():
        x, w, logits = _make(300, 8, 64, 64, torch.float16, device, seed=6)
        a = submission.moe_forward(x, w, logits, 2)
        b = submission.moe_forward(x, w, logits, 2)
        return bool(torch.equal(a, b)), "two runs give identical bits"

    c.check("moe_forward is deterministic from run to run", deterministic)

    # Performance. A 4096-token, top-2, 8-expert layer with D = 2048 and N = 4096 is 2 * 8192 * 2048 * 4096 =
    # 137.4 GFLOP, 0.44 ms at the 312 TFLOP/s peak. The Triton matmul tutorial reports about 220 TFLOP/s for a
    # plain matmul on an A100, so 20% of the peak is a loose bar for a correct grouped kernel.
    def perf():
        t_big = _perf_case(c, submission, "big", 4096, 8, 2, 2048, 4096, 0.0, device, 0.5)
        c.add(
            "big: at least 20% of the 312 TFLOP/s bf16 peak",
            t_big >= 0.2 * A100.TENSOR_BF16_TFLOPS,
            f"{t_big:.1f} against {0.2 * A100.TENSOR_BF16_TFLOPS:.1f} TFLOP/s",
        )
        # Sixty-four experts of about 128 rows each: the per-expert loop launches 64 small matmuls.
        _perf_case(c, submission, "many", 4096, 64, 2, 1024, 1024, 0.0, device, 0.5)
        _perf_case(c, submission, "skewed", 4096, 64, 2, 1024, 1024, 3.0, device, 0.5)

        # The whole layer, routing and permutation included, against the naive loop.
        x, w, logits = _make(4096, 64, 1024, 1024, torch.bfloat16, device, skew=1.0, seed=8)
        ms_fwd = cuda_time(lambda: submission.moe_forward(x, w, logits, 2), warmup=3, iters=10)

        def naive():
            ids, gates = submission.route_topk(logits, 2)
            return _naive_loop(x, w, ids, gates)

        ms_naive = cuda_time(naive, warmup=3, iters=10)
        c.metric("moe_forward_ms", round(ms_fwd, 4))
        c.metric("naive_moe_ms", round(ms_naive, 4))
        c.metric("speedup_end_to_end", round(ms_naive / ms_fwd, 2))
        del x, w, logits
        return True, "the timing cases ran"

    # A starter that still raises NotImplementedError fails here with a message, not a crash.
    c.check("the timing cases run", perf)

    _sparse_24(c, submission, device)
    return c.finish()
