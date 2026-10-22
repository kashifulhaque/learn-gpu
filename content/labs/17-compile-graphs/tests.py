import torch
from lab_common import A100, Checks, bandwidth_gbs, close, cuda_time, require_cuda

# Thresholds are fractions of arithmetic from docs/FACTS.md, never measured values.
#
# Part 1. N = 2^24 float32 is 64 MiB per tensor, more than the 40 MiB L2, so the
# byte count holds. Fused: 1 read + 1 write = 8N bytes = 134.2 MB, a floor of
# 0.105 ms at the 1275 GB/s copy ceiling. Eager: 6 reads + 6 writes = 48N bytes
# = 805.3 MB, a floor of 0.632 ms. The ideal speedup is 6x. The tests ask for
# half the ceiling, and a third of the ideal speedup for the hand-fused kernel
# (2x), and a quarter for the compiled one (1.5x).
N = 2**24
TRITON_MIN_FRACTION = 0.5
TRITON_MIN_SPEEDUP = 2.0
COMPILE_MIN_SPEEDUP = 1.5
# Part 2. 128 kernels, each a few microseconds. Eager is bound by the host at
# about 5 to 10 us per launch (FACTS.md 4.5); the graph replaces 128 launches
# with one. A loose 1.2x.
GRAPH_MIN_SPEEDUP = 1.2
DECODE_HIDDEN = 256
DECODE_LAYERS = 64

SIZES = [1, 7, 1000, 1024, 4097, 2**20 + 3]


def reference(x):
    t = x * 1.5
    t = t + 0.5
    t = torch.sin(t)
    t = t * t
    t = t - 0.25
    return torch.relu(t)


def decode_reference(h, weights, biases):
    for w, b in zip(weights, biases):
        h = torch.tanh(torch.addmm(b, h, w))
    return h


def make_decode_state(hidden, layers, batch, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    scale = hidden**-0.5
    weights = [torch.randn(hidden, hidden, device="cuda", generator=g) * scale for _ in range(layers)]
    biases = [torch.randn(hidden, device="cuda", generator=g) * 0.1 for _ in range(layers)]
    h = torch.randn(batch, hidden, device="cuda", generator=g)
    return weights, biases, h


def count_launches(fn):
    """Best effort: (host kernel launches, device kernels) from one profiled call.

    Returns (None, None) when the profiler's output doesn't look as expected.
    The numbers are reported, not asserted against, unless both are non-zero.
    """
    try:
        from torch.autograd import DeviceType
        from torch.profiler import ProfilerActivity, profile

        fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        host = 0
        device = 0
        for evt in prof.key_averages():
            if evt.key.startswith(("cudaLaunchKernel", "cuLaunchKernel", "cudaGraphLaunch")):
                host += evt.count
            if evt.device_type == DeviceType.CUDA and not evt.key.startswith(("Memcpy", "Memset")):
                device += evt.count
        if host == 0 or device == 0:
            return None, None
        return host, device
    except Exception as exc:  # noqa: BLE001
        print(f"      (launch counting unavailable: {type(exc).__name__}: {exc})", flush=True)
        return None, None


def run(submission):
    c = Checks()
    needed = [
        "chain_eager", "fused_chain", "compile_chain", "inductor_code",
        "decode_step", "GraphedDecode", "chain_opaque", "chain_traceable",
    ]
    if not c.require(submission, *needed):
        return c.finish()
    device = require_cuda()

    # ---------------- Part 1: fusion by hand ----------------
    print("\nPart 1: a chain of six pointwise ops", flush=True)
    torch.manual_seed(0)
    for n in SIZES:
        x = torch.randn(n, device=device)
        c.check(
            f"fused_chain matches eager, n = {n}",
            lambda x=x: close(submission.fused_chain(x), reference(x), 1e-4),
        )
    c.check(
        "fused_chain handles an empty tensor",
        lambda: submission.fused_chain(torch.empty(0, device=device)).numel() == 0,
    )
    base = torch.randn(64, 33, device=device)
    c.check(
        "fused_chain handles a non-contiguous view",
        lambda: close(submission.fused_chain(base[:, ::2]), reference(base[:, ::2]), 1e-4),
    )
    c.check(
        "chain_eager is unchanged",
        lambda: close(submission.chain_eager(base), reference(base), 1e-6),
    )

    # ---------------- Part 1: torch.compile ----------------
    print("\nPart 1: torch.compile", flush=True)
    import torch._dynamo as dynamo

    dynamo.reset()
    compiled = None

    def compile_and_check():
        nonlocal compiled
        compiled = submission.compile_chain()
        # Several sizes: the second distinct size triggers one recompile with a
        # dynamic dimension, and size 1 is specialized.
        for n in (1000, 4097, 2**20 + 3, 1):
            x = torch.randn(n, device=device)
            ok, detail = close(compiled(x), reference(x), 1e-4)
            if not ok:
                return False, f"n = {n}: {detail}"
        return True, "four sizes, one function"

    c.check("the compiled chain matches eager at several sizes", compile_and_check)

    x_src = torch.randn(4096, device=device)
    n_kernels = None
    try:
        source = submission.inductor_code(submission.chain_eager, x_src)
        n_kernels = source.count("@triton.jit")
        c.metric("inductor_triton_kernels", n_kernels)
        c.add(
            "Inductor's output contains a Triton kernel",
            n_kernels >= 1,
            "found no @triton.jit in the source inductor_code returned",
        )
        c.add(
            "Inductor fused the six ops into at most 2 kernels",
            1 <= n_kernels <= 2,
            f"found {n_kernels} @triton.jit kernels; six unfused ops would give 6",
        )
    except Exception as exc:  # noqa: BLE001
        c.add("inductor_code returns Inductor's source", False, f"{type(exc).__name__}: {exc}")

    # ---------------- Part 1: timing ----------------
    print("\nPart 1: timing at N = 2^24 float32 (64 MiB per tensor)", flush=True)
    x = torch.randn(N, device=device)
    y_eager = submission.chain_eager(x)
    try:
        fused_ok = close(submission.fused_chain(x), y_eager, 1e-4)[0]
    except Exception:  # noqa: BLE001 - the starter raises NotImplementedError
        fused_ok = False
    if fused_ok:
        eager_ms = cuda_time(lambda: submission.chain_eager(x))
        triton_ms = cuda_time(lambda: submission.fused_chain(x))
        c.metric("eager_ms", round(eager_ms, 4))
        c.metric("triton_ms", round(triton_ms, 4))
        fused_bytes = 2 * N * 4
        gbs = bandwidth_gbs(fused_bytes, triton_ms)
        c.metric("triton_gbs", round(gbs, 1))
        speedup = eager_ms / triton_ms
        c.metric("speedup_triton", round(speedup, 2))
        print(
            f"      predicted floors at {A100.COPY_CEILING_GBS:.0f} GB/s: eager "
            f"{48 * N * 1.0 / (A100.COPY_CEILING_GBS * 1e6):.3f} ms, fused "
            f"{fused_bytes / (A100.COPY_CEILING_GBS * 1e6):.3f} ms (ideal speedup 6x)",
            flush=True,
        )
        c.add(
            f"the fused kernel reaches at least {TRITON_MIN_FRACTION:.0%} of the copy ceiling",
            gbs >= TRITON_MIN_FRACTION * A100.COPY_CEILING_GBS,
            f"{gbs:.0f} GB/s counting one read and one write ({fused_bytes / 1e6:.1f} MB)",
        )
        c.add(
            f"the fused kernel is at least {TRITON_MIN_SPEEDUP}x faster than eager",
            speedup >= TRITON_MIN_SPEEDUP,
            f"{speedup:.2f}x",
        )
        if compiled is not None:
            try:
                compiled(x)
                compile_ms = cuda_time(lambda: compiled(x))
                c.metric("compile_ms", round(compile_ms, 4))
                s_compile = eager_ms / compile_ms
                c.metric("speedup_compile", round(s_compile, 2))
                c.add(
                    f"torch.compile is at least {COMPILE_MIN_SPEEDUP}x faster than eager",
                    s_compile >= COMPILE_MIN_SPEEDUP,
                    f"{s_compile:.2f}x",
                )
            except Exception as exc:  # noqa: BLE001
                c.add("the compiled chain runs at N = 2^24", False, f"{type(exc).__name__}: {exc}")

        h_eager, k_eager = count_launches(lambda: submission.chain_eager(x))
        h_fused, k_fused = count_launches(lambda: submission.fused_chain(x))
        c.metric("launches_eager", k_eager)
        c.metric("launches_fused", k_fused)
        if k_eager is not None and k_fused is not None:
            c.metric("launches_saved", k_eager - k_fused)
            c.add(
                "fusion removes kernel launches",
                k_fused < k_eager,
                f"{k_eager} kernels eager, {k_fused} fused",
            )
    else:
        c.add("the fused kernel is correct at N = 2^24", False, "fix the correctness checks first")

    # ---------------- Part 2: CUDA graph ----------------
    print("\nPart 2: a decode-like loop of tiny kernels", flush=True)
    for hidden, layers, batch in [(8, 5, 1), (33, 1, 3), (DECODE_HIDDEN, DECODE_LAYERS, 1)]:
        weights, biases, h0 = make_decode_state(hidden, layers, batch)

        def check_graph(weights=weights, biases=biases, h0=h0):
            graphed = submission.GraphedDecode(weights, biases, h0)
            g = torch.Generator(device="cuda").manual_seed(7)
            for _ in range(3):  # three replays with different inputs
                h = torch.randn(h0.shape, device="cuda", generator=g)
                out = graphed(h).clone()
                ok, detail = close(out, decode_reference(h, weights, biases), 1e-5)
                if not ok:
                    return False, detail
            return True, "three replays, three different inputs"

        c.check(
            f"the graph replay matches eager (hidden {hidden}, {layers} layers, batch {batch})",
            check_graph,
        )

    weights, biases, h0 = make_decode_state(DECODE_HIDDEN, DECODE_LAYERS, 1)
    try:
        graphed = submission.GraphedDecode(weights, biases, h0)
        eager_fn = lambda: submission.decode_step(h0, weights, biases)  # noqa: E731
        graph_fn = lambda: graphed(h0)  # noqa: E731
        ok = close(graph_fn().clone(), decode_reference(h0, weights, biases), 1e-5)[0]
        if ok:
            decode_eager_ms = cuda_time(eager_fn, warmup=20, iters=100)
            decode_graph_ms = cuda_time(graph_fn, warmup=20, iters=100)
            speedup_graph = decode_eager_ms / decode_graph_ms
            c.metric("decode_eager_ms", round(decode_eager_ms, 4))
            c.metric("decode_graph_ms", round(decode_graph_ms, 4))
            c.metric("speedup_graph", round(speedup_graph, 2))
            kernels = 2 * DECODE_LAYERS
            c.metric("decode_kernels", kernels)
            print(
                f"      per kernel: eager {decode_eager_ms * 1000 / kernels:.2f} us, "
                f"graph {decode_graph_ms * 1000 / kernels:.2f} us "
                f"(rule of thumb for a launch: 5 to 10 us)",
                flush=True,
            )
            weight_bytes = DECODE_LAYERS * (DECODE_HIDDEN * DECODE_HIDDEN + DECODE_HIDDEN) * 4
            print(
                f"      the bandwidth floor is {weight_bytes / (A100.COPY_CEILING_GBS * 1e3):.1f} us "
                f"for {weight_bytes / 1e6:.1f} MB of weights: the loop is nowhere near it",
                flush=True,
            )
            c.add(
                f"graph replay is at least {GRAPH_MIN_SPEEDUP}x faster than eager launches",
                speedup_graph >= GRAPH_MIN_SPEEDUP,
                f"{speedup_graph:.2f}x ({decode_eager_ms:.3f} ms to {decode_graph_ms:.3f} ms)",
            )
            h_e, _ = count_launches(eager_fn)
            h_g, _ = count_launches(graph_fn)
            if h_e is not None and h_g is not None:
                print(f"      host launches per step: eager {h_e}, graph {h_g}", flush=True)
    except Exception as exc:  # noqa: BLE001
        c.add("the decode graph runs", False, f"{type(exc).__name__}: {exc}")

    # ---------------- Part 3: custom ops ----------------
    print("\nPart 3: custom ops under torch.compile(fullgraph=True)", flush=True)
    xs = torch.randn(1000, device=device)
    c.check(
        "chain_opaque matches eager",
        lambda: close(submission.chain_opaque(xs), reference(xs), 1e-4),
    )
    c.check(
        "chain_traceable matches eager",
        lambda: close(submission.chain_traceable(xs), reference(xs), 1e-4),
    )

    def fake_works():
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            fx = torch.empty(5, 7, device="cuda")
            out = submission.chain_opaque(fx)
        return (
            out.shape == (5, 7) and out.dtype == torch.float32 and out.device.type == "cuda",
            f"shape {tuple(out.shape)}, dtype {out.dtype}",
        )

    c.check("chain_opaque has a fake implementation (works on fake tensors)", fake_works)

    def opcheck_works():
        torch.library.opcheck(
            submission.chain_opaque, (torch.randn(129, device="cuda"),),
            test_utils=("test_schema", "test_faketensor"),
        )
        return True

    c.check("torch.library.opcheck accepts chain_opaque", opcheck_works)

    def registered():
        ns = torch.ops.learn_gpu17
        return hasattr(ns, "chain_opaque") and hasattr(ns, "chain_traceable"), "torch.ops.learn_gpu17"

    c.check("both ops are registered in the learn_gpu17 namespace", registered)

    for label, op in (("chain_opaque", submission.chain_opaque), ("chain_traceable", submission.chain_traceable)):

        def compiled_ok(op=op):
            dynamo.reset()

            def f(x):
                return op(x * 2.0) + 1.0

            cf = torch.compile(f, fullgraph=True)
            for n in (1000, 4097, 1):
                x = torch.randn(n, device="cuda")
                ok, detail = close(cf(x), reference(x * 2.0) + 1.0, 1e-4)
                if not ok:
                    return False, f"n = {n}: {detail}"
            return True, "fullgraph=True, three sizes"

        c.check(f"{label} works inside torch.compile(fullgraph=True)", compiled_ok)

    def breaks():
        dynamo.reset()

        def f(x):
            return submission.chain_traceable(x * 2.0) + 1.0

        out = dynamo.explain(f)(torch.randn(1000, device="cuda"))
        c.metric("graph_breaks", out.graph_break_count)
        return out.graph_break_count == 0, f"{out.graph_break_count} graph breaks"

    c.check("torch._dynamo.explain reports no graph breaks around chain_traceable", breaks)
    return c.finish()
