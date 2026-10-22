import time

import torch
from lab_common import Checks, bandwidth_gbs, cuda_time, require_cuda

# Per-direction PCIe Gen4 x16 rate: 16 GT/s x 128/130 x 16 lanes / 8 = 31.5 GB/s
# (docs/FACTS.md, section 7). NVIDIA's datasheet prints 64, which sums both directions.
PCIE_GEN4_X16_GBS = 31.5

# The kernel maps x -> A x + B, `iters` times. In float32 the constants round, so the
# reference uses the rounded values and the closed form x_k = c + (x_0 - c) A^k, with
# the fixed point c = B / (1 - A).
A32 = float(torch.tensor(0.999, dtype=torch.float32))
B32 = float(torch.tensor(0.001, dtype=torch.float32))
FIXED = B32 / (1.0 - A32)
TOL = 1e-3  # a float32 iteration drifts by about iters * 6e-8, which is under 1e-3 for 4000 iterations

N_BIG = 2**26  # 256 MiB of float32
CHUNKS = 8


def reference(x: torch.Tensor, iters: int) -> torch.Tensor:
    return FIXED + (x.double() - FIXED) * (A32**iters)


def max_error(out: torch.Tensor, x: torch.Tensor, iters: int) -> float:
    """Largest absolute error of `out` against the float64 reference, computed on the GPU."""
    ref = reference(x.cuda(), iters)
    return float((out.cuda().double() - ref).abs().max().item())


def make_buffers(submission, n):
    h_in = submission.make_pinned(n)
    h_in.normal_()
    h_out = submission.make_pinned(n)
    h_out.fill_(float("nan"))
    d_in = torch.empty(n, dtype=torch.float32, device="cuda")
    d_out = torch.full((n,), float("nan"), dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    return h_in, h_out, d_in, d_out


def run(submission):
    c = Checks()
    names = (
        "make_pinned",
        "chunk_bounds",
        "pipeline_streams",
        "pipeline_staged",
        "pipeline_time_model",
        "overlap_bound_ms",
        "overlap_efficiency",
        "launch_kernel",
    )
    if not c.require(submission, *names):
        return c.finish()

    # --- Pure Python: the chunking and the time model ----------------------------------
    def bounds_cover():
        for n, k in [(1, 8), (7, 8), (8, 8), (1000, 8), (1_000_003, 8), (2**20 + 5, 3), (9, 8)]:
            b = submission.chunk_bounds(n, k)
            if not b or len(b) > k:
                return False, f"n={n}, chunks={k}: got {len(b)} chunks"
            if b[0][0] != 0 or b[-1][1] != n:
                return False, f"n={n}, chunks={k}: the chunks must start at 0 and end at {n}, got {b[0][0]} to {b[-1][1]}"
            for (a0, a1), (b0, b1) in zip(b, b[1:]):
                if a1 != b0:
                    return False, f"n={n}, chunks={k}: a gap or overlap between {a1} and {b0}"
            if any(hi <= lo for lo, hi in b):
                return False, f"n={n}, chunks={k}: an empty chunk"
        return True, "ranges are contiguous, non-empty, and cover [0, n)"

    c.check("chunk_bounds covers the array, including sizes below and between multiples", bounds_cover)

    def model_values():
        m = submission.pipeline_time_model
        cases = [
            ((8.52, 8.52, 8.52, 1), 25.56),
            ((8.52, 8.52, 8.52, 8), 10.65),  # 8.52 * (8 + 2) / 8
            ((8.0, 12.0, 6.0, 6), 14.3333),  # 4.3333 + 5 * 2
        ]
        for args, want in cases:
            got = m(*args)
            if abs(got - want) > 0.02:
                return False, f"pipeline_time_model{args} = {got:.3f}, expected {want}"
        if submission.overlap_bound_ms(8.0, 12.0, 6.0) != 12.0:
            return False, "overlap_bound_ms must return the largest stage"
        eff = submission.overlap_efficiency(8.52, 10.65)
        if abs(eff - 0.8) > 0.005:
            return False, f"overlap_efficiency(8.52, 10.65) = {eff:.3f}, expected 0.800"
        return True, "T = s1 + s2 + s3 + (n - 1) max(s); efficiency = bound / measured"

    c.check("the time model and the bound match the chapter's arithmetic", model_values)

    # --- GPU ----------------------------------------------------------------------------
    require_cuda()

    c.check(
        "make_pinned returns pinned float32 memory",
        lambda: (
            (lambda t: t.is_pinned() and t.dtype == torch.float32 and t.numel() == 1000 and t.device.type == "cpu")(
                submission.make_pinned(1000)
            ),
            "A copy from pageable memory runs but can't overlap, so the pipeline needs tensor.is_pinned() to be True.",
        ),
    )

    # Warm up: compile the Triton kernel and create the streams before anything is timed.
    def warm_up():
        h_in, h_out, d_in, d_out = make_buffers(submission, 4096)
        submission.pipeline_streams(h_in, h_out, d_in, d_out, 4, 2, 2)
        submission.pipeline_staged(h_in, h_out, d_in, d_out, 4, 2)
        torch.cuda.synchronize()
        return True

    # A starter that still raises NotImplementedError stops here with a failed check, not a crash.
    if not c.check("the pipelines run once on a small tensor", warm_up):
        return c.finish()

    # --- Correctness over awkward sizes ---------------------------------------------------
    def correct_for(n):
        def inner():
            for label, fn in (
                ("one chunk, one stream", lambda *a: submission.pipeline_streams(*a, 1, 1)),
                ("8 chunks, 3 streams", lambda *a: submission.pipeline_streams(*a, CHUNKS, 3)),
                ("8 chunks, staged", lambda *a: submission.pipeline_staged(*a, CHUNKS)),
            ):
                h_in, h_out, d_in, d_out = make_buffers(submission, n)
                fn(h_in, h_out, d_in, d_out, 16)
                torch.cuda.synchronize()
                if torch.isnan(h_out).any():
                    return False, f"n={n}, {label}: NaNs in the output, so some chunk was read back before its kernel finished, or never copied"
                err = max_error(h_out, h_in, 16)
                if err > TOL:
                    return False, f"n={n}, {label}: max abs error {err:.2e} (tolerance {TOL:.0e})"
            return True, f"n={n}: all three pipelines match the float64 reference"

        return inner

    for n in (1, 7, 1000, 1_000_003, 2**20 + 5):
        c.check(f"the output is correct for n = {n:,}", correct_for(n))

    # --- A dependency stress test -----------------------------------------------------------
    # The kernel runs for several times as long as the copies, so a copy-out that starts before
    # its kernel finishes reads the NaN sentinel.
    def stress():
        n = 2**24
        iters = 4000
        for label, fn in (
            ("stream per chunk", lambda *a: submission.pipeline_streams(*a, 16, 4)),
            ("stream per stage", lambda *a: submission.pipeline_staged(*a, 16)),
        ):
            h_in, h_out, d_in, d_out = make_buffers(submission, n)
            fn(h_in, h_out, d_in, d_out, iters)
            torch.cuda.synchronize()
            bad = int(torch.isnan(h_out).sum())
            if bad:
                return False, f"{label}: {bad:,} NaNs. A copy-out ran before its kernel finished: check wait_event and wait_stream."
            err = max_error(h_out, h_in, iters)
            if err > TOL:
                return False, f"{label}: max abs error {err:.2e}"
        return True, "no copy-out ran ahead of its kernel"

    c.check("events and stream waits order every chunk (kernel-heavy stress test)", stress)

    # --- Timing on the lab tensor ---------------------------------------------------------------
    h_in, h_out, d_in, d_out = make_buffers(submission, N_BIG)
    nbytes = N_BIG * 4

    h2d_ms = cuda_time(lambda: d_in.copy_(h_in, non_blocking=True), warmup=3, iters=10)
    c.metric("h2d_ms", round(h2d_ms, 3))
    h2d_gbs = bandwidth_gbs(nbytes, h2d_ms)  # one direction: count the bytes once
    c.metric("h2d_gbs", round(h2d_gbs, 1))
    c.metric("pcie_h2d_fraction", round(h2d_gbs / PCIE_GEN4_X16_GBS, 3))
    floor_ms = nbytes / (PCIE_GEN4_X16_GBS * 1e9) * 1e3
    print(
        f"      the PCIe Gen4 x16 floor for {nbytes / 2**20:.0f} MiB is {floor_ms:.2f} ms "
        f"at {PCIE_GEN4_X16_GBS} GB/s per direction",
        flush=True,
    )

    # Choose the kernel's iteration count so its time matches the copy time: the balanced
    # pipeline, which has the most to gain from overlap.
    probe = 256
    t_probe = cuda_time(lambda: submission.launch_kernel(d_in, d_out, probe), warmup=2, iters=5)
    iters = int(max(16, min(20000, round(probe * h2d_ms / t_probe))))
    kernel_ms = cuda_time(lambda: submission.launch_kernel(d_in, d_out, iters), warmup=2, iters=5)
    d2h_ms = cuda_time(lambda: h_out.copy_(d_out, non_blocking=True), warmup=3, iters=10)
    c.metric("kernel_iters", iters)
    c.metric("kernel_ms", round(kernel_ms, 3))
    c.metric("d2h_ms", round(d2h_ms, 3))
    sum_ms = h2d_ms + kernel_ms + d2h_ms
    bound_ms = submission.overlap_bound_ms(h2d_ms, kernel_ms, d2h_ms)
    c.metric("sum_of_stages_ms", round(sum_ms, 3))
    c.metric("bound_ms", round(bound_ms, 3))

    def timed(fn):
        return cuda_time(fn, warmup=2, iters=8)

    one_ms = timed(lambda: submission.pipeline_streams(h_in, h_out, d_in, d_out, iters, CHUNKS, 1))
    many_ms = timed(lambda: submission.pipeline_streams(h_in, h_out, d_in, d_out, iters, CHUNKS, 4))
    staged_ms = timed(lambda: submission.pipeline_staged(h_in, h_out, d_in, d_out, iters, CHUNKS))
    c.metric("one_stream_ms", round(one_ms, 3))
    c.metric("stream_per_chunk_ms", round(many_ms, 3))
    c.metric("staged_ms", round(staged_ms, 3))

    model_ms = submission.pipeline_time_model(h2d_ms, kernel_ms, d2h_ms, CHUNKS)
    best_ms = min(many_ms, staged_ms)
    efficiency = submission.overlap_efficiency(bound_ms, best_ms)
    c.metric("model_ms", round(model_ms, 3))
    c.metric("speedup_over_one_stream", round(one_ms / best_ms, 3))
    c.metric("overlap_efficiency", round(efficiency, 3))
    print(
        f"      one stream {one_ms:.2f} ms, stream per chunk {many_ms:.2f} ms, staged {staged_ms:.2f} ms; "
        f"the model says {model_ms:.2f} ms and the bound is {bound_ms:.2f} ms",
        flush=True,
    )

    # The last timed run left its output behind. It must be right at full size and full length.
    torch.cuda.synchronize()
    err = max_error(h_out, h_in, iters)
    c.add("the full-size output is correct", err <= TOL, f"max abs error {err:.2e}, {iters} iterations")

    # --- The checks on how the reader used streams ------------------------------------------------
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    submission.pipeline_streams(h_in, h_out, d_in, d_out, iters, CHUNKS, 4)
    host_ms = (time.perf_counter() - t0) * 1e3
    torch.cuda.synchronize()
    c.metric("host_return_ms", round(host_ms, 3))
    c.add(
        "the pipeline returns to the host before the GPU finishes",
        host_ms < 0.5 * many_ms,
        f"{host_ms:.2f} ms to queue the work against {many_ms:.2f} ms of GPU time. A larger figure means a "
        "host synchronize in the loop: a pageable copy, .item(), .cpu(), or torch.cuda.synchronize().",
    )

    c.add(
        "a stream per chunk beats one stream",
        many_ms < 0.85 * one_ms,
        f"{many_ms:.2f} ms against {one_ms:.2f} ms. With balanced stages the model predicts "
        f"{submission.pipeline_time_model(h2d_ms, kernel_ms, d2h_ms, CHUNKS) / sum_ms:.2f} of the sequential time.",
    )
    c.add(
        "a stream per stage, joined by events, beats one stream",
        staged_ms < 0.85 * one_ms,
        f"{staged_ms:.2f} ms against {one_ms:.2f} ms",
    )
    c.add(
        "the best pipeline reaches 40% of the max(copy, compute) bound",
        efficiency >= 0.4,
        f"efficiency {efficiency:.2f} = bound {bound_ms:.2f} ms / best {best_ms:.2f} ms. "
        f"The model predicts {bound_ms / model_ms:.2f} for {CHUNKS} equal chunks.",
    )

    # The control: the same pipeline on pageable memory. A metric, not a pass or fail.
    h_in_pg = torch.empty(N_BIG, dtype=torch.float32).copy_(h_in)  # an explicit pageable CPU tensor
    h_out_pg = torch.empty(N_BIG, dtype=torch.float32)  # an explicit pageable CPU tensor
    pageable_ms = timed(lambda: submission.pipeline_streams(h_in_pg, h_out_pg, d_in, d_out, iters, CHUNKS, 4))
    c.metric("pageable_ms", round(pageable_ms, 3))
    print(
        f"      pageable host memory: {pageable_ms:.2f} ms against {many_ms:.2f} ms pinned "
        f"({pageable_ms / many_ms:.2f}x). Pageable copies stage through a driver buffer and block the host.",
        flush=True,
    )

    return c.finish()
