import torch
from lab_common import A100, Checks, close, cuda_time, require_cuda, tflops

STAGE_COUNTS = (1, 2, 3, 4, 5)

# (M, K, N): the first three are tile-sized or smaller, the rest are awkward.
# K and N that aren't multiples of 8 exercise the wrapper's padding, M that
# isn't a multiple of 128 exercises the zero-filled rows, and K = 8 or 64 gives
# fewer k-tiles than a deep pipeline has stages.
SHAPES = [
    (128, 64, 128),
    (1, 8, 8),
    (1, 1, 1),
    (1, 64, 64),
    (100, 72, 136),
    (257, 129, 255),
    (333, 555, 777),
    (1000, 1000, 1000),
    (512, 2048, 512),
]


def simulate_schedule(sub, num_k_tiles, stages):
    """Replay `schedule` on a model ring buffer and return a problem string, or None."""
    sched = sub.schedule(num_k_tiles, stages)
    prologue, steps = sched["prologue"], sched["steps"]
    if len(steps) != num_k_tiles:
        return f"expected {num_k_tiles} steps, got {len(steps)}"
    if stages == 1:
        if prologue:
            return "one stage has no prologue"
        for kt, s in enumerate(steps):
            if (s["load_tile"], s["load_slot"], s["compute_tile"], s["compute_slot"], s["wait_pending"]) != (kt, 0, kt, 0, 0):
                return f"step {kt}: with one stage, load and compute tile {kt} in slot 0 and wait for 0 groups, got {s}"
        return None
    if len(prologue) != stages - 1:
        return f"the prologue commits {stages - 1} groups, got {len(prologue)}"
    slots = [None] * stages  # the k-tile each ring slot holds
    committed = 0
    for slot, tile in enumerate(prologue):
        committed += 1
        if tile is None:
            if slot < num_k_tiles:
                return f"prologue entry {slot} is None but tile {slot} exists"
            continue
        if tile != slot:
            return f"prologue entry {slot} should load tile {slot}, got {tile}"
        slots[slot] = tile
    for kt, s in enumerate(steps):
        if s["kt"] != kt or s["compute_tile"] != kt:
            return f"step {kt} should compute tile {kt}, got {s}"
        # Tile kt is group kt. After the wait, committed - wait_pending groups are done.
        done = committed - s["wait_pending"]
        if done < kt + 1:
            return f"step {kt}: wait_group {s['wait_pending']} leaves tile {kt} possibly in flight"
        if done > kt + 1:
            return f"step {kt}: wait_group {s['wait_pending']} waits for more than tile {kt}, so it stalls the pipeline"
        want = kt + stages - 1
        if s["load_tile"] != (want if want < num_k_tiles else None):
            return f"step {kt}: should load tile {want if want < num_k_tiles else None}, got {s['load_tile']}"
        if s["load_tile"] is not None:
            held = slots[s["load_slot"]]
            if held is not None and held >= kt:
                return f"step {kt}: the copy into slot {s['load_slot']} overwrites tile {held}, which is not consumed yet"
            slots[s["load_slot"]] = s["load_tile"]
        committed += 1  # a group is committed even when the tile is past the end
        if slots[s["compute_slot"]] != kt:
            return f"step {kt}: slot {s['compute_slot']} holds {slots[s['compute_slot']]}, not tile {kt}"
    return None


def run(submission):
    c = Checks()

    # ---- Part 1: the arithmetic and the schedule, which run anywhere --------
    if c.require(submission, "stage_smem_bytes", "max_stages", "schedule"):
        def expect(name, fn, want):
            # Run the student's helper inside c.check, so an unfinished starter fails the check instead of crashing the run.
            def go():
                got = fn()
                return got == want, f"got {got}, expected {want}"
            c.check(name, go)

        expect("one stage holds a 128x40 tile of A and a 32x136 tile of B in bf16: 18944 bytes",
               lambda: submission.stage_smem_bytes(1), 18944)
        expect("four stages need 75776 bytes", lambda: submission.stage_smem_bytes(4), 75776)
        expect("one block per SM: floor(166912 / 18944) = 8 stages", lambda: submission.max_stages(), 8)
        expect("two blocks per SM: floor((83968 - 1024) / 18944) = 4 stages",
               lambda: submission.max_stages(blocks_per_sm=2), 4)
        expect("three blocks per SM: 2 stages", lambda: submission.max_stages(blocks_per_sm=3), 2)
        for stages in STAGE_COUNTS:
            for tiles in (1, 2, 3, 6, 16):
                msg = None
                try:
                    msg = simulate_schedule(submission, tiles, stages)
                except Exception as exc:  # noqa: BLE001
                    msg = f"{type(exc).__name__}: {exc}"
                c.add(f"schedule({tiles} k-tiles, {stages} stages) is race-free and waits no longer than it must", msg is None, msg or "")

    # ---- Part 2: the kernel --------------------------------------------------
    if not c.require(submission, "gemm"):
        return c.finish()
    device = require_cuda()
    torch.backends.cuda.matmul.allow_tf32 = False  # the reference must be exact fp32
    props = torch.cuda.get_device_properties(0)
    print(f"      device = {props.name}", flush=True)
    if props.multi_processor_count != A100.SM_COUNT:
        print(f"      note: this is not the course's A100 ({A100.SM_COUNT} SMs).", flush=True)

    gen = torch.Generator(device="cpu").manual_seed(12)
    for stages in STAGE_COUNTS:
        for m, k, n in SHAPES:
            a = torch.randn(m, k, generator=gen).to(device=device, dtype=torch.bfloat16)
            b = torch.randn(k, n, generator=gen).to(device=device, dtype=torch.bfloat16)
            ref = a.float() @ b.float()
            # bf16 inputs are exact in fp32, so only the order of the fp32 sums differs.
            tol = 1e-3 * max(1.0, ref.abs().max().item())

            def attempt(a=a, b=b, ref=ref, tol=tol, stages=stages):
                out = submission.gemm(a, b, stages)
                torch.cuda.synchronize()
                if out.dtype != torch.float32:
                    return False, f"output dtype {out.dtype}, expected float32"
                return close(out, ref, tol)

            c.check(f"{stages} stage(s), M={m} K={k} N={n}", attempt)

    # ---- Part 3: performance --------------------------------------------------
    size = 4096
    a = torch.randn(size, size, device=device).to(torch.bfloat16)
    b = torch.randn(size, size, device=device).to(torch.bfloat16)
    flops = 2.0 * size**3
    rates = {}
    for stages in STAGE_COUNTS:
        try:
            ms = cuda_time(lambda s=stages: submission.gemm(a, b, s), warmup=5, iters=20)
        except Exception as exc:  # noqa: BLE001
            c.add(f"{stages} stage(s) runs at {size}^3", False, f"{type(exc).__name__}: {exc}")
            continue
        rates[stages] = tflops(flops, ms)
        print(f"      {stages} stage(s): {ms:.3f} ms, {rates[stages]:.1f} TFLOP/s", flush=True)

    cublas_ms = cuda_time(lambda: torch.matmul(a, b), warmup=5, iters=20)
    cublas_rate = tflops(flops, cublas_ms)
    c.metric("cublas_tflops", round(cublas_rate, 1))

    if 1 in rates and any(s in rates for s in (2, 3, 4)):
        piped = {s: rates[s] for s in (2, 3, 4) if s in rates}
        best_stages = max(piped, key=piped.get)
        best = piped[best_stages]
        c.metric("tflops_1stage", round(rates[1], 1))
        c.metric("tflops_best", round(best, 1))
        c.metric("best_stages", best_stages)
        c.metric("speedup", round(best / rates[1], 2))
        c.metric("pct_of_peak", round(100.0 * best / A100.TENSOR_BF16_TFLOPS, 1))
        c.metric("pct_of_cublas", round(100.0 * best / cublas_rate, 1))
        # The FP32 CUDA-core peak is 19.5 TFLOP/s, so a result above it means
        # the tensor cores did the work.
        c.add(
            "the pipelined kernel beats the 19.5 TFLOP/s FP32 CUDA-core peak, so it uses tensor cores",
            best > A100.FP32_TFLOPS,
            f"{best:.1f} TFLOP/s",
        )
        c.add(
            "pipelining does not slow the kernel down (at least 0.9x of one stage)",
            best >= 0.9 * rates[1],
            f"{best / rates[1]:.2f}x with {best_stages} stages",
        )
    else:
        c.add("the 1-stage and a pipelined kernel both run at 4096^3", False, "no timings")
    return c.finish()
