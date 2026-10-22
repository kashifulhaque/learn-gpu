import torch
from lab_common import A100, Checks, close, require_cuda

# Configurations worked out by hand in the chapter from the cc 8.0 limits in
# docs/FACTS.md section 4.2 and the arithmetic in section 4.3:
# (threads per block, registers per thread, shared memory bytes, blocks per SM, limiter or None).
# A limiter of None means two limits tie, so the name isn't checked.
KNOWN = [
    (256, 32, 0, 8, None),  # 8192 registers per block; 8 by registers and 8 by warps
    (256, 64, 0, 4, "registers"),
    (256, 128, 0, 2, "registers"),
    (256, 255, 0, 1, "registers"),  # 255 is charged as 256 per thread
    (256, 32, 48 * 1024, 3, "shared_memory"),  # 48 KiB + 1 KiB reserved = 49 KiB; 164 / 49 = 3.35
    (256, 32, 163 * 1024, 1, "shared_memory"),  # the per-block maximum fills the SM
    (32, 32, 0, 32, "blocks"),  # one-warp blocks hit the 32-block cap at 50% occupancy
    (128, 40, 0, 12, "registers"),  # 1280 registers per warp; 48 warps by registers
    (1024, 32, 0, 2, None),
    (1024, 255, 0, 0, "registers"),  # 255 registers x 1024 threads exceeds the register file
]

# Ways the calculator is cross-checked against cudaOccupancyMaxActiveBlocksPerMultiprocessor.
PROBE_VARIANTS = [0, 1, 2, 3]
BLOCK_SIZES = [32, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
DYN_SMEM = [0, 8 * 1024, 24 * 1024, 48 * 1024]

BLOCK = 256  # the kernel's block size; the reference needs it to know each thread's warp and lane


def reference(x: torch.Tensor, ways: int, by_warp: bool, iters: int, coef: torch.Tensor):
    """Closed form of v <- c * v + 1 applied `iters` times, in float64."""
    n = x.numel()
    tid = torch.arange(n) % BLOCK
    lane, warp = tid % 32, tid // 32
    group = (warp % ways) if by_warp else (lane % ways)
    c = coef.double().cpu()[group]
    cn = c**iters
    return (cn * x.double().cpu() + (1.0 - cn) / (1.0 - c)).float()


def run(submission):
    c = Checks()
    needed = [
        "occupancy",
        "max_regs_for_occupancy",
        "run_branchy",
        "probe_registers",
        "probe_static_smem",
        "api_blocks_per_sm",
        "branch_time_ms",
        "divergence_slowdown",
    ]
    if not c.require(submission, *needed):
        return c.finish()
    require_cuda()

    # -- Part 1: the calculator against hand-worked configurations -------------
    def known_case(threads, regs, smem, blocks, limiter):
        got = submission.occupancy(threads, regs, smem)
        ok = got["blocks_per_sm"] == blocks
        detail = f"got {got['blocks_per_sm']} blocks per SM, expected {blocks}"
        if ok and limiter is not None:
            ok = got["limiter"] == limiter
            detail = f"got limiter {got['limiter']!r}, expected {limiter!r}"
        if ok:
            want_occ = blocks * -(-threads // 32) / 64
            ok = abs(got["occupancy"] - want_occ) < 1e-9
            detail = f"got occupancy {got['occupancy']}, expected {want_occ}"
        return ok, detail

    for threads, regs, smem, blocks, limiter in KNOWN:
        label = f"occupancy({threads}, {regs}, {smem // 1024} KiB) is {blocks} blocks per SM"
        c.check(label, lambda a=(threads, regs, smem, blocks, limiter): known_case(*a))

    def regs_budget():
        got = [submission.max_regs_for_occupancy(256, t) for t in (1.0, 0.5, 0.25)]
        return got == [32, 64, 128], f"got {got} for targets 100%, 50%, 25%; expected [32, 64, 128]"

    c.check("max_regs_for_occupancy finds the register budget for 100%, 50%, 25%", regs_budget)

    # -- Part 2: the calculator against the CUDA occupancy API -----------------
    regs = {v: submission.probe_registers(v) for v in PROBE_VARIANTS}
    smem_static = {v: submission.probe_static_smem(v) for v in PROBE_VARIANTS}
    c.metric("probe_regs_max", max(regs.values()))
    print(f"      registers per thread of the four probe kernels: {regs}", flush=True)

    agree = {"regs_only": [0, 0], "with_smem": [0, 0]}
    mismatches = []
    for v in PROBE_VARIANTS:
        for block in BLOCK_SIZES:
            for dyn in DYN_SMEM:
                api = submission.api_blocks_per_sm(v, block, dyn)
                if api < 0:
                    continue
                mine = submission.occupancy(block, regs[v], smem_static[v] + dyn)["blocks_per_sm"]
                bucket = agree["regs_only" if dyn == 0 else "with_smem"]
                bucket[1] += 1
                if mine == api:
                    bucket[0] += 1
                else:
                    mismatches.append((v, regs[v], block, dyn, mine, api))
    for v, r, block, dyn, mine, api in mismatches[:12]:
        print(
            f"      mismatch: probe {v} ({r} regs), block {block}, {dyn // 1024} KiB dynamic smem: "
            f"calculator {mine}, API {api}",
            flush=True,
        )
    pct = lambda pair: 100.0 * pair[0] / pair[1] if pair[1] else 0.0  # noqa: E731
    reg_pct, smem_pct = pct(agree["regs_only"]), pct(agree["with_smem"])
    c.metric("api_agree_pct", round(reg_pct, 1))
    c.metric("api_agree_smem_pct", round(smem_pct, 1))
    c.add(
        "the calculator agrees with the occupancy API when no shared memory is requested",
        agree["regs_only"][1] > 0 and reg_pct >= 80.0,
        f"{agree['regs_only'][0]} of {agree['regs_only'][1]} configurations agree ({reg_pct:.0f}%, need 80%)",
    )
    c.add(
        "the calculator agrees with the occupancy API on at least half of the shared-memory configurations",
        agree["with_smem"][1] > 0 and smem_pct >= 50.0,
        f"{agree['with_smem'][0]} of {agree['with_smem'][1]} agree ({smem_pct:.0f}%, need 50%)",
    )

    # -- Part 3: the branchy kernel is correct ---------------------------------
    coef = (0.5 + 0.01 * torch.arange(32, dtype=torch.float32)).cuda()
    for n, ways, by_warp in [(1, 1, False), (1000, 2, False), (4097, 4, False), (4097, 32, False), (4097, 4, True), (1 << 16, 32, True)]:
        iters = 256
        x = torch.rand(n, device="cuda")

        def case(n=n, ways=ways, by_warp=by_warp, iters=iters, x=x):
            out = submission.run_branchy(x, ways, by_warp, iters)
            torch.cuda.synchronize()
            return close(out.cpu(), reference(x, ways, by_warp, iters, coef), 1e-4)

        kind = "warp-uniform" if by_warp else "lane-diverged"
        c.check(f"branchy is correct for n={n}, {ways} ways, {kind}", case)

    # -- Part 4: what divergence costs -----------------------------------------
    uniform_ms = submission.branch_time_ms(1, False)
    c.metric("uniform_ms", round(uniform_ms, 4))
    c.add("a uniform launch takes measurable time", uniform_ms > 0.0, f"{uniform_ms:.4f} ms")

    slow = {w: submission.divergence_slowdown(w) for w in (2, 4, 8, 16, 32)}
    print(f"      lane-diverged slowdown by ways: { {w: round(s, 2) for w, s in slow.items()} }", flush=True)
    c.metric("slowdown_2way", round(slow[2], 2))
    c.metric("slowdown_32way", round(slow[32], 2))
    # Every branch costs a full pass over the warp's instructions, so the slowdown
    # should track the number of ways. Half of that prediction leaves a wide margin.
    for w, s in slow.items():
        c.add(
            f"{w}-way divergence costs at least {w / 2:.0f}x (the prediction is {w}x)",
            s >= w / 2,
            f"measured {s:.2f}x",
        )

    by_warp_ms = submission.branch_time_ms(32, True)
    ratio = by_warp_ms / uniform_ms
    c.metric("by_warp_32way_ratio", round(ratio, 2))
    c.add(
        "32 branches chosen per warp cost about the same as one (at most 1.5x)",
        ratio <= 1.5,
        f"{by_warp_ms:.4f} ms against {uniform_ms:.4f} ms, ratio {ratio:.2f}",
    )
    return c.finish()
