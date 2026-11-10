"""Tests for lab 39.

The CPU checks need no GPU. The two kernel checks at the end need CUDA and
are skipped, not failed, without it.

release_only_message_passing is a plausible wrong must_stay_ordered. It
orders a release store against the earlier writes and forgets that the
reader's acquire load must order the later reads. On message passing the
reader's two loads then swap, and the weak outcome r0 = 1, r1 = 0 stays
possible. The weak-model check fails that implementation.

count_reset_as_release is a plausible wrong grid barrier, the classic reuse
bug. The last block resets the counter to 0, and the waiting blocks spin
until the counter reads 0. A fast block that leaves and re-arrives for the
next round bumps the counter before a slow waiter has looked, so the slow
waiter never sees 0. The barrier check fails it on a deadlock.

no_generation is the other plausible wrong barrier. It releases with a flag
that nothing clears, so round 2 passes immediately. The barrier check fails
it on a block that passed before another block's flag was written.
"""

from __future__ import annotations

from fixture_gridsim import run_grid
from lab_common import Checks

NAMES = (
    "amdahl_speedup",
    "brent_bounds",
    "reduction_work_span",
    "scan_work_span",
    "must_stay_ordered",
    "allowed_outcomes",
    "generation_barrier",
    "fits_cooperative_launch",
    "mp_orders",
)

R = "relaxed"


def mp(store_order: str, load_order: str, fenced: bool = False):
    """Message passing: T0 writes x then y, T1 reads y then x."""
    t0 = [("st", "x", 1, R)] + ([("fence", "sc")] if fenced else []) + [("st", "y", 1, store_order)]
    t1 = [("ld", "y", "r0", load_order)] + ([("fence", "sc")] if fenced else []) + [("ld", "x", "r1", R)]
    return [t0, t1]


def sb(store_order: str, load_order: str, fenced: bool = False):
    """Store buffering: each thread writes its variable, then reads the other's."""
    fence = [("fence", "sc")] if fenced else []
    return [
        [("st", "x", 1, store_order)] + fence + [("ld", "y", "r0", load_order)],
        [("st", "y", 1, store_order)] + fence + [("ld", "x", "r1", load_order)],
    ]


def lb(store_order: str, load_order: str):
    """Load buffering: each thread reads one variable, then writes the other."""
    return [
        [("ld", "x", "r0", load_order), ("st", "y", 1, store_order)],
        [("ld", "y", "r1", load_order), ("st", "x", 1, store_order)],
    ]


ALL = {(0, 0), (0, 1), (1, 0), (1, 1)}
MP_SC = {(0, 0), (0, 1), (1, 1)}  # (r0, r1): never r0 = 1 with r1 = 0
SB_SC = {(0, 1), (1, 0), (1, 1)}  # never both 0
LB_SC = {(0, 0), (0, 1), (1, 0)}  # never both 1


def _amdahl(f: float, p: float) -> float:
    return 1.0 / ((1.0 - f) + f / p)


def _shown(values) -> str:
    return str(sorted(values))


def _outcomes_case(reader, label, threads, model, want):
    try:
        got = reader.allowed_outcomes(threads, model)
        got = {tuple(o) for o in got}
    except Exception as exc:  # noqa: BLE001
        return f"{label}: expected {_shown(want)}, actual {type(exc).__name__}: {exc}"
    if got != set(want):
        return f"{label}: expected {_shown(want)}, actual {_shown(got)}"
    return None


def run(submission):
    c = Checks()
    if not c.require(submission, *NAMES):
        return c.finish()
    reader = submission

    # Check 1 calls only amdahl_speedup. Implementing that one function passes it.
    def amdahl():
        cases = ((0.95, 8), (0.9, 1), (0.5, 1_000_000), (1.0, 108))
        for f, p in cases:
            want = _amdahl(f, p)
            try:
                got = float(reader.amdahl_speedup(f, p))
            except Exception as exc:  # noqa: BLE001
                return False, f"expected {want:.4f} for f={f}, p={p}, actual {type(exc).__name__}: {exc}"
            if abs(got - want) > 1e-6 * max(1.0, want):
                return False, f"expected {want:.4f} for f={f}, p={p}, actual {got:.4f}"
        return True, "expected 5.9259 for f=0.95 on 8 processors, actual 5.9259"

    c.check("Amdahl's law: f = 0.95 on 8 processors is 5.93 times", amdahl)

    def work_span():
        try:
            lo, hi = reader.brent_bounds(1000, 10, 8)
            red = reader.reduction_work_span(1 << 26)
            ks = reader.scan_work_span(256, "kogge_stone")
            bl = reader.scan_work_span(256, "blelloch")
            serial = reader.scan_work_span(256, "serial")
        except Exception as exc:  # noqa: BLE001
            return False, f"expected (125.0, 135.0) and (67108863, 26), actual {type(exc).__name__}: {exc}"
        want = ((125.0, 135.0), (67108863, 26), (1793, 8), (510, 16), (255, 255))
        got = ((float(lo), float(hi)), tuple(red), tuple(ks), tuple(bl), tuple(serial))
        if got != want:
            return False, f"expected {want}, actual {got}"
        return True, f"expected {want}, actual {got}"

    c.check("Brent's bound, and the work and span of a reduction and of three scans", work_span)

    def ordering():
        st_rel = ("st", "y", 1, "release")
        st_rlx = ("st", "y", 1, R)
        ld_acq = ("ld", "y", "r0", "acquire")
        ld_rlx = ("ld", "y", "r0", R)
        x_ld = ("ld", "x", "r1", R)
        x_st = ("st", "x", 1, R)
        cases = (
            (("sc", x_st, st_rlx), True, "sc orders everything"),
            (("weak", x_st, st_rlx), False, "relaxed stores to different addresses reorder"),
            (("weak", x_st, ("st", "x", 2, R)), True, "same address stays in order"),
            (("weak", x_st, st_rel), True, "a release store waits for the earlier write"),
            (("weak", ld_acq, x_ld), True, "an acquire load comes before the later read"),
            (("weak", ld_rlx, x_ld), False, "relaxed loads reorder"),
            (("weak", st_rlx, x_ld), False, "a load may pass an earlier store to another address"),
            (("weak", x_st, ("fence", "sc")), True, "a fence orders both sides"),
        )
        for (model, earlier, later), want, why in cases:
            try:
                got = bool(reader.must_stay_ordered(earlier, later, model))
            except Exception as exc:  # noqa: BLE001
                return False, f"expected {want} ({why}), actual {type(exc).__name__}: {exc}"
            if got != want:
                return False, f"expected {want} ({why}), actual {got}"
        return True, "expected 8 of 8 orderings, actual 8 of 8"

    c.check("the reorder rule: what may not move past what", ordering)

    def sequential():
        for label, threads, want in (
            ("MP under sc", mp(R, R), MP_SC),
            ("SB under sc", sb(R, R), SB_SC),
            ("LB under sc", lb(R, R), LB_SC),
        ):
            problem = _outcomes_case(reader, label, threads, "sc", want)
            if problem:
                return False, problem
        return True, f"expected MP {_shown(MP_SC)}, actual {_shown(MP_SC)}"

    c.check("sequential consistency forbids r0 = 1 with r1 = 0 in MP, both zeros in SB, both ones in LB", sequential)

    def weak():
        # release_only_message_passing fails the half-sided MP cases below.
        cases = (
            ("MP, relaxed", mp(R, R), ALL),
            ("MP, release store only", mp("release", R), ALL),
            ("MP, acquire load only", mp(R, "acquire"), ALL),
            ("MP, release store and acquire load", mp("release", "acquire"), MP_SC),
            ("MP, fence on each side", mp(R, R, fenced=True), MP_SC),
            ("SB, relaxed", sb(R, R), ALL),
            ("SB, release and acquire", sb("release", "acquire"), ALL),
            ("SB, fence on each side", sb(R, R, fenced=True), SB_SC),
            ("LB, relaxed", lb(R, R), ALL),
            ("LB, acquire loads and release stores", lb("release", "acquire"), LB_SC),
        )
        for label, threads, want in cases:
            problem = _outcomes_case(reader, label, threads, "weak", want)
            if problem:
                return False, problem
        return True, "expected 10 of 10 outcome sets, actual 10 of 10"

    c.check("the weak model allows the weak outcomes, and release/acquire or fences remove the right ones", weak)

    def barrier_safe():
        # count_reset_as_release and no_generation fail here: the first
        # deadlocks on round 2, the second lets a block through early.
        for n in (2, 3, 5, 8):
            for seed in range(120):
                try:
                    result = run_grid(reader.generation_barrier, n, n, 3, seed)
                except Exception as exc:  # noqa: BLE001
                    return False, f"expected a finished run, actual {type(exc).__name__}: {exc}"
                if result["deadlocked"]:
                    return False, (
                        f"expected {n} blocks to finish 3 rounds, actual a deadlock "
                        f"(seed {seed}, {result['finished']} finished)"
                    )
                if result["violations"]:
                    block, rnd, other = result["violations"][0]
                    return False, (
                        f"expected every flag set, actual block {block} left round {rnd} "
                        f"before block {other} arrived (n {n}, seed {seed})"
                    )
        return True, "expected 480 runs with 0 early exits and 0 deadlocks, actual 480 runs with 0"

    c.check("the generation barrier never lets a block pass early, over 480 seeded interleavings", barrier_safe)

    def residency():
        try:
            fits = (
                reader.fits_cooperative_launch(864, 8, 108),
                reader.fits_cooperative_launch(865, 8, 108),
            )
        except Exception as exc:  # noqa: BLE001
            return False, f"expected (True, False) for 864 and 865 blocks, actual {type(exc).__name__}: {exc}"
        if fits != (True, False):
            return False, f"expected (True, False) for 864 and 865 blocks, actual {fits}"
        dead = run_grid(reader.generation_barrier, 8, 4, 3, 7)
        if not dead["deadlocked"]:
            return False, (
                "expected a deadlock with 8 blocks and room for 4, "
                f"actual {dead['finished']} blocks finished"
            )
        live = run_grid(reader.generation_barrier, 8, 8, 3, 7)
        if live["deadlocked"]:
            return False, "expected 8 blocks with room for 8 to finish, actual a deadlock"
        return True, "expected a deadlock at 4 of 8 resident, actual a deadlock"

    c.check("a barrier over more blocks than fit on the GPU deadlocks, so grid.sync needs co-residency", residency)

    def orders():
        try:
            store_order, load_order = reader.mp_orders()
            threads = mp(store_order, load_order)
            got = {tuple(o) for o in reader.allowed_outcomes(threads, "weak")}
        except Exception as exc:  # noqa: BLE001
            return False, f"expected ('release', 'acquire'), actual {type(exc).__name__}: {exc}"
        if (1, 0) in got:
            return False, (
                f"expected the weak outcome (1, 0) to be impossible, actual it is allowed "
                f"with store {store_order!r} and load {load_order!r}"
            )
        return True, f"expected no (1, 0), actual {_shown(got)}"

    c.check("the memory orders you chose for the CUDA kernel forbid the weak outcome in your own model", orders)

    _gpu_section(c, reader)
    return c.finish()


def _skip_reason() -> str | None:
    try:
        import torch
    except ImportError:
        return "SKIP: the CUDA litmus kernel and the grid barriers were not run. torch is not installed, so there is no GPU."
    if not torch.cuda.is_available():
        return "SKIP: the CUDA litmus kernel and the grid barriers were not run. No CUDA GPU is visible."
    major, _ = torch.cuda.get_device_capability()
    if major < 7:
        return (
            "SKIP: the CUDA litmus kernel and the grid barriers were not run. "
            f"They need compute capability 7.0 or later for acquire and release, and the GPU is {major}.x."
        )
    return None


def _gpu_section(c: Checks, reader) -> None:
    reason = _skip_reason()
    if reason is not None:
        print(reason, flush=True)
        for key in ("mp_relaxed_weak", "mp_ordered_forbidden", "barrier_atomic_violations", "coresident_blocks"):
            c.metric(key, "skipped (no GPU)")
        return

    from fixture_cuda import CPP_SRC, FUNCTIONS, build_source
    from lab_common import load_cuda

    store_order, load_order = reader.mp_orders()
    try:
        source = build_source(store_order, load_order)
    except ValueError as exc:
        c.add("the CUDA message-passing kernel never shows the forbidden outcome with your release and acquire", False, f"expected a store order and a load order the kernel accepts, actual {exc}")
        return
    mod = load_cuda("memory_model", source, CPP_SRC, FUNCTIONS)
    pairs, launches = 1 << 17, 100  # 13.1 million message-passing trials per variant

    def mp_kernel():
        relaxed = mod.mp_litmus(pairs, launches, False).tolist()
        ordered = mod.mp_litmus(pairs, launches, True).tolist()
        c.metric("mp_relaxed_weak", int(relaxed[1]))
        c.metric("mp_ordered_forbidden", int(ordered[1]))
        if ordered[0] < 1000:
            return False, f"expected at least 1000 consumers to see the flag, actual {ordered[0]}"
        if ordered[1] != 0:
            return False, (
                f"expected 0 forbidden outcomes with {store_order} and {load_order}, "
                f"actual {ordered[1]} of {ordered[0]}"
            )
        return True, (
            f"expected 0 forbidden of {ordered[0]} observed, actual 0. "
            f"The relaxed run saw {relaxed[1]} of {relaxed[0]}, a metric and not a requirement."
        )

    c.check("the CUDA message-passing kernel never shows the forbidden outcome with your release and acquire", mp_kernel)

    def grid_barriers():
        import torch

        sms = torch.cuda.get_device_properties(0).multi_processor_count
        threads = 256
        resident = int(mod.coresident_blocks(threads))
        c.metric("coresident_blocks", resident)
        atomic_bad = int(mod.grid_barrier(2 * sms, threads, 20, False).item())
        coop_bad = int(mod.grid_barrier(resident, threads, 20, True).item())
        c.metric("barrier_atomic_violations", atomic_bad)
        c.metric("barrier_coop_violations", coop_bad)
        try:
            mod.grid_barrier(resident + sms, threads, 1, True)
            rejected = False
        except RuntimeError:
            rejected = True
        c.metric("cooperative_too_large_rejected", rejected)
        if atomic_bad or coop_bad:
            return False, f"expected 0 early exits, actual {atomic_bad} (atomics) and {coop_bad} (cooperative groups)"
        return True, f"expected 0 early exits over 20 rounds, actual 0 and 0, with {resident} co-resident blocks"

    c.check("a barrier from atomics and cooperative groups' grid sync both hold for 20 rounds", grid_barriers)
