"""Tests for lab 40.

Nine checks run on the CPU with plain Python and torch-CPU. The tenth measures
`torch.cuda.max_memory_allocated` on a GPU and is skipped without one.

Plausible wrong implementations that these checks catch:

- one_adam_copy (check 1): state_bytes counts Adam as one fp32 copy instead of
  two. fp32 training then costs 12 bytes per parameter, not 16.
- sum_only_peak (check 3): peak_activation_bytes adds the stored tensors and
  forgets the layer that a checkpoint rebuilds during the backward pass.
- freed_segments_return (checks 5 and 6): the allocator hands a segment back to
  the device when its last block is freed, so `reserved` shrinks. The real one
  keeps the segment and reuses it.
- no_merge (checks 5 and 6): a free that doesn't merge with its free
  neighbors leaves two small blocks where one large block should be.
- free_over_reserved (check 7): fragmentation measured as free bytes over
  reserved bytes, not as 1 minus the largest free block over all free bytes.
- double_count_free (check 8): the peak replay subtracts on both
  `free_requested` and `free_completed`, so it undercounts the peak.
- transient_ignored (check 9): the checkpoint chooser accepts a set whose
  stored bytes fit but whose rebuild of one layer does not.
"""

from __future__ import annotations

import itertools
import random

from lab_common import Checks

import fixture_snapshot as F

MiB = 1 << 20

NAMES = (
    "state_bytes",
    "activation_layer_bytes",
    "peak_activation_bytes",
    "mlp_stack_saved_bytes",
    "round_size",
    "segment_size",
    "CachingAllocator",
    "run_trace",
    "reserved_and_allocated",
    "largest_free_block",
    "fragmentation_ratio",
    "peak_allocation",
    "classify_oom",
    "choose_checkpoints",
    "measure_peaks",
)


def _fmt(x):
    return repr(x)


def _expect(got, want):
    return (got == want), f"expected {_fmt(want)}, actual {_fmt(got)}"


def _tiny_snapshot(blocks, trace):
    """One large segment. blocks is a list of (MiB, state)."""
    out, addr = [], 0
    for mib, state in blocks:
        size = mib * MiB
        out.append({"address": addr, "size": size, "requested_size": size, "state": state, "frames": []})
        addr += size
    seg = {
        "address": 0,
        "total_size": addr,
        "segment_type": "large",
        "stream": 0,
        "allocated_size": 0,
        "active_size": 0,
        "blocks": out,
    }
    return {"segments": [seg], "device_traces": [trace]}


def _ref_peak(full, boundary, ckpt):
    """The reference peak, written again here so the chooser check is independent."""
    stored = [boundary[i] if i in ckpt else full[i] for i in range(len(full))]
    peak = sum(stored)
    prefix = 0
    for i in range(len(full)):
        prefix += stored[i]
        if i in ckpt:
            peak = max(peak, prefix + full[i])
    return peak


def _best_by_brute_force(full, boundary, cost, budget):
    n = len(full)
    best = None
    for mask in range(1 << n):
        ckpt = {i for i in range(n) if mask >> i & 1}
        if _ref_peak(full, boundary, ckpt) <= budget:
            c = sum(cost[i] for i in ckpt)
            if best is None or c < best:
                best = c
    return best


def run(submission):
    c = Checks()
    if not c.require(submission, *NAMES):
        return c.finish()
    r = submission

    # Check 1 needs only state_bytes. Implementing that one function passes it.
    def state():
        n = 1_000_000
        fp32 = r.state_bytes(n, "fp32")
        mixed = r.state_bytes(n, "mixed")
        want_fp32 = {"weights": 4_000_000, "grads": 4_000_000, "master": 0,
                     "adam_m": 4_000_000, "adam_v": 4_000_000, "total": 16_000_000}
        want_mixed = {"weights": 2_000_000, "grads": 2_000_000, "master": 4_000_000,
                      "adam_m": 4_000_000, "adam_v": 4_000_000, "total": 16_000_000}
        # one_adam_copy fails here: its fp32 total is 12,000,000.
        if dict(fp32) != want_fp32:
            return False, f"fp32 expected {want_fp32}, actual {dict(fp32)}"
        if dict(mixed) != want_mixed:
            return False, f"mixed expected {want_mixed}, actual {dict(mixed)}"
        big = r.state_bytes(7_000_000_000, "mixed")["total"]
        return _expect(big, 112_000_000_000)

    c.check("weights, gradients, and two Adam states cost 16 bytes per parameter", state)

    def activations():
        s, b, h, a = 4096, 1, 4096, 32
        # 34 * s * b * h = 34 * 16,777,216 and 5 * a * s^2 * b = 160 * 16,777,216.
        got = (r.activation_layer_bytes(s, b, h, a), r.activation_layer_bytes(s, b, h, a, flash=True))
        want = (3_254_779_904, 570_425_344)
        ok, msg = _expect(got, want)
        if not ok:
            return False, msg
        try:
            import torch
        except ImportError:
            return True, msg + " (the torch saved-tensor check was skipped: torch is not installed)"
        layers, tokens, hidden = 3, 8, 16
        seen, params = {}, set()
        model = torch.nn.Sequential(
            *[torch.nn.Sequential(torch.nn.Linear(hidden, hidden, bias=False), torch.nn.GELU()) for _ in range(layers)]
        )
        for p in model.parameters():
            params.add(p.data_ptr())

        def pack(t):
            if t.data_ptr() not in params:
                seen[t.data_ptr()] = t.numel() * t.element_size()
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            model(torch.randn(tokens, hidden))
        measured = sum(seen.values())
        predicted = r.mlp_stack_saved_bytes(layers, tokens, hidden, 4)
        ok2, msg2 = _expect(predicted, measured)
        return ok2, "layer bytes: " + msg + "; torch saved tensors: " + msg2

    c.check("per-layer activation bytes match the formula and torch's saved tensors", activations)

    def peak():
        full, bnd = [10, 10, 10, 10], [2, 2, 2, 2]
        # Hand-derived. Stored bytes, then the rebuild of each checkpointed layer.
        # none: 40. all: stored 8, last rebuild 8 + 10 = 18.
        # {0, 1}: stored 24, rebuild of 1 is 4 + 10 = 14, so 24.
        # {2, 3}: stored 24, rebuild of 2 is 22 + 10 = 32, of 3 is 24 + 10 = 34.
        cases = [(set(), 40), ({0, 1, 2, 3}, 18), ({0, 1}, 24), ({2, 3}, 34)]
        for ckpt, want in cases:
            got = r.peak_activation_bytes(full, bnd, ckpt)
            if got != want:
                # sum_only_peak returns 8 for the all-checkpointed case.
                return False, f"checkpointed layers {sorted(ckpt)}: expected {want}, actual {got}"
        return True, "expected 40, 18, 24, 34, actual 40, 18, 24, 34"

    c.check("the activation peak is the stored bytes or the rebuild, whichever is larger", peak)

    def rounding():
        rs = [r.round_size(n) for n in (1, 512, 513, 1_000_000)]
        ok, msg = _expect(rs, [512, 512, 1024, 1_000_448])
        if not ok:
            return False, "round_size " + msg
        sizes = [r.segment_size(n) for n in (1 * MiB, 1 * MiB + 512, 10 * MiB - 512, 10 * MiB, 11 * MiB)]
        want = [2 * MiB, 20 * MiB, 20 * MiB, 10 * MiB, 12 * MiB]
        return _expect(sizes, want)

    c.check("sizes round to 512 bytes and map to 2 MiB, 20 MiB, or 2 MiB multiples", rounding)

    def never_returns():
        ev = [("alloc", "x", 1000), ("alloc", "y", 1), ("alloc", "z", 3 * MiB),
              ("free", "x"), ("free", "y"), ("free", "z"), ("alloc", "w", 2 * MiB + MiB // 2)]
        out = r.run_trace(ev)
        want_alloc = [1024, 1536, 3_147_264, 3_146_240, 3_145_728, 0, 2_621_440]
        want_res = [2 * MiB, 2 * MiB] + [22 * MiB] * 5
        if list(out["allocated"]) != want_alloc:
            return False, f"allocated expected {want_alloc}, actual {list(out['allocated'])}"
        # freed_segments_return fails here: reserved falls once everything is freed.
        if list(out["reserved"]) != want_res:
            return False, f"reserved expected {want_res}, actual {list(out['reserved'])}"
        return True, "allocated and reserved match; reserved stays 22 MiB after every free"

    c.check("freed memory goes back to the cache and reserved never shrinks", never_returns)

    def fragmentation():
        out = r.run_trace(F.ALLOC_TRACE)
        want_alloc = [m * MiB for m in (12, 24, 36, 48, 36, 24, 40, 44, 28, 48)]
        want_res = [m * MiB for m in (12, 24, 36, 48, 48, 48, 64, 64, 64, 84)]
        if list(out["allocated"]) != want_alloc:
            return False, f"allocated expected {[a // MiB for a in want_alloc]} MiB, actual {[a // MiB for a in out['allocated']]} MiB"
        if list(out["reserved"]) != want_res:
            return False, f"reserved expected {[a // MiB for a in want_res]} MiB, actual {[a // MiB for a in out['reserved']]} MiB"
        if (out["peak_allocated"], out["peak_reserved"]) != (48 * MiB, 84 * MiB):
            return False, f"peaks expected (48, 84) MiB, actual ({out['peak_allocated'] // MiB}, {out['peak_reserved'] // MiB}) MiB"
        # Block f (4 MiB) best-fits the first 12 MiB segment and splits it into 4 live and 8 free.
        seg0 = out["allocator"].snapshot()["segments"][0]
        layout = [(b["size"] // MiB, b["state"]) for b in seg0["blocks"]]
        if layout != [(4, "active_allocated"), (8, "inactive")]:
            return False, f"first segment expected [(4, active), (8, inactive)], actual {layout}"
        # With a device of 80 MiB, g's cudaMalloc fails, the allocator frees the two fully free
        # segments (12 and 16 MiB), and retries: 64 - 28 + 20 = 56 MiB reserved.
        tight = r.run_trace(F.ALLOC_TRACE, capacity=80 * MiB)
        if tight["oom_at"] is not None or tight["reserved"][-1] != 56 * MiB:
            return False, f"capacity 80 MiB expected no OOM and reserved 56 MiB, actual oom_at={tight['oom_at']}, reserved={tight['reserved'][-1] // MiB} MiB"
        # With 40 MiB, the fourth 12 MiB block does not fit and nothing is cached to release.
        small = r.run_trace(F.ALLOC_TRACE, capacity=40 * MiB)
        return _expect(small["oom_at"], 3)

    c.check("the simulator reproduces allocated and reserved, splits, and retries after an OOM", fragmentation)

    def free_blocks():
        snap = F.SNAPSHOT
        got = (
            r.reserved_and_allocated(snap),
            r.largest_free_block(snap),
            round(r.fragmentation_ratio(snap), 4),
            r.classify_oom(snap),
        )
        # reserved 2 + 20 + 20 + 64 = 106 MiB, active 55 MiB, free 51 MiB, largest free 24 MiB,
        # fragmentation 1 - 24 / 51 = 27 / 51 = 0.5294. The OOM asks for 30 MiB: 51 free, 24 contiguous.
        want = ((106 * MiB, 55 * MiB), 24 * MiB, 0.5294, "fragmentation")
        ok, msg = _expect(got, want)
        if not ok:
            return False, msg
        # free_over_reserved gives 51 / 106 = 0.4811 above.
        tiny = _tiny_snapshot([(2, "inactive"), (3, "active_allocated"), (3, "inactive")],
                              [{"action": "oom", "addr": 0, "size": 6 * MiB, "stream": 0, "frames": [], "device_free": 0}])
        got2 = (r.largest_free_block(tiny), round(r.fragmentation_ratio(tiny), 4), r.classify_oom(tiny))
        ok2, msg2 = _expect(got2, (3 * MiB, 0.4, "capacity"))
        if not ok2:
            return False, "second snapshot " + msg2
        full = _tiny_snapshot([(4, "active_allocated")], [])
        got3 = (r.largest_free_block(full), r.fragmentation_ratio(full), r.classify_oom(full))
        return _expect(got3, (0, 0.0, None))

    c.check("a snapshot gives the largest free block, the fragmentation ratio, and the OOM kind", free_blocks)

    def peak_event():
        got = r.peak_allocation(F.SNAPSHOT)
        want = {"peak_bytes": 79 * MiB, "event": 13, "size": 24 * MiB, "frame": "loss.py:31 log_softmax"}
        have = {k: got.get(k) for k in want}
        # double_count_free finds 71 MiB: it subtracts the 8 MiB block twice.
        ok, msg = _expect(have, want)
        if not ok:
            return False, msg
        tr = [
            {"action": "alloc", "addr": 1, "size": 5, "stream": 0, "frames": [{"filename": "a.py", "line": 1, "name": "f"}]},
            {"action": "free_requested", "addr": 1, "size": 5, "stream": 0, "frames": []},
            {"action": "free_completed", "addr": 1, "size": 5, "stream": 0, "frames": []},
            {"action": "alloc", "addr": 2, "size": 3, "stream": 0, "frames": [{"filename": "b.py", "line": 2, "name": "g"}]},
        ]
        got2 = r.peak_allocation({"segments": [], "device_traces": [tr]})
        return _expect((got2["peak_bytes"], got2["event"], got2["frame"]), (5, 0, "a.py:1 f"))

    c.check("the trace replay finds the peak and the allocation that reached it", peak_event)

    def chooser():
        full, bnd, cost = [10] * 4, [2] * 4, [1] * 4
        # Budget 24: two layers, for example {0, 1}. Budget 18 and 16: three, {0, 1, 2}, whose
        # stored bytes are 16 and whose rebuild of layer 2 is 6 + 10 = 16. Budget 15: none fits.
        hand = [(24, 2), (18, 3), (16, 3), (15, None)]
        for budget, want in hand:
            got = r.choose_checkpoints(full, bnd, cost, budget)
            have = None if got is None else got[1]
            if have != want:
                return False, f"hand case budget {budget}: expected cost {want}, actual {got!r}"
        rng = random.Random(40)
        tried = 0
        for _ in range(40):
            n = rng.randint(3, 8)
            fl = [rng.randint(4, 12) for _ in range(n)]
            bd = [rng.randint(1, min(3, f)) for f in fl]
            co = [rng.randint(1, 9) for _ in range(n)]
            floor = min(_ref_peak(fl, bd, {i for i in range(n) if m >> i & 1}) for m in range(1 << n))
            budget = rng.randint(floor - 2, sum(fl))
            want = _best_by_brute_force(fl, bd, co, budget)
            got = r.choose_checkpoints(fl, bd, co, budget)
            tried += 1
            if want is None:
                if got is not None:
                    return False, f"budget {budget} is infeasible, expected None, actual {got!r} for full={fl} boundary={bd}"
                continue
            if got is None:
                return False, f"expected cost {want}, actual None for full={fl} boundary={bd} cost={co} budget={budget}"
            chosen, spent = got
            true_peak = _ref_peak(fl, bd, set(chosen))
            # transient_ignored fails here: its set fits in stored bytes but not in the rebuild.
            if true_peak > budget:
                return False, f"chosen layers {list(chosen)} peak at {true_peak}, over the budget {budget}, for full={fl} boundary={bd}"
            if spent != want or sum(co[i] for i in chosen) != spent:
                return False, f"expected cost {want}, actual {spent} for full={fl} boundary={bd} cost={co} budget={budget}"
        return True, f"matched brute force on {tried} seeded cases and 4 hand cases"

    c.check("the chooser matches brute force on small cases", chooser)

    _gpu_section(c, r)
    return c.finish()


def _skip_reason():
    try:
        import torch
    except ImportError:
        return "SKIP: the GPU measurement was not run. torch is not installed."
    if not torch.cuda.is_available():
        return "SKIP: the GPU measurement was not run. No CUDA GPU is visible."
    return None


def _gpu_section(c, r):
    reason = _skip_reason()
    if reason is not None:
        print(reason, flush=True)
        c.metric("peak_reduction_mib", "skipped (no GPU)")
        return

    layers, tokens, hidden = 16, 4096, 2048
    full = [2 * tokens * hidden * 4] * layers  # Linear input plus GELU input, fp32
    bnd = [tokens * hidden * 4] * layers  # the input a checkpoint keeps
    plain_pred = r.peak_activation_bytes(full, bnd, set())
    ckpt_pred = r.peak_activation_bytes(full, bnd, set(range(layers)))
    want = plain_pred - ckpt_pred

    def measured():
        plain, ckpt = r.measure_peaks(layers, tokens, hidden)
        got = plain - ckpt
        c.metric("peak_plain_mib", round(plain / MiB, 1))
        c.metric("peak_ckpt_mib", round(ckpt / MiB, 1))
        c.metric("peak_reduction_mib", round(got / MiB, 1))
        ok = got > 0 and abs(got - want) <= 0.35 * want
        return ok, (
            f"expected a reduction within 35% of {want / MiB:.0f} MiB, actual {got / MiB:.0f} MiB "
            f"(peaks {plain / MiB:.0f} MiB plain, {ckpt / MiB:.0f} MiB checkpointed)"
        )

    c.check("checkpointing cuts the measured peak by about what the calculator predicts", measured)
