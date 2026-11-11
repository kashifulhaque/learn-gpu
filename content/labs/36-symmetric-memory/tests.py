"""Tests for lab 36.

Three plausible wrong implementations appear below. Each check that uses one
asserts that the wrong implementation fails the property, so you can see that
the check can tell right from wrong.

signal_without_fence: puts data, then signals, and never fences. Under the
signals_first policy the signal arrives before the data, and a consumer that
sees the signal reads stale memory. It is the fence-versus-quiet mix-up: quiet
completes everything but waits, and a fence orders without waiting, and a
sender that calls neither lets the signal overtake the data.

all_gather_no_wait: every PE reads its buffer without waiting on the signals.
It reads a chunk before that chunk arrived.

eager_ag_gemm: computes tile i before its chunk's signal, so it reads an
unsignaled chunk. The audit flags it, and its output differs from A @ B.
"""

from __future__ import annotations

import torch

from lab_common import Checks, close

# Reference values of the alpha-beta model, in microseconds, from
# scripts/figures/36-symmetric-memory.py. Link 300 GB/s, 312 TFLOP/s, alpha 5 us,
# launch 5 us, signal 2 us, SM share 0.10, 8 GPUs.
PARAMS = dict(p=8, link_gbs=300.0, peak_tflops=312.0, alpha_s=5e-6, launch_s=5e-6, signal_s=2e-6, sm_steal=0.10)
CASE_A = dict(n_bytes=8192 * 8192 * 2, flops=2 * 8192 * 8192 * 1024)
CASE_B = dict(n_bytes=256 * 8192 * 2, flops=2 * 256 * 8192 * 1024)
MODEL_CASES = [
    # name, case, chunks, serial us, overlapped us, fused us, winner
    ("A, 8 chunks", CASE_A, 8, 866.978, 737.038, 535.494, "fused"),
    ("A, 1 chunk", CASE_A, 1, 866.978, 916.029, 878.029, "serial"),
    ("B, 4 chunks", CASE_B, 4, 60.999, 161.019, 24.019, "fused"),
    ("B, 64 chunks", CASE_B, 64, 60.999, 2257.470, 140.470, "serial"),
]


def signal_without_fence(heap, src, dst, data_off, sig_off, values):
    """Wrong: the signal can overtake the data."""
    heap.put(src, dst, data_off, values)
    heap.signal(src, dst, sig_off, 1.0)


def all_gather_no_wait(chunks, policy, heap_cls):
    """Wrong: reads the buffer without waiting for the signals."""
    n, c = len(chunks), chunks[0].numel()
    heap = heap_cls(n, n * c + n, policy)
    buf = heap.malloc(n * c)
    sig = heap.malloc(n)
    for pe in range(n):
        heap.mem[pe, buf + pe * c : buf + (pe + 1) * c] = chunks[pe].reshape(-1)
    for src in range(n):
        for dst in range(n):
            if dst != src:
                heap.put(src, dst, buf + src * c, chunks[src])
                heap.fence(src)
                heap.signal(src, dst, sig + src, 1.0)
    return [heap.get(pe, pe, buf, n * c) for pe in range(n)]


def eager_ag_gemm(shards, weights, heap_cls, tile_order):
    """Wrong: computes every tile in order without waiting for a signal."""
    n = len(shards)
    rows, k = shards[0].shape
    heap = heap_cls(n, n * rows * k + n, "fifo")
    buf = heap.malloc(n * rows * k)
    sig = heap.malloc(n)
    for pe in range(n):
        heap.mem[pe, buf + pe * rows * k : buf + (pe + 1) * rows * k] = shards[pe].reshape(-1)
    for src in range(n):
        for dst in range(n):
            if dst != src:
                heap.put(src, dst, buf + src * rows * k, shards[src])
                heap.fence(src)
                heap.signal(src, dst, sig + src, 1.0)
    log, outs = [], []
    for pe in range(n):
        out = torch.zeros(n * rows, weights[pe].shape[1])
        for chunk in tile_order(pe, n):
            tile = heap.get(pe, pe, buf + chunk * rows * k, rows * k).reshape(rows, k)
            out[chunk * rows : (chunk + 1) * rows] = tile @ weights[pe]
            log.append(("compute", pe, chunk))
        outs.append(out)
    return outs, log


def audit(log):
    """Return the tiles that read a remote chunk whose signal this PE had not waited on."""
    waited = set()
    bad = []
    for event in log:
        kind, pe, chunk = event
        if kind == "wait":
            waited.add((pe, chunk))
        elif chunk != pe and (pe, chunk) not in waited:
            bad.append(event)
    return bad


def _address(sub):
    heap = sub.SymmetricHeap(4, 64)
    got = [heap.address(0, 0), heap.address(1, 3), heap.address(3, 10)]
    want = [
        sub.HEAP_BASE,
        sub.HEAP_BASE + sub.PE_STRIDE + 12,
        sub.HEAP_BASE + 3 * sub.PE_STRIDE + 40,
    ]
    return got == want, f"got {[hex(x) for x in got]}, expected {[hex(x) for x in want]}"


def _translate(sub):
    heap = sub.SymmetricHeap(4, 64)
    a1 = heap.address(1, 7)
    got = heap.translate(a1, 1, 3)
    want = heap.address(3, 7)
    if got != want:
        return False, f"translate to PE 3 gave {got:#x}, expected {want:#x}"
    try:
        heap.translate(heap.address(2, 0) + 64 * 4, 2, 0)
    except ValueError:
        return True, f"PE 1 offset 7 maps to PE 3 offset 7 ({want:#x}); an address past the heap raises ValueError"
    return False, "an address one element past the end of PE 2's heap should raise ValueError, and it did not"


def _malloc(sub):
    heap = sub.SymmetricHeap(3, 10)
    first, second = heap.malloc(4), heap.malloc(6)
    if (first, second) != (0, 4):
        return False, f"offsets {(first, second)}, expected (0, 4)"
    try:
        heap.malloc(1)
    except MemoryError:
        return True, "offsets (0, 4), and a third allocation of 1 element on a full heap raises MemoryError"
    return False, "a heap of 10 elements with 10 allocated should raise MemoryError on the next malloc"


def _put_quiet_get(sub):
    heap = sub.SymmetricHeap(3, 16)
    off = heap.malloc(4)
    data = torch.tensor([1.0, 2.0, 3.0, 4.0])
    heap.put(0, 1, off, data)
    data.zero_()  # the caller may reuse its buffer after a put returns
    before = heap.get(2, 1, off, 4).tolist()
    heap.quiet(2)  # another PE's quiet must not complete PE 0's put
    other = heap.get(2, 1, off, 4).tolist()
    heap.quiet(0)
    after = heap.get(2, 1, off, 4).tolist()
    ok = before == [0.0] * 4 and other == [0.0] * 4 and after == [1.0, 2.0, 3.0, 4.0]
    return ok, (
        f"before quiet {before}, after PE 2's quiet {other}, after PE 0's quiet {after}; "
        "expected [0, 0, 0, 0], [0, 0, 0, 0], [1, 2, 3, 4]"
    )


def _never_early(heap_factory, send, steps_limit=20):
    """Run the network one step at a time. True if a visible signal ever sat beside stale data."""
    heap = heap_factory()
    data_off, sig_off = heap.malloc(3), heap.malloc(1)
    payload = torch.tensor([5.0, 6.0, 7.0])
    send(heap, payload, data_off, sig_off)
    for _ in range(steps_limit):
        if heap.mem[1, sig_off].item() == 1.0 and heap.get(1, 1, data_off, 3).tolist() != payload.tolist():
            return False
        if not heap.step():
            break
    return heap.get(1, 1, data_off, 3).tolist() == payload.tolist()


def _fence(sub):
    def make():
        return sub.SymmetricHeap(2, 8, "signals_first")

    def with_fence(heap, payload, data_off, sig_off):
        heap.put(0, 1, data_off, payload)
        heap.fence(0)
        heap.signal(0, 1, sig_off, 1.0)

    def without_fence(heap, payload, data_off, sig_off):
        signal_without_fence(heap, 0, 1, data_off, sig_off, payload)

    right = _never_early(make, with_fence)
    wrong = _never_early(make, without_fence)
    ok = right and not wrong
    return ok, (
        f"put, fence, signal keeps the data first: {right}, expected True. "
        f"signal_without_fence keeps the data first: {wrong}, expected False under signals_first."
    )


def _wait(sub):
    heap = sub.SymmetricHeap(2, 8, "reverse")
    sig = heap.malloc(1)
    heap.signal(0, 1, sig, 1.0)
    heap.wait_until(1, sig, 1.0)
    arrived = heap.mem[1, sig].item() == 1.0
    try:
        heap.wait_until(1, sig, 2.0)
    except sub.SignalNotReady:
        return arrived, "a signal in flight completes the wait, and a wait for a value nothing will write raises SignalNotReady"
    return False, "a wait for the value 2 with nothing in flight should raise SignalNotReady, and it returned"


def _all_gather(sub):
    gen = torch.Generator().manual_seed(36)
    chunks = [torch.randn(5, generator=gen) for _ in range(4)]
    want = torch.cat(chunks)
    for policy in ("fifo", "reverse", "signals_first"):
        outs = sub.all_gather_with_puts(chunks, policy)
        if len(outs) != 4:
            return False, f"policy {policy}: {len(outs)} outputs, expected 4"
        for pe, out in enumerate(outs):
            ok, detail = close(out, want, 0.0)
            if not ok:
                return False, f"policy {policy}, PE {pe}: {detail}"
    heap_cls = sub.SymmetricHeap
    wrong = all_gather_no_wait(chunks, "fifo", heap_cls)
    wrong_bad = any((out - want).abs().max().item() > 0 for out in wrong)
    return wrong_bad, (
        "equals torch.cat on 4 PEs under fifo, reverse, and signals_first (max abs error 0). "
        f"all_gather_no_wait reads before the signals and differs from the reference: {wrong_bad}"
    )


def _ag_gemm(sub):
    gen = torch.Generator().manual_seed(7)
    n, rows, k, cols = 4, 3, 5, 2
    shards = [torch.randn(rows, k, generator=gen) for _ in range(n)]
    weights = [torch.randn(k, cols, generator=gen) for _ in range(n)]
    full = torch.cat(shards)
    if [sub.tile_order(1, 4), sub.tile_order(3, 4)] != [[1, 2, 3, 0], [3, 0, 1, 2]]:
        return False, f"tile_order(1, 4) = {sub.tile_order(1, 4)}, expected [1, 2, 3, 0]; tile_order(3, 4) = {sub.tile_order(3, 4)}, expected [3, 0, 1, 2]"
    for policy in ("fifo", "signals_first"):
        outs, log = sub.ag_gemm(shards, weights, policy)
        for pe in range(n):
            ok, detail = close(outs[pe], full @ weights[pe], 1e-5)
            if not ok:
                return False, f"policy {policy}, PE {pe}: {detail}"
        bad = audit(log)
        if bad:
            return False, f"policy {policy}: tiles read an unsignaled chunk: {bad[:3]}, expected none"
        computed = [[c for kind, p, c in log if kind == "compute" and p == pe] for pe in range(n)]
        want_order = [sub.tile_order(pe, n) for pe in range(n)]
        if computed != want_order:
            return False, f"compute order {computed}, expected {want_order}"
    wrong_outs, wrong_log = eager_ag_gemm(shards, weights, sub.SymmetricHeap, sub.tile_order)
    flagged = len(audit(wrong_log)) > 0
    differs = any((wrong_outs[pe] - full @ weights[pe]).abs().max().item() > 1e-5 for pe in range(n))
    return flagged and differs, (
        "equals A @ B on every PE, no tile reads an unsignaled chunk, and each PE starts with its own chunk. "
        f"eager_ag_gemm is flagged by the audit: {flagged}, and is numerically wrong: {differs}"
    )


def _model(c, sub):
    names = []
    for name, case, chunks, serial, overlapped, fused, winner in MODEL_CASES:
        got = sub.overlap_model(**case, **PARAMS, chunks=chunks)
        row = (got["serial_s"] * 1e6, got["overlapped_s"] * 1e6, got["fused_s"] * 1e6)
        want = (serial, overlapped, fused)
        if any(abs(a - b) > 0.002 for a, b in zip(row, want)) or got["winner"] != winner:
            return False, (
                f"case {name}: got {tuple(round(x, 3) for x in row)} us and winner {got['winner']}, "
                f"expected {want} us and winner {winner}"
            )
        names.append(name)
    best = sub.overlap_model(**CASE_A, **PARAMS, chunks=8)
    c.metric("model_serial_us", round(best["serial_s"] * 1e6, 3))
    c.metric("model_overlapped_us", round(best["overlapped_s"] * 1e6, 3))
    c.metric("model_fused_us", round(best["fused_s"] * 1e6, 3))
    return True, f"serial, overlapped, fused, and the winner match for {', '.join(names)}"


def _gpu(c):
    import torch as t

    count = t.cuda.device_count() if t.cuda.is_available() else 0
    if count < 2:
        c.add(
            "two-GPU all-gather plus matmul overlap",
            True,
            f"skipped: this machine has {count} GPU(s), and the check needs 2. "
            "It was not run, and no timing is recorded. A missing two-GPU endpoint does not send it to one A100.",
        )
        return
    import fixture_symm

    try:
        r = fixture_symm.launch_overlap()
    except Exception as exc:  # noqa: BLE001
        c.add(
            "two-GPU all-gather plus matmul overlap",
            False,
            f"expected a finished two-GPU run, got {type(exc).__name__}: {exc}",
        )
        return
    speedup = r["serial_ms"] / r["overlap_ms"]
    # Correctness is the hard check. Timing is a metric with a loose bound: a
    # chunked pipeline may lose on a small problem, but not by more than half.
    ok = r["rel_err"] <= 2e-2 and r["overlap_ms"] <= 1.5 * r["serial_ms"]
    c.add(
        "two-GPU all-gather plus matmul overlap",
        ok,
        f"relative error {r['rel_err']:.2e}, expected <= 2e-2; serial {r['serial_ms']:.3f} ms, "
        f"chunked {r['overlap_ms']:.3f} ms, expected chunked <= 1.5 x serial",
    )
    c.metric("gpu_serial_ms", round(r["serial_ms"], 3))
    c.metric("gpu_overlap_ms", round(r["overlap_ms"], 3))
    c.metric("gpu_overlap_speedup", round(speedup, 3))


def run(submission):
    c = Checks()
    if not c.require(submission, "SymmetricHeap", "all_gather_with_puts", "tile_order", "ag_gemm", "overlap_model"):
        return c.finish()
    c.check("address maps (pe, offset) to a virtual address", lambda: _address(submission))
    c.check("translate swaps the base and rejects a foreign address", lambda: _translate(submission))
    c.check("malloc returns one offset that is valid on every PE", lambda: _malloc(submission))
    c.check("a put arrives only after the issuing PE's quiet", lambda: _put_quiet_get(submission))
    c.check("put, fence, signal keeps the data ahead of the signal", lambda: _fence(submission))
    c.check("wait_until spins on the network and detects a deadlock", lambda: _wait(submission))
    c.check("an all-gather from puts equals the concatenation", lambda: _all_gather(submission))
    c.check("chunked all-gather plus GEMM equals A @ B and never reads an unsignaled chunk", lambda: _ag_gemm(submission))
    c.check("the alpha-beta model picks serial, overlapped, or fused", lambda: _model(c, submission))
    _gpu(c)
    return c.finish()
