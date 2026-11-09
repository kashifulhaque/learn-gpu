"""Tests for lab 25.

allreduce_no_divide is a plausible wrong all-reduce. It sums the per-rank
gradients and does not divide by the world size. On world size 2 the local
gradients in this lab are 1 and 2, so allreduce_no_divide returns 3 and the
single-process mean is 1.5. The gradient check fails that implementation.

full_state_on_every_rank is a plausible wrong ZeRO-1 shard. It stores Adam's
first and second moments for every parameter on every rank. For 96 float32
parameters that is 768 bytes on each rank. The byte check expects 384, which
is 768 / 2, so that implementation fails the check.
"""

from __future__ import annotations

from lab_common import Checks


def allreduce_no_divide(local_grads: list[float]) -> float:
    """Sum the per-rank gradients and skip the division by the world size."""
    return float(sum(local_grads))


def full_state_on_every_rank(numel: int, dtype_bytes: int = 4) -> int:
    """Bytes of both Adam moments for the full parameter vector, on every rank."""
    return 2 * numel * dtype_bytes


def _bucket_example(submission):
    got = submission.assign_buckets([16, 16, 16, 48], 128, 4)
    want = [[3], [2, 1], [0]]
    return got == want, f"got {got}, expected {want}"


def _bucket_oversize(submission):
    got = submission.assign_buckets([10, 100, 10], 40, 4)
    want = [[2], [1], [0]]
    return got == want, f"got {got}, expected {want}"


def _ddp(c, submission):
    import fixture_ddp

    try:
        result = fixture_ddp.launch(2, {"name": "ddp"}, submission.__file__, timeout=60)
    except Exception as exc:  # noqa: BLE001
        c.add(
            "bucketed DDP runs to completion on gloo",
            False,
            f"expected a finished two-rank run, got {type(exc).__name__}: {exc}",
        )
        return
    if result.get("error"):
        tail = result["error"].strip().splitlines()[-1]
        c.add(
            "bucketed DDP runs to completion on gloo",
            False,
            f"expected a finished two-rank run, got {tail}",
        )
        return

    c.add(
        "BucketedDDP broadcasts rank 0 so the replicas start equal",
        result["init_err"] == 0.0,
        f"max abs diff {result['init_err']:.3e}, expected 0",
    )
    # The mean of local gradients 1 and 2 is 1.5. allreduce_no_divide returns 3.
    wrong = allreduce_no_divide([1.0, 2.0])
    mean = 1.5
    grad_ok = result["grad_err"] <= 1e-6 and wrong != mean
    c.add(
        "the averaged gradient equals the single-process mean",
        grad_ok,
        f"max abs error {result['grad_err']:.3e}, expected 0 (mean {mean}). "
        f"allreduce_no_divide returns {wrong}.",
    )
    c.add(
        "backward launches one asynchronous all-reduce per bucket",
        result["calls"] == 3 and result["async_calls"] == 3 and result["backward_calls"] == 3,
        f"all-reduce calls {result['calls']}, expected 3; "
        f"async calls {result['async_calls']}, expected 3; "
        f"calls during backward {result['backward_calls']}, expected 3",
    )
    c.add(
        "parameters match across ranks after the optimizer step",
        result["step_err"] == 0.0,
        f"max abs diff {result['step_err']:.3e}, expected 0",
    )
    c.metric("ddp_buckets", result["calls"])
    c.metric("grad_max_abs_err", result["grad_err"])


def _zero(c, submission):
    import fixture_ddp

    try:
        result = fixture_ddp.launch(2, {"name": "zero"}, submission.__file__, timeout=60)
    except Exception as exc:  # noqa: BLE001
        c.add(
            "ZeRO-1 runs to completion on gloo",
            False,
            f"expected a finished two-rank step, got {type(exc).__name__}: {exc}",
        )
        return
    if result.get("error"):
        tail = result["error"].strip().splitlines()[-1]
        c.add(
            "ZeRO-1 runs to completion on gloo",
            False,
            f"expected a finished two-rank step, got {tail}",
        )
        return

    # 96 float32 parameters, two moments, two ranks: 2 * 96 * 4 / 2 = 384.
    expected = 2 * result["numel"] * 4 // result["world"]
    wrong = full_state_on_every_rank(result["numel"])
    c.add(
        "ZeRO-1 state bytes on each rank are 1/N of the full moments",
        result["state_bytes"] == expected and wrong != expected,
        f"state bytes {result['state_bytes']}, expected {expected}. "
        f"full_state_on_every_rank stores {wrong}.",
    )
    c.add(
        "ZeRO-1 weights match a single-process Adam step, using reduce-scatter and all-gather",
        result["weight_err"] <= 1e-5 and result["reduce_scatter"] == 2 and result["all_gather"] == 2,
        f"max abs error {result['weight_err']:.3e}, expected <= 1e-5; "
        f"reduce-scatter calls {result['reduce_scatter']}, expected 2; "
        f"all-gather calls {result['all_gather']}, expected 2",
    )
    c.metric("zero_state_bytes", result["state_bytes"])
    c.metric("zero_full_state_bytes", wrong)


def _nccl(c):
    import torch

    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if count < 2:
        c.add(
            "NCCL bus-bandwidth sweep",
            True,
            f"skipped: this machine has {count} GPU(s), and the sweep needs 2. "
            "It was not run. A missing two-GPU endpoint does not send it to one A100.",
        )
        return
    import fixture_ddp

    try:
        result = fixture_ddp.launch_nccl()
    except Exception as exc:  # noqa: BLE001
        c.add(
            "NCCL bus-bandwidth sweep",
            False,
            f"expected a finished two-GPU sweep, got {type(exc).__name__}: {exc}",
        )
        return
    bandwidth = result["bandwidth_gbs"]
    alpha_us = result["alpha_s"] * 1e6
    ok = bandwidth > 0 and alpha_us == alpha_us  # finite, and not a claimed ceiling
    c.add(
        "NCCL bus-bandwidth sweep fits a positive bandwidth",
        ok,
        f"fit bandwidth {bandwidth:.3f} GB/s, expected > 0; alpha {alpha_us:.3f} us. "
        "This is the machine's measurement, not the 300 GB/s NVLink ceiling.",
    )
    c.metric("fit_alpha_us", round(alpha_us, 3))
    c.metric("fit_bandwidth_gbs", round(bandwidth, 3))


def run(submission):
    c = Checks()
    if not hasattr(submission, "assign_buckets"):
        c.add(
            "assign_buckets is defined",
            False,
            "expected a function assign_buckets, got no such attribute",
        )
        return c.finish()

    # These two checks call only assign_buckets. A NotImplementedError fails
    # inside check(), so the starter does not crash the harness.
    c.check("assign_buckets returns the worked example", lambda: _bucket_example(submission))
    c.check(
        "assign_buckets gives an oversized parameter its own bucket",
        lambda: _bucket_oversize(submission),
    )
    if hasattr(submission, "BucketedDDP"):
        _ddp(c, submission)
    else:
        c.add("BucketedDDP is defined", False, "expected a class BucketedDDP, got no such attribute")
    if hasattr(submission, "Zero1Adam"):
        _zero(c, submission)
    else:
        c.add("Zero1Adam is defined", False, "expected a class Zero1Adam, got no such attribute")
    _nccl(c)
    return c.finish()
