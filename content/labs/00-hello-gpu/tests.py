import time

import torch
from lab_common import A100, Checks, cuda_time, require_cuda


def run(submission):
    c = Checks()
    if not c.require(submission, "describe_device", "time_ms", "copy_bandwidth_gbs", "ceiling_fraction"):
        return c.finish()

    require_cuda()
    props = torch.cuda.get_device_properties(0)

    info = c.check("describe_device returns a dict", lambda: isinstance(submission.describe_device(), dict))
    if info:
        got = submission.describe_device()
        c.add("name is the device's name", got.get("name") == props.name, f"got {got.get('name')!r}")
        c.add(
            "sm_count matches the device",
            got.get("sm_count") == props.multi_processor_count,
            f"got {got.get('sm_count')}, the device has {props.multi_processor_count}",
        )
        memory = got.get("memory_gb", 0)
        c.add(
            "memory_gb is in decimal GB",
            abs(memory - props.total_memory / 1e9) < 0.5,
            f"got {memory}, expected {props.total_memory / 1e9:.1f}",
        )
        c.add(
            "capability is (major, minor)",
            tuple(got.get("capability", ())) == (props.major, props.minor),
            f"got {got.get('capability')}",
        )
        c.metric("sm_count", props.multi_processor_count)
        print(f"      device = {props.name}", flush=True)
        if props.multi_processor_count != A100.SM_COUNT:
            print(
                f"      note: this is not the course's A100 ({A100.SM_COUNT} SMs); "
                "bandwidth targets below assume one.",
                flush=True,
            )

    # The timer. 256 MiB of float32, copied: 2 * 256 MiB moved.
    n_bytes = 256 * 2**20
    src = torch.empty(n_bytes // 4, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)
    src.normal_()

    def copy():
        dst.copy_(src)

    # The fastest a copy of this size could run is the traffic over twice the
    # rated bandwidth, so a reading below it means the timer did not wait.
    floor_ms = 2 * n_bytes / (2 * A100.HBM_BW_GBS * 1e9) * 1e3
    ms = None

    def timer_works():
        nonlocal ms
        ms = submission.time_ms(copy, warmup=5, iters=20)
        ok = isinstance(ms, float) or isinstance(ms, int)
        return ok and ms > floor_ms, (
            f"{ms!r} ms. A copy this size can't finish in under {floor_ms:.3f} ms, so a smaller "
            "reading means the timer read before the GPU finished: synchronize on the end event."
        )

    c.check("time_ms waits for the GPU", timer_works)

    if isinstance(ms, (int, float)) and ms > 0:
        reference = cuda_time(copy, warmup=5, iters=20)
        c.add(
            "time_ms agrees with lab_common.cuda_time",
            0.5 < ms / reference < 2.0,
            f"yours {ms:.3f} ms, reference {reference:.3f} ms",
        )
        # The contrast the chapter teaches: a host timer with no synchronize.
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        copy()
        host_ms = (time.perf_counter() - t0) * 1e3
        torch.cuda.synchronize()
        print(
            f"      host timer without synchronize = {host_ms:.3f} ms; events say {ms:.3f} ms",
            flush=True,
        )
        c.metric("host_no_sync_ms", round(host_ms, 4))
        c.metric("copy_ms", round(ms, 4))

    measured = None

    def bandwidth_plausible():
        nonlocal measured
        measured = float(submission.copy_bandwidth_gbs())
        return (
            100.0 < measured < 2 * A100.HBM_BW_GBS,
            f"{measured:.0f} GB/s. A figure outside 100 to {2 * A100.HBM_BW_GBS:.0f} usually means the "
            "bytes or the time are wrong, for example counting the read but not the write.",
        )

    c.check("the copy bandwidth is plausible", bandwidth_plausible)
    if measured is None:
        return c.finish()
    c.metric("copy_gbs", round(measured, 1))

    # Loose performance check: half of the measured copy ceiling, on an A100-class card.
    if props.multi_processor_count == A100.SM_COUNT:
        target = 0.5 * A100.COPY_CEILING_GBS
        c.add(
            "the copy reaches half of the 1275 GB/s ceiling",
            measured >= target,
            f"{measured:.0f} GB/s against a floor of {target:.0f} GB/s",
        )

    def fraction_ok():
        fraction = submission.ceiling_fraction(measured)
        c.metric("ceiling_fraction", round(fraction, 3))
        return (
            abs(fraction - measured / A100.COPY_CEILING_GBS) < 1e-6,
            f"got {fraction:.3f}, expected {measured / A100.COPY_CEILING_GBS:.3f}",
        )

    c.check("ceiling_fraction divides by the measured copy ceiling", fraction_ok)
    # On the lab A100 this lands near 1.0. Another card is not a failure.
    return c.finish()
