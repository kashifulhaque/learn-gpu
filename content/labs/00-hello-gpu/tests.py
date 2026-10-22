import torch
from lab_common import A100, Checks, require_cuda


def run(submission):
    c = Checks()
    if not c.require(submission, "describe_device", "copy_bandwidth_gbs", "ceiling_fraction"):
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

    measured = submission.copy_bandwidth_gbs()
    c.check(
        "the copy bandwidth is plausible",
        lambda: (
            100.0 < measured < 4000.0,
            f"{measured:.0f} GB/s. A figure outside 100 to 4000 usually means the "
            "bytes or the time are wrong, for example counting the read but not the write.",
        ),
    )
    c.metric("copy_gbs", round(measured, 1))

    fraction = submission.ceiling_fraction(measured)
    c.check(
        "ceiling_fraction divides by the measured copy ceiling",
        lambda: (
            abs(fraction - measured / A100.COPY_CEILING_GBS) < 1e-6,
            f"got {fraction:.3f}, expected {measured / A100.COPY_CEILING_GBS:.3f}",
        ),
    )
    c.metric("ceiling_fraction", round(fraction, 3))
    # On the lab A100 this lands near 1.0. Another card is not a failure.
    return c.finish()
