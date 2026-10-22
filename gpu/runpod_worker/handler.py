"""RunPod serverless handler: runs one lab and streams its events.

Build and push the image in this directory, then point a RunPod serverless
endpoint at it and set RUNPOD_ENDPOINT_ID in the app's .env.
"""

import sys
from typing import Any, Iterator

sys.path.insert(0, "/repo/gpu")

import runpod  # noqa: E402
from lab_runner import run_lab_stream  # noqa: E402


def handler(job: dict[str, Any]) -> Iterator[dict[str, Any]]:
    payload = job.get("input", {})
    lab_id = payload.get("lab_id")
    code = payload.get("code", "")
    timeout = int(payload.get("timeout", 900))

    if not lab_id:
        yield {"type": "error", "message": "No lab_id in the job input"}
        return

    yield from run_lab_stream(lab_id=lab_id, code=code, timeout=timeout)


runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
