"""The compute panel: what is running on the GPU providers, and how to stop it.

Labs run on someone else's GPU, and the two failure modes that cost real money
are a container nobody noticed and a job left queued after a browser tab
closed. This module gathers every provider's live state into one snapshot, and
dispatches the stop actions the panel offers.
"""

import asyncio
import time
from dataclasses import asdict, is_dataclass
from typing import Any

from . import db
from .config import get_settings
from .providers import ProviderError, get_provider, provider_names

# Actions the panel may ask for. Anything that deletes an endpoint, a volume, or
# a pod's disk is deliberately absent: the RunPod serverless endpoint costs
# nothing while idle.
ALLOWED_ACTIONS = frozenset(
    {
        "stop-pod",
        "purge-queue",
        "cancel-job",
        "scale-to-zero",
        "cancel-run",
    }
)


def _plain(value: Any) -> Any:
    """Make dataclasses and their nested lists JSON-serialisable."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


async def snapshot() -> dict[str, Any]:
    """Every provider's live state, plus this app's own unfinished runs."""
    settings = get_settings()
    names = provider_names()
    results = await asyncio.gather(
        *(_provider_snapshot(name) for name in names), return_exceptions=False
    )

    providers = []
    active = 0
    for name, result in zip(names, results):
        result["name"] = name
        result["default"] = name == settings.gpu_provider
        active += sum(1 for item in result.get("instances", []) if item.get("active"))
        providers.append(result)

    return {
        "generated_at": time.time(),
        "active": active,
        "providers": providers,
        "runs": _active_runs(),
    }


async def _provider_snapshot(name: str) -> dict[str, Any]:
    """One provider's state. A provider that errors must not blank the panel."""
    try:
        provider = get_provider(name)
        result = _plain(await provider.snapshot())
        result.setdefault("instances", [])
        result.setdefault("notices", [])
        result.setdefault("facts", [])
        result["console_url"] = getattr(provider, "console_url", "")
        return result
    except Exception as exc:  # noqa: BLE001 - the panel reports it and stays up
        return {
            "available": False,
            "reason": f"Could not read {name}: {exc}",
            "instances": [],
            "notices": [],
            "facts": [],
            "console_url": "",
        }


def _active_runs() -> list[dict[str, Any]]:
    """Runs this app started and never saw finish."""
    runs = []
    for run in db.list_active_runs():
        age = time.time() - run["started_at"]
        runs.append(
            {
                **run,
                "age": round(age),
                "cancellable": bool(run.get("job_id")),
                "hint": (
                    "Cancel the job on the provider."
                    if run.get("job_id")
                    else "No provider job id was recorded. Stop the worker in "
                    f"the {run['provider']} section instead."
                ),
            }
        )
    return runs


async def act(provider_name: str, action: str, target: str) -> dict[str, Any]:
    """Perform one panel action and return a message for the user."""
    if action not in ALLOWED_ACTIONS:
        raise ProviderError(f"Unknown action: {action}")

    if action == "cancel-run":
        return {"message": await _cancel_run(target)}

    provider = get_provider(provider_name)
    if not hasattr(provider, "act"):
        raise ProviderError(f"{provider_name} has no management API")
    message = await provider.act(action, target)
    return {"message": message}


async def _cancel_run(run_id: str) -> str:
    """Cancel a run that is still marked running, and close it out locally."""
    run = db.find_run(run_id)
    if not run:
        raise ProviderError("No such run")
    if run["status"] != "running":
        return f"Run {run_id[:8]} already finished as {run['status']}."

    message = ""
    if run.get("job_id"):
        provider = get_provider(run["provider"])
        # If this raises, the record stays open on purpose: a cancel that did
        # not go through must not be reported as one.
        message = await provider.act("cancel-job", run["job_id"])
    else:
        message = (
            "No provider job id was recorded for this run, so only this app's "
            f"record was closed. Check the {run['provider']} section below."
        )
    db.finish_run(run_id, status="cancelled", error="Cancelled from the compute panel")
    return message
