"""One lab run, shared by `/api/run` and the MCP server's `run_lab` tool.

`start_run` saves the draft, picks the provider, records the run, and returns
the event stream. The stream finishes the run record, so both callers have the
same side effects and the same cost: a lab that needs a GPU starts a RunPod job.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

from . import curriculum, db
from .providers import OutOfCredits, ProviderError, get_provider


@dataclass
class Run:
    run_id: str
    provider: str
    events: AsyncIterator[dict[str, Any]]


def start_run(user: str, lab_id: str, code: str, provider_name: str | None = None) -> Run | None:
    """Prepare a run, or return None when the lab doesn't exist.

    Raises ValueError for an unknown provider name.
    """
    lab = curriculum.get_lab(lab_id)
    if not lab:
        return None

    # A draft is saved on every run, so a closed tab never loses work.
    db.save_draft(user, lab_id, code)
    provider = get_provider(provider_name)
    run_id = db.create_run(user, lab_id, provider.name)

    async def events() -> AsyncIterator[dict[str, Any]]:
        log_lines: list[str] = []
        result: dict[str, Any] | None = None
        stream_error: str | None = None
        started = time.time()
        yield {"type": "start", "run_id": run_id, "provider": provider.name}
        try:
            async for event in provider.run_lab(
                lab_id=lab_id,
                code=code,
                gpu=lab["gpu"],
                timeout=lab["timeout"],
                gpus=int(lab.get("gpus") or 1),
            ):
                kind = event.get("type")
                if kind == "job":
                    # The provider's own id for this job. Recorded so the
                    # compute panel can cancel a run whose tab has gone away.
                    db.set_run_job(run_id, event.get("job_id", ""))
                    continue
                if kind == "log":
                    log_lines.append(event.get("line", ""))
                elif kind == "result":
                    result = event
                elif kind == "error":
                    # The provider reports failures as events rather than
                    # exceptions, so record them as errors, not as a failed run.
                    stream_error = event.get("message", "")
                yield event

            passed = bool(result.get("passed")) if result else False
            metrics = result.get("metrics", {}) if result else {}
            db.finish_run(
                run_id,
                status="error" if stream_error else "finished",
                passed=None if stream_error else passed,
                metrics=metrics,
                log="\n".join(log_lines),
                error=stream_error,
            )
            if stream_error:
                return
            yield {
                "type": "done",
                "run_id": run_id,
                "passed": passed,
                "seconds": round(time.time() - started, 2),
            }
        except OutOfCredits as exc:
            db.finish_run(
                run_id, status="out_of_credits", log="\n".join(log_lines), error=str(exc)
            )
            hint = f"{provider.name} is out of credit. Add credit, then run again."
            yield {
                "type": "out_of_credits",
                "provider": provider.name,
                "message": str(exc),
                "hint": hint,
            }
        except (ProviderError, Exception) as exc:  # noqa: BLE001
            db.finish_run(
                run_id, status="error", log="\n".join(log_lines), error=str(exc)
            )
            yield {"type": "error", "message": str(exc)}

    return Run(run_id=run_id, provider=provider.name, events=events())
