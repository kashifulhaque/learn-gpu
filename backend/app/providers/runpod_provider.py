"""The lab executor: a RunPod serverless endpoint.

The worker image is in gpu/runpod_worker/. RunPod streams generator output
through /stream/{job_id}, which this provider polls.
"""

import asyncio
import time
from typing import Any, AsyncIterator

import httpx

from ..config import get_settings
from .base import Instance, OutOfCredits, ProviderError

API_BASE = "https://api.runpod.ai/v2"
REST_BASE = "https://rest.runpod.io/v1"
GRAPHQL_URL = "https://api.runpod.io/graphql"
CONSOLE_URL = "https://console.runpod.io"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}

# Worker states RunPod reports from /health. "throttled" means RunPod has no
# capacity for the worker right now; it is not billed.
BILLED_WORKER_STATES = ("running", "ready", "initializing", "idle")


class RunPodProvider:
    name = "runpod"
    console_url = CONSOLE_URL

    def __init__(self) -> None:
        self.settings = get_settings()

    def available(self) -> tuple[bool, str]:
        if not self.settings.runpod_api_key:
            return False, "RUNPOD_API_KEY is not set"
        if not self.settings.runpod_endpoint_id:
            return False, "RUNPOD_ENDPOINT_ID is not set"
        return True, ""

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.runpod_api_key}",
            "Content-Type": "application/json",
        }

    async def run_lab(
        self, lab_id: str, code: str, gpu: str, timeout: int
    ) -> AsyncIterator[dict[str, Any]]:
        ok, why = self.available()
        if not ok:
            raise ProviderError(why)

        endpoint = f"{API_BASE}/{self.settings.runpod_endpoint_id}"
        payload = {
            "input": {"lab_id": lab_id, "code": code, "gpu": gpu, "timeout": timeout}
        }

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{endpoint}/run", json=payload, headers=self._headers
            )
            if response.status_code in (402, 403):
                raise OutOfCredits(f"RunPod rejected the job: {response.text}")
            response.raise_for_status()
            job_id = response.json()["id"]

            # The compute panel needs the job id to cancel a run whose browser
            # tab has gone away.
            yield {"type": "job", "provider": self.name, "job_id": job_id}

            deadline = asyncio.get_running_loop().time() + timeout + 120
            while True:
                if asyncio.get_running_loop().time() > deadline:
                    raise ProviderError("RunPod job exceeded its deadline")

                stream = await client.get(
                    f"{endpoint}/stream/{job_id}", headers=self._headers
                )
                stream.raise_for_status()
                body = stream.json()

                for chunk in body.get("stream", []):
                    event = chunk.get("output")
                    if isinstance(event, dict):
                        yield event

                status = body.get("status")
                if status in TERMINAL:
                    if status != "COMPLETED":
                        yield {
                            "type": "error",
                            "message": f"RunPod job ended with status {status}",
                        }
                    return
                await asyncio.sleep(1.0)

    # --- management ---------------------------------------------------------

    async def snapshot(self) -> dict[str, Any]:
        """Endpoint workers, pods, balance, and recent spend."""
        if not self.settings.runpod_api_key:
            return {"available": False, "reason": "RUNPOD_API_KEY is not set"}

        # The account is readable with the key alone, so the panel still shows
        # pods when RUNPOD_ENDPOINT_ID is missing. `available` stays
        # honest about whether labs can actually run here.
        runnable, why = self.available()
        endpoint_id = self.settings.runpod_endpoint_id
        async with httpx.AsyncClient(timeout=30, headers=self._headers) as client:
            account, endpoints, pods, spend, health = await asyncio.gather(
                self._account(client),
                self._get(client, f"{REST_BASE}/endpoints"),
                self._get(client, f"{REST_BASE}/pods"),
                self._spend(client),
                self._health(client, endpoint_id),
                return_exceptions=True,
            )

        errors = [
            str(item) for item in (account, endpoints, pods, spend, health)
            if isinstance(item, BaseException)
        ]
        account = account if isinstance(account, dict) else {}
        endpoints = endpoints if isinstance(endpoints, list) else []
        pods = pods if isinstance(pods, list) else []
        spend = spend if isinstance(spend, dict) else {}
        health = health if isinstance(health, dict) else {}

        instances = self._endpoint_instances(endpoints, health, endpoint_id)
        instances += self._pod_instances(pods)

        notices = []
        if endpoint_id and not any(e.get("id") == endpoint_id for e in endpoints):
            notices.append(
                f"RUNPOD_ENDPOINT_ID is {endpoint_id}, but no endpoint with that id "
                "exists on this account."
            )
        if not endpoint_id:
            notices.append(
                "RUNPOD_ENDPOINT_ID is not set, so labs cannot fall back to RunPod."
            )
        throttled = (health.get("workers") or {}).get("throttled", 0)
        if throttled:
            notices.append(
                f"{throttled} worker(s) are throttled: RunPod has no capacity for "
                "them right now. Throttled workers are not billed, but a lab sent "
                "to this endpoint will wait."
            )
        notices += errors

        return {
            "available": runnable,
            "reason": why,
            "instances": instances,
            "notices": notices,
            "facts": self._facts(account, spend, endpoints, endpoint_id),
            "account": account,
            "spend": spend,
            "jobs": health.get("jobs") or {},
        }

    async def act(self, action: str, target: str) -> str:
        """Stop a pod, drop queued jobs, cancel one job, or park the workers."""
        if not self.settings.runpod_api_key:
            raise ProviderError("RUNPOD_API_KEY is not set")
        endpoint_id = self.settings.runpod_endpoint_id
        if action in ("purge-queue", "cancel-job") and not (target or endpoint_id):
            raise ProviderError("RUNPOD_ENDPOINT_ID is not set")
        async with httpx.AsyncClient(timeout=30, headers=self._headers) as client:
            if action == "stop-pod":
                await self._post(client, f"{REST_BASE}/pods/{target}/stop")
                return (
                    f"Stopped pod {target}. Its disk is kept, so it can be started "
                    "again from the RunPod console."
                )
            if action == "purge-queue":
                body = await self._post(
                    client, f"{API_BASE}/{target or endpoint_id}/purge-queue"
                )
                removed = (body or {}).get("removed", "queued")
                return f"Purged the queue: {removed} job(s) dropped."
            if action == "cancel-job":
                cancelled = await self._post(
                    client,
                    f"{API_BASE}/{endpoint_id}/cancel/{target}",
                    missing_ok=True,
                )
                if cancelled is None:
                    return (
                        f"RunPod no longer has job {target}, so it had already "
                        "finished or expired."
                    )
                return f"Cancelled job {target}."
            if action == "scale-to-zero":
                await self._patch(
                    client,
                    f"{REST_BASE}/endpoints/{target or endpoint_id}",
                    {"workersMin": 0},
                )
                return (
                    "Set the endpoint's always-on workers to zero. It stays "
                    "deployed and costs nothing while idle."
                )
        raise ProviderError(f"RunPod cannot do '{action}'")

    # --- HTTP helpers -------------------------------------------------------

    async def _get(self, client: httpx.AsyncClient, url: str) -> Any:
        return self._body(await client.get(url))

    async def _post(
        self,
        client: httpx.AsyncClient,
        url: str,
        body: dict[str, Any] | None = None,
        missing_ok: bool = False,
    ) -> Any:
        """POST, returning None instead of raising when the target is gone."""
        response = await client.post(url, json=body or {})
        if missing_ok and response.status_code == 404:
            return None
        return self._body(response)

    async def _patch(
        self, client: httpx.AsyncClient, url: str, body: dict[str, Any]
    ) -> Any:
        return self._body(await client.patch(url, json=body))

    @staticmethod
    def _body(response: httpx.Response) -> Any:
        """The JSON body, or a readable error. httpx's own text is not one."""
        if response.status_code in (402, 403):
            raise OutOfCredits(f"RunPod refused the request: {response.text[:200]}")
        if response.status_code >= 400:
            raise ProviderError(
                f"RunPod answered {response.status_code} for "
                f"{response.request.url.path}: {response.text[:200]}"
            )
        try:
            return response.json()
        except ValueError:
            return {}

    async def _health(self, client: httpx.AsyncClient, endpoint_id: str) -> dict:
        if not endpoint_id:
            return {}
        return await self._get(client, f"{API_BASE}/{endpoint_id}/health")

    async def _account(self, client: httpx.AsyncClient) -> dict[str, Any]:
        """Balance and burn rate, which only the GraphQL API reports."""
        query = (
            "query { myself { clientBalance currentSpendPerHr spendLimit "
            "minBalance } }"
        )
        payload = self._body(await client.post(GRAPHQL_URL, json={"query": query}))
        if payload.get("errors"):
            raise ProviderError(f"RunPod GraphQL: {payload['errors'][0].get('message')}")
        return (payload.get("data") or {}).get("myself") or {}

    async def _spend(self, client: httpx.AsyncClient) -> dict[str, Any]:
        """What this account has spent over the last day and week."""
        now = time.time()
        window = {
            "startTime": _iso(now - 7 * 86400),
            "endTime": _iso(now),
            "bucketSize": "1d",
        }
        sources = ("endpoints", "pods", "networkvolumes")
        results = await asyncio.gather(
            *(
                self._get(
                    client,
                    f"{REST_BASE}/billing/{source}?"
                    + "&".join(f"{k}={v}" for k, v in window.items()),
                )
                for source in sources
            ),
            return_exceptions=True,
        )

        week = 0.0
        day = 0.0
        cutoff = now - 86400
        by_source: dict[str, float] = {}
        for source, rows in zip(sources, results):
            if not isinstance(rows, list):
                continue
            for row in rows:
                amount = float(row.get("amount") or 0) + float(
                    row.get("highPerformanceStorageAmount") or 0
                )
                week += amount
                by_source[source] = by_source.get(source, 0.0) + amount
                stamp = _parse_stamp(row.get("time") or row.get("startDate"))
                if stamp and stamp >= cutoff:
                    day += amount
        return {"day": round(day, 4), "week": round(week, 4), "by_source": by_source}

    # --- shaping ------------------------------------------------------------

    def _endpoint_instances(
        self, endpoints: list[dict], health: dict, endpoint_id: str
    ) -> list[Instance]:
        out = []
        for endpoint in endpoints:
            ours = endpoint.get("id") == endpoint_id
            workers = (health.get("workers") or {}) if ours else {}
            jobs = (health.get("jobs") or {}) if ours else {}
            live = sum(int(workers.get(state, 0)) for state in BILLED_WORKER_STATES)
            queued = int(jobs.get("inQueue", 0)) + int(jobs.get("inProgress", 0))
            always_on = int(endpoint.get("workersMin") or 0)

            actions = []
            if ours and int(jobs.get("inQueue", 0)):
                actions.append(
                    {
                        "action": "purge-queue",
                        "label": "Purge queue",
                        "confirm": "Drop every queued job on this endpoint?",
                    }
                )
            if always_on:
                actions.append(
                    {
                        "action": "scale-to-zero",
                        "label": "Park workers",
                        "confirm": "Set always-on workers to zero? The endpoint "
                        "stays deployed.",
                    }
                )

            detail = (
                f"{endpoint.get('gpuCount', 1)}× "
                f"{', '.join(endpoint.get('gpuTypeIds') or ['unspecified GPU'])}. "
                f"Workers min {always_on}, max {endpoint.get('workersMax')}, "
                f"idle timeout {endpoint.get('idleTimeout')}s."
            )
            if not always_on:
                detail += " Scales to zero, so it costs nothing while idle."
            if not ours:
                detail += " Not the endpoint this app is pointed at."

            out.append(
                Instance(
                    provider="runpod",
                    kind="endpoint",
                    id=endpoint.get("id", ""),
                    label=endpoint.get("name") or endpoint.get("id", ""),
                    status=_endpoint_status(live, queued, always_on),
                    active=bool(live or queued or always_on),
                    gpu=", ".join(endpoint.get("gpuTypeIds") or []),
                    started_at=_parse_stamp(endpoint.get("createdAt")),
                    age_label="created",
                    detail=detail,
                    actions=actions,
                )
            )
        return out

    def _pod_instances(self, pods: list[dict]) -> list[Instance]:
        out = []
        for pod in pods:
            status = str(pod.get("desiredStatus") or "unknown").lower()
            running = status == "running"
            gpu = pod.get("machine", {}).get("gpuTypeId") or pod.get("gpuTypeId") or ""
            out.append(
                Instance(
                    provider="runpod",
                    kind="pod",
                    id=pod.get("id", ""),
                    label=pod.get("name") or pod.get("id", ""),
                    status=status,
                    active=running,
                    gpu=f"{pod.get('gpuCount', 1)}× {gpu}" if gpu else "",
                    started_at=_parse_stamp(pod.get("lastStartedAt")),
                    detail=(
                        f"Pod on {pod.get('machineId', 'unknown machine')}, "
                        f"${pod.get('costPerHr', '?')}/hr while running."
                    ),
                    actions=(
                        [
                            {
                                "action": "stop-pod",
                                "label": "Stop pod",
                                "confirm": "Stop this pod? Its disk is kept and it "
                                "can be started again.",
                            }
                        ]
                        if running
                        else []
                    ),
                )
            )
        return out

    def _facts(
        self,
        account: dict[str, Any],
        spend: dict[str, Any],
        endpoints: list[dict],
        endpoint_id: str,
    ) -> list[dict[str, str]]:
        ours = next((e for e in endpoints if e.get("id") == endpoint_id), {})
        facts = []
        if account.get("clientBalance") is not None:
            facts.append(
                {"label": "Balance", "value": f"${float(account['clientBalance']):.2f}"}
            )
        if account.get("currentSpendPerHr") is not None:
            facts.append(
                {
                    "label": "Burn rate",
                    "value": f"${float(account['currentSpendPerHr']):.3f}/hr",
                }
            )
        if spend:
            facts.append({"label": "Spent, 24h", "value": f"${spend.get('day', 0):.2f}"})
            facts.append({"label": "Spent, 7d", "value": f"${spend.get('week', 0):.2f}"})
        if ours:
            facts.append(
                {"label": "Lab endpoint", "value": ours.get("name") or endpoint_id}
            )
        return facts


def _endpoint_status(live: int, queued: int, always_on: int) -> str:
    if live:
        return f"{live} worker(s) up"
    if queued:
        return f"{queued} job(s) waiting"
    if always_on:
        return f"{always_on} always-on worker(s)"
    return "idle"


def _iso(stamp: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stamp))


def _parse_stamp(value: Any) -> float | None:
    """RunPod mixes "2026-08-30T17:45:30.354Z" and "2026-08-30 00:00:00"."""
    if not isinstance(value, str) or not value:
        return None
    text = value.replace("Z", "+00:00").replace(" ", "T", 1)
    try:
        from datetime import datetime, timezone

        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except ValueError:
        return None
