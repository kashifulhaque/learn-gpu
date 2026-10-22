"""Shared interface for the GPU backends that execute labs."""

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol


class ProviderError(RuntimeError):
    """The provider could not run the lab."""


class OutOfCredits(ProviderError):
    """The provider rejected the job because the account is out of credit.

    The app catches this to tell the reader to add credit.
    """


@dataclass
class LabResult:
    passed: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    detail: str = ""


@dataclass
class Instance:
    """One thing that is running, or could run, and may cost money.

    A RunPod pod, or the worker pool behind a RunPod
    serverless endpoint. The compute panel renders these and offers `actions`
    as buttons.
    """

    provider: str
    kind: str
    id: str
    label: str
    status: str
    active: bool
    gpu: str = ""
    started_at: float | None = None
    # How to read `started_at`: a container has been "up" that long, whereas a
    # serverless endpoint was "created" then and bills only while it works.
    age_label: str = "up"
    detail: str = ""
    actions: list[dict[str, str]] = field(default_factory=list)


class Provider(Protocol):
    name: str

    def available(self) -> tuple[bool, str]:
        """Whether the provider is configured, and why not if it isn't."""

    def run_lab(
        self, lab_id: str, code: str, gpu: str, timeout: int
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield events while the lab runs.

        Each event is a dict with a "type" of "log", "result", or "error".
        """


class Manager(Protocol):
    """The management half of a provider: what is running, and how to stop it.

    Kept separate from `Provider` because the compute panel needs it and lab
    execution does not.
    """

    name: str
    console_url: str

    async def snapshot(self) -> dict[str, Any]:
        """Instances and spend, for the compute panel."""

    async def act(self, action: str, target: str) -> str:
        """Run one action from an instance's `actions` list. Returns a message."""
