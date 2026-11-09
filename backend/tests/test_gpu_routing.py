"""Lab routing by meta.json gpu, and the unavailable event when an endpoint is missing."""

import asyncio
import json

import pytest

from backend.app.config import get_settings
from backend.app.providers.registry import endpoint_id_for, unavailable_event

ENDPOINT_VARS = (
    "RUNPOD_API_KEY",
    "RUNPOD_ENDPOINT_ID",
    "RUNPOD_ENDPOINT_A100",
    "RUNPOD_ENDPOINT_H100",
    "RUNPOD_ENDPOINT_2XA100",
)


@pytest.fixture
def endpoints(monkeypatch):
    """Blank every endpoint variable so a developer .env cannot leak into the test."""
    for name in ENDPOINT_VARS:
        monkeypatch.setenv(name, "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def use_env(monkeypatch, **values):
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()


def test_a100_uses_runpod_endpoint_id_when_the_specific_variable_is_unset(endpoints, monkeypatch):
    use_env(monkeypatch, RUNPOD_ENDPOINT_ID="a100-legacy")
    assert endpoint_id_for("A100-80GB") == "a100-legacy"
    assert endpoint_id_for(None) == "a100-legacy"
    assert endpoint_id_for("") == "a100-legacy"


def test_explicit_a100_endpoint_wins_over_the_fallback(endpoints, monkeypatch):
    use_env(
        monkeypatch,
        RUNPOD_ENDPOINT_ID="legacy",
        RUNPOD_ENDPOINT_A100="a100-specific",
    )
    assert endpoint_id_for("A100-80GB") == "a100-specific"


def test_h100_and_two_gpu_labs_do_not_use_the_a100_endpoint(endpoints, monkeypatch):
    use_env(
        monkeypatch,
        RUNPOD_ENDPOINT_ID="a100-legacy",
        RUNPOD_ENDPOINT_H100="h100-id",
        RUNPOD_ENDPOINT_2XA100="two-id",
    )
    assert endpoint_id_for("H100-80GB") == "h100-id"
    assert endpoint_id_for("A100-80GB", gpus=2) == "two-id"
    assert endpoint_id_for("H100-80GB", gpus=2) == "two-id"
    # An unknown card is not sent to the A100 endpoint.
    assert endpoint_id_for("B200") == ""


def test_missing_h100_endpoint_reports_the_deployment_sentence(endpoints, monkeypatch):
    use_env(monkeypatch, RUNPOD_ENDPOINT_ID="a100-legacy")
    event = unavailable_event("H100-80GB")
    assert event == {
        "type": "unavailable",
        "gpu": "H100-80GB",
        "gpus": 1,
        "message": "needs H100, not configured on this deployment",
    }
    assert endpoint_id_for("H100-80GB") == ""
    two = unavailable_event("A100-80GB", gpus=2)
    assert two["message"] == "needs 2 A100 GPUs, not configured on this deployment"
    assert two["gpus"] == 2


def test_a_lab_without_gpus_counts_as_one(content):
    from backend.app import curriculum

    lab = content / "labs" / "h100-lab"
    lab.mkdir()
    (lab / "meta.json").write_text(
        json.dumps({"id": "h100-lab", "gpu": "H100-80GB", "title": "T"})
    )
    (lab / "starter.py").write_text("x = 1\n")
    (lab / "solution.py").write_text("x = 2\n")
    pair = content / "labs" / "two-lab"
    pair.mkdir()
    (pair / "meta.json").write_text(
        json.dumps({"id": "two-lab", "gpu": "A100-80GB", "gpus": 2})
    )
    (pair / "starter.py").write_text("x = 1\n")
    (pair / "solution.py").write_text("x = 2\n")
    omitted = content / "labs" / "plain-lab"
    omitted.mkdir()
    (omitted / "meta.json").write_text(json.dumps({"id": "plain-lab"}))
    (omitted / "starter.py").write_text("x = 1\n")
    (omitted / "solution.py").write_text("x = 2\n")
    curriculum.reload_content()

    assert curriculum.get_lab("h100-lab")["gpus"] == 1
    assert curriculum.get_lab("h100-lab")["gpu"] == "H100-80GB"
    assert curriculum.get_lab("two-lab")["gpus"] == 2
    assert curriculum.get_lab("plain-lab")["gpu"] == "A100-80GB"
    assert curriculum.get_lab("plain-lab")["gpus"] == 1


def _events(provider, **kwargs):
    async def collect():
        return [event async for event in provider.run_lab(**kwargs)]

    return asyncio.run(collect())


def test_missing_h100_is_unavailable_and_cpu_checks_still_run(
    endpoints, monkeypatch, tmp_path
):
    """A missing H100 endpoint does not call RunPod and does not raise."""
    use_env(monkeypatch, RUNPOD_ENDPOINT_ID="a100-legacy", RUNPOD_API_KEY="test-key")
    lab = tmp_path / "gated"
    lab.mkdir()
    (lab / "meta.json").write_text(json.dumps({"id": "gated", "gpu": "H100-80GB"}))
    (lab / "tests.py").write_text(
        "def run(submission):\n"
        "    return {'passed': True, 'checks': "
        "[{'name': 'cpu add', 'passed': submission.add(1, 1) == 2, 'detail': '1+1'}],"
        " 'metrics': {}}\n"
    )
    monkeypatch.setenv("LG_LABS_DIR", str(tmp_path))

    def refuse_runpod(*args, **kwargs):
        raise AssertionError("the lab was sent to RunPod")

    import httpx

    monkeypatch.setattr(httpx.AsyncClient, "__init__", refuse_runpod)

    from backend.app.providers.runpod_provider import RunPodProvider

    events = _events(
        RunPodProvider(),
        lab_id="gated",
        code="def add(a, b):\n    return a + b\n",
        gpu="H100-80GB",
        timeout=30,
    )
    assert events[0] == {
        "type": "unavailable",
        "gpu": "H100-80GB",
        "gpus": 1,
        "message": "needs H100, not configured on this deployment",
    }
    result = next(event for event in events if event["type"] == "result")
    assert result["passed"] is True
    assert result["checks"][0]["name"] == "cpu add"
    assert not any(event["type"] == "error" for event in events)


def test_a100_run_posts_to_the_legacy_endpoint(endpoints, monkeypatch):
    use_env(monkeypatch, RUNPOD_ENDPOINT_ID="a100-legacy", RUNPOD_API_KEY="test-key")
    posts = []

    class FakeResponse:
        def __init__(self, payload):
            self.status_code = 200
            self._payload = payload
            self.text = ""

        def json(self):
            return self._payload

        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json=None, headers=None):
            posts.append(url)
            return FakeResponse({"id": "job-1"})

        async def get(self, url, headers=None):
            return FakeResponse({"status": "COMPLETED", "stream": []})

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    from backend.app.providers.runpod_provider import RunPodProvider

    events = _events(
        RunPodProvider(),
        lab_id="03-bandwidth-probe",
        code="x = 1\n",
        gpu="A100-80GB",
        timeout=30,
    )
    assert posts == ["https://api.runpod.ai/v2/a100-legacy/run"]
    assert events[0]["type"] == "job"
    assert events[0]["job_id"] == "job-1"
    assert not any(event["type"] == "unavailable" for event in events)


def test_two_gpus_do_not_fall_back_to_the_a100_endpoint(endpoints, monkeypatch, tmp_path):
    use_env(monkeypatch, RUNPOD_ENDPOINT_ID="a100-legacy", RUNPOD_API_KEY="test-key")
    lab = tmp_path / "nccl"
    lab.mkdir()
    (lab / "tests.py").write_text(
        "def run(submission):\n"
        "    return {'passed': True, 'checks': "
        "[{'name': 'cpu', 'passed': True, 'detail': ''}], 'metrics': {}}\n"
    )
    monkeypatch.setenv("LG_LABS_DIR", str(tmp_path))

    def refuse_runpod(*args, **kwargs):
        raise AssertionError("a 2-GPU lab was sent to the A100 endpoint")

    import httpx

    monkeypatch.setattr(httpx.AsyncClient, "__init__", refuse_runpod)

    from backend.app.providers.runpod_provider import RunPodProvider

    events = _events(
        RunPodProvider(),
        lab_id="nccl",
        code="x = 1\n",
        gpu="A100-80GB",
        timeout=30,
        gpus=2,
    )
    assert events[0]["type"] == "unavailable"
    assert events[0]["message"] == "needs 2 A100 GPUs, not configured on this deployment"
    assert events[0]["gpus"] == 2
    assert any(event["type"] == "result" and event["passed"] for event in events)
