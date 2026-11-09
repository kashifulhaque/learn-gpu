"""Chooses which GPU provider to use and which RunPod endpoint a lab needs.

A lab names its card in `meta.json["gpu"]`. The field defaults to
`A100-80GB` when it is absent. `"gpus": 2` selects the two-A100 endpoint
instead of the single-card one. A missing endpoint does not fall through to
a different card.
"""

from typing import Any

from ..config import get_settings
from .base import Provider
from .runpod_provider import RunPodProvider

A100 = "A100-80GB"
H100 = "H100-80GB"

# Order matters: the first entry is the preferred provider, and the compute
# panel renders the providers in this order.
PREFERRED = "runpod"
_BUILDERS = {"runpod": RunPodProvider}


def provider_names() -> list[str]:
    """Every provider this build knows about, preferred one first."""
    return list(_BUILDERS)


def get_provider(name: str | None = None) -> Provider:
    settings = get_settings()
    key = (name or settings.gpu_provider or PREFERRED).lower()
    builder = _BUILDERS.get(key)
    if builder is None:
        raise ValueError(f"Unknown GPU provider: {key}")
    return builder()


def _clean(value: str | None) -> str:
    return (value or "").strip()


def endpoint_id_for(gpu: str | None, gpus: int = 1) -> str:
    """The RunPod endpoint id for this lab, or "" when it is not configured.

    `A100-80GB` is the default, including when `meta.json` omits `gpu`.
    `RUNPOD_ENDPOINT_ID` supplies that endpoint when `RUNPOD_ENDPOINT_A100`
    is unset. `H100-80GB` uses `RUNPOD_ENDPOINT_H100`. A lab with `gpus` of
    2 or more uses `RUNPOD_ENDPOINT_2XA100` and is not sent to a single-GPU
    endpoint.
    """
    settings = get_settings()
    count = gpus if isinstance(gpus, int) and gpus > 0 else 1
    if count >= 2:
        return _clean(settings.runpod_endpoint_2xa100)
    name = _clean(gpu) or A100
    if name == H100:
        return _clean(settings.runpod_endpoint_h100)
    if name == A100:
        return _clean(settings.runpod_endpoint_a100) or _clean(settings.runpod_endpoint_id)
    return ""


def any_endpoint_configured() -> bool:
    """True when this deployment can run at least one GPU type."""
    settings = get_settings()
    return bool(
        _clean(settings.runpod_endpoint_a100)
        or _clean(settings.runpod_endpoint_id)
        or _clean(settings.runpod_endpoint_h100)
        or _clean(settings.runpod_endpoint_2xa100)
    )


def unavailable_event(gpu: str | None, gpus: int = 1) -> dict[str, Any]:
    """The event the lab pane shows when this deployment has no such endpoint.

    An unconfigured H100 uses the sentence "needs H100, not configured on
    this deployment". Other missing endpoints use the same shape. The lab
    is not sent to a different GPU.
    """
    name = _clean(gpu) or A100
    count = gpus if isinstance(gpus, int) and gpus > 0 else 1
    if count >= 2:
        phrase = "2 A100 GPUs"
    elif name == H100:
        phrase = "H100"
    elif name == A100:
        phrase = "A100"
    else:
        phrase = name
    return {
        "type": "unavailable",
        "gpu": name,
        "gpus": count,
        "message": f"needs {phrase}, not configured on this deployment",
    }


def provider_status() -> dict[str, Any]:
    """Each provider's state right now, for the lab pane and the compute panel."""
    settings = get_settings()
    providers = []
    for key, builder in _BUILDERS.items():
        ok, why = builder().available()
        providers.append(
            {
                "name": key,
                "available": ok,
                "reason": why,
                "preferred": key == PREFERRED,
            }
        )
    return {
        "default": settings.gpu_provider or PREFERRED,
        "preferred": PREFERRED,
        "providers": providers,
    }
