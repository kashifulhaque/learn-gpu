"""Chooses which GPU provider to use and reports what is configured."""

from typing import Any

from ..config import get_settings
from .base import Provider
from .runpod_provider import RunPodProvider

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
