"""Tracker registry: maps a name string to a constructed adapter."""
from __future__ import annotations

from .base import Track, TrackerAdapter
from .boxmot_adapter import BoxmotAdapter, _BOXMOT_TRACKERS
from .external_adapter import ExternalTrackerAdapter

_EXTERNAL = {"fasttrack", "tracktrack"}


def build_tracker(name: str, **kwargs) -> TrackerAdapter:
    """Construct a tracker adapter by name.

    boxmot trackers (bytetrack/botsort/ocsort/deepocsort/...) and external
    trackers (fasttrack/tracktrack) are resolved transparently.
    """
    name = name.lower()
    if name in _BOXMOT_TRACKERS:
        return BoxmotAdapter(name, **kwargs)
    if name in _EXTERNAL:
        device = kwargs.get("device", "cuda:0")
        return ExternalTrackerAdapter(name, device=device)
    raise ValueError(
        f"Unknown tracker '{name}'. "
        f"Known: {sorted(set(_BOXMOT_TRACKERS) | _EXTERNAL)}"
    )


def available_trackers() -> list[str]:
    return sorted(set(_BOXMOT_TRACKERS) | _EXTERNAL)


__all__ = [
    "Track",
    "TrackerAdapter",
    "build_tracker",
    "available_trackers",
]
