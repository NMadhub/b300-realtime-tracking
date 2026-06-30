"""Pluggable tracker adapter interface.

Every tracker in the benchmark is wrapped so that the orchestrator can treat
them identically: feed identical per-frame detections in, get back a uniform
list of tracks out. This is what makes the comparison fair -- detection happens
once, and only the association/tracking stage differs between tiles.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Track:
    """One tracked object in one frame, in a tracker-agnostic format."""

    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int

    def ltrb(self) -> tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)


class TrackerAdapter(Protocol):
    """Interface every tracker wrapper must implement.

    Detections are passed as an (N, 6) float array of
    ``[x1, y1, x2, y2, conf, cls]`` in absolute pixel coordinates -- the exact
    format boxmot expects -- plus the raw BGR frame (needed by appearance/ReID
    trackers such as BoTSORT and DeepOCSORT).
    """

    name: str

    def update(self, detections: np.ndarray, frame: np.ndarray) -> list[Track]:
        ...
