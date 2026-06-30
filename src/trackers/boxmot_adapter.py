"""Adapter for trackers provided by the `boxmot` library (v18+ API).

boxmot 18 replaced the per-class constructors with a unified factory,
``boxmot.trackers.tracker_zoo.create_tracker(tracker_type, ...)``, and trackers
now return a ``TrackResults`` ndarray-subclass exposing ``.xyxy/.id/.conf/.cls``.
This adapter wraps that so the orchestrator sees a uniform list of ``Track``.

Covers ByteTrack, OcSort (motion-only) and BotSort, DeepOcSort, StrongSort,
BoostSort, etc. (appearance/ReID) -- the ReID ones get a ReID checkpoint.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Track

# Our short names -> boxmot create_tracker type strings (they match in v18).
_BOXMOT_TRACKERS = {
    "bytetrack": "bytetrack",
    "botsort": "botsort",
    "ocsort": "ocsort",
    "deepocsort": "deepocsort",
    "strongsort": "strongsort",
    "boosttrack": "boosttrack",
    "hybridsort": "hybridsort",
    "sfsort": "sfsort",
}


class BoxmotAdapter:
    def __init__(
        self,
        name: str,
        device: str = "cuda:0",
        half: bool = True,
        reid_weights: str = "osnet_x0_25_msmt17.pt",
    ) -> None:
        if name not in _BOXMOT_TRACKERS:
            raise ValueError(
                f"Unknown boxmot tracker '{name}'. "
                f"Available: {sorted(_BOXMOT_TRACKERS)}"
            )
        self.name = name
        self._tracker = self._build(name, device, half, reid_weights)

    @staticmethod
    def _build(name: str, device: str, half: bool, reid_weights: str):
        from boxmot.trackers.tracker_zoo import (
            REID_TRACKERS,
            create_tracker,
            get_tracker_config,
        )

        ttype = _BOXMOT_TRACKERS[name]
        cfg = get_tracker_config(ttype)
        kwargs: dict = {}
        if ttype in REID_TRACKERS:
            kwargs = dict(
                reid_weights=Path(reid_weights), device=device, half=half
            )
        return create_tracker(ttype, tracker_config=cfg, **kwargs)

    def update(self, detections: np.ndarray, frame: np.ndarray) -> list[Track]:
        if detections is None or len(detections) == 0:
            detections = np.empty((0, 6), dtype=np.float32)
        res = self._tracker.update(detections.astype(np.float32), frame)
        if res is None or len(res) == 0:
            return []
        xyxy = np.asarray(res.xyxy)
        ids = np.asarray(res.id)
        conf = np.asarray(res.conf)
        cls = np.asarray(res.cls)
        tracks: list[Track] = []
        for i in range(len(res)):
            x1, y1, x2, y2 = xyxy[i]
            tracks.append(
                Track(
                    track_id=int(ids[i]),
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    conf=float(conf[i]),
                    cls=int(cls[i]),
                )
            )
        return tracks
