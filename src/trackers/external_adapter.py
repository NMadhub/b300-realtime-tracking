"""Adapters for trackers that are NOT in boxmot: FastTrack and TrackTrack.

These two appear in the reference image but ship as standalone research repos,
not as a pip package. Rather than hard-fail, this module gives you:

  1. A clear, single place to wire each repo in (`_load_fasttrack` /
     `_load_tracktrack`), and
  2. A graceful fallback so the 6-up grid still renders even before you've
     cloned them -- the tile is drawn with a "STUB" banner so the comparison is
     honest about what's real.

To make a tile real: clone the upstream repo into ./third_party/, import its
tracker inside the matching `_load_*` function, and convert its output rows into
`Track` objects (same shape boxmot uses: x1,y1,x2,y2,id,conf,cls).

Upstream references:
  TrackTrack (CVPR 2025): https://github.com/kamkyu94/TrackTrack
  FastTrack:              see the post author's repo / your chosen implementation
"""
from __future__ import annotations

import numpy as np

from .base import Track


class ExternalTrackerAdapter:
    def __init__(self, name: str, device: str = "cuda:0") -> None:
        self.name = name
        self.device = device
        self.is_stub = False
        self._impl = None

        if name == "fasttrack":
            self._impl = self._load_fasttrack()
        elif name == "tracktrack":
            self._impl = self._load_tracktrack()
        else:
            raise ValueError(f"No external adapter for '{name}'")

        if self._impl is None:
            self.is_stub = True

    # ----- wire your real implementations in here -------------------------- #
    def _load_fasttrack(self):
        try:
            # from third_party.fasttrack import FastTrack
            # return FastTrack(device=self.device)
            raise ImportError
        except ImportError:
            return None

    def _load_tracktrack(self):
        try:
            # from third_party.tracktrack import TrackTrack
            # return TrackTrack(device=self.device)
            raise ImportError
        except ImportError:
            return None

    # ---------------------------------------------------------------------- #
    def update(self, detections: np.ndarray, frame: np.ndarray) -> list[Track]:
        if self.is_stub:
            # No real tracker yet: pass detections through with no IDs so the
            # tile still shows boxes (clearly marked STUB by the visualizer).
            tracks: list[Track] = []
            if detections is not None:
                for x1, y1, x2, y2, conf, cls in detections:
                    tracks.append(
                        Track(-1, x1, y1, x2, y2, float(conf), int(cls))
                    )
            return tracks

        out = self._impl.update(detections.astype(np.float32), frame)
        return [
            Track(int(r[4]), r[0], r[1], r[2], r[3], float(r[5]), int(r[6]))
            for r in out
        ]
