"""Rendering: draw tracked boxes + IDs, label each tile, build the 2x3 grid.

The reference image is a 2x3 mosaic where each tile is the same scene under a
different tracker, with a colored name banner in the top-left. This module
reproduces that layout.
"""
from __future__ import annotations

import colorsys

import cv2
import numpy as np

from .trackers import Track

# Per-tile banner color, roughly matching the reference image's palette.
TILE_COLORS = {
    "botsort": (255, 110, 180),
    "bytetrack": (60, 120, 255),
    "ocsort": (40, 200, 120),
    "deepocsort": (255, 200, 40),
    "fasttrack": (80, 200, 255),
    "tracktrack": (200, 120, 255),
}


def _id_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic, well-spread BGR color per track id (golden-ratio hue)."""
    if track_id < 0:
        return (160, 160, 160)
    h = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


def draw_tracks(
    frame: np.ndarray,
    tracks: list[Track],
    tracker_name: str,
    is_stub: bool = False,
) -> np.ndarray:
    """Draw boxes + IDs on a copy of the frame, with a labeled banner."""
    img = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = t.ltrb()
        color = _id_color(t.track_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{t.track_id}" if t.track_id >= 0 else "?"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            img,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    _banner(img, tracker_name, is_stub)
    return img


def _banner(img: np.ndarray, name: str, is_stub: bool) -> None:
    color = TILE_COLORS.get(name, (60, 160, 60))
    text = name + ("  [STUB]" if is_stub else "")
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(img, (8, 8), (8 + tw + 16, 8 + th + 16), color, -1)
    cv2.putText(
        img,
        text,
        (16, 8 + th + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def make_grid(
    tiles: list[np.ndarray],
    cols: int = 2,
    tile_w: int = 640,
    tile_h: int = 360,
) -> np.ndarray:
    """Compose tiles into a grid (defaults to 2 cols -> 2x3 for 6 trackers)."""
    rows = (len(tiles) + cols - 1) // cols
    canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        resized = cv2.resize(tile, (tile_w, tile_h))
        canvas[
            r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w
        ] = resized
    return canvas


class GridVideoWriter:
    def __init__(self, path: str, fps: float = 30.0) -> None:
        self.path = path
        self.fps = fps
        self._writer: cv2.VideoWriter | None = None

    def write(self, grid: np.ndarray) -> None:
        if self._writer is None:
            h, w = grid.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                self.path, fourcc, self.fps, (w, h)
            )
        self._writer.write(grid)

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
