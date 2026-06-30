"""Multi-tracker benchmark orchestrator.

Runs ONE detector over a video and fans the identical per-frame detections out
to N trackers, rendering a side-by-side grid video (the 2x3 mosaic from the
reference image) plus per-tracker timing stats.

Usage (fallback detector, runs anywhere with a GPU + ultralytics):
    python -m src.benchmark \
        --video assets/crowd.mp4 \
        --backend ultralytics \
        --trackers bytetrack botsort ocsort deepocsort strongsort boosttrack \
        --output output/comparison.mp4

Usage (DeepStream, inside the DeepStream container):
    python -m src.benchmark \
        --video assets/crowd.mp4 \
        --backend deepstream \
        --pgie-config configs/pgie_yolo_config.txt \
        --trackers bytetrack botsort ocsort deepocsort fasttrack tracktrack \
        --output output/comparison.mp4
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

from .trackers import available_trackers, build_tracker
from .visualize import GridVideoWriter, draw_tracks, make_grid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-object tracker benchmark grid")
    p.add_argument("--video", required=True, help="input video path")
    p.add_argument(
        "--backend",
        choices=["deepstream", "ultralytics"],
        default="ultralytics",
        help="detection backend",
    )
    p.add_argument(
        "--trackers",
        nargs="+",
        default=["bytetrack", "botsort", "ocsort", "deepocsort"],
        help=f"trackers to compare. options: {available_trackers()}",
    )
    p.add_argument("--output", default="output/comparison.mp4")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--cols", type=int, default=2, help="grid columns")
    p.add_argument("--tile-w", type=int, default=640)
    p.add_argument("--tile-h", type=int, default=360)
    p.add_argument("--max-frames", type=int, default=None)
    # ultralytics backend
    p.add_argument("--yolo", default="yolov8m.pt")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument(
        "--classes", type=int, nargs="+", default=[0], help="class ids to keep"
    )
    # deepstream backend
    p.add_argument("--pgie-config", default="configs/pgie_yolo_config.txt")
    p.add_argument("--mux-width", type=int, default=1920)
    p.add_argument("--mux-height", type=int, default=1080)
    return p.parse_args()


def build_detector(args):
    if args.backend == "deepstream":
        from .detector import DeepStreamDetector

        return DeepStreamDetector(
            video_path=args.video,
            pgie_config=args.pgie_config,
            muxer_width=args.mux_width,
            muxer_height=args.mux_height,
            max_frames=args.max_frames,
        )
    from .detector import UltralyticsDetector

    return UltralyticsDetector(
        video_path=args.video,
        model=args.yolo,
        conf=args.conf,
        classes=args.classes,
        device=args.device,
        max_frames=args.max_frames,
    )


def main() -> None:
    args = parse_args()

    print(f"[init] backend={args.backend} trackers={args.trackers}")
    detector = build_detector(args)
    trackers = [
        build_tracker(name, device=args.device) for name in args.trackers
    ]
    writer = GridVideoWriter(args.output, fps=args.fps)

    timings: dict[str, float] = defaultdict(float)
    id_seen: dict[str, set[int]] = defaultdict(set)
    n_frames = 0

    t_start = time.time()
    for idx, frame, dets in detector.frames():
        tiles = []
        for tr in trackers:
            t0 = time.perf_counter()
            tracks = tr.update(dets, frame)
            timings[tr.name] += time.perf_counter() - t0

            for t in tracks:
                if t.track_id >= 0:
                    id_seen[tr.name].add(t.track_id)

            is_stub = getattr(tr, "is_stub", False)
            tiles.append(draw_tracks(frame, tracks, tr.name, is_stub))

        grid = make_grid(
            tiles, cols=args.cols, tile_w=args.tile_w, tile_h=args.tile_h
        )
        writer.write(grid)
        n_frames += 1
        if n_frames % 30 == 0:
            print(f"[run] processed {n_frames} frames "
                  f"({n_frames / (time.time() - t_start):.1f} fps overall)")

    writer.release()
    _report(timings, id_seen, n_frames, args.output)


def _report(timings, id_seen, n_frames, output) -> None:
    print("\n" + "=" * 60)
    print(f"Benchmark complete: {n_frames} frames -> {output}")
    print("=" * 60)
    print(f"{'tracker':<14}{'avg ms/frame':>14}{'fps':>10}{'unique IDs':>12}")
    print("-" * 60)
    for name, total in timings.items():
        ms = (total / max(n_frames, 1)) * 1000
        fps = (n_frames / total) if total > 0 else 0.0
        print(f"{name:<14}{ms:>14.2f}{fps:>10.1f}{len(id_seen[name]):>12}")
    print("=" * 60)
    print("Note: 'unique IDs' is a rough proxy -- inflated counts often mean "
          "more ID switches. For real MOTA/IDF1, run TrackEval against GT.")


if __name__ == "__main__":
    main()
