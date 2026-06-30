#!/usr/bin/env bash
# Convenience launcher. Build the image, then run the benchmark inside the
# DeepStream container with GPU access.
set -euo pipefail

IMAGE=${IMAGE:-mot-tracker-benchmark:latest}
VIDEO=${VIDEO:-assets/crowd.mp4}
OUTPUT=${OUTPUT:-output/comparison.mp4}
BACKEND=${BACKEND:-deepstream}
TRACKERS=${TRACKERS:-"bytetrack botsort ocsort deepocsort fasttrack tracktrack"}

case "${1:-run}" in
  build)
    docker build -t "$IMAGE" .
    ;;
  run)
    docker run --rm --gpus all \
      -v "$(pwd)":/workspace/mot-benchmark \
      -w /workspace/mot-benchmark \
      "$IMAGE" \
      python3 -m src.benchmark \
        --video "$VIDEO" \
        --backend "$BACKEND" \
        --pgie-config configs/pgie_yolo_config.txt \
        --trackers $TRACKERS \
        --output "$OUTPUT"
    ;;
  shell)
    docker run --rm -it --gpus all \
      -v "$(pwd)":/workspace/mot-benchmark \
      -w /workspace/mot-benchmark \
      "$IMAGE" bash
    ;;
  *)
    echo "usage: $0 {build|run|shell}"
    exit 1
    ;;
esac
