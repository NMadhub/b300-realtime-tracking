"""B300 showcase benchmark.

Measures REAL numbers on the NVIDIA B300 SXM6 to demonstrate what one GPU (and
the full node) can do for real-time multi-object tracking:

  1. Raw tensor compute  -- sustained BF16 matmul TFLOPS, per-GPU and aggregate
     across every GPU in the node (run concurrently).
  2. Applied detection   -- YOLO people-detector throughput (FP16, batched) in
     frames/sec, swept over batch size.
  3. Applied tracking    -- ByteTrack association throughput in frames/sec.
  4. Derived capacity    -- how many 1080p@30fps streams can be tracked in real
     time on one B300 and on the whole node.

Results are written to output/b300_results.json for the canvas dashboard.
All numbers are measured here; nothing is hard-coded.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torch.multiprocessing as mp

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "b300_results.json")


# --------------------------------------------------------------------------- #
# 1. Raw compute: sustained BF16 matmul TFLOPS
# --------------------------------------------------------------------------- #
def matmul_tflops(device: str, n: int = 16384, iters: int = 50, warmup: int = 10) -> float:
    torch.cuda.set_device(device)
    a = torch.randn(n, n, device=device, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=device, dtype=torch.bfloat16)
    for _ in range(warmup):
        c = a @ b
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0
    flops = 2.0 * (n ** 3) * iters
    del a, b, c
    torch.cuda.empty_cache()
    return flops / dt / 1e12


def _tflops_worker(gpu_id: int):
    return matmul_tflops(f"cuda:{gpu_id}")


def aggregate_tflops(num_gpus: int) -> tuple[float, list[float]]:
    """Run the matmul on every GPU concurrently; return (sum, per-gpu)."""
    ctx = mp.get_context("spawn")
    with ctx.Pool(num_gpus) as pool:
        per_gpu = pool.map(_tflops_worker, list(range(num_gpus)))
    return float(sum(per_gpu)), [float(x) for x in per_gpu]


# --------------------------------------------------------------------------- #
# 2. Applied: YOLO detection throughput (FP16, batched)
# --------------------------------------------------------------------------- #
def yolo_throughput(device: str = "cuda:0", imgsz: int = 640) -> dict:
    from ultralytics import YOLO

    model = YOLO("yolov8m.pt").model.eval().to(device).half()
    out = {"imgsz": imgsz, "by_batch": []}
    for bs in [1, 8, 16, 32, 64]:
        x = torch.randn(bs, 3, imgsz, imgsz, device=device, dtype=torch.float16)
        with torch.no_grad():
            for _ in range(5):  # warmup
                model(x)
        torch.cuda.synchronize(device)
        iters = 30
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(iters):
                model(x)
        torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        fps = bs * iters / dt
        out["by_batch"].append({"batch": bs, "fps": round(fps, 1)})
        del x
        torch.cuda.empty_cache()
    out["peak_fps"] = max(b["fps"] for b in out["by_batch"])
    return out


# --------------------------------------------------------------------------- #
# 3. Applied: ByteTrack association throughput
# --------------------------------------------------------------------------- #
def tracking_throughput(n_objects: int = 40, frames: int = 300) -> dict:
    from boxmot.trackers.tracker_zoo import create_tracker, get_tracker_config

    tracker = create_tracker("bytetrack", tracker_config=get_tracker_config("bytetrack"))
    rng = np.random.default_rng(0)
    blank = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # warmup
    for _ in range(20):
        tracker.update(_synth_dets(rng, n_objects), blank)
    t0 = time.perf_counter()
    for _ in range(frames):
        tracker.update(_synth_dets(rng, n_objects), blank)
    dt = time.perf_counter() - t0
    return {"objects_per_frame": n_objects, "fps": round(frames / dt, 1)}


def _synth_dets(rng, n: int) -> np.ndarray:
    cx = rng.uniform(50, 1870, n)
    cy = rng.uniform(50, 1030, n)
    w = rng.uniform(30, 80, n)
    h = rng.uniform(60, 160, n)
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    conf = rng.uniform(0.5, 0.95, n)
    cls = np.zeros(n)
    return np.stack([x1, y1, x2, y2, conf, cls], axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
def main() -> None:
    assert torch.cuda.is_available(), "CUDA not available"
    num_gpus = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)

    results: dict = {
        "gpu_name": gpu_name,
        "num_gpus": num_gpus,
        "mem_per_gpu_gb": round(props.total_memory / 1e9, 1),
        "torch": torch.__version__,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    print(f"[b300] {num_gpus}x {gpu_name} ({results['mem_per_gpu_gb']} GB each)")

    print("[1/4] single-GPU BF16 matmul TFLOPS ...")
    results["tflops_single"] = round(matmul_tflops("cuda:0"), 1)
    print(f"      -> {results['tflops_single']} TFLOPS")

    print(f"[2/4] aggregate BF16 TFLOPS across {num_gpus} GPUs (concurrent) ...")
    agg, per_gpu = aggregate_tflops(num_gpus)
    results["tflops_aggregate"] = round(agg, 1)
    results["tflops_per_gpu"] = [round(x, 1) for x in per_gpu]
    results["pflops_aggregate"] = round(agg / 1000, 2)
    print(f"      -> {results['pflops_aggregate']} PFLOPS aggregate")

    try:
        print("[3/4] YOLOv8m detection throughput (FP16, batched) ...")
        results["detection"] = yolo_throughput()
        print(f"      -> peak {results['detection']['peak_fps']} fps")
    except Exception as e:  # noqa: BLE001
        results["detection_error"] = f"{type(e).__name__}: {e}"
        print(f"      !! detection failed: {e}")

    try:
        print("[4/4] ByteTrack association throughput ...")
        results["tracking"] = tracking_throughput()
        print(f"      -> {results['tracking']['fps']} fps")
    except Exception as e:  # noqa: BLE001
        results["tracking_error"] = f"{type(e).__name__}: {e}"
        print(f"      !! tracking failed: {e}")

    # Derived real-time capacity at 1080p@30fps (detection is the GPU bottleneck;
    # trackers run in parallel on CPU cores).
    if "detection" in results:
        det = results["detection"]["peak_fps"]
        per_gpu_streams = det / 30.0
        results["realtime_streams_per_gpu"] = int(per_gpu_streams)
        results["realtime_streams_node"] = int(per_gpu_streams * num_gpus)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] wrote {RESULTS_PATH}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
