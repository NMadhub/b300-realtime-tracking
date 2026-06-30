"""In-container environment smoke test.

Verifies the DeepStream + tracking stack is wired up before running the full
benchmark: pyds bindings, Torch CUDA on this GPU, boxmot/opencv, the GStreamer
nvidia plugins, and that our CLI imports cleanly.
"""
import shutil
import subprocess


def check(name, fn):
    try:
        print(f"[ OK ] {name}: {fn()}")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def _pyds():
    import pyds  # noqa: F401
    return "import succeeded"


def _torch():
    import torch
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-cuda"
    return f"torch={torch.__version__} cuda_avail={torch.cuda.is_available()} dev={dev}"


def _boxmot():
    import boxmot
    import cv2
    return f"boxmot={boxmot.__version__} cv2={cv2.__version__}"


def _gst(plugin):
    if shutil.which("gst-inspect-1.0") is None:
        raise RuntimeError("gst-inspect-1.0 not found")
    r = subprocess.run(
        ["gst-inspect-1.0", plugin], capture_output=True, text=True
    )
    if r.returncode != 0:
        raise RuntimeError("plugin not registered")
    return "registered"


def _cli():
    import importlib
    importlib.import_module("src.benchmark")
    return "src.benchmark imports"


if __name__ == "__main__":
    check("pyds (DeepStream python bindings)", _pyds)
    check("torch CUDA", _torch)
    check("boxmot + opencv", _boxmot)
    check("gst nvinfer", lambda: _gst("nvinfer"))
    check("gst nvstreammux", lambda: _gst("nvstreammux"))
    check("gst nvtracker", lambda: _gst("nvtracker"))
    check("our CLI", _cli)
