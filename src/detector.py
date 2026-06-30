"""Detection sources.

The benchmark needs, per frame: the BGR image plus an (N, 6) detection array
``[x1, y1, x2, y2, conf, cls]``. Detection runs ONCE and is shared across all
trackers -- that is the whole point of a fair tracker comparison.

Two backends are provided:

  * DeepStreamDetector  -- GPU decode + nvinfer (YOLO) inside a GStreamer
    pipeline, with detections pulled out via a pyds pad probe and frames via an
    appsink. This is the intended production path (DeepStream + pyds).

  * UltralyticsDetector -- a pure-PyTorch fallback (ultralytics YOLO + OpenCV
    decode). Use it to develop/validate the tracking + visualization logic on
    any machine before you have the DeepStream container running.

Both expose the same generator: `frames()` yielding (idx, frame_bgr, dets).
"""
from __future__ import annotations

from typing import Iterator, Protocol

import numpy as np


class DetectionSource(Protocol):
    def frames(self) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        ...


# --------------------------------------------------------------------------- #
# Fallback: ultralytics YOLO + OpenCV (no DeepStream required)
# --------------------------------------------------------------------------- #
class UltralyticsDetector:
    def __init__(
        self,
        video_path: str,
        model: str = "yolov8m.pt",
        conf: float = 0.3,
        classes: list[int] | None = None,
        device: str = "cuda:0",
        max_frames: int | None = None,
    ) -> None:
        self.video_path = video_path
        self.conf = conf
        self.classes = classes if classes is not None else [0]  # person
        self.max_frames = max_frames
        from ultralytics import YOLO

        self.model = YOLO(model)
        self.device = device

    def frames(self) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        import cv2

        cap = cv2.VideoCapture(self.video_path)
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if self.max_frames is not None and idx >= self.max_frames:
                    break
                res = self.model.predict(
                    frame,
                    conf=self.conf,
                    classes=self.classes,
                    device=self.device,
                    verbose=False,
                )[0]
                if res.boxes is not None and len(res.boxes) > 0:
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    cfs = res.boxes.conf.cpu().numpy().reshape(-1, 1)
                    cls = res.boxes.cls.cpu().numpy().reshape(-1, 1)
                    dets = np.concatenate([xyxy, cfs, cls], axis=1)
                else:
                    dets = np.empty((0, 6), dtype=np.float32)
                yield idx, frame, dets.astype(np.float32)
                idx += 1
        finally:
            cap.release()


# --------------------------------------------------------------------------- #
# DeepStream: GPU decode + nvinfer detection, frames + dets via pyds
# --------------------------------------------------------------------------- #
class DeepStreamDetector:
    """DeepStream detection source using pyds.

    Pipeline:
        filesrc -> decodebin -> nvstreammux -> nvinfer(PGIE/YOLO)
                -> nvvideoconvert -> capsfilter(RGBA) -> appsink

    The pad probe on the appsink pad reads NvDsObjectMeta (detections) and
    `pyds.get_nvds_buf_surface` gives the frame as a numpy array. Both are
    queued so `frames()` can yield them together.

    NOTE: this requires running inside the DeepStream container (pyds + the GST
    nvidia plugins). It cannot run on a machine without them.
    """

    def __init__(
        self,
        video_path: str,
        pgie_config: str,
        muxer_width: int = 1920,
        muxer_height: int = 1080,
        person_class_id: int = 0,
        max_frames: int | None = None,
    ) -> None:
        self.video_path = video_path
        self.pgie_config = pgie_config
        self.muxer_width = muxer_width
        self.muxer_height = muxer_height
        self.person_class_id = person_class_id
        self.max_frames = max_frames
        self._queue: list[tuple[int, np.ndarray, np.ndarray]] = []

    def _osd_sink_pad_probe(self, pad, info, _u):
        import pyds
        from gi.repository import Gst

        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            # frame as numpy (RGBA) -> BGR copy
            surf = pyds.get_nvds_buf_surface(
                hash(gst_buffer), frame_meta.batch_id
            )
            frame_rgba = np.array(surf, copy=True, order="C")
            frame_bgr = frame_rgba[:, :, [2, 1, 0]].copy()

            dets = []
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                obj = pyds.NvDsObjectMeta.cast(l_obj.data)
                r = obj.rect_params
                x1, y1 = r.left, r.top
                x2, y2 = r.left + r.width, r.top + r.height
                dets.append(
                    [x1, y1, x2, y2, obj.confidence, obj.class_id]
                )
                l_obj = l_obj.next

            arr = (
                np.array(dets, dtype=np.float32)
                if dets
                else np.empty((0, 6), dtype=np.float32)
            )
            self._queue.append(
                (frame_meta.frame_num, frame_bgr, arr)
            )
            l_frame = l_frame.next
        return Gst.PadProbeReturn.OK

    def frames(self) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst, GLib

        Gst.init(None)
        pipeline = Gst.Pipeline()

        def mk(factory, name):
            el = Gst.ElementFactory.make(factory, name)
            if not el:
                raise RuntimeError(f"Failed to create element: {factory}")
            pipeline.add(el)
            return el

        source = mk("filesrc", "file-source")
        source.set_property("location", self.video_path)
        decoder = mk("decodebin", "decoder")
        streammux = mk("nvstreammux", "stream-muxer")
        streammux.set_property("batch-size", 1)
        streammux.set_property("width", self.muxer_width)
        streammux.set_property("height", self.muxer_height)
        streammux.set_property("batched-push-timeout", 4000000)
        pgie = mk("nvinfer", "primary-inference")
        pgie.set_property("config-file-path", self.pgie_config)
        nvvidconv = mk("nvvideoconvert", "convertor")
        capsfilter = mk("capsfilter", "caps")
        caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
        capsfilter.set_property("caps", caps)
        sink = mk("fakesink", "sink")
        sink.set_property("sync", 0)

        def on_pad_added(_dbin, pad):
            sinkpad = streammux.request_pad_simple("sink_0")
            pad.link(sinkpad)

        decoder.connect("pad-added", on_pad_added)

        source.link(decoder)
        streammux.link(pgie)
        pgie.link(nvvidconv)
        nvvidconv.link(capsfilter)
        capsfilter.link(sink)

        sinkpad = sink.get_static_pad("sink")
        sinkpad.add_probe(
            Gst.PadProbeType.BUFFER, self._osd_sink_pad_probe, 0
        )

        loop = GLib.MainLoop()
        bus = pipeline.get_bus()
        bus.add_signal_watch()

        def on_message(_bus, message):
            t = message.type
            if t == Gst.MessageType.EOS:
                loop.quit()
            elif t == Gst.MessageType.ERROR:
                err, dbg = message.parse_error()
                print(f"[DeepStream ERROR] {err}: {dbg}")
                loop.quit()

        bus.connect("message", on_message)
        pipeline.set_state(Gst.State.PLAYING)

        # Drain the probe queue while the pipeline runs, in another thread.
        import threading

        thread = threading.Thread(target=loop.run, daemon=True)
        thread.start()

        emitted = 0
        try:
            while thread.is_alive() or self._queue:
                if not self._queue:
                    thread.join(timeout=0.01)
                    continue
                item = self._queue.pop(0)
                yield item
                emitted += 1
                if self.max_frames is not None and emitted >= self.max_frames:
                    break
        finally:
            pipeline.set_state(Gst.State.NULL)
