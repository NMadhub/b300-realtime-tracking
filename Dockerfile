# Multi-object tracker benchmark on top of the DeepStream NGC container.
#
# The DeepStream image already provides: CUDA, TensorRT, GStreamer + the nvidia
# plugins (nvstreammux, nvinfer, nvvideoconvert, ...) and the `pyds` bindings.
# We only layer the Python tracking/detection deps on top.
#
# Pick the tag that matches your driver/GPU; triton variant includes pyds.
#   https://catalog.ngc.nvidia.com/orgs/nvidia/teams/deepstream/containers/deepstream
# 9.0 is the current release (Mar 2026) and the first to certify Blackwell/B300.
# Older tags (e.g. 7.1) reject the B300 with "No supported GPU(s) detected".
ARG DS_IMAGE=nvcr.io/nvidia/deepstream:9.0-triton-multiarch
FROM ${DS_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace/mot-benchmark

# DeepStream 9.0 does NOT bundle the `pyds` Python bindings, and drops some
# codecs (needed for MP4 demux). Install both:
#   1. codecs via the SDK helper script
#   2. the matching prebuilt pyds wheel (cp312 / x86_64 for DS 9.0 on U24.04)
ARG PYDS_WHL=https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.2/pyds-1.2.2-cp312-cp312-linux_x86_64.whl
RUN if [ -f /opt/nvidia/deepstream/deepstream/user_additional_install.sh ]; then \
        /opt/nvidia/deepstream/deepstream/user_additional_install.sh || true; \
    fi && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3-gi python3-gst-1.0 gir1.2-gst-rtsp-server-1.0 curl && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fsSL "$PYDS_WHL" -O --output-dir /tmp && \
    python3 -m pip install --no-cache-dir --break-system-packages /tmp/pyds-*.whl && \
    rm -f /tmp/pyds-*.whl

COPY requirements.txt .
# DeepStream 9.0 is Ubuntu 24.04: pip is Debian-managed (PEP 668), so we install
# into the system Python (where pyds lives) with --break-system-packages rather
# than trying to upgrade/replace the distro pip.
RUN python3 -m pip install --no-cache-dir --break-system-packages \
    -r requirements.txt

COPY . .

# Default: show CLI help. Override with your own command at `docker run`.
CMD ["python3", "-m", "src.benchmark", "--help"]
