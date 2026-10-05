# The HTTP API service: the pipeline, its backends and ffmpeg, behind
# `audio-transcriber serve`. CPU only; local models run on the CPU here.
#
#   docker build -t audio-transcriber .
#   docker build --build-arg EXTRAS=api,eval,elevenlabs -t audio-transcriber .
#
# EXTRAS picks the optional extras installed. `api` is required to serve;
# `eval` scores jobs against a ground truth; each backend is its own extra.

FROM python:3.12-slim

ARG EXTRAS=api,eval,faster-whisper,elevenlabs

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Decoding is ffmpeg as a subprocess, as on a workstation.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 app

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[${EXTRAS}]"

USER app
# Model downloads land in HF_HOME, which compose mounts as a volume so they
# survive the container. Jobs' uploads and runs live in the work directory.
ENV HF_HOME=/home/app/.cache/huggingface \
    AUDIO_TRANSCRIBER_WORK_DIR=/home/app/work
# Created as `app`, so a fresh named volume mounted here is writable by it.
RUN mkdir -p "$HF_HOME" "$AUDIO_TRANSCRIBER_WORK_DIR"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

CMD ["audio-transcriber", "serve", "--host", "0.0.0.0"]
