"""A stub backend and stub decoder for the API tests.

The API tests drive the real routes, runner and pipeline, but never a real
engine or ffmpeg: no test may build a hosted backend, so a `.env` holding a
real API key can never lead to a paid call.

The stub's behavior is chosen by its options, exactly as a client would choose
a real backend's:

- `fail`: `"transcription"` raises `TranscriptionError` while transcribing,
  `"requirement"` raises `BackendRequirementError` from `check_requirements`.
- `wait`: true blocks transcription until `StubBackend.gate` is set, so a test
  can observe a job while it is running.

A file whose bytes start with `not audio` fails to decode.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from audio_transcriber import pipeline, registry
from audio_transcriber.backend import TranscriptionBackend
from audio_transcriber.errors import (
    AudioDecodeError,
    BackendRequirementError,
    TranscriptionError,
)
from audio_transcriber.schema import Segment, Transcript

STUB_NAME = "stub"
UNDECODABLE = b"not audio"

_OPTIONS = {"fail": None, "wait": False}


class StubBackend(TranscriptionBackend):
    """Transcribes any audio as one English segment, `hello world`."""

    name = STUB_NAME

    #: Sources transcribed, across every instance, in order.
    calls: list[str] = []
    #: Holds `wait` transcriptions until set.
    gate = threading.Event()

    def __init__(self, **config: Any) -> None:
        unknown = sorted(set(config) - set(_OPTIONS))
        if unknown:
            raise BackendRequirementError(
                self.name,
                f"unknown option(s): {', '.join(unknown)}.",
                f"Valid options are: {', '.join(sorted(_OPTIONS))}.",
            )
        super().__init__(**{**_OPTIONS, **config})

    def check_requirements(self) -> None:
        if self.config["fail"] == "requirement":
            raise BackendRequirementError(
                self.name, "STUB_API_KEY is not set.", "Set STUB_API_KEY in .env."
            )

    def model_version(self) -> str:
        return "stub-1.0"

    def transcribe(self, audio_path: Path, source: str) -> Transcript:
        if self.config["wait"]:
            assert StubBackend.gate.wait(timeout=10), "test never opened the gate"
        StubBackend.calls.append(source)
        if self.config["fail"] == "transcription":
            raise TranscriptionError(source, self.name, "the engine said no.")
        return Transcript(
            source=source,
            segments=(
                Segment(
                    start_time=0.0,
                    end_time=1.0,
                    text="hello world",
                    speaker="SPEAKER_00",
                    spoken_language="en",
                ),
            ),
            language="en",
            duration=1.0,
            backend=self.name,
            model_version=self.model_version(),
        )


def install(monkeypatch) -> type[StubBackend]:
    """Make the stub the only backend, and stub out ffmpeg.

    Built-ins are hidden, so nothing a test does can reach a real engine.

    Returns:
        The stub class, with its call log cleared and its gate closed.
    """
    monkeypatch.setattr(registry, "_REGISTRY", {STUB_NAME: StubBackend})
    monkeypatch.setattr(registry, "_BUILTINS", ())
    monkeypatch.setattr(
        registry,
        "_unavailable",
        {
            "missing": "not installed: it needs 'x'. Install its extra: pip install -e \".[x]\""
        },
    )
    monkeypatch.setattr(registry, "_builtins_loaded", True)
    monkeypatch.setattr(pipeline.audio, "check_ffmpeg", lambda: None)

    @contextmanager
    def _fake_decoded(source: Path):
        if source.read_bytes().startswith(UNDECODABLE):
            raise AudioDecodeError(str(source), "ffmpeg could not decode it.")
        yield source

    monkeypatch.setattr(pipeline.audio, "decoded", _fake_decoded)
    StubBackend.calls = []
    StubBackend.gate = threading.Event()
    return StubBackend
