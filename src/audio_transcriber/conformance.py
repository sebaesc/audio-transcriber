"""Shared conformance checks every backend must satisfy.

These assertions are the executable half of the backend contract: whatever an
engine returns natively, what reaches the pipeline must pass every check here.
They live in the package rather than in `tests/` so a backend developed outside
this repo can run the same suite.

`tests/test_conformance.py` applies these to every registered backend.
"""

from __future__ import annotations

from pathlib import Path

from audio_transcriber.backend import TranscriptionBackend
from audio_transcriber.schema import SCHEMA_VERSION, Segment, Transcript


class ConformanceError(AssertionError):
    """Raised when a backend's output violates the shared contract."""


def _fail(backend: str, message: str) -> None:
    raise ConformanceError(f"{backend}: {message}")


def assert_segment_conforms(segment: Segment, backend: str, index: int) -> None:
    """Check one segment against the shared schema's guarantees.

    Args:
        segment: The segment to check.
        backend: Name of the backend that produced it, for the error message.
        index: Position of the segment, for the error message.

    Raises:
        ConformanceError: If the segment violates the contract.
    """
    where = f"segment {index}"
    if not isinstance(segment, Segment):
        _fail(backend, f"{where} is {type(segment).__name__}, not a Segment")
    if segment.start < 0:
        _fail(backend, f"{where} starts at {segment.start}, before zero")
    if segment.end < segment.start:
        _fail(backend, f"{where} ends at {segment.end}, before its start")
    if segment.text != segment.text.strip():
        _fail(backend, f"{where} text is not stripped: {segment.text!r}")
    if segment.confidence is not None and not 0.0 <= segment.confidence <= 1.0:
        _fail(backend, f"{where} confidence {segment.confidence} is outside 0.0-1.0")


def assert_transcript_conforms(
    transcript: Transcript, backend: str, source: str
) -> None:
    """Check a transcript against the shared schema's guarantees.

    Args:
        transcript: The transcript a backend returned.
        backend: The name the backend is registered under.
        source: The source path the backend was handed.

    Raises:
        ConformanceError: If the transcript violates the contract.
    """
    if not isinstance(transcript, Transcript):
        _fail(
            backend,
            f"transcribe returned {type(transcript).__name__}, not a Transcript. "
            "Backends normalize their own output.",
        )
    if transcript.source != source:
        _fail(
            backend,
            f"transcript source is {transcript.source!r}, expected {source!r}. "
            "Record the original input path, not the decoded temporary file.",
        )
    if transcript.backend != backend:
        _fail(
            backend,
            f"transcript backend is {transcript.backend!r}, expected {backend!r}",
        )
    if not transcript.model_version:
        _fail(backend, "transcript has no model_version; the manifest needs it")
    if transcript.schema_version != SCHEMA_VERSION:
        _fail(
            backend,
            f"transcript schema_version is {transcript.schema_version!r}, "
            f"expected {SCHEMA_VERSION!r}",
        )

    previous_end = 0.0
    for index, segment in enumerate(transcript.segments):
        assert_segment_conforms(segment, backend, index)
        if segment.start < previous_end:
            _fail(
                backend,
                f"segment {index} starts at {segment.start}, overlapping the "
                f"previous segment which ended at {previous_end}",
            )
        previous_end = segment.end

    if transcript.duration is not None and previous_end > transcript.duration + 0.5:
        _fail(
            backend,
            f"segments run to {previous_end}s, past the reported "
            f"duration of {transcript.duration}s",
        )


def assert_serializable(transcript: Transcript, backend: str) -> None:
    """Check a transcript survives a round trip through JSON data.

    Raises:
        ConformanceError: If the transcript does not round-trip unchanged.
    """
    restored = Transcript.from_dict(transcript.to_dict())
    if restored != transcript:
        _fail(backend, "transcript does not survive a round trip through JSON")


def check_backend(
    backend: TranscriptionBackend, audio_path: Path, source: str
) -> Transcript:
    """Run a backend over one file and check everything it must guarantee.

    Args:
        backend: An instantiated backend whose requirements are already met.
        audio_path: Decoded mono 16 kHz WAV to transcribe.
        source: The original input path to hand the backend.

    Returns:
        The transcript, so a caller can make further engine-specific assertions.

    Raises:
        ConformanceError: If the backend violates the contract.
    """
    if not backend.name:
        raise ConformanceError(
            f"{type(backend).__name__} has no name and cannot be selected"
        )
    if not backend.model_version():
        _fail(backend.name, "model_version() is empty; the manifest needs it")

    transcript = backend.transcribe(audio_path, source)
    assert_transcript_conforms(transcript, backend.name, source)
    assert_serializable(transcript, backend.name)
    return transcript
