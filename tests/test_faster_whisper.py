"""Tests for the faster-whisper backend.

Normalization and device selection are the parts worth testing: the first is
required by the Testing Policy in `conductor/workflow.md`, the second turns a
mid-batch crash into an up-front error.

These use synthetic engine output rather than real inference, so they run in
milliseconds. The live engine is covered by the conformance suite.
"""

from dataclasses import dataclass

import pytest

pytest.importorskip(
    "faster_whisper", reason="the faster-whisper extra is not installed"
)

from audio_transcriber.backends.faster_whisper import (  # noqa: E402
    FasterWhisperBackend,
)
from audio_transcriber.errors import BackendRequirementError  # noqa: E402


@dataclass
class _NativeSegment:
    """Stands in for one segment as faster-whisper yields it."""

    start: float
    end: float
    text: str


@pytest.fixture
def backend() -> FasterWhisperBackend:
    return FasterWhisperBackend(model="tiny")


def test_normalize_maps_native_segments_onto_the_schema(backend):
    segments = backend._normalize(
        [_NativeSegment(0.0, 1.5, " Hello there. "), _NativeSegment(1.5, 3.0, "Bye.")],
        "en",
    )

    assert [s.to_dict() for s in segments] == [
        {
            "start_time": 0.0,
            "end_time": 1.5,
            "speaker": None,
            "spoken_language": "en",
            "text": "Hello there.",
        },
        {
            "start_time": 1.5,
            "end_time": 3.0,
            "speaker": None,
            "spoken_language": "en",
            "text": "Bye.",
        },
    ]


def test_normalize_drops_empty_segments(backend):
    """Whisper emits blank segments; they are noise, not content."""
    segments = backend._normalize(
        [
            _NativeSegment(0.0, 1.0, "   "),
            _NativeSegment(1.0, 2.0, "real text"),
            _NativeSegment(2.0, 3.0, ""),
        ],
        "en",
    )

    assert [s.text for s in segments] == ["real text"]


def test_normalize_clamps_overlapping_segments(backend):
    """The schema forbids overlap, so a few milliseconds of it is corrected here."""
    segments = backend._normalize(
        [_NativeSegment(0.0, 2.0, "first"), _NativeSegment(1.95, 3.0, "second")],
        "en",
    )

    assert segments[1].start_time == 2.0
    assert segments[0].end_time == 2.0


def test_normalize_fixes_a_segment_ending_before_it_starts(backend):
    segments = backend._normalize([_NativeSegment(5.0, 4.0, "backwards")], "en")

    assert segments[0].end_time >= segments[0].start_time


def test_normalize_leaves_speaker_unset(backend):
    """This backend does not diarize, so speaker stays null."""
    segments = backend._normalize([_NativeSegment(0.0, 1.0, "hello")], "en")

    assert segments[0].speaker is None
    assert backend.capabilities.diarization is False


def test_normalize_carries_the_detected_language_onto_every_segment(backend):
    segments = backend._normalize(
        [_NativeSegment(0.0, 1.0, "hola"), _NativeSegment(1.0, 2.0, "adios")],
        "es",
    )

    assert [s.spoken_language for s in segments] == ["es", "es"]


def test_normalize_handles_no_segments(backend):
    """Silence yields an empty transcript, which is a success."""
    assert backend._normalize([], "en") == ()


def test_model_version_pins_both_library_and_model(backend):
    version = backend.model_version()

    assert version.startswith("faster-whisper-")
    assert version.endswith("/tiny")
    assert "latest" not in version


def test_unknown_model_is_rejected_with_the_valid_names():
    backend = FasterWhisperBackend(model="enormous")

    with pytest.raises(BackendRequirementError, match="unknown model 'enormous'") as e:
        backend.check_requirements()

    assert "tiny" in str(e.value)


def test_defaults_are_recorded_in_the_effective_config():
    """Nothing is left implicit in a flag default; the manifest gets real values."""
    config = FasterWhisperBackend().effective_config()

    assert config["model"] == "base"
    assert config["beam_size"] == 5
    assert config["vad_filter"] is True


def test_explicit_config_overrides_defaults():
    config = FasterWhisperBackend(model="small", beam_size=1).effective_config()

    assert config["model"] == "small"
    assert config["beam_size"] == 1


def test_auto_device_tries_cuda_first_when_one_is_present(backend, monkeypatch):
    monkeypatch.setattr(
        "audio_transcriber.backends.faster_whisper.ctranslate2.get_cuda_device_count",
        lambda: 1,
    )

    assert backend._candidate_devices() == ("cuda", "cpu")


def test_auto_device_skips_cuda_when_none_is_present(backend, monkeypatch):
    monkeypatch.setattr(
        "audio_transcriber.backends.faster_whisper.ctranslate2.get_cuda_device_count",
        lambda: 0,
    )

    assert backend._candidate_devices() == ("cpu",)


def test_explicit_device_is_not_second_guessed(monkeypatch):
    """An explicit device failing is an error, not a silent fallback to CPU."""
    monkeypatch.setattr(
        "audio_transcriber.backends.faster_whisper.ctranslate2.get_cuda_device_count",
        lambda: 1,
    )

    assert FasterWhisperBackend(device="cuda")._candidate_devices() == ("cuda",)


def test_device_failure_reports_every_attempt_and_a_remedy(backend, monkeypatch):
    def _explode(*args, **kwargs):
        raise RuntimeError("Library libcublas.so.12 is not found")

    monkeypatch.setattr(
        "audio_transcriber.backends.faster_whisper.ctranslate2.get_cuda_device_count",
        lambda: 1,
    )
    monkeypatch.setattr(
        "audio_transcriber.backends.faster_whisper.WhisperModel", _explode
    )

    with pytest.raises(BackendRequirementError) as caught:
        backend.check_requirements()

    message = str(caught.value)
    assert "cuda: " in message and "cpu: " in message
    assert "set device=cpu" in message


def test_resolved_device_is_recorded_for_reproducibility(backend, monkeypatch):
    """A run that fell back to CPU must be reproducible as a CPU run."""
    monkeypatch.setattr(
        "audio_transcriber.backends.faster_whisper.ctranslate2.get_cuda_device_count",
        lambda: 0,
    )
    monkeypatch.setattr(
        "audio_transcriber.backends.faster_whisper.WhisperModel",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(FasterWhisperBackend, "_smoke_test", lambda self, model: None)

    backend.check_requirements()

    assert backend.effective_config()["device"] == "cpu"
