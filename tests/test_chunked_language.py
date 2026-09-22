"""Tests for per-chunk language detection.

The engine-facing parts are stubbed: what these cover is the logic that decides
a language and the arithmetic that puts chunk-relative timings back onto the
recording's timeline. A silent off-by-one there would shift every timestamp in
a bilingual transcript without failing anything.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip(
    "faster_whisper", reason="the faster-whisper extra is not installed"
)

from audio_transcriber.backends.chunking import SAMPLE_RATE  # noqa: E402
from audio_transcriber.backends.faster_whisper import (  # noqa: E402
    FasterWhisperBackend,
)
from audio_transcriber.errors import BackendRequirementError  # noqa: E402


class _NativeSegment:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text


class _Model:
    """Stands in for the engine: fixed language probabilities and segments."""

    def __init__(self, probabilities, segments_by_language=None):
        self.probabilities = probabilities
        self.segments_by_language = segments_by_language or {}
        self.transcribe_calls = []

    def detect_language(self, audio=None):
        best = max(self.probabilities, key=lambda item: item[1])
        return best[0], best[1], self.probabilities

    def transcribe(self, audio, **kwargs):
        self.transcribe_calls.append(kwargs)
        language = kwargs.get("language")
        segments = self.segments_by_language.get(
            language, [_NativeSegment(0.0, 1.0, f"text-{language}")]
        )

        class _Info:
            pass

        return iter(segments), _Info()


def _backend(**config) -> FasterWhisperBackend:
    backend = FasterWhisperBackend(
        model="tiny", language_detection="chunked", languages=["en", "es"], **config
    )
    backend._resolved_device = "cpu"
    return backend


def _patch_audio(monkeypatch, seconds: float, chunk_bounds):
    """Give the backend fixed audio and fixed chunk boundaries."""
    samples = np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)
    monkeypatch.setattr(
        "audio_transcriber.backends.faster_whisper.decode_audio",
        lambda *a, **k: samples,
    )

    class _Chunk:
        def __init__(self, start, end):
            self.start, self.end = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)

        @property
        def start_time(self):
            return self.start / SAMPLE_RATE

        @property
        def duration(self):
            return (self.end - self.start) / SAMPLE_RATE

    monkeypatch.setattr(
        "audio_transcriber.backends.faster_whisper.speech_chunks",
        lambda *a, **k: [_Chunk(a, b) for a, b in chunk_bounds],
    )
    return samples


def test_detection_is_restricted_to_the_allowed_languages():
    """Portuguese outranking Spanish on Spanish audio must not win."""
    backend = _backend()
    model = _Model([("pt", 0.7), ("es", 0.25), ("en", 0.05)])

    language, probability = backend._detect_language(model, None)

    assert language == "es"
    assert probability == 0.25


def test_detection_without_restriction_takes_the_best_overall():
    backend = FasterWhisperBackend(model="tiny", languages=None)
    model = _Model([("pt", 0.7), ("es", 0.25)])

    assert backend._detect_language(model, None)[0] == "pt"


def test_languages_may_be_given_as_a_comma_separated_string():
    """The CLI passes --option languages=en,es as text."""
    backend = FasterWhisperBackend(model="tiny", languages="en, es")

    assert backend._allowed_languages() == ("en", "es")


def test_no_allowed_language_ranked_reports_nothing():
    backend = FasterWhisperBackend(model="tiny", languages=["de"])
    model = _Model([("pt", 0.7), ("es", 0.25)])

    assert backend._detect_language(model, None) == (None, 0.0)


def test_chunk_times_are_shifted_onto_the_recording_timeline(monkeypatch):
    """A segment 1s into the third chunk is not 1s into the recording."""
    backend = _backend()
    _patch_audio(monkeypatch, 90.0, [(0, 25), (25, 50), (50, 75)])
    model = _Model(
        [("en", 0.9), ("es", 0.1)],
        {"en": [_NativeSegment(0.0, 2.0, "hello")]},
    )

    transcript = backend._transcribe_chunked(model, Path("a.wav"), "a.wav")

    starts = [s.start_time for s in transcript.segments]
    assert starts == [0.0, 25.0, 50.0]


def test_segments_stay_ordered_and_non_overlapping(monkeypatch):
    backend = _backend()
    _patch_audio(monkeypatch, 60.0, [(0, 25), (20, 45)])
    model = _Model([("en", 0.9)], {"en": [_NativeSegment(0.0, 10.0, "overlapping")]})

    transcript = backend._transcribe_chunked(model, Path("a.wav"), "a.wav")

    for earlier, later in zip(
        transcript.segments, transcript.segments[1:], strict=False
    ):
        assert later.start_time >= earlier.end_time


def test_segments_never_run_past_the_recording(monkeypatch):
    backend = _backend()
    _patch_audio(monkeypatch, 30.0, [(0, 25)])
    model = _Model([("en", 0.9)], {"en": [_NativeSegment(0.0, 60.0, "too long")]})

    transcript = backend._transcribe_chunked(model, Path("a.wav"), "a.wav")

    assert max(s.end_time for s in transcript.segments) <= 30.0


def test_a_low_confidence_chunk_falls_back_to_the_file_language(monkeypatch):
    """A guess the model is unsure of is worse than the file-level answer."""
    backend = _backend(language_threshold=0.8)
    _patch_audio(monkeypatch, 30.0, [(0, 25)])
    model = _Model([("en", 0.55), ("es", 0.45)])

    backend._transcribe_chunked(model, Path("a.wav"), "a.wav")

    assert model.transcribe_calls[0]["language"] == "en"


def test_a_confident_chunk_uses_its_own_language(monkeypatch):
    backend = _backend(language_threshold=0.5)
    _patch_audio(monkeypatch, 30.0, [(0, 25)])
    model = _Model([("es", 0.92), ("en", 0.08)])

    backend._transcribe_chunked(model, Path("a.wav"), "a.wav")

    assert model.transcribe_calls[0]["language"] == "es"


def test_vad_is_not_run_again_inside_a_chunk(monkeypatch):
    """The chunk is already speech; re-running VAD only trims its edges."""
    backend = _backend()
    _patch_audio(monkeypatch, 30.0, [(0, 25)])
    model = _Model([("en", 0.9)])

    backend._transcribe_chunked(model, Path("a.wav"), "a.wav")

    assert model.transcribe_calls[0]["vad_filter"] is False


def test_the_transcript_records_the_whole_recording_duration(monkeypatch):
    backend = _backend()
    _patch_audio(monkeypatch, 90.0, [(0, 25)])
    model = _Model([("en", 0.9)])

    transcript = backend._transcribe_chunked(model, Path("a.wav"), "a.wav")

    assert transcript.duration == pytest.approx(90.0)


def test_the_original_source_path_is_preserved(monkeypatch):
    backend = _backend()
    _patch_audio(monkeypatch, 30.0, [(0, 25)])
    model = _Model([("en", 0.9)])

    transcript = backend._transcribe_chunked(
        model, Path("/tmp/decoded.wav"), "audio/meeting.m4a"
    )

    assert transcript.source == "audio/meeting.m4a"


def test_chunked_mode_is_recorded_in_the_effective_config():
    config = _backend().effective_config()

    assert config["language_detection"] == "chunked"
    assert config["languages"] == ["en", "es"]
    assert "language_threshold" in config and "chunk_seconds" in config


def test_file_detection_remains_the_default():
    """The shipped behaviour does not change unless asked."""
    assert FasterWhisperBackend().effective_config()["language_detection"] == "file"


def test_pipeline_options_are_not_passed_to_the_decoder(monkeypatch):
    """language_detection is ours; transcribe() would reject it."""
    backend = _backend()
    _patch_audio(monkeypatch, 30.0, [(0, 25)])
    model = _Model([("en", 0.9)])

    backend._transcribe_chunked(model, Path("a.wav"), "a.wav")

    for key in (
        "language_detection",
        "languages",
        "language_threshold",
        "chunk_seconds",
    ):
        assert key not in model.transcribe_calls[0]


def test_an_unknown_option_is_still_rejected():
    with pytest.raises(BackendRequirementError, match="unknown option"):
        FasterWhisperBackend(langauges=["en"])
