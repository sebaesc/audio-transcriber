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


class _Chunk:
    def __init__(self, start, end):
        self.start, self.end = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)

    @property
    def start_time(self):
        return self.start / SAMPLE_RATE

    @property
    def duration(self):
        return (self.end - self.start) / SAMPLE_RATE


def _chunks(bounds):
    return [_Chunk(a, b) for a, b in bounds]


def _patch_audio(monkeypatch, seconds: float, chunk_bounds):
    """Give the backend fixed audio and fixed chunk boundaries."""
    samples = np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)
    monkeypatch.setattr(
        "audio_transcriber.backends.faster_whisper.decode_audio",
        lambda *a, **k: samples,
    )

    monkeypatch.setattr(
        "audio_transcriber.backends.faster_whisper.speech_chunks",
        lambda *a, **k: _chunks(chunk_bounds),
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
    backend = FasterWhisperBackend(
        model="tiny", languages="en, es", language_detection="chunked"
    )

    assert backend._allowed_languages() == ("en", "es")


def test_no_allowed_language_ranked_reports_nothing():
    backend = FasterWhisperBackend(
        model="tiny", languages=["de"], language_detection="chunked"
    )
    model = _Model([("pt", 0.7), ("es", 0.25)])

    assert backend._detect_language(model, None) == (None, 0.0)


def test_span_times_are_shifted_onto_the_recording_timeline(monkeypatch):
    """A segment 1s into the third span is not 1s into the recording.

    Spans are used rather than chunks because same-language chunks are merged;
    alternating the detected language keeps them separate so the shift is
    observable.
    """
    backend = _backend()
    _patch_audio(monkeypatch, 90.0, [(0, 25), (25, 50), (50, 75)])

    class _Alternating(_Model):
        def __init__(self):
            super().__init__([], {None: [_NativeSegment(0.0, 2.0, "hello")]})
            self.calls = 0

        def detect_language(self, audio=None):
            self.calls += 1
            code = "en" if self.calls % 2 else "es"
            probs = [(code, 0.95)]
            return code, 0.95, probs

        def transcribe(self, audio, **kwargs):
            self.transcribe_calls.append(kwargs)

            class _Info:
                pass

            return iter([_NativeSegment(0.0, 2.0, "hello")]), _Info()

    transcript = backend._transcribe_chunked(_Alternating(), Path("a.wav"), "a.wav")

    assert [s.start_time for s in transcript.segments] == [0.0, 25.0, 50.0]


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


def test_adjacent_chunks_of_one_language_become_a_single_span(monkeypatch):
    """Every decode boundary costs content, so agreeing neighbours are merged."""
    backend = _backend()
    samples = _patch_audio(monkeypatch, 75.0, [(0, 25), (25, 50), (50, 75)])
    model = _Model([("en", 0.9), ("es", 0.1)])

    spans = backend._language_spans(
        model, samples, _chunks([(0, 25), (25, 50), (50, 75)]), "en", 0.5
    )

    assert spans == [(0, 75 * SAMPLE_RATE, "en")]


def test_a_language_change_starts_a_new_span(monkeypatch):
    backend = _backend()
    samples = np.zeros(75 * SAMPLE_RATE, dtype=np.float32)

    class _Switching(_Model):
        def __init__(self):
            super().__init__([])
            self.calls = 0

        def detect_language(self, audio=None):
            self.calls += 1
            probs = (
                [("en", 0.9), ("es", 0.1)]
                if self.calls == 1
                else [("es", 0.9), ("en", 0.1)]
            )
            return probs[0][0], probs[0][1], probs

    spans = backend._language_spans(
        _Switching(), samples, _chunks([(0, 25), (25, 50)]), "en", 0.5
    )

    assert len(spans) == 2
    assert [s[2] for s in spans] == ["en", "es"]


def test_merging_reduces_the_number_of_decodes(monkeypatch):
    """Fewer decodes is the point: it is why coverage recovers."""
    backend = _backend()
    _patch_audio(monkeypatch, 75.0, [(0, 25), (25, 50), (50, 75)])
    model = _Model([("en", 0.9), ("es", 0.1)])

    backend._transcribe_chunked(model, Path("a.wav"), "a.wav")

    assert len(model.transcribe_calls) == 1


def test_a_segment_squashed_onto_the_span_end_is_dropped(monkeypatch):
    """Each span is normalized against its own length, so the clamp that
    squashes a hallucination runs per span — and so must the density check."""
    _patch_audio(monkeypatch, 10.0, [(0.0, 9.0), (9.99, 10.0)])
    backend = _backend()

    class _Squashing:
        def detect_language(self, audio):
            return "en", 1.0, [("en", 1.0)]

        def transcribe(self, audio, **kwargs):
            # Relative to its span: starts 5 ms before the span ends.
            return iter(
                [
                    _NativeSegment(0.0, 5.0, "a real sentence here"),
                    _NativeSegment(
                        9.995, 12.0, "For more info, see the links in the description."
                    ),
                ]
            ), None

    transcript = backend._transcribe_chunked(_Squashing(), Path("a.wav"), "a.wav")

    assert [s.text for s in transcript.segments] == ["a real sentence here"]


# --- Options that contradict the detection mode ---
#
# Both of these were recorded in the manifest and silently ignored: chunked mode
# overwrote a forced `language` with whatever it detected, and file mode never
# read the chunked-only options at all.


def test_a_forced_language_is_rejected_in_chunked_mode():
    with pytest.raises(BackendRequirementError, match="language"):
        FasterWhisperBackend(language="es", language_detection="chunked")


@pytest.mark.parametrize(
    "option",
    [
        {"languages": ["en", "es"]},
        {"language_threshold": 0.7},
        {"chunk_seconds": 20.0},
    ],
)
def test_chunked_only_options_are_rejected_in_file_mode(option):
    with pytest.raises(BackendRequirementError, match="language_detection=chunked"):
        FasterWhisperBackend(**option)


def test_a_forced_language_is_accepted_in_file_mode():
    assert FasterWhisperBackend(language="es").effective_config()["language"] == "es"


def test_chunks_default_to_fifteen_seconds():
    """Chosen from the eval: better than 25 s on 4 of 5 recordings."""
    config = FasterWhisperBackend(language_detection="chunked").effective_config()

    assert config["chunk_seconds"] == 15.0
