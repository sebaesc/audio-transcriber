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


def test_cuda_preload_is_idempotent_and_never_raises(monkeypatch):
    """Preloading runs before every CUDA attempt, and a missing library is not fatal.

    The device smoke test decides whether CUDA works; preloading only makes the
    pip-installed libraries findable.
    """
    from audio_transcriber.backends import faster_whisper as module

    monkeypatch.setattr(module, "_cuda_preloaded", False)
    monkeypatch.setattr(module.site, "getsitepackages", lambda: ["/nonexistent"])

    module._preload_cuda_libraries()
    module._preload_cuda_libraries()


def test_cuda_preload_skips_work_once_done(monkeypatch):
    from audio_transcriber.backends import faster_whisper as module

    calls = []
    monkeypatch.setattr(module, "_cuda_preloaded", True)
    monkeypatch.setattr(module.site, "getsitepackages", lambda: calls.append(1) or [])

    module._preload_cuda_libraries()

    assert calls == []


def test_normalize_drops_hallucinations_past_the_end_of_audio(backend):
    """Whisper pads short audio to 30s and hallucinates into the silence.

    Observed with large-v3 on a 10.4s recording: 18 segments running to 40.38s,
    including text in another language. Anything starting past the end of the
    audio is not a transcription of it.
    """
    segments = backend._normalize(
        [
            _NativeSegment(0.0, 10.4, "real speech"),
            _NativeSegment(12.0, 14.0, "hallucinated"),
            _NativeSegment(38.0, 40.4, "그리고 camembert cheese"),
        ],
        "en",
        duration=10.44,
    )

    assert [s.text for s in segments] == ["real speech"]


def test_normalize_clamps_a_segment_overhanging_the_end(backend):
    segments = backend._normalize(
        [_NativeSegment(9.0, 30.0, "trails off")], "en", duration=10.44
    )

    assert segments[0].end_time == 10.44


def test_normalize_drops_a_segment_starting_at_or_after_the_end(backend):
    segments = backend._normalize(
        [_NativeSegment(10.44, 30.0, "noise")], "en", duration=10.435
    )

    assert segments == ()


def test_normalize_keeps_a_segment_starting_just_inside_the_audio(backend):
    """Clamped to the end rather than dropped: it began while audio remained."""
    segments = backend._normalize(
        [_NativeSegment(10.40, 30.0, "trailing word")], "en", duration=10.435
    )

    assert len(segments) == 1
    assert segments[0].end_time == 10.435


def test_normalize_without_a_duration_keeps_everything(backend):
    """A backend that cannot report duration must not lose real segments."""
    segments = backend._normalize(
        [_NativeSegment(0.0, 1.0, "a"), _NativeSegment(40.0, 41.0, "b")], "en"
    )

    assert len(segments) == 2


# --- Reproducibility ---
#
# "The same input plus the same config produces the same output" is a product
# principle, and it did not hold: transcribing one recording twice produced 689
# vs 642 segments and a WER of 0.409 vs 0.441. faster-whisper defaults
# temperature to [0.0, 0.2, ..., 1.0] and falls back up that ladder when a decode
# trips its quality thresholds. Above 0.0 it samples, so the output is not
# reproducible — and condition_on_previous_text then carries the divergence into
# every later segment.


def test_temperature_defaults_to_deterministic_decoding():
    """A single temperature of 0.0 means greedy decoding, not sampling."""
    config = FasterWhisperBackend().effective_config()

    assert config["temperature"] == 0.0


def test_decoding_options_are_recorded_in_the_effective_config():
    """A run is only reproducible if the manifest says what produced it."""
    config = FasterWhisperBackend().effective_config()

    for key in ("temperature", "condition_on_previous_text", "beam_size"):
        assert key in config


def test_temperature_is_forwarded_to_the_engine(backend, monkeypatch, tmp_path):
    """Recording the value is not enough; it has to reach the decoder."""
    seen = {}

    class _Model:
        def transcribe(self, audio, **kwargs):
            seen.update(kwargs)

            class _Info:
                language = "en"
                duration = 1.0

            return iter(()), _Info()

    monkeypatch.setattr(backend, "_model", _Model())
    backend.transcribe(tmp_path / "a.wav", "a.wav")

    assert seen["temperature"] == 0.0
    assert seen["condition_on_previous_text"] is True


def test_the_fallback_ladder_can_be_restored_by_config():
    """Determinism is the default, not a cage: the ladder stays available."""
    ladder = [0.0, 0.2, 0.4]
    backend = FasterWhisperBackend(temperature=ladder)

    assert backend.effective_config()["temperature"] == ladder


# --- Config that reaches the decoder ---
#
# The backend accepted any keyword, stored it, and recorded it in the manifest,
# while transcribe() forwarded only the handful it knew about. A run configured
# with no_repeat_ngram_size=3 produced byte-identical output to one without it,
# and the manifest claimed the option had been used. A manifest that records
# options which were never applied is worse than one that records nothing.


def test_an_unknown_option_is_rejected():
    """Silently ignoring it would put a false claim in the manifest."""
    with pytest.raises(BackendRequirementError, match="unknown option"):
        FasterWhisperBackend(no_such_option=1)


def test_the_rejection_lists_the_options_that_do_exist():
    with pytest.raises(BackendRequirementError) as caught:
        FasterWhisperBackend(beem_size=5)

    assert "beam_size" in str(caught.value)


def test_every_recorded_option_reaches_the_decoder(monkeypatch, tmp_path):
    """Whatever the manifest records must actually have been applied."""
    seen = {}

    class _Model:
        def transcribe(self, audio, **kwargs):
            seen.update(kwargs)

            class _Info:
                language = "en"
                duration = 1.0

            return iter(()), _Info()

    backend = FasterWhisperBackend(model="tiny")
    monkeypatch.setattr(backend, "_model", _Model())
    backend.transcribe(tmp_path / "a.wav", "a.wav")

    recorded = set(backend.effective_config())
    # Options describing how the model is loaded rather than how it decodes.
    load_time = {"model", "device", "compute_type"}
    assert (recorded - load_time) <= set(seen), (
        f"recorded but never forwarded: {sorted((recorded - load_time) - set(seen))}"
    )


def test_repetition_controls_are_forwarded(monkeypatch, tmp_path):
    """These are the deterministic alternative to temperature sampling."""
    seen = {}

    class _Model:
        def transcribe(self, audio, **kwargs):
            seen.update(kwargs)

            class _Info:
                language = "en"
                duration = 1.0

            return iter(()), _Info()

    backend = FasterWhisperBackend(repetition_penalty=1.1, no_repeat_ngram_size=3)
    monkeypatch.setattr(backend, "_model", _Model())
    backend.transcribe(tmp_path / "a.wav", "a.wav")

    assert seen["repetition_penalty"] == 1.1
    assert seen["no_repeat_ngram_size"] == 3
