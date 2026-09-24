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


class _Weights:
    """Stands in for the CTranslate2 model a WhisperModel wraps."""

    def unload_model(self):
        pass


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

    assert config["model"] == "large-v3"
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


def test_decoding_options_are_recorded_in_the_effective_config():
    """A run is only reproducible if the manifest says what produced it."""
    config = FasterWhisperBackend().effective_config()

    for key in ("temperature", "condition_on_previous_text", "beam_size"):
        assert key in config


def test_temperature_is_forwarded_to_the_engine(backend, monkeypatch, tmp_path):
    """Recording the value is not enough; it has to reach the decoder."""
    seen = {}

    class _Model:
        model = _Weights()

        def transcribe(self, audio, **kwargs):
            seen.update(kwargs)

            class _Info:
                language = "en"
                duration = 1.0

            return iter(()), _Info()

    stub = _Model()
    monkeypatch.setattr(backend, "_model", stub)
    monkeypatch.setattr(backend, "_build_model", lambda device: stub)
    backend._resolved_device = "cpu"
    backend.transcribe(tmp_path / "a.wav", "a.wav")

    assert (
        seen["temperature"] == FasterWhisperBackend().effective_config()["temperature"]
    )
    assert seen["condition_on_previous_text"] is True


def test_greedy_decoding_can_be_selected_by_config():
    """The ladder is the default, not a cage: greedy stays available."""
    assert (
        FasterWhisperBackend(temperature=0.0).effective_config()["temperature"] == 0.0
    )


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
        model = _Weights()

        def transcribe(self, audio, **kwargs):
            seen.update(kwargs)

            class _Info:
                language = "en"
                duration = 1.0

            return iter(()), _Info()

    monkeypatch.setattr(
        "audio_transcriber.backends.faster_whisper.ctranslate2.set_random_seed",
        lambda value: None,
    )
    backend = FasterWhisperBackend(model="tiny", seed=None)
    monkeypatch.setattr(backend, "_model", _Model())
    backend.transcribe(tmp_path / "a.wav", "a.wav")

    from audio_transcriber.backends.faster_whisper import _NON_DECODE_OPTIONS

    recorded = set(backend.effective_config())
    assert (recorded - _NON_DECODE_OPTIONS) <= set(seen), (
        "recorded but never forwarded: "
        f"{sorted((recorded - _NON_DECODE_OPTIONS) - set(seen))}"
    )


def test_repetition_controls_are_forwarded(monkeypatch, tmp_path):
    """These are the deterministic alternative to temperature sampling."""
    seen = {}

    class _Model:
        model = _Weights()

        def transcribe(self, audio, **kwargs):
            seen.update(kwargs)

            class _Info:
                language = "en"
                duration = 1.0

            return iter(()), _Info()

    monkeypatch.setattr(
        "audio_transcriber.backends.faster_whisper.ctranslate2.set_random_seed",
        lambda value: None,
    )
    backend = FasterWhisperBackend(
        repetition_penalty=1.1, no_repeat_ngram_size=3, seed=None
    )
    monkeypatch.setattr(backend, "_model", _Model())
    backend.transcribe(tmp_path / "a.wav", "a.wav")

    assert seen["repetition_penalty"] == 1.1
    assert seen["no_repeat_ngram_size"] == 3


# --- Seeded decoding ---
#
# The temperature ladder re-decodes segments that trip Whisper's quality
# thresholds, and it is worth keeping: on one recording it scored WER 0.434
# against 0.619 for the best deterministic configuration. Its sampling is what
# made runs unreproducible — but CTranslate2's RNG takes a seed, so seeding it
# gives both. Verified: the same seed produced byte-identical transcripts, and a
# different seed produced different ones.


def test_temperature_defaults_to_the_fallback_ladder():
    """Quality needs it: 14.5% of segments fall back, and some need 1.0."""
    temperature = FasterWhisperBackend().effective_config()["temperature"]

    assert isinstance(temperature, list)
    assert temperature[0] == 0.0
    assert max(temperature) > 0.0


def test_seed_is_recorded_in_the_effective_config():
    """The seed is part of what reproduces a run, so the manifest needs it."""
    assert FasterWhisperBackend().effective_config()["seed"] == 0


def test_a_seed_rebuilds_the_model_for_every_file(monkeypatch, tmp_path):
    """The seed binds when the model is built, so applying it means rebuilding.

    An earlier version called `set_random_seed` between decodes on one model and
    asserted only that the call happened. It happened, and it did nothing: the
    same recording still scored 0.375 alone and 0.476 as the fifth file of a run.
    """
    builds = []

    class _Model:
        model = _Weights()

        def transcribe(self, audio, **kwargs):
            class _Info:
                language = "en"
                duration = 1.0

            return iter(()), _Info()

    backend = FasterWhisperBackend(seed=7)
    backend._resolved_device = "cpu"
    monkeypatch.setattr(backend, "_model", _Model())
    monkeypatch.setattr(
        backend, "_build_model", lambda device: builds.append(device) or _Model()
    )

    backend.transcribe(tmp_path / "a.wav", "a.wav")
    backend.transcribe(tmp_path / "b.wav", "b.wav")

    assert builds == ["cpu", "cpu"]


def test_the_previous_model_is_unloaded_before_the_next_is_built(monkeypatch):
    """Two copies of the weights must never be resident at once.

    Dropping the Python reference was assumed to free the old model. It did not
    — something still held it — so every seeded run peaked at twice the model:
    7.9 GB for large-v3 on an 8 GB card. Adding diarization overflowed that,
    WSL2 paged GPU memory to system RAM, and a 57 s diarization took over 20
    minutes. Unloading frees the weights whatever else refers to the object.
    """
    events = []

    class _Weights:
        def unload_model(self):
            events.append("unload")

    class _Model:
        def __init__(self):
            self.model = _Weights()

    backend = FasterWhisperBackend(seed=0)
    backend._resolved_device = "cpu"
    backend._model = _Model()

    def _build(device):
        events.append("build")
        return _Model()

    monkeypatch.setattr(backend, "_build_model", _build)

    backend._model_for_file()
    backend._model_for_file()

    assert events == ["unload", "build", "unload", "build"]


def test_no_seed_reuses_the_loaded_model(monkeypatch, tmp_path):
    """Opting out of reproducibility should also opt out of its cost."""
    builds = []

    class _Model:
        model = _Weights()

        def transcribe(self, audio, **kwargs):
            class _Info:
                language = "en"
                duration = 1.0

            return iter(()), _Info()

    backend = FasterWhisperBackend(seed=None)
    monkeypatch.setattr(backend, "_model", _Model())
    monkeypatch.setattr(
        backend, "_build_model", lambda device: builds.append(device) or _Model()
    )

    backend.transcribe(tmp_path / "a.wav", "a.wav")

    assert builds == []


def test_build_model_seeds_before_construction(monkeypatch):
    """Seeding after construction is too late, which is the whole bug."""
    order = []
    monkeypatch.setattr(
        "audio_transcriber.backends.faster_whisper.ctranslate2.set_random_seed",
        lambda value: order.append(f"seed:{value}"),
    )
    monkeypatch.setattr(
        "audio_transcriber.backends.faster_whisper.WhisperModel",
        lambda *args, **kwargs: order.append("build"),
    )

    FasterWhisperBackend(seed=99, model="tiny")._build_model("cpu")

    assert order == ["seed:99", "build"]


def test_the_seed_is_not_passed_to_the_decoder(monkeypatch, tmp_path):
    """It is applied to the RNG; transcribe() would reject it as a keyword."""
    seen = {}
    monkeypatch.setattr(
        "audio_transcriber.backends.faster_whisper.ctranslate2.set_random_seed",
        lambda value: None,
    )

    class _Model:
        model = _Weights()

        def transcribe(self, audio, **kwargs):
            seen.update(kwargs)

            class _Info:
                language = "en"
                duration = 1.0

            return iter(()), _Info()

    backend = FasterWhisperBackend(seed=None)
    monkeypatch.setattr(backend, "_model", _Model())
    backend.transcribe(tmp_path / "a.wav", "a.wav")

    assert "seed" not in seen


# --- Implausibly dense segments ---


def test_normalize_drops_a_hallucination_squashed_by_the_end_clamp(backend):
    """Observed with large-v3: eight words clamped into the last 15 ms of audio.

    The segment was a hallucination running past the end of the recording.
    Starting just inside the audio, it was clamped rather than dropped, which
    left a stock YouTube outro claiming to have been said in 15 ms.
    """
    segments = backend._normalize(
        [
            _NativeSegment(0.0, 5.0, "and that was the whole story"),
            _NativeSegment(
                10.42, 14.0, "For more info, see the links in the description."
            ),
        ],
        "en",
        duration=10.435,
    )

    assert [s.text for s in segments] == ["and that was the whole story"]


def test_normalize_keeps_fast_but_real_speech(backend):
    """Fast conversational speech runs around 4-5 words per second."""
    segments = backend._normalize(
        [_NativeSegment(0.0, 2.0, "no no I told you that already")], "en"
    )

    assert len(segments) == 1


def test_normalize_keeps_a_short_single_word(backend):
    """A lone backchannel is short, and its timestamps are coarse.

    One word in a tenth of a second is a density no sentence reaches, but it is
    what "yeah" looks like when the decoder places it tightly. Density is only
    evidence of hallucination when there is enough text for it to mean anything.
    """
    segments = backend._normalize([_NativeSegment(3.0, 3.1, "Yeah.")], "en")

    assert len(segments) == 1
