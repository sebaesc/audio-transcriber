"""Tests for the ElevenLabs Scribe v2 backend.

Nothing here touches the network. Option handling, normalization and error
mapping are tested against hand-written responses shaped like the API's
documented `SpeechToTextChunkResponseModel`, through a fake client. The live
API is covered only by the conformance suite, and only on explicit opt-in.
"""

import pytest

pytest.importorskip("elevenlabs", reason="the elevenlabs extra is not installed")

from elevenlabs.types import (  # noqa: E402
    MultichannelSpeechToTextResponseModel,
    SpeechToTextChunkResponseModel,
)

from audio_transcriber.backends import credentials, elevenlabs_scribe  # noqa: E402
from audio_transcriber.backends.elevenlabs_scribe import (  # noqa: E402
    ElevenLabsScribeBackend,
)
from audio_transcriber.conformance import assert_transcript_conforms  # noqa: E402
from audio_transcriber.errors import (  # noqa: E402
    BackendRequirementError,
    TranscriptionError,
)

# --- Options ---


def test_defaults_are_recorded_in_the_effective_config():
    config = ElevenLabsScribeBackend().effective_config()

    assert config["seed"] == 0
    assert config["temperature"] == 0.0
    assert config["diarization"] is False
    assert config["language"] is None
    assert config["max_speakers"] is None
    assert config["diarization_threshold"] is None
    assert config["timeout"] == elevenlabs_scribe.DEFAULT_TIMEOUT


def test_explicit_config_overrides_defaults():
    config = ElevenLabsScribeBackend(seed=7, language="es").effective_config()

    assert config["seed"] == 7
    assert config["language"] == "es"


def test_an_unknown_option_is_rejected_with_the_valid_names():
    with pytest.raises(BackendRequirementError, match="unknown option") as error:
        ElevenLabsScribeBackend(beam_size=5)

    assert "max_speakers" in str(error.value)


@pytest.mark.parametrize("option", ["num_speakers", "min_speakers"])
def test_an_exact_or_minimum_speaker_count_points_at_max_speakers(option):
    """Scribe's count is a maximum; accepting these would silently change them."""
    with pytest.raises(BackendRequirementError, match="max_speakers"):
        ElevenLabsScribeBackend(diarization=True, **{option: 2})


@pytest.mark.parametrize("value", [0, 33, -1])
def test_max_speakers_outside_the_api_range_is_rejected(value):
    with pytest.raises(BackendRequirementError, match="between 1 and 32"):
        ElevenLabsScribeBackend(diarization=True, max_speakers=value)


def test_max_speakers_within_range_is_accepted():
    backend = ElevenLabsScribeBackend(diarization=True, max_speakers=32)

    assert backend.effective_config()["max_speakers"] == 32


def test_max_speakers_without_diarization_is_rejected():
    with pytest.raises(BackendRequirementError, match="diarization=true"):
        ElevenLabsScribeBackend(max_speakers=2)


def test_diarization_threshold_without_diarization_is_rejected():
    with pytest.raises(BackendRequirementError, match="diarization=true"):
        ElevenLabsScribeBackend(diarization_threshold=0.3)


def test_diarization_threshold_with_a_speaker_count_is_rejected():
    """The API accepts a threshold only when no speaker count is set."""
    with pytest.raises(BackendRequirementError, match="max_speakers"):
        ElevenLabsScribeBackend(
            diarization=True, max_speakers=2, diarization_threshold=0.3
        )


def test_diarization_threshold_alone_is_accepted():
    backend = ElevenLabsScribeBackend(diarization=True, diarization_threshold=0.3)

    assert backend.effective_config()["diarization_threshold"] == 0.3


def test_a_timeout_that_is_not_positive_is_rejected():
    with pytest.raises(BackendRequirementError, match="timeout"):
        ElevenLabsScribeBackend(timeout=0)


def test_diarization_is_declared_only_when_enabled():
    assert ElevenLabsScribeBackend().capabilities.diarization is False
    assert ElevenLabsScribeBackend(diarization=True).capabilities.diarization is True


def test_language_detection_and_word_timestamps_are_declared():
    capabilities = ElevenLabsScribeBackend().capabilities

    assert capabilities.language_detection is True
    assert capabilities.word_timestamps is True
    assert capabilities.translation is False


def test_the_api_key_is_redacted_from_the_effective_config():
    config = ElevenLabsScribeBackend(api_key="sk_secret").effective_config()

    assert config["api_key"] == "<redacted>"
    assert "sk_secret" not in repr(config)


def test_model_version_pins_the_model_and_the_sdk():
    version = ElevenLabsScribeBackend().model_version()

    assert version == f"elevenlabs-{elevenlabs_scribe.SDK_VERSION}/scribe_v2"


# --- Requirements ---


@pytest.fixture
def no_key(monkeypatch):
    """No key anywhere: a developer running this may have a real one in .env."""
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setattr(credentials, "_dotenv_loaded", True)


def test_a_missing_key_fails_up_front_and_says_where_to_put_it(no_key):
    with pytest.raises(BackendRequirementError) as error:
        ElevenLabsScribeBackend().check_requirements()

    message = str(error.value)
    assert "ELEVENLABS_API_KEY" in message
    assert ".env" in message
    assert ".env.example" in message


def test_the_key_is_read_from_the_environment(no_key, monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk_env")
    backend = ElevenLabsScribeBackend()

    backend.check_requirements()

    assert backend._api_key == "sk_env"


def test_an_explicit_key_beats_the_environment(no_key, monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk_env")
    backend = ElevenLabsScribeBackend(api_key="sk_explicit")

    backend.check_requirements()

    assert backend._api_key == "sk_explicit"


def test_a_dotenv_file_supplies_the_key(no_key, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("ELEVENLABS_API_KEY=sk_from_dotenv\n")
    monkeypatch.setattr(credentials, "_dotenv_loaded", False)
    backend = ElevenLabsScribeBackend()

    backend.check_requirements()

    assert backend._api_key == "sk_from_dotenv"


# --- Language codes ---


@pytest.mark.parametrize(
    ("native", "expected"),
    [
        ("eng", "en"),
        ("spa", "es"),
        ("en", "en"),
        ("es", "es"),
        ("ENG", "en"),
        (" spa ", "es"),
        ("zho", "zh"),
        ("fas", "fa"),
        ("nor", "no"),
    ],
)
def test_language_codes_become_the_two_letter_form_faster_whisper_emits(
    native, expected
):
    assert elevenlabs_scribe.normalize_language(native) == expected


@pytest.mark.parametrize("native", ["yue", "fil", "ceb", "xyz"])
def test_a_code_with_no_two_letter_form_passes_through(native):
    """Cantonese, Filipino and Cebuano have no ISO 639-1 code; guessing would lie."""
    assert elevenlabs_scribe.normalize_language(native) == native


@pytest.mark.parametrize("native", [None, "", "  "])
def test_no_language_stays_none(native):
    assert elevenlabs_scribe.normalize_language(native) is None


# --- Normalization ---
#
# Responses are built with the SDK's own models, so a fixture that does not
# match the documented shape fails here rather than passing on a guess.


def _word(text, start, end, speaker=None, kind="word"):
    return {
        "text": text,
        "type": kind,
        "start": start,
        "end": end,
        "logprob": -0.1,
        "speaker_id": speaker,
    }


def _spoken(*words):
    """Interleave (text, start, end[, speaker]) words with spacing tokens."""
    tokens = []
    for index, item in enumerate(words):
        text, start, end, *rest = item
        speaker = rest[0] if rest else None
        if index:
            gap_start = tokens[-1]["end"]
            tokens.append(_word(" ", gap_start, start, speaker, kind="spacing"))
        tokens.append(_word(text, start, end, speaker))
    return tokens


def _response(tokens, language="eng", duration=10.0):
    return SpeechToTextChunkResponseModel.model_validate(
        {
            "language_code": language,
            "language_probability": 0.97,
            "text": "".join(token["text"] for token in tokens),
            "words": tokens,
            "audio_duration_secs": duration,
        }
    )


@pytest.fixture
def backend() -> ElevenLabsScribeBackend:
    return ElevenLabsScribeBackend()


@pytest.fixture
def diarizing() -> ElevenLabsScribeBackend:
    return ElevenLabsScribeBackend(diarization=True)


def _segments(transcript):
    return [segment.to_dict() for segment in transcript.segments]


def test_words_become_one_segment_with_the_file_language(backend):
    response = _response(_spoken(("Hello", 0.0, 0.4), ("there.", 0.5, 0.9)))

    transcript = backend._normalize(response, "audio/a.wav", None)

    assert _segments(transcript) == [
        {
            "start_time": 0.0,
            "end_time": 0.9,
            "speaker": None,
            "spoken_language": "en",
            "text": "Hello there.",
        }
    ]
    assert transcript.language == "en"
    assert transcript.source == "audio/a.wav"
    assert transcript.backend == "elevenlabs-scribe"
    assert transcript.model_version == backend.model_version()


def test_a_sentence_end_starts_a_new_segment(backend):
    response = _response(
        _spoken(
            ("Hi.", 0.0, 0.3), ("How", 0.4, 0.6), ("are", 0.6, 0.8), ("you?", 0.8, 1.1)
        )
    )

    texts = [s.text for s in backend._normalize(response, "a.wav", None).segments]

    assert texts == ["Hi.", "How are you?"]


def test_a_long_pause_starts_a_new_segment(backend):
    gap = elevenlabs_scribe.PAUSE_SECONDS + 0.1
    response = _response(_spoken(("one", 0.0, 0.3), ("two", 0.3 + gap, 0.6 + gap)))

    segments = backend._normalize(response, "a.wav", None).segments

    assert [s.text for s in segments] == ["one", "two"]
    assert segments[1].start_time == pytest.approx(0.3 + gap)


def test_a_short_pause_does_not_split(backend):
    gap = elevenlabs_scribe.PAUSE_SECONDS - 0.1
    response = _response(_spoken(("one", 0.0, 0.3), ("two", 0.3 + gap, 0.6 + gap)))

    assert [s.text for s in backend._normalize(response, "a.wav", None).segments] == [
        "one two"
    ]


def test_a_speaker_change_starts_a_new_segment_when_diarizing(diarizing):
    response = _response(
        _spoken(
            ("yes", 0.0, 0.3, "speaker_0"),
            ("no", 0.4, 0.6, "speaker_1"),
            ("maybe", 0.7, 1.0, "speaker_1"),
        )
    )

    segments = diarizing._normalize(response, "a.wav", None).segments

    assert [(s.speaker, s.text) for s in segments] == [
        ("speaker_0", "yes"),
        ("speaker_1", "no maybe"),
    ]


def test_speaker_ids_are_ignored_without_diarization(backend):
    """The API may label speakers anyway; unrequested labels are not recorded."""
    response = _response(
        _spoken(("yes", 0.0, 0.3, "speaker_0"), ("no", 0.4, 0.6, "speaker_1"))
    )

    segments = backend._normalize(response, "a.wav", None).segments

    assert [(s.speaker, s.text) for s in segments] == [(None, "yes no")]


def test_every_segment_carries_the_file_language(backend):
    response = _response(
        _spoken(("Hola.", 0.0, 0.3), ("Hello.", 2.0, 2.3)), language="spa"
    )

    segments = backend._normalize(response, "a.wav", None).segments

    assert {s.spoken_language for s in segments} == {"es"}


def test_audio_events_are_dropped(backend):
    tokens = _spoken(("so", 0.0, 0.2), ("funny", 0.3, 0.6))
    tokens.insert(1, _word("(laughter)", 0.2, 0.3, kind="audio_event"))

    assert [
        s.text for s in backend._normalize(_response(tokens), "a.wav", None).segments
    ] == ["so funny"]


def test_text_keeps_the_spacing_scribe_returned(backend):
    """Languages written without spaces get no spacing tokens, and none are added."""
    tokens = [_word("こんにちは", 0.0, 0.5), _word("世界", 0.5, 0.9)]

    transcript = backend._normalize(_response(tokens, language="jpn"), "a.wav", None)

    assert [s.text for s in transcript.segments] == ["こんにちは世界"]
    assert transcript.language == "ja"


def test_a_word_without_timings_joins_its_segment_without_moving_it(backend):
    tokens = _spoken(("before", 0.0, 0.3), ("middle", None, None), ("after", 0.5, 0.8))

    assert _segments(backend._normalize(_response(tokens), "a.wav", None)) == [
        {
            "start_time": 0.0,
            "end_time": 0.8,
            "speaker": None,
            "spoken_language": "en",
            "text": "before middle after",
        }
    ]


def test_a_segment_with_no_timings_at_all_is_dropped(backend):
    tokens = [_word("ghost", None, None)]

    assert backend._normalize(_response(tokens), "a.wav", None).segments == ()


def test_overlapping_words_across_speakers_become_non_overlapping_segments(diarizing):
    response = _response(
        _spoken(("well", 0.0, 1.0, "speaker_0"), ("right", 0.8, 1.2, "speaker_1"))
    )

    segments = diarizing._normalize(response, "a.wav", None).segments

    assert segments[1].start_time >= segments[0].end_time


def test_duration_comes_from_the_response(backend):
    transcript = backend._normalize(
        _response(_spoken(("hi", 0.0, 0.2)), duration=12.5), "a.wav", 99.0
    )

    assert transcript.duration == 12.5


def test_duration_falls_back_to_the_audio_when_the_response_has_none(backend):
    transcript = backend._normalize(
        _response(_spoken(("hi", 0.0, 0.2)), duration=None), "a.wav", 3.0
    )

    assert transcript.duration == 3.0


def test_no_words_is_an_empty_transcript_not_an_error(backend):
    transcript = backend._normalize(_response([]), "a.wav", None)

    assert transcript.segments == ()
    assert transcript.language == "en"


def test_only_spacing_and_audio_events_is_an_empty_transcript(backend):
    tokens = [
        _word(" ", 0.0, 0.5, kind="spacing"),
        _word("(music)", 0.5, 3.0, kind="audio_event"),
    ]

    assert backend._normalize(_response(tokens), "a.wav", None).segments == ()


def test_a_single_word_is_one_segment(backend):
    segments = backend._normalize(_response([_word("Sí.", 0.1, 0.4)]), "a.wav", None)

    assert [s.text for s in segments.segments] == ["Sí."]


def test_a_segment_overhanging_the_end_is_clamped(backend):
    response = _response(_spoken(("late", 0.5, 1.4)), duration=1.0)

    segments = backend._normalize(response, "a.wav", None).segments

    assert segments[-1].end_time == 1.0


def test_a_segment_starting_at_or_after_the_end_is_dropped(backend):
    response = _response(
        _spoken(("real.", 0.0, 0.5), ("phantom.", 1.0, 1.3)), duration=1.0
    )

    assert [s.text for s in backend._normalize(response, "a.wav", None).segments] == [
        "real."
    ]


def test_a_normalized_transcript_meets_the_shared_contract(diarizing):
    response = _response(
        _spoken(
            ("Hola,", 0.0, 0.4, "speaker_0"),
            ("¿qué", 0.5, 0.7, "speaker_0"),
            ("tal?", 0.7, 1.0, "speaker_0"),
            ("Good,", 1.2, 1.5, "speaker_1"),
            ("thanks.", 1.5, 1.9, "speaker_1"),
        ),
        duration=2.0,
    )

    transcript = diarizing._normalize(response, "audio/a.wav", None)

    assert_transcript_conforms(transcript, "elevenlabs-scribe", "audio/a.wav")


def test_a_multichannel_response_is_rejected_clearly(backend):
    response = MultichannelSpeechToTextResponseModel.model_validate({"transcripts": []})

    with pytest.raises(TranscriptionError, match="multichannel"):
        backend._normalize(response, "a.wav", None)
