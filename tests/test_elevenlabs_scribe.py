"""Tests for the ElevenLabs Scribe v2 backend.

Nothing here touches the network. Option handling, normalization and error
mapping are tested against hand-written responses shaped like the API's
documented `SpeechToTextChunkResponseModel`, through a fake client. The live
API is covered only by the conformance suite, and only on explicit opt-in.
"""

import httpx
import pytest

pytest.importorskip("elevenlabs", reason="the elevenlabs extra is not installed")

from elevenlabs.core.api_error import ApiError  # noqa: E402
from elevenlabs.errors import UnauthorizedError  # noqa: E402
from elevenlabs.types import (  # noqa: E402
    MultichannelSpeechToTextResponseModel,
    SpeechToTextChunkResponseModel,
)

from audio_transcriber.backends import credentials, elevenlabs_scribe  # noqa: E402
from audio_transcriber.backends.elevenlabs_scribe import (  # noqa: E402
    ElevenLabsScribeBackend,
)
from audio_transcriber.conformance import (  # noqa: E402
    assert_transcript_conforms,
    check_backend,
)
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


# --- The API call ---


class _FakeSpeechToText:
    """Stands in for `client.speech_to_text`, recording what it was sent."""

    def __init__(self, result):
        self.result = result
        self.calls: list[dict] = []
        self.uploaded: bytes | None = None

    def convert(self, **kwargs):
        name, handle, content_type = kwargs.pop("file")
        self.uploaded = handle.read()
        self.calls.append({**kwargs, "_file": (name, content_type)})
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class _FakeClient:
    def __init__(self, result):
        self.speech_to_text = _FakeSpeechToText(result)


def _with_client(backend, result):
    """Give a backend a fake client, as check_requirements would a real one."""
    backend._api_key = "sk_test_key"
    backend._client = _FakeClient(result)
    return backend._client.speech_to_text


_HELLO = _response(_spoken(("Hello.", 0.0, 0.5)), duration=1.0)


def test_transcribe_uploads_the_decoded_wav_and_normalizes(backend, wav_file):
    fake = _with_client(backend, _HELLO)

    transcript = backend.transcribe(wav_file, "audio/original.mp3")

    assert fake.uploaded == wav_file.read_bytes()
    assert fake.calls[0]["_file"] == ("sample.wav", "audio/wav")
    assert [s.text for s in transcript.segments] == ["Hello."]
    assert transcript.source == "audio/original.mp3"


def test_the_default_request_sends_exactly_the_recorded_settings(backend, wav_file):
    fake = _with_client(backend, _HELLO)

    backend.transcribe(wav_file, "a.wav")

    sent = {key: value for key, value in fake.calls[0].items() if key != "_file"}
    assert sent == {
        "model_id": "scribe_v2",
        "timestamps_granularity": "word",
        "tag_audio_events": False,
        "diarize": False,
        "seed": 0,
        "temperature": 0.0,
    }


def test_configured_options_reach_the_request(wav_file):
    backend = ElevenLabsScribeBackend(
        language="es", diarization=True, max_speakers=2, seed=5, temperature=0.2
    )
    fake = _with_client(backend, _HELLO)

    backend.transcribe(wav_file, "a.wav")

    sent = fake.calls[0]
    assert sent["language_code"] == "es"
    assert sent["diarize"] is True
    assert sent["num_speakers"] == 2
    assert sent["seed"] == 5
    assert sent["temperature"] == 0.2
    assert "diarization_threshold" not in sent


def test_a_diarization_threshold_reaches_the_request(wav_file):
    backend = ElevenLabsScribeBackend(diarization=True, diarization_threshold=0.3)
    fake = _with_client(backend, _HELLO)

    backend.transcribe(wav_file, "a.wav")

    assert fake.calls[0]["diarization_threshold"] == 0.3
    assert "num_speakers" not in fake.calls[0]


def test_options_left_unset_are_not_sent(wav_file):
    backend = ElevenLabsScribeBackend(seed=None, temperature=None)
    fake = _with_client(backend, _HELLO)

    backend.transcribe(wav_file, "a.wav")

    for key in ("seed", "temperature", "language_code", "num_speakers"):
        assert key not in fake.calls[0]


def test_every_recorded_option_affects_the_request_or_the_client():
    """An option recorded in the manifest but never applied is a silent lie."""
    applied_elsewhere = {"api_key", "timeout", "diarization", "language"}
    request_keys = {"max_speakers", "diarization_threshold", "seed", "temperature"}

    assert set(elevenlabs_scribe._DEFAULTS) == applied_elsewhere | request_keys


def test_the_client_gets_the_key_and_the_timeout(monkeypatch, no_key, wav_file):
    built = {}

    class _Recorder:
        def __init__(self, **kwargs):
            built.update(kwargs)
            self.speech_to_text = _FakeSpeechToText(_HELLO)

    monkeypatch.setattr(elevenlabs_scribe, "ElevenLabs", _Recorder)
    backend = ElevenLabsScribeBackend(api_key="sk_given", timeout=123)

    backend.transcribe(wav_file, "a.wav")

    assert built == {"api_key": "sk_given", "timeout": 123.0}


def test_duration_falls_back_to_the_wav_header(backend, wav_file):
    _with_client(backend, _response(_spoken(("hi", 0.0, 0.2)), duration=None))

    assert backend.transcribe(wav_file, "a.wav").duration == pytest.approx(1.0)


# --- Errors ---


def _api_error(status, body=None):
    return ApiError(status_code=status, headers={}, body=body)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (UnauthorizedError(body={"detail": {"status": "invalid_api_key"}}), "401"),
        (_api_error(403, {"detail": "missing permission"}), "403"),
        (_api_error(422, {"detail": [{"msg": "file too short"}]}), "file too short"),
        (_api_error(429, {"detail": "too many concurrent requests"}), "429"),
        (_api_error(503, "unavailable"), "503"),
    ],
)
def test_api_errors_become_per_file_failures(backend, wav_file, error, expected):
    _with_client(backend, error)

    with pytest.raises(TranscriptionError, match=expected) as raised:
        backend.transcribe(wav_file, "audio/a.wav")

    assert raised.value.source == "audio/a.wav"
    assert raised.value.backend == "elevenlabs-scribe"


def test_an_auth_failure_says_to_check_the_key(backend, wav_file):
    _with_client(backend, UnauthorizedError(body={"detail": "invalid"}))

    with pytest.raises(TranscriptionError, match="ELEVENLABS_API_KEY"):
        backend.transcribe(wav_file, "a.wav")


def test_rate_limiting_says_the_sdk_already_retried(backend, wav_file):
    _with_client(backend, _api_error(429))

    with pytest.raises(TranscriptionError, match="retr"):
        backend.transcribe(wav_file, "a.wav")


def test_a_timeout_names_the_option_that_raises_it(backend, wav_file):
    _with_client(backend, httpx.ReadTimeout("timed out"))

    with pytest.raises(TranscriptionError, match="timeout=") as raised:
        backend.transcribe(wav_file, "a.wav")

    assert f"{elevenlabs_scribe.DEFAULT_TIMEOUT:g}" in str(raised.value)


def test_a_connection_failure_is_a_per_file_failure(backend, wav_file):
    _with_client(backend, httpx.ConnectError("name resolution failed"))

    with pytest.raises(TranscriptionError, match="could not reach"):
        backend.transcribe(wav_file, "a.wav")


def test_a_long_error_body_is_truncated(backend, wav_file):
    _with_client(backend, _api_error(400, "x" * 5000))

    with pytest.raises(TranscriptionError) as raised:
        backend.transcribe(wav_file, "a.wav")

    assert len(str(raised.value)) < 600


def test_the_api_key_never_appears_in_an_error(backend, wav_file):
    """Even if the API were to echo it back."""
    _with_client(backend, _api_error(400, {"detail": "bad key sk_test_key"}))

    with pytest.raises(TranscriptionError) as raised:
        backend.transcribe(wav_file, "a.wav")

    assert "sk_test_key" not in str(raised.value)


# --- Conformance, offline ---
#
# The shared suite skips this backend unless AUDIO_TRANSCRIBER_LIVE_TESTS=1, so
# it is run here against a fake client on every run.


@pytest.mark.conformance
@pytest.mark.parametrize(
    "config", [{}, {"diarization": True}], ids=["default", "diarized"]
)
def test_the_shared_contract_holds_offline(config, wav_file):
    backend = ElevenLabsScribeBackend(**config)
    _with_client(
        backend,
        _response(
            _spoken(
                ("Hola,", 0.0, 0.3, "speaker_0"),
                ("¿qué", 0.35, 0.5, "speaker_0"),
                ("tal?", 0.5, 0.7, "speaker_0"),
                ("Fine.", 0.8, 1.2, "speaker_1"),
            ),
            language="spa",
            duration=1.0,
        ),
    )

    check_backend(backend, wav_file, "audio/sample.wav")


@pytest.mark.conformance
def test_silence_conforms_offline(backend, silent_wav_file):
    _with_client(backend, _response([], duration=1.0))

    transcript = check_backend(backend, silent_wav_file, "audio/silence.wav")

    assert transcript.segments == ()
