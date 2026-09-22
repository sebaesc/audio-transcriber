"""Tests for the backend interface itself.

The interface is what keeps engines swappable, so its shape is tested directly:
what a backend must implement, and what it is handed.
"""

from pathlib import Path

import pytest

from audio_transcriber.backend import (
    TARGET_SAMPLE_RATE,
    Capabilities,
    TranscriptionBackend,
)
from audio_transcriber.errors import BackendRequirementError, TranscriptionError
from audio_transcriber.schema import Transcript


class _StubBackend(TranscriptionBackend):
    name = "stub"
    capabilities = Capabilities(language_detection=True)

    def check_requirements(self) -> None:
        return None

    def model_version(self) -> str:
        return "stub-1.0"

    def transcribe(self, audio_path: Path, source: str) -> Transcript:
        return Transcript(source=source, backend=self.name)


def test_backend_cannot_be_instantiated_without_implementing_the_interface():
    class _Incomplete(TranscriptionBackend):
        name = "incomplete"

    with pytest.raises(TypeError):
        _Incomplete()


def test_backend_records_its_config():
    backend = _StubBackend(language="en", beam_size=5)

    assert backend.effective_config() == {"language": "en", "beam_size": 5}


def test_effective_config_is_a_copy_not_the_live_dict():
    """The manifest must not change when a backend mutates its config later."""
    backend = _StubBackend(language="en")
    recorded = backend.effective_config()
    backend.config["language"] = "de"

    assert recorded == {"language": "en"}


def test_transcribe_returns_a_transcript_naming_the_original_source():
    """The backend is handed a decoded temp file but records the user's path."""
    backend = _StubBackend()

    transcript = backend.transcribe(Path("/tmp/decoded.wav"), "audio/meeting.m4a")

    assert transcript.source == "audio/meeting.m4a"
    assert transcript.backend == "stub"


def test_capabilities_default_to_false():
    """A backend declares what it supports; nothing is assumed on its behalf."""
    capabilities = Capabilities()

    assert capabilities.to_dict() == {
        "diarization": False,
        "language_detection": False,
        "word_timestamps": False,
        "translation": False,
    }


def test_capabilities_are_serializable_for_the_manifest():
    assert Capabilities(diarization=True).to_dict()["diarization"] is True


def test_target_sample_rate_is_fixed_for_every_backend():
    assert TARGET_SAMPLE_RATE == 16_000


def test_requirement_error_states_how_to_fix_it():
    error = BackendRequirementError(
        "whisper-api", "no API key found.", "Set OPENAI_API_KEY."
    )

    assert str(error) == "whisper-api: no API key found. Set OPENAI_API_KEY."
    assert error.backend == "whisper-api"


def test_transcription_error_names_the_file_and_backend():
    error = TranscriptionError("audio/a.wav", "stub", "decoder returned no frames.")

    assert "audio/a.wav" in str(error)
    assert "stub" in str(error)
    assert error.source == "audio/a.wav"
