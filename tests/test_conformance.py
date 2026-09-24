"""The shared backend conformance suite.

Parameterized over every registered backend. A backend is not done until it
passes this file: `pytest -k conformance`, quality gate 3 in
`conductor/workflow.md`.

With no backends registered this collects nothing and is trivially green, which
is the expected state until the first backend lands in Phase 3.
"""

from pathlib import Path

import pytest

from audio_transcriber import registry
from audio_transcriber.backend import Capabilities, TranscriptionBackend
from audio_transcriber.conformance import (
    ConformanceError,
    assert_transcript_conforms,
    check_backend,
)
from audio_transcriber.errors import BackendRequirementError
from audio_transcriber.schema import Segment, Transcript

pytestmark = pytest.mark.conformance


#: Configurations beyond the default that change what a backend emits, so each
#: must meet the contract on its own. A mode that is only ever tested at its
#: defaults is a mode the suite has not checked.
_EXTRA_CONFIGS: dict[str, list[dict]] = {
    "faster-whisper": [
        {"language_detection": "chunked", "languages": ["en", "es"]},
        {"diarization": True},
        {
            "language_detection": "chunked",
            "languages": ["en", "es"],
            "diarization": True,
        },
        {"diarization": True, "speaker_assignment": "segment"},
        {
            "language_detection": "chunked",
            "languages": ["en", "es"],
            "diarization": True,
            "speaker_assignment": "segment",
        },
    ],
}


def _backend_configs() -> list:
    return [
        pytest.param((name, config), id=f"{name}-{index}")
        for name in registry.available()
        for index, config in enumerate([{}, *_EXTRA_CONFIGS.get(name, [])])
    ]


@pytest.fixture(params=_backend_configs())
def registered_backend(request) -> TranscriptionBackend:
    """Each registered backend in each configuration, skipped if unrunnable."""
    name, config = request.param
    backend = registry.create(name, **config)
    try:
        backend.check_requirements()
    except BackendRequirementError as error:
        pytest.skip(f"requirements not met: {error}")
    return backend


def test_registered_backend_conforms(registered_backend, wav_file):
    """Every registered backend satisfies the whole shared contract."""
    check_backend(registered_backend, wav_file, "audio/sample.wav")


def test_registered_backend_handles_silence(registered_backend, silent_wav_file):
    """Silence is a valid transcript with no segments, not an error."""
    transcript = check_backend(registered_backend, silent_wav_file, "audio/silence.wav")

    assert isinstance(transcript, Transcript)


def test_registered_backend_declares_capabilities(registered_backend):
    assert isinstance(registered_backend.capabilities, Capabilities)


def test_registered_backend_reports_a_pinned_model_version(registered_backend):
    """A moving tag makes a transcript untraceable."""
    version = registered_backend.model_version()

    assert version
    assert version != "latest"


# --- The suite's own tests: it must reject a backend that breaks the contract. ---


class _ConformingBackend(TranscriptionBackend):
    name = "conforming"

    def check_requirements(self) -> None:
        return None

    def model_version(self) -> str:
        return "conforming-1.0"

    def transcribe(self, audio_path: Path, source: str) -> Transcript:
        return Transcript(
            source=source,
            segments=(Segment(start_time=0.0, end_time=1.0, text="hello"),),
            backend=self.name,
            model_version=self.model_version(),
            duration=1.0,
        )


def test_suite_accepts_a_conforming_backend(wav_file):
    check_backend(_ConformingBackend(), wav_file, "audio/a.wav")


def test_suite_rejects_a_backend_returning_the_wrong_source():
    """A backend returning its decoded temp path breaks downstream traceability."""
    transcript = Transcript(
        source="/tmp/decoded.wav",
        backend="broken",
        model_version="1.0",
    )

    with pytest.raises(ConformanceError, match="Record the original input path"):
        assert_transcript_conforms(transcript, "broken", "audio/a.wav")


def test_suite_rejects_a_backend_without_a_model_version():
    transcript = Transcript(source="audio/a.wav", backend="broken")

    with pytest.raises(ConformanceError, match="no model_version"):
        assert_transcript_conforms(transcript, "broken", "audio/a.wav")


def test_suite_rejects_unstripped_segment_text():
    transcript = Transcript(
        source="audio/a.wav",
        segments=(Segment(start_time=0.0, end_time=1.0, text="  hello  "),),
        backend="broken",
        model_version="1.0",
    )

    with pytest.raises(ConformanceError, match="not stripped"):
        assert_transcript_conforms(transcript, "broken", "audio/a.wav")


def test_suite_rejects_segments_running_past_the_reported_duration():
    transcript = Transcript(
        source="audio/a.wav",
        segments=(Segment(start_time=0.0, end_time=30.0, text="hello"),),
        backend="broken",
        model_version="1.0",
        duration=5.0,
    )

    with pytest.raises(ConformanceError, match="past the reported"):
        assert_transcript_conforms(transcript, "broken", "audio/a.wav")


def test_suite_rejects_a_backend_returning_the_wrong_type():
    with pytest.raises(ConformanceError, match="not a Transcript"):
        assert_transcript_conforms([], "broken", "a.wav")


def test_suite_rejects_a_segment_with_the_wrong_keys(monkeypatch):
    """The five-key output contract is enforced, not just the field values."""
    from audio_transcriber import conformance

    monkeypatch.setattr(conformance, "SEGMENT_KEYS", ("start", "end", "text"))
    transcript = Transcript(
        source="audio/a.wav",
        segments=(Segment(start_time=0.0, end_time=1.0, text="hello"),),
        backend="broken",
        model_version="1.0",
    )

    with pytest.raises(ConformanceError, match="serializes as"):
        assert_transcript_conforms(transcript, "broken", "audio/a.wav")
