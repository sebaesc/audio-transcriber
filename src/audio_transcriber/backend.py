"""The backend interface.

A transcription engine becomes usable by implementing `TranscriptionBackend` and
registering itself by name. Pipeline code holds only this interface: it never
imports an engine's library and never branches on a backend name.

Engine-specific abilities are declared through `Capabilities`, not discovered by
inspecting the backend's type. A pipeline feature that only one backend can
support does not belong in the pipeline.

The interface is a project contract: per `conductor/workflow.md`, changing it
requires approval.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audio_transcriber.schema import Transcript

#: Sample rate every backend receives decoded audio at.
TARGET_SAMPLE_RATE = 16_000


@dataclass(frozen=True, slots=True)
class Capabilities:
    """What a backend can do, declared rather than inferred.

    Attributes:
        diarization: Whether the backend labels segments by speaker.
        language_detection: Whether the backend detects the language itself.
        word_timestamps: Whether the backend reports per-word timings.
        translation: Whether the backend can translate while transcribing.
    """

    diarization: bool = False
    language_detection: bool = False
    word_timestamps: bool = False
    translation: bool = False

    def to_dict(self) -> dict[str, bool]:
        """Return the capabilities as JSON-serializable data."""
        return {
            "diarization": self.diarization,
            "language_detection": self.language_detection,
            "word_timestamps": self.word_timestamps,
            "translation": self.translation,
        }


class TranscriptionBackend(ABC):
    """One transcription engine behind the pipeline's stable interface.

    A backend owns its own normalization: `transcribe` returns a `Transcript`
    conforming to the shared schema, never the engine's native output. It also
    owns its requirement checks, so a missing model or credential is reported
    before a run starts rather than part-way through a batch.

    Subclasses set `name` and implement `check_requirements`, `model_version`,
    and `transcribe`.

    Attributes:
        name: The name this backend is selected by, as passed to `--backend`.
        capabilities: What this backend supports.
    """

    name: str = ""
    capabilities: Capabilities = Capabilities()

    def __init__(self, **config: Any) -> None:
        """Store the backend's effective configuration.

        Args:
            **config: Backend-specific options. Every value is recorded in the
                run manifest, so nothing is left implicit.
        """
        self.config = config

    @abstractmethod
    def check_requirements(self) -> None:
        """Verify everything this backend needs is present.

        Called once during up-front validation, before any file is processed.

        Raises:
            BackendRequirementError: If a model, credential, binary, or device
                the backend needs is unavailable. The error states how to fix it.
        """

    @abstractmethod
    def model_version(self) -> str:
        """Return the version identifier of the model in use.

        Recorded in the run manifest, so a transcript can be traced back to what
        produced it. Never a moving tag like "latest".
        """

    @abstractmethod
    def transcribe(self, audio_path: Path, source: str) -> Transcript:
        """Transcribe one decoded audio file.

        Args:
            audio_path: Path to audio already decoded to mono 16-bit PCM WAV at
                `TARGET_SAMPLE_RATE`. The backend does not decode audio itself.
            source: Path of the original input file, recorded on the transcript
                so it points at the user's file rather than a temporary one.

        Returns:
            A `Transcript` conforming to the shared schema, with segments in
            ascending, non-overlapping time order.

        Raises:
            TranscriptionError: If this file cannot be transcribed. The run
                collects the failure and continues with the next file.
        """

    def effective_config(self) -> dict[str, Any]:
        """Return the fully resolved config for the run manifest.

        Subclasses override this when defaults are applied inside the backend, so
        the manifest records what was actually used rather than what was passed.
        """
        return dict(self.config)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
