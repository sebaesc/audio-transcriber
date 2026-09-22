"""The shared transcript schema.

Every backend normalizes its native output into these types. Downstream consumers
read this schema and nothing else, so a backend swap does not change their input.

The schema is a project contract: per `conductor/workflow.md`, changing it requires
approval. Fields are added rather than repurposed, and `schema_version` moves when
the shape changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Self

SCHEMA_VERSION = "1.0"

_SEGMENT_REQUIRED = ("start", "end", "text")


class SchemaError(ValueError):
    """Raised when data does not satisfy the transcript schema."""


@dataclass(frozen=True, slots=True)
class Segment:
    """A contiguous span of transcribed speech.

    Attributes:
        start: Segment start, in seconds from the beginning of the audio.
        end: Segment end, in seconds. Never earlier than `start`.
        text: Transcribed text, stripped of leading and trailing whitespace.
        speaker: Speaker label, or None when the backend does not diarize.
        confidence: Backend-reported confidence in 0.0-1.0, or None if unavailable.
    """

    start: float
    end: float
    text: str
    speaker: str | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.start < 0:
            raise SchemaError(f"segment start must be >= 0, got {self.start}")
        if self.end < self.start:
            raise SchemaError(
                f"segment end ({self.end}) is before its start ({self.start})"
            )
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise SchemaError(
                f"segment confidence must be in 0.0-1.0, got {self.confidence}"
            )

    @property
    def duration(self) -> float:
        """Length of the segment in seconds."""
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        """Return the segment as JSON-serializable data."""
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "speaker": self.speaker,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Build a segment from deserialized JSON data.

        Raises:
            SchemaError: If a required key is missing or a value is out of range.
        """
        missing = [key for key in _SEGMENT_REQUIRED if key not in data]
        if missing:
            raise SchemaError(f"segment is missing required keys: {missing}")
        return cls(
            start=float(data["start"]),
            end=float(data["end"]),
            text=str(data["text"]),
            speaker=data.get("speaker"),
            confidence=data.get("confidence"),
        )


@dataclass(frozen=True, slots=True)
class Transcript:
    """A normalized transcript of one audio file.

    Attributes:
        source: Path of the audio file this transcript came from, as given to the run.
        segments: Transcribed segments in non-overlapping ascending time order.
        language: Detected or configured language code, or None if the backend
            does not report one.
        duration: Audio duration in seconds, or None if the backend does not report it.
        backend: Name of the backend that produced this transcript.
        model_version: Version identifier of the model used, recorded for reproducibility.
        schema_version: Version of this schema, for forward compatibility.
    """

    source: str
    segments: tuple[Segment, ...] = ()
    language: str | None = None
    duration: float | None = None
    backend: str = ""
    model_version: str = ""
    schema_version: str = field(default=SCHEMA_VERSION)

    def __post_init__(self) -> None:
        if not self.source:
            raise SchemaError("transcript source must not be empty")
        previous_end = 0.0
        for index, segment in enumerate(self.segments):
            if segment.start < previous_end:
                raise SchemaError(
                    f"segment {index} starts at {segment.start}, "
                    f"before the previous segment ended at {previous_end}"
                )
            previous_end = segment.end

    @property
    def text(self) -> str:
        """The full transcript as a single string, segments joined by spaces."""
        return " ".join(segment.text for segment in self.segments if segment.text)

    def to_dict(self) -> dict[str, Any]:
        """Return the transcript as JSON-serializable data."""
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "language": self.language,
            "duration": self.duration,
            "backend": self.backend,
            "model_version": self.model_version,
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Build a transcript from deserialized JSON data.

        Raises:
            SchemaError: If the data does not satisfy the schema.
        """
        if "source" not in data:
            raise SchemaError("transcript is missing required key: source")
        return cls(
            source=str(data["source"]),
            segments=tuple(
                Segment.from_dict(item) for item in data.get("segments", ())
            ),
            language=data.get("language"),
            duration=data.get("duration"),
            backend=data.get("backend", ""),
            model_version=data.get("model_version", ""),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )
