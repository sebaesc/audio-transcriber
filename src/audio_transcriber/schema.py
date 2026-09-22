"""The shared transcript schema.

Every backend normalizes its native output into these types. Downstream consumers
read this schema and nothing else, so a backend swap does not change their input.

A transcript **file** is a bare JSON array of segment objects, each with exactly
five keys: `start_time`, `end_time`, `speaker`, `spoken_language`, `text`. A
consumer that only wants the segments reads the file and iterates it, with no
wrapper to unpack.

Per-file metadata — duration, language, segment count — is not in that array. It
is recorded as an optional sidecar on the run manifest's input records, so it is
available for traceability and ignorable by consumers that do not want it.

The schema is a project contract: per `conductor/workflow.md`, changing it requires
approval. Keys are added rather than repurposed, and `SCHEMA_VERSION` moves when
the shape changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

SCHEMA_VERSION = "2.0"

#: The keys of a serialized segment, in the order they are written.
SEGMENT_KEYS = ("start_time", "end_time", "speaker", "spoken_language", "text")

_SEGMENT_REQUIRED = ("start_time", "end_time", "text")


class SchemaError(ValueError):
    """Raised when data does not satisfy the transcript schema."""


@dataclass(frozen=True, slots=True)
class Segment:
    """A contiguous span of transcribed speech.

    One of these, serialized, is one object in a transcript file's array.

    Attributes:
        start_time: Segment start, in seconds from the beginning of the audio.
        end_time: Segment end, in seconds. Never earlier than `start_time`.
        text: Transcribed text, stripped of leading and trailing whitespace.
        speaker: Speaker label, or None when the backend does not diarize.
        spoken_language: Language code of this segment's speech, or None when the
            backend reports no language. Per segment rather than per file, so a
            backend that detects a language switch mid-audio can express it.
    """

    start_time: float
    end_time: float
    text: str
    speaker: str | None = None
    spoken_language: str | None = None

    def __post_init__(self) -> None:
        if self.start_time < 0:
            raise SchemaError(f"segment start_time must be >= 0, got {self.start_time}")
        if self.end_time < self.start_time:
            raise SchemaError(
                f"segment end_time ({self.end_time}) is before its "
                f"start_time ({self.start_time})"
            )

    @property
    def duration(self) -> float:
        """Length of the segment in seconds."""
        return self.end_time - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Return the segment as one object of a transcript file's array."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "speaker": self.speaker,
            "spoken_language": self.spoken_language,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Build a segment from one object of a transcript file's array.

        Raises:
            SchemaError: If a required key is missing or a value is out of range.
        """
        missing = [key for key in _SEGMENT_REQUIRED if key not in data]
        if missing:
            raise SchemaError(f"segment is missing required keys: {missing}")
        return cls(
            start_time=float(data["start_time"]),
            end_time=float(data["end_time"]),
            text=str(data["text"]),
            speaker=data.get("speaker"),
            spoken_language=data.get("spoken_language"),
        )


@dataclass(frozen=True, slots=True)
class Transcript:
    """A normalized transcript of one audio file.

    The in-memory object carries the metadata a run needs for its manifest. Only
    `segments` is written to the transcript file, as a bare array.

    Attributes:
        source: Path of the audio file this transcript came from, as given to the run.
        segments: Transcribed segments in non-overlapping ascending time order.
        language: Language code for the file as a whole, or None if the backend
            does not report one. Segments carry their own `spoken_language`.
        duration: Audio duration in seconds, or None if the backend does not report it.
        backend: Name of the backend that produced this transcript.
        model_version: Version identifier of the model used, recorded for reproducibility.
    """

    source: str
    segments: tuple[Segment, ...] = ()
    language: str | None = None
    duration: float | None = None
    backend: str = ""
    model_version: str = ""

    def __post_init__(self) -> None:
        if not self.source:
            raise SchemaError("transcript source must not be empty")
        previous_end = 0.0
        for index, segment in enumerate(self.segments):
            if segment.start_time < previous_end:
                raise SchemaError(
                    f"segment {index} starts at {segment.start_time}, "
                    f"before the previous segment ended at {previous_end}"
                )
            previous_end = segment.end_time

    @property
    def text(self) -> str:
        """The full transcript as a single string, segments joined by spaces."""
        return " ".join(segment.text for segment in self.segments if segment.text)

    def to_json_data(self) -> list[dict[str, Any]]:
        """Return the transcript file's contents: a bare array of segments."""
        return [segment.to_dict() for segment in self.segments]

    def metadata(self) -> dict[str, Any]:
        """Return the optional per-file sidecar recorded on the run manifest.

        Consumers that only want the segments never need this.
        """
        return {
            "schema_version": SCHEMA_VERSION,
            "language": self.language,
            "duration": self.duration,
            "segment_count": len(self.segments),
        }

    @classmethod
    def from_json_data(
        cls,
        data: list[dict[str, Any]],
        source: str,
        **metadata: Any,
    ) -> Self:
        """Rebuild a transcript from a transcript file's array.

        Args:
            data: The array read from a transcript file.
            source: Path of the audio the transcript came from. The array does not
                carry it, so the caller supplies it from the manifest or filename.
            **metadata: Optional `language`, `duration`, `backend`, `model_version`.

        Raises:
            SchemaError: If the data does not satisfy the schema.
        """
        if not isinstance(data, list):
            raise SchemaError(
                f"a transcript file is a JSON array of segments, got {type(data).__name__}"
            )
        return cls(
            source=source,
            segments=tuple(Segment.from_dict(item) for item in data),
            language=metadata.get("language"),
            duration=metadata.get("duration"),
            backend=metadata.get("backend", ""),
            model_version=metadata.get("model_version", ""),
        )
