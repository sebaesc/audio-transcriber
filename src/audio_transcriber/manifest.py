"""The run manifest.

Every run writes a manifest recording what produced its transcripts: the backend,
the model version, the effective config, and a checksum of each input. Given a
manifest and the same inputs, a run can be reproduced exactly.

The manifest format is a project contract: per `conductor/workflow.md`, changing it
requires approval.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from audio_transcriber.schema import SCHEMA_VERSION

MANIFEST_VERSION = "1.0"

_CHECKSUM_CHUNK_BYTES = 1024 * 1024


def checksum_file(path: Path) -> str:
    """Return the SHA-256 checksum of a file, as `sha256:<hex>`.

    The file is read in chunks so that audio files larger than memory are handled.

    Args:
        path: Path to the file to checksum.

    Returns:
        The checksum, prefixed with its algorithm.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHECKSUM_CHUNK_BYTES):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class InputRecord:
    """One input file as the run saw it, and what it produced.

    A transcript file is a bare array of segments and carries no metadata of its
    own. The per-file metadata lives here instead: it is available for tracing a
    transcript back to its input, and ignorable by a consumer that only wants the
    segments.

    Attributes:
        path: Path of the input file, as given to the run.
        checksum: Checksum of the file's bytes, `sha256:<hex>`.
        size_bytes: Size of the file in bytes.
        transcript: Path of the transcript written for this input, relative to the
            run directory, or None if the file failed.
        duration: Audio duration in seconds, or None if unreported.
        language: Language code for the file as a whole, or None if unreported.
        segment_count: Number of segments in the transcript, or None if it failed.
    """

    path: str
    checksum: str
    size_bytes: int
    transcript: str | None = None
    duration: float | None = None
    language: str | None = None
    segment_count: int | None = None

    @classmethod
    def from_path(cls, path: Path) -> Self:
        """Build a record by reading the file at `path`.

        The result describes the input only. The fields describing what it produced
        are filled in with `with_result` once the file has been transcribed.
        """
        return cls(
            path=str(path),
            checksum=checksum_file(path),
            size_bytes=path.stat().st_size,
        )

    def with_result(self, transcript_path: str, metadata: dict[str, Any]) -> Self:
        """Return a copy carrying what this input produced.

        Args:
            transcript_path: Path of the transcript file, relative to the run directory.
            metadata: The transcript's optional sidecar, from `Transcript.metadata()`.
        """
        return replace(
            self,
            transcript=transcript_path,
            duration=metadata.get("duration"),
            language=metadata.get("language"),
            segment_count=metadata.get("segment_count"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the record as JSON-serializable data."""
        return {
            "path": self.path,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
            "transcript": self.transcript,
            "duration": self.duration,
            "language": self.language,
            "segment_count": self.segment_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Build a record from deserialized JSON data."""
        return cls(
            path=str(data["path"]),
            checksum=str(data["checksum"]),
            size_bytes=int(data["size_bytes"]),
            transcript=data.get("transcript"),
            duration=data.get("duration"),
            language=data.get("language"),
            segment_count=data.get("segment_count"),
        )


@dataclass(slots=True)
class Manifest:
    """What produced a run's transcripts.

    Attributes:
        run_id: Identifier of the run, matching its output directory name.
        backend: Name of the backend used.
        model_version: Version identifier of the model, as the backend reports it.
        config: The effective config, with every value resolved. Nothing is left
            implicit in a CLI flag default.
        inputs: One record per input file, in the run's processing order.
        started_at: ISO 8601 UTC timestamp of when the run started.
        finished_at: ISO 8601 UTC timestamp of when the run ended, or None while
            the run is in progress.
        succeeded: Count of files transcribed successfully.
        failed: Count of files that failed.
        tool_version: Version of audio-transcriber that produced the run.
        manifest_version: Version of this manifest format.
        schema_version: Version of the transcript schema the run's transcript files
            follow. Recorded here because those files are bare arrays with no room
            for a version of their own.
    """

    run_id: str
    backend: str
    model_version: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    inputs: list[InputRecord] = field(default_factory=list)
    started_at: str = field(default_factory=utc_now)
    finished_at: str | None = None
    succeeded: int = 0
    failed: int = 0
    tool_version: str = ""
    manifest_version: str = MANIFEST_VERSION
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return the manifest as JSON-serializable data."""
        return {
            "manifest_version": self.manifest_version,
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "tool_version": self.tool_version,
            "backend": self.backend,
            "model_version": self.model_version,
            "config": self.config,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "inputs": [record.to_dict() for record in self.inputs],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Build a manifest from deserialized JSON data."""
        return cls(
            run_id=str(data["run_id"]),
            backend=str(data["backend"]),
            model_version=data.get("model_version", ""),
            config=data.get("config", {}),
            inputs=[InputRecord.from_dict(item) for item in data.get("inputs", ())],
            started_at=data.get("started_at", ""),
            finished_at=data.get("finished_at"),
            succeeded=int(data.get("succeeded", 0)),
            failed=int(data.get("failed", 0)),
            tool_version=data.get("tool_version", ""),
            manifest_version=data.get("manifest_version", MANIFEST_VERSION),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )
