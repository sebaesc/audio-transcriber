"""Run orchestration.

A run discovers its inputs, validates everything it can before touching a file,
then decodes and transcribes each input in turn. Per-file failures are collected
rather than fatal: a batch finishes and reports, and partial results are kept.

Nothing here knows which engine is in use. The backend arrives as a name, is
resolved through the registry, and is called through the interface.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from audio_transcriber import audio, discovery
from audio_transcriber.backend import TranscriptionBackend
from audio_transcriber.conformance import ConformanceError, assert_transcript_conforms
from audio_transcriber.errors import (
    AudioTranscriberError,
    ConfigError,
    TranscriptionError,
)
from audio_transcriber.manifest import InputRecord, Manifest, utc_now
from audio_transcriber.registry import create
from audio_transcriber.schema import Transcript

MANIFEST_FILENAME = "manifest.json"
ERRORS_FILENAME = "errors.json"

_logger = logging.getLogger(__name__)


class RunStatus(Enum):
    """How a run finished. The CLI maps these onto exit codes."""

    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True)
class FileFailure:
    """One input that could not be transcribed.

    Attributes:
        source: Path of the input file.
        error: The error class name, for grouping failures by cause.
        message: What happened and how to fix it.
    """

    source: str
    error: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Return the failure as JSON-serializable data."""
        return {"source": self.source, "error": self.error, "message": self.message}


@dataclass(slots=True)
class RunResult:
    """What a run produced.

    Attributes:
        run_directory: Where the outputs were written.
        manifest: The run's manifest, as written to disk.
        transcripts: Successful transcripts, in processing order.
        failures: Inputs that failed, in processing order.
    """

    run_directory: Path
    manifest: Manifest
    transcripts: list[Transcript] = field(default_factory=list)
    failures: list[FileFailure] = field(default_factory=list)

    @property
    def status(self) -> RunStatus:
        """Whether every file succeeded, some failed, or nothing succeeded."""
        if not self.failures:
            return RunStatus.SUCCEEDED
        if self.transcripts:
            return RunStatus.PARTIAL
        return RunStatus.ABORTED


def new_run_id() -> str:
    """Return a run identifier: a UTC timestamp, safe as a directory name."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")


def transcript_filenames(sources: Iterable[Path]) -> dict[Path, str]:
    """Map each input to a unique transcript filename.

    Two inputs in different directories can share a stem, so a bare stem is not
    enough: the second would silently overwrite the first. Collisions are
    disambiguated with a counter, in discovery order.

    Args:
        sources: The input paths, in processing order.

    Returns:
        A mapping from input path to output filename.
    """
    names: dict[Path, str] = {}
    used: set[str] = set()
    for source in sources:
        stem = source.stem or "transcript"
        candidate = f"{stem}.json"
        counter = 2
        while candidate in used:
            candidate = f"{stem}-{counter}.json"
            counter += 1
        used.add(candidate)
        names[source] = candidate
    return names


def prepare_run_directory(output_root: Path, run_id: str) -> Path:
    """Create the directory for this run's outputs.

    Args:
        output_root: The `--out` directory, created if missing.
        run_id: Identifier for this run, used as the directory name.

    Returns:
        The run directory.

    Raises:
        ConfigError: If the run directory already holds results. Prior runs are
            never overwritten in place.
    """
    run_directory = output_root / run_id
    if run_directory.exists() and any(run_directory.iterdir()):
        raise ConfigError(
            f"output directory already has results: {run_directory}. "
            "Use a different --out, or remove it."
        )
    run_directory.mkdir(parents=True, exist_ok=True)
    return run_directory


def validate(
    input_path: Path,
    backend: TranscriptionBackend,
    pattern: str = discovery.DEFAULT_GLOB,
) -> list[Path]:
    """Check everything knowable before the run starts.

    Validating once, up front, means a missing ffmpeg or an unavailable model is
    reported immediately rather than on the first file, or worse, the last.

    Args:
        input_path: Directory or file to transcribe.
        backend: The instantiated backend.
        pattern: Glob pattern for discovery.

    Returns:
        The discovered input files.

    Raises:
        ConfigError: If the input is missing or nothing matched.
        BackendRequirementError: If the backend is not ready to run.
    """
    sources = discovery.discover(input_path, pattern)
    if not sources:
        raise ConfigError(
            f"no audio files matched in {input_path}. Check the input path or --glob."
        )
    audio.check_ffmpeg()
    backend.check_requirements()
    return sources


def run(
    input_path: Path,
    backend_name: str,
    output_root: Path,
    pattern: str = discovery.DEFAULT_GLOB,
    backend_config: dict[str, Any] | None = None,
    run_id: str | None = None,
    on_progress: Callable[[int, int, Path], None] | None = None,
) -> RunResult:
    """Transcribe every matching input and write the results.

    Args:
        input_path: Directory or file to transcribe.
        backend_name: Name of the backend to use.
        output_root: Directory that will hold this run's output directory.
        pattern: Glob pattern for discovery.
        backend_config: Backend-specific options, recorded in the manifest.
        run_id: Identifier for this run. Defaults to a UTC timestamp.
        on_progress: Called with (index, total, source) before each file.

    Returns:
        The run's result, including any per-file failures.

    Raises:
        ConfigError: If validation fails. Nothing is written in that case.
        BackendRequirementError: If the backend is not ready to run.
    """
    backend = create(backend_name, **(backend_config or {}))
    sources = validate(input_path, backend, pattern)

    run_id = run_id or new_run_id()
    run_directory = prepare_run_directory(output_root, run_id)
    filenames = transcript_filenames(sources)

    manifest = Manifest(
        run_id=run_id,
        backend=backend.name,
        model_version=backend.model_version(),
        config=_recordable(backend.effective_config()),
        tool_version=_tool_version(),
        started_at=utc_now(),
    )
    result = RunResult(run_directory=run_directory, manifest=manifest)

    for index, source in enumerate(sources, start=1):
        if on_progress is not None:
            on_progress(index, len(sources), source)
        record = InputRecord.from_path(source)
        try:
            transcript = _transcribe_one(backend, source)
        except AudioTranscriberError as error:
            _logger.warning("%s", error)
            result.failures.append(
                FileFailure(
                    source=str(source),
                    error=type(error).__name__,
                    message=str(error),
                )
            )
            manifest.inputs.append(record)
            continue

        filename = filenames[source]
        _write_json(run_directory / filename, transcript.to_json_data())
        result.transcripts.append(transcript)
        manifest.inputs.append(record.with_result(filename, transcript.metadata()))

    manifest.succeeded = len(result.transcripts)
    manifest.failed = len(result.failures)
    manifest.finished_at = utc_now()

    _write_json(run_directory / MANIFEST_FILENAME, manifest.to_dict())
    _write_json(
        run_directory / ERRORS_FILENAME,
        [failure.to_dict() for failure in result.failures],
    )
    return result


def _transcribe_one(backend: TranscriptionBackend, source: Path) -> Transcript:
    """Decode one file, transcribe it, and check the result against the contract.

    The conformance suite proves a backend *can* honor the schema; it cannot
    prove every transcript does. Whisper, for one, hallucinates into the silence
    it pads short audio with and returns segments past the end of the recording.
    Checking here means a backend that breaks the contract produces a reported
    failure rather than a bad transcript that looks like a success.

    Raises:
        AudioDecodeError: If the file cannot be decoded.
        TranscriptionError: If the backend fails, or returns a transcript that
            violates the shared schema.
    """
    with audio.decoded(source) as decoded_path:
        transcript = backend.transcribe(decoded_path, str(source))

    try:
        assert_transcript_conforms(transcript, backend.name, str(source))
    except ConformanceError as error:
        raise TranscriptionError(
            str(source),
            backend.name,
            f"returned a transcript that violates the schema: {error}",
        ) from error
    return transcript


def _write_json(path: Path, data: Any) -> None:
    """Write JSON to a file, newline-terminated and UTF-8."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _recordable(config: dict[str, Any]) -> dict[str, Any]:
    """Return a config safe to serialize into the manifest.

    A backend may carry a value JSON cannot represent, such as a Path. The
    manifest is for reading later, so those become strings rather than failing
    the run at the last step.
    """
    recordable: dict[str, Any] = {}
    for key, value in config.items():
        if value is None or isinstance(value, bool | int | float | str):
            recordable[key] = value
        else:
            recordable[key] = str(value)
    return recordable


def _tool_version() -> str:
    """Return this package's version, for the manifest."""
    from audio_transcriber import __version__

    return __version__
