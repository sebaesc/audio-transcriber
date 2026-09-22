"""Running an evaluation over the reference corpus.

The runner takes a backend name and config, transcribes every recording in the
corpus, scores each one, and writes a JSON report. Two reports can then be
compared to judge whether a change helped.

It goes through the registry and the backend interface, exactly as the pipeline
does, so what is measured is what a run would produce.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from audio_transcriber import __version__, audio, registry
from audio_transcriber.evaluation import corpus
from audio_transcriber.evaluation.scoring import Comparison, Score, score
from audio_transcriber.evaluation.text import clean_reference, normalize
from audio_transcriber.manifest import utc_now

_logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RecordingResult:
    """One recording's outcome.

    Attributes:
        name: Recording identifier.
        duration_s: Length of the audio.
        elapsed_s: Wall-clock time spent transcribing it.
        detected_languages: The distinct `spoken_language` values produced.
        segments: Number of segments in the transcript.
        score: How the transcript compared to the reference, or None if it failed.
        error: Why it failed, if it did.
    """

    name: str
    duration_s: float
    elapsed_s: float = 0.0
    detected_languages: list[str] = field(default_factory=list)
    segments: int = 0
    score: Score | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the result as JSON-serializable data."""
        return {
            "name": self.name,
            "duration_s": round(self.duration_s, 1),
            "elapsed_s": round(self.elapsed_s, 1),
            "realtime_factor": (
                round(self.duration_s / self.elapsed_s, 1) if self.elapsed_s else None
            ),
            "segments": self.segments,
            "detected_languages": self.detected_languages,
            "score": self.score.to_dict() if self.score else None,
            "error": self.error,
        }


@dataclass(slots=True)
class Report:
    """The outcome of one evaluation.

    Attributes:
        label: A name for this configuration, used when comparing reports.
        backend: Backend name.
        model_version: Model identifier the backend reported.
        config: Effective backend config.
        results: One entry per recording.
        created_at: When the evaluation ran.
        tool_version: Version of audio-transcriber used.
    """

    label: str
    backend: str
    model_version: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    results: list[RecordingResult] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    tool_version: str = __version__

    @property
    def scored(self) -> list[RecordingResult]:
        """Recordings that produced a transcript."""
        return [item for item in self.results if item.score is not None]

    def mean(self, attribute: str) -> float:
        """Return the mean of a score attribute across scored recordings.

        Args:
            attribute: A `Score` attribute or property name, e.g. `"wer"`.

        Returns:
            The mean, or 0.0 when nothing was scored.
        """
        values = [getattr(item.score, attribute) for item in self.scored]
        return sum(values) / len(values) if values else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return the report as JSON-serializable data."""
        return {
            "label": self.label,
            "created_at": self.created_at,
            "tool_version": self.tool_version,
            "backend": self.backend,
            "model_version": self.model_version,
            "config": self.config,
            "corpus": {"dataset": corpus.DATASET, "license": corpus.LICENSE},
            "summary": {
                "recordings": len(self.results),
                "scored": len(self.scored),
                "mean_wer": round(self.mean("wer"), 4),
                "mean_coverage": round(self.mean("coverage"), 4),
                "mean_language_mix_error": round(self.mean("language_mix_error"), 4),
            },
            "recordings": [item.to_dict() for item in self.results],
        }

    def write(self, path: Path) -> Path:
        """Write the report as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        return path


def evaluate(
    backend_name: str,
    label: str,
    backend_config: dict[str, Any] | None = None,
    only: list[str] | None = None,
    transcript_dir: Path | None = None,
) -> Report:
    """Transcribe and score every recording in the corpus.

    Args:
        backend_name: Backend to evaluate, resolved through the registry.
        label: A name for this configuration, e.g. `"baseline"`.
        backend_config: Backend options.
        only: Recording names to limit the run to. None evaluates all of them.
        transcript_dir: Where to write each hypothesis, for inspecting what the
            scores are actually measuring. None skips writing them.

    Returns:
        The report.

    Raises:
        ConfigError: If the corpus cannot be fetched.
        BackendRequirementError: If the backend is not ready to run.
    """
    backend = registry.create(backend_name, **(backend_config or {}))
    audio.check_ffmpeg()
    backend.check_requirements()

    recordings = corpus.load()
    if only:
        wanted = set(only)
        recordings = [item for item in recordings if item.name in wanted]

    report = Report(
        label=label,
        backend=backend.name,
        model_version=backend.model_version(),
        config=_recordable(backend.effective_config()),
    )

    for recording in recordings:
        _logger.info(
            "evaluating %s (%.1f min)", recording.name, recording.duration_s / 60
        )
        result = RecordingResult(name=recording.name, duration_s=recording.duration_s)
        started = time.monotonic()
        try:
            with audio.decoded(recording.audio_path) as decoded:
                transcript = backend.transcribe(decoded, str(recording.audio_path))
        except Exception as error:  # noqa: BLE001 - one recording failing is not fatal
            result.error = f"{type(error).__name__}: {error}"
            result.elapsed_s = time.monotonic() - started
            report.results.append(result)
            continue

        result.elapsed_s = time.monotonic() - started
        result.segments = len(transcript.segments)
        result.detected_languages = sorted(
            {s.spoken_language for s in transcript.segments if s.spoken_language}
        )
        hypothesis = normalize(transcript.text)
        result.score = score(clean_reference(recording.transcript), hypothesis)
        report.results.append(result)

        if transcript_dir is not None:
            transcript_dir.mkdir(parents=True, exist_ok=True)
            (transcript_dir / f"{recording.name}.{label}.txt").write_text(
                hypothesis, encoding="utf-8"
            )

    return report


def compare(baseline: Report, candidate: Report) -> dict[str, Any]:
    """Compare two reports, recording by recording.

    A change is judged per recording, not on the mean. The chunked prototype
    improved one recording by 23% and made another 36% worse; a mean would have
    hidden both.

    Args:
        baseline: The report to beat.
        candidate: The report being judged.

    Returns:
        JSON-serializable comparison data.
    """
    by_name = {item.name: item for item in candidate.scored}
    per_recording: dict[str, Any] = {}
    improved = 0

    for item in baseline.scored:
        other = by_name.get(item.name)
        if other is None or item.score is None or other.score is None:
            continue
        comparison = Comparison(baseline=item.score, candidate=other.score)
        per_recording[item.name] = comparison.to_dict()
        improved += int(comparison.improved)

    return {
        "baseline_label": baseline.label,
        "candidate_label": candidate.label,
        "recordings_compared": len(per_recording),
        "recordings_improved": improved,
        "improved_everywhere": bool(per_recording) and improved == len(per_recording),
        "mean_wer_delta": round(candidate.mean("wer") - baseline.mean("wer"), 4),
        "mean_coverage_delta": round(
            candidate.mean("coverage") - baseline.mean("coverage"), 4
        ),
        "mean_language_mix_error_delta": round(
            candidate.mean("language_mix_error") - baseline.mean("language_mix_error"),
            4,
        ),
        "per_recording": per_recording,
    }


def _recordable(config: dict[str, Any]) -> dict[str, Any]:
    """Return a config safe to serialize into a report."""
    return {
        key: value
        if value is None or isinstance(value, bool | int | float | str)
        else str(value)
        for key, value in config.items()
    }
