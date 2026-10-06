"""Scoring a finished job against a ground truth, without transcribing again.

The job's transcript is already on disk beside its run's manifest, so this is
the evaluation CLI's existing-transcript mode (`score_transcripts`) on one
pair: the same scores, the same report, and no backend run, so no credits are
spent re-scoring a hosted backend's output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from audio_transcriber.api.jobs import Job, JobStatus
from audio_transcriber.errors import AudioTranscriberError
from audio_transcriber.evaluation.reference import Pair
from audio_transcriber.evaluation.runner import score_transcripts

#: The extra that provides each optional scoring library.
_EXTRAS = {"jiwer": "eval", "pyannote": "diarization"}


class JobNotScorableError(AudioTranscriberError):
    """Raised when a job has no transcript to score: unfinished, or failed."""

    def __init__(self, job: Job) -> None:
        if job.status is JobStatus.FAILED:
            self.remedy = "Submit the audio again; a failed job has no transcript."
            reason = "failed"
        else:
            self.remedy = "Poll the job until its status is succeeded."
            reason = f"is still {job.status.value}"
        super().__init__(f"job {job.id} {reason}, so there is no transcript to score.")


class ScoringUnavailableError(AudioTranscriberError):
    """Raised when a library scoring needs is not installed on the server."""

    def __init__(self, error: ModuleNotFoundError) -> None:
        module = (error.name or "").partition(".")[0]
        extra = _EXTRAS.get(module, "eval")
        self.remedy = f'Install it on the server: pip install -e ".[{extra}]"'
        super().__init__(f"scoring needs {module or 'a module'!r}, not installed.")


def score_job(job: Job, reference: Path) -> dict[str, Any]:
    """Score a succeeded job's transcript against a ground-truth file.

    Args:
        job: The job. Its id is the report's label.
        reference: The ground truth, in the evaluation CLI's format.

    Returns:
        The report, as the evaluation CLI writes it.

    Raises:
        JobNotScorableError: If the job has not succeeded.
        ConfigError: If the ground truth is invalid.
        ScoringUnavailableError: If a scoring library is not installed.
    """
    if job.status is not JobStatus.SUCCEEDED or job.transcript_path is None:
        raise JobNotScorableError(job)
    pair = Pair(
        name=Path(job.filename).stem, media=job.transcript_path, reference=reference
    )
    try:
        report = score_transcripts(label=job.id, pairs=[pair])
    except ModuleNotFoundError as error:
        raise ScoringUnavailableError(error) from error
    return report.to_dict()
