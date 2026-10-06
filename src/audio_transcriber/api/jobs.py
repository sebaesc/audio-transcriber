"""Jobs: the API's record of one transcription request.

A job is immutable. Each status change returns a new job, and only forward
moves are legal: queued → running → succeeded, or failed from either of the
first two. A client polling a job therefore never sees a finished one restart.

Jobs live in a `JobStore`. The in-memory store here loses them on restart; the
protocol is the seam a persistent store replaces it through.
"""

from __future__ import annotations

import dataclasses
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, Self, runtime_checkable

from audio_transcriber.manifest import utc_now


class JobStatus(StrEnum):
    """Where a job is in its life. The values are the names on the wire."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.QUEUED: frozenset({JobStatus.RUNNING, JobStatus.FAILED}),
    JobStatus.RUNNING: frozenset({JobStatus.SUCCEEDED, JobStatus.FAILED}),
    JobStatus.SUCCEEDED: frozenset(),
    JobStatus.FAILED: frozenset(),
}


class InvalidTransitionError(RuntimeError):
    """Raised when a job is moved to a status it cannot reach from its own.

    A programming error, not a client one: the runner drives every transition.
    """


class UnknownJobError(LookupError):
    """Raised when a store is asked to update a job it does not hold."""


@dataclass(frozen=True, slots=True)
class JobError:
    """Why a job failed, in the shape the API returns it.

    Attributes:
        type: The error class name, for grouping failures by cause.
        message: What happened and how to fix it.
        remedy: The command or config change that fixes it, when known.
    """

    type: str
    message: str
    remedy: str = ""

    @classmethod
    def from_exception(cls, error: BaseException) -> Self:
        """Describe an exception, keeping its remedy when it carries one."""
        return cls(
            type=type(error).__name__,
            message=str(error),
            remedy=getattr(error, "remedy", "") or "",
        )

    def to_dict(self) -> dict[str, str]:
        """Return the error as JSON-serializable data."""
        return {"type": self.type, "message": self.message, "remedy": self.remedy}


@dataclass(frozen=True, slots=True)
class Job:
    """One transcription request and its outcome.

    Attributes:
        id: Opaque identifier, unique per job.
        backend: The backend name the client asked for.
        options: Backend options, as the client sent them.
        filename: The uploaded file's name, as the client sent it.
        status: Where the job is.
        created_at: When it was submitted, ISO 8601 UTC.
        started_at: When it started running, or None.
        finished_at: When it succeeded or failed, or None.
        transcript: The transcript, as the CLI writes it, once succeeded.
        transcript_path: Where that transcript is on disk, beside its run's
            manifest. Server-side only: it is not part of the API response.
        error: Why it failed, once failed.
    """

    id: str
    backend: str
    options: dict[str, Any]
    filename: str
    status: JobStatus = JobStatus.QUEUED
    created_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    finished_at: str | None = None
    transcript: list[dict[str, Any]] | None = None
    transcript_path: Path | None = None
    error: JobError | None = None

    @classmethod
    def new(cls, backend: str, options: dict[str, Any], filename: str) -> Self:
        """Return a queued job with a fresh id."""
        return cls(
            id=uuid.uuid4().hex, backend=backend, options=options, filename=filename
        )

    def start(self) -> Job:
        """Return this job, running."""
        return self._move(JobStatus.RUNNING, started_at=utc_now())

    def succeed(
        self, transcript: list[dict[str, Any]], transcript_path: Path | None
    ) -> Job:
        """Return this job, succeeded with its transcript."""
        return self._move(
            JobStatus.SUCCEEDED,
            finished_at=utc_now(),
            transcript=transcript,
            transcript_path=transcript_path,
        )

    def fail(self, error: JobError) -> Job:
        """Return this job, failed with why."""
        return self._move(JobStatus.FAILED, finished_at=utc_now(), error=error)

    def _move(self, status: JobStatus, **changes: Any) -> Job:
        if status not in _TRANSITIONS[self.status]:
            raise InvalidTransitionError(
                f"job {self.id} cannot move from {self.status} to {status}"
            )
        return dataclasses.replace(self, status=status, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Return the job as the API serves it."""
        return {
            "id": self.id,
            "status": self.status.value,
            "backend": self.backend,
            "options": self.options,
            "filename": self.filename,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "transcript": self.transcript,
            "error": self.error.to_dict() if self.error else None,
        }


@runtime_checkable
class JobStore(Protocol):
    """Where jobs are kept. Implementations must be safe across threads."""

    def create(self, job: Job) -> Job:
        """Store a new job and return it."""
        ...

    def get(self, job_id: str) -> Job | None:
        """Return the job with this id, or None."""
        ...

    def update(self, job_id: str, change: Callable[[Job], Job]) -> Job:
        """Replace a job with `change(job)`, atomically, and return the result.

        Raises:
            UnknownJobError: If there is no job with this id.
        """
        ...


class InMemoryJobStore:
    """A `JobStore` in a dict. Jobs are lost when the process exits."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, job: Job) -> Job:
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, change: Callable[[Job], Job]) -> Job:
        with self._lock:
            try:
                current = self._jobs[job_id]
            except KeyError:
                raise UnknownJobError(job_id) from None
            updated = change(current)
            self._jobs[job_id] = updated
            return updated
