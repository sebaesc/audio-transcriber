"""Running jobs: the pipeline, in the background, one upload per job.

Backends block, so jobs run on a thread pool rather than the event loop. The
pool is sized by the settings, one job at a time by default, so a local model
does not overcommit the CPU or GPU.

A job is an ordinary pipeline run over a single file. Its run directory, with
the transcript and `manifest.json`, is kept under the settings' runs directory,
so the job can be scored later without transcribing it again.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from audio_transcriber import pipeline
from audio_transcriber.api.jobs import Job, JobError, JobStore
from audio_transcriber.api.settings import ApiSettings
from audio_transcriber.errors import AudioTranscriberError

_logger = logging.getLogger(__name__)


class JobRunner:
    """Executes jobs on a thread pool and records their outcome in a store.

    Args:
        store: Where jobs are recorded.
        settings: Pool size and where runs are written.
    """

    def __init__(self, store: JobStore, settings: ApiSettings) -> None:
        self._store = store
        self._settings = settings
        self._executor = ThreadPoolExecutor(
            max_workers=settings.concurrency, thread_name_prefix="transcription"
        )

    def submit(self, job: Job, upload: Path) -> Job:
        """Store a queued job and schedule it.

        Args:
            job: The queued job.
            upload: The uploaded audio. The runner owns its directory from now
                on and removes it when the job finishes, however it finishes.

        Returns:
            The job, as stored.
        """
        stored = self._store.create(job)
        self._executor.submit(self._execute, job.id, upload)
        return stored

    def shutdown(self, wait: bool = True) -> None:
        """Stop taking jobs, and by default wait for those already submitted."""
        self._executor.shutdown(wait=wait, cancel_futures=not wait)

    def _execute(self, job_id: str, upload: Path) -> None:
        try:
            job = self._store.update(job_id, Job.start)
            self._store.update(job_id, self._outcome(job, upload))
        except Exception as error:  # noqa: BLE001 - a job must never hang
            if not isinstance(error, AudioTranscriberError):
                _logger.exception("job %s failed unexpectedly", job_id)
            failure = JobError.from_exception(error)
            self._store.update(job_id, lambda current: current.fail(failure))
        finally:
            shutil.rmtree(upload.parent, ignore_errors=True)

    def _outcome(self, job: Job, upload: Path) -> Callable[[Job], Job]:
        """Run the pipeline for a job, and return how to record the result.

        Raises:
            AudioTranscriberError: If the run could not start: a missing
                requirement or an invalid option.
        """
        result = pipeline.run(
            input_path=upload,
            backend_name=job.backend,
            output_root=self._settings.runs_dir,
            backend_config=job.options,
            run_id=job.id,
        )
        if result.failures:
            failure = result.failures[0]
            error = JobError(type=failure.error, message=failure.message)
            return lambda current: current.fail(error)

        record = result.manifest.inputs[0]
        path = result.run_directory / record.transcript
        transcript = result.transcripts[0].to_json_data()
        return lambda current: current.succeed(transcript, path)
