"""Tests for the job runner: the pipeline, run in the background, one job a file.

These run the real pipeline over the stub backend and stub decoder.
"""

import json
import time
import uuid
from pathlib import Path

import pytest

from audio_transcriber.api.jobs import InMemoryJobStore, Job, JobStatus
from audio_transcriber.api.runner import JobRunner
from audio_transcriber.api.settings import ApiSettings
from tests import api_stub


@pytest.fixture
def settings(tmp_path) -> ApiSettings:
    return ApiSettings(work_dir=tmp_path / "work")


@pytest.fixture
def store() -> InMemoryJobStore:
    return InMemoryJobStore()


@pytest.fixture
def runner(store, settings):
    runner = JobRunner(store, settings)
    yield runner
    api_stub.StubBackend.gate.set()
    runner.shutdown()


def _upload(settings: ApiSettings, content: bytes = b"fake audio") -> Path:
    path = settings.uploads_dir / uuid.uuid4().hex / "meeting.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _wait_for(store, job_id: str, *statuses: JobStatus, timeout: float = 5.0) -> Job:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = store.get(job_id)
        if job.status in statuses:
            return job
        time.sleep(0.01)
    raise AssertionError(f"job stayed {store.get(job_id).status}")


def _submit(runner, settings, options=None, content=b"fake audio") -> Job:
    job = Job.new(backend="stub", options=options or {}, filename="meeting.wav")
    return runner.submit(job, _upload(settings, content))


def test_a_submitted_job_is_stored_queued(stub, runner, store, settings):
    job = _submit(runner, settings, {"wait": True})

    assert job.status is JobStatus.QUEUED
    assert store.get(job.id) is not None


def test_a_job_runs_then_succeeds_with_the_transcript_the_cli_writes(
    stub, runner, store, settings
):
    job = _submit(runner, settings, {"wait": True})

    _wait_for(store, job.id, JobStatus.RUNNING)
    stub.gate.set()
    done = _wait_for(store, job.id, JobStatus.SUCCEEDED, JobStatus.FAILED)

    assert done.status is JobStatus.SUCCEEDED, done.error
    written = json.loads(done.transcript_path.read_text(encoding="utf-8"))
    assert done.transcript == written
    assert written[0]["text"] == "hello world"
    assert done.transcript_path.name == "meeting.json"


def test_a_job_keeps_its_run_directory_with_the_manifest(stub, runner, store, settings):
    job = _submit(runner, settings)

    done = _wait_for(store, job.id, JobStatus.SUCCEEDED, JobStatus.FAILED)

    run_directory = done.transcript_path.parent
    assert run_directory == settings.runs_dir / job.id
    manifest = json.loads((run_directory / "manifest.json").read_text())
    assert manifest["backend"] == "stub"
    assert manifest["model_version"] == "stub-1.0"


def test_jobs_run_one_at_a_time_by_default(stub, runner, store, settings):
    first = _submit(runner, settings, {"wait": True})
    second = _submit(runner, settings, {"wait": True})

    _wait_for(store, first.id, JobStatus.RUNNING)
    time.sleep(0.05)
    assert store.get(second.id).status is JobStatus.QUEUED

    stub.gate.set()
    _wait_for(store, second.id, JobStatus.SUCCEEDED)


def test_concurrency_comes_from_the_settings(stub, store, tmp_path):
    settings = ApiSettings(work_dir=tmp_path / "work", concurrency=2)
    runner = JobRunner(store, settings)
    try:
        first = _submit(runner, settings, {"wait": True})
        second = _submit(runner, settings, {"wait": True})

        _wait_for(store, first.id, JobStatus.RUNNING)
        _wait_for(store, second.id, JobStatus.RUNNING)
    finally:
        stub.gate.set()
        runner.shutdown()


@pytest.mark.parametrize(
    ("options", "content", "error_type", "remedy"),
    [
        ({}, api_stub.UNDECODABLE, "AudioDecodeError", ""),
        ({"fail": "requirement"}, b"fake audio", "BackendRequirementError", "Set"),
        ({"fail": "transcription"}, b"fake audio", "TranscriptionError", ""),
    ],
    ids=["decode", "requirement", "transcription"],
)
def test_a_failure_is_recorded_on_the_job_with_a_structured_error(
    stub, runner, store, settings, options, content, error_type, remedy
):
    job = _submit(runner, settings, options, content)

    done = _wait_for(store, job.id, JobStatus.SUCCEEDED, JobStatus.FAILED)

    assert done.status is JobStatus.FAILED
    assert done.error.type == error_type
    assert done.error.message
    assert done.error.remedy.startswith(remedy)
    assert done.transcript is None


def test_an_unexpected_exception_fails_the_job_rather_than_hanging_it(
    stub, runner, store, settings, monkeypatch
):
    def _explode(**kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr("audio_transcriber.api.runner.pipeline.run", _explode)
    job = _submit(runner, settings)

    done = _wait_for(store, job.id, JobStatus.SUCCEEDED, JobStatus.FAILED)

    assert done.status is JobStatus.FAILED
    assert done.error.type == "RuntimeError"
    assert "disk on fire" in done.error.message


@pytest.mark.parametrize(
    ("options", "content"),
    [
        ({}, b"fake audio"),
        ({}, api_stub.UNDECODABLE),
        ({"fail": "requirement"}, b"fake audio"),
        ({"fail": "transcription"}, b"fake audio"),
    ],
    ids=["succeeded", "decode", "requirement", "transcription"],
)
def test_the_upload_is_deleted_in_every_outcome(
    stub, runner, store, settings, options, content
):
    upload = _upload(settings, content)
    job = runner.submit(
        Job.new(backend="stub", options=options, filename="meeting.wav"), upload
    )

    _wait_for(store, job.id, JobStatus.SUCCEEDED, JobStatus.FAILED)

    assert not upload.exists()
    assert not upload.parent.exists()
