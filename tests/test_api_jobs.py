"""Tests for the API's job model and in-memory store.

A job is the API's record of one transcription request. Its status only moves
forward, so a client polling it never sees a finished job start again.
"""

import dataclasses
import json
import threading
import time

import pytest

from audio_transcriber.api.jobs import (
    InMemoryJobStore,
    InvalidTransitionError,
    Job,
    JobError,
    JobStatus,
    JobStore,
    UnknownJobError,
)
from audio_transcriber.errors import BackendRequirementError, TranscriptionError

TRANSCRIPT = [
    {
        "start_time": 0.0,
        "end_time": 1.0,
        "speaker": None,
        "spoken_language": "en",
        "text": "hello",
    }
]


def _job() -> Job:
    return Job.new(backend="stub", options={"beam_size": 1}, filename="a.wav")


def test_a_new_job_is_queued_with_a_creation_time():
    job = _job()

    assert job.status is JobStatus.QUEUED
    assert job.id
    assert job.created_at
    assert job.started_at is None
    assert job.finished_at is None
    assert job.transcript is None
    assert job.error is None


def test_new_jobs_get_distinct_ids():
    assert _job().id != _job().id


def test_status_values_are_the_wire_names():
    assert [status.value for status in JobStatus] == [
        "queued",
        "running",
        "succeeded",
        "failed",
    ]


def test_a_job_runs_then_succeeds_with_its_transcript(tmp_path):
    running = _job().start()
    done = running.succeed(TRANSCRIPT, tmp_path / "x.json")

    assert running.status is JobStatus.RUNNING
    assert running.started_at
    assert done.status is JobStatus.SUCCEEDED
    assert done.finished_at
    assert done.transcript == TRANSCRIPT
    assert done.transcript_path == tmp_path / "x.json"


def test_a_job_can_fail_while_running_or_before_it_starts():
    error = JobError(type="TranscriptionError", message="engine said no.")

    assert _job().start().fail(error).status is JobStatus.FAILED
    assert _job().fail(error).status is JobStatus.FAILED
    assert _job().fail(error).error == error


@pytest.mark.parametrize(
    "move",
    [
        lambda job, path: job.succeed(TRANSCRIPT, path),
        lambda job, path: job.start().start(),
        lambda job, path: job.start().succeed(TRANSCRIPT, path).start(),
        lambda job, path: job.start()
        .succeed(TRANSCRIPT, path)
        .fail(JobError("X", "late")),
        lambda job, path: job.fail(JobError("X", "no")).start(),
    ],
    ids=[
        "queued-to-succeeded",
        "running-to-running",
        "succeeded-to-running",
        "succeeded-to-failed",
        "failed-to-running",
    ],
)
def test_illegal_transitions_are_refused(move, tmp_path):
    with pytest.raises(InvalidTransitionError):
        move(_job(), tmp_path / "x.json")


def test_transitions_leave_the_original_job_unchanged():
    job = _job()
    job.start()

    assert job.status is JobStatus.QUEUED


def test_error_from_an_exception_keeps_type_message_and_remedy():
    error = JobError.from_exception(
        BackendRequirementError("stub", "no API key.", "Set STUB_KEY in .env.")
    )

    assert error.type == "BackendRequirementError"
    assert "no API key." in error.message
    assert error.remedy == "Set STUB_KEY in .env."


def test_error_from_an_exception_without_a_remedy_has_an_empty_one():
    error = JobError.from_exception(TranscriptionError("a.wav", "stub", "boom."))

    assert error.type == "TranscriptionError"
    assert error.remedy == ""


def test_a_succeeded_job_serializes_to_json(tmp_path):
    job = _job().start().succeed(TRANSCRIPT, tmp_path / "x.json")

    data = json.loads(json.dumps(job.to_dict()))

    assert data["id"] == job.id
    assert data["status"] == "succeeded"
    assert data["backend"] == "stub"
    assert data["options"] == {"beam_size": 1}
    assert data["filename"] == "a.wav"
    assert data["transcript"] == TRANSCRIPT
    assert data["error"] is None
    assert {"created_at", "started_at", "finished_at"} <= data.keys()
    assert "transcript_path" not in data, "server paths are not part of the API"


def test_a_failed_job_serializes_its_structured_error():
    job = _job().fail(JobError("AudioDecodeError", "cannot decode.", "Re-encode."))

    data = json.loads(json.dumps(job.to_dict()))

    assert data["status"] == "failed"
    assert data["transcript"] is None
    assert data["error"] == {
        "type": "AudioDecodeError",
        "message": "cannot decode.",
        "remedy": "Re-encode.",
    }


def test_the_in_memory_store_satisfies_the_protocol():
    assert isinstance(InMemoryJobStore(), JobStore)


def test_the_store_creates_gets_and_updates_jobs():
    store = InMemoryJobStore()
    job = store.create(_job())

    assert store.get(job.id) == job
    updated = store.update(job.id, Job.start)
    assert updated.status is JobStatus.RUNNING
    assert store.get(job.id) == updated


def test_the_store_returns_none_for_an_unknown_job():
    assert InMemoryJobStore().get("nope") is None


def test_updating_an_unknown_job_raises():
    with pytest.raises(UnknownJobError):
        InMemoryJobStore().update("nope", Job.start)


def test_a_refused_update_leaves_the_stored_job_unchanged():
    store = InMemoryJobStore()
    job = store.create(_job())

    with pytest.raises(InvalidTransitionError):
        store.update(job.id, lambda j: j.succeed(TRANSCRIPT, None))
    assert store.get(job.id) == job


def test_concurrent_updates_are_serialized():
    """Each update reads the latest job, so none is lost to a race."""
    store = InMemoryJobStore()
    job = store.create(dataclasses.replace(_job(), options={"n": 0}))

    def bump(current: Job) -> Job:
        seen = current.options["n"]
        time.sleep(0.001)  # widen the read-modify-write window
        return dataclasses.replace(current, options={"n": seen + 1})

    threads = [
        threading.Thread(target=lambda: store.update(job.id, bump)) for _ in range(20)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert store.get(job.id).options == {"n": 20}
