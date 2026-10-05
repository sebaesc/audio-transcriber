"""Contract tests for scoring a finished job against a ground truth.

Scoring reads the transcript the job already wrote. It must never run a
backend: on a hosted one that would spend credits again.
"""

import json

import pytest

pytest.importorskip("fastapi", reason="the api extra is not installed")
pytest.importorskip("jiwer", reason="the eval extra is not installed")

from audio_transcriber import registry  # noqa: E402
from audio_transcriber.evaluation.reference import Pair  # noqa: E402
from audio_transcriber.evaluation.runner import score_transcripts  # noqa: E402
from tests import api_stub  # noqa: E402
from tests.api_fixtures import (  # noqa: E402
    assert_error_body,
    submit,
    wait_until_finished,
)

#: Matches the stub's `hello world` except one word, without speakers, so
#: scoring needs neither diarization metrics nor a perfect transcript.
GROUND_TRUTH = [{"start_time": 0.0, "end_time": 1.0, "text": "hello there"}]


def evaluate(client, job_id: str, ground_truth=GROUND_TRUTH, raw: bytes | None = None):
    content = raw if raw is not None else json.dumps(ground_truth).encode()
    return client.post(
        f"/transcriptions/{job_id}/evaluation",
        files={"ground_truth": ("meeting.json", content, "application/json")},
    )


@pytest.fixture
def finished_job(client) -> dict:
    job_id = submit(client, name="meeting.wav").json()["id"]
    job = wait_until_finished(client, job_id)
    assert job["status"] == "succeeded"
    return job


def test_a_succeeded_job_is_scored_against_the_ground_truth(client, finished_job):
    response = evaluate(client, finished_job["id"])

    assert response.status_code == 200
    report = response.json()
    assert report["summary"]["scored"] == 1
    assert report["summary"]["mean_wer"] == 0.5
    assert report["backend"] == "stub", "read from the job's manifest"
    assert report["model_version"] == "stub-1.0"
    assert report["corpus"] == {"dataset": "custom ground truth"}
    (recording,) = report["recordings"]
    assert recording["name"] == "meeting"
    assert recording["score"]["wer"] == 0.5


def test_scoring_never_runs_a_backend(client, finished_job, stub, monkeypatch):
    calls_before = list(stub.calls)

    def _no_backends(*args, **kwargs):
        raise AssertionError("scoring built a backend")

    monkeypatch.setattr(registry, "create", _no_backends)
    monkeypatch.setattr(registry, "get", _no_backends)

    assert evaluate(client, finished_job["id"]).status_code == 200
    assert stub.calls == calls_before


def test_the_scores_equal_the_evaluation_clis(client, finished_job, settings, tmp_path):
    """Phase 3 verification: same transcript and ground truth, same scores."""
    reference = tmp_path / "meeting.json"
    reference.write_text(json.dumps(GROUND_TRUTH), encoding="utf-8")
    transcript = settings.runs_dir / finished_job["id"] / "meeting.json"
    expected = score_transcripts(
        label=finished_job["id"],
        pairs=[Pair(name="meeting", media=transcript, reference=reference)],
    ).to_dict()

    actual = evaluate(client, finished_job["id"]).json()

    for report in (expected, actual):
        report.pop("created_at")
        report["recordings"][0]["reference"].pop("file")
    assert actual == expected


@pytest.mark.parametrize(
    ("options", "content"),
    [({"wait": True}, b"fake audio"), ({}, api_stub.UNDECODABLE)],
    ids=["unfinished", "failed"],
)
def test_a_job_without_a_transcript_is_a_409(client, stub, options, content):
    job_id = submit(client, options=options, content=content).json()["id"]
    if not options:
        wait_until_finished(client, job_id)

    response = evaluate(client, job_id)

    assert response.status_code == 409
    assert_error_body(response, "JobNotScorableError")
    stub.gate.set()


def test_an_unknown_job_is_a_404(client):
    response = evaluate(client, "does-not-exist")

    assert response.status_code == 404
    assert_error_body(response, "UnknownJobError")


@pytest.mark.parametrize(
    ("raw", "fragment"),
    [
        (b"not json", "not valid JSON"),
        (b'{"text": "hi"}', "JSON array"),
        (b"[]", "empty"),
        (b'[{"end_time": 1, "text": "hi"}]', "segment 0"),
    ],
    ids=["not-json", "object", "empty", "bad-segment"],
)
def test_an_invalid_ground_truth_is_a_422(client, finished_job, raw, fragment):
    response = evaluate(client, finished_job["id"], raw=raw)

    assert response.status_code == 422
    assert fragment in assert_error_body(response, "ConfigError")["message"]


def test_the_uploaded_ground_truth_is_removed_after_scoring(
    client, finished_job, settings
):
    evaluate(client, finished_job["id"])
    evaluate(client, finished_job["id"], raw=b"not json")

    assert not any(settings.uploads_dir.rglob("*.json"))


def test_a_missing_scoring_extra_is_a_501_saying_what_to_install(
    client, finished_job, monkeypatch
):
    def _missing(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'jiwer'", name="jiwer")

    monkeypatch.setattr("audio_transcriber.api.scoring.score_transcripts", _missing)

    response = evaluate(client, finished_job["id"])

    assert response.status_code == 501
    error = assert_error_body(response, "ScoringUnavailableError")
    assert '".[eval]"' in error["remedy"]


def test_speakers_are_scored_when_the_ground_truth_labels_them(client, finished_job):
    pytest.importorskip("pyannote.metrics", reason="the diarization extra is absent")
    labelled = [dict(GROUND_TRUTH[0], speaker="Ana")]

    report = evaluate(client, finished_job["id"], labelled).json()

    (recording,) = report["recordings"]
    assert recording["ground_truth"]["diarization"]["der"] == 0.0
