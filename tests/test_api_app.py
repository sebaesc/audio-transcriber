"""Contract tests for the HTTP API's transcription, backend and health routes.

They drive the app through FastAPI's test client, over the real runner and
pipeline, with the stub backend and decoder. No real engine is ever built.
Fixtures (`stub`, `settings`, `client`) come from `tests/api_fixtures.py`.
"""

import json

import pytest

from tests import api_stub
from tests.api_fixtures import assert_error_body, submit, wait_until_finished

# POST /transcriptions


def test_submitting_returns_202_with_the_job_id_and_its_location(client):
    response = submit(client, options={"wait": True})

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert response.headers["location"] == f"/transcriptions/{body['id']}"


def test_options_default_to_empty(client, stub):
    response = submit(client)

    assert wait_until_finished(client, response.json()["id"])["options"] == {}


def test_an_unknown_backend_is_a_400_and_nothing_is_queued(client, settings):
    response = submit(client, backend="nope")

    assert response.status_code == 400
    error = assert_error_body(response, "UnknownBackendError")
    assert "stub" in error["message"], "lists the available backends"
    assert not any(settings.runs_dir.glob("*"))


def test_an_uninstalled_backend_is_a_400_saying_what_to_install(client):
    response = submit(client, backend="missing")

    assert response.status_code == 400
    error = assert_error_body(response, "BackendNotInstalledError")
    assert "pip install" in error["message"]


def test_an_unknown_backend_option_is_a_400_with_the_valid_ones(client):
    response = submit(client, options={"bogus": 1})

    assert response.status_code == 400
    error = assert_error_body(response, "BackendRequirementError")
    assert "fail" in error["remedy"]


@pytest.mark.parametrize("options", ["not json", "[1, 2]", "3"])
def test_options_that_are_not_a_json_object_are_a_400(client, options):
    response = client.post(
        "/transcriptions",
        data={"backend": "stub", "options": options},
        files={"file": ("a.wav", b"fake audio", "audio/wav")},
    )

    assert response.status_code == 400
    assert "JSON object" in assert_error_body(response, "ConfigError")["message"]


def test_a_missing_file_is_a_422_in_the_same_error_shape(client):
    response = client.post("/transcriptions", data={"backend": "stub"})

    assert response.status_code == 422
    assert "file" in assert_error_body(response, "RequestValidationError")["message"]


def test_an_oversize_upload_is_a_413_and_is_not_kept(client, settings):
    response = submit(client, content=b"x" * 2048)

    assert response.status_code == 413
    assert_error_body(response, "UploadTooLargeError")
    assert not any(settings.uploads_dir.rglob("*.wav"))


def test_an_oversize_content_length_is_refused_before_reading(client):
    response = client.post(
        "/transcriptions",
        content=b"x" * (2 * 1024 * 1024),
        headers={"content-type": "multipart/form-data; boundary=x"},
    )

    assert response.status_code == 413


def test_an_upload_name_cannot_escape_the_upload_directory(client, settings):
    response = submit(client, name="../../evil.wav")

    job = wait_until_finished(client, response.json()["id"])
    assert job["status"] == "succeeded"
    assert job["filename"] == "evil.wav"
    assert not (settings.work_dir / "evil.wav").exists()


# GET /transcriptions/{id}


def test_a_job_is_queued_or_running_until_it_finishes(client, stub):
    job_id = submit(client, options={"wait": True}).json()["id"]

    assert client.get(f"/transcriptions/{job_id}").json()["status"] in (
        "queued",
        "running",
    )
    stub.gate.set()
    assert wait_until_finished(client, job_id)["status"] == "succeeded"


def test_a_succeeded_job_carries_the_transcript_the_cli_writes(client, settings):
    job_id = submit(client, name="meeting.wav").json()["id"]

    body = wait_until_finished(client, job_id)

    written = settings.runs_dir / job_id / "meeting.json"
    assert body["status"] == "succeeded"
    assert body["transcript"] == json.loads(written.read_text(encoding="utf-8"))
    assert body["transcript"] == [
        {
            "start_time": 0.0,
            "end_time": 1.0,
            "speaker": "SPEAKER_00",
            "spoken_language": "en",
            "text": "hello world",
        }
    ]
    assert body["error"] is None
    assert body["backend"] == "stub"
    assert body["filename"] == "meeting.wav"


@pytest.mark.parametrize(
    ("options", "content", "error_type"),
    [
        ({}, api_stub.UNDECODABLE, "AudioDecodeError"),
        ({"fail": "requirement"}, b"fake audio", "BackendRequirementError"),
        ({"fail": "transcription"}, b"fake audio", "TranscriptionError"),
    ],
    ids=["decode", "credentials", "transcription"],
)
def test_an_in_job_failure_is_reported_on_the_job(client, options, content, error_type):
    response = submit(client, options=options, content=content)

    assert response.status_code == 202
    body = wait_until_finished(client, response.json()["id"])
    assert body["status"] == "failed"
    assert body["transcript"] is None
    assert body["error"]["type"] == error_type
    assert body["error"]["message"]


def test_an_unknown_job_is_a_404(client):
    response = client.get("/transcriptions/does-not-exist")

    assert response.status_code == 404
    assert_error_body(response, "UnknownJobError")


# GET /backends and /health


def test_backends_lists_the_installed_ones_and_why_the_others_are_not(client):
    body = client.get("/backends").json()

    assert body["available"] == ["stub"]
    assert body["unavailable"] == {
        "missing": "not installed: it needs 'x'. Install its extra: pip install -e \".[x]\""
    }


def test_health_is_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_the_api_transcript_equals_the_one_the_cli_writes(client, tmp_path):
    """Phase 2 verification: same stub output, same transcript JSON."""
    from typer.testing import CliRunner

    from audio_transcriber.cli import app as cli_app

    audio = tmp_path / "cli-in" / "meeting.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"fake audio")
    result = CliRunner().invoke(
        cli_app,
        [
            "--input",
            str(audio),
            "--backend",
            "stub",
            "--out",
            str(tmp_path / "cli-out"),
        ],
    )
    assert result.exit_code == 0, result.output
    (cli_file,) = (tmp_path / "cli-out").glob("*/meeting.json")

    job_id = submit(client, name="meeting.wav").json()["id"]
    body = wait_until_finished(client, job_id)

    assert body["transcript"] == json.loads(cli_file.read_text(encoding="utf-8"))
