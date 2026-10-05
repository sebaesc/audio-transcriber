"""Fixtures and helpers shared by the HTTP API tests.

Loaded as a pytest plugin from `conftest.py`. The web framework is imported only
inside the fixtures, so the rest of the suite runs without the `api` extra.
"""

import json
import time

import pytest

from audio_transcriber.api.jobs import InMemoryJobStore
from audio_transcriber.api.settings import ApiSettings
from tests import api_stub


@pytest.fixture
def stub(monkeypatch):
    return api_stub.install(monkeypatch)


@pytest.fixture
def settings(tmp_path) -> ApiSettings:
    return ApiSettings(work_dir=tmp_path / "work", max_upload_bytes=1024)


@pytest.fixture
def client(stub, settings):
    pytest.importorskip("fastapi", reason="the api extra is not installed")
    from fastapi.testclient import TestClient

    from audio_transcriber.api.app import create_app

    with TestClient(create_app(settings, InMemoryJobStore())) as client:
        yield client
        stub.gate.set()


def submit(client, backend="stub", options=None, content=b"fake audio", name="a.wav"):
    data = {"backend": backend}
    if options is not None:
        data["options"] = json.dumps(options)
    return client.post(
        "/transcriptions", data=data, files={"file": (name, content, "audio/wav")}
    )


def wait_until_finished(client, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/transcriptions/{job_id}").json()
        if body["status"] in ("succeeded", "failed"):
            return body
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} never finished")


def assert_error_body(response, error_type: str) -> dict:
    error = response.json()["error"]
    assert error["type"] == error_type
    assert error["message"]
    assert "remedy" in error
    return error
