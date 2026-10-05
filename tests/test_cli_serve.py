"""CLI contract tests for `audio-transcriber serve`.

The server is never started: `uvicorn.run` is replaced, and what it would have
been given is checked instead.
"""

import sys

import pytest
from typer.testing import CliRunner

from audio_transcriber.cli import ExitCode, app

runner = CliRunner()


@pytest.fixture
def served(monkeypatch, tmp_path):
    """Capture what `serve` would run, without listening on a port."""
    uvicorn = pytest.importorskip("uvicorn", reason="the api extra is not installed")
    monkeypatch.setenv("AUDIO_TRANSCRIBER_WORK_DIR", str(tmp_path / "work"))
    calls = []
    monkeypatch.setattr(
        uvicorn,
        "run",
        lambda application, **kwargs: calls.append((application, kwargs)),
    )
    return calls


def test_serve_runs_the_app_on_localhost_by_default(served):
    result = runner.invoke(app, ["serve"])

    assert result.exit_code == ExitCode.SUCCESS, result.output
    ((application, kwargs),) = served
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 8000
    assert application.state.settings.concurrency == 1


def test_serve_takes_host_port_and_concurrency(served):
    result = runner.invoke(
        app,
        ["serve", "--host", "0.0.0.0", "--port", "9001", "--workers-concurrency", "3"],
    )

    assert result.exit_code == ExitCode.SUCCESS, result.output
    ((application, kwargs),) = served
    assert (kwargs["host"], kwargs["port"]) == ("0.0.0.0", 9001)
    assert application.state.settings.concurrency == 3


def test_serve_reports_unusable_settings_as_a_config_error(served, monkeypatch):
    monkeypatch.setenv("AUDIO_TRANSCRIBER_CONCURRENCY", "lots")

    result = runner.invoke(app, ["serve"])

    assert result.exit_code == ExitCode.CONFIG_ERROR
    assert "error: AUDIO_TRANSCRIBER_CONCURRENCY" in result.output
    assert not served


def test_serve_without_the_api_extra_says_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "uvicorn", None)

    result = runner.invoke(app, ["serve"])

    assert result.exit_code == ExitCode.CONFIG_ERROR
    assert "error:" in result.output
    assert 'pip install -e ".[api]"' in result.output


def test_the_run_form_still_needs_no_subcommand():
    result = runner.invoke(app, ["--input", "x", "--backend", "nope"])

    assert result.exit_code == ExitCode.CONFIG_ERROR
    assert "unknown backend" in result.output or "not found" in result.output


def test_help_lists_serve():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "serve" in result.output
    assert "--input" in result.output
