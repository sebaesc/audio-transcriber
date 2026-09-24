"""Tests for the CLI contract: flags, exit codes, and output routing.

The CLI is what callers script against, so its exit codes and stream discipline
are covered explicitly. A stub backend stands in for an engine.
"""

import json
from contextlib import contextmanager
from pathlib import Path

import pytest
from typer.testing import CliRunner

from audio_transcriber import pipeline, registry
from audio_transcriber.backend import TranscriptionBackend
from audio_transcriber.cli import ExitCode, app
from audio_transcriber.errors import TranscriptionError
from audio_transcriber.schema import Segment, Transcript

runner = CliRunner()


class _StubBackend(TranscriptionBackend):
    name = "stub"

    def check_requirements(self) -> None:
        return None

    def model_version(self) -> str:
        return "stub-1.0"

    def transcribe(self, audio_path: Path, source: str) -> Transcript:
        if "broken" in source:
            raise TranscriptionError(source, self.name, "engine said no.")
        return Transcript(
            source=source,
            segments=(
                Segment(
                    start_time=0.0, end_time=1.0, text="hello", spoken_language="en"
                ),
            ),
            language="en",
            duration=1.0,
            backend=self.name,
            model_version=self.model_version(),
        )


@pytest.fixture(autouse=True)
def stub_environment(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", {"stub": _StubBackend})
    monkeypatch.setattr(registry, "_BUILTINS", ())
    monkeypatch.setattr(registry, "_unavailable", {})
    monkeypatch.setattr(registry, "_builtins_loaded", True)
    monkeypatch.setattr(pipeline.audio, "check_ffmpeg", lambda: None)

    @contextmanager
    def _fake_decoded(source: Path):
        yield source

    monkeypatch.setattr(pipeline.audio, "decoded", _fake_decoded)


@pytest.fixture
def audio_dir(tmp_path) -> Path:
    directory = tmp_path / "audio"
    directory.mkdir()
    (directory / "a.wav").write_bytes(b"fake audio")
    return directory


def _run(*args):
    return runner.invoke(app, list(args))


def test_clean_run_exits_zero(audio_dir, tmp_path):
    result = _run(
        "--input", str(audio_dir), "--backend", "stub", "--out", str(tmp_path / "o")
    )

    assert result.exit_code == ExitCode.SUCCESS


def test_partial_failure_exits_one(audio_dir, tmp_path):
    (audio_dir / "broken.wav").write_bytes(b"fake audio")

    result = _run(
        "--input", str(audio_dir), "--backend", "stub", "--out", str(tmp_path / "o")
    )

    assert result.exit_code == ExitCode.PARTIAL


def test_total_failure_exits_two(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "broken.wav").write_bytes(b"fake audio")

    result = _run(
        "--input", str(audio_dir), "--backend", "stub", "--out", str(tmp_path / "o")
    )

    assert result.exit_code == ExitCode.ABORTED


def test_unknown_backend_exits_with_config_error(audio_dir, tmp_path):
    result = _run(
        "--input", str(audio_dir), "--backend", "nope", "--out", str(tmp_path / "o")
    )

    assert result.exit_code == ExitCode.CONFIG_ERROR
    assert "unknown backend" in result.output
    assert "stub" in result.output


def test_a_backend_without_its_extra_says_what_to_install(
    monkeypatch, audio_dir, tmp_path
):
    monkeypatch.setattr(
        registry,
        "_unavailable",
        {
            "ghost": "needs 'ghost_engine'. Install it: pip install -e \".[ghost-extra]\""
        },
    )

    result = _run(
        "--input", str(audio_dir), "--backend", "ghost", "--out", str(tmp_path / "o")
    )

    assert result.exit_code == ExitCode.CONFIG_ERROR
    assert 'pip install -e ".[ghost-extra]"' in result.output
    assert not (tmp_path / "o").exists()


def test_missing_input_exits_with_config_error(tmp_path):
    result = _run("--input", str(tmp_path / "nope"), "--backend", "stub")

    assert result.exit_code == ExitCode.CONFIG_ERROR
    assert "input path not found" in result.output


def test_no_matching_files_says_what_to_check(audio_dir, tmp_path):
    result = _run(
        "--input",
        str(audio_dir),
        "--backend",
        "stub",
        "--out",
        str(tmp_path / "o"),
        "--glob",
        "*.nothing",
    )

    assert result.exit_code == ExitCode.CONFIG_ERROR
    assert "--glob" in result.output


def test_missing_required_options_are_named(tmp_path):
    result = _run("--out", str(tmp_path / "o"))

    assert result.exit_code == ExitCode.CONFIG_ERROR
    assert "--input" in result.output and "--backend" in result.output


def test_errors_are_prefixed(audio_dir, tmp_path):
    result = _run(
        "--input", str(audio_dir), "--backend", "nope", "--out", str(tmp_path / "o")
    )

    assert "error:" in result.output


def test_partial_run_warns_and_points_at_errors_json(audio_dir, tmp_path):
    (audio_dir / "broken.wav").write_bytes(b"fake audio")

    result = _run(
        "--input", str(audio_dir), "--backend", "stub", "--out", str(tmp_path / "o")
    )

    assert "warning:" in result.output
    assert "errors.json" in result.output


def test_summary_reports_count_and_destination(audio_dir, tmp_path):
    result = _run(
        "--input", str(audio_dir), "--backend", "stub", "--out", str(tmp_path / "o")
    )

    assert "Transcribed 1 files" in result.output
    assert str(tmp_path / "o") in result.output


def test_progress_is_reported_per_file(audio_dir, tmp_path):
    result = _run(
        "--input", str(audio_dir), "--backend", "stub", "--out", str(tmp_path / "o")
    )

    assert "[1/1]" in result.output


def test_quiet_suppresses_progress_but_keeps_the_summary(audio_dir, tmp_path):
    result = _run(
        "--input",
        str(audio_dir),
        "--backend",
        "stub",
        "--out",
        str(tmp_path / "o"),
        "--quiet",
    )

    assert "[1/1]" not in result.output
    assert "Transcribed 1 files" in result.output


def test_backend_options_reach_the_manifest(audio_dir, tmp_path):
    out = tmp_path / "o"
    _run(
        "--input",
        str(audio_dir),
        "--backend",
        "stub",
        "--out",
        str(out),
        "--option",
        "beam_size=1",
        "--option",
        "language=es",
    )

    manifest = json.loads(next(out.glob("*/manifest.json")).read_text())

    assert manifest["config"] == {"beam_size": 1, "language": "es"}


def test_option_values_keep_their_json_types(audio_dir, tmp_path):
    """beam_size=1 must arrive as an int, not the string '1'."""
    out = tmp_path / "o"
    _run(
        "--input",
        str(audio_dir),
        "--backend",
        "stub",
        "--out",
        str(out),
        "--option",
        "beam_size=1",
        "--option",
        "vad_filter=true",
        "--option",
        "model=large-v3",
    )

    config = json.loads(next(out.glob("*/manifest.json")).read_text())["config"]

    assert config["beam_size"] == 1
    assert config["vad_filter"] is True
    assert config["model"] == "large-v3"


def test_malformed_option_is_rejected(audio_dir, tmp_path):
    result = _run(
        "--input",
        str(audio_dir),
        "--backend",
        "stub",
        "--out",
        str(tmp_path / "o"),
        "--option",
        "nonsense",
    )

    assert result.exit_code != ExitCode.SUCCESS
    assert "key=value" in result.output


def test_list_backends_prints_them(tmp_path):
    result = _run("--list-backends")

    assert result.exit_code == ExitCode.SUCCESS
    assert "stub" in result.output


def test_list_backends_notes_uninstalled_ones_on_stderr(monkeypatch):
    """stdout stays one installed name per line, for scripts."""
    monkeypatch.setattr(
        registry,
        "_unavailable",
        {"ghost": 'not installed. Install its extra: pip install -e ".[ghost-extra]"'},
    )

    result = _run("--list-backends")

    assert result.exit_code == ExitCode.SUCCESS
    assert result.stdout.split() == ["stub"]
    assert "ghost is not installed. Install its extra: pip" in result.stderr


def test_list_backends_with_none_installed_says_how_to_get_one(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", {})

    result = _run("--list-backends")

    assert result.exit_code == ExitCode.CONFIG_ERROR
    assert "No backends installed" in result.output


def test_version_flag_prints_the_version():
    from audio_transcriber import __version__

    result = _run("--version")

    assert result.exit_code == ExitCode.SUCCESS
    assert __version__ in result.output


def test_help_lists_every_flag():
    result = _run("--help")

    for flag in ("--input", "--backend", "--out", "--glob", "--option", "--quiet"):
        assert flag in result.output


def test_output_layout_matches_the_documented_contract(audio_dir, tmp_path):
    out = tmp_path / "o"
    _run("--input", str(audio_dir), "--backend", "stub", "--out", str(out))

    run_directory = next(out.iterdir())
    names = sorted(p.name for p in run_directory.iterdir())

    assert names == ["a.json", "errors.json", "manifest.json"]
