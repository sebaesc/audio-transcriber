"""Tests for run orchestration.

These use a stub backend and a stub decoder, so they test the pipeline rather
than an engine or ffmpeg. The real engine is covered by the conformance suite.
"""

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from audio_transcriber import pipeline, registry
from audio_transcriber.backend import TranscriptionBackend
from audio_transcriber.errors import (
    BackendRequirementError,
    ConfigError,
    TranscriptionError,
)
from audio_transcriber.pipeline import RunStatus
from audio_transcriber.schema import Segment, Transcript


class _StubBackend(TranscriptionBackend):
    """Transcribes anything, and fails any file whose name contains 'broken'."""

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
def stub_environment(monkeypatch, tmp_path):
    """Register the stub backend and stub out ffmpeg for every test here."""
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
    for name in ("a.wav", "b.wav"):
        (directory / name).write_bytes(b"fake audio")
    return directory


def test_run_writes_one_transcript_per_input(audio_dir, tmp_path):
    result = pipeline.run(audio_dir, "stub", tmp_path / "out", run_id="run-1")

    assert sorted(p.name for p in result.run_directory.glob("*.json")) == [
        "a.json",
        "b.json",
        "errors.json",
        "manifest.json",
    ]


def test_transcript_file_is_a_bare_array(audio_dir, tmp_path):
    result = pipeline.run(audio_dir, "stub", tmp_path / "out", run_id="run-1")

    data = json.loads((result.run_directory / "a.json").read_text())

    assert isinstance(data, list)
    assert list(data[0]) == [
        "start_time",
        "end_time",
        "speaker",
        "spoken_language",
        "text",
    ]


def test_manifest_records_what_produced_the_run(audio_dir, tmp_path):
    result = pipeline.run(
        audio_dir,
        "stub",
        tmp_path / "out",
        backend_config={"language": "en"},
        run_id="run-1",
    )

    manifest = json.loads((result.run_directory / "manifest.json").read_text())

    assert manifest["backend"] == "stub"
    assert manifest["model_version"] == "stub-1.0"
    assert manifest["config"] == {"language": "en"}
    assert manifest["succeeded"] == 2
    assert manifest["failed"] == 0
    assert manifest["finished_at"]


def test_manifest_records_a_checksum_per_input(audio_dir, tmp_path):
    result = pipeline.run(audio_dir, "stub", tmp_path / "out", run_id="run-1")

    manifest = json.loads((result.run_directory / "manifest.json").read_text())

    assert len(manifest["inputs"]) == 2
    assert all(item["checksum"].startswith("sha256:") for item in manifest["inputs"])


def test_manifest_carries_the_per_file_metadata_sidecar(audio_dir, tmp_path):
    result = pipeline.run(audio_dir, "stub", tmp_path / "out", run_id="run-1")

    manifest = json.loads((result.run_directory / "manifest.json").read_text())
    first = manifest["inputs"][0]

    assert first["transcript"] == "a.json"
    assert first["language"] == "en"
    assert first["segment_count"] == 1


def test_a_failing_file_does_not_abort_the_batch(audio_dir, tmp_path):
    (audio_dir / "broken.wav").write_bytes(b"fake audio")

    result = pipeline.run(audio_dir, "stub", tmp_path / "out", run_id="run-1")

    assert result.status is RunStatus.PARTIAL
    assert len(result.transcripts) == 2
    assert len(result.failures) == 1


def test_partial_results_are_kept_on_failure(audio_dir, tmp_path):
    (audio_dir / "broken.wav").write_bytes(b"fake audio")

    result = pipeline.run(audio_dir, "stub", tmp_path / "out", run_id="run-1")

    assert (result.run_directory / "a.json").exists()
    assert not (result.run_directory / "broken.json").exists()


def test_failures_are_collected_into_errors_json(audio_dir, tmp_path):
    (audio_dir / "broken.wav").write_bytes(b"fake audio")

    result = pipeline.run(audio_dir, "stub", tmp_path / "out", run_id="run-1")

    errors = json.loads((result.run_directory / "errors.json").read_text())

    assert len(errors) == 1
    assert errors[0]["error"] == "TranscriptionError"
    assert "broken.wav" in errors[0]["source"]
    assert "engine said no" in errors[0]["message"]


def test_errors_json_is_written_even_when_nothing_failed(audio_dir, tmp_path):
    """A consumer can always read it, rather than handling a missing file."""
    result = pipeline.run(audio_dir, "stub", tmp_path / "out", run_id="run-1")

    assert json.loads((result.run_directory / "errors.json").read_text()) == []


def test_a_run_where_everything_fails_is_aborted_not_partial(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "broken.wav").write_bytes(b"fake audio")

    result = pipeline.run(audio_dir, "stub", tmp_path / "out", run_id="run-1")

    assert result.status is RunStatus.ABORTED


def test_a_clean_run_reports_success(audio_dir, tmp_path):
    result = pipeline.run(audio_dir, "stub", tmp_path / "out", run_id="run-1")

    assert result.status is RunStatus.SUCCEEDED


def test_a_prior_run_is_never_overwritten(audio_dir, tmp_path):
    pipeline.run(audio_dir, "stub", tmp_path / "out", run_id="run-1")

    with pytest.raises(ConfigError, match="already has results"):
        pipeline.run(audio_dir, "stub", tmp_path / "out", run_id="run-1")


def test_inputs_sharing_a_stem_get_distinct_transcripts(tmp_path):
    """Two directories, one filename: the second must not overwrite the first."""
    audio_dir = tmp_path / "audio"
    (audio_dir / "monday").mkdir(parents=True)
    (audio_dir / "tuesday").mkdir(parents=True)
    (audio_dir / "monday" / "standup.wav").write_bytes(b"a")
    (audio_dir / "tuesday" / "standup.wav").write_bytes(b"b")

    result = pipeline.run(audio_dir, "stub", tmp_path / "out", run_id="run-1")

    written = sorted(p.name for p in result.run_directory.glob("standup*.json"))
    assert written == ["standup-2.json", "standup.json"]
    assert len(result.transcripts) == 2


def test_validation_failure_writes_nothing(tmp_path):
    """A run that cannot start leaves no half-made output directory behind."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    with pytest.raises(ConfigError, match="no audio files matched"):
        pipeline.run(audio_dir, "stub", tmp_path / "out", run_id="run-1")

    assert not (tmp_path / "out" / "run-1").exists()


def test_missing_input_path_fails_before_any_work(tmp_path):
    with pytest.raises(ConfigError, match="input path not found"):
        pipeline.run(tmp_path / "nope", "stub", tmp_path / "out")


def test_unmet_backend_requirements_fail_before_any_work(
    audio_dir, tmp_path, monkeypatch
):
    def _unmet(self):
        raise BackendRequirementError("stub", "model missing.", "Download it.")

    monkeypatch.setattr(_StubBackend, "check_requirements", _unmet)

    with pytest.raises(BackendRequirementError, match="Download it"):
        pipeline.run(audio_dir, "stub", tmp_path / "out", run_id="run-1")

    assert not (tmp_path / "out" / "run-1").exists()


def test_ffmpeg_is_checked_before_any_file_is_read(audio_dir, tmp_path, monkeypatch):
    def _missing():
        raise ConfigError("ffmpeg not found on PATH. Install it: apt install ffmpeg")

    monkeypatch.setattr(pipeline.audio, "check_ffmpeg", _missing)

    with pytest.raises(ConfigError, match="apt install ffmpeg"):
        pipeline.run(audio_dir, "stub", tmp_path / "out", run_id="run-1")


def test_progress_is_reported_per_file(audio_dir, tmp_path):
    seen = []

    pipeline.run(
        audio_dir,
        "stub",
        tmp_path / "out",
        run_id="run-1",
        on_progress=lambda index, total, source: seen.append((index, total)),
    )

    assert seen == [(1, 2), (2, 2)]


def test_run_ids_are_directory_safe():
    run_id = pipeline.new_run_id()

    assert "/" not in run_id and ":" not in run_id


def test_unserializable_config_values_do_not_break_the_manifest(audio_dir, tmp_path):
    """The manifest is written last; a Path in the config must not lose the run."""
    result = pipeline.run(
        audio_dir,
        "stub",
        tmp_path / "out",
        backend_config={"model_path": Path("/models/x.bin")},
        run_id="run-1",
    )

    manifest = json.loads((result.run_directory / "manifest.json").read_text())

    assert manifest["config"]["model_path"] == "/models/x.bin"


class _NonConformingBackend(_StubBackend):
    """Returns segments past the end of the audio, as Whisper can."""

    name = "stub"

    def transcribe(self, audio_path: Path, source: str) -> Transcript:
        return Transcript(
            source=source,
            segments=(Segment(start_time=0.0, end_time=40.0, text="hallucinated"),),
            language="en",
            duration=10.0,
            backend=self.name,
            model_version=self.model_version(),
        )


def test_a_non_conforming_transcript_is_a_failure_not_an_output(
    audio_dir, tmp_path, monkeypatch
):
    """The contract is enforced at runtime, not only in the conformance suite.

    A backend that breaks the schema must produce a reported failure rather than
    a bad transcript that reads like a success.
    """
    monkeypatch.setattr(registry, "_REGISTRY", {"stub": _NonConformingBackend})

    result = pipeline.run(audio_dir, "stub", tmp_path / "out", run_id="run-1")

    assert result.status is RunStatus.ABORTED
    assert not list(result.run_directory.glob("a.json"))

    errors = json.loads((result.run_directory / "errors.json").read_text())
    assert "violates the schema" in errors[0]["message"]
    assert "past the reported" in errors[0]["message"]


def test_a_conforming_transcript_is_written_unchanged(audio_dir, tmp_path):
    """The runtime check must not reject valid output."""
    result = pipeline.run(audio_dir, "stub", tmp_path / "out", run_id="run-1")

    assert result.status is RunStatus.SUCCEEDED
    assert len(result.transcripts) == 2
