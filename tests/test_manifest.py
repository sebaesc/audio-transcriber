"""Tests for the run manifest.

Reproducibility rests on the manifest, so these cover checksums, the recorded
config, and round-tripping through JSON.
"""

import json

from audio_transcriber.manifest import (
    MANIFEST_VERSION,
    InputRecord,
    Manifest,
    checksum_file,
)
from audio_transcriber.schema import SCHEMA_VERSION, Segment, Transcript


def test_checksum_is_stable_for_identical_content(tmp_path):
    first = tmp_path / "a.wav"
    second = tmp_path / "b.wav"
    first.write_bytes(b"audio bytes")
    second.write_bytes(b"audio bytes")

    assert checksum_file(first) == checksum_file(second)


def test_checksum_changes_with_content(tmp_path):
    path = tmp_path / "a.wav"
    path.write_bytes(b"audio bytes")
    before = checksum_file(path)
    path.write_bytes(b"different bytes")

    assert checksum_file(path) != before


def test_checksum_is_algorithm_prefixed(tmp_path):
    path = tmp_path / "a.wav"
    path.write_bytes(b"x")

    assert checksum_file(path).startswith("sha256:")


def test_checksum_reads_files_larger_than_one_chunk(tmp_path):
    """Files are read in chunks, so a multi-chunk file must still checksum."""
    path = tmp_path / "big.wav"
    path.write_bytes(b"x" * (1024 * 1024 * 2 + 17))

    assert checksum_file(path).startswith("sha256:")


def test_input_record_reads_checksum_and_size(tmp_path):
    path = tmp_path / "a.wav"
    path.write_bytes(b"audio bytes")

    record = InputRecord.from_path(path)

    assert record.path == str(path)
    assert record.size_bytes == len(b"audio bytes")
    assert record.checksum == checksum_file(path)


def test_manifest_records_backend_model_version_and_config():
    """The three things a transcript must be traceable back to."""
    manifest = Manifest(
        run_id="2026-09-22T10-00-00",
        backend="reference",
        model_version="tiny-1.0",
        config={"language": "en", "beam_size": 5},
    )

    data = manifest.to_dict()

    assert data["backend"] == "reference"
    assert data["model_version"] == "tiny-1.0"
    assert data["config"] == {"language": "en", "beam_size": 5}


def test_manifest_round_trips_through_json(tmp_path):
    path = tmp_path / "a.wav"
    path.write_bytes(b"audio bytes")
    manifest = Manifest(
        run_id="run-1",
        backend="reference",
        model_version="tiny-1.0",
        config={"language": "en"},
        inputs=[InputRecord.from_path(path)],
        finished_at="2026-09-22T10:05:00+00:00",
        succeeded=1,
        failed=0,
        tool_version="0.1.0",
    )

    restored = Manifest.from_dict(json.loads(json.dumps(manifest.to_dict())))

    assert restored == manifest


def test_manifest_records_its_format_version():
    manifest = Manifest(run_id="run-1", backend="reference")

    assert manifest.to_dict()["manifest_version"] == MANIFEST_VERSION


def test_manifest_starts_unfinished():
    """finished_at stays None until the run ends, so partial runs are visible."""
    manifest = Manifest(run_id="run-1", backend="reference")

    assert manifest.finished_at is None
    assert manifest.started_at


def test_manifest_preserves_input_order():
    records = [
        InputRecord(path=f"{name}.wav", checksum="sha256:x", size_bytes=1)
        for name in ("c", "a", "b")
    ]
    manifest = Manifest(run_id="run-1", backend="reference", inputs=records)

    paths = [item["path"] for item in manifest.to_dict()["inputs"]]

    assert paths == ["c.wav", "a.wav", "b.wav"]


def test_input_record_carries_the_optional_per_file_sidecar(tmp_path):
    """Transcript files are bare arrays, so their metadata is recorded here."""
    path = tmp_path / "a.wav"
    path.write_bytes(b"audio bytes")
    transcript = Transcript(
        source=str(path),
        segments=(Segment(start_time=0.0, end_time=1.0, text="hello"),),
        language="en",
        duration=1.0,
    )

    record = InputRecord.from_path(path).with_result("a.json", transcript.metadata())

    assert record.transcript == "a.json"
    assert record.language == "en"
    assert record.duration == 1.0
    assert record.segment_count == 1


def test_input_record_sidecar_is_empty_until_the_file_is_transcribed():
    """A failed input keeps its checksum and leaves the result fields null."""
    record = InputRecord(path="a.wav", checksum="sha256:x", size_bytes=1)

    data = record.to_dict()

    assert data["transcript"] is None
    assert data["segment_count"] is None


def test_with_result_does_not_mutate_the_original_record():
    original = InputRecord(path="a.wav", checksum="sha256:x", size_bytes=1)

    original.with_result("a.json", {"segment_count": 3})

    assert original.transcript is None


def test_manifest_records_the_transcript_schema_version():
    """The array files have no room for a version, so the manifest holds it."""
    manifest = Manifest(run_id="run-1", backend="reference")

    assert manifest.to_dict()["schema_version"] == SCHEMA_VERSION
