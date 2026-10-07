"""Tests for scoring transcripts the pipeline already wrote, with no backend run.

This is how a hosted backend's output is re-scored without spending credits
again, so it must give exactly the scores a backend run would.
"""

import json
from pathlib import Path

import pytest

from audio_transcriber.errors import ConfigError
from audio_transcriber.evaluation.reference import Pair, transcript_pairs
from audio_transcriber.evaluation.runner import (
    UNKNOWN,
    evaluate_custom,
    score_transcripts,
)
from audio_transcriber.manifest import InputRecord, Manifest
from tests.test_evaluation_custom import GROUND_TRUTH, HYPOTHESIS, _pair

pytest.importorskip("jiwer")

# The scripted backend and its environment, for the run this is checked against.
from tests.test_evaluation_custom import scripted_environment  # noqa: E402, F401


def _transcript(path: Path, segments=HYPOTHESIS) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([s.to_dict() for s in segments]), encoding="utf-8")
    return path


def _reference(path: Path, ground_truth=GROUND_TRUTH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ground_truth), encoding="utf-8")
    return path


def _manifest(run: Path) -> None:
    manifest = Manifest(
        run_id=run.name,
        backend="elevenlabs-scribe",
        model_version="scribe_v2",
        config={"diarize": True},
        inputs=[
            InputRecord(
                path="audio/talk.mp4",
                checksum="sha256:00",
                size_bytes=1,
                transcript="talk.json",
            )
        ],
    )
    (run / "manifest.json").write_text(json.dumps(manifest.to_dict()))
    (run / "errors.json").write_text("[]")


def test_a_single_transcript_is_scored_without_a_backend(tmp_path: Path):
    pair = Pair(
        name="talk",
        media=_transcript(tmp_path / "talk.json"),
        reference=_reference(tmp_path / "truth" / "talk.json"),
    )

    (result,) = score_transcripts(label="again", pairs=[pair]).results

    assert result.score.wer == pytest.approx(0.1)
    assert result.reference["file"] == "talk.json"
    assert result.elapsed_s == 0.0


def test_the_backend_is_read_from_the_run_manifest(tmp_path: Path):
    run = tmp_path / "out" / "2026-10-02T10-00-00Z"
    _transcript(run / "talk.json")
    _manifest(run)
    pairs = transcript_pairs(run, _reference(tmp_path / "truth" / "talk.json").parent)

    report = score_transcripts(label="again", pairs=pairs)

    assert (report.backend, report.model_version) == ("elevenlabs-scribe", "scribe_v2")
    assert report.config == {"diarize": True}


def test_without_a_manifest_the_backend_is_unknown(tmp_path: Path):
    pair = Pair(
        name="talk",
        media=_transcript(tmp_path / "talk.json"),
        reference=_reference(tmp_path / "truth" / "talk.json"),
    )

    report = score_transcripts(label="again", pairs=[pair])

    assert (report.backend, report.model_version) == (UNKNOWN, UNKNOWN)
    assert report.config == {}


def test_a_run_folder_is_matched_to_a_reference_folder_by_name(tmp_path: Path):
    run = tmp_path / "run"
    for name in ("b", "a"):
        _transcript(run / f"{name}.json")
        _reference(tmp_path / "truth" / f"{name}.json")
    (tmp_path / "truth" / "a.mp4").write_bytes(b"the recording, not needed")
    _manifest(run)

    pairs = transcript_pairs(run, tmp_path / "truth")

    assert [(p.name, p.media, p.reference.name) for p in pairs] == [
        ("a", run / "a.json", "a.json"),
        ("b", run / "b.json", "b.json"),
    ]


@pytest.mark.parametrize(
    ("transcripts", "references", "named"),
    [
        (("a", "b"), ("a",), "b.json"),
        (("a",), ("a", "c"), "c.json"),
    ],
)
def test_an_unmatched_transcript_or_reference_is_an_error(
    tmp_path: Path, transcripts, references, named
):
    for name in transcripts:
        _transcript(tmp_path / "run" / f"{name}.json")
    for name in references:
        _reference(tmp_path / "truth" / f"{name}.json")

    with pytest.raises(ConfigError, match=named.replace(".", r"\.")):
        transcript_pairs(tmp_path / "run", tmp_path / "truth")


def test_only_filters_the_run_folder(tmp_path: Path):
    for name in ("a", "b"):
        _transcript(tmp_path / "run" / f"{name}.json")
    _reference(tmp_path / "truth" / "a.json")

    pairs = transcript_pairs(tmp_path / "run", tmp_path / "truth", only=["a"])

    assert [pair.name for pair in pairs] == ["a"]


def test_a_malformed_transcript_is_an_error_naming_it(tmp_path: Path):
    pair = Pair(
        name="talk",
        media=_transcript(tmp_path / "talk.json", segments=()),
        reference=_reference(tmp_path / "truth" / "talk.json"),
    )
    pair.media.write_text('{"segments": []}')

    with pytest.raises(ConfigError, match="talk.json"):
        score_transcripts(label="again", pairs=[pair])


def test_duration_without_audio_is_the_last_speech(tmp_path: Path):
    pair = Pair(
        name="talk",
        media=_transcript(tmp_path / "talk.json"),
        reference=_reference(tmp_path / "truth" / "talk.json"),
    )

    (result,) = score_transcripts(label="again", pairs=[pair]).results

    assert result.duration_s == pytest.approx(7.0)


def test_scores_match_the_run_that_produced_the_transcript(tmp_path: Path):
    """Re-scoring must not move a number, or it cannot stand in for a run."""
    pytest.importorskip("pyannote.metrics")
    pair = _pair(tmp_path)
    out = tmp_path / "transcripts"
    run = evaluate_custom(
        backend_name="scripted", label="run", pairs=[pair], transcript_dir=out
    )

    again = score_transcripts(
        label="again",
        pairs=[
            Pair(name="talk", media=out / "talk.run.json", reference=pair.reference)
        ],
    )

    before, after = run.results[0], again.results[0]
    assert after.score == before.score
    assert after.annotation == before.annotation
    assert after.reference == before.reference


# --- Clips ---


def _clip_ground_truth(clip_start: float | None) -> dict | list:
    """`GROUND_TRUTH` written against a video, starting 600 s in."""
    segments = [
        {**s, "start_time": s["start_time"] + 600, "end_time": s["end_time"] + 600}
        for s in GROUND_TRUTH
    ]
    if clip_start is None:
        return segments
    return {"clip_start": clip_start, "segments": segments}


def _run_with_duration(run: Path, duration: float) -> None:
    manifest = Manifest(
        run_id=run.name,
        backend="scripted",
        model_version="scripted-1.0",
        config={},
        inputs=[
            InputRecord(
                path="audio/talk.mp4",
                checksum="sha256:00",
                size_bytes=1,
                transcript="talk.json",
                duration=duration,
            )
        ],
    )
    (run / "manifest.json").write_text(json.dumps(manifest.to_dict()))


def test_a_clip_rescored_matches_its_ground_truth_in_its_own_times(tmp_path: Path):
    plain = Pair(
        name="talk",
        media=_transcript(tmp_path / "talk.json"),
        reference=_reference(tmp_path / "truth" / "talk.json"),
    )
    clip = Pair(
        name="talk",
        media=plain.media,
        reference=_reference(tmp_path / "clip" / "talk.json", _clip_ground_truth(600)),
    )

    (expected,) = score_transcripts(label="plain", pairs=[plain]).results
    (result,) = score_transcripts(label="clip", pairs=[clip]).results

    assert result.score == expected.score
    assert result.annotation == expected.annotation


def test_the_recorded_duration_bounds_the_ground_truth(tmp_path: Path):
    run = tmp_path / "run"
    _transcript(run / "talk.json")
    _run_with_duration(run, 6.0)
    pairs = transcript_pairs(run, _reference(tmp_path / "truth" / "talk.json").parent)

    (result,) = score_transcripts(label="again", pairs=pairs).results

    assert result.duration_s == 6.0
    assert result.reference["outside_s"] == 1.0


@pytest.mark.parametrize("with_manifest", [True, False], ids=["manifest", "bare"])
def test_a_forgotten_clip_start_is_refused_when_rescoring(
    tmp_path: Path, with_manifest
):
    run = tmp_path / "run"
    _transcript(run / "talk.json")
    if with_manifest:
        _run_with_duration(run, 8.0)
    truth = _reference(tmp_path / "truth" / "talk.json", _clip_ground_truth(None))

    with pytest.raises(ConfigError, match="is clip_start missing or wrong"):
        score_transcripts(label="again", pairs=transcript_pairs(run, truth.parent))
