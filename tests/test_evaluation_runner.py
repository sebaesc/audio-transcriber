"""Tests for the eval runner and report comparison.

These use a stub backend rather than an engine: what is tested is the reporting
and the verdict logic, not transcription quality.
"""

import json
from pathlib import Path

import pytest

from audio_transcriber.evaluation.runner import (
    RecordingResult,
    Report,
    compare,
)
from audio_transcriber.evaluation.scoring import Score


def _score(wer: float, hypothesis_words: int = 95, hyp_share: float = 0.5) -> Score:
    return Score(
        wer=wer,
        substitutions=10,
        deletions=5,
        insertions=2,
        hits=80,
        reference_words=100,
        hypothesis_words=hypothesis_words,
        reference_spanish_share=0.6,
        hypothesis_spanish_share=hyp_share,
    )


def _report(label: str, scores: dict[str, Score]) -> Report:
    return Report(
        label=label,
        backend="stub",
        model_version="stub-1.0",
        results=[
            RecordingResult(
                name=name,
                duration_s=600.0,
                elapsed_s=120.0,
                segments=50,
                detected_languages=["en", "es"],
                score=value,
            )
            for name, value in scores.items()
        ],
    )


def test_report_summarizes_the_mean_across_recordings():
    report = _report("baseline", {"a": _score(0.4), "b": _score(0.6)})

    assert report.mean("wer") == pytest.approx(0.5)


def test_report_ignores_failed_recordings_in_the_mean():
    report = _report("baseline", {"a": _score(0.4)})
    report.results.append(
        RecordingResult(name="broken", duration_s=60.0, error="RuntimeError: nope")
    )

    assert report.mean("wer") == pytest.approx(0.4)
    assert len(report.scored) == 1
    assert report.to_dict()["summary"]["recordings"] == 2


def test_report_records_realtime_factor():
    report = _report("baseline", {"a": _score(0.4)})

    assert report.to_dict()["recordings"][0]["realtime_factor"] == pytest.approx(5.0)


def test_report_round_trips_through_json(tmp_path: Path):
    report = _report("baseline", {"a": _score(0.4)})

    path = report.write(tmp_path / "baseline.json")
    data = json.loads(path.read_text())

    assert data["label"] == "baseline"
    assert data["recordings"][0]["score"]["wer"] == 0.4


def test_report_names_the_corpus_and_its_licence(tmp_path: Path):
    """The corpus is CC BY-SA; a report that uses it should say so."""
    data = _report("baseline", {"a": _score(0.4)}).to_dict()

    assert "Bangor" in data["corpus"]["dataset"]
    assert data["corpus"]["license"]


def test_comparison_counts_recordings_improved():
    baseline = _report("base", {"a": _score(0.44), "b": _score(0.42)})
    candidate = _report("cand", {"a": _score(0.34), "b": _score(0.57)})

    result = compare(baseline, candidate)

    assert result["recordings_compared"] == 2
    assert result["recordings_improved"] == 1
    assert result["improved_everywhere"] is False


def test_comparison_requires_every_recording_to_improve():
    """The prototype improved one recording and regressed another.

    A mean would have called that a wash. The verdict is per recording.
    """
    baseline = _report("base", {"a": _score(0.44), "b": _score(0.42)})
    candidate = _report("cand", {"a": _score(0.34), "b": _score(0.41)})

    assert compare(baseline, candidate)["improved_everywhere"] is True


def test_comparison_rejects_a_win_bought_with_coverage():
    baseline = _report("base", {"a": _score(0.44, hypothesis_words=95)})
    candidate = _report("cand", {"a": _score(0.30, hypothesis_words=70)})

    result = compare(baseline, candidate)

    assert result["per_recording"]["a"]["coverage_regressed"] is True
    assert result["recordings_improved"] == 0


def test_comparison_reports_language_mix_movement():
    baseline = _report("base", {"a": _score(0.44, hyp_share=0.30)})
    candidate = _report("cand", {"a": _score(0.44, hyp_share=0.60)})

    result = compare(baseline, candidate)

    assert result["mean_language_mix_error_delta"] == pytest.approx(-0.30, abs=1e-4)


def test_comparison_skips_a_recording_missing_from_the_candidate():
    baseline = _report("base", {"a": _score(0.44), "b": _score(0.42)})
    candidate = _report("cand", {"a": _score(0.34)})

    result = compare(baseline, candidate)

    assert result["recordings_compared"] == 1


def test_comparison_of_nothing_is_not_an_improvement():
    """An empty comparison must not read as success."""
    result = compare(_report("base", {}), _report("cand", {}))

    assert result["improved_everywhere"] is False
