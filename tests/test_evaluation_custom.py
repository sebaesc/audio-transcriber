"""Tests for scoring a backend run against a user's own ground truth.

A scripted backend stands in for an engine, returning a fixed transcript per
recording, so what is tested is the wiring: references loaded up front, the
existing scores applied to them, and a report `compare` accepts.
"""

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from audio_transcriber import registry
from audio_transcriber.backend import TranscriptionBackend
from audio_transcriber.errors import ConfigError
from audio_transcriber.evaluation import runner
from audio_transcriber.evaluation.reference import Pair
from audio_transcriber.evaluation.runner import compare, evaluate_custom
from audio_transcriber.schema import Segment, Transcript
from tests.conftest import write_wav

pytest.importorskip("jiwer")

GROUND_TRUTH = [
    {
        "start_time": 0.0,
        "end_time": 4.0,
        "speaker": "Bob",
        "spoken_language": "en",
        "text": "So the meeting is at three.",
    },
    {
        "start_time": 4.0,
        "end_time": 7.0,
        "speaker": "Ana",
        "spoken_language": "es",
        "text": "¿Y tú qué piensas?",
    },
]

#: One substitution in ten words; the Spanish turn labelled English.
HYPOTHESIS = (
    Segment(0.0, 4.0, "so the meeting is at tree", "SPEAKER_00", "en"),
    Segment(4.0, 7.0, "y tu que piensas", "SPEAKER_01", "en"),
)


class _ScriptedBackend(TranscriptionBackend):
    """Returns `HYPOTHESIS` for every recording; `speakers=False` drops labels."""

    name = "scripted"
    calls: list[str] = []

    def check_requirements(self) -> None:
        return None

    def model_version(self) -> str:
        return "scripted-1.0"

    def transcribe(self, audio_path: Path, source: str) -> Transcript:
        type(self).calls.append(source)
        if "broken" in source:
            raise RuntimeError("engine said no")
        segments = HYPOTHESIS
        if not self.config.get("speakers", True):
            segments = tuple(
                Segment(s.start_time, s.end_time, s.text, None, s.spoken_language)
                for s in segments
            )
        return Transcript(source=source, segments=segments, backend=self.name)


@pytest.fixture(autouse=True)
def scripted_environment(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", {"scripted": _ScriptedBackend})
    monkeypatch.setattr(registry, "_BUILTINS", ())
    monkeypatch.setattr(registry, "_unavailable", {})
    monkeypatch.setattr(registry, "_builtins_loaded", True)
    monkeypatch.setattr(runner.audio, "check_ffmpeg", lambda: None)
    monkeypatch.setattr(_ScriptedBackend, "calls", [])

    @contextmanager
    def _already_wav(source: Path):
        yield source

    monkeypatch.setattr(runner.audio, "decoded", _already_wav)


def _pair(tmp_path: Path, name: str = "talk", ground_truth=GROUND_TRUTH) -> Pair:
    media = write_wav(tmp_path / f"{name}.wav", seconds=8.0, frequency=0.0)
    reference = tmp_path / f"{name}.json"
    reference.write_text(json.dumps(ground_truth), encoding="utf-8")
    return Pair(name=name, media=media, reference=reference)


def _run(pairs, label="try", **config):
    return evaluate_custom(
        backend_name="scripted", label=label, pairs=pairs, backend_config=config
    )


def test_each_recording_gets_wer_and_retained(tmp_path: Path):
    (result,) = _run([_pair(tmp_path)]).results

    assert result.score.wer == pytest.approx(0.1)
    assert result.score.retained == pytest.approx(1.0)
    assert result.segments == 2


def test_duration_comes_from_the_decoded_audio(tmp_path: Path):
    (result,) = _run([_pair(tmp_path)]).results

    assert result.duration_s == pytest.approx(8.0)


def test_speakers_and_language_are_scored_when_the_ground_truth_has_labels(
    tmp_path: Path,
):
    pytest.importorskip("pyannote.metrics")

    scores = _run([_pair(tmp_path)]).results[0].annotation

    assert scores["diarization"]["confusion"] == pytest.approx(0.0)
    assert scores["language"]["accuracy"] == pytest.approx(4 / 7, abs=1e-4)
    assert scores["language"]["recall"]["es"] == pytest.approx(0.0)


def test_unlabelled_dimensions_are_skipped_with_a_reason(tmp_path: Path):
    bare = [
        {"start_time": s["start_time"], "end_time": s["end_time"], "text": s["text"]}
        for s in GROUND_TRUTH
    ]

    scores = _run([_pair(tmp_path, ground_truth=bare)]).results[0].annotation

    assert scores["diarization"] is None
    assert "no speaker labels" in scores["diarization_skipped"]
    assert scores["language"] is None
    assert "no spoken_language labels" in scores["language_skipped"]


def test_a_transcript_without_speakers_says_so(tmp_path: Path):
    scores = _run([_pair(tmp_path)], speakers=False).results[0].annotation

    assert scores["diarization"] is None
    assert "transcript has no speaker labels" in scores["diarization_skipped"]


def test_an_invalid_ground_truth_stops_the_run_before_transcribing(tmp_path: Path):
    good = _pair(tmp_path, "good")
    bad = _pair(tmp_path, "bad", ground_truth=[{"start_time": 1.0, "end_time": 0.5}])

    with pytest.raises(ConfigError, match="bad.json"):
        _run([good, bad])

    assert _ScriptedBackend.calls == []


def test_one_recording_failing_does_not_stop_the_others(tmp_path: Path):
    report = _run([_pair(tmp_path, "broken"), _pair(tmp_path, "fine")])

    assert "engine said no" in report.results[0].error
    assert report.results[1].score is not None


def test_the_reference_text_is_written_beside_the_hypothesis(tmp_path: Path):
    out = tmp_path / "transcripts"

    evaluate_custom(
        backend_name="scripted",
        label="try",
        pairs=[_pair(tmp_path)],
        transcript_dir=out,
    )

    assert (out / "talk.try.txt").read_text() == (
        "so the meeting is at tree y tu que piensas"
    )
    assert (out / "talk.reference.txt").read_text() == (
        "so the meeting is at three y tu que piensas"
    )
    assert json.loads((out / "talk.try.json").read_text())[0]["speaker"] == (
        "SPEAKER_00"
    )


def test_the_report_traces_each_score_to_its_ground_truth(tmp_path: Path):
    pair = _pair(tmp_path)

    data = _run([pair]).to_dict()

    reference = data["recordings"][0]["reference"]
    assert reference["file"] == "talk.json"
    assert reference["checksum"].startswith("sha256:")
    assert data["corpus"]["dataset"] == runner.CUSTOM_DATASET
    assert "talkbank" not in data["summary"]
    assert "talkbank" not in data["recordings"][0]


def test_the_report_summarizes_ground_truth_scores(tmp_path: Path):
    pytest.importorskip("pyannote.metrics")

    summary = _run([_pair(tmp_path)]).to_dict()["summary"]

    assert summary["ground_truth"]["mean_speaker_confusion"] == pytest.approx(0.0)
    assert summary["ground_truth"]["mean_language_accuracy"] == pytest.approx(
        0.5714, abs=1e-4
    )


def test_compare_accepts_two_custom_reports(tmp_path: Path):
    pytest.importorskip("pyannote.metrics")
    from audio_transcriber.evaluation.__main__ import _load

    pairs = [_pair(tmp_path)]
    baseline = _run(pairs, label="base").write(tmp_path / "base.json")
    candidate = _run(pairs, label="cand", speakers=False).write(tmp_path / "c.json")

    result = compare(_load(baseline), _load(candidate))

    assert result["recordings_compared"] == 1
    assert result["mean_wer_delta"] == 0.0
    assert result["mean_language_accuracy_delta"] == 0.0


def test_a_reloaded_custom_report_keeps_its_shape(tmp_path: Path):
    from audio_transcriber.evaluation.__main__ import _load

    path = _run([_pair(tmp_path)]).write(tmp_path / "r.json")

    def untimed(recordings):
        # A stub run takes ~0 s, which rounds away the realtime factor.
        return [{**item, "realtime_factor": None} for item in recordings]

    reloaded = _load(path).to_dict()["recordings"]
    assert untimed(reloaded) == untimed(json.loads(path.read_text())["recordings"])


# --- Clips: a ground truth written against the longer video ---


def _in_video_times(ground_truth=GROUND_TRUTH, clip_start: float = 600.0) -> list:
    """The ground truth as written against a video the 8 s clip was cut from."""
    return [
        {
            **segment,
            "start_time": segment["start_time"] + clip_start,
            "end_time": segment["end_time"] + clip_start,
        }
        for segment in ground_truth
    ]


def test_a_clip_scores_as_its_ground_truth_in_the_clips_own_times(tmp_path: Path):
    plain = _run([_pair(tmp_path, "plain")]).results[0]
    clip = _run(
        [
            _pair(
                tmp_path,
                "clip",
                ground_truth={"clip_start": "10:00", "segments": _in_video_times()},
            )
        ]
    ).results[0]

    assert clip.score == plain.score
    assert clip.annotation == plain.annotation
    assert clip.reference["clip_start"] == 600.0


def test_a_forgotten_clip_start_stops_the_run_before_transcribing(tmp_path: Path):
    """Scored anyway, speaker confusion would read a perfect 0."""
    pair = _pair(tmp_path, ground_truth=_in_video_times())

    with pytest.raises(ConfigError, match="is clip_start missing or wrong"):
        _run([_pair(tmp_path, "fine"), pair])

    assert _ScriptedBackend.calls == []


def test_speech_past_the_end_of_the_recording_is_not_scored(tmp_path: Path):
    extra = [
        *GROUND_TRUTH,
        {"start_time": 8.5, "end_time": 9.0, "text": "and one more thing"},
    ]

    (result,) = _run([_pair(tmp_path, ground_truth=extra)]).results

    assert result.score.wer == pytest.approx(0.1)
    assert result.reference["outside_s"] == 0.5
