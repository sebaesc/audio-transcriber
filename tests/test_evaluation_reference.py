"""Tests for reading a user's ground truth and pairing it with recordings.

Every fixture is a few hand-written segments. No real recording or ground truth
is committed: they are the team's data, not ours to vendor.
"""

import json
from pathlib import Path

import pytest

from audio_transcriber.errors import ConfigError
from audio_transcriber.evaluation.reference import (
    discover_pairs,
    load_reference,
    single_pair,
)

_SEGMENTS = [
    {
        "start_time": 4.0,
        "end_time": 6.5,
        "speaker": "Ana",
        "spoken_language": "es",
        "text": "¿Y tú qué piensas?",
    },
    {
        "start_time": 0.0,
        "end_time": 4.2,
        "speaker": "Bob",
        "spoken_language": "en",
        "text": "So, the meeting's at three.",
    },
]


def _write(path: Path, data) -> Path:
    path.write_text(
        data if isinstance(data, str) else json.dumps(data), encoding="utf-8"
    )
    return path


# --- Loading ---


def test_a_valid_ground_truth_loads_in_time_order(tmp_path: Path):
    reference = load_reference(_write(tmp_path / "talk.json", _SEGMENTS))

    assert [segment.start_time for segment in reference.segments] == [0.0, 4.0]
    assert reference.path == tmp_path / "talk.json"
    assert reference.checksum.startswith("sha256:")


def test_overlapping_speech_is_accepted(tmp_path: Path):
    """Two speakers at once is real speech, not a malformed ground truth."""
    reference = load_reference(_write(tmp_path / "talk.json", _SEGMENTS))

    assert reference.segments[0].end_time > reference.segments[1].start_time


@pytest.mark.parametrize(
    ("content", "problem"),
    [
        ("{not json", "not valid JSON"),
        ({"segments": _SEGMENTS}, "JSON array"),
        ([], "empty"),
        (["hello"], "segment 0"),
        ([_SEGMENTS[0], {"start_time": 1.0, "end_time": 2.0}], "segment 1"),
        (
            [{"start_time": 3.0, "end_time": 2.0, "text": "backwards"}],
            "segment 0",
        ),
        ([{"start_time": "soon", "end_time": 2.0, "text": "x"}], "segment 0"),
    ],
)
def test_a_malformed_ground_truth_names_the_file_and_the_problem(
    tmp_path: Path, content, problem
):
    path = _write(tmp_path / "bad.json", content)

    with pytest.raises(ConfigError) as caught:
        load_reference(path)

    assert "bad.json" in str(caught.value)
    assert problem in str(caught.value)


@pytest.mark.parametrize(
    ("written", "seconds"),
    [
        ("19:52", 1192.0),
        ("00:07", 7.0),
        ("75:10", 4510.0),
        ("1:02:03", 3723.0),
        ("01:02:03.5", 3723.5),
        ("12.5", 12.5),
        (12.5, 12.5),
    ],
)
def test_times_may_be_written_as_clock_times(tmp_path: Path, written, seconds):
    """The team writes ground truth by ear, as MM:SS read off a player."""
    segment = {"start_time": written, "end_time": 9999, "text": "hi"}

    reference = load_reference(_write(tmp_path / "talk.json", [segment]))

    assert reference.segments[0].start_time == pytest.approx(seconds)


@pytest.mark.parametrize("written", ["19.52.10", "12:75", "1:60:00", "-0:05", ""])
def test_a_time_that_is_neither_seconds_nor_a_clock_time_is_rejected(
    tmp_path: Path, written
):
    segment = {"start_time": 0, "end_time": written, "text": "hi"}
    path = _write(tmp_path / "bad.json", [segment])

    with pytest.raises(ConfigError, match=r"bad\.json, segment 0: end_time"):
        load_reference(path)


def test_a_missing_key_is_named(tmp_path: Path):
    path = _write(tmp_path / "bad.json", [{"start_time": 0.0, "end_time": 1.0}])

    with pytest.raises(ConfigError, match="text"):
        load_reference(path)


def test_end_before_start_is_explained(tmp_path: Path):
    path = _write(
        tmp_path / "bad.json", [{"start_time": 3.0, "end_time": 2.0, "text": "x"}]
    )

    with pytest.raises(ConfigError, match="before its start_time"):
        load_reference(path)


def test_a_missing_file_is_a_config_error(tmp_path: Path):
    with pytest.raises(ConfigError, match="nowhere.json"):
        load_reference(tmp_path / "nowhere.json")


# --- Utterances, labels and text ---


def test_segments_become_timed_utterances(tmp_path: Path):
    reference = load_reference(_write(tmp_path / "talk.json", _SEGMENTS))

    first, second = reference.utterances

    assert (first.speaker, first.start, first.end, first.language) == (
        "Bob",
        0.0,
        4.2,
        "en",
    )
    assert second.speaker == "Ana" and second.language == "es"
    assert first.timed


def test_a_fully_labelled_ground_truth_scores_both_dimensions(tmp_path: Path):
    reference = load_reference(_write(tmp_path / "talk.json", _SEGMENTS))

    assert reference.speakers_unscorable() is None
    assert reference.languages_unscorable() is None


def test_missing_labels_are_reported_as_a_reason(tmp_path: Path):
    """All null, or the keys left out: either way there is nothing to score."""
    segments = [
        {"start_time": 0.0, "end_time": 1.0, "speaker": None, "text": "hi"},
        {"start_time": 1.0, "end_time": 2.0, "text": "there"},
    ]
    reference = load_reference(_write(tmp_path / "talk.json", segments))

    assert "no speaker labels" in reference.speakers_unscorable()
    assert "no spoken_language labels" in reference.languages_unscorable()


def test_partly_labelled_speakers_cannot_be_scored(tmp_path: Path):
    """An unlabelled stretch of speech would be scored as nobody's."""
    segments = [
        {"start_time": 0.0, "end_time": 1.0, "speaker": "Ana", "text": "hi"},
        {"start_time": 1.0, "end_time": 2.0, "speaker": None, "text": "there"},
    ]
    reference = load_reference(_write(tmp_path / "talk.json", segments))

    assert "1 of 2 segments" in reference.speakers_unscorable()


def test_partly_labelled_languages_are_still_scored(tmp_path: Path):
    """Unlabelled segments are left out of language scoring, as in TalkBank."""
    segments = [
        {"start_time": 0.0, "end_time": 1.0, "spoken_language": "en", "text": "hi"},
        {"start_time": 1.0, "end_time": 2.0, "text": "um"},
    ]
    reference = load_reference(_write(tmp_path / "talk.json", segments))

    assert reference.languages_unscorable() is None


def test_reference_text_is_normalized_in_time_order(tmp_path: Path):
    reference = load_reference(_write(tmp_path / "talk.json", _SEGMENTS))

    assert reference.text == "so the meeting's at three y tu que piensas"


def test_end_time_is_the_latest_segment_end(tmp_path: Path):
    reference = load_reference(_write(tmp_path / "talk.json", _SEGMENTS))

    assert reference.end_time == 6.5


# --- Pairing recordings with ground truth ---


def test_a_single_pair_is_named_after_the_recording(tmp_path: Path):
    audio = _write(tmp_path / "standup.mp4", "video")
    truth = _write(tmp_path / "truth.json", _SEGMENTS)

    (pair,) = single_pair(audio, truth)

    assert (pair.name, pair.media, pair.reference) == ("standup", audio, truth)


def test_a_single_pair_with_a_missing_file_is_an_error(tmp_path: Path):
    truth = _write(tmp_path / "truth.json", _SEGMENTS)

    with pytest.raises(ConfigError, match="gone.wav"):
        single_pair(tmp_path / "gone.wav", truth)


def test_a_folder_pairs_recordings_with_ground_truth_by_name(tmp_path: Path):
    for name in ("b.mp3", "b.json", "a.mkv", "a.json"):
        _write(tmp_path / name, "x")

    pairs = discover_pairs(tmp_path)

    assert [(p.name, p.media.name, p.reference.name) for p in pairs] == [
        ("a", "a.mkv", "a.json"),
        ("b", "b.mp3", "b.json"),
    ]


@pytest.mark.parametrize(
    ("files", "named"),
    [
        (("a.wav", "a.json", "b.wav"), "b.wav"),
        (("a.wav", "a.json", "c.json"), "c.json"),
        (("a.wav", "a.mp3", "a.json"), "a.mp3"),
    ],
)
def test_an_unpaired_or_ambiguous_file_is_an_error_naming_it(
    tmp_path: Path, files, named
):
    for name in files:
        _write(tmp_path / name, "x")

    with pytest.raises(ConfigError, match=named.replace(".", r"\.")):
        discover_pairs(tmp_path)


def test_other_files_in_the_folder_are_ignored(tmp_path: Path):
    for name in ("a.wav", "a.json", "notes.txt", ".DS_Store"):
        _write(tmp_path / name, "x")

    assert [pair.name for pair in discover_pairs(tmp_path)] == ["a"]


def test_only_filters_the_folder_by_name(tmp_path: Path):
    """Filtering first, so an unrelated unpaired file does not block a run."""
    for name in ("a.wav", "a.json", "b.wav", "b.json", "stray.wav"):
        _write(tmp_path / name, "x")

    assert [pair.name for pair in discover_pairs(tmp_path, only=["b"])] == ["b"]


def test_only_naming_nothing_is_an_error(tmp_path: Path):
    for name in ("a.wav", "a.json"):
        _write(tmp_path / name, "x")

    with pytest.raises(ConfigError, match="zzz"):
        discover_pairs(tmp_path, only=["zzz"])


def test_an_empty_or_missing_folder_is_an_error(tmp_path: Path):
    with pytest.raises(ConfigError, match="no recordings"):
        discover_pairs(tmp_path)
    with pytest.raises(ConfigError, match="not a directory"):
        discover_pairs(tmp_path / "missing")
