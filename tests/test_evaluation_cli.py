"""Tests for the evaluation CLI's modes, flags and exit codes.

Corpus mode, a custom ground truth and scoring an existing transcript are
mutually exclusive. Misuse exits 2 (argparse), a run that cannot start exits 3.
"""

import json
from pathlib import Path

import pytest

from audio_transcriber.evaluation import runner
from audio_transcriber.evaluation.__main__ import main
from tests.test_evaluation_custom import HYPOTHESIS, _pair

pytest.importorskip("jiwer")

from tests.test_evaluation_custom import scripted_environment  # noqa: E402, F401


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    """A folder of two recordings, each with its ground truth."""
    directory = tmp_path / "recordings"
    directory.mkdir()
    for name in ("a", "b"):
        _pair(directory, name)
    return directory


@pytest.fixture
def run_folder(tmp_path: Path) -> Path:
    """A pipeline run's output: one transcript, no manifest."""
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "a.json").write_text(json.dumps([s.to_dict() for s in HYPOTHESIS]))
    return directory


def _report(out: Path, label: str) -> dict:
    return json.loads((out / "custom" / f"{label}.json").read_text())


def test_a_single_pair_runs_the_backend_and_writes_under_custom(tmp_path: Path):
    pair = _pair(tmp_path)
    out = tmp_path / "eval"

    code = main(
        ["-b", "scripted", "-l", "one", "--out", str(out)]
        + ["--audio", str(pair.media), "--reference", str(pair.reference)]
    )

    assert code == 0
    report = _report(out, "one")
    assert report["recordings"][0]["name"] == "talk"
    assert report["corpus"]["dataset"] == runner.CUSTOM_DATASET
    assert (out / "custom" / "transcripts" / "talk.reference.txt").is_file()


def test_a_folder_of_pairs_can_be_filtered_with_only(folder: Path, tmp_path: Path):
    out = tmp_path / "eval"

    code = main(
        ["-b", "scripted", "-l", "dir", "--out", str(out)]
        + ["--references", str(folder), "--only", "b"]
    )

    assert code == 0
    assert [r["name"] for r in _report(out, "dir")["recordings"]] == ["b"]


def test_an_existing_transcript_is_scored_without_a_backend(
    run_folder: Path, folder: Path, tmp_path: Path
):
    out = tmp_path / "eval"

    code = main(
        ["-l", "again", "--out", str(out)]
        + ["--transcript", str(run_folder / "a.json")]
        + ["--reference", str(folder / "a.json")]
    )

    assert code == 0
    report = _report(out, "again")
    assert report["backend"] == runner.UNKNOWN
    assert report["recordings"][0]["score"]["wer"] == pytest.approx(0.1)


def test_a_run_folder_is_scored_against_a_reference_folder(
    run_folder: Path, folder: Path, tmp_path: Path
):
    out = tmp_path / "eval"

    code = main(
        ["-l", "again", "--out", str(out)]
        + ["--transcripts", str(run_folder), "--references", str(folder)]
        + ["--only", "a"]
    )

    assert code == 0
    assert [r["name"] for r in _report(out, "again")["recordings"]] == ["a"]


def test_an_invalid_ground_truth_exits_3_naming_it(tmp_path: Path, capsys):
    pair = _pair(tmp_path, ground_truth=[{"start_time": 0.0, "end_time": 1.0}])

    code = main(
        ["-b", "scripted", "-l", "x", "--out", str(tmp_path / "eval")]
        + ["--audio", str(pair.media), "--reference", str(pair.reference)]
    )

    assert code == 3
    assert "talk.json" in capsys.readouterr().err


def test_an_unpaired_folder_entry_exits_3(folder: Path, tmp_path: Path, capsys):
    (folder / "stray.wav").write_bytes(b"")

    code = main(
        ["-b", "scripted", "-l", "x", "--out", str(tmp_path / "eval")]
        + ["--references", str(folder)]
    )

    assert code == 3
    assert "stray.wav" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "complaint"),
    [
        # Modes cannot be combined.
        (["-b", "s", "-l", "x", "--audio", "a", "--transcript", "t"], "one of"),
        (["-b", "s", "-l", "x", "--audio", "a", "--references", "d"], "--audio"),
        (["--compare", "a", "b", "--references", "d"], "--compare"),
        # Each file needs its partner.
        (["-b", "s", "-l", "x", "--audio", "a"], "--reference"),
        (["-l", "x", "--transcript", "t"], "--reference"),
        (["-l", "x", "--transcripts", "d"], "--references"),
        (["-b", "s", "-l", "x", "--reference", "r"], "--reference"),
        (["-l", "x", "--transcripts", "d", "--reference", "r"], "--references"),
        # Flags that belong to another mode.
        (["-b", "s", "-l", "x", "--references", "d", "--talkbank"], "--talkbank"),
        (["-l", "x", "--transcript", "t", "--reference", "r", "-b", "s"], "--backend"),
        (
            ["-l", "x", "--transcript", "t", "--reference", "r", "--option", "a=1"],
            "--option",
        ),
        (
            ["-b", "s", "-l", "x", "--audio", "a", "--reference", "r", "--only", "n"],
            "--only",
        ),
        # What every run needs.
        (["--references", "d", "-l", "x"], "--backend"),
        (["--transcript", "t", "--reference", "r"], "--label"),
    ],
)
def test_misuse_exits_2_and_says_why(argv, complaint, capsys):
    with pytest.raises(SystemExit) as caught:
        main(argv)

    assert caught.value.code == 2
    assert complaint in capsys.readouterr().err


def test_corpus_mode_still_needs_a_backend_and_label(capsys):
    with pytest.raises(SystemExit) as caught:
        main(["--only", "zeledon4"])

    assert caught.value.code == 2
    assert "--backend and --label" in capsys.readouterr().err
