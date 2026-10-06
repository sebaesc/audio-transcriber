"""Tests for the side-by-side table of reports, and its CLI mode."""

import json
from pathlib import Path

import pytest

from audio_transcriber.errors import ConfigError
from audio_transcriber.evaluation.__main__ import main
from audio_transcriber.evaluation.runner import CUSTOM_DATASET
from audio_transcriber.evaluation.table import (
    COLUMNS,
    comparison_table,
    discover_reports,
)


def _recording(name: str, wer: float, annotation: dict | None = None) -> dict:
    return {
        "name": name,
        "realtime_factor": 5.0,
        "score": {
            "wer": wer,
            "retained": 0.9,
            "coverage": 1.0,
            "language_mix_error": 0.05,
        },
        "talkbank": annotation,
        "error": None,
    }


def _write(path: Path, label: str, recordings: list[dict], custom=False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = CUSTOM_DATASET if custom else "corpus"
    if custom:
        for item in recordings:
            item["ground_truth"] = item.pop("talkbank")
    data = {
        "label": label,
        "backend": "faster-whisper",
        "model_version": "faster-whisper-1.2.1/large-v3",
        "config": {"diarization": True},
        "corpus": {"dataset": dataset},
        "recordings": recordings,
    }
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def out(tmp_path: Path) -> Path:
    out = tmp_path / "eval"
    diarized = {"diarization": {"der": 0.25}, "language": {"accuracy": 0.8}}
    _write(out / "whisper.json", "whisper", [_recording("a", 0.4, diarized)])
    scribe = _recording("b", 0.2)
    _write(out / "scribe.json", "scribe", [_recording("a", 0.3), scribe])
    _write(out / "tried.local.json", "tried", [_recording("a", 0.9)])
    _write(out / "custom" / "mine.json", "mine", [_recording("x", 0.1, diarized)], True)
    return out


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip("|").split("|")]


def _rows(table: str, first: str) -> list[list[str]]:
    rows = [_cells(line) for line in table.splitlines() if line.startswith("|")]
    return [row for row in rows if row[0] == first]


def test_discovery_keeps_corpus_then_custom_and_drops_local_reports(out: Path):
    names = [path.name for path in discover_reports(out)]

    assert names == ["scribe.json", "whisper.json", "mine.json"]


def test_each_recording_has_one_row_per_model_with_its_metrics(out: Path):
    table = comparison_table(discover_reports(out))

    rows = _rows(table, "a")
    assert [row[1] for row in rows] == ["scribe", "whisper"]
    assert rows[1][2] == "faster-whisper large-v3"
    assert {"0.400", "0.250", "0.800"} <= set(rows[1])
    assert len(_rows(table, "b")) == 1


def test_corpus_and_custom_are_separate_sections(out: Path):
    table = comparison_table(discover_reports(out))

    corpus, custom = table.split("## Custom ground truth")
    assert not _rows(corpus, "x")
    assert _rows(custom, "x")[0][1] == "mine"


def test_means_count_only_the_scored_recordings(out: Path):
    scribe = _write(
        out / "scribe.json",
        "scribe",
        [_recording("a", 0.3), {"name": "b", "error": "boom"}],
    )

    table = comparison_table([scribe])

    assert _rows(table, "b")[0][3] == "error"
    assert _rows(table, "scribe")[0][2:4] == ["1/2", "0.300"]


def test_unmeasured_metrics_show_a_dash(out: Path):
    table = comparison_table(discover_reports(out))

    scribe_a = _rows(table, "a")[0]
    assert scribe_a.count("–") == 6  # diarization and language


def test_clashing_labels_are_told_apart_by_file(out: Path):
    other = _write(out / "again.json", "whisper", [_recording("a", 0.5)])

    table = comparison_table([out / "whisper.json", other])

    assert "whisper (again.json)" in table and "whisper (whisper.json)" in table


def test_every_line_of_a_table_is_the_same_width(out: Path):
    table = comparison_table(discover_reports(out))

    blocks = [[]]
    for line in table.splitlines():
        if line.startswith("|"):
            blocks[-1].append(line)
        elif blocks[-1]:
            blocks.append([])
    for block in filter(None, blocks):
        assert len({len(line) for line in block}) == 1
        pipes = {tuple(i for i, c in enumerate(line) if c == "|") for line in block}
        assert len(pipes) == 1


def test_a_file_that_is_not_a_report_is_refused(tmp_path: Path):
    path = tmp_path / "x.json"
    path.write_text("[]")

    with pytest.raises(ConfigError, match="not an evaluation report"):
        comparison_table([path])


def test_the_cli_prints_and_writes_the_table(out: Path, capsys):
    code = main(["--table", "--out", str(out)])

    assert code == 0
    written = (out / "comparison.md").read_text()
    assert written == capsys.readouterr().out
    assert _rows(written, "a")[1][1] == "whisper"


def test_the_cli_takes_explicit_reports_local_ones_included(out: Path, capsys):
    code = main(["--table", str(out / "tried.local.json"), "--out", str(out)])

    assert code == 0
    assert _rows(capsys.readouterr().out, "a")[0][1] == "tried"


def test_the_cli_exits_3_when_there_is_nothing_to_tabulate(tmp_path: Path):
    assert main(["--table", "--out", str(tmp_path / "empty")]) == 3


@pytest.mark.parametrize(
    "extra", [["-b", "scripted"], ["-l", "x"], ["--compare", "a", "b"], ["--talkbank"]]
)
def test_the_table_takes_no_run_flags(extra: list[str], tmp_path: Path):
    with pytest.raises(SystemExit) as raised:
        main(["--table", "--out", str(tmp_path), *extra])

    assert raised.value.code == 2


def test_metrics_defines_every_column_of_the_table(capsys):
    assert main(["--metrics"]) == 0

    printed = capsys.readouterr().out
    for column in COLUMNS:
        assert f"## {column.header}\n" in printed
        assert f"`{column.field}`" in printed
    assert "`score.wer`" in printed
    assert "`<talkbank|ground_truth>.diarization.confusion`" in printed


@pytest.mark.parametrize("extra", [["--table"], ["-b", "scripted"], ["--talkbank"]])
def test_metrics_takes_no_mode(extra: list[str]):
    with pytest.raises(SystemExit) as raised:
        main(["--metrics", *extra])

    assert raised.value.code == 2
