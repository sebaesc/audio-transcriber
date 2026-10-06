"""A side-by-side table of every model's scores, recording by recording.

`compare` judges one candidate against one baseline. This reads any number of
reports instead, corpus and custom alike, and lays their scores out as one
Markdown table per source: a block of rows per recording, one row per model,
the transcription and diarization metrics as columns, and each model's means
at the end.

It reads the reports as JSON and nothing else, so no backend runs and no
credits are spent: a model's transcripts are scored once, then tabulated as
often as needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audio_transcriber.errors import ConfigError
from audio_transcriber.evaluation.runner import CUSTOM_DATASET

#: Where the table is written, under the report directory.
TABLE_FILENAME = "comparison.md"


@dataclass(frozen=True)
class Column:
    """One metric column: where its value is in a recording, and how to read it."""

    header: str
    section: str  # `score`, `diarization`, `language`, or `` for the recording
    key: str
    lower_is_better: bool | None
    meaning: str
    definition: str
    decimals: int = 3

    @property
    def field(self) -> str:
        """Where the value is in a report's recording."""
        if self.section in ("", "score"):
            return ".".join(filter(None, (self.section, self.key)))
        return f"<talkbank|ground_truth>.{self.section}.{self.key}"

    @property
    def better(self) -> str:
        if self.lower_is_better is None:
            return _NEITHER.get(self.key, "neither")
        return "lower" if self.lower_is_better else "higher"


#: How to read a column with no better direction.
_NEITHER = {
    "coverage": "nearest 1.0",
    "mixed_share": "neither: a property of the recording",
}


COLUMNS = (
    Column(
        "WER",
        "score",
        "wer",
        True,
        "word error rate",
        "Word error rate: (substitutions + deletions + insertions) / reference "
        "words, after both sides are lowercased, stripped of punctuation and "
        "accent-folded. It rewards emitting less, so read it with Retained.",
    ),
    Column(
        "Retained",
        "score",
        "retained",
        False,
        "reference words recovered",
        "Reference words transcribed as anything, a hit or a substitution, rather "
        "than deleted, / reference words. The guard against a WER gain bought by "
        "dropping audio.",
    ),
    Column(
        "Coverage",
        "score",
        "coverage",
        None,
        "hypothesis words per reference word; 1 is right, below drops, above adds",
        "Hypothesis words / reference words. Near 1.0 is right; well above 1.0 "
        "means the transcriber invents words. Invented and dropped words offset "
        "each other, so it is not a measure of dropped content.",
    ),
    Column(
        "Lang mix err",
        "score",
        "language_mix_error",
        True,
        "gap between the reference's and the hypothesis's share of Spanish",
        "|hypothesis - reference| share of Spanish, counted over marker function "
        "words (que, pero, the, because). It catches English written out as "
        "Spanish and the reverse, but only in aggregate: per-segment errors that "
        "balance out go unseen.",
    ),
    Column(
        "DER",
        "diarization",
        "der",
        True,
        "diarization error rate",
        "Diarization error rate: Missed + False alarm + Spk conf, as shares of "
        "reference speech, with a 0.25 s collar. Scores the transcript's segments, "
        "so segment boundaries inflate it. Only when the run diarized.",
    ),
    Column(
        "Missed",
        "diarization",
        "missed",
        True,
        "speech with no speaker",
        "Reference speech with no labelled segment over it: pauses the reference "
        "spans and the transcript does not, and speech never transcribed.",
    ),
    Column(
        "False alarm",
        "diarization",
        "false_alarm",
        True,
        "speaker on no speech",
        "Labelled segments over no reference speech.",
    ),
    Column(
        "Spk conf",
        "diarization",
        "confusion",
        True,
        "speech on the wrong speaker",
        "Reference speech attributed to the wrong speaker, after mapping our "
        "labels onto the reference speakers by best agreement. The number to "
        "read for speaker-label quality.",
    ),
    Column(
        "Lang acc",
        "language",
        "accuracy",
        False,
        "share of speech labeled with the right language",
        "Share of English and Spanish reference speech whose overlapping segment "
        "carries the right spoken_language, by time. Mixed utterances are left "
        "out. Stronger than Lang mix err: per-segment errors cannot cancel out.",
    ),
    Column(
        "Mixed",
        "language",
        "mixed_share",
        None,
        "share of the reference's speech that mixes both languages",
        "Share of reference speech in utterances holding both languages. A "
        "property of the recording, not of the model: no single language label "
        "is right for it, so it bounds what Lang acc can reach.",
    ),
    Column(
        "RTF",
        "",
        "realtime_factor",
        False,
        "audio seconds per second of compute",
        "Real-time factor: audio duration / wall-clock transcription time. 27.6 "
        "means 27.6 s of audio per second. Absent for a re-scored transcript, "
        "which was not timed.",
        decimals=1,
    ),
)


def metric_definitions() -> str:
    """What each column of the table means, where it comes from, and its direction."""
    lines = ["# Metrics", ""]
    for column in COLUMNS:
        lines += [
            f"## {column.header}",
            "",
            column.definition,
            "",
            f"- Better: {column.better}",
            f"- Report field: `{column.field}`",
            "",
        ]
    lines += [
        "Diarization and language metrics come from annotations: TalkBank's for "
        "the corpus (--talkbank) and yours for a custom ground truth. A – in the "
        "table means the metric was not measured.",
    ]
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class _Report:
    name: str
    custom: bool
    backend: str
    model_version: str
    config: dict[str, Any]
    recordings: list[dict[str, Any]]

    @property
    def annotation_key(self) -> str:
        return "ground_truth" if self.custom else "talkbank"

    @property
    def engine(self) -> str:
        """The backend and its model, short: `faster-whisper large-v3`."""
        model = self.model_version.rpartition("/")[2]
        if self.backend == model or model == "unknown":
            return self.backend
        return f"{self.backend} {model}"


#: The suffix of a report kept out of version control: a one-off experiment.
LOCAL_SUFFIX = ".local.json"


def discover_reports(out: Path) -> list[Path]:
    """The kept reports under a report directory: the corpus's, then the custom.

    `*.local.json` reports are one-off experiments and are left out; name them
    explicitly to tabulate them.
    """
    found = sorted(out.glob("*.json")) + sorted((out / "custom").glob("*.json"))
    return [path for path in found if not path.name.endswith(LOCAL_SUFFIX)]


def comparison_table(paths: list[Path]) -> str:
    """Tabulate the reports at `paths` as Markdown.

    Raises:
        ConfigError: If there are no reports, or one is not a report.
    """
    if not paths:
        raise ConfigError("no reports to tabulate")
    reports = _named([_read(path) for path in paths])
    lines = ["# Model comparison", ""]
    for title, custom in (("Corpus", False), ("Custom ground truth", True)):
        section = [report for report in reports if report.custom is custom]
        if section:
            lines += _section(title, section)
    lines += _models(reports)
    lines += _legend()
    return "\n".join(lines) + "\n"


def _read(path: Path) -> tuple[Path, dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ConfigError(f"cannot read report {path}: {error}") from error
    if not isinstance(data, dict) or "recordings" not in data or "label" not in data:
        raise ConfigError(f"{path} is not an evaluation report")
    return path, data


def _named(items: list[tuple[Path, dict[str, Any]]]) -> list[_Report]:
    """Build the reports, naming each by its label, or its file where labels clash."""
    reports = []
    for path, data in items:
        custom = data.get("corpus", {}).get("dataset") == CUSTOM_DATASET
        clashes = sum(
            1
            for _, other in items
            if other["label"] == data["label"]
            and (other.get("corpus", {}).get("dataset") == CUSTOM_DATASET) is custom
        )
        name = data["label"] if clashes == 1 else f"{data['label']} ({path.name})"
        reports.append(
            _Report(
                name=name,
                custom=custom,
                backend=data.get("backend", "unknown"),
                model_version=data.get("model_version") or "unknown",
                config=data.get("config", {}),
                recordings=data["recordings"],
            )
        )
    return sorted(reports, key=lambda report: report.name)


def _section(title: str, reports: list[_Report]) -> list[str]:
    rows = []
    names = sorted({item["name"] for report in reports for item in report.recordings})
    for name in names:
        for report in reports:
            item = next((i for i in report.recordings if i["name"] == name), None)
            if item is not None:
                rows.append([name, report.name, report.engine, *_cells(item, report)])
    means = []
    for report in reports:
        scored = [item for item in report.recordings if not item.get("error")]
        count = f"{len(scored)}/{len(report.recordings)}"
        values = [
            _format(_mean(scored, report, column), column.decimals)
            for column in COLUMNS
        ]
        means.append([report.name, report.engine, count, *values])
    metrics = [_arrowed(column) for column in COLUMNS]
    return [
        f"## {title}",
        "",
        *_render(["Recording", "Model", "Backend", *metrics], rows, numeric_from=3),
        "",
        "### Means over each model's scored recordings",
        "",
        *_render(["Model", "Backend", "Scored", *metrics], means, numeric_from=2),
        "",
    ]


def _cells(item: dict[str, Any], report: _Report) -> list[str]:
    if item.get("error"):
        return ["error"] + ["–"] * (len(COLUMNS) - 1)
    return [
        _format(_value(item, report, column), column.decimals) for column in COLUMNS
    ]


def _value(item: dict[str, Any], report: _Report, column: Column) -> float | None:
    if column.section == "":
        value = item.get(column.key)
    elif column.section == "score":
        value = (item.get("score") or {}).get(column.key)
    else:
        annotation = item.get(report.annotation_key) or {}
        value = (annotation.get(column.section) or {}).get(column.key)
    return value if isinstance(value, int | float) else None


def _mean(items: list[dict[str, Any]], report: _Report, column: Column) -> float | None:
    values = [_value(item, report, column) for item in items]
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _format(value: float | None, decimals: int = 3) -> str:
    """A fixed number of decimals, so the points line up down a column."""
    return "–" if value is None else f"{value:.{decimals}f}"


def _arrowed(column: Column) -> str:
    arrow = {True: " ↓", False: " ↑", None: ""}[column.lower_is_better]
    return column.header + arrow


def _render(
    headers: list[str], rows: list[list[str]], numeric_from: int | None = None
) -> list[str]:
    """Render a Markdown table padded so its columns line up in plain text too.

    Columns from `numeric_from` on are right-aligned, the rest left-aligned.
    """
    cells = [[text.replace("|", "\\|") for text in row] for row in [headers, *rows]]
    widths = [max(3, *(len(row[i]) for row in cells)) for i in range(len(headers))]
    right = [numeric_from is not None and i >= numeric_from for i in range(len(widths))]

    def line(row: list[str]) -> str:
        padded = (
            text.rjust(width) if is_right else text.ljust(width)
            for text, width, is_right in zip(row, widths, right, strict=True)
        )
        return "| " + " | ".join(padded) + " |"

    rule = (
        "| "
        + " | ".join(
            "-" * (width - 1) + ":" if is_right else ":" + "-" * (width - 1)
            for width, is_right in zip(widths, right, strict=True)
        )
        + " |"
    )
    return [line(cells[0]), rule, *(line(row) for row in cells[1:])]


def _models(reports: list[_Report]) -> list[str]:
    rows = []
    for report in reports:
        diarization = report.config.get("diarization")
        rows.append(
            [
                report.name,
                "custom" if report.custom else "corpus",
                report.backend,
                report.model_version,
                "–" if diarization is None else ("on" if diarization else "off"),
            ]
        )
    headers = ["Model", "Source", "Backend", "Model version", "Diarization"]
    return ["## Models", "", *_render(headers, rows), ""]


def _legend() -> list[str]:
    lines = ["## Metrics", ""]
    for column in COLUMNS:
        lines.append(f"- **{_arrowed(column)}**: {column.meaning}.")
    lines += [
        "",
        "↓ lower is better, ↑ higher is better. – means not measured: no "
        "diarization, no annotations, or no timing for a re-scored transcript. "
        "Corpus annotations are TalkBank's, and are scored only with --talkbank. "
        "Means cover only the recordings each model scored, so compare them "
        "only where the Scored counts match. For each metric in full: "
        "python -m audio_transcriber.evaluation --metrics",
    ]
    return lines
