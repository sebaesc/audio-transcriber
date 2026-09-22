"""Entry point for `python -m audio_transcriber.evaluation`.

Kept separate from the pipeline CLI: evaluation is a development tool, not part
of transcribing audio.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from audio_transcriber.errors import AudioTranscriberError
from audio_transcriber.evaluation.runner import Report, compare, evaluate


def _parse_option(raw: str) -> tuple[str, object]:
    """Parse one `key=value` backend option, reading the value as JSON."""
    key, separator, value = raw.partition("=")
    if not separator or not key:
        raise SystemExit(f"error: expected key=value, got {raw!r}")
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


def main(argv: list[str] | None = None) -> int:
    """Run an evaluation, or compare two existing reports."""
    parser = argparse.ArgumentParser(
        prog="python -m audio_transcriber.evaluation",
        description="Measure transcription quality against the reference corpus.",
    )
    parser.add_argument("--backend", "-b", help="Backend to evaluate.")
    parser.add_argument("--label", "-l", help="Name for this configuration.")
    parser.add_argument(
        "--option", action="append", metavar="KEY=VALUE", help="Backend option."
    )
    parser.add_argument(
        "--only", action="append", metavar="NAME", help="Limit to these recordings."
    )
    parser.add_argument(
        "--out", "-o", type=Path, default=Path("eval"), help="Report directory."
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE", "CANDIDATE"),
        type=Path,
        help="Compare two existing reports instead of running one.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    if args.compare:
        baseline, candidate = (_load(path) for path in args.compare)
        result = compare(baseline, candidate)
        print(json.dumps(result, indent=2))
        return 0 if result["improved_everywhere"] else 1

    if not args.backend or not args.label:
        parser.error("--backend and --label are required unless --compare is given")

    config = dict(_parse_option(item) for item in args.option or [])
    try:
        report = evaluate(
            backend_name=args.backend,
            label=args.label,
            backend_config=config,
            only=args.only,
            transcript_dir=args.out / "transcripts",
        )
    except AudioTranscriberError as error:
        print(f"error: {error}", file=sys.stderr)
        return 3

    path = report.write(args.out / f"{args.label}.json")
    print(json.dumps(report.to_dict()["summary"], indent=2))
    print(f"report written to {path}", file=sys.stderr)
    return 0


def _load(path: Path) -> Report:
    """Rebuild enough of a report from JSON to compare it."""
    from audio_transcriber.evaluation.runner import RecordingResult
    from audio_transcriber.evaluation.scoring import Score

    data = json.loads(path.read_text(encoding="utf-8"))
    report = Report(
        label=data["label"],
        backend=data["backend"],
        model_version=data.get("model_version", ""),
        config=data.get("config", {}),
        created_at=data.get("created_at", ""),
    )
    for item in data["recordings"]:
        raw = item.get("score")
        report.results.append(
            RecordingResult(
                name=item["name"],
                duration_s=item["duration_s"],
                elapsed_s=item["elapsed_s"],
                detected_languages=item.get("detected_languages", []),
                segments=item.get("segments", 0),
                score=Score(
                    wer=raw["wer"],
                    substitutions=raw["substitutions"],
                    deletions=raw["deletions"],
                    insertions=raw["insertions"],
                    hits=raw["hits"],
                    reference_words=raw["reference_words"],
                    hypothesis_words=raw["hypothesis_words"],
                    reference_spanish_share=raw["reference_spanish_share"],
                    hypothesis_spanish_share=raw["hypothesis_spanish_share"],
                )
                if raw
                else None,
                error=item.get("error"),
            )
        )
    return report


if __name__ == "__main__":
    raise SystemExit(main())
