"""Entry point for `python -m audio_transcriber.evaluation`.

Kept separate from the pipeline CLI: evaluation is a development tool, not part
of transcribing audio.

Four mutually exclusive modes:

- **corpus** (the default): run a backend over the reference corpus;
- **custom**: run a backend over the user's recordings, `--audio FILE
  --reference FILE` or `--references DIR`, scoring each against its ground truth;
- **existing**: score transcripts the pipeline already wrote, `--transcript FILE
  --reference FILE` or `--transcripts DIR --references DIR`, with no backend run;
- **compare**: compare two existing reports.

Custom and existing reports go under `<out>/custom/`, apart from the corpus's.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from audio_transcriber.errors import AudioTranscriberError
from audio_transcriber.evaluation.reference import (
    discover_pairs,
    single_pair,
    transcript_pairs,
)
from audio_transcriber.evaluation.runner import (
    Report,
    compare,
    evaluate,
    evaluate_custom,
    score_transcripts,
)

#: The subdirectory of `--out` holding reports scored against a custom ground truth.
CUSTOM_SUBDIRECTORY = "custom"


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
        description="Measure transcription quality against the reference corpus, "
        "or against your own ground truth.",
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
        "--talkbank",
        action="store_true",
        help="Also score speakers and per-segment language against TalkBank's "
        "annotations. Needs the transcripts archive; see docs/evaluation.md.",
    )
    custom = parser.add_argument_group(
        "your own recordings",
        "Score against a ground truth you wrote: a JSON array of transcript "
        "segments. See docs/evaluation.md.",
    )
    custom.add_argument(
        "--audio", type=Path, metavar="FILE", help="A recording (audio or video)."
    )
    custom.add_argument(
        "--reference",
        type=Path,
        metavar="FILE",
        help="The ground truth for --audio or --transcript.",
    )
    custom.add_argument(
        "--references",
        type=Path,
        metavar="DIR",
        help="A folder of ground truths, name.json. With no --transcripts, each "
        "is paired with the recording of the same name in the folder.",
    )
    custom.add_argument(
        "--transcript",
        type=Path,
        metavar="FILE",
        help="Score this existing transcript instead of running a backend.",
    )
    custom.add_argument(
        "--transcripts",
        type=Path,
        metavar="DIR",
        help="Score a pipeline run's output folder instead of running a backend.",
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

    mode = _mode(args, parser)
    if mode == "compare":
        baseline, candidate = (_load(path) for path in args.compare)
        result = compare(baseline, candidate)
        print(json.dumps(result, indent=2))
        return 0 if result["improved_everywhere"] else 1

    out = args.out if mode == "corpus" else args.out / CUSTOM_SUBDIRECTORY
    config = dict(_parse_option(item) for item in args.option or [])
    try:
        if mode == "corpus":
            report = evaluate(
                backend_name=args.backend,
                label=args.label,
                backend_config=config,
                only=args.only,
                transcript_dir=out / "transcripts",
                with_talkbank=args.talkbank,
            )
        elif mode == "custom":
            pairs = (
                single_pair(args.audio, args.reference)
                if args.audio
                else discover_pairs(args.references, only=args.only)
            )
            report = evaluate_custom(
                backend_name=args.backend,
                label=args.label,
                pairs=pairs,
                backend_config=config,
                transcript_dir=out / "transcripts",
            )
        else:
            pairs = (
                single_pair(args.transcript, args.reference)
                if args.transcript
                else transcript_pairs(args.transcripts, args.references, args.only)
            )
            report = score_transcripts(
                label=args.label, pairs=pairs, transcript_dir=out / "transcripts"
            )
    except AudioTranscriberError as error:
        print(f"error: {error}", file=sys.stderr)
        return 3

    path = report.write(out / f"{args.label}.json")
    print(json.dumps(report.to_dict()["summary"], indent=2))
    print(f"report written to {path}", file=sys.stderr)
    return 0


def _mode(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    """Work out which mode the flags ask for, rejecting any misuse.

    Returns:
        `compare`, `corpus`, `custom` or `existing`.
    """
    inputs = {
        "--audio": args.audio,
        "--reference": args.reference,
        "--references": args.references,
        "--transcript": args.transcript,
        "--transcripts": args.transcripts,
    }
    if args.compare:
        used = [flag for flag, value in inputs.items() if value]
        if used:
            parser.error(f"--compare reads two reports; it takes no {', '.join(used)}")
        return "compare"

    if args.audio and (args.transcript or args.transcripts):
        parser.error(
            "use only one of --audio (run a backend on a recording) and "
            "--transcript/--transcripts (score a transcript already made)"
        )
    if args.transcript and args.transcripts:
        parser.error("use only one of --transcript and --transcripts")
    if args.audio and args.references:
        parser.error(
            "--audio takes one recording with --reference FILE; --references DIR "
            "takes a folder. Use one or the other."
        )

    single = args.audio or args.transcript
    if single and not args.reference:
        flag = "--audio" if args.audio else "--transcript"
        parser.error(f"{flag} needs its ground truth: --reference FILE")
    if args.reference and not single:
        parser.error(
            "--reference FILE goes with --audio or --transcript; for a folder, "
            "use --references DIR"
        )
    if args.transcripts and not args.references:
        parser.error("--transcripts needs the ground-truth folder: --references DIR")

    if args.transcript or args.transcripts:
        mode = "existing"
    elif args.audio or args.references:
        mode = "custom"
    else:
        mode = "corpus"

    if mode != "corpus" and args.talkbank:
        parser.error("--talkbank scores the corpus only")
    if single and args.only:
        parser.error("--only filters a folder (--references or --transcripts)")
    if mode == "existing":
        if args.backend:
            parser.error(
                "--backend is not used when scoring existing transcripts; it is "
                "read from the run's manifest"
            )
        if args.option:
            parser.error("--option is not used when scoring existing transcripts")
        if not args.label:
            parser.error("--label is required")
    elif mode == "custom":
        if not args.backend:
            parser.error(
                "--backend is required to transcribe your recordings; to score "
                "transcripts already made, use --transcript or --transcripts"
            )
        if not args.label:
            parser.error("--label is required")
    elif not args.backend or not args.label:
        parser.error("--backend and --label are required unless --compare is given")
    return mode


def _load(path: Path) -> Report:
    """Rebuild enough of a report from JSON to compare it."""
    from audio_transcriber.evaluation.runner import CUSTOM_DATASET, RecordingResult
    from audio_transcriber.evaluation.scoring import Score

    data = json.loads(path.read_text(encoding="utf-8"))
    report = Report(
        label=data["label"],
        backend=data["backend"],
        model_version=data.get("model_version", ""),
        config=data.get("config", {}),
        created_at=data.get("created_at", ""),
        ground_truth=data.get("corpus", {}).get("dataset") == CUSTOM_DATASET,
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
                annotation=item.get(report.annotation_key),
                reference=item.get("reference"),
                error=item.get("error"),
            )
        )
    return report


if __name__ == "__main__":
    raise SystemExit(main())
