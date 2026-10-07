"""Running an evaluation over the reference corpus, or the team's own audio.

The runner takes a backend name and config, transcribes every recording in the
corpus, scores each one, and writes a JSON report. Two reports can then be
compared to judge whether a change helped.

The recordings can instead be the user's own, each scored against a ground
truth they wrote (`evaluation.reference`), or transcripts the pipeline already
produced can be scored without running a backend at all. Either way the scores
and the report are the same as for the corpus, so `compare` works on them.

It goes through the registry and the backend interface, exactly as the pipeline
does, so what is measured is what a run would produce.
"""

from __future__ import annotations

import json
import logging
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from audio_transcriber import __version__, audio, pipeline, registry
from audio_transcriber.backend import TranscriptionBackend
from audio_transcriber.errors import AudioDecodeError, ConfigError
from audio_transcriber.evaluation import corpus, talkbank
from audio_transcriber.evaluation.annotation import (
    Utterance,
    diarization_error,
    diarization_scorable,
    language_confusion,
)
from audio_transcriber.evaluation.reference import GroundTruth, Pair, load_reference
from audio_transcriber.evaluation.scoring import Comparison, Score, score
from audio_transcriber.evaluation.text import clean_reference, normalize
from audio_transcriber.manifest import Manifest, utc_now
from audio_transcriber.schema import Segment, Transcript

_logger = logging.getLogger(__name__)

#: What a report scored against a user's ground truth names as its corpus.
CUSTOM_DATASET = "custom ground truth"

#: Backend and model version of a transcript whose run left no manifest.
UNKNOWN = "unknown"


@dataclass(slots=True)
class RecordingResult:
    """One recording's outcome.

    Attributes:
        name: Recording identifier.
        duration_s: Length of the audio.
        elapsed_s: Wall-clock time spent transcribing it.
        detected_languages: The distinct `spoken_language` values produced.
        segments: Number of segments in the transcript.
        score: How the transcript compared to the reference, or None if it failed.
        annotation: Speaker and language scores against timed annotations —
            TalkBank's, or the user's ground truth — or None when there are none.
        reference: The ground-truth file scored against, as `file` and
            `checksum`, or None for the corpus.
        error: Why it failed, if it did.
    """

    name: str
    duration_s: float
    elapsed_s: float = 0.0
    detected_languages: list[str] = field(default_factory=list)
    segments: int = 0
    score: Score | None = None
    annotation: dict[str, Any] | None = None
    reference: dict[str, str] | None = None
    error: str | None = None

    @property
    def talkbank(self) -> dict[str, Any] | None:
        """The corpus-mode name for `annotation`."""
        return self.annotation

    @talkbank.setter
    def talkbank(self, value: dict[str, Any] | None) -> None:
        self.annotation = value

    def to_dict(self, annotation_key: str = "talkbank") -> dict[str, Any]:
        """Return the result as JSON-serializable data.

        Args:
            annotation_key: The key the annotation scores are written under:
                `talkbank` for the corpus, `ground_truth` for the user's own.
        """
        data: dict[str, Any] = {
            "name": self.name,
            "duration_s": round(self.duration_s, 1),
            "elapsed_s": round(self.elapsed_s, 1),
            "realtime_factor": (
                round(self.duration_s / self.elapsed_s, 1) if self.elapsed_s else None
            ),
            "segments": self.segments,
            "detected_languages": self.detected_languages,
            "score": self.score.to_dict() if self.score else None,
            annotation_key: self.annotation,
        }
        if self.reference is not None:
            data["reference"] = self.reference
        data["error"] = self.error
        return data


@dataclass(slots=True)
class Report:
    """The outcome of one evaluation.

    Attributes:
        label: A name for this configuration, used when comparing reports.
        backend: Backend name.
        model_version: Model identifier the backend reported.
        config: Effective backend config.
        results: One entry per recording.
        created_at: When the evaluation ran.
        tool_version: Version of audio-transcriber used.
        ground_truth: Whether the recordings were scored against the user's own
            ground truth rather than the corpus.
    """

    label: str
    backend: str
    model_version: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    results: list[RecordingResult] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    tool_version: str = __version__
    ground_truth: bool = False

    @property
    def annotation_key(self) -> str:
        """Where annotation scores go in the report: named for their source."""
        return "ground_truth" if self.ground_truth else "talkbank"

    @property
    def scored(self) -> list[RecordingResult]:
        """Recordings that produced a transcript."""
        return [item for item in self.results if item.score is not None]

    def mean(self, attribute: str) -> float:
        """Return the mean of a score attribute across scored recordings.

        Args:
            attribute: A `Score` attribute or property name, e.g. `"wer"`.

        Returns:
            The mean, or 0.0 when nothing was scored.
        """
        values = [getattr(item.score, attribute) for item in self.scored]
        return sum(values) / len(values) if values else 0.0

    def mean_annotation(self, section: str, key: str) -> float | None:
        """Return the mean of an annotation score over recordings that have it.

        Args:
            section: `diarization` or `language`.
            key: A score within it, e.g. `confusion`.

        Returns:
            The mean, or None when no recording has that score — diarization,
            for one, is absent from any run that did not diarize.
        """
        values = [
            item.annotation[section][key]
            for item in self.results
            if item.annotation
            and item.annotation.get(section)
            and item.annotation[section].get(key) is not None
        ]
        return sum(values) / len(values) if values else None

    def _annotation_summary(self) -> dict[str, Any] | None:
        if not any(item.annotation for item in self.results):
            return None

        def rounded(value: float | None) -> float | None:
            return None if value is None else round(value, 4)

        return {
            "mean_der": rounded(self.mean_annotation("diarization", "der")),
            "mean_speaker_confusion": rounded(
                self.mean_annotation("diarization", "confusion")
            ),
            "mean_language_accuracy": rounded(
                self.mean_annotation("language", "accuracy")
            ),
            "mean_mixed_share": rounded(
                self.mean_annotation("language", "mixed_share")
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the report as JSON-serializable data."""
        summary: dict[str, Any] = {
            "recordings": len(self.results),
            "scored": len(self.scored),
            "mean_wer": round(self.mean("wer"), 4),
            "mean_retained": round(self.mean("retained"), 4),
            "mean_coverage": round(self.mean("coverage"), 4),
            "mean_language_mix_error": round(self.mean("language_mix_error"), 4),
        }
        corpus_info: dict[str, Any] = (
            {"dataset": CUSTOM_DATASET}
            if self.ground_truth
            else {"dataset": corpus.DATASET, "license": corpus.LICENSE}
        )
        annotation_summary = self._annotation_summary()
        if annotation_summary is not None:
            summary[self.annotation_key] = annotation_summary
            if not self.ground_truth:
                corpus_info["annotations"] = talkbank.CITATION
        return {
            "label": self.label,
            "created_at": self.created_at,
            "tool_version": self.tool_version,
            "backend": self.backend,
            "model_version": self.model_version,
            "config": self.config,
            "corpus": corpus_info,
            "summary": summary,
            "recordings": [item.to_dict(self.annotation_key) for item in self.results],
        }

    def write(self, path: Path) -> Path:
        """Write the report as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        return path


def evaluate(
    backend_name: str,
    label: str,
    backend_config: dict[str, Any] | None = None,
    only: list[str] | None = None,
    transcript_dir: Path | None = None,
    with_talkbank: bool = False,
) -> Report:
    """Transcribe and score every recording in the corpus.

    Args:
        backend_name: Backend to evaluate, resolved through the registry.
        label: A name for this configuration, e.g. `"baseline"`.
        backend_config: Backend options.
        only: Recording names to limit the run to. None evaluates all of them.
        transcript_dir: Where to write each hypothesis, for inspecting what the
            scores are actually measuring. None skips writing them.
        with_talkbank: Also score speakers and per-segment language against
            TalkBank's annotations, from the local archive.

    Returns:
        The report.

    Raises:
        ConfigError: If the corpus cannot be fetched.
        BackendRequirementError: If the backend is not ready to run.
    """
    backend = _ready_backend(backend_name, backend_config)

    recordings = corpus.load()
    if only:
        wanted = set(only)
        recordings = [item for item in recordings if item.name in wanted]

    # Loaded before transcribing anything, so a missing archive fails in
    # seconds rather than after the first half-hour recording.
    references = (
        {item.name: talkbank.load(item.name) for item in recordings}
        if with_talkbank
        else {}
    )

    report = _report_for(backend, label)

    for recording in recordings:
        _logger.info(
            "evaluating %s (%.1f min)", recording.name, recording.duration_s / 60
        )
        result = RecordingResult(name=recording.name, duration_s=recording.duration_s)
        report.results.append(result)
        transcript = _transcribe(backend, recording.audio_path, result)
        if transcript is None:
            continue

        _describe(result, transcript)
        hypothesis = normalize(transcript.text)
        result.score = score(clean_reference(recording.transcript), hypothesis)
        if recording.name in references:
            result.annotation = _talkbank_scores(
                references[recording.name], transcript, recording.duration_s
            )
        if transcript_dir is not None:
            _write_hypothesis(transcript_dir, recording.name, label, transcript)

    return report


def evaluate_custom(
    backend_name: str,
    label: str,
    pairs: list[Pair],
    backend_config: dict[str, Any] | None = None,
    transcript_dir: Path | None = None,
) -> Report:
    """Transcribe the user's own recordings and score them against their ground truth.

    Args:
        backend_name: Backend to evaluate, resolved through the registry.
        label: A name for this configuration.
        pairs: Each recording (audio or video) with its ground-truth file.
        backend_config: Backend options.
        transcript_dir: Where to write each hypothesis and its normalized
            reference, so `diff` shows which words cost the WER. None skips them.

    Returns:
        The report, shaped like a corpus report so `compare` accepts it.

    Raises:
        ConfigError: If a ground truth is invalid, or does not lie within its
            recording. Every one is read, and checked against its recording's
            length, before anything is transcribed.
        BackendRequirementError: If the backend is not ready to run.
    """
    references = {pair.name: load_reference(pair.reference) for pair in pairs}
    backend = _ready_backend(backend_name, backend_config)
    for pair in pairs:
        duration = _media_seconds(pair.media)
        if duration is not None:
            references[pair.name] = references[pair.name].fit(duration)
    report = _report_for(backend, label)
    report.ground_truth = True

    for pair in pairs:
        _logger.info("evaluating %s", pair.name)
        result = RecordingResult(name=pair.name, duration_s=0.0)
        report.results.append(result)
        transcript = _transcribe(backend, pair.media, result, measure=True)
        if transcript is not None:
            _score_against(
                result, transcript, references[pair.name], label, transcript_dir
            )

    return report


def score_transcripts(
    label: str,
    pairs: list[Pair],
    transcript_dir: Path | None = None,
) -> Report:
    """Score transcripts the pipeline already wrote, without running a backend.

    This is how a hosted backend's output is re-scored without paying for it
    again. The scores are the ones `evaluate_custom` gives the same transcript.

    The backend, model version and config come from the run's `manifest.json`
    beside the transcripts, when there is one, and are `unknown` otherwise. No
    audio is read. A recording's duration is the one the manifest records;
    without it, it is taken as the later of the last reference and transcript
    speech, which scores the same: nothing after it is speech on either side.
    The ground truth is checked against that duration, or, without one, against
    where the transcript's speech ends.

    Args:
        label: A name for this scoring.
        pairs: Each transcript file with its ground-truth file.
        transcript_dir: Where to write each hypothesis and its normalized
            reference. None skips them.

    Returns:
        The report, shaped like any other so `compare` accepts it.

    Raises:
        ConfigError: If a transcript, ground truth or manifest is invalid, or
            a ground truth does not lie within its recording. Everything is
            read before anything is scored.
    """
    references = {pair.name: load_reference(pair.reference) for pair in pairs}
    transcripts = {pair.name: _load_transcript(pair.media) for pair in pairs}
    manifest = _find_manifest(pairs)
    durations = _recorded_durations(manifest)
    for pair in pairs:
        duration = durations.get(pair.media.name)
        if duration is not None:
            references[pair.name] = references[pair.name].fit(duration)
            continue
        last_speech = _last_speech(transcripts[pair.name])
        if last_speech:
            references[pair.name].fit(
                last_speech, trim=False, bound="the transcript's speech"
            )

    report = Report(
        label=label,
        backend=manifest.backend if manifest else UNKNOWN,
        model_version=(manifest.model_version or UNKNOWN) if manifest else UNKNOWN,
        config=manifest.config if manifest else {},
        ground_truth=True,
    )
    for pair in pairs:
        transcript, reference = transcripts[pair.name], references[pair.name]
        duration = durations.get(pair.media.name)
        if duration is None:
            duration = max(reference.end_time, _last_speech(transcript))
        result = RecordingResult(name=pair.name, duration_s=duration)
        report.results.append(result)
        _score_against(result, transcript, reference, label, transcript_dir)
    return report


def _last_speech(transcript: Transcript) -> float:
    """When a transcript's last speech ends, or 0 if it has none."""
    return max((segment.end_time for segment in transcript.segments), default=0.0)


def _recorded_durations(manifest: Manifest | None) -> dict[str, float]:
    """Each transcript's audio duration as the run recorded it, by file name."""
    if manifest is None:
        return {}
    return {
        Path(record.transcript).name: record.duration
        for record in manifest.inputs
        if record.transcript and record.duration
    }


def _load_transcript(path: Path) -> Transcript:
    """Read a transcript file the pipeline wrote.

    Raises:
        ConfigError: If it is not valid JSON or not a valid transcript.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Transcript.from_json_data(data, source=str(path))
    except (OSError, ValueError, TypeError) as error:
        # SchemaError and JSONDecodeError are both ValueErrors.
        raise ConfigError(f"cannot score transcript {path}: {error}") from error


def _find_manifest(pairs: list[Pair]) -> Manifest | None:
    """Read the manifest of the run the transcripts came from, if there is one.

    Read only: the manifest is the pipeline's record, not the harness's.

    Raises:
        ConfigError: If the transcripts come from more than one folder, or the
            manifest there cannot be read.
    """
    folders = {pair.media.parent for pair in pairs}
    if len(folders) != 1:
        raise ConfigError(
            "transcripts to score together must come from one run folder, got "
            f"{', '.join(sorted(map(str, folders)))}"
        )
    path = folders.pop() / pipeline.MANIFEST_FILENAME
    if not path.is_file():
        _logger.info("no %s beside the transcripts; backend unknown", path.name)
        return None
    try:
        return Manifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ConfigError(f"cannot read run manifest {path}: {error}") from error


def _ready_backend(
    backend_name: str, backend_config: dict[str, Any] | None
) -> TranscriptionBackend:
    """Create a backend and check it, and ffmpeg, can run."""
    backend = registry.create(backend_name, **(backend_config or {}))
    audio.check_ffmpeg()
    backend.check_requirements()
    return backend


def _report_for(backend: TranscriptionBackend, label: str) -> Report:
    """Start a report recording what the backend is and how it is configured."""
    return Report(
        label=label,
        backend=backend.name,
        model_version=backend.model_version(),
        config=_recordable(backend.effective_config()),
    )


def _transcribe(
    backend: TranscriptionBackend,
    media: Path,
    result: RecordingResult,
    measure: bool = False,
) -> Transcript | None:
    """Transcribe one recording, recording the time taken or the failure.

    Args:
        backend: The backend to run.
        media: The recording, in any format ffmpeg reads.
        result: Where the elapsed time, and any error, are recorded.
        measure: Also set the result's duration from the decoded audio.

    Returns:
        The transcript, or None if it failed. One failing is not fatal.
    """
    started = time.monotonic()
    try:
        with audio.decoded(media) as decoded:
            if measure:
                result.duration_s = _wav_seconds(decoded)
            transcript = backend.transcribe(decoded, str(media))
    except Exception as error:  # noqa: BLE001 - one recording failing is not fatal
        result.error = f"{type(error).__name__}: {error}"
        transcript = None
    result.elapsed_s = time.monotonic() - started
    return transcript


def _media_seconds(media: Path) -> float | None:
    """Return a recording's length, or None if it cannot be decoded.

    Decoded once more when transcribed: decoding takes seconds, and checking
    every ground truth against its recording first means a wrong one stops the
    run before any transcription is paid for. A recording that will not decode
    is left for `_transcribe` to record as failed.
    """
    try:
        with audio.decoded(media) as decoded:
            return _wav_seconds(decoded)
    except AudioDecodeError:
        return None


def _wav_seconds(path: Path) -> float:
    """Return the length of a decoded WAV."""
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def _describe(result: RecordingResult, transcript: Transcript) -> None:
    """Record what a transcript holds, as distinct from how it scored."""
    result.segments = len(transcript.segments)
    result.detected_languages = sorted(
        {s.spoken_language for s in transcript.segments if s.spoken_language}
    )


def _score_against(
    result: RecordingResult,
    transcript: Transcript,
    reference: GroundTruth,
    label: str,
    transcript_dir: Path | None,
) -> None:
    """Score a transcript against a user's ground truth, into `result`."""
    _describe(result, transcript)
    result.score = score(reference.text, normalize(transcript.text))
    result.annotation = _ground_truth_scores(
        reference, list(transcript.segments), result.duration_s
    )
    result.reference = reference.traced
    if transcript_dir is not None:
        _write_hypothesis(transcript_dir, result.name, label, transcript)
        (transcript_dir / f"{result.name}.reference.txt").write_text(
            reference.text, encoding="utf-8"
        )


def _write_hypothesis(
    transcript_dir: Path, name: str, label: str, transcript: Transcript
) -> None:
    """Write the scored hypothesis, as normalized text and as segments."""
    transcript_dir.mkdir(parents=True, exist_ok=True)
    (transcript_dir / f"{name}.{label}.txt").write_text(
        normalize(transcript.text), encoding="utf-8"
    )
    # The segments too, not just the text: timing and density are what
    # expose a hallucination, and the flattened text has neither.
    (transcript_dir / f"{name}.{label}.json").write_text(
        json.dumps(transcript.to_json_data(), indent=1, ensure_ascii=False),
        encoding="utf-8",
    )


def _talkbank_scores(
    reference: talkbank.ChatTranscript, transcript: Any, duration: float
) -> dict[str, Any]:
    """Score one transcript's speakers and languages against TalkBank."""
    return _annotation_scores(
        list(reference.utterances), list(transcript.segments), duration
    )


def _ground_truth_scores(
    reference: GroundTruth, segments: list[Segment], duration: float
) -> dict[str, Any]:
    """Score one transcript's speakers and languages against a ground truth.

    Unlike TalkBank, which always has both, a user's ground truth may label
    neither, and a dimension it does not label is skipped with the reason.
    """
    return _annotation_scores(
        reference.utterances,
        segments,
        duration,
        speakers_unscorable=reference.speakers_unscorable(),
        languages_unscorable=reference.languages_unscorable(),
        undiarized="the transcript has no speaker labels; the run did not diarize",
    )


def _annotation_scores(
    utterances: list[Utterance],
    segments: list[Segment],
    duration: float,
    speakers_unscorable: str | None = None,
    languages_unscorable: str | None = None,
    undiarized: str | None = None,
) -> dict[str, Any]:
    """Score speakers and languages against timed reference utterances.

    Args:
        utterances: The reference.
        segments: The transcript's segments.
        duration: Length of the recording in seconds.
        speakers_unscorable: Why the reference cannot score speakers, if so.
        languages_unscorable: Why it cannot score languages, if so.
        undiarized: The reason to give when the transcript has no speakers.
            None leaves diarization absent without one, as a corpus run that
            did not diarize always has.

    Returns:
        `diarization` and `language` scores, None where skipped, with a
        `<dimension>_skipped` reason beside any that has one.
    """
    scores: dict[str, Any] = {"diarization": None, "language": None}
    if languages_unscorable:
        scores["language_skipped"] = languages_unscorable
    else:
        scores["language"] = _rounded(language_confusion(utterances, segments))

    if not any(segment.speaker for segment in segments):
        reason = speakers_unscorable or undiarized
    else:
        reason = speakers_unscorable or diarization_scorable(utterances)
        if not reason:
            scores["diarization"] = _rounded(
                diarization_error(utterances, segments, duration)
            )
    if reason:
        scores["diarization_skipped"] = reason
    return scores


def _rounded(data: Any) -> Any:
    """Round every float in nested data to four places, for a readable report."""
    if isinstance(data, float):
        return round(data, 4)
    if isinstance(data, dict):
        return {key: _rounded(value) for key, value in data.items()}
    return data


def compare(baseline: Report, candidate: Report) -> dict[str, Any]:
    """Compare two reports, recording by recording.

    A change is judged per recording, not on the mean. The chunked prototype
    improved one recording by 23% and made another 36% worse; a mean would have
    hidden both.

    Args:
        baseline: The report to beat.
        candidate: The report being judged.

    Returns:
        JSON-serializable comparison data.
    """
    by_name = {item.name: item for item in candidate.scored}
    per_recording: dict[str, Any] = {}
    improved = 0

    for item in baseline.scored:
        other = by_name.get(item.name)
        if other is None or item.score is None or other.score is None:
            continue
        comparison = Comparison(baseline=item.score, candidate=other.score)
        per_recording[item.name] = comparison.to_dict()
        improved += int(comparison.improved)

    talkbank_deltas = {}
    for name, section, key in (
        ("mean_speaker_confusion_delta", "diarization", "confusion"),
        ("mean_language_accuracy_delta", "language", "accuracy"),
    ):
        before = baseline.mean_annotation(section, key)
        after = candidate.mean_annotation(section, key)
        if before is not None and after is not None:
            talkbank_deltas[name] = round(after - before, 4)

    return {
        **talkbank_deltas,
        "baseline_label": baseline.label,
        "candidate_label": candidate.label,
        "recordings_compared": len(per_recording),
        "recordings_improved": improved,
        "improved_everywhere": bool(per_recording) and improved == len(per_recording),
        "mean_wer_delta": round(candidate.mean("wer") - baseline.mean("wer"), 4),
        "mean_retained_delta": round(
            candidate.mean("retained") - baseline.mean("retained"), 4
        ),
        "mean_coverage_delta": round(
            candidate.mean("coverage") - baseline.mean("coverage"), 4
        ),
        "mean_language_mix_error_delta": round(
            candidate.mean("language_mix_error") - baseline.mean("language_mix_error"),
            4,
        ),
        "per_recording": per_recording,
    }


def _recordable(config: dict[str, Any]) -> dict[str, Any]:
    """Return a config safe to serialize into a report."""
    return {
        key: value
        if value is None or isinstance(value, bool | int | float | str)
        else str(value)
        for key, value in config.items()
    }
