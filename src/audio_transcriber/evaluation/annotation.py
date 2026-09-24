"""Scoring a transcript against timed reference annotations.

Word error rate compares text. These compare *time*: whether each stretch of
reference speech was attributed to the right speaker, and labelled with the
right language. The reference is a list of timed utterances, as read from
TalkBank by `evaluation.talkbank`.

What is scored is the transcript's segments — the speaker and language labels
a user actually receives — not the diarizer's raw output. So the diarization
error has two kinds of component:

- **confusion**, speech attributed to the wrong person, is diarization quality;
- **missed** and **false alarm** mostly measure where Whisper's segments start
  and stop, and what it failed to transcribe at all, which `retained` already
  measures in words.

Read confusion for "are the speaker labels right".
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from audio_transcriber.evaluation.talkbank import Utterance
from audio_transcriber.schema import Segment

#: Seconds forgiven either side of each reference boundary. Human timestamps
#: on conversational speech are not exact, and 0.25 s is the usual allowance.
DEFAULT_COLLAR = 0.25

#: Reference languages that a single segment label can be right or wrong about.
_SCORED_LANGUAGES = ("en", "es")


def scorable_region(
    utterances: list[Utterance], duration: float
) -> list[tuple[float, float]]:
    """Return the stretches of the recording the reference fully describes.

    An untimed utterance was spoken somewhere between its timed neighbours, so
    that gap is left out: scoring it would count whatever was transcribed there
    as a false alarm, or its absence as nothing at all.

    Args:
        utterances: Reference utterances, in file order.
        duration: Length of the recording in seconds.

    Returns:
        Non-overlapping (start, end) spans, in order.
    """
    excluded: list[tuple[float, float]] = []
    previous_end = 0.0
    pending = False
    for utterance in utterances:
        if not utterance.timed:
            pending = True
            continue
        if pending:
            excluded.append((previous_end, utterance.start))
            pending = False
        previous_end = max(previous_end, utterance.end)
    if pending:
        excluded.append((previous_end, duration))

    region: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in excluded:
        if start > cursor:
            region.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        region.append((cursor, duration))
    return region


def diarization_error(
    utterances: list[Utterance],
    segments: list[Segment],
    duration: float,
    collar: float = DEFAULT_COLLAR,
) -> dict[str, Any]:
    """Score the transcript's speaker labels against the reference speakers.

    Our labels are arbitrary (`SPEAKER_00`), so they are mapped onto the
    reference speakers by the assignment that agrees most, as standard. A
    segment without a speaker contributes no hypothesis speech.

    Args:
        utterances: Reference utterances.
        segments: The transcript's segments.
        duration: Length of the recording in seconds.
        collar: Seconds forgiven either side of each reference boundary.

    Returns:
        `der` and its components `missed`, `false_alarm`, `confusion`, each a
        fraction of the reference speech scored, and `reference_seconds`.
    """
    from pyannote.core import Annotation, Timeline
    from pyannote.core import Segment as Span
    from pyannote.metrics.diarization import DiarizationErrorRate

    reference = Annotation()
    for index, utterance in enumerate(utterances):
        if utterance.timed and utterance.end > utterance.start:
            reference[Span(utterance.start, utterance.end), index] = utterance.speaker

    hypothesis = Annotation()
    for index, segment in enumerate(segments):
        if segment.speaker and segment.end_time > segment.start_time:
            hypothesis[Span(segment.start_time, segment.end_time), index] = (
                segment.speaker
            )

    uem = Timeline(
        [Span(start, end) for start, end in scorable_region(utterances, duration)]
    )
    # pyannote's collar is the total width removed around a boundary.
    metric = DiarizationErrorRate(collar=2 * collar, skip_overlap=False)
    detail = metric(reference, hypothesis, uem=uem, detailed=True)

    total = detail["total"]

    def share(key: str) -> float:
        return detail[key] / total if total else 0.0

    return {
        "der": detail["diarization error rate"] if total else 0.0,
        "missed": share("missed detection"),
        "false_alarm": share("false alarm"),
        "confusion": share("confusion"),
        "reference_seconds": total,
    }


def language_confusion(
    utterances: list[Utterance], segments: list[Segment]
) -> dict[str, Any]:
    """Score segment languages against reference utterance languages, by time.

    Every second where a segment overlaps a reference utterance is counted
    under (reference language, segment language). Mixed utterances have no
    single right label, so they are reported as a share and left out of
    accuracy; utterances with no known language, or no timestamp, are ignored.

    Args:
        utterances: Reference utterances.
        segments: The transcript's segments.

    Returns:
        `accuracy` over English and Spanish reference speech (None if there is
        none), `recall` per language, `mixed_share` of all overlapped speech,
        `scored_seconds`, and the full `confusion` in seconds.
    """
    reference = sorted(
        (u.start, u.end, u.language)
        for u in utterances
        if u.timed and u.language is not None and u.end > u.start
    )
    hypothesis = sorted(
        (s.start_time, s.end_time, s.spoken_language or "none") for s in segments
    )

    seconds: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    first = 0
    for start, end, language in reference:
        while first < len(hypothesis) and hypothesis[first][1] <= start:
            first += 1
        for h_start, h_end, h_language in hypothesis[first:]:
            if h_start >= end:
                break
            shared = min(end, h_end) - max(start, h_start)
            if shared > 0:
                seconds[language][h_language] += shared

    scored = {
        language: sum(seconds[language].values()) for language in _SCORED_LANGUAGES
    }
    scored_total = sum(scored.values())
    mixed = sum(seconds["mixed"].values())
    correct = sum(seconds[language][language] for language in _SCORED_LANGUAGES)

    return {
        "accuracy": correct / scored_total if scored_total else None,
        "recall": {
            language: seconds[language][language] / scored[language]
            if scored[language]
            else None
            for language in _SCORED_LANGUAGES
        },
        "mixed_share": mixed / (scored_total + mixed) if scored_total + mixed else 0.0,
        "scored_seconds": scored_total,
        "confusion": {
            reference: dict(hypotheses) for reference, hypotheses in seconds.items()
        },
    }
