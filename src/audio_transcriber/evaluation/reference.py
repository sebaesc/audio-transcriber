"""A user's own ground truth, for scoring the team's recordings.

The corpus measures a backend on someone else's audio. What decides which
backend and settings to use is how they do on ours, so the harness also scores
against a ground truth the user writes in the pipeline's own transcript format:
a bare JSON array of `start_time`, `end_time`, `speaker`, `spoken_language`,
`text` segments.

Each segment is validated as a `Segment`, but the file is not read as a
`Transcript`. A transcript's segments never overlap; a human ground truth may
have two speakers at once, and diarization error handles that. So overlap is
allowed and order is not required.

Times may be seconds, as the pipeline writes them, or clock times — `MM:SS` or
`HH:MM:SS`, optionally with fractional seconds — as a person writes them while
listening. Clock times are converted to seconds when read.

A recording is often a clip cut from a longer video, and its ground truth is
written against the video it was cut from, with that video's times. Such a file
is an object instead of a bare array, `{"clip_start": ..., "segments": [...]}`,
and every time in it is moved back by `clip_start` when read. Speech that then
falls outside the clip is cut, and if much of it does, the file is refused:
scored against the wrong stretch of time, speaker and language scores are
meaningless yet can look perfect (see `GroundTruth.fit`).

The recordings and ground-truth files are the team's data and stay outside the
repo. Reports record each ground truth's checksum, so a score can be traced back
to the file it was measured against.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

from audio_transcriber.discovery import is_audio
from audio_transcriber.errors import ConfigError
from audio_transcriber.evaluation.annotation import Utterance
from audio_transcriber.evaluation.text import normalize
from audio_transcriber.manifest import checksum_file
from audio_transcriber.schema import SchemaError, Segment

REFERENCE_SUFFIX = ".json"

#: `MM:SS` or `HH:MM:SS`, seconds optionally fractional. Minutes are unbounded
#: without hours, so an hour-long recording can be written `75:10`.
_CLOCK = re.compile(r"^(?:(\d+):([0-5]\d):|(\d+):)([0-5]\d(?:\.\d+)?)$")

_TIME_KEYS = ("start_time", "end_time")

#: The keys of a ground truth written as an object, for a clip.
_CLIP_KEYS = frozenset({"clip_start", "segments"})

#: The share of a ground truth's speech that may fall outside the recording
#: before the file is refused. A clip cut a second off from its ground truth
#: loses a segment or two at either end; a missing or wrong `clip_start` puts
#: nearly all of it outside.
MAX_OUTSIDE_SHARE = 0.2

#: Files a pipeline run writes beside its transcripts that are not transcripts.
_RUN_FILES = frozenset({"manifest.json", "errors.json"})


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """A validated ground-truth file.

    Attributes:
        path: Where it was read from.
        checksum: Checksum of the file's bytes, `sha256:<hex>`.
        segments: Its segments, ordered by start then end time, in seconds
            from the start of the recording: moved back by `clip_start`, and
            cut to the recording.
        clip_start: Where the recording starts in the video the ground truth
            was written against, in seconds. 0 for a bare array.
        speech_s: Seconds of speech the file holds, before any was cut.
            Overlapping segments count twice.
        outside_s: Seconds of that speech cut for falling outside the
            recording.
    """

    path: Path
    checksum: str
    segments: tuple[Segment, ...]
    clip_start: float = 0.0
    speech_s: float = 0.0
    outside_s: float = 0.0

    @property
    def traced(self) -> dict[str, str | float]:
        """What a report records of this file, to trace a score back to it.

        `clip_start` and `outside_s` only when set, so a bare array is traced
        as it always was.
        """
        data: dict[str, str | float] = {
            "file": self.path.name,
            "checksum": self.checksum,
        }
        if self.clip_start:
            data["clip_start"] = round(self.clip_start, 3)
        if self.outside_s:
            data["outside_s"] = round(self.outside_s, 1)
        return data

    def fit(
        self, duration: float, *, trim: bool = True, bound: str = "the audio"
    ) -> GroundTruth:
        """Check the ground truth lies within a recording, and cut it to it.

        Cuts speech after `duration`: it is not in the recording, so neither
        its words nor its time are scored. Speech before 0 was already cut when
        the file was read.

        Args:
            duration: Length of the recording, in seconds.
            trim: Cut the speech after `duration`. False when `duration` only
                bounds where the recording ends, so it is checked but not cut.
            bound: What `duration` is the end of, for the error message.

        Returns:
            The ground truth with the speech outside the recording cut.

        Raises:
            ConfigError: If more than `MAX_OUTSIDE_SHARE` of the speech falls
                outside the recording: the ground truth is for other times, so
                `clip_start` is most likely missing or wrong.
        """
        kept = []
        outside = self.outside_s
        for segment in self.segments:
            if segment.start_time >= duration:
                outside += segment.duration
                continue
            if segment.end_time > duration:
                outside += segment.end_time - duration
                segment = replace(segment, end_time=duration)
            kept.append(segment)

        share = outside / self.speech_s if self.speech_s else 0.0
        if share > MAX_OUTSIDE_SHARE or not kept:
            first = min(s.start_time for s in self.segments) + self.clip_start
            last = self.end_time + self.clip_start
            where = (
                f"after moving it back by clip_start ({self.clip_start:g} s)"
                if self.clip_start
                else "with no clip_start"
            )
            raise ConfigError(
                f"ground truth {self.path}: {share:.0%} of its speech falls outside "
                f"{bound} (0–{duration:.0f} s) {where}. It is written for "
                f"{first:.0f}–{last:.0f} s: is clip_start missing or wrong? "
                "See docs/evaluation.md."
            )
        if not trim:
            return self
        return replace(self, segments=tuple(kept), outside_s=outside)

    @property
    def utterances(self) -> list[Utterance]:
        """The segments as timed reference utterances, for `annotation`.

        A segment with no speaker gets an empty one; diarization is not scored
        unless every segment has a speaker (see `speakers_unscorable`).
        """
        return [
            Utterance(
                speaker=segment.speaker or "",
                start=segment.start_time,
                end=segment.end_time,
                language=segment.spoken_language,
            )
            for segment in self.segments
        ]

    @property
    def text(self) -> str:
        """The reference words, in time order, normalized for scoring.

        `normalize` rather than `clean_reference`: the latter strips CHAT markup,
        which a user's JSON does not contain, and could eat text that is real.
        """
        return normalize(" ".join(segment.text for segment in self.segments))

    @property
    def end_time(self) -> float:
        """When the last reference speech ends."""
        return max(segment.end_time for segment in self.segments)

    def speakers_unscorable(self) -> str | None:
        """Return why speakers cannot be scored against this file, or None.

        A segment with no speaker is speech that belongs to nobody. Scored, any
        hypothesis speaker there would be a false alarm, so it is all or none.
        """
        unlabelled = sum(1 for segment in self.segments if not segment.speaker)
        if unlabelled == len(self.segments):
            return "the ground truth has no speaker labels"
        if unlabelled:
            return (
                f"{unlabelled} of {len(self.segments)} segments in the ground "
                "truth have no speaker label"
            )
        return None

    def languages_unscorable(self) -> str | None:
        """Return why languages cannot be scored against this file, or None.

        Unlike speakers, a partly labelled file can be scored: unlabelled
        segments are left out, as TalkBank utterances of unknown language are.
        """
        if not any(segment.spoken_language for segment in self.segments):
            return "the ground truth has no spoken_language labels"
        return None


def load_reference(path: Path) -> GroundTruth:
    """Read and validate one ground-truth file.

    Args:
        path: A JSON array of five-key transcript segments, or, for a clip, an
            object holding them as `segments` beside the clip's `clip_start`.

    Returns:
        The ground truth, its segments sorted by time and moved back by
        `clip_start`. Speech that then starts before 0 is cut; speech after
        the end of the recording is cut by `GroundTruth.fit`.

    Raises:
        ConfigError: If the file cannot be read or is not a valid ground truth.
            The message names the file and, for a bad segment, its index.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(f"cannot read ground truth {path}: {error}") from error
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ConfigError(
            f"ground truth {path} is not valid JSON: {error}. It should be an "
            "array of transcript segments; see docs/evaluation.md."
        ) from error

    data, clip_start = _unwrap_clip(path, data)
    if not isinstance(data, list):
        raise ConfigError(
            f"ground truth {path} must be a JSON array of segments, got a "
            f"{type(data).__name__}. See docs/evaluation.md for the format."
        )
    if not data:
        raise ConfigError(f"ground truth {path} is empty: it has no segments.")

    segments = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ConfigError(
                f"ground truth {path}, segment {index}: expected an object with "
                f"start_time, end_time and text, got {type(item).__name__}."
            )
        try:
            item = {**item, **_times_in_seconds(item)}
            # Validated in the file's own times, so an error names times the
            # user can find in it.
            segments.append(Segment.from_dict(item))
        except (SchemaError, TypeError, ValueError) as error:
            raise ConfigError(
                f"ground truth {path}, segment {index}: {error}"
            ) from error

    speech = sum(segment.duration for segment in segments)
    kept, outside = _shift(segments, clip_start)
    if not kept:
        raise ConfigError(
            f"ground truth {path}: all of its speech ends before clip_start "
            f"({clip_start:g} s), so none of it is in the recording."
        )

    return GroundTruth(
        path=path,
        checksum=checksum_file(path),
        segments=tuple(sorted(kept, key=lambda s: (s.start_time, s.end_time))),
        clip_start=clip_start,
        speech_s=speech,
        outside_s=outside,
    )


def _unwrap_clip(path: Path, data):
    """Return a ground truth's segments and its `clip_start`.

    A bare array is its own segments, starting at 0. An object must hold
    `segments` and may hold `clip_start`; nothing else, so a misspelt
    `clip_start` is an error rather than silently 0.

    Raises:
        ConfigError: If an object has other keys, no `segments`, or a
            `clip_start` that is not a time at or after 0.
    """
    if not isinstance(data, dict):
        return data, 0.0
    if "segments" not in data:
        keys = ", ".join(sorted(data)) or "none"
        raise ConfigError(
            f"ground truth {path} must be a JSON array of segments, or for a clip "
            "an object holding them as segments beside clip_start; got an "
            f"object with keys: {keys}. See docs/evaluation.md for the format."
        )
    unknown = sorted(set(data) - _CLIP_KEYS)
    if unknown:
        raise ConfigError(
            f"ground truth {path} has unknown keys: {', '.join(unknown)}. An "
            "object holds only segments and clip_start; see docs/evaluation.md."
        )
    clip_start = data.get("clip_start", 0.0)
    try:
        clip_start = _seconds("clip_start", clip_start)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"ground truth {path}: {error}") from error
    if clip_start < 0:
        raise ConfigError(
            f"ground truth {path}: clip_start must be >= 0, got {clip_start:g}"
        )
    return data["segments"], clip_start


def _shift(segments: list[Segment], clip_start: float) -> tuple[list[Segment], float]:
    """Move segments back by `clip_start`, cutting speech before 0.

    Returns:
        The segments that still hold speech, and the seconds of speech cut.
    """
    if not clip_start:
        return segments, 0.0
    kept, outside = [], 0.0
    for segment in segments:
        # Rounded so 104.2 - 100 reads 4.2, not 4.200000000000003.
        start = round(segment.start_time - clip_start, 6)
        end = round(segment.end_time - clip_start, 6)
        if end <= 0:
            outside += segment.duration
            continue
        if start < 0:
            outside += -start
            start = 0.0
        kept.append(replace(segment, start_time=start, end_time=end))
    return kept, outside


def _times_in_seconds(item: dict) -> dict:
    """Return a segment's times as seconds, converting any clock times.

    Raises:
        ValueError: If a time is a string that is neither a number nor a clock
            time.
    """
    return {
        key: _seconds(key, item[key])
        for key in _TIME_KEYS
        if isinstance(item.get(key), str)
    }


def _seconds(key: str, value) -> float:
    """Return a time as seconds: a number, or a string of seconds or a clock time.

    Raises:
        ValueError: If a string is neither a number nor a clock time.
        TypeError: If the value is neither a number nor a string.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(
            f"{key} must be seconds or a clock time, got {type(value).__name__}"
        )
    if not isinstance(value, str):
        return float(value)
    clock = _CLOCK.match(value.strip())
    if clock:
        hours, minutes_after_hours, minutes, seconds = clock.groups()
        return (
            int(hours or 0) * 3600
            + int(minutes_after_hours or minutes) * 60
            + float(seconds)
        )
    try:
        return float(value)
    except ValueError:
        raise ValueError(
            f"{key} {value!r} is neither seconds nor a clock time (MM:SS or HH:MM:SS)"
        ) from None


@dataclass(frozen=True, slots=True)
class Pair:
    """A recording and the ground truth it is scored against.

    Attributes:
        name: The recording's name, as it appears in reports.
        media: The audio or video file, or the transcript already made from it.
        reference: Its ground-truth file.
    """

    name: str
    media: Path
    reference: Path


def single_pair(media: Path, reference: Path) -> list[Pair]:
    """Pair one file with one ground truth, named after the file.

    Raises:
        ConfigError: If either file does not exist.
    """
    for path in (media, reference):
        if not path.is_file():
            raise ConfigError(f"file not found: {path}")
    return [Pair(name=media.stem, media=media, reference=reference)]


def discover_pairs(
    directory: Path,
    only: list[str] | None = None,
    reference_directory: Path | None = None,
    is_media=is_audio,
) -> list[Pair]:
    """Pair each file in a folder with the ground truth of the same name.

    `standup.mp4` pairs with `standup.json`. Other files are ignored. Only the
    top level is searched.

    Args:
        directory: Holds the media, and the ground truth unless
            `reference_directory` is given.
        only: Names to limit the pairs to. Applied before pairing, so an
            unrelated unpaired file does not block a run.
        reference_directory: Where the ground-truth files are, when they are
            not beside the media.
        is_media: Which files in `directory` are media. Audio and video by
            default.

    Returns:
        The pairs, ordered by name.

    Raises:
        ConfigError: If a folder is missing or holds nothing to score, a name in
            `only` matches nothing, or a file has no partner or more than one.
    """
    reference_directory = reference_directory or directory
    for folder in {directory, reference_directory}:
        if not folder.is_dir():
            raise ConfigError(f"not a directory: {folder}")

    media = _by_name(path for path in directory.iterdir() if is_media(path))
    references = _by_name(
        path
        for path in reference_directory.iterdir()
        if path.suffix.lower() == REFERENCE_SUFFIX
    )

    if only:
        unknown = sorted(set(only) - set(media) - set(references))
        if unknown:
            raise ConfigError(
                f"--only names nothing in the folder: {', '.join(unknown)}"
            )
        media = {name: paths for name, paths in media.items() if name in only}
        references = {name: paths for name, paths in references.items() if name in only}

    if not media and not references:
        raise ConfigError(f"no recordings with ground truth found in {directory}")

    pairs = []
    for name in sorted(set(media) | set(references)):
        found_media = media.get(name, [])
        found_reference = references.get(name, [])
        if len(found_media) > 1:
            raise ConfigError(
                f"more than one file is named {name!r}: "
                f"{', '.join(path.name for path in found_media)}. Rename all but one."
            )
        if not found_reference:
            raise ConfigError(
                f"{found_media[0]} has no ground truth: expected "
                f"{reference_directory / (name + REFERENCE_SUFFIX)}"
            )
        if not found_media:
            raise ConfigError(
                f"ground truth {found_reference[0]} has nothing to score: no file "
                f"named {name!r} in {directory}"
            )
        pairs.append(
            Pair(name=name, media=found_media[0], reference=found_reference[0])
        )
    return pairs


def transcript_pairs(
    directory: Path, reference_directory: Path, only: list[str] | None = None
) -> list[Pair]:
    """Pair each transcript in a pipeline run's folder with its ground truth.

    The pipeline names a transcript after its recording, so `standup.json` in
    the run folder pairs with `standup.json` in the ground-truth folder. The
    run's `manifest.json` and `errors.json` are not transcripts. Anything else
    in the ground-truth folder, such as the recordings, is ignored.

    Raises:
        ConfigError: As `discover_pairs`.
    """
    return discover_pairs(
        directory,
        only=only,
        reference_directory=reference_directory,
        is_media=_is_transcript,
    )


def _is_transcript(path: Path) -> bool:
    return path.suffix.lower() == REFERENCE_SUFFIX and path.name not in _RUN_FILES


def _by_name(paths) -> dict[str, list[Path]]:
    """Group files by stem."""
    grouped: dict[str, list[Path]] = {}
    for path in sorted(paths):
        if path.is_file():
            grouped.setdefault(path.stem, []).append(path)
    return grouped
