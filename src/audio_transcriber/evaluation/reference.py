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

The recordings and ground-truth files are the team's data and stay outside the
repo. Reports record each ground truth's checksum, so a score can be traced back
to the file it was measured against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from audio_transcriber.discovery import is_audio
from audio_transcriber.errors import ConfigError
from audio_transcriber.evaluation.annotation import Utterance
from audio_transcriber.evaluation.text import normalize
from audio_transcriber.manifest import checksum_file
from audio_transcriber.schema import SchemaError, Segment

REFERENCE_SUFFIX = ".json"

#: Files a pipeline run writes beside its transcripts that are not transcripts.
_RUN_FILES = frozenset({"manifest.json", "errors.json"})


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """A validated ground-truth file.

    Attributes:
        path: Where it was read from.
        checksum: Checksum of the file's bytes, `sha256:<hex>`.
        segments: Its segments, ordered by start then end time.
    """

    path: Path
    checksum: str
    segments: tuple[Segment, ...]

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
        path: A JSON array of five-key transcript segments.

    Returns:
        The ground truth, its segments sorted by time.

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
            segments.append(Segment.from_dict(item))
        except (SchemaError, TypeError, ValueError) as error:
            raise ConfigError(
                f"ground truth {path}, segment {index}: {error}"
            ) from error

    return GroundTruth(
        path=path,
        checksum=checksum_file(path),
        segments=tuple(sorted(segments, key=lambda s: (s.start_time, s.end_time))),
    )


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
