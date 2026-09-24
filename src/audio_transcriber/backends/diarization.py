"""Speaker diarization, via pyannote.

Lives under `backends/` because it is engine-specific: nothing in the pipeline
imports it, and a backend that does not diarize never loads it.

pyannote's models are MIT-licensed but gated on the Hugging Face Hub. Using them
requires accepting their conditions and supplying a token, so the requirement
check here reports exactly which pages to visit rather than surfacing a bare 401.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from pathlib import Path

from audio_transcriber.backends import credentials
from audio_transcriber.errors import BackendRequirementError
from audio_transcriber.schema import Segment

#: pyannote 4.x's recommended pipeline. It bundles its embedding and scoring
#: weights, so only its own gate has to be accepted.
DEFAULT_MODEL = "pyannote/speaker-diarization-community-1"

#: The gates a pipeline needs accepted. `speaker-diarization-3.1` references its
#: segmentation model externally, so accepting only the pipeline leaves the
#: second fetch failing with a bare 401 that explains nothing.
GATES: dict[str, tuple[str, ...]] = {
    "pyannote/speaker-diarization-community-1": (
        "pyannote/speaker-diarization-community-1",
    ),
    "pyannote/speaker-diarization-3.1": (
        "pyannote/speaker-diarization-3.1",
        "pyannote/segmentation-3.0",
    ),
}


def gates_for(model: str) -> tuple[str, ...]:
    """Return the Hub pages whose conditions must be accepted for `model`.

    An unknown model is assumed to gate only itself; that is the common case and
    a wrong guess costs the reader one extra click, not a failed run.
    """
    return GATES.get(model, (model,))


#: The longest run of words, flanked on both sides by one other speaker, that is
#: folded into that speaker rather than split out. Word timings drift by tens of
#: milliseconds, so a single word flipping at a turn boundary is more often
#: timing error than a one-word turn.
MAX_FOLDED_WORDS = 1

_TOKEN_VARIABLES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")


@dataclass(frozen=True, slots=True)
class SpeakerTurn:
    """One speaker speaking, over a span of the recording.

    Attributes:
        start: Start of the turn, in seconds.
        end: End of the turn, in seconds.
        speaker: Label, stable within one recording but not across recordings.
    """

    start: float
    end: float
    speaker: str

    def overlap(self, start: float, end: float) -> float:
        """Return how many seconds this turn shares with the given span."""
        return max(0.0, min(self.end, end) - max(self.start, start))


@dataclass(frozen=True, slots=True)
class Word:
    """One transcribed word and when it was spoken.

    Internal to the backends that assign speakers per word: the transcript
    schema carries segments, not words.

    Attributes:
        start: Start of the word, in seconds.
        end: End of the word, in seconds.
        text: The word as the engine spelled it, including any leading space
            and attached punctuation, so words join back into segment text.
    """

    start: float
    end: float
    text: str


def find_token(explicit: str | None = None) -> str | None:
    """Return the Hugging Face token.

    Looked up in order: the backend config, then the environment, then a `.env`
    beside the project, as every backend credential is (see `credentials`).
    """
    return credentials.find_credential(explicit, _TOKEN_VARIABLES)


def _gate_instructions(model: str = DEFAULT_MODEL) -> str:
    """Return the steps that make `model` fetchable."""
    pages = "\n  ".join(f"https://huggingface.co/{name}" for name in gates_for(model))
    plural = "each model" if len(gates_for(model)) > 1 else "the model"
    return (
        f"Accept the conditions for {plural}, then set HF_TOKEN:\n  "
        f"{pages}\n"
        "Create a token at https://huggingface.co/settings/tokens, then export "
        "it as HF_TOKEN or put it in a .env file beside the project."
    )


def check_requirements(model: str, token: str | None, backend: str) -> None:
    """Verify diarization can run before a run starts.

    Args:
        model: The pyannote pipeline to load.
        token: Hugging Face token, or None to read it from the environment.
        backend: Name of the calling backend, for the error message.

    Raises:
        BackendRequirementError: If the extra is not installed, no token is
            available, or the gated model cannot be fetched.
    """
    try:
        import pyannote.audio  # noqa: F401
    except ImportError as error:
        raise BackendRequirementError(
            backend,
            "diarization is enabled but pyannote.audio is not installed.",
            'Install it: pip install -e ".[diarization]"',
        ) from error

    resolved = find_token(token)
    if not resolved:
        raise BackendRequirementError(
            backend,
            "diarization needs a Hugging Face token; the models are gated.",
            _gate_instructions(model),
        )

    try:
        load_pipeline(model, resolved)
    except BackendRequirementError:
        raise
    except Exception as error:  # noqa: BLE001 - reported with its remedy
        raise BackendRequirementError(
            backend,
            f"could not load the diarization model {model!r}: {error}",
            "If the conditions have not been accepted for this token, "
            f"{_gate_instructions(model)} "
            "Otherwise the cause is above, not the gate.",
        ) from error


def load_pipeline(model: str, token: str | None, device: str | None = None):
    """Load a pyannote pipeline.

    pyannote loads onto the CPU and stays there unless moved. On CPU it runs
    under 2x realtime, slower than transcribing the same audio on a GPU; on CUDA
    it ran 23x realtime on a 21.8-minute recording.

    Args:
        model: Pipeline identifier on the Hub.
        token: Hugging Face token.
        device: `cpu` or `cuda` to move the pipeline to. None leaves it where
            it loaded.

    Returns:
        The loaded pipeline.

    Raises:
        BackendRequirementError: If the Hub returns nothing, which is what an
            unaccepted gate looks like from here.
    """
    from pyannote.audio import Pipeline

    # pyannote 4.x renamed this keyword from `use_auth_token` to `token`.
    pipeline = Pipeline.from_pretrained(model, token=token)
    if pipeline is None:
        # from_pretrained returns None rather than raising when access is denied.
        raise BackendRequirementError(
            "diarization",
            f"the Hub returned no pipeline for {model!r}, which means the gate "
            "has not been accepted for this token.",
            _gate_instructions(model),
        )
    if device is not None:
        import torch

        pipeline.to(torch.device(device))
    return pipeline


def version() -> str:
    """Return the installed pyannote version, for the run manifest."""
    try:
        from pyannote.audio import __version__

        return f"pyannote.audio-{__version__}"
    except ImportError:
        return ""


def diarize(
    pipeline,
    audio_path: Path,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[SpeakerTurn]:
    """Run diarization over one decoded file.

    Args:
        pipeline: A loaded pyannote pipeline.
        audio_path: Mono 16 kHz WAV, as the pipeline decodes it.
        num_speakers: Exact speaker count, when known.
        min_speakers: Lower bound on the speaker count.
        max_speakers: Upper bound on the speaker count.

    Returns:
        Speaker turns in ascending time order.
    """
    constraints = {
        key: value
        for key, value in (
            ("num_speakers", num_speakers),
            ("min_speakers", min_speakers),
            ("max_speakers", max_speakers),
        )
        if value is not None
    }
    # On audio with little or no speech, pyannote's clustering averages empty
    # embedding sets and numpy warns about it — "Mean of empty slice", "invalid
    # value encountered in divide". The result is simply no turns, which the
    # caller already handles; the warnings are noise, so they stop here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        output = pipeline(str(audio_path), **constraints)
    # pyannote 4.x returns a DiarizeOutput whose serialize() is the documented
    # shape; 3.x returned an Annotation iterated with itertracks(). Both are
    # supported so the version pin is not the only thing holding this together.
    if hasattr(output, "serialize"):
        turns = [
            SpeakerTurn(
                start=float(entry["start"]),
                end=float(entry["end"]),
                speaker=str(entry["speaker"]),
            )
            for entry in output.serialize().get("diarization", [])
        ]
    else:
        turns = [
            SpeakerTurn(
                start=float(segment.start), end=float(segment.end), speaker=str(label)
            )
            for segment, _, label in output.itertracks(yield_label=True)
        ]
    return sorted(turns, key=lambda turn: turn.start)


def speaker_for(turns: list[SpeakerTurn], start: float, end: float) -> str | None:
    """Return the speaker who holds most of a span.

    Args:
        turns: Speaker turns for the recording.
        start: Span start, in seconds.
        end: Span end, in seconds.

    Returns:
        The label with the greatest overlap, or None when nothing overlaps.
        A transcript segment can fall in a gap between turns, and inventing a
        speaker for it would be worse than admitting there is none.
    """
    best: str | None = None
    best_overlap = 0.0
    for turn in turns:
        if turn.start >= end:
            break
        shared = turn.overlap(start, end)
        if shared > best_overlap:
            best, best_overlap = turn.speaker, shared
    return best


def word_speakers(turns: list[SpeakerTurn], words: list[Word]) -> list[str | None]:
    """Return a speaker for each word.

    Each word takes the speaker holding most of it, as a segment does. A word
    that overlaps no turn — pyannote leaves short gaps, and Whisper's word times
    drift into them — takes the previous word's speaker, or the next word's at
    the start. Only when no word overlaps any turn is the result all None.

    Args:
        turns: Speaker turns for the recording, in time order.
        words: One segment's words, in order.

    Returns:
        One label per word, or None for every word if none overlaps a turn.
    """
    found = [_word_speaker(turns, word) for word in words]
    known = next((speaker for speaker in found if speaker is not None), None)
    filled: list[str | None] = []
    for speaker in found:
        known = speaker or known
        filled.append(known)
    return filled


def _word_speaker(turns: list[SpeakerTurn], word: Word) -> str | None:
    """Return the speaker for one word.

    Whisper collapses some words to an instant, which overlaps nothing, so
    those take the turn they fall inside.
    """
    if word.end > word.start:
        return speaker_for(turns, word.start, word.end)
    for turn in turns:
        if turn.start > word.start:
            break
        if word.start < turn.end:
            return turn.speaker
    return None


def smooth(
    speakers: list[str | None], max_folded: int = MAX_FOLDED_WORDS
) -> list[str | None]:
    """Fold short runs flanked on both sides by one other speaker into it.

    A run at either edge is kept, since nothing on its other side confirms it
    is noise: a short reply ending a segment is what per-word assignment is
    for. So is a run between two different speakers.

    Args:
        speakers: One label per word, in order.
        max_folded: The longest run that is folded.

    Returns:
        The labels, with short flanked runs relabelled.
    """
    runs = _runs(speakers)
    smoothed: list[str | None] = []
    for index, (speaker, length) in enumerate(runs):
        if 0 < index < len(runs) - 1 and length <= max_folded:
            before, after = runs[index - 1][0], runs[index + 1][0]
            if before == after:
                speaker = before
        smoothed.extend([speaker] * length)
    return smoothed


def _runs(speakers: list[str | None]) -> list[tuple[str | None, int]]:
    """Return consecutive equal labels as (label, count) runs."""
    runs: list[tuple[str | None, int]] = []
    for speaker in speakers:
        if runs and runs[-1][0] == speaker:
            runs[-1] = (speaker, runs[-1][1] + 1)
        else:
            runs.append((speaker, 1))
    return runs


def split_by_speaker(
    segment: Segment, words: list[Word], turns: list[SpeakerTurn]
) -> list[Segment]:
    """Split a segment wherever its speaker changes, word by word.

    The pieces are contiguous and cover exactly the segment's span: the first
    starts where the segment starts, the last ends where it ends, and each
    boundary falls where the next speaker's first word starts. Each piece
    keeps the segment's language. Its text is its words joined, which
    reproduces the segment's text because each word carries its own leading
    space and punctuation.

    A segment that stays with one speaker comes back whole, with its own text
    and times. One without word timings, or whose words overlap no turn, takes
    the speaker holding most of it, as in segment assignment.

    Args:
        segment: One transcript segment.
        words: Its words, clamped to its span.
        turns: Speaker turns for the recording, in time order.

    Returns:
        The segment's pieces, in order, each with a speaker.
    """
    # A blank word would become a piece with no text.
    words = [word for word in words if word.text.strip()]
    speakers = smooth(word_speakers(turns, words))
    if not speakers or speakers[0] is None:
        speaker = speaker_for(turns, segment.start_time, segment.end_time)
        return [replace(segment, speaker=speaker)]

    groups: list[tuple[str | None, list[Word]]] = []
    for word, speaker in zip(words, speakers, strict=True):
        if groups and groups[-1][0] == speaker:
            groups[-1][1].append(word)
        else:
            groups.append((speaker, [word]))
    if len(groups) == 1:
        return [replace(segment, speaker=groups[0][0])]

    pieces: list[Segment] = []
    start = segment.start_time
    for index, (speaker, group) in enumerate(groups):
        if index + 1 < len(groups):
            next_start = groups[index + 1][1][0].start
            end = min(max(next_start, start), segment.end_time)
        else:
            end = segment.end_time
        pieces.append(
            replace(
                segment,
                start_time=start,
                end_time=end,
                text="".join(word.text for word in group).strip(),
                speaker=speaker,
            )
        )
        start = end
    return pieces
