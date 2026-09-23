"""Speaker diarization, via pyannote.

Lives under `backends/` because it is engine-specific: nothing in the pipeline
imports it, and a backend that does not diarize never loads it.

pyannote's models are MIT-licensed but gated on the Hugging Face Hub. Using them
requires accepting their conditions and supplying a token, so the requirement
check here reports exactly which pages to visit rather than surfacing a bare 401.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from audio_transcriber.errors import BackendRequirementError

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


def find_token(explicit: str | None = None) -> str | None:
    """Return the Hugging Face token, from config or the environment."""
    if explicit:
        return explicit
    for name in _TOKEN_VARIABLES:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _gate_instructions(model: str = DEFAULT_MODEL) -> str:
    """Return the steps that make `model` fetchable."""
    pages = "\n  ".join(f"https://huggingface.co/{name}" for name in gates_for(model))
    plural = "each model" if len(gates_for(model)) > 1 else "the model"
    return (
        f"Accept the conditions for {plural}, then set HF_TOKEN:\n  "
        f"{pages}\n"
        "Create a token at https://huggingface.co/settings/tokens and export it "
        "as HF_TOKEN."
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
            _gate_instructions(model),
        ) from error


def load_pipeline(model: str, token: str | None):
    """Load a pyannote pipeline.

    Args:
        model: Pipeline identifier on the Hub.
        token: Hugging Face token.

    Returns:
        The loaded pipeline.

    Raises:
        BackendRequirementError: If the Hub returns nothing, which is what an
            unaccepted gate looks like from here.
    """
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(model, use_auth_token=token)
    if pipeline is None:
        # from_pretrained returns None rather than raising when access is denied.
        raise BackendRequirementError(
            "diarization",
            f"the Hub returned no pipeline for {model!r}, which means the gate "
            "has not been accepted for this token.",
            _gate_instructions(model),
        )
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
    annotation = pipeline(str(audio_path), **constraints)
    turns = [
        SpeakerTurn(
            start=float(segment.start), end=float(segment.end), speaker=str(label)
        )
        for segment, _, label in annotation.itertracks(yield_label=True)
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
