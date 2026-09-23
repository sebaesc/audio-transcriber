"""Splitting audio into chunks for per-chunk language detection.

Lives under `backends/` because it depends on faster-whisper's bundled Silero
VAD. Nothing in the pipeline imports it.

Chunks exist so that language can be detected more than once per recording. The
target length is a compromise: long enough for detection to be confident, short
enough that one decision does not cover minutes of bilingual conversation.
"""

from __future__ import annotations

from dataclasses import dataclass

from faster_whisper.vad import VadOptions, get_speech_timestamps

SAMPLE_RATE = 16_000

#: Default chunk length. Chosen from the eval, not from Whisper's 30-second
#: window: against 25 s, 15 s improved WER on 4 of the 5 evaluation recordings,
#: three beyond the seed noise floor, at the same speed. Shorter chunks mean more
#: decode boundaries, but consecutive chunks that agree on a language are
#: decoded as one span, so finer detection costs little. One recording,
#: `maria20`, was worse (+0.048 WER).
DEFAULT_TARGET_SECONDS = 15.0

#: Chunks shorter than this are merged into their neighbour rather than being
#: detected on their own: a fragment of a second is not enough audio to identify
#: a language from.
MIN_CHUNK_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class Chunk:
    """A span of speech, in samples.

    Attributes:
        start: First sample, inclusive.
        end: Last sample, exclusive.
    """

    start: int
    end: int

    @property
    def duration(self) -> float:
        """Length of the chunk in seconds."""
        return (self.end - self.start) / SAMPLE_RATE

    @property
    def start_time(self) -> float:
        """Chunk start, in seconds from the beginning of the audio."""
        return self.start / SAMPLE_RATE

    @property
    def end_time(self) -> float:
        """Chunk end, in seconds from the beginning of the audio."""
        return self.end / SAMPLE_RATE


def vad_options(target_seconds: float = DEFAULT_TARGET_SECONDS) -> VadOptions:
    """Return VAD options that actually bound region length.

    Silero defaults to `max_speech_duration_s=inf`, so in conversation with few
    long silences it returns speech regions minutes long. Setting the bound lets
    the VAD split them at the best silence it can find, which is a far better
    boundary than an arbitrary cut.
    """
    return VadOptions(max_speech_duration_s=target_seconds)


def group_regions(
    regions: list[tuple[int, int]],
    target_samples: int,
    min_samples: int,
) -> list[Chunk]:
    """Group speech regions into chunks no longer than the target.

    Separated from the VAD call so the grouping can be tested directly. Two
    things matter here, and the prototype got both wrong:

    - A region longer than the target is **split**, not emitted whole. Avoiding
      further merging is not enough.
    - A chunk shorter than the minimum is merged into its predecessor rather
      than detected on its own.

    Args:
        regions: Speech regions as (start, end) sample pairs, in order.
        target_samples: Maximum chunk length in samples.
        min_samples: Below this, a chunk is merged into its predecessor.

    Returns:
        Chunks covering the given regions, none longer than the target.
    """
    if target_samples <= 0:
        raise ValueError("target_samples must be positive")

    split: list[tuple[int, int]] = []
    for start, end in regions:
        while end - start > target_samples:
            split.append((start, start + target_samples))
            start += target_samples
        if end > start:
            split.append((start, end))

    chunks: list[Chunk] = []
    current_start: int | None = None
    current_end = 0
    for start, end in split:
        if current_start is None:
            current_start, current_end = start, end
            continue
        if end - current_start <= target_samples:
            current_end = end
        else:
            chunks.append(Chunk(current_start, current_end))
            current_start, current_end = start, end
    if current_start is not None:
        chunks.append(Chunk(current_start, current_end))

    return _merge_short(chunks, min_samples, target_samples)


def _merge_short(
    chunks: list[Chunk], min_samples: int, target_samples: int
) -> list[Chunk]:
    """Fold chunks below the minimum into their predecessor where it fits."""
    merged: list[Chunk] = []
    for chunk in chunks:
        if (
            merged
            and chunk.end - chunk.start < min_samples
            and chunk.end - merged[-1].start <= target_samples
        ):
            previous = merged.pop()
            merged.append(Chunk(previous.start, chunk.end))
        else:
            merged.append(chunk)
    return merged


def speech_chunks(
    audio,
    target_seconds: float = DEFAULT_TARGET_SECONDS,
    min_seconds: float = MIN_CHUNK_SECONDS,
) -> list[Chunk]:
    """Split audio into speech chunks of at most `target_seconds`.

    Args:
        audio: Mono 16 kHz float32 samples.
        target_seconds: Maximum chunk length.
        min_seconds: Below this, a chunk is merged into its predecessor.

    Returns:
        Chunks in ascending time order, covering the detected speech.
    """
    regions = [
        (region["start"], region["end"])
        for region in get_speech_timestamps(audio, vad_options(target_seconds))
    ]
    return group_regions(
        regions,
        target_samples=int(target_seconds * SAMPLE_RATE),
        min_samples=int(min_seconds * SAMPLE_RATE),
    )


def speech_coverage(chunks: list[Chunk], total_samples: int) -> float:
    """Return the fraction of the audio the chunks cover.

    The guard against buying a WER improvement by discarding audio. A run whose
    chunks cover materially less than the baseline transcribed is not doing
    better, it is doing less.
    """
    if total_samples <= 0:
        return 0.0
    covered = sum(chunk.end - chunk.start for chunk in chunks)
    return covered / total_samples
