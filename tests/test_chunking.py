"""Tests for VAD chunking.

The grouping is tested directly with synthetic regions, because the bug it fixes
was in the grouping rather than in the VAD: a region longer than the target used
to be emitted whole, so a 30-minute recording produced 11 chunks.
"""

import pytest

pytest.importorskip(
    "faster_whisper", reason="the faster-whisper extra is not installed"
)

from audio_transcriber.backends.chunking import (  # noqa: E402
    SAMPLE_RATE,
    Chunk,
    group_regions,
    speech_coverage,
    vad_options,
)

TARGET = 25 * SAMPLE_RATE
MIN = 1 * SAMPLE_RATE


def seconds(*pairs):
    """Build sample-pair regions from second-pairs, for readability."""
    return [(int(a * SAMPLE_RATE), int(b * SAMPLE_RATE)) for a, b in pairs]


def test_vad_options_bound_region_length():
    """Silero defaults to inf, which is what produced minutes-long regions."""
    assert vad_options(25.0).max_speech_duration_s == 25.0


def test_short_regions_are_merged_up_to_the_target():
    chunks = group_regions(seconds((0, 5), (6, 11), (12, 17)), TARGET, MIN)

    assert len(chunks) == 1
    assert chunks[0].duration <= 25.0


def test_a_region_longer_than_the_target_is_split():
    """The actual bug: a 164-second region used to be emitted as one chunk."""
    chunks = group_regions(seconds((0, 164)), TARGET, MIN)

    assert len(chunks) > 1
    assert all(chunk.duration <= 25.0 for chunk in chunks)


def test_no_chunk_ever_exceeds_the_target():
    chunks = group_regions(seconds((0, 300), (305, 400), (402, 403)), TARGET, MIN)

    assert chunks
    assert max(chunk.duration for chunk in chunks) <= 25.0


def test_a_thirty_minute_recording_produces_many_chunks():
    """Regression: 30 minutes of speech used to yield 11 chunks."""
    chunks = group_regions(seconds((0, 1800)), TARGET, MIN)

    assert len(chunks) >= 70


def test_chunks_stay_in_order():
    chunks = group_regions(seconds((0, 30), (40, 80), (90, 100)), TARGET, MIN)

    assert chunks == sorted(chunks, key=lambda c: c.start)


def test_chunks_do_not_overlap():
    chunks = group_regions(seconds((0, 100), (110, 200)), TARGET, MIN)

    for earlier, later in zip(chunks, chunks[1:], strict=False):
        assert earlier.end <= later.start


def test_silence_between_regions_is_not_covered():
    """Chunks describe speech; the gaps are not transcribed."""
    chunks = group_regions(seconds((0, 10), (100, 110)), TARGET, MIN)

    assert sum(c.end - c.start for c in chunks) == 20 * SAMPLE_RATE


def test_a_tiny_trailing_region_is_merged_into_its_neighbour():
    """Half a second is not enough audio to identify a language from."""
    chunks = group_regions(seconds((0, 10), (10.1, 10.4)), TARGET, MIN)

    assert len(chunks) == 1


def test_a_tiny_region_is_kept_when_merging_would_exceed_the_target():
    chunks = group_regions(seconds((0, 25), (25.1, 25.4)), TARGET, MIN)

    assert len(chunks) == 2
    assert all(chunk.duration <= 25.0 for chunk in chunks)


def test_a_lone_short_region_is_kept():
    """Dropping it would lose the only speech there is."""
    chunks = group_regions(seconds((0, 0.4)), TARGET, MIN)

    assert len(chunks) == 1


def test_no_regions_yields_no_chunks():
    assert group_regions([], TARGET, MIN) == []


def test_target_must_be_positive():
    with pytest.raises(ValueError, match="must be positive"):
        group_regions(seconds((0, 10)), 0, MIN)


def test_chunk_reports_its_times():
    chunk = Chunk(start=SAMPLE_RATE, end=3 * SAMPLE_RATE)

    assert chunk.start_time == 1.0
    assert chunk.end_time == 3.0
    assert chunk.duration == 2.0


def test_coverage_is_the_fraction_of_audio_in_chunks():
    chunks = group_regions(seconds((0, 50)), TARGET, MIN)

    assert speech_coverage(chunks, 100 * SAMPLE_RATE) == pytest.approx(0.5)


def test_coverage_of_empty_audio_is_zero_not_an_error():
    assert speech_coverage([], 0) == 0.0


def test_coverage_of_no_chunks_is_zero():
    assert speech_coverage([], 100 * SAMPLE_RATE) == 0.0


@pytest.mark.parametrize("target_seconds", [5.0, 10.0, 25.0, 28.0])
def test_the_target_is_respected_at_any_setting(target_seconds):
    target = int(target_seconds * SAMPLE_RATE)
    chunks = group_regions(seconds((0, 600)), target, MIN)

    assert max(chunk.duration for chunk in chunks) <= target_seconds
