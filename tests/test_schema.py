"""Tests for the shared transcript schema.

The schema is a project contract, so these cover serialization, round-tripping,
and the validation that keeps malformed transcripts out of the output directory.
"""

import json

import pytest

from audio_transcriber.schema import (
    SCHEMA_VERSION,
    SchemaError,
    Segment,
    Transcript,
)


def test_segment_round_trips_through_dict():
    segment = Segment(start=0.0, end=1.5, text="hello", confidence=0.9)

    assert Segment.from_dict(segment.to_dict()) == segment


def test_segment_duration_is_end_minus_start():
    assert Segment(start=2.0, end=5.5, text="x").duration == 3.5


def test_segment_speaker_defaults_to_none():
    """Speaker fields exist in the schema but stay unpopulated without diarization."""
    segment = Segment(start=0.0, end=1.0, text="hello")

    assert segment.speaker is None
    assert segment.to_dict()["speaker"] is None


def test_segment_speaker_survives_round_trip_when_populated():
    segment = Segment(start=0.0, end=1.0, text="hello", speaker="SPEAKER_01")

    assert Segment.from_dict(segment.to_dict()).speaker == "SPEAKER_01"


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"start": -1.0, "end": 1.0, "text": "x"}, "start must be >= 0"),
        ({"start": 5.0, "end": 2.0, "text": "x"}, "before its start"),
        ({"start": 0.0, "end": 1.0, "text": "x", "confidence": 1.5}, "0.0-1.0"),
    ],
)
def test_segment_rejects_invalid_values(kwargs, message):
    with pytest.raises(SchemaError, match=message):
        Segment(**kwargs)


def test_segment_from_dict_reports_missing_keys():
    with pytest.raises(SchemaError, match="missing required keys"):
        Segment.from_dict({"start": 0.0, "text": "no end"})


def test_transcript_round_trips_through_json():
    transcript = Transcript(
        source="audio/meeting.wav",
        segments=(
            Segment(start=0.0, end=1.0, text="hello"),
            Segment(start=1.0, end=2.0, text="world"),
        ),
        language="en",
        duration=2.0,
        backend="reference",
        model_version="tiny-1.0",
    )

    restored = Transcript.from_dict(json.loads(json.dumps(transcript.to_dict())))

    assert restored == transcript


def test_transcript_text_joins_segments():
    transcript = Transcript(
        source="a.wav",
        segments=(
            Segment(start=0.0, end=1.0, text="hello"),
            Segment(start=1.0, end=2.0, text="world"),
        ),
    )

    assert transcript.text == "hello world"


def test_empty_transcript_is_valid_and_has_empty_text():
    """A file with no detected speech is a successful run, not a failure."""
    transcript = Transcript(source="silence.wav")

    assert transcript.segments == ()
    assert transcript.text == ""


def test_transcript_records_schema_version_by_default():
    assert Transcript(source="a.wav").to_dict()["schema_version"] == SCHEMA_VERSION


def test_transcript_rejects_overlapping_segments():
    with pytest.raises(SchemaError, match="before the previous segment ended"):
        Transcript(
            source="a.wav",
            segments=(
                Segment(start=0.0, end=2.0, text="first"),
                Segment(start=1.0, end=3.0, text="overlaps"),
            ),
        )


def test_transcript_rejects_empty_source():
    with pytest.raises(SchemaError, match="source must not be empty"):
        Transcript(source="")


def test_transcript_from_dict_reports_missing_source():
    with pytest.raises(SchemaError, match="missing required key: source"):
        Transcript.from_dict({"segments": []})
