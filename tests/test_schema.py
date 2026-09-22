"""Tests for the shared transcript schema.

The schema is a project contract, so these cover the exact file format, the
round trip, and the validation that keeps malformed transcripts out of the
output directory.
"""

import json

import pytest

from audio_transcriber.schema import (
    SCHEMA_VERSION,
    SEGMENT_KEYS,
    SchemaError,
    Segment,
    Transcript,
)


def test_segment_serializes_exactly_the_contract_keys_in_order():
    """The five keys, in this order, are the output contract."""
    segment = Segment(start_time=0.0, end_time=1.5, text="hello")

    assert tuple(segment.to_dict()) == SEGMENT_KEYS
    assert SEGMENT_KEYS == (
        "start_time",
        "end_time",
        "speaker",
        "spoken_language",
        "text",
    )


def test_segment_round_trips_through_dict():
    segment = Segment(
        start_time=0.0,
        end_time=1.5,
        text="hello",
        speaker="SPEAKER_01",
        spoken_language="en",
    )

    assert Segment.from_dict(segment.to_dict()) == segment


def test_segment_duration_is_end_minus_start():
    assert Segment(start_time=2.0, end_time=5.5, text="x").duration == 3.5


def test_segment_speaker_and_language_default_to_none():
    """Both fields are in every object, null until a backend populates them."""
    data = Segment(start_time=0.0, end_time=1.0, text="hello").to_dict()

    assert data["speaker"] is None
    assert data["spoken_language"] is None


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"start_time": -1.0, "end_time": 1.0, "text": "x"}, "start_time must be >= 0"),
        ({"start_time": 5.0, "end_time": 2.0, "text": "x"}, "before its"),
    ],
)
def test_segment_rejects_invalid_values(kwargs, message):
    with pytest.raises(SchemaError, match=message):
        Segment(**kwargs)


def test_segment_from_dict_reports_missing_keys():
    with pytest.raises(SchemaError, match="missing required keys"):
        Segment.from_dict({"start_time": 0.0, "text": "no end"})


def test_transcript_file_is_a_bare_array_of_segments():
    """A consumer reads the file and iterates it, with no wrapper to unpack."""
    transcript = Transcript(
        source="audio/meeting.wav",
        segments=(
            Segment(start_time=0.0, end_time=1.0, text="hello", spoken_language="en"),
            Segment(start_time=1.0, end_time=2.0, text="world", spoken_language="en"),
        ),
    )

    data = json.loads(json.dumps(transcript.to_json_data()))

    assert isinstance(data, list)
    assert data == [
        {
            "start_time": 0.0,
            "end_time": 1.0,
            "speaker": None,
            "spoken_language": "en",
            "text": "hello",
        },
        {
            "start_time": 1.0,
            "end_time": 2.0,
            "speaker": None,
            "spoken_language": "en",
            "text": "world",
        },
    ]


def test_transcript_round_trips_with_metadata_supplied_separately():
    transcript = Transcript(
        source="audio/meeting.wav",
        segments=(Segment(start_time=0.0, end_time=1.0, text="hello"),),
        language="en",
        duration=2.0,
        backend="reference",
        model_version="tiny-1.0",
    )

    restored = Transcript.from_json_data(
        json.loads(json.dumps(transcript.to_json_data())),
        source=transcript.source,
        language=transcript.language,
        duration=transcript.duration,
        backend=transcript.backend,
        model_version=transcript.model_version,
    )

    assert restored == transcript


def test_metadata_sidecar_carries_what_the_array_cannot():
    transcript = Transcript(
        source="a.wav",
        segments=(Segment(start_time=0.0, end_time=1.0, text="hello"),),
        language="en",
        duration=12.5,
    )

    assert transcript.metadata() == {
        "schema_version": SCHEMA_VERSION,
        "language": "en",
        "duration": 12.5,
        "segment_count": 1,
    }


def test_transcript_text_joins_segments():
    transcript = Transcript(
        source="a.wav",
        segments=(
            Segment(start_time=0.0, end_time=1.0, text="hello"),
            Segment(start_time=1.0, end_time=2.0, text="world"),
        ),
    )

    assert transcript.text == "hello world"


def test_empty_transcript_writes_an_empty_array():
    """A file with no detected speech is a successful run, not a failure."""
    transcript = Transcript(source="silence.wav")

    assert transcript.to_json_data() == []
    assert transcript.text == ""


def test_transcript_rejects_overlapping_segments():
    with pytest.raises(SchemaError, match="before the previous segment ended"):
        Transcript(
            source="a.wav",
            segments=(
                Segment(start_time=0.0, end_time=2.0, text="first"),
                Segment(start_time=1.0, end_time=3.0, text="overlaps"),
            ),
        )


def test_transcript_rejects_empty_source():
    with pytest.raises(SchemaError, match="source must not be empty"):
        Transcript(source="")


def test_from_json_data_rejects_a_wrapped_object():
    """Guards against a consumer or backend writing the old object shape."""
    with pytest.raises(SchemaError, match="is a JSON array"):
        Transcript.from_json_data({"segments": []}, source="a.wav")
