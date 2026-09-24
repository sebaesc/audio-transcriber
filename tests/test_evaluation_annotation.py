"""Tests for scoring a transcript against timed reference annotations.

The references are hand-built utterances, not corpus data. What is tested is
the arithmetic: which time counts, and what it counts as.
"""

import pytest

from audio_transcriber.evaluation.annotation import (
    diarization_error,
    diarization_scorable,
    language_confusion,
    scorable_region,
)
from audio_transcriber.evaluation.talkbank import Utterance
from audio_transcriber.schema import Segment


def _utt(speaker, start, end, language="en"):
    return Utterance(speaker=speaker, start=start, end=end, language=language)


def _seg(start, end, speaker=None, language="en"):
    return Segment(
        start_time=start,
        end_time=end,
        text="x",
        speaker=speaker,
        spoken_language=language,
    )


# --- Which time is scored ---


def test_a_fully_timed_reference_scores_the_whole_recording():
    utterances = [_utt("A", 1.0, 2.0), _utt("B", 3.0, 4.0)]

    assert scorable_region(utterances, duration=10.0) == [(0.0, 10.0)]


def test_the_gap_holding_an_untimed_utterance_is_not_scored():
    """Observed on maria20: a third of its utterances carry no timestamp.

    Their speech is somewhere between the neighbouring timed utterances, and
    scoring that gap would call anything transcribed there a false alarm.
    """
    utterances = [
        _utt("A", 1.0, 2.0),
        _utt("B", None, None),
        _utt("A", 5.0, 6.0),
    ]

    assert scorable_region(utterances, duration=10.0) == [(0.0, 2.0), (5.0, 10.0)]


def test_untimed_speech_after_the_last_timestamp_excludes_the_tail():
    utterances = [_utt("A", 1.0, 2.0), _utt("B", None, None)]

    assert scorable_region(utterances, duration=10.0) == [(0.0, 2.0)]


# --- Diarization error ---


def test_perfect_attribution_scores_zero():
    reference = [_utt("A", 0.0, 4.0), _utt("B", 5.0, 9.0)]
    hypothesis = [_seg(0.0, 4.0, "SPEAKER_00"), _seg(5.0, 9.0, "SPEAKER_01")]

    result = diarization_error(reference, hypothesis, duration=10.0, collar=0.0)

    assert result["der"] == pytest.approx(0.0)


def test_labels_are_mapped_not_compared_by_name():
    """Our `SPEAKER_01` can be their `A`; only consistency matters."""
    reference = [_utt("A", 0.0, 4.0), _utt("B", 5.0, 9.0)]
    hypothesis = [_seg(0.0, 4.0, "SPEAKER_01"), _seg(5.0, 9.0, "SPEAKER_00")]

    assert diarization_error(reference, hypothesis, 10.0, collar=0.0)["der"] == 0.0


def test_speech_given_to_the_wrong_speaker_is_confusion():
    reference = [_utt("A", 0.0, 4.0), _utt("B", 4.0, 8.0)]
    hypothesis = [_seg(0.0, 8.0, "SPEAKER_00")]

    result = diarization_error(reference, hypothesis, 10.0, collar=0.0)

    assert result["confusion"] == pytest.approx(0.5)
    assert result["missed"] == pytest.approx(0.0)


def test_a_segment_without_a_speaker_is_missed_speech():
    """A `null` speaker is an admitted gap, and scored as one."""
    reference = [_utt("A", 0.0, 4.0)]
    hypothesis = [_seg(0.0, 4.0, None)]

    assert diarization_error(reference, hypothesis, 10.0, collar=0.0)["missed"] == 1.0


def test_speech_outside_the_scorable_region_is_not_a_false_alarm():
    reference = [_utt("A", 0.0, 2.0), _utt("B", None, None), _utt("A", 6.0, 8.0)]
    hypothesis = [
        _seg(0.0, 2.0, "S0"),
        _seg(3.0, 5.0, "S1"),  # the untimed utterance, probably
        _seg(6.0, 8.0, "S0"),
    ]

    result = diarization_error(reference, hypothesis, 10.0, collar=0.0)

    assert result["false_alarm"] == pytest.approx(0.0)


def test_the_collar_forgives_boundary_imprecision():
    reference = [_utt("A", 0.0, 4.0), _utt("B", 4.0, 8.0)]
    hypothesis = [_seg(0.0, 4.2, "S0"), _seg(4.2, 8.0, "S1")]

    strict = diarization_error(reference, hypothesis, 10.0, collar=0.0)
    forgiving = diarization_error(reference, hypothesis, 10.0, collar=0.5)

    assert strict["der"] > 0.0
    assert forgiving["der"] == pytest.approx(0.0)


def test_the_reference_speech_scored_is_reported():
    reference = [_utt("A", 0.0, 4.0), _utt("B", 5.0, 9.0)]

    result = diarization_error(reference, [], 10.0, collar=0.0)

    assert result["reference_seconds"] == pytest.approx(8.0)
    assert result["missed"] == pytest.approx(1.0)


# --- Language ---


def test_language_is_weighted_by_overlapping_time():
    reference = [_utt("A", 0.0, 6.0, "en"), _utt("B", 6.0, 10.0, "es")]
    hypothesis = [_seg(0.0, 10.0, language="en")]

    result = language_confusion(reference, hypothesis)

    assert result["accuracy"] == pytest.approx(0.6)
    assert result["recall"] == {"en": pytest.approx(1.0), "es": pytest.approx(0.0)}


def test_mixed_utterances_are_reported_apart_from_accuracy():
    """No single label is right for a mixed utterance, so none is scored."""
    reference = [_utt("A", 0.0, 5.0, "en"), _utt("B", 5.0, 10.0, "mixed")]
    hypothesis = [_seg(0.0, 10.0, language="en")]

    result = language_confusion(reference, hypothesis)

    assert result["accuracy"] == pytest.approx(1.0)
    assert result["mixed_share"] == pytest.approx(0.5)


def test_utterances_without_a_known_language_are_ignored():
    reference = [_utt("A", 0.0, 5.0, None), _utt("B", 5.0, 10.0, "es")]
    hypothesis = [_seg(0.0, 10.0, language="es")]

    result = language_confusion(reference, hypothesis)

    assert result["accuracy"] == pytest.approx(1.0)
    assert result["scored_seconds"] == pytest.approx(5.0)


def test_untimed_utterances_are_ignored_for_language():
    reference = [_utt("A", None, None, "es"), _utt("B", 0.0, 4.0, "en")]
    hypothesis = [_seg(0.0, 4.0, language="en")]

    assert language_confusion(reference, hypothesis)["accuracy"] == 1.0


def test_nothing_to_score_reports_none_rather_than_dividing_by_zero():
    result = language_confusion([_utt("A", 0.0, 4.0, "en")], [])

    assert result["accuracy"] is None


# --- Whether a reference can score speakers at all ---


def test_a_reference_timing_every_speaker_can_score_diarization():
    utterances = [_utt("A", 0.0, 1.0), _utt("B", 1.0, 2.0), _utt("B", None, None)]

    assert diarization_scorable(utterances) is None


def test_a_reference_timing_only_one_speaker_cannot():
    """Observed on maria20: every utterance of one of its two speakers is untimed.

    With one speaker timed, any speech given to the other counts as confusion —
    46% on maria20 — which says nothing about the diarizer.
    """
    utterances = [_utt("A", 0.0, 1.0), _utt("B", None, None), _utt("B", None, None)]

    assert "B" in diarization_scorable(utterances)
