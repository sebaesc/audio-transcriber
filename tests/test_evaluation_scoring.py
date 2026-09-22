"""Tests for scoring.

The point of these is that the eval cannot be gamed by transcribing less, which
is exactly how the chunked prototype produced a misleading WER win.
"""

import pytest

from audio_transcriber.evaluation.scoring import (
    ENGLISH_MARKERS,
    SPANISH_MARKERS,
    Comparison,
    Score,
    score,
    spanish_share,
)


def test_identical_text_scores_zero():
    result = score("hello world", "hello world")

    assert result.wer == 0.0
    assert result.coverage == 1.0


def test_substitution_is_counted():
    result = score("hello world", "hello there")

    assert result.substitutions == 1
    assert result.wer == pytest.approx(0.5)


def test_deletion_is_counted():
    result = score("hello big world", "hello world")

    assert result.deletions == 1


def test_insertion_is_counted():
    result = score("hello world", "hello big world")

    assert result.insertions == 1


def test_coverage_falls_when_words_are_dropped():
    """The signal that a WER win was bought rather than earned."""
    result = score("one two three four five six", "one two three")

    assert result.coverage == pytest.approx(0.5)


def test_coverage_exceeds_one_when_the_transcriber_invents():
    result = score("one two", "one two three four")

    assert result.coverage == pytest.approx(2.0)


def test_coverage_of_an_empty_reference_is_zero_not_an_error():
    assert score("", "anything").coverage == 0.0


def test_spanish_share_of_spanish_text():
    assert spanish_share("yo no se que es esto pero") > 0.9


def test_spanish_share_of_english_text():
    assert spanish_share("the thing is that you know what they think") < 0.1


def test_spanish_share_of_mixed_text_is_between():
    share = spanish_share("yo creo que the party is nice pero no se")

    assert 0.2 < share < 0.9


def test_spanish_share_without_markers_is_zero():
    assert spanish_share("xyzzy plugh frobnicate") == 0.0


def test_marker_lists_do_not_overlap():
    """A word counted for both languages would make the measure meaningless."""
    assert not (SPANISH_MARKERS & ENGLISH_MARKERS)


def test_language_mix_error_is_the_gap_from_the_reference():
    result = score("yo no se que es esto", "the thing is that you know")

    assert result.language_mix_error > 0.8


def test_matching_language_mix_scores_near_zero_error():
    text = "yo creo que the party is nice pero no se"
    result = score(text, text)

    assert result.language_mix_error == 0.0


def _score(wer: float, coverage_words: int) -> Score:
    return Score(
        wer=wer,
        substitutions=0,
        deletions=0,
        insertions=0,
        hits=0,
        reference_words=100,
        hypothesis_words=coverage_words,
        reference_spanish_share=0.5,
        hypothesis_spanish_share=0.5,
    )


def test_a_lower_wer_with_stable_coverage_is_an_improvement():
    comparison = Comparison(baseline=_score(0.44, 95), candidate=_score(0.34, 95))

    assert comparison.improved
    assert not comparison.coverage_regressed


def test_a_lower_wer_bought_by_dropping_content_is_not_an_improvement():
    """The prototype's failure mode: WER 0.44 to 0.34 while losing 14% of words."""
    comparison = Comparison(baseline=_score(0.44, 95), candidate=_score(0.34, 81))

    assert comparison.coverage_regressed
    assert not comparison.improved


def test_a_higher_wer_is_never_an_improvement():
    comparison = Comparison(baseline=_score(0.42, 95), candidate=_score(0.57, 95))

    assert not comparison.improved


def test_small_coverage_movement_is_tolerated():
    comparison = Comparison(baseline=_score(0.44, 95), candidate=_score(0.34, 94))

    assert not comparison.coverage_regressed
    assert comparison.improved


def test_comparison_reports_its_deltas():
    data = Comparison(baseline=_score(0.44, 95), candidate=_score(0.34, 81)).to_dict()

    assert data["wer_delta"] == pytest.approx(-0.10, abs=1e-4)
    assert data["coverage_delta"] == pytest.approx(-0.14, abs=1e-4)
    assert data["improved"] is False
