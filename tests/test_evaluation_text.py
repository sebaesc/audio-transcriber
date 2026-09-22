"""Tests for reference cleaning.

The reference is the yardstick, so markup leaking into it would inflate every
WER this project ever reports. Each case here is markup observed in the Bangor
Miami transcripts.
"""

from audio_transcriber.evaluation.text import clean_reference, normalize


def test_pause_markers_are_removed():
    assert clean_reference("el tema (.) es de lo que") == "el tema es de lo que"


def test_retracing_brackets_are_removed_but_the_words_are_kept():
    """The speaker did say those words, so they count."""
    assert clean_reference("es <de lo que <de lo del") == "es de lo que de lo del"


def test_utterance_continuations_are_removed():
    text = "+ no para otra persona .\n+, yo respeto tu forma ."

    assert clean_reference(text) == "no para otra persona yo respeto tu forma"


def test_partial_words_are_completed():
    """`mani(cure) manicure` is one cut-off attempt then the full word."""
    assert clean_reference("del mani(cure) manicure y") == "del manicure manicure y"


def test_inline_codes_are_removed():
    assert clean_reference("yo creo [//] yo pienso") == "yo creo yo pienso"


def test_event_codes_are_removed():
    assert clean_reference("bueno &=laughs si &-uh claro") == "bueno si claro"


def test_case_is_folded():
    assert normalize("He Hoped There") == "he hoped there"


def test_punctuation_is_removed():
    assert normalize("qué piensa usted de eso ?") == "que piensa usted de eso"


def test_accents_are_folded():
    """Whisper in English mode writes Spanish unaccented; that is not an error."""
    assert normalize("está aquí más años") == "esta aqui mas anos"


def test_apostrophes_are_kept():
    """`it's` and `its` are different words."""
    assert normalize("it's nice") == "it's nice"


def test_whitespace_is_collapsed():
    assert normalize("a   b\n\nc\t d") == "a b c d"


def test_cleaning_is_idempotent():
    """Cleaning already-clean text must not change it."""
    once = clean_reference("el tema (.) es <de lo que mani(cure) .")

    assert clean_reference(once) == once


def test_a_real_transcript_excerpt_survives_cleaning():
    raw = (
        "el tema (.) es <de lo que <de lo del mani(cure) manicure y pedicure "
        "que los hombres se hacen .\n"
        "qué piensa usted de eso ?\n"
        "+ no <para otra persona (.) para otra persona ."
    )

    cleaned = clean_reference(raw)

    assert "(" not in cleaned and ")" not in cleaned
    assert "<" not in cleaned and ">" not in cleaned
    assert "+" not in cleaned and "?" not in cleaned
    assert cleaned.startswith("el tema es de lo que de lo del manicure manicure")


def test_empty_input_is_empty_output():
    assert clean_reference("") == ""
