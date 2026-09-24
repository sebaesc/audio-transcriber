"""Tests for reading TalkBank's CHAT annotations of the Bangor Miami corpus.

Every transcript here is hand-written in CHAT's format. TalkBank's licence keeps
the corpus on the local device, so no real line of it belongs in the repo, test
fixtures included.
"""

import zipfile
from pathlib import Path

import pytest

from audio_transcriber.errors import ConfigError
from audio_transcriber.evaluation import talkbank

B = "\x15"  # CHAT's media-bullet delimiter


def _chat(*utterances: str, languages: str = "eng, spa") -> str:
    """Build a minimal CHAT file around the given utterance lines."""
    header = [
        "@UTF8",
        "@Begin",
        f"@Languages:\t{languages}",
        "@Participants:\tAAA Ana Adult, BBB Bea Adult",
        "@Media:\tsample, audio",
    ]
    return "\n".join([*header, *utterances, "@End", ""])


def _utt(speaker: str, text: str, start_ms: int | None, end_ms: int | None) -> str:
    bullet = f" {B}{start_ms}_{end_ms}{B}" if start_ms is not None else ""
    return f"*{speaker}:\t{text}{bullet}"


# --- Language of an utterance ---


def test_unmarked_words_are_in_the_file_base_language():
    assert talkbank.utterance_language("I told her already .", "eng") == "en"


def test_a_word_marked_at_s_is_in_the_other_language():
    """`@s` marks a switch away from the utterance's base language."""
    assert talkbank.utterance_language("I told her ya@s .", "eng") == "mixed"


def test_a_precode_switches_the_whole_utterance():
    """`[- spa]` makes Spanish the base, so unmarked words are Spanish."""
    assert talkbank.utterance_language("[- spa] no sé .", "eng") == "es"


def test_at_s_inside_a_precoded_utterance_switches_back():
    assert talkbank.utterance_language("[- spa] no sé party@s .", "eng") == "mixed"


def test_every_word_switched_makes_the_other_language():
    assert talkbank.utterance_language("hola@s amiga@s .", "eng") == "es"


def test_an_ambiguous_word_counts_for_neither_language():
    """`@s:eng&spa` belongs to both languages, so it is evidence of neither."""
    assert talkbank.utterance_language("okay@s:eng&spa .", "eng") is None
    assert talkbank.utterance_language("I okay@s:eng+spa .", "eng") == "en"


def test_unintelligible_words_carry_no_language():
    assert talkbank.utterance_language("xxx yyy .", "eng") is None


def test_markup_is_not_counted_as_words():
    """Pauses, events, fillers, retracings and omitted words are not speech."""
    text = "&-um <I think> [/] I (.) &=laughs 0am know +/."

    assert talkbank.utterance_language(text, "eng") == "en"


# --- Parsing a file ---


def test_utterances_are_timed_in_seconds_with_their_speaker():
    parsed = talkbank.parse_chat(
        _chat(_utt("AAA", "hello there .", 1500, 2750), _utt("BBB", "hi .", 3000, 3400))
    )

    first, second = parsed.utterances
    assert (first.speaker, first.start, first.end) == ("AAA", 1.5, 2.75)
    assert (second.speaker, second.language) == ("BBB", "en")


def test_the_base_language_comes_from_the_languages_header():
    parsed = talkbank.parse_chat(
        _chat(_utt("AAA", "no sé .", 0, 1000), languages="spa, eng")
    )

    assert parsed.languages == ("es", "en")
    assert parsed.utterances[0].language == "es"


def test_an_untimed_utterance_is_kept_without_times():
    """Untimed speech still happened; scoring has to know where it might be."""
    parsed = talkbank.parse_chat(
        _chat(_utt("AAA", "one .", 0, 1000), _utt("BBB", "two .", None, None))
    )

    assert parsed.utterances[1].start is None
    assert parsed.utterances[1].speaker == "BBB"


def test_dependent_tiers_are_ignored():
    parsed = talkbank.parse_chat(
        _chat(
            _utt("AAA", "[- spa] hola .", 0, 1000),
            "%eng:\thello .",
            "%mor:\tco|hola .",
        )
    )

    assert len(parsed.utterances) == 1
    assert parsed.utterances[0].language == "es"


# --- Finding a recording in the archive ---


def _archive(tmp_path: Path, files: dict[str, str]) -> Path:
    path = tmp_path / "Miami.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return path


def test_a_recording_name_is_matched_with_its_number_zero_padded(tmp_path):
    """The Hugging Face corpus says `sastre8`; TalkBank's file is `sastre08`."""
    archive = _archive(
        tmp_path, {"eng/sastre/sastre08.cha": _chat(_utt("AAA", "hi .", 0, 500))}
    )

    parsed = talkbank.load("sastre8", archive)

    assert parsed.utterances[0].speaker == "AAA"


def test_the_bilingual_annotation_wins_over_a_monolingual_one(tmp_path):
    """Observed: `zeledon14` exists twice, once as a one-speaker English pass."""
    archive = _archive(
        tmp_path,
        {
            "eng/zeledon/zeledon14.cha": _chat(
                _utt("PAR", "hello .", 0, 500), languages="eng"
            ),
            "spa/zeledon/zeledon14.cha": _chat(
                _utt("HER", "hola .", 0, 500), languages="spa, eng"
            ),
        },
    )

    assert talkbank.load("zeledon14", archive).utterances[0].speaker == "HER"


def test_a_missing_archive_says_how_to_get_it(tmp_path):
    with pytest.raises(ConfigError, match="talkbank.org"):
        talkbank.load("herring17", tmp_path / "absent.zip")


def test_a_recording_absent_from_the_archive_is_reported(tmp_path):
    archive = _archive(tmp_path, {"eng/maria/maria01.cha": _chat()})

    with pytest.raises(ConfigError, match="maria20"):
        talkbank.load("maria20", archive)


def test_the_archive_location_honors_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("AUDIO_TRANSCRIBER_TALKBANK_ARCHIVE", str(tmp_path / "m.zip"))

    assert talkbank.archive_path() == tmp_path / "m.zip"
