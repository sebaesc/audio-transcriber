"""TalkBank's original annotations of the Bangor Miami corpus.

The Hugging Face copy the harness transcribes against has its CHAT markup
stripped, speaker tiers and language marks included. TalkBank's original
transcripts keep both, and their timestamps align with the Hugging Face audio:
matching transcript words against our own segments peaks at a 0 s offset on
all five `mixed` recordings. So they can score what the stripped copy cannot —
who spoke, and which language each stretch of speech is in.

**Licence.** TalkBank data is CC BY-NC-SA 3.0. Its ground rules keep downloaded
data on the local device and forbid passing it on, so the archive is read from
a local cache and never committed; only aggregate scores are. Any use must
cite: Deuchar, M., & Parafita Couto, M. (2011). The Bangor Miami Corpus.
TalkBank. doi:10.21415/T5J01D.

Access needs a free TalkBank account. Download the transcripts, while logged
in, from https://talkbank.org/data/biling/Bangor/Miami?f=zip and save the zip
as `archive_path()`.
"""

from __future__ import annotations

import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from audio_transcriber.errors import ConfigError

DOWNLOAD_URL = "https://talkbank.org/data/biling/Bangor/Miami?f=zip"
CITATION = (
    "Deuchar, M., & Parafita Couto, M. (2011). The Bangor Miami Corpus. "
    "TalkBank. doi:10.21415/T5J01D"
)

#: CHAT's three-letter language codes, mapped to the codes the schema uses.
_ISO = {"eng": "en", "spa": "es"}

#: In a bilingual file, `@s` switches to whichever language is not the base.
_OTHER = {"eng": "spa", "spa": "eng"}

_BULLET = re.compile("\x15(\\d+)_(\\d+)\x15")
_PRECODE = re.compile(r"\[- (\w+)\]")
_BRACKETED = re.compile(r"\[[^\]]*\]")
_SWITCH = re.compile(r"^(.*?)@s(?::([\w&+]+))?$")
_PAUSE = re.compile(r"^\(\.+\)$")
_NAME = re.compile(r"^([a-z]+)(\d+)$")

#: CHAT's placeholders for speech that could not be made out.
_UNINTELLIGIBLE = frozenset({"xxx", "yyy", "www"})


@dataclass(frozen=True, slots=True)
class Utterance:
    """One reference utterance.

    Attributes:
        speaker: CHAT speaker code, e.g. `JAM`.
        start: Start in seconds, or None when the utterance is untimed.
        end: End in seconds, or None when the utterance is untimed.
        language: `en`, `es`, `mixed` when it holds both, or None when it has
            no word whose language is known.
    """

    speaker: str
    start: float | None
    end: float | None
    language: str | None

    @property
    def timed(self) -> bool:
        """Whether the utterance has a place on the recording's timeline."""
        return self.start is not None and self.end is not None


@dataclass(frozen=True, slots=True)
class ChatTranscript:
    """A parsed CHAT file.

    Attributes:
        languages: The file's languages, base language first.
        utterances: Every utterance, in file order, timed or not.
    """

    languages: tuple[str, ...]
    utterances: tuple[Utterance, ...]


def archive_path() -> Path:
    """Return where the TalkBank transcripts zip is expected.

    Honors `AUDIO_TRANSCRIBER_TALKBANK_ARCHIVE`.
    """
    override = os.environ.get("AUDIO_TRANSCRIBER_TALKBANK_ARCHIVE")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "audio-transcriber" / "talkbank" / "Miami.zip"


def _word_language(token: str, base: str) -> str | None:
    """Return the CHAT language of one token, or None if it is not a word."""
    token = token.strip("<>")
    if (
        not token
        or token.startswith(("&", "0", "+"))
        or _PAUSE.match(token)
        or not any(character.isalpha() for character in token)
    ):
        return None
    match = _SWITCH.match(token)
    if match:
        word, explicit = match.groups()
        if explicit and ("&" in explicit or "+" in explicit):
            return None
        language = explicit or _OTHER.get(base, base)
    else:
        word, language = token.split("@", 1)[0], base
    if word.lower() in _UNINTELLIGIBLE:
        return None
    return language


def utterance_language(text: str, base: str) -> str | None:
    """Return the language of one utterance's words.

    Unmarked words are in the base language. A `[- spa]` precode replaces the
    base for the whole utterance; a word marked `@s` is in the other language;
    `@s:eng&spa` marks a word belonging to both, which is evidence of neither.

    Args:
        text: The utterance's text, CHAT markup included.
        base: The file's base language as a CHAT code, e.g. `eng`.

    Returns:
        `en` or `es` when every word is in one language, `mixed` when both
        occur, or None when no word has a known language.
    """
    precode = _PRECODE.search(text)
    if precode:
        base = precode.group(1)
    text = _BRACKETED.sub(" ", text)

    found = {
        language
        for token in text.split()
        if (language := _word_language(token, base)) is not None
    }
    if not found:
        return None
    if len(found) > 1:
        return "mixed"
    (language,) = found
    return _ISO.get(language, language)


def parse_chat(text: str) -> ChatTranscript:
    """Parse the utterances of a CHAT file.

    Dependent tiers (`%mor`, `%eng`, ...) are ignored; only main tiers are
    speech.
    """
    languages: tuple[str, ...] = ()
    utterances: list[Utterance] = []
    for line in text.splitlines():
        if line.startswith("@Languages:"):
            languages = tuple(code.strip() for code in line.split(":", 1)[1].split(","))
            continue
        if not line.startswith("*"):
            continue
        speaker, _, body = line[1:].partition(":")
        bullet = _BULLET.search(body)
        start = end = None
        if bullet:
            start, end = int(bullet.group(1)) / 1000, int(bullet.group(2)) / 1000
            body = body[: bullet.start()]
        base = languages[0] if languages else "eng"
        utterances.append(
            Utterance(
                speaker=speaker.strip(),
                start=start,
                end=end,
                language=utterance_language(body, base),
            )
        )
    return ChatTranscript(
        languages=tuple(_ISO.get(code, code) for code in languages),
        utterances=tuple(utterances),
    )


def _file_name(recording: str) -> str:
    """Return TalkBank's file name for a recording, e.g. `sastre8` -> `sastre08.cha`."""
    match = _NAME.match(recording)
    if not match:
        return f"{recording}.cha"
    stem, number = match.groups()
    return f"{stem}{int(number):02d}.cha"


def load(recording: str, archive: Path | None = None) -> ChatTranscript:
    """Load the reference annotation for one recording.

    Some recordings are annotated more than once — `zeledon14` also has a
    one-speaker English-only pass. The bilingual annotation is preferred, then
    the one with the most timed utterances.

    Args:
        recording: Recording name as the harness corpus spells it.
        archive: The TalkBank zip. Defaults to `archive_path()`.

    Raises:
        ConfigError: If the archive is missing or does not hold the recording.
    """
    archive = archive or archive_path()
    if not archive.is_file():
        raise ConfigError(
            f"no TalkBank archive at {archive}. With a TalkBank account, download "
            f"{DOWNLOAD_URL} while logged in and save it there, or set "
            "AUDIO_TRANSCRIBER_TALKBANK_ARCHIVE."
        )

    wanted = _file_name(recording)
    with zipfile.ZipFile(archive) as bundle:
        candidates = [
            parse_chat(bundle.read(name).decode("utf-8"))
            for name in bundle.namelist()
            if Path(name).name == wanted
        ]
    if not candidates:
        raise ConfigError(f"{recording} ({wanted}) is not in the TalkBank archive.")
    return max(
        candidates,
        key=lambda parsed: (
            len(parsed.languages) > 1,
            sum(utterance.timed for utterance in parsed.utterances),
        ),
    )
