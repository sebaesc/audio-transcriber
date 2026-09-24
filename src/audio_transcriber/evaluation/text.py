"""Making a reference transcript comparable to ASR output.

The Bangor Miami transcripts come from CHAT-format corpus files. The published
dataset strips the speaker tiers and most annotation, but conversational markup
survives: pause markers, retracing brackets, utterance continuations, and
partial words written with their completion in parentheses.

None of that is speech, so none of it should count against a transcriber. What
remains is normalized on both sides — reference and hypothesis — so that a
difference in case, punctuation, or accents is not scored as a word error.
"""

from __future__ import annotations

import re
import unicodedata

#: `(.)` marks a short pause.
_PAUSE = re.compile(r"\(\.+\)")

#: `<...>` brackets speech the speaker then retraced. The words inside were
#: spoken, so only the brackets are removed.
_RETRACE_BRACKETS = re.compile(r"[<>]")

#: A line beginning `+`, `+,` or `+!` continues the previous utterance.
_CONTINUATION = re.compile(r"(?m)^\s*\+[,!.]?\s*")

#: `mani(cure)` is a word the speaker cut short, written with its completion.
#: The parenthesized part was not spoken, but keeping the whole word scores
#: closer to what a transcriber will produce than keeping the fragment.
_PARTIAL_WORD = re.compile(r"\b(\w+)\((\w+)\)")

#: `&=laughs` and `&-uh` style event and filler codes.
_EVENT_CODE = re.compile(r"&[=\-]?\w+")

#: `[/]`, `[//]`, `[: text]` and similar inline CHAT codes.
_INLINE_CODE = re.compile(r"\[[^\]]*\]")

_PUNCTUATION = re.compile(r"[^\w\s']", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def clean_reference(text: str) -> str:
    """Strip conversational markup from a reference transcript, then normalize.

    Args:
        text: The transcript as published, with CHAT markup.

    Returns:
        A normalized string of the words actually spoken.
    """
    text = _CONTINUATION.sub(" ", text)
    text = _INLINE_CODE.sub(" ", text)
    text = _EVENT_CODE.sub(" ", text)
    text = _PAUSE.sub(" ", text)
    text = _RETRACE_BRACKETS.sub(" ", text)
    text = _PARTIAL_WORD.sub(r"\1\2", text)
    return normalize(text)


def normalize(text: str) -> str:
    """Lowercase, strip punctuation and accents, and collapse whitespace.

    Accents are folded deliberately. Whisper decoding under an English language
    token writes Spanish without them, and counting `esta` against `está` as a
    word error would measure orthography rather than recognition.

    Args:
        text: Any transcript text.

    Returns:
        The normalized form, suitable for word-level comparison.
    """
    text = text.lower()
    text = _PUNCTUATION.sub(" ", text)
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return _WHITESPACE.sub(" ", text).strip()
