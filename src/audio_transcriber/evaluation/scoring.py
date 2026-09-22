"""Scoring a hypothesis transcript against a reference.

Word error rate alone is not enough for this corpus. A transcriber can lower WER
by emitting less: deletions cost one error each, while a wrong guess costs a
substitution and risks an insertion too. A run that silently drops a third of the
audio can therefore look like an improvement.

So every score carries a **coverage** figure alongside WER, and the language mix
each transcript reports. A change that improves WER while coverage falls is not
an improvement, and the eval report says so rather than leaving it to be noticed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Spanish function words. Short, common, and rare in English, so counting them
#: estimates how much of a transcript was written as Spanish. This measures
#: orthography, not meaning: it is how we detect English being transcribed under
#: a Spanish language token, and vice versa.
SPANISH_MARKERS = frozenset(
    """que de la el en los las un una por para con pero como mas muy yo tu es no
    si se lo le me te esta eso esto aqui alli porque cuando donde tambien ahora
    entonces bueno todo nada algo siempre nunca hacer tiene tengo vamos verdad
    del al su sus mi mis ese esa esos esas hay ya solo otro otra""".split()
)

ENGLISH_MARKERS = frozenset(
    """the and is was that this with for you he she they what when where because
    about would there their here just like know think going have has had been
    will can could should from them your our its were are am""".split()
)


@dataclass(frozen=True, slots=True)
class Score:
    """How one hypothesis compared to its reference.

    Attributes:
        wer: Word error rate.
        substitutions: Words transcribed as a different word.
        deletions: Reference words missing from the hypothesis.
        insertions: Hypothesis words absent from the reference.
        hits: Words matched.
        reference_words: Length of the reference.
        hypothesis_words: Length of the hypothesis.
        reference_spanish_share: Spanish share of the reference, by marker words.
        hypothesis_spanish_share: Spanish share of the hypothesis.
    """

    wer: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    reference_words: int
    hypothesis_words: int
    reference_spanish_share: float
    hypothesis_spanish_share: float

    @property
    def coverage(self) -> float:
        """Hypothesis length as a fraction of reference length.

        A proxy for how much of the audio was transcribed at all. Well below 1.0
        means content was dropped; well above means the transcriber is inventing.
        """
        if not self.reference_words:
            return 0.0
        return self.hypothesis_words / self.reference_words

    @property
    def language_mix_error(self) -> float:
        """Absolute difference between hypothesis and reference Spanish share.

        The measure this track exists to improve. A transcript can have a
        respectable WER while attributing every word to the wrong language.
        """
        return abs(self.hypothesis_spanish_share - self.reference_spanish_share)

    def to_dict(self) -> dict[str, Any]:
        """Return the score as JSON-serializable data."""
        return {
            "wer": round(self.wer, 4),
            "coverage": round(self.coverage, 4),
            "language_mix_error": round(self.language_mix_error, 4),
            "reference_spanish_share": round(self.reference_spanish_share, 4),
            "hypothesis_spanish_share": round(self.hypothesis_spanish_share, 4),
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "hits": self.hits,
            "reference_words": self.reference_words,
            "hypothesis_words": self.hypothesis_words,
        }


@dataclass(slots=True)
class Comparison:
    """Two scores and the verdict on whether the change helped.

    Attributes:
        baseline: The score to beat.
        candidate: The score being judged.
        coverage_tolerance: How much coverage may fall before a WER improvement
            is treated as bought rather than earned.
    """

    baseline: Score
    candidate: Score
    coverage_tolerance: float = 0.02
    notes: list[str] = field(default_factory=list)

    @property
    def coverage_regressed(self) -> bool:
        """Whether the candidate transcribed materially less than the baseline."""
        return (
            self.candidate.coverage < self.baseline.coverage - self.coverage_tolerance
        )

    @property
    def improved(self) -> bool:
        """Whether the candidate is better on WER without losing coverage."""
        return self.candidate.wer < self.baseline.wer and not self.coverage_regressed

    def to_dict(self) -> dict[str, Any]:
        """Return the comparison as JSON-serializable data."""
        return {
            "improved": self.improved,
            "coverage_regressed": self.coverage_regressed,
            "wer_delta": round(self.candidate.wer - self.baseline.wer, 4),
            "coverage_delta": round(
                self.candidate.coverage - self.baseline.coverage, 4
            ),
            "language_mix_error_delta": round(
                self.candidate.language_mix_error - self.baseline.language_mix_error, 4
            ),
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
        }


def spanish_share(text: str) -> float:
    """Estimate the Spanish share of a normalized transcript.

    Args:
        text: Normalized text, as produced by `evaluation.text.normalize`.

    Returns:
        Spanish marker words as a fraction of all marker words, or 0.0 when the
        text contains none.
    """
    words = text.split()
    spanish = sum(1 for word in words if word in SPANISH_MARKERS)
    english = sum(1 for word in words if word in ENGLISH_MARKERS)
    total = spanish + english
    return spanish / total if total else 0.0


def score(reference: str, hypothesis: str) -> Score:
    """Score a hypothesis against a reference.

    Args:
        reference: Cleaned, normalized reference text.
        hypothesis: Normalized hypothesis text.

    Returns:
        The score, including coverage and language mix.
    """
    import jiwer

    measured = jiwer.process_words(reference or " ", hypothesis or " ")
    return Score(
        wer=measured.wer,
        substitutions=measured.substitutions,
        deletions=measured.deletions,
        insertions=measured.insertions,
        hits=measured.hits,
        reference_words=len(reference.split()),
        hypothesis_words=len(hypothesis.split()),
        reference_spanish_share=spanish_share(reference),
        hypothesis_spanish_share=spanish_share(hypothesis),
    )
