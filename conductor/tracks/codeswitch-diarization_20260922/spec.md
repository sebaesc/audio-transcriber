# Specification: Code-Switching Language Detection and Speaker Diarization

**Track ID:** codeswitch-diarization_20260922
**Type:** Feature
**Created:** 2026-09-22
**Status:** Draft

## Summary

Label each transcript segment with the language actually spoken and the speaker who spoke it, for
audio that switches between Spanish and English within a single recording.

## Context

The corpus this project transcribes is mostly English with Spanish speakers, and a single file
routinely contains both. The pipeline shipped in `core-pipeline_20260922` cannot express that:
faster-whisper detects one language per file and decodes the whole recording under it.

Measured on a real 21.8-minute Bangor Miami conversation (`zeledon4`, 43% Spanish words):

| | Result |
| --- | --- |
| File-level language detected | `es` |
| Segments | 658 |
| Distinct `spoken_language` values | `{es}` — every segment, including the English ones |
| Reference Spanish share | 0.63 |
| Transcribed Spanish share | 0.77 |
| WER | 0.439 |

This is worse than a missing feature. `spoken_language` is populated with a value that is wrong
for 37% of the audio, and nothing downstream can detect that it happened. Decoding English under
a Spanish language token also drags the English content toward Spanish, which is where the 0.77
against a reference of 0.63 comes from.

A prototype split the audio on speech boundaries, detected the language of each chunk among
`{en, es}` only, and transcribed each chunk under its own language. **It did not replicate across
two recordings:**

| | `zeledon4` (21.8 min) | `herring17` (30 min) |
| --- | --- | --- |
| Baseline WER | 0.439 | 0.419 |
| Chunked WER | 0.336 | 0.568 |
| Reference Spanish share | 0.629 | 0.598 |
| Baseline Spanish share | 0.770 | 0.383 |
| Chunked Spanish share | 0.628 | 0.685 |
| Chunks produced | 19 (~68 s each) | 11 (~164 s each) |

The prototype improved one recording by 23% relative and made the other 36% worse. The chunk
counts explain why: the grouping logic is broken, producing chunks one to three minutes long
against a 28 s target. At that granularity a single language decision covers minutes of bilingual
conversation, which forces long stretches into one language — hence the Spanish share overshooting
to 0.685 on `herring17`.

Two further observations from the same runs:

- **Per-file detection is unstable across this corpus.** The baseline detected `es` for one
  recording and `en` for the other, despite both being 20–80% Spanish. Whichever it picks, the
  other language is mislabelled and distorted.
- **The prototype dropped content.** 3683 hypothesis words against a 4269-word reference on
  `zeledon4`, with 666 deletions. Part of that recording's WER "win" came from VAD discarding
  speech rather than transcribing it better.

So the approach is unproven, not proven. That is the premise of this track: build the measurement
first, fix the chunking, and only then decide whether per-chunk detection earns its place.

Speaker diarization is included because in bilingual conversation the speaker and the language
usually switch together: the same chunk boundaries serve both questions, and the transcript
schema already carries `speaker` alongside `spoken_language`, unpopulated since the first track.

## User Story

As a member of the internal team, I want each transcript segment labelled with the language
actually spoken and the speaker who spoke it, so that a bilingual recording produces usable,
correctly attributed text instead of one language's guess applied to the whole file.

## Acceptance Criteria

- [ ] Language detection runs per chunk, not per file, and is restricted to `{en, es}`
- [ ] `spoken_language` varies within a transcript when the audio switches language
- [ ] `speaker` is populated with stable labels within a run's transcript
- [ ] A reproducible eval harness reports WER and language-mix error against the Bangor Miami
      ground truth, and is runnable on demand
- [ ] Measured WER improves on the recorded baseline across **all five** `mixed` recordings, not
      one, and transcribed Spanish share lands within 0.05 of each reference share
- [ ] Chunked detection is opt-in configuration on the existing `faster-whisper` backend; the
      shipped per-file default behaviour is unchanged unless it is enabled
- [ ] No pipeline or schema changes — the existing contracts already express both fields
- [ ] The backend doc page covers the new options, their cost, and their limits

## Dependencies

- `core-pipeline_20260922` — complete and merged. Provides the interface, schema, manifest, CLI,
  and the `faster-whisper` backend this track extends.

External:

- The Bangor Miami Spanish-English Corpus on Hugging Face
  (`BrunoHays/Bangor-Miami-Spanish-English-Corpus`, CC BY-SA 3.0), `mixed` config: 5 recordings,
  ~2.5 h, 20–80% Spanish share. Downloaded on demand, never committed.
- A diarization model. Most candidates (pyannote) are gated on Hugging Face and need an accepted
  licence plus a token.

## Out of Scope

- Languages other than Spanish and English — detection is restricted to `{en, es}` deliberately
- Intra-sentential switching below chunk granularity. Real code-switching in this corpus happens
  mid-sentence ("me lo hago simplemente para if I have a party"); chunk-level detection cannot
  express that, and word-level language ID is a separate problem
- Translation between the two languages
- Changing the transcript schema, the backend interface, or the manifest format
- Making chunked detection the default. It is opt-in until the eval justifies otherwise
- Diarization for any backend other than `faster-whisper`
- Speaker identification — labels are stable within a transcript, not across files

## Technical Notes

- **Placement:** configuration on the existing `faster-whisper` backend, e.g.
  `--option language_detection=chunked`, defaulting to the current per-file behaviour. One
  backend, one doc page, and the shipped default is untouched.
- **Chunking:** `faster_whisper.vad.get_speech_timestamps` for speech boundaries, grouped to a
  target length. The prototype's grouping was too coarse — 19 chunks for 21.8 minutes, roughly
  68 s each, against a 28 s target — so the grouping logic needs fixing, not just tuning.
- **Detection:** `model.detect_language()` per chunk, taking the best of `{en, es}` from
  `all_language_probs` rather than the global argmax. A confidence threshold below which the
  chunk falls back to the file-level language needs choosing from data.
- **Known regression to fix:** the prototype raised deletions to 666 and produced 3683 words
  against a 4269-word reference. VAD is discarding speech the baseline captured. This must be
  resolved before the feature is considered done — a WER win bought by dropping audio is not a win.
- **Evaluate on all five recordings, never one.** The prototype's result inverted between the two
  recordings tested. Any claim about this feature is made against the whole `mixed` subset.
- **Diarization:** candidate models are gated. Whatever is chosen is an optional extra, loaded
  only when diarization is enabled, and the backend's `capabilities.diarization` flag must reflect
  whether it is active.
- **Eval:** WER via `jiwer` after normalizing case, punctuation, and accents, plus a language-mix
  measure. The reference needs CHAT markup stripped — `(.)` pauses, `<>` retracing, `+`
  continuations, `mani(cure)` partial words.
- **Measured throughput** for sizing the eval: `large-v3` on an RTX 3060 runs real conversational
  audio at about 5.1x realtime, so a 30-minute recording takes roughly 4 minutes per pass.

## Risks

- **No diarization ground truth.** The Hugging Face dataset ships transcripts with CHAT markup
  stripped, including the speaker tiers — there are no speaker labels in it. The original TalkBank
  source has them, but `media.talkbank.org` is behind an authentication modal. Diarization quality
  therefore cannot be measured against this corpus as things stand. Resolve by obtaining TalkBank
  credentials, finding another labelled corpus, or accepting qualitative evaluation on our own
  audio — decided in Phase 1, before diarization work starts.
- **Spanish and English are close.** The prototype's per-chunk detection was validated at high
  confidence on English vs Hindi. Spanish against English is a harder discrimination and should be
  expected to produce more borderline chunks.
- **Chunked decoding costs a second model pass** over each chunk for detection. It came out faster
  than the baseline here because VAD skips silence, but that will not hold for dense audio.

---

_Generated by Conductor. Review and edit as needed._
