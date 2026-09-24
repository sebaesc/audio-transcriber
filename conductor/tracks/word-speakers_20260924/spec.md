# Specification: Per-Word Speaker Assignment

**Track ID:** word-speakers_20260924
**Type:** Feature
**Created:** 2026-09-24
**Status:** Draft

## Summary

Add a `speaker_assignment` option to the `faster-whisper` backend. With `word`, each word gets a
speaker, and a segment is split wherever the speaker changes. With `segment`, the default, the
current behaviour is unchanged: the whole segment goes to whoever speaks most of it.

## Context

Diarization shipped in `codeswitch-diarization_20260922`. pyannote produces speaker turns, and
`diarization.speaker_for` gives each transcript segment the speaker whose turns overlap it the
most. Whisper segments are often several seconds long and in conversation can hold more than one
speaker, so interruptions, quick back-and-forth and short replies ("yeah", "sí, claro") are
labelled with the wrong speaker.

Measured against TalkBank in that track, speaker confusion was 3.7–9.7% on the four recordings
that can be scored. The evaluation scores the transcript's segments, not the diarizer's raw
turns (see `docs/evaluation.md`), so a segment attributed to the wrong speaker as a whole shows
up directly in `confusion`.

WhisperX solves this with wav2vec2 forced alignment followed by per-word assignment. It was
considered and rejected (2026-09-24):

- It aligns with one language model per file, which breaks on this corpus's code-switched
  audio and on `language_detection=chunked`.
- It pins its own faster-whisper, pyannote and torch versions.
- It would duplicate the backend rather than extend it.

faster-whisper's own word timestamps are less precise than forced alignment, but pyannote's
turns are coarser still, so they are precise enough to decide which speaker said each word.

## User Story

As a member of the internal team, I want speaker labels to follow speaker changes inside a
Whisper segment, so that interruptions and short replies in conversational recordings are
attributed to the person who said them.

## Acceptance Criteria

- [ ] `--option speaker_assignment=word` gives each word a speaker and splits segments where the
      speaker changes; `speaker_assignment=segment` (the default) produces output byte-identical
      to the current release
- [ ] Setting `speaker_assignment` to a value other than the default without `diarization=true`
      is rejected with a `BackendRequirementError`, the same way chunked-only options are
      rejected in `file` mode; an unknown value is rejected too
- [ ] Words overlapping no speaker turn take a neighbouring word's speaker rather than `None`,
      and a speaker change lasting a single word is not split out on its own (see Technical Notes)
- [ ] Works with both `language_detection=file` and `language_detection=chunked`; split segments
      keep their segment's `spoken_language`
- [ ] The option is recorded in the manifest's resolved config
- [ ] Both modes are scored with `evaluation --talkbank` on the four scorable recordings, the
      results are recorded in the plan, and the default is set from them
- [ ] `docs/backends/faster-whisper.md` documents the option, its cost and its limits

## Dependencies

- `codeswitch-diarization_20260922` (complete): diarization, `speaker_for`, chunked detection
  and the TalkBank speaker scoring this track is judged by.
- TalkBank annotations for the Bangor Miami `mixed` recordings, as used by `--talkbank`.
- A GPU run of `large-v3` for the evaluation: about 4 minutes per 30-minute recording per pass,
  plus diarization, times two modes.

## Out of Scope

- WhisperX, or wav2vec2 forced alignment of any kind
- Exposing word timings in the output. The transcript schema keeps exactly its five keys, and
  words stay internal to the backend
- Word-level language ID. Split segments inherit their segment's language
- Changes to the diarization model, pyannote settings or `speaker_for`'s overlap rule
- The `elevenlabs-scribe` backend, which returns per-word speakers natively
- Changing the transcript schema, the backend interface or the manifest format

## Technical Notes

- **Option:** `speaker_assignment: "segment" | "word"`, added to `_DEFAULTS` and
  `_PIPELINE_OPTIONS` so it never reaches `transcribe()`. The check goes in
  `_reject_contradictions`, matching the existing chunked-only check.
- **Word timestamps only in `word` mode.** `word_timestamps=True` changes how faster-whisper
  places segment boundaries, not just what it returns, so turning it on unconditionally would
  change `segment` mode output. Set it in `_decode_options()` only when
  `diarization` is on and `speaker_assignment=word`.
- **Words have to survive normalization.** `_normalize` currently builds `Segment`s and drops
  everything else, and `_transcribe_chunked` shifts segment times onto the recording's timeline.
  Word mode needs an internal record per segment that carries its words, with the same shifting,
  overlap clamping, end-of-audio clamping and density drop applied. A word past the clamped end
  is dropped along with its time. `Segment` itself does not change.
- **Assignment:** each word takes `speaker_for(turns, word.start, word.end)`.
- **Gaps:** a word overlapping no turn takes the previous word's speaker, or the next one's at
  the start of a segment. A segment where no word overlaps a turn keeps `None`, as today.
- **Smoothing:** a run of a single word whose speaker differs from the words on both sides is
  folded into its neighbours. Whisper word timings drift by tens of milliseconds, and a
  one-word flip at a turn boundary is more often timing error than a real one-word turn. The
  threshold (one word, or a duration) is chosen from the evaluation, not assumed.
- **Splitting:** each run of same-speaker words becomes one segment. Its start and end are its
  first word's start and last word's end, and its text is its words joined. Adjacent pieces must
  stay non-overlapping and in order, as the schema requires.
- **Text fidelity:** joining faster-whisper's words must reproduce the segment text. Its words
  carry their leading spaces, so concatenating and stripping should match `segment.text`. A test
  pins this, including punctuation and Spanish text.
- **Engine-specific:** all of this lives in `backends/faster_whisper.py` and
  `backends/diarization.py`. Nothing in the pipeline changes.

## Risks

- **Word timestamps may cost WER.** They change segmentation. The evaluation compares WER between
  modes as well as speaker confusion, so a win on speakers bought with worse text is visible.
- **Hallucinated words carry made-up timings.** The density drop runs per segment before
  splitting, so a hallucination is still discarded whole rather than scattered across speakers.
- **Overlapping speech.** Crosstalk is common in this corpus. A word inside overlapping turns goes
  to whichever turn overlaps it most, and at crosstalk that may alternate. Smoothing limits the
  fragmentation this causes.
- **The gain may be small.** If most speaker changes already fall between segments, the default
  stays `segment` and the option remains for conversational audio. That is an acceptable outcome.

---

_Generated by Conductor. Review and edit as needed._
