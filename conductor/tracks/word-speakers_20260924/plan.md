# Implementation Plan: Per-Word Speaker Assignment

**Track ID:** word-speakers_20260924
**Spec:** [spec.md](./spec.md)
**Created:** 2026-09-24
**Status:** [ ] Not Started

## Overview

Extend the `faster-whisper` backend with `speaker_assignment=segment|word`. Build it in four
layers:

1. The option and its validation.
2. Carrying word timings through normalization, in both detection modes.
3. Pure assignment logic that turns words and speaker turns into split segments, tested without
   a model.
4. Wiring, docs, and an evaluation on TalkBank that decides the default.

`segment` stays the default until Phase 4 says otherwise, and its output must not change at all.
The transcript schema, backend interface and manifest format stay unchanged, so no approval pause
for a contract change is expected. If one turns out to be necessary, stop and ask.

## Phase 1: Option and Validation

Make the option selectable and reject configurations it would contradict, without changing any
output yet.

### Tasks

- [ ] Task 1.1: Write the option tests first: default is `segment`; `word` and `segment` are
      accepted with `diarization=true`; `speaker_assignment=word` without diarization is rejected
      with a `BackendRequirementError` naming `diarization=true`; an unknown value is rejected and
      lists the valid ones; `speaker_assignment=segment` without diarization is accepted, since
      it is the default
- [ ] Task 1.2: Add `speaker_assignment` to `_DEFAULTS` and `_PIPELINE_OPTIONS`, and the checks
      to `_reject_contradictions`
- [ ] Task 1.3: Test that `effective_config()` records `speaker_assignment`, and that
      `_decode_options()` never contains it

### Verification

- [ ] `pytest tests/test_faster_whisper.py` passes, and so does the full suite

## Phase 2: Word Timings Through Normalization

Keep faster-whisper's words alongside each segment inside the backend, without changing what
`segment` mode produces.

### Tasks

- [ ] Task 2.1: Test that `_decode_options()` sets `word_timestamps=True` only when `diarization`
      is on and `speaker_assignment=word`, and leaves the key out otherwise
- [ ] Task 2.2: Introduce an internal word record (start, end, text) and a way for `_normalize`
      to return each kept segment with its words. `Segment` is not changed. Apply the existing
      rules to words too: overlap clamping, end-of-audio clamping, words past the clamped end
      dropped, and the density drop still decided per segment
- [ ] Task 2.3: Shift word times onto the recording's timeline in `_transcribe_chunked`, exactly
      as segment times are shifted
- [ ] Task 2.4: Test with fake native segments: words survive normalization in both modes; times
      are shifted in chunked mode; a hallucinated segment dropped for density takes its words
      with it; a segment overhanging the audio's end loses the words past it
- [ ] Task 2.5: Regression test that `segment` mode output is unchanged, using the existing
      normalization fixtures

### Verification

- [ ] Full suite passes; no test fixture for `segment` mode needed updating

## Phase 3: Assignment and Splitting

Turn words plus speaker turns into schema segments. Pure functions in `backends/diarization.py`,
tested without a model or pyannote.

### Tasks

- [ ] Task 3.1: Write tests first for per-word assignment: each word takes
      `speaker_for(turns, start, end)`; a word overlapping no turn takes the previous word's
      speaker, or the next word's at the start of a segment; a segment with no word overlapping
      any turn keeps `None`
- [ ] Task 3.2: Write tests first for smoothing: a single word whose speaker differs from both
      neighbours is folded into them; the same word at a segment edge is not folded; two
      consecutive words by the other speaker are kept as a real change. Make the threshold a
      module constant so Phase 4 can tune it
- [ ] Task 3.3: Write tests first for splitting: one segment with a speaker change becomes two,
      with times taken from their first and last words, text joined from their words, and the
      parent's `spoken_language`; pieces are ordered and non-overlapping; a segment with one
      speaker comes back identical to the `segment` mode result
- [ ] Task 3.4: Write tests first for text fidelity: joining a segment's words reproduces its
      text, including punctuation, Spanish accents and `¿`/`¡`. If faster-whisper's words do not
      reproduce the text exactly, record the deviation here and decide which one wins
- [ ] Task 3.5: Implement assignment, smoothing and splitting to pass 3.1–3.4

### Verification

- [ ] `pytest tests/test_diarization.py` passes, and so does the full suite

## Phase 4: Wiring, Documentation and Evaluation

Connect the pieces, measure both modes, and pick the default from the numbers.

### Tasks

- [ ] Task 4.1: Branch `_maybe_diarize` on `speaker_assignment`, using Phase 3's functions in
      `word` mode; test with fake turns and fake native segments in both detection modes
- [ ] Task 4.2: Extend `tests/test_conformance.py` so `speaker_assignment=word` is one of the
      modes checked
- [ ] Task 4.3: Real run over sample audio in both modes; check the transcripts by eye at a
      known interruption
- [ ] Task 4.4: `evaluation --talkbank` on the Bangor Miami `mixed` recordings, `segment` and
      `word` modes, same seed. Record per recording: speaker `confusion`, `der`, WER and segment
      count. `maria20` is excluded from speaker scoring as before
- [ ] Task 4.5: If `word` mode fragments output or confusion does not improve, try the smoothing
      threshold at one or two other values and record every result; do not tune to a single
      recording
- [ ] Task 4.6: Decide the default. Switch to `word` only if confusion falls on the scorable
      recordings on average without a WER regression beyond the seed noise floor noted in
      `_DEFAULTS`. Record the decision and its numbers here
- [ ] Task 4.7: Document `speaker_assignment` in `docs/backends/faster-whisper.md`: what it does,
      that it needs `diarization=true`, the extra decode cost, the measured results, and the
      limits (overlapping speech, word-timing drift, one language per split segment)
- [ ] Task 4.8: Update the README example if the default changes

### Verification

- [ ] Evaluation results and the default decision recorded in this plan

## Final Verification

- [ ] All acceptance criteria met
- [ ] `ruff check . && ruff format --check . && pytest` clean
- [ ] `faster-whisper` doc page current
- [ ] Manifest records `speaker_assignment`
- [ ] A real run over sample audio produces the expected output
- [ ] Manual sign-off recorded here

---

_Generated by Conductor. Tasks will be marked [~] in progress and [x] complete._
