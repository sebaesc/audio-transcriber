# Implementation Plan: Code-Switching Language Detection and Speaker Diarization

**Track ID:** codeswitch-diarization_20260922
**Spec:** [spec.md](./spec.md)
**Created:** 2026-09-22
**Status:** [ ] Not Started

## Overview

Build the measurement before the thing being measured. Phase 1 turns the spike's throwaway eval
scripts into a committed harness with a recorded baseline, because without it no later change can
be shown to help. Phase 2 fixes the chunking, which the prototype got wrong in two ways — too
coarse, and dropping speech. Phase 3 puts per-chunk language detection behind backend config and
proves it against the baseline. Phase 4 adds diarization, gated on the ground-truth question
settled in Phase 1. Phase 5 documents and tunes.

Nothing in this track touches the pipeline, the schema, the interface, or the manifest. Both
fields already exist; this fills them. If a change here appears to need a contract change, that is
a signal to stop and reconsider the approach, per the approval rules in `workflow.md`.

## Phase 1: Eval Harness and Baseline

Make the improvement measurable. The spike's numbers live in a scratch directory and are not
reproducible; this phase makes them so.

### Tasks

- [x] Task 1.1: Add an `eval` optional extra (`jiwer`, `pyarrow`) — never a core or backend dependency
- [~] Task 1.2: Write the dataset fetcher: download the `mixed` config on demand into a gitignored cache, verify it, and never commit audio
- [ ] Task 1.3: Implement reference cleaning for CHAT markup — `(.)` pauses, `<>` retracing, `+` continuations, `mani(cure)` partials — with tests over hand-written examples
- [ ] Task 1.4: Implement scoring: WER via `jiwer`, plus a language-mix measure reporting transcribed Spanish share against the reference share
- [ ] Task 1.5: Write the eval runner — takes a backend name and config, runs the `mixed` subset, writes a JSON report
- [ ] Task 1.6: Record the per-file baseline for all 5 recordings and commit the report as the number to beat
- [ ] Task 1.7: **Settle the diarization ground-truth question** (see Risks in the spec): TalkBank credentials, another labelled corpus, or qualitative-only. Record the decision here before Phase 4 is planned in detail

### Verification

- [ ] `pytest` passes; reference cleaning is covered by tests
- [ ] The eval runs end to end and reproduces a WER near 0.44 for `zeledon4` under the current default
- [ ] The baseline report is committed, and Task 1.7's decision is written into this plan

## Phase 2: Chunking

Fix what the prototype got wrong. This phase changes no behaviour visible to a user — it produces
correct chunk boundaries and proves they lose nothing.

### Tasks

- [ ] Task 2.1: Implement VAD-based chunk grouping with a configurable target length, replacing the prototype's logic which produced ~68 s chunks against a 28 s target
- [ ] Task 2.2: Write tests asserting chunk lengths stay within the target and boundaries land on speech gaps
- [ ] Task 2.3: **Fix the content-loss regression.** The prototype produced 3683 words against a 4269-word reference, with 666 deletions; measure what VAD discards and close the gap
- [ ] Task 2.4: Add a coverage assertion to the eval — transcribed audio duration against total speech duration — so a future change cannot buy WER by dropping speech

### Verification

- [ ] Chunk lengths respect the configured target
- [ ] Transcribing chunk-by-chunk under the file-level language scores no worse than the baseline; this isolates chunking from detection
- [ ] Coverage assertion in place and passing

## Phase 3: Per-Chunk Language Detection

The measured win. Opt-in, restricted to `{en, es}`.

### Tasks

- [ ] Task 3.1: Implement per-chunk detection taking the best of `{en, es}` from `all_language_probs`, not the global argmax
- [ ] Task 3.2: Implement the confidence threshold and its fallback to the file-level language; choose the value from the eval rather than by intuition
- [ ] Task 3.3: Add `language_detection` backend config (`file` default, `chunked` opt-in) and the allowed-language set as config
- [ ] Task 3.4: Populate `spoken_language` per segment from its chunk's detected language
- [ ] Task 3.5: Record the detection mode, threshold, and allowed languages in the effective config, so a run's manifest explains its own output
- [ ] Task 3.6: Write tests — detection restricted to the allowed set, threshold fallback, and `spoken_language` varying within one transcript
- [ ] Task 3.7: Run the eval and compare against the Phase 1 baseline

### Verification

- [ ] `pytest -k conformance` still passes in both modes
- [ ] WER improves on the baseline and Spanish share lands within 0.05 of the reference
- [ ] Coverage assertion still passing — the win is not bought with deletions
- [ ] The default path is byte-for-byte unchanged when `language_detection=file`

## Phase 4: Speaker Diarization

Scope depends on Task 1.7. If no ground truth is available, this phase ships the capability with
qualitative evaluation only, and says so plainly in the doc page.

### Tasks

- [ ] Task 4.1: Select a diarization model and record why, including its licence and gating
- [ ] Task 4.2: Add it as its own optional extra, loaded only when diarization is enabled
- [ ] Task 4.3: Implement requirement checks — model availability, credentials, device — reported up front like every other backend requirement
- [ ] Task 4.4: Assign speaker labels to segments, stable within a transcript
- [ ] Task 4.5: Make `capabilities.diarization` reflect whether it is actually active, rather than what the engine could do
- [ ] Task 4.6: Record the diarization model and its version in the manifest, so a transcript's speaker labels are traceable
- [ ] Task 4.7: Evaluate against whatever Task 1.7 settled on, and write the result down

### Verification

- [ ] `speaker` is populated and stable within a transcript when enabled, `null` when not
- [ ] Disabled by default; nothing about the Phase 3 output changes when it is off
- [ ] `pytest -k conformance` passes with diarization both on and off

## Phase 5: Tuning and Documentation

### Tasks

- [ ] Task 5.1: Tune chunk length and confidence threshold against the eval; record the chosen values and what they cost
- [ ] Task 5.2: Measure and document throughput for each mode — chunked detection adds a pass, diarization adds a model
- [ ] Task 5.3: Update `docs/backends/faster-whisper.md` with the new options, their cost, and their limits
- [ ] Task 5.4: Document the eval harness: how to run it, what it measures, and how to read the report
- [ ] Task 5.5: Record in the spec's known limits that intra-sentential switching is not expressible at chunk granularity
- [ ] Task 5.6: Decide whether `chunked` should become the default, with the eval as the argument either way

### Verification

- [ ] Doc page current, per gate 6
- [ ] A real run over bilingual audio produces varying `spoken_language` and populated `speaker`

## Final Verification

- [ ] All acceptance criteria in [spec.md](./spec.md) met
- [ ] Gate 1 — `pytest` passing
- [ ] Gate 2 — detection, chunking, and normalization covered by tests
- [ ] Gate 3 — `pytest -k conformance` passing in every mode
- [ ] Gate 4 — `ruff check .` clean
- [ ] Gate 5 — `ruff format --check .` clean
- [ ] Gate 6 — the `faster-whisper` doc page covers every new option
- [ ] Gate 7 — manifest records detection mode, threshold, languages, and diarization model
- [ ] Eval report committed, showing the improvement over the recorded baseline
- [ ] Manual sign-off recorded here

---

_Generated by Conductor. Tasks will be marked [~] in progress and [x] complete._
