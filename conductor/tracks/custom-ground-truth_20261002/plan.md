# Implementation Plan: Evaluate Against a Custom Ground Truth

**Track ID:** custom-ground-truth_20261002
**Spec:** [spec.md](./spec.md)
**Created:** 2026-10-02
**Status:** [~] In Progress

## Overview

Add a ground-truth loader that reads the five-key JSON array into timed reference utterances, then
let the runner score against those instead of the Bangor Miami corpus, either by running a backend
on the user's recording or by reading a transcript the pipeline already wrote. The existing
`scoring` and `annotation` functions are reused unchanged; the new code is input discovery,
validation, and wiring. Everything stays inside `evaluation/`: no change to the pipeline CLI, the
transcript schema or the run manifest format, so no contract-approval pause is needed.

## Phase 1: Ground-Truth Loading and Validation

Read and validate ground-truth files, and pair them with recordings, before anything is transcribed.

### Tasks

- [ ] Task 1.1: Tests for the ground-truth loader: valid file; overlapping segments accepted;
      not-an-array, empty array, missing key, `end_time` before `start_time` and invalid JSON each
      rejected with file name and segment index in the message
- [ ] Task 1.2: `evaluation/reference.py`: `load_reference(path)` validates each segment with
      `Segment.from_dict` (not `Transcript`, so overlaps are allowed), sorts by time, and returns
      the segments plus the file's checksum
- [ ] Task 1.3: Convert ground-truth segments into timed `Utterance`s (speaker, start, end,
      language), and report which dimensions have labels; move `Utterance` to a shared module if
      `talkbank.py` is the wrong home for it
- [ ] Task 1.4: Reference text: segments' `text` joined in time order and run through `normalize`
- [ ] Task 1.5: Tests and implementation for pair discovery: a single `--audio`/`--reference` pair,
      and a folder of `name.<audio/video ext>` + `name.json`; an unpaired file on either side is an
      error naming it; `--only` filters by name

### Verification

- [ ] `pytest tests/evaluation` passes; every malformed fixture fails with a message that names
      the file and the problem

## Phase 2: Scoring a Backend Run Against Custom Ground Truth

Run a backend on each custom recording and score it with the existing metrics.

### Tasks

- [ ] Task 2.1: Tests (fake backend, hand-written ground truth): WER/`retained` per recording;
      diarization and language scores present when labels exist; skipped with a reason when not
- [ ] Task 2.2: Generalize `runner.evaluate` so the recordings and their references come from either
      the corpus or a custom source, leaving corpus-mode behavior and its reports unchanged
- [ ] Task 2.3: Score speakers and language from the ground truth through `annotation`, reusing the
      TalkBank path's skip logic and reasons; the duration comes from the decoded audio
- [ ] Task 2.4: Write the normalized reference `.txt` next to the hypothesis `.txt` and `.json`
- [ ] Task 2.5: Record each reference file's name and checksum in the report; check that
      `compare` works on two custom-ground-truth reports (test)

### Verification

- [ ] Existing corpus-mode tests still pass unchanged; a custom run over fixtures produces a
      report that `compare` accepts

## Phase 3: Scoring an Existing Transcript

Score a transcript the pipeline already wrote, with no backend run.

### Tasks

- [ ] Task 3.1: Tests: single transcript + reference; a pipeline output folder matched to a
      reference folder by name; backend/model/config read from the run manifest when present and
      marked unknown when absent; an unmatched transcript or reference is an error
- [ ] Task 3.2: Load a transcript file through `Transcript`/`Segment` and score it with the Phase 2
      scoring path, without creating a backend
- [ ] Task 3.3: Find the run manifest beside the transcripts and take backend, model version and
      config from it (read only; the manifest format is not changed)
- [ ] Task 3.4: Duration without audio: the later of the last reference and hypothesis `end_time`

### Verification

- [ ] Scoring an existing transcript gives the same scores as the Phase 2 run that produced it
      (test with the fake backend)

## Phase 4: CLI, Documentation and a Real Run

Expose the modes on the command line, document them, and measure real audio.

### Tasks

- [ ] Task 4.1: CLI tests and implementation: `--audio`/`--reference`, `--references DIR`,
      `--transcript`/`--transcripts DIR`; the three modes are mutually exclusive; `--talkbank` is
      corpus-only; clear errors and exit codes for misuse
- [ ] Task 4.2: `docs/evaluation.md`: writing a ground truth (with an example), both input forms,
      scoring an existing transcript, what each metric means on custom data, `en`/`es`-only language
      accuracy, overlap allowed, and where reports go (`eval/custom/`)
- [ ] Task 4.3: Real run: one of the team's ~20 min recordings with its ground truth, on
      faster-whisper `large-v3`; check the scores and the reference/hypothesis diff look right
- [ ] Task 4.4: Score the same transcript again in existing-transcript mode and confirm the scores
      match; commit the report under `eval/custom/`
- [ ] Task 4.5: Update `tech-stack.md` (decision log entry) and the README/pipeline overview if
      they describe the evaluation modes

### Verification

- [ ] `ruff check . && ruff format --check . && pytest` clean; the real-run report is committed

## Final Verification

- [ ] All acceptance criteria met
- [ ] Tests passing
- [ ] Documentation updated (`docs/evaluation.md`, `tech-stack.md`)
- [ ] Manual sign-off recorded in this plan
- [ ] Ready for review

---

_Generated by Conductor. Tasks will be marked [~] in progress and [x] complete._
