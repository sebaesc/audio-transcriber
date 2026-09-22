# Implementation Plan: Core Pipeline and Backend Interface

**Track ID:** core-pipeline_20260922
**Spec:** [spec.md](./spec.md)
**Created:** 2026-09-22
**Status:** [x] Complete

## Overview

Build the contracts first, then the code that depends on them. Phase 1 creates the package and
tooling. Phase 2 defines the transcript schema, the backend interface, and the registry — the
three things `workflow.md` protects behind an approval gate — together with the conformance test
suite that every backend must pass. Phase 3 implements the reference backend against that suite,
proving the interface is implementable before anything is built on top of it. Phase 4 adds the
run orchestration (discovery, audio decode, manifest, error collection), and Phase 5 puts the CLI
in front of it and finishes the documentation.

Per `workflow.md`, tests are written for the backend interface, output normalization, the manifest,
and the CLI contract — not for thin wrappers or glue. For bug fixes found along the way, the failing
test comes first.

## Phase 1: Project Scaffolding

Create the Python package, build config, and lint setup. No pipeline logic yet.

### Tasks

- [x] Task 1.1: Create `pyproject.toml` — package metadata, Python 3.12 requirement, `[project.optional-dependencies]` with `dev` and a placeholder for backend extras
- [x] Task 1.2: Configure `[tool.ruff]` in `pyproject.toml` (lint + format), per `conductor/code_styleguides/python.md`
- [x] Task 1.3: Create the package layout: `src/audio_transcriber/` with `__init__.py`, `__main__.py`, and empty `backends/` package; `tests/` alongside
- [x] Task 1.4: Add `pytest` config and confirm `.venv/bin/pip install -e ".[dev]"` succeeds
- [x] Task 1.5: Add a placeholder `README.md` describing the package and how to install it

### Verification

- [x] `.venv/bin/pip install -e ".[dev]"` installs cleanly; `ruff check .`, `ruff format --check .`, and `pytest` all run (zero tests is acceptable here)

## Phase 2: Contracts — Schema, Interface, Registry

Define the three artifacts that every backend and consumer depends on. **This phase ends at an
approval gate:** per `workflow.md`, the backend interface, transcript schema, and manifest format
cannot change without sign-off, so they are reviewed before anything is built on them.

### Tasks

- [x] Task 2.1: Define the transcript schema — `Transcript`, `Segment`, and metadata types, with JSON serialization. Segments carry start, end, text; speaker fields present but unpopulated
- [x] Task 2.2: Write tests for schema serialization and round-tripping, including the unpopulated speaker fields
- [x] Task 2.3: Define the run manifest schema — backend name, model version, effective config, input checksums, timestamps — with tests
- [x] Task 2.4: Define the backend interface (abstract base class): `transcribe()`, capability flags, requirement checks, and the error types a backend may raise
- [x] Task 2.5: Implement the backend registry — an explicit name-to-class dict with `register()` and `get(name)`, raising a clear error listing known names on an unknown backend
- [x] Task 2.6: Write the shared conformance test suite that any backend must pass — parameterized over registered backends, asserting schema validity, timestamp ordering, and error behavior
- [x] Task 2.7: Write `docs/backend-interface.md` — the contract a new backend implements, and what normalization it owns

### Tasks (added at the approval gate)

- [x] Task 2.8: Reshape the transcript file to a bare array of five-key segment objects (`start_time`, `end_time`, `speaker`, `spoken_language`, `text`)
- [x] Task 2.9: Move per-file metadata to the manifest's input record as an optional sidecar

### Verification

- [x] `pytest` passes; conformance suite runs (and is trivially green with no backends registered)
- [x] **Approval gate:** interface, transcript schema, and manifest format reviewed and signed off before Phase 3 begins

### Deviations

**2026-09-22 — transcript file shape.** The spec's original draft wrapped segments in a metadata
object. At the approval gate the output contract was changed to a bare JSON array of segment
objects with exactly five keys, and per-file metadata moved to the manifest's input record as an
optional sidecar. `spoken_language` became a per-segment field; `confidence` was dropped from the
contract. Caught before any backend, writer, or CLI was built on the old shape. `spec.md` records
the revision.

## Phase 3: Reference Backend (faster-whisper)

Implement one backend against the interface, proving it is implementable and that normalization
lives inside the backend module.

### Tasks

- [x] Task 3.1: Add `faster-whisper` as an optional extra in `pyproject.toml` (`[faster-whisper]`), never a core dependency
- [x] Task 3.2: Implement `backends/faster_whisper.py` — load the model, transcribe, and normalize native output to the transcript schema inside the module
- [x] Task 3.3: Implement its requirement check — model availability and device selection — reporting what is missing and how to install it
- [x] Task 3.4: Report model version and effective config back to the manifest
- [x] Task 3.5: Register the backend and confirm it passes the shared conformance suite against a short fixture audio file
- [x] Task 3.6: Write `docs/backends/faster-whisper.md` — usage, dependencies, credentials (none), hardware, known limits

### Verification

- [x] `pytest -k conformance` passes with the backend registered
- [x] Backend doc page exists and covers usage, dependencies, hardware, and limits (Gate 6 in `workflow.md`)

### Deviations

**2026-09-22 — device selection.** Task 3.3 was specified as "model availability and device
selection". Implementation found that CTranslate2 reports a CUDA device whenever the driver is
visible, so `device: auto` selected a device that then failed lazily on the first segment, which
would abort a batch part-way through. `check_requirements()` therefore runs a real inference over
a silent buffer on each candidate device, falls back to CPU under `auto`, and records the
resolved device in the effective config. Task 3.1 was already satisfied by Task 1.1, which
declared the extra.

**2026-09-22 — GPU support.** Follow-up to the above, after confirming the development machine has
an RTX 3060. The CUDA runtime libraries are installed as wheels rather than system-wide, and the
backend preloads them with `RTLD_GLOBAL` because `LD_LIBRARY_PATH` is read too early to be set
from inside Python. `device: auto` now resolves to `cuda`. Measured VRAM per model size is in the
backend doc page.

## Phase 4: Run Orchestration

Assemble the pipeline: discover inputs, decode audio, run the backend per file, and write outputs.
No engine-specific code anywhere in this phase.

### Tasks

- [x] Task 4.1: Implement input discovery over a directory, with a `--glob` pattern and deterministic ordering
- [x] Task 4.2: Implement audio decode/resample via ffmpeg subprocess, with an up-front presence check that fails fast with an install hint
- [x] Task 4.3: Implement up-front validation — input paths, config, output directory, and the selected backend's requirements — before any file is processed
- [x] Task 4.4: Implement the run loop: per-file transcription, writing `out/<run-id>/<name>.json`, never overwriting a prior run's output
- [x] Task 4.5: Implement error collection — per-file failures recorded and written to `errors.json` without aborting the batch, with partial results kept
- [x] Task 4.6: Implement manifest writing — backend name, model version, effective config, and input checksums to `manifest.json`, with tests
- [x] Task 4.7: Write a test asserting no pipeline module imports or branches on a specific backend

### Verification

- [x] `pytest` passes; a run over a fixture directory produces per-file transcripts, `manifest.json`, and `errors.json` in the expected layout
- [x] A run with one deliberately broken input completes, keeps the good results, and records the failure

### Deviations

**2026-09-22 — filename collisions.** Not in the task list, but two inputs in different
directories can share a stem (`monday/standup.wav`, `tuesday/standup.wav`), and the second
transcript would have silently overwritten the first. Collisions are now disambiguated with a
counter in discovery order.

**2026-09-22 — ffmpeg not installed.** The decode tests that need a real ffmpeg skip on this
machine (4 skipped). Everything else, including the full orchestration path, is covered with a
stub decoder. The end-to-end verification against real audio is outstanding until
`apt install ffmpeg`.

## Phase 5: CLI and Polish

Put the CLI in front of the pipeline and finish the error handling and documentation.

### Tasks

- [x] Task 5.1: Implement the Typer CLI: `--input`, `--backend`, `--out`, `--glob`, and backend config passthrough
- [x] Task 5.2: Implement exit codes distinguishing all-succeeded, partial failure, and aborted run, with tests
- [x] Task 5.3: Route progress and errors to stderr, transcript data to files, so piping works (`product-guidelines.md`)
- [x] Task 5.4: Apply the error message structure — what happened, why, how to fix — across validation and run failures
- [x] Task 5.5: Write CLI contract tests — flags, exit codes, and output layout
- [x] Task 5.6: Update `README.md` with install, a worked example run, and links to the backend docs
- [x] Task 5.7: Pin the libraries landed by this track in `tech-stack.md` (Typer, faster-whisper, ffmpeg), replacing the TBD rows

### Verification

- [x] CLI contract tests pass; `--help` is accurate; exit codes behave as specified
- [x] A real run over sample audio produces the expected output

## Final Verification

- [x] All acceptance criteria in [spec.md](./spec.md) met
- [x] All tasks marked `[x]`
- [x] Gate 1 — `pytest` passing
- [x] Gate 2 — interface, normalization, and CLI covered by tests
- [x] Gate 3 — `pytest -k conformance` passing for the reference backend
- [x] Gate 4 — `ruff check .` clean
- [x] Gate 5 — `ruff format --check .` clean
- [x] Gate 6 — every touched backend has a current doc page
- [x] Gate 7 — run manifest records backend, version, and config
- [x] Manual sign-off recorded here

### Phase 5 deviations

**2026-09-22 — Whisper hallucination past the end of audio.** The real-audio verification found
`large-v3` returning 18 segments running to 40.38s for a 10.435s recording, including text in
another language: hallucination into the silence Whisper pads short audio with. Two fixes:
normalization now drops segments starting past the audio duration and clamps overhanging ones,
and the **pipeline validates every transcript against the shared conformance assertions before
writing it**, so a backend that breaks the contract produces a reported failure rather than
plausible-looking bad data. The second is a change in where the contract is enforced, recorded as
a decision in `tech-stack.md`.

**2026-09-22 — the leakage guard caught the CLI.** Task 4.7's test flagged `cli.py` for naming
`faster-whisper` in a help string. Not a branch, but the pipeline should not name an engine even
in an example, so the message now points at `docs/backends/` instead.

### Verification run

```
$ python -m audio_transcriber -i <audio> -b faster-whisper -o <out> --option model=large-v3
warning: 1 of 5 files failed. See <out>/2026-09-22T15-56-52Z/errors.json
Transcribed 4 files in 32s → <out>/2026-09-22T15-56-52Z/
EXIT: 1
```

Five inputs across four container formats (`.flac`, `.m4a`, `.mp3`, `.ogg`) plus one deliberately
corrupt file, two of them sharing a stem across directories. Four transcribed on GPU, the corrupt
one isolated into `errors.json`, distinct filenames preserved, exit code 1 for partial. English
and Hindi both transcribed with correct `spoken_language`.

**Signed off:** 2026-09-22.

### Correction, 2026-09-22 (during `codeswitch-diarization_20260922`)

**Gate 7 passed on an untested assumption.** The gate reads "run manifest records backend,
version, config", and the manifest did record them. What was never checked is whether replaying
that record reproduces the transcript. It did not. Two defects, both found by the evaluation
harness built in the next track, neither visible to this track's test suite:

1. **Decoding was not reproducible.** faster-whisper defaults `temperature` to a fallback ladder
   and samples above 0.0. The same recording transcribed twice gave 689 vs 642 segments and WER
   0.409 vs 0.441. `product-guidelines.md` asks for seeds to be set "where the backend exposes
   them"; CTranslate2 exposes one and the backend did not use it. Fixed by pinning a seed, which
   keeps the ladder's accuracy — removing it would have cost 0.185 WER.

2. **The manifest could record options that were never applied.** The backend accepted any
   keyword, recorded it, and forwarded only those `transcribe()` named inline. A run configured
   with `no_repeat_ngram_size=3` produced byte-identical output to one without it while the
   manifest claimed otherwise. Fixed by deriving decode options from the config and rejecting
   unknown ones.

**What this says about the gate, not just the code.** Gate 7 verified that fields were written,
not that the record was sufficient to reproduce the run. A reproducibility gate has to re-run
something and compare. Any future track asserting reproducibility should transcribe one file
twice and diff the output, which is now a two-minute check with the eval harness in place.

---

_Generated by Conductor. Tasks will be marked [~] in progress and [x] complete._
