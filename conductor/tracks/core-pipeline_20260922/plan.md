# Implementation Plan: Core Pipeline and Backend Interface

**Track ID:** core-pipeline_20260922
**Spec:** [spec.md](./spec.md)
**Created:** 2026-09-22
**Status:** [ ] Not Started

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
- [ ] Task 1.5: Add a placeholder `README.md` describing the package and how to install it

### Verification

- [ ] `.venv/bin/pip install -e ".[dev]"` installs cleanly; `ruff check .`, `ruff format --check .`, and `pytest` all run (zero tests is acceptable here)

## Phase 2: Contracts — Schema, Interface, Registry

Define the three artifacts that every backend and consumer depends on. **This phase ends at an
approval gate:** per `workflow.md`, the backend interface, transcript schema, and manifest format
cannot change without sign-off, so they are reviewed before anything is built on them.

### Tasks

- [ ] Task 2.1: Define the transcript schema — `Transcript`, `Segment`, and metadata types, with JSON serialization. Segments carry start, end, text; speaker fields present but unpopulated
- [ ] Task 2.2: Write tests for schema serialization and round-tripping, including the unpopulated speaker fields
- [ ] Task 2.3: Define the run manifest schema — backend name, model version, effective config, input checksums, timestamps — with tests
- [ ] Task 2.4: Define the backend interface (abstract base class): `transcribe()`, capability flags, requirement checks, and the error types a backend may raise
- [ ] Task 2.5: Implement the backend registry — an explicit name-to-class dict with `register()` and `get(name)`, raising a clear error listing known names on an unknown backend
- [ ] Task 2.6: Write the shared conformance test suite that any backend must pass — parameterized over registered backends, asserting schema validity, timestamp ordering, and error behavior
- [ ] Task 2.7: Write `docs/backend-interface.md` — the contract a new backend implements, and what normalization it owns

### Verification

- [ ] `pytest` passes; conformance suite runs (and is trivially green with no backends registered)
- [ ] **Approval gate:** interface, transcript schema, and manifest format reviewed and signed off before Phase 3 begins

## Phase 3: Reference Backend (faster-whisper)

Implement one backend against the interface, proving it is implementable and that normalization
lives inside the backend module.

### Tasks

- [ ] Task 3.1: Add `faster-whisper` as an optional extra in `pyproject.toml` (`[faster-whisper]`), never a core dependency
- [ ] Task 3.2: Implement `backends/faster_whisper.py` — load the model, transcribe, and normalize native output to the transcript schema inside the module
- [ ] Task 3.3: Implement its requirement check — model availability and device selection — reporting what is missing and how to install it
- [ ] Task 3.4: Report model version and effective config back to the manifest
- [ ] Task 3.5: Register the backend and confirm it passes the shared conformance suite against a short fixture audio file
- [ ] Task 3.6: Write `docs/backends/faster-whisper.md` — usage, dependencies, credentials (none), hardware, known limits

### Verification

- [ ] `pytest -k conformance` passes with the backend registered
- [ ] Backend doc page exists and covers usage, dependencies, hardware, and limits (Gate 6 in `workflow.md`)

## Phase 4: Run Orchestration

Assemble the pipeline: discover inputs, decode audio, run the backend per file, and write outputs.
No engine-specific code anywhere in this phase.

### Tasks

- [ ] Task 4.1: Implement input discovery over a directory, with a `--glob` pattern and deterministic ordering
- [ ] Task 4.2: Implement audio decode/resample via ffmpeg subprocess, with an up-front presence check that fails fast with an install hint
- [ ] Task 4.3: Implement up-front validation — input paths, config, output directory, and the selected backend's requirements — before any file is processed
- [ ] Task 4.4: Implement the run loop: per-file transcription, writing `out/<run-id>/<name>.json`, never overwriting a prior run's output
- [ ] Task 4.5: Implement error collection — per-file failures recorded and written to `errors.json` without aborting the batch, with partial results kept
- [ ] Task 4.6: Implement manifest writing — backend name, model version, effective config, and input checksums to `manifest.json`, with tests
- [ ] Task 4.7: Write a test asserting no pipeline module imports or branches on a specific backend

### Verification

- [ ] `pytest` passes; a run over a fixture directory produces per-file transcripts, `manifest.json`, and `errors.json` in the expected layout
- [ ] A run with one deliberately broken input completes, keeps the good results, and records the failure

## Phase 5: CLI and Polish

Put the CLI in front of the pipeline and finish the error handling and documentation.

### Tasks

- [ ] Task 5.1: Implement the Typer CLI: `--input`, `--backend`, `--out`, `--glob`, and backend config passthrough
- [ ] Task 5.2: Implement exit codes distinguishing all-succeeded, partial failure, and aborted run, with tests
- [ ] Task 5.3: Route progress and errors to stderr, transcript data to files, so piping works (`product-guidelines.md`)
- [ ] Task 5.4: Apply the error message structure — what happened, why, how to fix — across validation and run failures
- [ ] Task 5.5: Write CLI contract tests — flags, exit codes, and output layout
- [ ] Task 5.6: Update `README.md` with install, a worked example run, and links to the backend docs
- [ ] Task 5.7: Pin the libraries landed by this track in `tech-stack.md` (Typer, faster-whisper, ffmpeg), replacing the TBD rows

### Verification

- [ ] CLI contract tests pass; `--help` is accurate; exit codes behave as specified
- [ ] A real run over sample audio produces the expected output

## Final Verification

- [ ] All acceptance criteria in [spec.md](./spec.md) met
- [ ] All tasks marked `[x]`
- [ ] Gate 1 — `pytest` passing
- [ ] Gate 2 — interface, normalization, and CLI covered by tests
- [ ] Gate 3 — `pytest -k conformance` passing for the reference backend
- [ ] Gate 4 — `ruff check .` clean
- [ ] Gate 5 — `ruff format --check .` clean
- [ ] Gate 6 — every touched backend has a current doc page
- [ ] Gate 7 — run manifest records backend, version, and config
- [ ] Manual sign-off recorded here

---

_Generated by Conductor. Tasks will be marked [~] in progress and [x] complete._
