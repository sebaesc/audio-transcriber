# Implementation Plan: HTTP API Service

**Track ID:** http-api_20261005
**Spec:** [spec.md](./spec.md)
**Created:** 2026-10-05
**Status:** [ ] Not Started

## Overview

Add an `audio_transcriber.api` package behind an optional `api` extra (FastAPI + uvicorn +
python-multipart). Build it in layers: a job model and in-memory store behind a small interface,
an executor that runs the existing `pipeline` on a thread pool, the HTTP routes and error mapping,
then evaluation, the `serve` command, Docker, and docs. The pipeline, backend interface,
transcript schema and manifest format stay unchanged. If a contract change turns out to be
needed, stop and ask.

Every test drives the API through FastAPI's `TestClient` with a registered stub backend. No test
may build a hosted backend, so a `.env` with `ELEVENLABS_API_KEY` never leads to a paid call.

## Phase 1: Foundation — Extra, Package, Job Model

Make the API importable and give it a job model, without any routes yet.

### Tasks

- [ ] Task 1.1: Add the `api` extra (`fastapi`, `uvicorn[standard]`, `python-multipart`, pinned
      to major ranges) to `pyproject.toml`. Install it in `.venv` and record the versions in
      `tech-stack.md`
- [ ] Task 1.2: Make sure the core package never imports `fastapi`/`uvicorn`. Extend
      `tests/test_no_engine_leakage.py` (or add a sibling test) so that importing
      `audio_transcriber` and its CLI module does not import them
- [ ] Task 1.3: Write tests first for the job model: `JobStatus`
      (`queued`/`running`/`succeeded`/`failed`), legal transitions only, timestamps, a structured
      error (`type`, `message`, `remedy`), and JSON serialization
- [ ] Task 1.4: Implement `Job` and a `JobStore` protocol with an `InMemoryJobStore`
      (thread-safe; `create`, `get`, `update`). The protocol is the seam for the backlogged
      persistent store
- [ ] Task 1.5: Add `ApiSettings` (read from the environment): max upload size, worker
      concurrency (default 1), temp/output directory

### Verification

- [ ] `pytest` passes; `ruff check` clean; `pip install -e .` without `[api]` still works and the
      CLI runs

## Phase 2: Core — Job Execution and Transcription Routes

Run transcription jobs in the background and expose them over HTTP.

### Tasks

- [ ] Task 2.1: Add a stub backend fixture for API tests (deterministic transcript, a
      configurable delay, and a configurable failure mode). Reuse the existing conformance stub
      if one fits
- [ ] Task 2.2: Write tests first for `JobRunner`: it submits to a `ThreadPoolExecutor` sized by
      settings, moves a job `queued → running → succeeded` with the transcript stored, and
      records `failed` with a structured error for `AudioDecodeError`,
      `BackendRequirementError` and `TranscriptionError`. Temp uploads are deleted in every
      outcome
- [ ] Task 2.3: Implement `JobRunner` on top of `pipeline.run`/`validate`. Keep each job's run
      directory (transcript JSON + `manifest.json`) for the evaluation step
- [ ] Task 2.4: Write API contract tests first: `POST /transcriptions` (multipart: `file`,
      `backend`, `options` as JSON) → 202 + `{id, status}` + `Location`; unknown/uninstalled
      backend → 400 before queuing; oversize upload → 413; `GET /transcriptions/{id}` → status
      and transcript (same JSON the CLI writes) or error; unknown id → 404
- [ ] Task 2.5: Implement the FastAPI app factory (`create_app(settings, store)`), the two
      routes, and the exception handlers that map the `errors.py` hierarchy to JSON bodies
- [ ] Task 2.6: Test and implement `GET /backends` (`registry.available()` +
      `registry.unavailable()` reasons) and `GET /health`

### Verification

- [ ] A test submits a job, polls until it succeeds, and gets back a transcript identical to
      what the CLI writes for the same stub output

## Phase 3: Evaluation Endpoint

Score a finished job's transcript against a ground truth without re-transcribing.

### Tasks

- [ ] Task 3.1: Write tests first for `POST /transcriptions/{id}/evaluation` (multipart ground
      truth file): it returns the `Report` metrics (WER etc.) for a succeeded job; returns 409
      while the job is queued/running and for a failed job; returns 404 for an unknown id; returns
      422 for an invalid ground truth (`ConfigError` from `load_reference`); and never calls the
      backend (assert the stub was not invoked again)
- [ ] Task 3.2: Implement it by building a `Pair` from the job's stored transcript and the
      uploaded reference, then calling `evaluation.runner.score_transcripts` (which picks up the
      job's `manifest.json`). Return the report in the same shape the evaluation CLI writes
- [ ] Task 3.3: Clean up the uploaded reference after scoring

### Verification

- [ ] For the same transcript + ground truth, the endpoint's scores equal the evaluation CLI's
      `score_transcripts` output

## Phase 4: Integration — `serve` Command and Docker

Make the service runnable locally and in a container.

### Tasks

- [ ] Task 4.1: Add `audio-transcriber serve` (`--host`, `--port`, `--workers-concurrency`). If
      the `api` extra is missing, exit with the install command, following the CLI's existing
      error/exit-code style. Add CLI contract tests for both cases
- [ ] Task 4.2: Add a `Dockerfile`: `python:3.12-slim`, ffmpeg via apt, build arg
      `EXTRAS=api,faster-whisper,elevenlabs`, a non-root user, `EXPOSE 8000`, a `HEALTHCHECK` on
      `/health`, and `CMD audio-transcriber serve --host 0.0.0.0`. Add a `.dockerignore`
- [ ] Task 4.3: Add a `docker-compose.yml`: build args, `env_file: .env` (passes
      `ELEVENLABS_API_KEY`, `HF_TOKEN`), and a named volume for the Whisper/HF model cache
- [ ] Task 4.4: Build the image and run it, then smoke-test `/health` and `/backends`, plus one
      transcription with faster-whisper on a short sample. Do not call ElevenLabs

### Verification

- [ ] `docker compose up` serves the API; a job submitted with `curl` succeeds end to end

## Phase 5: Polish — Docs and Backlog

### Tasks

- [ ] Task 5.1: Add a README "HTTP API" section: install, `serve`, the endpoints, example `curl`
      requests (submit, poll, evaluate), the async job lifecycle, settings, and the limitation
      that jobs are in-memory and lost on restart
- [ ] Task 5.2: Add a README "Docker" section: build args, compose, `.env`, and the model cache
      volume
- [ ] Task 5.3: Update `tech-stack.md` (FastAPI/uvicorn, Docker image details) and the Mermaid
      diagrams if the run flow changed
- [ ] Task 5.4: Confirm that "Persistent job storage" is recorded in the tracks.md Backlog

### Verification

- [ ] Following the README from a clean checkout, a new user can start the service and transcribe
      a file

## Final Verification

- [ ] All acceptance criteria met
- [ ] Tests passing (`pytest`), `ruff format` / `ruff check` clean
- [ ] No test or smoke run spent ElevenLabs credits
- [ ] Documentation updated
- [ ] Ready for review

---

_Generated by Conductor. Tasks will be marked [~] in progress and [x] complete._
