# Implementation Plan: HTTP API Service

**Track ID:** http-api_20261005
**Spec:** [spec.md](./spec.md)
**Created:** 2026-10-05
**Status:** [~] In Progress

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

- [x] Task 1.1: Add the `api` extra (`fastapi`, `uvicorn[standard]`, `python-multipart`, pinned
      to major ranges) to `pyproject.toml`. Install it in `.venv` and record the versions in
      `tech-stack.md`
- [x] Task 1.2: Make sure the core package never imports `fastapi`/`uvicorn`. Extend
      `tests/test_no_engine_leakage.py` (or add a sibling test) so that importing
      `audio_transcriber` and its CLI module does not import them
- [x] Task 1.3: Write tests first for the job model: `JobStatus`
      (`queued`/`running`/`succeeded`/`failed`), legal transitions only, timestamps, a structured
      error (`type`, `message`, `remedy`), and JSON serialization
- [x] Task 1.4: Implement `Job` and a `JobStore` protocol with an `InMemoryJobStore`
      (thread-safe; `create`, `get`, `update`). The protocol is the seam for the backlogged
      persistent store
- [x] Task 1.5: Add `ApiSettings` (read from the environment): max upload size, worker
      concurrency (default 1), temp/output directory

### Notes

- 1.2 is a regression guard in its own file, `tests/test_api_optional.py`: it imports the core
  modules in a fresh interpreter, because the API tests import FastAPI into the test process
- 1.5 settings: `AUDIO_TRANSCRIBER_MAX_UPLOAD_MB` (default 1024), `AUDIO_TRANSCRIBER_CONCURRENCY`
  (default 1), `AUDIO_TRANSCRIBER_WORK_DIR` (default `<tmp>/audio-transcriber-api`)

### Verification

- [x] `pytest` passes; `ruff check` clean; `pip install -e .` without `[api]` still works and the
      CLI runs

## Phase 2: Core — Job Execution and Transcription Routes

Run transcription jobs in the background and expose them over HTTP.

### Tasks

- [x] Task 2.1: Add a stub backend fixture for API tests (deterministic transcript, a
      configurable delay, and a configurable failure mode). Reuse the existing conformance stub
      if one fits
- [x] Task 2.2: Write tests first for `JobRunner`: it submits to a `ThreadPoolExecutor` sized by
      settings, moves a job `queued → running → succeeded` with the transcript stored, and
      records `failed` with a structured error for `AudioDecodeError`,
      `BackendRequirementError` and `TranscriptionError`. Temp uploads are deleted in every
      outcome
- [x] Task 2.3: Implement `JobRunner` on top of `pipeline.run`/`validate`. Keep each job's run
      directory (transcript JSON + `manifest.json`) for the evaluation step
- [x] Task 2.4: Write API contract tests first: `POST /transcriptions` (multipart: `file`,
      `backend`, `options` as JSON) → 202 + `{id, status}` + `Location`; unknown/uninstalled
      backend → 400 before queuing; oversize upload → 413; `GET /transcriptions/{id}` → status
      and transcript (same JSON the CLI writes) or error; unknown id → 404
- [x] Task 2.5: Implement the FastAPI app factory (`create_app(settings, store)`), the two
      routes, and the exception handlers that map the `errors.py` hierarchy to JSON bodies
- [x] Task 2.6: Test and implement `GET /backends` (`registry.available()` +
      `registry.unavailable()` reasons) and `GET /health`

### Notes

- 2.1: the conformance stub is private to its test and has no delay or failure modes, so the API
  has its own, `tests/api_stub.py`. Its options choose the behavior (`fail`, `wait`) and it hides
  the built-ins, so no API test can build a real engine
- 2.2: found and fixed a race — a job's outcome was recorded before its upload was removed
- 2.4/2.5: bad backend options (constructing the backend is offline and cheap) are also a 400
  before queuing. Missing form fields are a 422 in the same error shape. An oversize upload is
  refused by its `Content-Length` before the multipart parser spools it, and byte-exact while
  copying. Error bodies are `{"error": {"type", "message", "remedy"}}`
- Deviation: Starlette 1.7 deprecates `httpx` for `TestClient`; `httpx2` joins the `dev` extra

### Verification

- [x] A test submits a job, polls until it succeeds, and gets back a transcript identical to
      what the CLI writes for the same stub output

## Phase 3: Evaluation Endpoint

Score a finished job's transcript against a ground truth without re-transcribing.

### Tasks

- [x] Task 3.1: Write tests first for `POST /transcriptions/{id}/evaluation` (multipart ground
      truth file): it returns the `Report` metrics (WER etc.) for a succeeded job; returns 409
      while the job is queued/running and for a failed job; returns 404 for an unknown id; returns
      422 for an invalid ground truth (`ConfigError` from `load_reference`); and never calls the
      backend (assert the stub was not invoked again)
- [x] Task 3.2: Implement it by building a `Pair` from the job's stored transcript and the
      uploaded reference, then calling `evaluation.runner.score_transcripts` (which picks up the
      job's `manifest.json`). Return the report in the same shape the evaluation CLI writes
- [x] Task 3.3: Clean up the uploaded reference after scoring

### Notes

- The multipart field is `ground_truth`; the report's label is the job id
- Addition: a missing scoring library (`jiwer` → `eval`, `pyannote.metrics` → `diarization`) is a
  501 naming the extra to install, rather than a 500. The Docker image therefore needs `eval`
  in its default extras (see Phase 4)
- The shared API fixtures moved to `tests/api_fixtures.py`, loaded as a pytest plugin

### Verification

- [x] For the same transcript + ground truth, the endpoint's scores equal the evaluation CLI's
      `score_transcripts` output

## Phase 4: Integration — `serve` Command and Docker

Make the service runnable locally and in a container.

### Tasks

- [x] Task 4.1: Add `audio-transcriber serve` (`--host`, `--port`, `--workers-concurrency`). If
      the `api` extra is missing, exit with the install command, following the CLI's existing
      error/exit-code style. Add CLI contract tests for both cases
- [x] Task 4.2: Add a `Dockerfile`: `python:3.12-slim`, ffmpeg via apt, build arg
      `EXTRAS=api,faster-whisper,elevenlabs`, a non-root user, `EXPOSE 8000`, a `HEALTHCHECK` on
      `/health`, and `CMD audio-transcriber serve --host 0.0.0.0`. Add a `.dockerignore`
- [x] Task 4.3: Add a `docker-compose.yml`: build args, `env_file: .env` (passes
      `ELEVENLABS_API_KEY`, `HF_TOKEN`), and a named volume for the Whisper/HF model cache
- [x] Task 4.4: Build the image and run it, then smoke-test `/health` and `/backends`, plus one
      transcription with faster-whisper on a short sample. Do not call ElevenLabs

### Notes

- 4.1: the CLI was a single Typer command; a second one would have made `--input` need a verb.
  The run form is now the app's callback (`invoke_without_command=True`), so existing usage and
  every existing CLI test are unchanged, and `serve` is a command beside it
- 4.2 deviation: the default `EXTRAS` adds `eval`, so the evaluation endpoint works in the image
  (`api,eval,faster-whisper,elevenlabs`). Image: 1.27 GB, CPU only
- 4.4 found a real bug: a fresh install resolves PyAV 19, which removed an argument
  faster-whisper 1.2.1 passes, so every faster-whisper transcription failed. Fixed by capping
  `av<19` in the `faster-whisper` extra
- 4.4 smoke run (2026-10-05): `/health` ok, `/backends` lists both, a 20 s clip transcribed with
  `{"model": "tiny"}` (3 segments), its evaluation scored, container `healthy`, running as `app`.
  ElevenLabs was not called

### Verification

- [x] `docker compose up` serves the API; a job submitted with `curl` succeeds end to end

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
