# Technology Stack

## Frontend

Not applicable — CLI only. No frontend framework, state management, or styling layer.

If a UI is ever added, it is a separate consumer of the library and does not change the
pipeline contract.

## Backend

### Language

**Choice:** Python
**Version:** 3.12 (detected: 3.12.3 in `.venv`)

**Rationale:**
The transcription ecosystem (Whisper, faster-whisper, torch, audio tooling) is Python-native.
A venv already exists at `.venv` with Python 3.12.3.

### Framework

**Choice:** None — Python library with a CLI entry point
**Version:** n/a

**Rationale:**
The deliverable is a pipeline, not a service. A library plus CLI keeps the core importable
and testable; an HTTP layer can be added later as a consumer without changing the pipeline.

### Database

#### Primary Database

**Choice:** None — files on disk
**Version:** n/a

**Rationale:**
Transcripts and run metadata are written as JSON/text files. This keeps runs portable and
inspectable, and suits a single-team CLI workflow. A database would be added only if multiple
consumers need to query results.

#### Secondary Database (if applicable)

**Choice:** None
**Purpose:** n/a

### Additional Backend Libraries

Pinned by the `core-pipeline_20260922` track.

**Core dependencies** — the pipeline and its CLI, nothing else:

| Library | Purpose                       | Version          |
| ------- | ----------------------------- | ---------------- |
| Typer   | CLI entry point               | `>=0.12,<0.20`   |

**Development dependencies:**

| Library | Purpose            | Version         |
| ------- | ------------------ | --------------- |
| pytest  | Test runner        | `>=8.0,<9.0`    |
| Ruff    | Lint and format    | `>=0.6,<0.15`   |

**Backend extras** — one per backend, installed individually:

| Extra            | Library          | Version        |
| ---------------- | ---------------- | -------------- |
| `faster-whisper` | faster-whisper   | `>=1.0,<2.0`   |

Backend-specific dependencies are installed as **optional extras**, never as required
dependencies of the core package — installing the pipeline must not pull in every engine.

Audio decode and resample is **ffmpeg invoked as a subprocess**, not a Python binding: it is the
same tool the docs tell users to install, and its absence is reported with the command that fixes
it. Decoding happens once, in the pipeline, so no backend handles container formats.

GPU inference needs the CUDA runtime libraries, installed as wheels
(`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`) rather than system-wide. They are not declared in
`pyproject.toml`: they are optional, large, and only relevant to backends that use CUDA.

## Infrastructure

### Hosting

**Provider:** Self-hosted
**Environment:** Docker

**Services Used:**

- Docker image containing the pipeline and its system dependencies (ffmpeg)
- Optional GPU host for local model backends (CUDA base image variant)
- Host filesystem / mounted volumes for audio input and transcript output

### CI/CD

**Platform:** TBD

**Pipeline Stages:**

1. Lint and format check (`ruff check`, `ruff format --check`)
2. Tests (`pytest`)
3. Build Docker image
4. Publish image

### Monitoring

**APM:** None — batch CLI tool
**Logging:** Python `logging` to stderr; per-run manifest and `errors.json` on disk
**Alerting:** None

### Additional Infrastructure

| Service | Purpose                                      | Provider   |
| ------- | -------------------------------------------- | ---------- |
| ffmpeg  | Audio decode/resample (system dependency)     | OS package |
| GPU host | Local model inference (optional)             | Self-hosted |

## Development Tools

### Package Manager

**Choice:** pip (venv at `.venv`)
**Version:** pip 26.2.1

### Testing

| Type        | Tool   | Coverage Target                       |
| ----------- | ------ | ------------------------------------- |
| Unit        | pytest | Complex logic (interface, normalization) |
| Integration | pytest | Backend conformance against fixtures  |
| E2E         | pytest | CLI run over a small sample set       |

Coverage is not gated at a fixed percentage — see `workflow.md`. Tests are expected on the
backend interface, output normalization, and the CLI contract.

### Linting & Formatting

**Linter:** Ruff
**Formatter:** Ruff (`ruff format`)
**Config:** `[tool.ruff]` in `pyproject.toml`

Ruff handles both lint and format; Black and flake8 are deliberately not used.

### Additional Dev Tools

| Tool    | Purpose                                     |
| ------- | ------------------------------------------- |
| Docker  | Reproducible runtime, incl. system deps     |
| ffmpeg  | Local audio conversion during development   |

## Decision Log

### Model backends are plug-ins behind a stable interface

**Date:** 2026-09-22
**Status:** Accepted

**Context:**
Transcription tools hard-wire one engine, so changing models means rewriting the pipeline.

**Decision:**
Define a backend interface. Every engine implements it and normalizes its own output to the
shared transcript schema. Backends are selected by name in config; pipeline code contains no
engine-specific branches.

**Consequences:**

- Adding an engine is one module plus one doc page
- Pipeline features are limited to what the interface can express across backends
- Engine-specific capabilities must be surfaced through explicit capability flags, not ad-hoc branches

---

### No database — results are files on disk

**Date:** 2026-09-22
**Status:** Accepted

**Context:**
Single internal team, CLI workflow, no multi-consumer query needs today.

**Decision:**
Write transcripts and run metadata as JSON/text files under a per-run output directory.

**Consequences:**

- Runs are portable and inspectable with ordinary tools
- No query layer; finding results across runs means filesystem search
- Revisit if concurrent writers or cross-run queries appear

---

### Ruff as the single lint/format tool

**Date:** 2026-09-22
**Status:** Accepted

**Context:**
Greenfield project with no existing lint configuration.

**Decision:**
Use Ruff for both linting and formatting, configured in `pyproject.toml`.

**Consequences:**

- One tool, one config, fast checks
- Ruff format is Black-compatible, so migration stays open
- Type checking is not covered; add mypy later if type errors become a real problem

### Transcript files are bare JSON arrays

**Date:** 2026-09-22
**Status:** Accepted

**Context:**
Consumers need the segments; only reproducibility needs the metadata. Wrapping segments in a
metadata object makes every consumer unwrap before doing anything.

**Decision:**
A transcript file is a bare JSON array of segment objects with exactly five keys: `start_time`,
`end_time`, `speaker`, `spoken_language`, `text`. Per-file metadata moves to the run manifest's
input record as an optional sidecar.

**Consequences:**

- A consumer reads the file and iterates it, with no wrapper and nothing to ignore
- A transcript file cannot carry its own schema version, so the manifest records it
- Tracing a transcript to its input requires the manifest, not the transcript alone

---

### Schema conformance is enforced at runtime, not only in tests

**Date:** 2026-09-22
**Status:** Accepted

**Context:**
The conformance suite proves a backend *can* honor the schema. It cannot prove every transcript
does. Whisper hallucinates into the silence it pads short audio with, returning segments past the
end of the recording — output the conformance suite would reject, written to disk as if valid.

**Decision:**
The pipeline validates every transcript against the shared assertions before writing it. A
violation becomes a reported per-file failure.

**Consequences:**

- A backend bug produces a visible failure rather than plausible-looking bad data
- Validation cost is paid on every file, though it is pure in-memory checks
- A backend with a systematic bug fails every file, which is the intended signal

---

## Version Compatibility Matrix

| Component      | Min Version | Max Version | Notes                                          |
| -------------- | ----------- | ----------- | ---------------------------------------------- |
| Python         | 3.12        | 3.12.x      | Matches `.venv` (3.12.3)                       |
| ffmpeg         | 6.0         | —           | System dependency; verified with 6.1.1         |
| Ruff           | 0.6         | <0.15       | Pinned in dev dependencies                     |
| Typer          | 0.12        | <0.20       | Core dependency                                |
| pytest         | 8.0         | <9.0        | Dev dependency                                 |
| faster-whisper | 1.0         | <2.0        | Optional extra; verified with 1.2.1            |
