# Conductor Hub

## Project: Audio Transcriber

A modular audio transcription pipeline with swappable model backends.

## Quick Links

### Core Documents

| Document                                      | Description                | Status   |
| --------------------------------------------- | -------------------------- | -------- |
| [Product Vision](./product.md)                | Product overview and goals | Complete |
| [Product Guidelines](./product-guidelines.md) | Voice, tone, and standards | Complete |
| [Tech Stack](./tech-stack.md)                 | Technology decisions       | Complete |
| [Workflow](./workflow.md)                     | Development process        | Complete |

### Track Management

| Document                        | Description            |
| ------------------------------- | ---------------------- |
| [Track Registry](./tracks.md)   | All development tracks |
| [Active Tracks](#active-tracks) | Currently in progress  |

### Style Guides

| Guide                                    | Language/Domain      |
| ---------------------------------------- | -------------------- |
| [General](./code_styleguides/general.md) | Universal principles |
| [Python](./code_styleguides/python.md)   | Python standards     |

## Active Tracks

<!-- Auto-populated by /conductor:new-track -->

| Track | Status | Priority | Spec | Plan |
| ----- | ------ | -------- | ---- | ---- |
| [GPU Docker Image](./tracks/gpu-docker_20261006/index.md) | Pending | Medium | [spec](./tracks/gpu-docker_20261006/spec.md) | [plan](./tracks/gpu-docker_20261006/plan.md) |
| [HTTP API Service](./tracks/http-api_20261005/index.md) | Complete | Medium | [spec](./tracks/http-api_20261005/spec.md) | [plan](./tracks/http-api_20261005/plan.md) |
| [Core Pipeline and Backend Interface](./tracks/_archive/core-pipeline_20260922/index.md) | Complete | Critical | [spec](./tracks/_archive/core-pipeline_20260922/spec.md) | [plan](./tracks/_archive/core-pipeline_20260922/plan.md) |
| [Code-Switching Language Detection and Speaker Diarization](./tracks/_archive/codeswitch-diarization_20260922/index.md) | Complete | High | [spec](./tracks/_archive/codeswitch-diarization_20260922/spec.md) | [plan](./tracks/_archive/codeswitch-diarization_20260922/plan.md) |
| [Default Whisper Model to large-v3](./tracks/_archive/default-large-v3_20260923/index.md) | Complete | Low | [spec](./tracks/_archive/default-large-v3_20260923/spec.md) | [plan](./tracks/_archive/default-large-v3_20260923/plan.md) |
| [Per-Word Speaker Assignment](./tracks/_archive/word-speakers_20260924/index.md) | Complete | Medium | [spec](./tracks/_archive/word-speakers_20260924/spec.md) | [plan](./tracks/_archive/word-speakers_20260924/plan.md) |
| [ElevenLabs Scribe v2 Backend](./tracks/_archive/elevenlabs-scribe_20260924/index.md) | Complete | Medium | [spec](./tracks/_archive/elevenlabs-scribe_20260924/spec.md) | [plan](./tracks/_archive/elevenlabs-scribe_20260924/plan.md) |
| [Standalone Backend Scripts and Pipeline Overview](./tracks/_archive/standalone-snapshot_20260925/index.md) | Complete | Medium | [spec](./tracks/_archive/standalone-snapshot_20260925/spec.md) | [plan](./tracks/_archive/standalone-snapshot_20260925/plan.md) |
| [Evaluate Against a Custom Ground Truth](./tracks/_archive/custom-ground-truth_20261002/index.md) | Complete | Medium | [spec](./tracks/_archive/custom-ground-truth_20261002/spec.md) | [plan](./tracks/_archive/custom-ground-truth_20261002/plan.md) |

## Recent Activity

| Date       | Track | Action                     |
| ---------- | ----- | -------------------------- |
| 2026-09-22 | —     | Conductor setup completed  |
| 2026-09-22 | —     | Git repository initialized |
| 2026-09-22 | core-pipeline_20260922 | Track created (5 phases, 32 tasks) |
| 2026-09-22 | core-pipeline_20260922 | Track complete (5 phases, 34 tasks, all gates passed) |
| 2026-09-22 | core-pipeline_20260922 | Merged into `main` |
| 2026-09-22 | codeswitch-diarization_20260922 | Track created (5 phases, 31 tasks) |
| 2026-09-23 | default-large-v3_20260923 | Track created (2 phases, 5 tasks) |
| 2026-09-23 | default-large-v3_20260923 | Track complete (2 phases, 5 tasks, all gates passed) |
| 2026-09-24 | word-speakers_20260924 | Track created (4 phases, 21 tasks) |
| 2026-09-24 | word-speakers_20260924 | Track complete (4 phases, 21 tasks, all gates passed); default `speaker_assignment` now `word` |
| 2026-09-24 | elevenlabs-scribe_20260924 | Track created (4 phases, 19 tasks) |
| 2026-09-24 | elevenlabs-scribe_20260924 | Track complete (4 phases, 20 tasks, all gates passed; evaluation scored 3 of 5 recordings, free-tier quota) |
| 2026-09-25 | standalone-snapshot_20260925 | Track created (4 phases, 18 tasks) |
| 2026-09-25 | standalone-snapshot_20260925 | Track complete (4 phases, 18 tasks, all gates passed); merged into `main` |
| 2026-10-02 | custom-ground-truth_20261002 | Track created (4 phases, 19 tasks) |
| 2026-10-02 | custom-ground-truth_20261002 | Track complete (4 phases, 19 tasks, all gates passed); merged into `main` |
| 2026-10-05 | http-api_20261005 | Track created (5 phases, 22 tasks) |
| 2026-10-06 | http-api_20261005 | Track complete (5 phases, 22 tasks, all gates passed); merged into `main` |
| 2026-10-06 | gpu-docker_20261006 | Track created (4 phases, 15 tasks) |

## Project Status

**Current Phase:** Core pipeline merged; `codeswitch-diarization_20260922` ready to start
**Overall Progress:** Core pipeline, backend interface, and one backend shipped

### Milestone Tracker

| Milestone                         | Target Date | Status  |
| --------------------------------- | ----------- | ------- |
| Core pipeline + backend interface | 2026-09-22  | Complete |
| First model backend + docs        | 2026-09-22  | Complete |
| Spanish/English code-switching    | TBD         | Pending |
| Docker packaging                  | TBD         | Pending |

## Project Context at a Glance

- **Language:** Python 3.12 (`.venv`, 3.12.3)
- **Shape:** Library + CLI. No web framework, no database — results are files on disk.
- **Deployment:** Self-hosted Docker
- **Lint/Format:** Ruff (both)
- **Testing:** Flexible TDD — required for the backend interface, output normalization, and CLI contract
- **Commits:** Conventional Commits
- **Review:** Self-review OK
- **Verification:** At track completion only (exception: backend-interface / schema / manifest changes need approval)

## Getting Started

1. Review [Product Vision](./product.md) for project context
2. Check [Tech Stack](./tech-stack.md) for technology decisions
3. Read [Workflow](./workflow.md) for the development process
4. Find your track in the [Track Registry](./tracks.md)
5. Follow the track spec and plan

## Commands Reference

```bash
# Setup
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

# Run
.venv/bin/python -m audio_transcriber --input ./audio --backend <name> --out ./out

# Test
pytest

# Validate
ruff check . && ruff format --check . && pytest

# Build
docker build -t audio-transcriber .
```

> Commands reflect the intended layout; they become real as the first track lands.

---

**Last Updated:** 2026-09-22
**Maintained By:** Internal team
