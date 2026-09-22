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
| [Core Pipeline and Backend Interface](./tracks/core-pipeline_20260922/index.md) | Complete | Critical | [spec](./tracks/core-pipeline_20260922/spec.md) | [plan](./tracks/core-pipeline_20260922/plan.md) |

## Recent Activity

| Date       | Track | Action                     |
| ---------- | ----- | -------------------------- |
| 2026-09-22 | —     | Conductor setup completed  |
| 2026-09-22 | —     | Git repository initialized |
| 2026-09-22 | core-pipeline_20260922 | Track created (5 phases, 32 tasks) |
| 2026-09-22 | core-pipeline_20260922 | Track complete (5 phases, 34 tasks, all gates passed) |

## Project Status

**Current Phase:** Core pipeline complete; ready for the next track
**Overall Progress:** Core pipeline, backend interface, and one backend shipped

### Milestone Tracker

| Milestone                         | Target Date | Status  |
| --------------------------------- | ----------- | ------- |
| Core pipeline + backend interface | 2026-09-22  | Complete |
| First model backend + docs        | 2026-09-22  | Complete |
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
