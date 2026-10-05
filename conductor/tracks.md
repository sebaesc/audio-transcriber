# Track Registry

This file maintains the registry of all development tracks for Audio Transcriber. Each track
represents a distinct body of work with its own spec and implementation plan.

## Status Legend

| Symbol | Status      | Description               |
| ------ | ----------- | ------------------------- |
| `[ ]`  | Pending     | Not yet started           |
| `[~]`  | In Progress | Currently being worked on |
| `[x]`  | Completed   | Finished and verified     |

## Active Tracks

| Status | Track ID | Title | Created | Updated |
| ------ | -------- | ----- | ------- | ------- |
| [ ] | [http-api_20261005](./tracks/http-api_20261005/index.md) | HTTP API Service | 2026-10-05 | 2026-10-05 |

<!-- Tracks registered by /conductor:new-track -->

---

## Completed Tracks

| Status | Track ID | Title | Created | Completed |
| ------ | -------- | ----- | ------- | --------- |

<!-- Move completed tracks here -->

---

## Backlog

Ideas deferred from tracks; promote one with `/conductor:new-track`.

| Item | Source Track | Added | Notes |
| ---- | ------------ | ----- | ----- |
| Persistent job storage for the HTTP API | http-api_20261005 | 2026-10-05 | Jobs are in-memory and lost on restart; implement the `JobStore` protocol over SQLite or similar |

---

## Archived Tracks

| Track ID | Type | Reason | Archived | Folder |
| -------- | ---- | ------ | -------- | ------ |
| core-pipeline_20260922 | feature | Completed | 2026-10-05 | [./tracks/_archive/core-pipeline_20260922/](./tracks/_archive/core-pipeline_20260922/) |
| codeswitch-diarization_20260922 | feature | Completed | 2026-10-05 | [./tracks/_archive/codeswitch-diarization_20260922/](./tracks/_archive/codeswitch-diarization_20260922/) |
| default-large-v3_20260923 | chore | Completed | 2026-10-05 | [./tracks/_archive/default-large-v3_20260923/](./tracks/_archive/default-large-v3_20260923/) |
| word-speakers_20260924 | feature | Completed | 2026-10-05 | [./tracks/_archive/word-speakers_20260924/](./tracks/_archive/word-speakers_20260924/) |
| elevenlabs-scribe_20260924 | feature | Completed | 2026-10-05 | [./tracks/_archive/elevenlabs-scribe_20260924/](./tracks/_archive/elevenlabs-scribe_20260924/) |
| standalone-snapshot_20260925 | chore | Completed | 2026-10-05 | [./tracks/_archive/standalone-snapshot_20260925/](./tracks/_archive/standalone-snapshot_20260925/) |
| custom-ground-truth_20261002 | feature | Completed | 2026-10-05 | [./tracks/_archive/custom-ground-truth_20261002/](./tracks/_archive/custom-ground-truth_20261002/) |

---

## Track Creation Checklist

When creating a new track:

1. [ ] Add entry to this registry
2. [ ] Create track folder: `./tracks/{{track-id}}/`
3. [ ] Create spec.md from template
4. [ ] Create plan.md from template
5. [ ] Create metadata.json from template
6. [ ] Update index.md with new track reference

## Notes

- Track IDs are lowercase with hyphens (e.g., `core-pipeline`, `whisper-backend`)
- Keep descriptions to one line
- Prioritize tracks as: critical, high, medium, low
- A track that adds a model backend is not complete without that backend's doc page
