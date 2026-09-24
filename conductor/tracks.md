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

<!-- Tracks registered by /conductor:new-track -->

---

## Completed Tracks

| Status | Track ID | Title | Created | Completed |
| ------ | -------- | ----- | ------- | --------- |
| `[x]` | [core-pipeline_20260922](./tracks/core-pipeline_20260922/index.md) | Core Pipeline and Backend Interface | 2026-09-22 | 2026-09-22 |
| `[x]` | [codeswitch-diarization_20260922](./tracks/codeswitch-diarization_20260922/index.md) | Code-Switching Language Detection and Speaker Diarization | 2026-09-22 | 2026-09-23 |

<!-- Move completed tracks here -->

---

## Archived Tracks

| Track ID | Type | Reason | Archived | Folder |
| -------- | ---- | ------ | -------- | ------ |

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
