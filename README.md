# Audio Transcriber

A modular audio transcription pipeline with swappable model backends.

The transcription model is a plug-in, not a dependency. Adding, swapping, or comparing engines
means implementing one interface — not rewriting the pipeline. Every backend normalizes its own
output to a single transcript schema, so downstream consumers do not break on a swap.

## Requirements

- Python 3.12
- ffmpeg 6.0 or newer on `PATH` (audio decode and resample)
- The dependencies of whichever backend you select, installed as an optional extra

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Backends are optional extras, installed one at a time. Installing the pipeline does not pull in
every engine:

```bash
.venv/bin/pip install -e ".[dev,faster-whisper]"
```

## Usage

```bash
.venv/bin/python -m audio_transcriber --input ./audio --backend <name> --out ./out
```

A run writes one transcript JSON per input file to `out/<run-id>/`, alongside a `manifest.json`
recording the backend, model version, effective config, and input checksums, and an `errors.json`
listing any files that failed.

> The CLI lands in Phase 5 of the `core-pipeline_20260922` track. Until then this command reports
> that it is not implemented.

## Development

```bash
.venv/bin/ruff format .
.venv/bin/ruff check --fix .
.venv/bin/pytest
```

Full validation, matching the quality gates in `conductor/workflow.md`:

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/pytest
```

## Documentation

- Backend interface contract: `docs/backend-interface.md` _(Phase 2)_
- Per-backend documentation: `docs/backends/` _(Phase 3)_
- Project context, tracks, and workflow: `conductor/`
