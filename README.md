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

See [`docs/backends/`](docs/backends/) for what each backend needs.

## Usage

```bash
.venv/bin/python -m audio_transcriber --input ./audio --backend faster-whisper --out ./out
```

Or, once installed, as `audio-transcriber`.

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| `--input`, `-i` | — | Directory of audio files, or a single file |
| `--backend`, `-b` | — | Backend to transcribe with |
| `--out`, `-o` | `out` | Directory to write the run into |
| `--glob` | `**/*` | Pattern for finding inputs, relative to `--input` |
| `--option KEY=VALUE` | — | Backend option; repeatable |
| `--quiet`, `-q` | off | Report only the final summary |
| `--list-backends` | — | List installed backends and exit |

Backend options are passed through and recorded in the manifest. Values are read as JSON, so
types are preserved:

```bash
.venv/bin/python -m audio_transcriber -i ./audio -b faster-whisper \
  --option model=large-v3 --option beam_size=1
```

### Exit codes

| Code | Meaning |
| ---- | ------- |
| `0` | Every file transcribed |
| `1` | Partial failure — some files failed, the rest succeeded |
| `2` | Nothing succeeded |
| `3` | The run could not start (bad config, missing input, unmet requirements) |
| `130` | Interrupted; partial results are kept |

Progress and errors go to stderr, transcript data to files, so piping works.

## Output

A run writes to `out/<run-id>/`, where `run-id` is a UTC timestamp. A prior run is never
overwritten.

```
out/2026-09-22T15-53-21Z/
├── interview.json      # one per input
├── standup.json
├── standup-2.json      # inputs sharing a stem stay distinct
├── manifest.json
└── errors.json
```

A **transcript file** is a bare JSON array of segments. There is no wrapper: read it and iterate.

```json
[
  {
    "start_time": 0.18,
    "end_time": 10.42,
    "speaker": null,
    "spoken_language": "en",
    "text": "he hoped there would be stew for dinner"
  }
]
```

Every object has exactly those five keys. `speaker` stays `null` until a backend diarizes.

**`manifest.json`** records what produced the run — backend, model version, resolved config, and
a checksum per input — so any transcript traces back to what made it. Per-file metadata
(duration, language, segment count) lives on each input's entry, and a consumer that only wants
segments can ignore all of it.

**`errors.json`** lists files that failed. It is written even when nothing failed, so consumers
never have to handle a missing file. A per-file failure never aborts the batch, and partial
results are kept.

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

Every backend must pass the shared contract suite:

```bash
.venv/bin/pytest -k conformance
```

To measure whether a change improves transcription quality, run the evaluation harness against
the Spanish-English reference corpus — see [`docs/evaluation.md`](docs/evaluation.md).

## Adding a backend

Implement one interface, register it by name, and write its doc page. See
[`docs/backend-interface.md`](docs/backend-interface.md) for the contract, and
[`docs/backends/faster-whisper.md`](docs/backends/faster-whisper.md) for a worked example.

## Documentation

- [Backend interface contract](docs/backend-interface.md)
- [Per-backend documentation](docs/backends/)
- [Evaluation harness](docs/evaluation.md)
- Project context, tracks, and workflow: [`conductor/`](conductor/)
