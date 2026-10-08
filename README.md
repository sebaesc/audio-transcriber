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

| Backend | Extra | Runs | Needs |
| ------- | ----- | ---- | ----- |
| [`faster-whisper`](docs/backends/faster-whisper.md) | `faster-whisper` | Locally, CPU or GPU | Nothing; `HF_TOKEN` for diarization |
| [`elevenlabs-scribe`](docs/backends/elevenlabs-scribe.md) | `elevenlabs` | Hosted by ElevenLabs; audio is uploaded, $0.22/h | `ELEVENLABS_API_KEY` |

Credentials go in a `.env` beside the project; copy [`.env.example`](.env.example) to start.
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

`audio-transcriber serve` starts the HTTP API instead; see [HTTP API](#http-api).

For bilingual audio, detect language per speech chunk rather than once per file, and label
speakers. Both are opt-in; see the
[`faster-whisper` doc page](docs/backends/faster-whisper.md) for what each costs and needs:

```bash
.venv/bin/python -m audio_transcriber -i ./audio -b faster-whisper \
  --option model=large-v3 \
  --option language_detection=chunked --option 'languages=["en","es"]' \
  --option diarization=true
```

Speakers are assigned per word, and a segment is split where the speaker changes. Add
`--option speaker_assignment=segment` to label each segment whole instead.

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

## HTTP API

The same pipeline as a service: upload audio, get a job, poll it for the transcript, and
optionally score it against a ground truth. Install the `api` extra beside a backend, plus `eval`
to score:

```bash
.venv/bin/pip install -e ".[api,eval,faster-whisper]"
.venv/bin/audio-transcriber serve                  # http://127.0.0.1:8000, docs at /docs
.venv/bin/audio-transcriber serve --host 0.0.0.0 --port 8000 --workers-concurrency 2
```

| Method and path | Does | Statuses |
| --------------- | ---- | -------- |
| `POST /transcriptions` | Queue a transcription. Multipart: `file`, `backend`, `options` (a JSON object, optional) | 202, 400, 413, 422 |
| `GET /transcriptions/{id}` | The job: `status`, and `transcript` or `error` once finished | 200, 404 |
| `POST /transcriptions/{id}/evaluation` | Score a succeeded job. Multipart: `ground_truth` | 200, 404, 409, 422, 501 |
| `GET /backends` | Installed backends, and why the others are unavailable | 200 |
| `GET /health` | `{"status": "ok"}` | 200 |

Submit, poll, then score:

```bash
curl -i -X POST localhost:8000/transcriptions \
  -F file=@meeting.mp4 -F backend=faster-whisper -F 'options={"model": "small"}'
# HTTP/1.1 202 Accepted
# location: /transcriptions/3f2c…
# {"id": "3f2c…", "status": "queued"}

curl localhost:8000/transcriptions/3f2c…
# {"id": "3f2c…", "status": "succeeded", "transcript": [ …segments… ], "error": null, …}

curl -X POST localhost:8000/transcriptions/3f2c…/evaluation -F ground_truth=@meeting.json
# The report the evaluation CLI writes: WER, retention, coverage, speakers, languages
```

**Postman.** [`docs/postman/audio-transcriber.postman_collection.json`](docs/postman/audio-transcriber.postman_collection.json)
holds the same requests, plus the refused ones and their errors. Import it, set the `baseUrl`,
`backend` and `options` collection variables, and pick the files in each request's Body tab.
`Submit transcription` saves the job id in `jobId`, so the poll and evaluate requests use that job.
Send `Get transcription` again until the job finishes.

**Jobs are asynchronous.** A job moves `queued` → `running` → `succeeded` or `failed`, and never
back. On success, `transcript` is the same JSON array the CLI writes. On failure, `error` says why:
`{"type", "message", "remedy"}`. A request no job could run is refused before anything is queued:
an unknown or uninstalled backend, or a bad option, is a 400. Anything that can only fail while
transcribing (undecodable audio, a missing API key, an engine error) is reported on the job.
Every error response has the same shape: `{"error": {"type", "message", "remedy"}}`.

**Scoring never transcribes again.** The evaluation endpoint reads the transcript the job already
wrote, so re-scoring a hosted backend's output spends no credits. A job that is unfinished or failed
is a 409. The ground truth format is the one in [`docs/evaluation.md`](docs/evaluation.md).

**Settings** come from the environment:

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `AUDIO_TRANSCRIBER_CONCURRENCY` | `1` | Jobs run at once. Keep 1 for a local model; `--workers-concurrency` overrides it |
| `AUDIO_TRANSCRIBER_MAX_UPLOAD_MB` | `1024` | Larger uploads are refused with 413 |
| `AUDIO_TRANSCRIBER_WORK_DIR` | `<tmp>/audio-transcriber-api` | Uploads while their job runs, and each job's run directory |

**Limitations.** Jobs live in memory: a restart loses them, though their run directories stay on
disk. There is no authentication, so keep the service on a trusted network.

## Docker

The image holds the pipeline, ffmpeg, and the extras named by the `EXTRAS` build argument
(`api,eval,faster-whisper,elevenlabs` by default). It runs `audio-transcriber serve` on port 8000
as a non-root user, CPU only.

```bash
cp .env.example .env            # fill in ELEVENLABS_API_KEY / HF_TOKEN if you use them
docker compose up --build       # http://localhost:8000
```

- **Credentials** are read from `.env` at run time by compose and are never copied into the image.
- **Models** download on first use into the `model-cache` volume, so a restart does not download
  `large-v3` (about 3 GB) again. For a quick check, pass `-F 'options={"model": "tiny"}'`.
- **A leaner image**: `EXTRAS=api,eval,elevenlabs docker compose build` leaves out the local model.
- `PORT`, `AUDIO_TRANSCRIBER_CONCURRENCY` and `AUDIO_TRANSCRIBER_MAX_UPLOAD_MB` can be set in the
  shell or `.env` when running compose.

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

Hosted backends run against a fake client by default, so a key in `.env` never turns a plain
`pytest` into paid API calls. `AUDIO_TRANSCRIBER_LIVE_TESTS=1` runs them against the real API too.

To measure whether a change improves transcription quality, run the evaluation harness against
the Spanish-English reference corpus, or against a ground truth you write for your own recordings
(by running a backend, or by scoring a transcript the pipeline already made) — see
[`docs/evaluation.md`](docs/evaluation.md). `--table` puts every report side by side, model by
model and recording by recording, and `--metrics` explains each metric.

### Evaluation data

The reference corpus is the Bangor Miami Spanish-English corpus (Deuchar & Parafita Couto, 2011):
naturalistic English-Spanish conversations. It comes from two places:

- **Audio and reference transcripts: Hugging Face.**
  [`BrunoHays/Bangor-Miami-Spanish-English-Corpus`](https://huggingface.co/datasets/BrunoHays/Bangor-Miami-Spanish-English-Corpus)
  (CC BY-SA 3.0). Nothing to do by hand: the 116 MB parquet downloads on first use into
  `~/.cache/audio-transcriber/bangor-miami` (override with `AUDIO_TRANSCRIBER_EVAL_CACHE`).
- **Speaker and per-segment language annotations: TalkBank.** Only needed for
  `evaluation --talkbank`, and it cannot be automated. Create a free account at
  <https://talkbank.org/biling/>, then while logged in download
  <https://talkbank.org/data/biling/Bangor/Miami?f=zip> and save it as
  `~/.cache/audio-transcriber/talkbank/Miami.zip` (or set `AUDIO_TRANSCRIBER_TALKBANK_ARCHIVE`).
  The data is CC BY-NC-SA 3.0 and must stay on your machine, so never commit it.

Neither is ever committed to the repo. Details and the required citation are in
[`docs/evaluation.md`](docs/evaluation.md).

## Adding a backend

Implement one interface, register it by name, and write its doc page. See
[`docs/backend-interface.md`](docs/backend-interface.md) for the contract, and
[`docs/backends/faster-whisper.md`](docs/backends/faster-whisper.md) for a worked example.

## Documentation

- [Backend interface contract](docs/backend-interface.md)
- [Per-backend documentation](docs/backends/)
- [Evaluation harness](docs/evaluation.md)
- Project context, tracks, and workflow: [`conductor/`](conductor/)
