# Backend: `elevenlabs-scribe`

ElevenLabs' hosted [Scribe v2](https://elevenlabs.io/docs/overview/capabilities/speech-to-text)
speech-to-text model, over its batch API. This is the first hosted backend: **audio leaves the
machine**, the service bills per audio hour, and ElevenLabs stores what it receives (see
[Data retention](#data-retention)).

| | |
| --- | --- |
| **Name** | `elevenlabs-scribe` |
| **Extra** | `.[elevenlabs]` |
| **Credentials** | `ELEVENLABS_API_KEY` |
| **Network** | Every file is uploaded to `api.elevenlabs.io` |
| **Cost** | $0.22 per audio hour; diarization adds nothing |
| **Diarization** | Opt-in (`diarization=true`), done by Scribe itself, up to 32 speakers |
| **Language detection** | Per file only, never per segment |
| **Word timestamps** | Yes (not surfaced by the pipeline yet) |
| **Translation** | No |

## Install

```bash
.venv/bin/pip install -e ".[elevenlabs]"
```

This installs the official `elevenlabs` Python SDK (2.x), which brings `httpx`, `pydantic`,
`requests` and `websockets`. It is not part of the core install. Without it the backend is not
registered, and `--backend elevenlabs-scribe` reports an unknown backend. `--list-backends` shows
what is installed.

### API key

1. Create a key at <https://elevenlabs.io/app/settings/api-keys>. It needs speech-to-text access.
2. Put it in a `.env` beside the project, copying `.env.example`:

   ```bash
   cp .env.example .env   # then fill in ELEVENLABS_API_KEY=
   ```

   Or export `ELEVENLABS_API_KEY`, or pass `--option api_key=...`. That is also the lookup order:
   the option, then the environment, then `.env`. An exported variable beats the file.

A missing key stops the run before any file is processed. The key is not tried against the API up
front. A wrong key fails every file with HTTP 401 and the message says to check it. The key is
redacted from the run manifest and from every error message.

## Usage

```bash
.venv/bin/python -m audio_transcriber --input ./audio --backend elevenlabs-scribe --out ./out
```

With speakers, at most two of them:

```bash
.venv/bin/python -m audio_transcriber -i ./audio -b elevenlabs-scribe --out ./out \
  --option diarization=true --option max_speakers=2
```

Programmatically:

```python
from audio_transcriber import registry

backend = registry.create("elevenlabs-scribe", diarization=True)
backend.check_requirements()
transcript = backend.transcribe(decoded_wav_path, "audio/meeting.m4a")
```

## Configuration

Pass options with `--option key=value` on the CLI, or as keywords to `registry.create`. Every
option is recorded in the run manifest, and an option the backend does not recognize is
**rejected**, not ignored. So is one the API would reject, or would read differently from the
same-named faster-whisper option.

| Option | Default | Meaning |
| --- | --- | --- |
| `api_key` | from environment / `.env` | ElevenLabs API key. Redacted from the manifest. |
| `language` | `null` (detect) | Force a language, ISO 639-1 (`es`) or 639-3 (`spa`). |
| `diarization` | `false` | Label segments by speaker. |
| `max_speakers` | `null` (up to 32) | The **most** speakers Scribe may assign, 1–32. Needs `diarization=true`. |
| `diarization_threshold` | `null` (API picks, about 0.22) | Higher merges more voices into one speaker. Needs `diarization=true` and no `max_speakers`. |
| `seed` | `0` | Sent for best-effort determinism; see [Known limits](#known-limits). `null` omits it. |
| `temperature` | `0.0` | 0.0–2.0. Higher is more varied and less repeatable. `null` omits it. |
| `timeout` | `900` | Seconds to wait on any one network operation. In practice, the wait for the transcript after the upload. |

### Speaker counts differ from faster-whisper

faster-whisper's `num_speakers` is an **exact** count, handed to pyannote. Scribe's own
`num_speakers` parameter is an **upper bound**. Passing the same option to both backends would
mean different things on each, so this backend accepts only `max_speakers`. It rejects
`num_speakers` and `min_speakers` rather than reinterpreting them.

### What is sent

Every request uses `model_id=scribe_v2`, `timestamps_granularity=word` and `tag_audio_events=false`,
plus `diarize` and whichever options above are set. The upload is the pipeline's decoded mono
16 kHz WAV, about 115 MB per hour of audio. Multichannel mode, webhooks, keyterms, entity detection
and speaker roles are not used. Keyterms, entity detection and speaker roles carry surcharges.

## How the transcript is built

Scribe returns a list of words, spacing tokens and audio events, not segments. The backend builds
segments from them:

- A segment ends at the end of a sentence (`.`, `?`, `!`, `…`), at a pause over 1 s, or, with
  diarization on, at a change of speaker.
- Text keeps the spacing Scribe returned, so languages written without spaces are not given any.
- Audio events are dropped. They are not requested, and a tag is not speech.
- Overlapping speech is made non-overlapping: a segment that starts before the previous one
  ended is moved to where it ended.
- Language codes come back as ISO 639-3 (`eng`, `spa`) and are mapped to the 639-1 codes
  faster-whisper uses (`en`, `es`). This makes transcripts from both backends comparable. Eight
  languages with no 639-1 code pass through unchanged: `ast`, `ceb`, `fil`, `kea`, `luo`, `nso`,
  `umb`, `yue`.
- Speaker labels are Scribe's (`speaker_0`, `speaker_1`, …). Like pyannote's, they are stable
  within one file and mean nothing across files.

## Errors and retries

The SDK retries HTTP 408, 409, 429 and 5xx twice, with backoff that honors `Retry-After`. After
that, each failure below becomes a failure **for that file only**, and the run moves on to the
next file:

| What happened | Message says |
| --- | --- |
| 401, `quota_exceeded` | The account is out of credits, with the API's count of what remains. ElevenLabs sends this as a 401, not a 429 |
| 401 / 403 | The key or its permissions were rejected; check `ELEVENLABS_API_KEY` |
| 429 | Rate limited or over the plan's concurrency, even after retries |
| 5xx | Server error, even after retries |
| 400 / 422 | The request was rejected, with the API's detail (truncated) |
| Timeout | No response within `timeout`; raise it with `--option timeout=SECONDS` |
| Network | Could not reach the API |

Timeouts and connection errors are not retried.

## Limits

- **File size:** up to 5 GB (raised from 3 GB on 2026-06-01). Some ElevenLabs pages still say
  3 GB.
- **Duration:** up to 10 hours per file; at least 100 ms.
- **Speakers:** up to 32.
- **Concurrency:** set by the ElevenLabs plan. The pipeline sends one file at a time.

## Cost

Scribe v2 costs **$0.22 per audio hour**, billed per minute. Diarization costs nothing extra. A full
evaluation over the Bangor Miami corpus, about 2.5 hours, costs about **$0.55** per configuration.
Prices are from <https://elevenlabs.io/pricing/api> as of 2026-09-24.

## Data retention

ElevenLabs keeps request logs and transcripts by default. Zero-retention mode
(`enable_logging=false`) is available only to enterprise customers, and this backend does not
offer it. **Send only audio you are allowed to share with ElevenLabs.**

For the evaluation corpus, the Hugging Face copy of the Bangor Miami audio (CC BY-SA 3.0) is what
gets uploaded. TalkBank's CHAT transcripts are never uploaded; they are only read locally to score
the results.

## Known limits

- **One language per file.** Scribe reports a single language for the whole file and none per
  word, so every segment carries it. On code-switched English-Spanish audio, stretches in the
  other language are mislabelled, though they are still transcribed. faster-whisper's
  `language_detection=chunked` can label per chunk; this backend cannot.
- **Determinism is best effort.** ElevenLabs says a seed makes the API "make a best effort to
  sample deterministically", and that "determinism is not guaranteed". Two runs over the same
  audio may differ, so compare results against this backend's measured noise floor, not against
  zero.
- **Word timestamps are not exposed** by the pipeline, although Scribe returns them.

## Measured

### Speed and the timeout

Timed on 2026-09-24 from a home connection, uploading the pipeline's 16 kHz mono WAV:

| Audio | Upload | Wall clock | Realtime factor |
| --- | --- | --- | --- |
| 30 s clip | 0.96 MB | 2.4 s, 2.6 s | about 12x |
| `sastre8`, 33.3 min | 64 MB | 46.9 s | about 43x |

`timeout` is not a limit on the whole request. The SDK passes it to httpx, which applies it to
each network operation separately: connecting, each write of the upload, and each read. The
longest single wait is for the transcript after the upload finishes. A 33 min recording finished
in under a minute end to end, so the 900 s default leaves about 19x headroom. Scaled linearly, a
file at the API's 10 h limit would take about 14 min. The SDK's own default is 240 s. Raise
`timeout` only on a slow uplink or for files of many hours.

Both runs of the 30 s clip used the same seed and temperature 0, and they came back with 13 and
14 segments. That is the best-effort determinism described under [Known limits](#known-limits).

<!-- Task 4.4 adds the evaluation against faster-whisper below. -->

## Conformance

```bash
.venv/bin/pytest -k conformance                                  # offline, fake client
AUDIO_TRANSCRIBER_LIVE_TESTS=1 .venv/bin/pytest -k conformance   # also calls the real API
```

The shared suite skips this backend unless `AUDIO_TRANSCRIBER_LIVE_TESTS=1` is set. Without that,
a key in `.env` would make every plain `pytest` spend money and fail offline. The offline run
checks the same contract against canned responses, in the default and `diarization=true`
configurations.
