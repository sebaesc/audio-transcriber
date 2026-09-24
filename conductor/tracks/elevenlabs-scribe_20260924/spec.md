# Specification: ElevenLabs Scribe v2 Backend

**Track ID:** elevenlabs-scribe_20260924
**Type:** Feature
**Created:** 2026-09-24
**Status:** Draft

## Summary

Add an `elevenlabs-scribe` backend that transcribes through ElevenLabs' hosted Scribe v2
speech-to-text API, behind the existing backend interface. It is the project's first hosted
backend.

## Context

The pipeline treats the transcription model as a plug-in: a backend implements
`TranscriptionBackend` (`src/audio_transcriber/backend.py`), registers by name, and normalizes its
own output to the shared five-key segment schema. faster-whisper is the only backend today. The
team's audio is mostly English with Spanish speakers who code-switch, and backend quality is
judged by the evaluation harness against the Bangor Miami corpus, per recording, with committed
reports.

## User Story

As a member of the internal team, I want to select `--backend elevenlabs-scribe` so that I can
transcribe our code-switched English/Spanish audio with a hosted model and compare it against
faster-whisper using the evaluation harness, without changing pipeline code.

## Acceptance Criteria

- [x] `--backend elevenlabs-scribe` transcribes audio via ElevenLabs Scribe v2; output passes
      runtime schema validation and the shared conformance tests
- [x] `check_requirements()` fails up front, with a fix-it message, when the `elevenlabs` SDK or
      `ELEVENLABS_API_KEY` is missing; the key is read from `.env` the same way `HF_TOKEN` is, and
      is redacted from the manifest. (A missing SDK is reported by the registry at selection,
      before `check_requirements()` can run; see plan Task 4.6.)
- [x] Speaker labels come from Scribe's own diarization when enabled; every segment's
      `spoken_language` is Scribe's file-level language, mapped to the schema's code format;
      capabilities are declared to match; the pinned model id is recorded in the manifest
- [x] API failures (auth, rate limit, timeout, oversized file) become a per-file
      `TranscriptionError` and the batch continues
- [x] `docs/backends/elevenlabs-scribe.md` covers usage, the extra, the credential, cost and
      limits (including per-file language only), and an evaluation report comparing it with
      faster-whisper on Bangor Miami is committed alongside the existing reports. (Deviation:
      3 of 5 recordings, as the free-tier quota ran out; see plan note 4.4.)

## Dependencies

Existing code from completed tracks: the backend interface and registry, the conformance suite,
runtime schema validation, `.env` credential loading (`backends/diarization.py`), and the
evaluation harness. No open track blocks this one.

## Out of Scope

- Scribe v2 Realtime / streaming transcription (batch file transcription only)
- Scribe v1 and other ElevenLabs models; only `scribe_v2` is pinned
- Any change to the backend interface, transcript schema, or manifest format
- Retry/backoff beyond the SDK's defaults
- Cost tracking or budget limits
- Translation, audio event tagging, keyterm prompting, entity detection, speaker roles
- Webhook/async delivery, multichannel mode, `source_url` uploads
- Making Scribe the default backend; faster-whisper stays the default

## Technical Notes

- **Language is per file, not per segment.** Scribe's response carries one top-level
  `language_code`; `words[]` has no language. Every segment gets the file language (decision (a)).
  The limitation is documented, and the evaluation report shows its cost on code-switched audio.
  Per-segment language ID on top of Scribe is a possible future track.
- Official `elevenlabs` Python SDK (`>=2.0,<3.0`; 2.68.0 checked) as a new optional extra
  `elevenlabs`. It brings httpx, pydantic, requests and websockets. The core package does not
  depend on it, and no engine import leaks outside the backend module.
- `POST /v1/speech-to-text` via `client.speech_to_text.convert(model_id="scribe_v2", ...)`,
  synchronous (no `webhook`), single channel (no `use_multi_channel`). `model_version()` returns
  `scribe_v2` plus the SDK version, since the hosted model has no finer version to pin.
- **Language codes.** The response's `language_code` is documented as ISO 639-3 (`eng`) in one
  place and shown as `en` in another. Normalize both forms to the ISO 639-1 codes faster-whisper
  emits (`en`, `es`). Codes that can't be mapped pass through unchanged. The `language` option
  accepts 639-1 or 639-3, as the API does.
- **Limits.** Files are limited to 5 GB (raised from 3 GB on 2026-06-01; the capabilities page still
  says 3 GB), 10 hours in standard mode, and a 100 ms minimum. The decoded WAV is about 115 MB per
  hour, so upload size is not a practical constraint.
- **Timeout.** The SDK's default timeout is 240 s. It is an httpx timeout, applied to each network
  operation (connect, each write, each read), not to the whole request. The wait that matters is
  the read while Scribe transcribes. Add a `timeout` option with a generous default, measured
  during implementation, and record it in the manifest. (Corrected in Task 4.1; this spec first
  called it a limit on the whole request. The measurement is in the plan.)
- **Retries.** The SDK retries 408/409/429/5xx twice by default with backoff that honors
  `Retry-After`. Timeouts and connection errors are not retried. After that, `ApiError`
  (`status_code`, `body`) and httpx transport errors become a `TranscriptionError`; the message
  includes the status but never the API key.
- **Diarization.** `diarize=true`, up to 32 speakers, with no surcharge. Scribe's `num_speakers`
  is a *maximum*, unlike pyannote's exact count. It is exposed as `max_speakers`; `num_speakers`
  and `min_speakers` are rejected rather than silently reinterpreted. `diarization_threshold` is
  valid only when no speaker count is set, and that is enforced up front.
- **Determinism.** `seed` (0–2147483647) is best effort: "determinism is not guaranteed".
  `temperature` ranges 0–2 and defaults to about 0. Both are sent (`seed` defaults to 0) and
  recorded. The evaluation cannot be byte-reproducible, so a repeat run measures this backend's
  noise floor.
- **Segments.** Request `timestamps_granularity="word"`. Group `words[]` into segments on speaker
  change and on pauses. Drop `spacing` tokens, and any `audio_event` tokens (`tag_audio_events`
  is sent as false).
- **Cost.** $0.22 per audio hour. A full Bangor Miami evaluation (about 2.5 h) costs about $0.55
  per configuration.
- **Data retention.** ElevenLabs keeps logs and transcripts by default (`enable_logging=true`).
  Zero-retention mode is available only to enterprise customers. The doc page states this
  plainly: audio sent to this backend leaves the machine and is stored by ElevenLabs. The Bangor
  Miami audio (Hugging Face copy, CC BY-SA 3.0) may be shared. TalkBank CHAT files are never
  uploaded; they are only read locally for scoring.
- Unit tests use recorded or fake API responses and never touch the network. Real API calls happen
  only in the manual verification run and the evaluation report.

---

_Generated by Conductor. Review and edit as needed._
