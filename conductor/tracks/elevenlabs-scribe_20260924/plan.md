# Implementation Plan: ElevenLabs Scribe v2 Backend

**Track ID:** elevenlabs-scribe_20260924
**Spec:** [spec.md](./spec.md)
**Created:** 2026-09-24
**Status:** [ ] Not Started

## Overview

Add `src/audio_transcriber/backends/elevenlabs_scribe.py`, registered as `elevenlabs-scribe`,
behind an optional `elevenlabs` extra. Build it in four layers: options and requirement checks,
normalization of Scribe's word list into schema segments, the API call and its error mapping, and
then docs and a measured evaluation. The first three layers are tested without network access. The
backend interface, transcript schema and manifest format stay unchanged, so no contract-change
approval pause is expected. If one turns out to be necessary, stop and ask.

The conformance suite builds every registered backend whose requirements are met. With
`ELEVENLABS_API_KEY` in `.env`, a plain `pytest` would therefore call the paid API. Phase 3 prevents
that: live conformance runs only on explicit opt-in, and a fake-client configuration covers the
contract on every run.

## Phase 1: Foundation — Extra, Registration, Options, Requirements

Make the backend selectable and able to validate itself, without transcribing yet.

### Tasks

- [ ] Task 1.1: Add the `elevenlabs = ["elevenlabs>=2.0,<3.0"]` extra to `pyproject.toml`; install
      it in `.venv` and record the installed version
- [ ] Task 1.2: Add `elevenlabs` to `ENGINE_LIBRARIES` in `tests/test_no_engine_leakage.py`, and
      the new module to the registry's list of built-in backend modules
- [ ] Task 1.3: Write the option tests first: defaults (`seed=0`, `temperature=0.0`,
      `diarization=False`, `timeout` default, `language=None`); unknown options rejected;
      `num_speakers` and `min_speakers` rejected with a pointer to `max_speakers`; `max_speakers`
      checked to be between 1 and 32; `diarization_threshold` rejected alongside `max_speakers` or
      without `diarization`; `diarization=True` flips the per-instance capability
- [ ] Task 1.4: Implement `ElevenLabsScribeBackend` skeleton: `name`, class `Capabilities`
      (`language_detection=True`, `word_timestamps=True`), `_DEFAULTS`, option validation,
      `effective_config()` with `api_key` redacted, `model_version()` → `scribe_v2` + SDK version
- [ ] Task 1.5: Shared `.env` credentials. Move `_load_dotenv_once()` and the lookup order
      (config, then environment, then `.env`) out of `backends/diarization.py` into a shared
      helper that both backends use. `find_dotenv(usecwd=True)` and `override=False` stay as they
      are, and existing diarization tests pass unchanged. Add a committed `.env.example` listing
      `HF_TOKEN=` and `ELEVENLABS_API_KEY=` with comments on where to get each; never write to the
      real `.env`
- [ ] Task 1.6: Test and implement `check_requirements()`: missing SDK → install command for the
      extra; missing key → names `ELEVENLABS_API_KEY`, `.env` and `.env.example`. The key is found
      through the shared helper from Task 1.5

### Verification

- [ ] `pytest tests/test_elevenlabs_scribe.py tests/test_no_engine_leakage.py tests/test_registry.py`
      passes; `--backend elevenlabs-scribe` with no key fails up front with the fix-it message

## Phase 2: Normalization — Scribe Response → Transcript

Turn Scribe's response into schema segments. This is pure logic, tested against hand-written
responses shaped like the documented `SpeechToTextChunkResponseModel`.

### Tasks

- [ ] Task 2.1: Test and implement language-code normalization: `eng`/`en` → `en`, `spa`/`es` →
      `es`, other 639-3 codes mapped where a 639-1 code exists, and unknown codes passed through
      unchanged
- [ ] Task 2.2: Test and implement word grouping. Drop `spacing` and `audio_event` tokens. Start a
      new segment on a `speaker_id` change or on a pause over a threshold. Segment text is built
      from the words and stripped. Timings are ascending and non-overlapping, and words with null
      `start`/`end` are handled. The pause threshold is a module constant, recorded in the
      manifest if it is made configurable
- [ ] Task 2.3: Test and implement `_normalize()` → `Transcript`: every segment gets the
      file-level `spoken_language`; `speaker` is set only when diarization is on; `duration`
      comes from `audio_duration_secs` or else the WAV header; `source` is the original input.
      Empty `words` → an empty transcript, not an error
- [ ] Task 2.4: Test edge cases: a response of only audio events or spacing, a single word,
      segments reaching past the audio's end (clamped or rejected so runtime validation doesn't
      turn them into a file failure unexpectedly), and a multichannel-shaped response rejected
      clearly

### Verification

- [ ] Normalized fixtures pass `assert_transcript_conforms`; all Phase 2 tests pass

## Phase 3: API Call, Error Mapping, Conformance

Wire the SDK call and make failures per-file, with the client injectable so no test touches the
network.

### Tasks

- [ ] Task 3.1: Implement `transcribe()`. Build the client once, with the `timeout` option and
      the SDK's default retries. Call `speech_to_text.convert` with `model_id="scribe_v2"`, the
      WAV file, `timestamps_granularity="word"`, `tag_audio_events=False`, `diarize`, `num_speakers`
      (from `max_speakers`), `diarization_threshold`, `language_code`, `seed` and `temperature`.
      Omitted options are not sent
- [ ] Task 3.2: Using a fake client, test that the request carries exactly the configured
      parameters, and that options left unset are omitted
- [ ] Task 3.3: Test and implement error mapping. `ApiError` subclasses become
      `TranscriptionError` with the status and a short reason: 401 means check the key, 422 means
      a validation message, 429 means rate limited after retries. `httpx.TimeoutException`
      becomes an error naming the `timeout` option. Other transport errors become a connection
      failure. The API key never appears in a message. The pipeline continues to the next file
- [ ] Task 3.4: Conformance. Run the shared suite against the backend with a fake client
      returning a canned response, on every `pytest` run. Skip the live registered configurations
      unless `AUDIO_TRANSCRIBER_LIVE_TESTS=1` is set, so a key in `.env` never makes the default
      suite call the paid API. Add `{"diarization": True}` to `_EXTRA_CONFIGS`

### Verification

- [ ] `pytest` passes offline with a key present in `.env`, making no network calls
- [ ] `AUDIO_TRANSCRIBER_LIVE_TESTS=1 pytest -k conformance` passes against the real API

## Phase 4: Documentation, Measured Defaults, Evaluation

Document the backend and measure it against faster-whisper.

### Tasks

- [ ] Task 4.1: Measure the `timeout` default. Time real requests on a short clip and on the
      longest Bangor Miami recording (`sastre8`, 33.4 min), then set the default with headroom and
      record the measurement in the doc page
- [ ] Task 4.2: Write `docs/backends/elevenlabs-scribe.md`: usage and options; the extra; the
      credential and `.env`; cost ($0.22/h, about $0.55 per Bangor Miami run); limits (5 GB,
      10 h, 100 ms minimum, 32 speakers); per-file language only; best-effort seed; the
      difference between `max_speakers` and pyannote's count; data retention (audio leaves the
      machine and ElevenLabs stores it; zero retention is enterprise only); errors and retries
- [ ] Task 4.3: Update `README.md` backend list, `docs/backend-interface.md` if it lists
      backends, and `conductor/tech-stack.md` (extras table, version matrix, a decision-log entry
      on the first hosted backend covering retention and live-test gating)
- [ ] Task 4.4: Evaluate on Bangor Miami. Run `elevenlabs-scribe` without diarization, run it
      again to measure the noise floor, and run once with diarization and `--talkbank` for speaker
      and language scores. Compare with the committed faster-whisper report per recording, and
      commit the reports and the comparison. Note where per-file language costs accuracy on
      code-switched recordings
- [ ] Task 4.5: Real pipeline run on sample audio via
      `python -m audio_transcriber --backend elevenlabs-scribe`; check the transcripts and that
      the manifest records the model version, the config with the key redacted, and the timeout

### Verification

- [ ] Doc page complete per workflow Step 8; evaluation reports committed with the TalkBank
      citation where TalkBank scores appear

## Final Verification

- [ ] All acceptance criteria met
- [ ] `ruff check . && ruff format --check . && pytest` pass (offline)
- [ ] `pytest -k conformance` passes, and passes live with `AUDIO_TRANSCRIBER_LIVE_TESTS=1`
- [ ] Documentation updated: backend doc page, README, tech-stack
- [ ] Manual sign-off recorded in this plan

---

_Generated by Conductor. Tasks will be marked [~] in progress and [x] complete._
