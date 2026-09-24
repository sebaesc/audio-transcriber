# Implementation Plan: ElevenLabs Scribe v2 Backend

**Track ID:** elevenlabs-scribe_20260924
**Spec:** [spec.md](./spec.md)
**Created:** 2026-09-24
**Status:** [x] Complete

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

- [x] Task 1.1: Add the `elevenlabs = ["elevenlabs>=2.0,<3.0"]` extra to `pyproject.toml`; install
      it in `.venv` and record the installed version
- [x] Task 1.2: Add `elevenlabs` to `ENGINE_LIBRARIES` in `tests/test_no_engine_leakage.py`, and
      the new module to the registry's list of built-in backend modules
- [x] Task 1.3: Write the option tests first: defaults (`seed=0`, `temperature=0.0`,
      `diarization=False`, `timeout` default, `language=None`); unknown options rejected;
      `num_speakers` and `min_speakers` rejected with a pointer to `max_speakers`; `max_speakers`
      checked to be between 1 and 32; `diarization_threshold` rejected alongside `max_speakers` or
      without `diarization`; `diarization=True` flips the per-instance capability
- [x] Task 1.4: Implement `ElevenLabsScribeBackend` skeleton: `name`, class `Capabilities`
      (`language_detection=True`, `word_timestamps=True`), `_DEFAULTS`, option validation,
      `effective_config()` with `api_key` redacted, `model_version()` → `scribe_v2` + SDK version
- [x] Task 1.5: Shared `.env` credentials. Move `_load_dotenv_once()` and the lookup order
      (config, then environment, then `.env`) out of `backends/diarization.py` into a shared
      helper that both backends use. `find_dotenv(usecwd=True)` and `override=False` stay as they
      are, and existing diarization tests pass unchanged. Add a committed `.env.example` listing
      `HF_TOKEN=` and `ELEVENLABS_API_KEY=` with comments on where to get each; never write to the
      real `.env`
- [x] Task 1.6: Test and implement `check_requirements()`: missing SDK → install command for the
      extra; missing key → names `ELEVENLABS_API_KEY`, `.env` and `.env.example`. The key is found
      through the shared helper from Task 1.5

### Verification

- [x] `pytest tests/test_elevenlabs_scribe.py tests/test_no_engine_leakage.py tests/test_registry.py`
      passes; `--backend elevenlabs-scribe` with no key fails up front with the fix-it message
      (exit 3, no output written)

### Notes and deviations

- **1.1:** `elevenlabs` 2.68.0 installed (brings httpx 0.28.1).
- **1.5:** Moving the flag to `backends/credentials.py` meant `tests/test_diarization.py` could
  not stay unchanged: its monkeypatches now target `credentials._dotenv_loaded` instead of
  `diarization._dotenv_loaded`. Only the patch target changed; the assertions did not. Keeping a
  dead alias in `diarization` just to leave the tests untouched would have been worse.
- **1.6:** A missing SDK cannot reach `check_requirements()`. The module imports the SDK at the
  top, so the registry skips the backend and `--backend elevenlabs-scribe` reports "unknown
  backend", the same as faster-whisper without its extra. Naming the missing extra there needs a
  registry change (remember why each built-in failed to import). Done in Task 4.6.
- **1.6:** The key is not tried against the API up front, because that would spend a request on
  every run. A wrong key fails every file with a 401 that names the key.

## Phase 2: Normalization — Scribe Response → Transcript

Turn Scribe's response into schema segments. This is pure logic, tested against hand-written
responses shaped like the documented `SpeechToTextChunkResponseModel`.

### Tasks

- [x] Task 2.1: Test and implement language-code normalization: `eng`/`en` → `en`, `spa`/`es` →
      `es`, other 639-3 codes mapped where a 639-1 code exists, and unknown codes passed through
      unchanged
- [x] Task 2.2: Test and implement word grouping. Drop `spacing` and `audio_event` tokens. Start a
      new segment on a `speaker_id` change or on a pause over a threshold. Segment text is built
      from the words and stripped. Timings are ascending and non-overlapping, and words with null
      `start`/`end` are handled. The pause threshold is a module constant, recorded in the
      manifest if it is made configurable
- [x] Task 2.3: Test and implement `_normalize()` → `Transcript`: every segment gets the
      file-level `spoken_language`; `speaker` is set only when diarization is on; `duration`
      comes from `audio_duration_secs` or else the WAV header; `source` is the original input.
      Empty `words` → an empty transcript, not an error
- [x] Task 2.4: Test edge cases: a response of only audio events or spacing, a single word,
      segments reaching past the audio's end (clamped or rejected so runtime validation doesn't
      turn them into a file failure unexpectedly), and a multichannel-shaped response rejected
      clearly

### Verification

- [x] Normalized fixtures pass `assert_transcript_conforms`; all Phase 2 tests pass

### Notes

- **2.1:** Built from Scribe v2's published list of 99 languages. Eight have no ISO 639-1 code
  and pass through as-is: `ast`, `ceb`, `fil`, `kea`, `luo`, `nso`, `umb`, `yue`.
- **2.2:** Segments break at a sentence end (`. ? ! …`), a speaker change (diarization only),
  or a pause over `PAUSE_SECONDS = 1.0`. Spacing tokens are kept inside a segment, so text keeps
  Scribe's own spacing, and none is invented for languages written without spaces. The
  threshold is a constant, not an option; revisit if the evaluation shows segmentation hurting
  anything.

## Phase 3: API Call, Error Mapping, Conformance

Wire the SDK call and make failures per-file, with the client injectable so no test touches the
network.

### Tasks

- [x] Task 3.1: Implement `transcribe()`. Build the client once, with the `timeout` option and
      the SDK's default retries. Call `speech_to_text.convert` with `model_id="scribe_v2"`, the
      WAV file, `timestamps_granularity="word"`, `tag_audio_events=False`, `diarize`, `num_speakers`
      (from `max_speakers`), `diarization_threshold`, `language_code`, `seed` and `temperature`.
      Omitted options are not sent
- [x] Task 3.2: Using a fake client, test that the request carries exactly the configured
      parameters, and that options left unset are omitted
- [x] Task 3.3: Test and implement error mapping. `ApiError` subclasses become
      `TranscriptionError` with the status and a short reason: 401 means check the key, 422 means
      a validation message, 429 means rate limited after retries. `httpx.TimeoutException`
      becomes an error naming the `timeout` option. Other transport errors become a connection
      failure. The API key never appears in a message. The pipeline continues to the next file
- [x] Task 3.4: Conformance. Run the shared suite against the backend with a fake client
      returning a canned response, on every `pytest` run. Skip the live registered configurations
      unless `AUDIO_TRANSCRIBER_LIVE_TESTS=1` is set, so a key in `.env` never makes the default
      suite call the paid API. Add `{"diarization": True}` to `_EXTRA_CONFIGS`

### Verification

- [x] `pytest` passes offline with a key present in `.env`, making no network calls. Run with
      `ELEVENLABS_API_KEY=sk_dummy_not_real`: the 8 hosted conformance cases skip with the opt-in
      message, and everything else passes; ruff is clean
- [x] `AUDIO_TRANSCRIBER_LIVE_TESTS=1 pytest -k conformance` passes against the real API
      (2026-09-24, after the connection fix in `49ca9fe`). Before a real key was available: With a dummy key, the live path reached the real endpoint and came back as a
      clean per-file 401 `TranscriptionError` that names `ELEVENLABS_API_KEY` and does not echo the
      key. So the SDK wiring and upload are correct as far as authentication

### Notes

- **3.1:** The upload is `("<name>.wav", handle, "audio/wav")`. `file_format` stays at the API
  default (`other`): `pcm_s16le_16` expects raw PCM without a WAV header.
- **3.3:** Messages are built from the status and body, never from `str(ApiError)`, which dumps
  the response headers. As a safeguard, the key is scrubbed from every message.
- **3.4:** Hosted backends are listed in `_HOSTED` in `tests/test_conformance.py`. The offline
  contract runs in `tests/test_elevenlabs_scribe.py` (default and `diarization=true`, plus
  silence), marked `conformance` so `pytest -k conformance` includes it.

## Phase 4: Documentation, Measured Defaults, Evaluation

Document the backend and measure it against faster-whisper.

### Tasks

- [x] Task 4.1: Measure the `timeout` default. Time real requests on a short clip and on the
      longest Bangor Miami recording (`sastre8`, 33.4 min), then set the default with headroom and
      record the measurement in the doc page
- [x] Task 4.2: Write `docs/backends/elevenlabs-scribe.md`: usage and options; the extra; the
      credential and `.env`; cost ($0.22/h, about $0.55 per Bangor Miami run); limits (5 GB,
      10 h, 100 ms minimum, 32 speakers); per-file language only; best-effort seed; the
      difference between `max_speakers` and pyannote's count; data retention (audio leaves the
      machine and ElevenLabs stores it; zero retention is enterprise only); errors and retries
- [x] Task 4.3: Update `README.md` backend list, `docs/backend-interface.md` if it lists
      backends, and `conductor/tech-stack.md` (extras table, version matrix, a decision-log entry
      on the first hosted backend covering retention and live-test gating)
- [x] Task 4.4: Evaluate on Bangor Miami. Run `elevenlabs-scribe` without diarization, run it
      again to measure the noise floor, and run once with diarization and `--talkbank` for speaker
      and language scores. Compare with the committed faster-whisper report per recording, and
      commit the reports and the comparison. Note where per-file language costs accuracy on
      code-switched recordings
- [x] Task 4.5: Real pipeline run on sample audio via
      `python -m audio_transcriber --backend elevenlabs-scribe`; check the transcripts and that
      the manifest records the model version, the config with the key redacted, and the timeout

### Notes

- **4.1:** Measured 2026-09-24 against the real API: a 30 s clip took 2.4 s and 2.6 s, and
  `sastre8` (33.3 min, 64 MB WAV) took 46.9 s end to end. `DEFAULT_TIMEOUT` stays at 900 s.
  Deviation: the spec called `timeout` a limit on the whole request. The SDK passes it to httpx,
  which applies it per operation, so it bounds the wait for the response, not upload plus
  response. The spec, the docstring and the doc page are corrected. The clip gave 13 and then 14
  segments with the same seed, which is early evidence that output is not deterministic.
- **4.4 (deviation):** the key is a free account with 10,000 credits a month, about 150 min of
  audio at 66.7 credits per minute. It ran out during the first evaluation run. `eval/scribe.json`
  scores `herring17`, `maria20` and `sastre8`; `zeledon14` and `zeledon4` failed with
  `quota_exceeded`, and those failures stay in the report. The repeat run (noise floor) and the
  diarized run got no credits and were dropped with the user's agreement. Their reports held only
  failures and were deleted. The doc page states all three gaps. The result does not depend on
  the noise floor: Scribe's WER is 0.11–0.19 lower on every scored recording, against
  faster-whisper's measured spread of 0.022. Scribe's TalkBank language accuracy is 0.11–0.52
  lower, because it labels each file with one language.
- **4.4 (bug found):** ElevenLabs returns `quota_exceeded` as HTTP 401, and the error mapping
  blamed the key. Fixed with a test first (`f8ff85f`).
- **Final verification (bug found):** live conformance failed on `ResourceWarning: unclosed
  <ssl.SSLSocket>`, which `filterwarnings = error` turns into a failure. Task 3.1 built the
  client once per backend, and the interface has no close hook, so the connection was never
  closed. Deviation from 3.1: a client is now opened per file and closed when the file is done,
  including on failure. Tested first. The fake-client tests never opened a socket, so only a
  live run could have caught it.
- **4.4 (follow-up, eval harness):** `--compare` averages each report over whichever recordings
  it scored, so its mean deltas mislead when two reports scored different sets. The
  per-recording deltas are unaffected. Not fixed here.
- **4.5:** a 10 s MP3 cut of `sastre8` through the CLI: exit 0, 4 segments, and the manifest
  records `elevenlabs-2.68.0/scribe_v2`, `timeout: 900.0` and `api_key: null`. No output file
  contains the key. Code-switched speech was transcribed in both languages, with every segment
  labelled `es`: the per-file language limit, as documented.
- **4.6:** checked for real by blocking the `elevenlabs` import in a subprocess.
  `--backend elevenlabs-scribe` printed `backend 'elevenlabs-scribe' is not installed: it needs
  'elevenlabs'. Install its extra: pip install -e ".[elevenlabs]"`, exited 3 and wrote nothing.
  One behaviour change: a built-in whose own module fails to import used to be skipped
  silently. It is now reported as broken. Test finding: a registration check that relies on
  re-import passes alone and fails in the full suite, since modules import once. The test reads
  the module's classes instead.
- **Key:** the `.env` key authenticates for speech-to-text but lacks `user_read`, so account
  endpoints such as subscription return 401. The backend does not use them.

- [x] Task 4.6: Name the missing extra. Added after 4.5 at the user's request, to resolve the
      1.6 follow-up. The registry declares each built-in's name, module and extra, and remembers
      why each import failed. Selecting one whose extra is missing raises
      `BackendNotInstalledError` (a subclass of `UnknownBackendError`) with the `pip install`
      command, and the CLI exits 3. `--list-backends` notes missing built-ins on stderr

### Verification

- [x] Doc page complete per workflow Step 8; evaluation reports committed with the TalkBank
      citation where TalkBank scores appear

## Final Verification

- [x] All acceptance criteria met
- [x] `ruff check . && ruff format --check . && pytest` pass (offline)
- [x] `pytest -k conformance` passes, and passes live with `AUDIO_TRANSCRIBER_LIVE_TESTS=1`
- [x] Documentation updated: backend doc page, README, tech-stack
- [x] Manual sign-off recorded in this plan: approved by Sebastian Escuder, 2026-09-24

---

_Generated by Conductor. Tasks will be marked [~] in progress and [x] complete._
