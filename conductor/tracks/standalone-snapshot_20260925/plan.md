# Implementation Plan: Standalone Backend Scripts and Pipeline Overview

**Track ID:** standalone-snapshot_20260925
**Spec:** [spec.md](./spec.md)
**Created:** 2026-09-25
**Status:** [~] In Progress

## Overview

Build `standalone/` as a detached snapshot: one script per backend, condensed from the backend
module it mirrors, and a README explaining the pipeline. Nothing under `src/` changes. The
scripts are verified by a real run, not by the test suite.

## Phase 1: Directory and Guardrails

Set up the directory so it can never break the main gate or leak a secret.

### Tasks

- [x] Task 1.1: Create `standalone/` with a `.gitignore` covering `.env`, `*.json` output and audio files
- [x] Task 1.2: Exclude `standalone/` from pytest collection and Ruff in `pyproject.toml`
  - **Deviation:** no change needed for pytest, since `testpaths = ["tests"]` never collects
    `standalone/`. Ruff is deliberately kept on: the scripts import nothing from `src/`, so drift
    cannot break their lint, and lint-clean scripts are the better thing to hand over.
- [x] Task 1.3: Prepare a 30–60 s two-speaker, code-switched sample clip locally, uncommitted, to run both scripts on
  - 45 s of Bangor Miami `herring17` from 2:25, chosen from the earlier `word-chunked` eval output
    as the window with the most language switches between two speakers. Kept in the session
    scratchpad, never in the repo.
  - pyannote found only one speaker in that clip, in the package too, so a second clip was added:
    the first 45 s of `zeledon4`, with two speakers and three language spans. It is the sample
    for everything after Phase 1.

### Verification

- [ ] `pytest` and `ruff check . && ruff format --check .` pass unchanged with `standalone/` present

## Phase 2: faster-whisper Script

`standalone/transcribe_faster_whisper.py`, condensed from `backends/faster_whisper.py`, `chunking.py`,
`diarization.py` and `audio.py`.

### Tasks

- [x] Task 2.1: Header docstring with requirements (`pip install faster-whisper pyannote.audio`, ffmpeg, `HF_TOKEN` and the Hugging Face gate, GPU optional) and the run command
- [x] Task 2.2: ffmpeg decode to 16 kHz mono float32, with a clear message when ffmpeg is missing
- [x] Task 2.3: Transcription with `large-v3`, CUDA when available, else CPU `int8`; seeded decoding
- [x] Task 2.4: Chunked language detection limited to English and Spanish: VAD chunks of at most 15 s, a 0.5 threshold falling back to the file language, and agreeing neighbours merged
- [x] Task 2.5: pyannote `speaker-diarization-community-1` with speakers assigned per word; a `--no-diarization` flag that sets `speaker` to `null`
- [x] Task 2.6: Emit the five-key segment array to stdout, with logs on stderr; comments mark where the full backend does more

### Verification

- [x] Copied alone into an empty directory with a fresh venv, the script runs on the sample and prints valid five-key segments, with both languages and two speakers where the audio has them
  - **Deviation:** renamed from `faster_whisper.py`, which shadows the `faster_whisper` library
    and failed on import in the isolated run. Both scripts are now `transcribe_<backend>.py`.
  - Output is identical to the package's (`--option diarization=true
    --option language_detection=chunked --option 'languages=["en","es"]'`) on both clips: same
    text, times to the millisecond, languages and speakers. The first isolated run was not
    identical: decoding to float32 directly, instead of through 16-bit PCM as the pipeline does,
    moved one chunk's detection from 0.514 to 0.474, across the 0.5 threshold. The decode now
    matches the pipeline's sample for sample.
  - `zeledon4` clip: 8 segments, `SPEAKER_00`/`SPEAKER_01`, spans `es en es`, 24 s on the RTX
    3060. `--no-diarization` gives `speaker: null` without `HF_TOKEN`; without the token and
    the flag, it exits before transcribing, naming the gate page.

## Phase 3: ElevenLabs Scribe Script

`standalone/elevenlabs_scribe.py`, condensed from `backends/elevenlabs_scribe.py`.

### Tasks

- [ ] Task 3.1: Header docstring with requirements (`pip install elevenlabs`, ffmpeg, `ELEVENLABS_API_KEY`, the note that audio leaves the machine) and the run command
- [ ] Task 3.2: Decode with ffmpeg, then call Scribe v2 through the SDK with `diarize=True`, reading the key from the environment only
- [ ] Task 3.3: Group words into segments at speaker changes and pauses, with the file language on every segment
- [ ] Task 3.4: Clear messages for a missing key, a rejected key and an exhausted quota
- [ ] Task 3.5: Confirm with the user before spending credits, then run it once live on the sample clip

### Verification

- [ ] Copied alone into an empty directory with a fresh venv, the script prints valid five-key segments with speaker labels

## Phase 4: Pipeline Document

`standalone/README.md`.

### Tasks

- [ ] Task 4.1: What the pipeline does and why: swappable backends, one transcript schema, code-switched English-Spanish audio
- [ ] Task 4.2: How it works: decode, backend interface, normalization, runtime schema validation, run manifest; a short diagram
- [ ] Task 4.3: The transcript schema with a real example segment from the sample run
- [ ] Task 4.4: Backends compared (language granularity, diarization, cost, where audio goes) and the committed evaluation results, as aggregate scores with the corpus citation
- [ ] Task 4.5: How to run each script, and what they simplify compared with the package; the snapshot's date and commit

### Verification

- [ ] The README contains no corpus transcript text, and every command in it has been run as written

## Final Verification

- [ ] All success criteria in spec.md met
- [ ] `pytest`, `ruff check .` and `ruff format --check .` pass
- [ ] No secrets in `standalone/` (`git grep` for key patterns)
- [ ] Manual sign-off recorded in this plan

---

_Generated by Conductor. Tasks will be marked [~] in progress and [x] complete._
