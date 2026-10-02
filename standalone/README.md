# Audio Transcriber — standalone snapshot

A progress snapshot of **Audio Transcriber**, a modular transcription pipeline built for audio
that is mostly English with Spanish speakers who switch languages mid-conversation.

This directory stands on its own. It holds one script per transcription backend. Each script is
a condensed copy of that backend and runs with one command, with no dependency on the rest of
the project. This README explains what the full pipeline does and how the scripts relate to it.

| File | What it runs | Where the audio goes |
| --- | --- | --- |
| [`transcribe_faster_whisper.py`](transcribe_faster_whisper.py) | Whisper `large-v3` via faster-whisper, per-chunk English/Spanish detection, pyannote diarization | Stays on the machine |
| [`transcribe_elevenlabs_scribe.py`](transcribe_elevenlabs_scribe.py) | ElevenLabs Scribe v2, with Scribe's own diarization | Uploaded to ElevenLabs |

**Snapshot of:** commit `166ddb1` on `main`, 2026-09-25. The scripts are copies and are not
kept in sync with the project.

---

## Quick start

Both scripts need Python 3.12 and ffmpeg (`apt install ffmpeg`, `brew install ffmpeg`). Use a
separate virtual environment for each; they share nothing.

### faster-whisper (local)

```bash
python3.12 -m venv .venv-fw
.venv-fw/bin/pip install "faster-whisper>=1.0,<2.0" "pyannote.audio>=4.0,<5.0"
.venv-fw/bin/pip install nvidia-cublas-cu12 nvidia-cudnn-cu12   # only with an NVIDIA GPU

export HF_TOKEN=hf_...    # see "Diarization token" below
.venv-fw/bin/python transcribe_faster_whisper.py recording.mp3 > transcript.json
```

Add `--no-diarization` to skip speakers; then no token is needed. The first run downloads about
3 GB of model weights. On an RTX 3060 a 45 s clip takes about 24 s, including loading both
models. On CPU, expect it to be many times slower.

**Diarization token.** The pyannote model is free (MIT) but gated. Once:

1. Accept its conditions at <https://huggingface.co/pyannote/speaker-diarization-community-1>.
2. Create a read token at <https://huggingface.co/settings/tokens> and export it as `HF_TOKEN`.

### ElevenLabs Scribe v2 (hosted)

```bash
python3.12 -m venv .venv-el
.venv-el/bin/pip install "elevenlabs>=2.0,<3.0"

export ELEVENLABS_API_KEY=...    # https://elevenlabs.io/app/settings/api-keys
.venv-el/bin/python transcribe_elevenlabs_scribe.py recording.mp3 > transcript.json
```

Add `--max-speakers N` to cap the number of speakers. Scribe costs about 67 credits per minute
of audio. A free account's 10,000 monthly credits cover about 150 minutes. A 45 s clip returned
in under 3 s.

Both scripts print the transcript to stdout and their progress to stderr, and exit non-zero with
a message saying what to fix: missing ffmpeg, a missing token or key, an unaccepted model gate,
an exhausted quota.

---

## What the pipeline does

Audio goes in; a **transcript** comes out, one per file: a list of segments, each saying when
it was spoken, by whom, in which language, and what was said.

Two problems drive the design.

1. **Transcription tools hard-wire one engine.** Once code is written against one model's API
   and output format, switching models means rewriting it. So here the model is a **plug-in**
   behind one interface. Every engine produces the same output. Comparing engines, or changing
   the default, touches no pipeline code.
2. **Our audio switches language mid-sentence.** Whisper decides one language per file and
   decodes everything under it. Spanish stretches of an English recording come out mislabelled,
   and sometimes transliterated into English spelling. So language is recorded **per segment**,
   and detected per stretch of speech rather than once per file.

### How a file moves through it

```
 input file (any format ffmpeg reads)
      │
      ▼
 ┌───────────────┐   ffmpeg, once, in the pipeline: every backend receives the
 │ decode        │   same mono 16 kHz WAV and never deals with containers
 └──────┬────────┘
        ▼
 ┌───────────────┐   the selected backend, behind one interface:
 │ backend       │   check_requirements() → transcribe() → normalized transcript
 │ (plug-in)     │   faster-whisper · elevenlabs-scribe · …
 └──────┬────────┘
        ▼
 ┌───────────────┐   every transcript is checked against the schema before it is
 │ validate      │   written; a violation is a reported failure, not bad data on disk
 └──────┬────────┘
        ▼
 out/<run>/<file>.json    the transcript: a bare JSON array of segments
 out/<run>/manifest.json  backend, model version, every option, per-file results
 out/<run>/errors.json    one entry per file that failed; the run carries on
```

What the project adds around that, and these scripts leave out:

- **Up-front checks.** A missing model, device, token or ffmpeg stops the run before any file is
  touched, with the command that fixes it. The run doesn't fail half-way through a batch.
- **A run manifest.** It records the backend, model and library versions, every option, and
  the device actually used, so any transcript can be reproduced. Credentials are redacted.
- **Batch runs.** A directory in, one transcript per file out. One file failing is reported and
  the rest continue.
- **A conformance suite.** Shared tests every backend must pass to prove it honours the schema.
  With a network-free fake, the hosted backend runs them on every test run.
- **An evaluation harness.** It scores backends against a Spanish-English reference corpus (see
  [Measured results](#measured-results)), or against a ground truth written for your own
  recordings, including transcripts already made (see `docs/evaluation.md`).

### The transcript schema

A transcript file is a bare JSON array. Each segment has exactly five keys:

| Key | Type | Meaning |
| --- | --- | --- |
| `start_time` | number | Seconds from the start of the audio |
| `end_time` | number | Seconds; never before `start_time` |
| `speaker` | string or null | `SPEAKER_00`, `speaker_0`, …; stable within one file, not across files. `null` when not diarized |
| `spoken_language` | string or null | ISO 639-1 code of *this segment's* speech: `en`, `es` |
| `text` | string | What was said |

Segments are in time order and never overlap. An example, with invented text:

```json
[
  {"start_time": 0.46, "end_time": 4.1, "speaker": "SPEAKER_01", "spoken_language": "es", "text": "¿Y tú fuiste a la reunión ayer?"},
  {"start_time": 4.52, "end_time": 7.9, "speaker": "SPEAKER_00", "spoken_language": "en", "text": "Yeah, but I left early, it was so long."}
]
```

Both scripts print exactly this shape. Language codes are the same from both backends: Scribe's
`spa` becomes `es`, as Whisper reports it. Speaker labels differ in spelling (`SPEAKER_00`
against `speaker_0`) because they come from different diarizers.

### The two backends

| | faster-whisper | ElevenLabs Scribe v2 |
| --- | --- | --- |
| Runs | Locally, GPU or CPU | ElevenLabs' servers |
| Audio | Never leaves the machine | Uploaded and retained under the account's settings |
| Cost | Hardware only | About $0.22 per audio hour |
| Language | **Per chunk** of speech, English or Spanish | **One per file** |
| Diarization | pyannote `speaker-diarization-community-1`, speakers per word | Scribe's own, at no extra cost |
| Reproducible | Yes: seeded, byte-identical run to run | Best effort only |
| Credentials | `HF_TOKEN`, for diarization | `ELEVENLABS_API_KEY` |

**How faster-whisper detects language per chunk.** Silero VAD finds the speech and cuts it into
chunks of at most 15 s at the best available silences. Each chunk's language is detected, and
only English and Spanish are allowed. Otherwise the model sometimes ranks Portuguese or Galician
first on Spanish. A chunk whose best score is under 0.5 keeps the whole file's language.
Neighbouring chunks that agree are then decoded together, since every decode boundary risks
cutting a word in two.

**How speakers are assigned.** pyannote returns speaker turns, and Whisper returns a time for each
word. Each word takes the speaker whose turn covers most of it. A lone word flanked by one other
speaker is folded into that speaker, because word times drift at turn boundaries. A segment is
split wherever its speaker changes, so an interruption is attributed to whoever made it.

**How Scribe output is shaped.** Scribe returns words, not segments. A new segment starts at a
speaker change, after a sentence ends, or at a pause over 1 s.

---

## Measured results

Quality is measured, not assumed. The harness scores each backend against the
[Bangor Miami Spanish-English corpus](https://huggingface.co/datasets/BrunoHays/Bangor-Miami-Spanish-English-Corpus)
(CC BY-SA 3.0): five naturalistic two-speaker conversations recorded in Miami, about 2.5 hours,
each 20–80% Spanish. Speaker and per-segment language scores use TalkBank's original annotations
of the same recordings. Results are judged recording by recording, never on the mean, against a
measured run-to-run noise floor of 0.022 WER per recording.

Absolute scores are pessimistic. The references keep unintelligible-speech markers and fillers
that no transcriber reproduces.

**Per-chunk language detection (faster-whisper `large-v3`)**, against one language per file:

- It reported both languages on every recording, where per-file detection reported one.
- Per-segment language accuracy rose on all five recordings, from 0.67 to 0.82 on average.
- The error in the share of Spanish written out fell from 0.164 to 0.044.
- WER improved on four of five recordings (mean −0.034).

**Speaker labels (faster-whisper + pyannote, per-word assignment).** On average 4.6% of labelled
speech went to the wrong speaker with per-chunk language detection (2.6–7.6% per recording
across configurations). That is on the four recordings whose reference times both speakers.

**Scribe against faster-whisper** (chunked language detection). Only three recordings were
scored before the free-tier quota ran out:

| Recording | WER, Scribe / faster-whisper | Language accuracy, Scribe / faster-whisper |
| --- | --- | --- |
| `herring17` | **0.307** / 0.416 | 0.635 / **0.778** |
| `maria20` | **0.224** / 0.411 | 0.301 / **0.817** |
| `sastre8` | **0.160** / 0.291 | 0.655 / **0.762** |

Scribe transcribes the words markedly better, with WER 0.11–0.19 lower on every recording, and
runs 28–59x realtime. But it labels a whole file with one language, so per-segment language
accuracy is 0.11–0.52 worse. faster-whisper stays the default where knowing which language each
segment is in matters. Scribe is the better choice where only the words matter.

> TalkBank scores cite: Deuchar, M., & Parafita Couto, M. (2011). The Bangor Miami Corpus.
> TalkBank. doi:10.21415/T5J01D

### Verification of these scripts

Both scripts were run from an empty directory, in a fresh virtual environment built only from
their own requirements, on two 45 s clips of corpus audio (not included here):

- `transcribe_faster_whisper.py`: the same text, times, languages and speakers as the full
  pipeline run with the same options, on both clips. On the two-speaker clip: 8 segments, two
  speakers, language spans `es → en → es`.
- `transcribe_elevenlabs_scribe.py`: one live call on the same clip gave 18 segments and two
  speakers, all labelled `es`. That is the one-language-per-file limit above, on a clip where
  faster-whisper found about 16 s of English. Its output matches the pipeline's normalization of
  the same response, and 3,000 generated responses agree offline too.

---

## What the scripts simplify

The scripts keep the behaviour described above. Compared with the project they leave out:

| Left out | In the project |
| --- | --- |
| Batch input, run manifest, `errors.json` | A directory in; a reproducible, auditable run out |
| Options | Model size, device, beam and temperature settings, speaker-count bounds, per-file language mode, `segment` speaker assignment, Scribe's diarization threshold, timeouts, … each validated and recorded |
| Other languages | The Scribe language table covers every language Scribe lists; here only `eng` and `spa` are mapped, and other codes pass through unchanged |
| Credential lookup | Config, then environment, then a `.env` file; here the environment only |
| Runtime schema validation | Every transcript is checked before it is written |
| Tests | 549 tests, including a shared conformance suite every backend passes |
