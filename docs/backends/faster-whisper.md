# Backend: `faster-whisper`

Whisper inference through [CTranslate2](https://github.com/OpenNMT/CTranslate2). This is the
reference backend: it exists to prove the interface is implementable and to give the pipeline a
real engine to run against.

| | |
| --- | --- |
| **Name** | `faster-whisper` |
| **Extra** | `.[faster-whisper]` |
| **Credentials** | None, unless diarization is enabled — then a Hugging Face token |
| **Network** | Only on first use, to download model weights |
| **Diarization** | Opt-in (`diarization=true`), via pyannote; needs the `diarization` extra |
| **Language detection** | Per file by default; per chunk with `language_detection=chunked` |
| **Word timestamps** | Yes (not surfaced by the pipeline yet) |
| **Translation** | Yes (not surfaced by the pipeline yet) |

## Install

```bash
.venv/bin/pip install -e ".[faster-whisper]"
```

This pulls in `ctranslate2`, `onnxruntime`, `av`, `tokenizers`, and `huggingface_hub`. It is not
part of the core install: the pipeline does not depend on any engine.

### For diarization

```bash
.venv/bin/pip install -e ".[faster-whisper,diarization]"
```

The `diarization` extra adds `pyannote.audio` 4.x, which brings `torch`. It is loaded only when
`diarization=true`.

pyannote's models are MIT-licensed but **gated** on the Hugging Face Hub:

1. Accept the conditions at
   [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)
   with your Hugging Face account. With `diarization_model=pyannote/speaker-diarization-3.1`,
   also accept [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0) —
   that pipeline fetches it separately.
2. Create a read token at <https://huggingface.co/settings/tokens>.
3. Export it as `HF_TOKEN`, or put `HF_TOKEN=...` in a `.env` file beside the project. An
   exported variable wins over the file.

A missing token or an unaccepted gate is reported by `check_requirements()`, before any file is
transcribed, with the pages to visit for the configured pipeline. The token is written to the
manifest as `<redacted>`.

## Usage

```bash
.venv/bin/python -m audio_transcriber --input ./audio --backend faster-whisper --out ./out
```

Bilingual audio, with speakers:

```bash
.venv/bin/python -m audio_transcriber -i ./audio -b faster-whisper --out ./out \
  --option model=large-v3 \
  --option language_detection=chunked --option 'languages=["en","es"]' \
  --option diarization=true
```

Programmatically:

```python
from audio_transcriber import registry

backend = registry.create("faster-whisper", model="small", language="es")
backend.check_requirements()
transcript = backend.transcribe(decoded_wav_path, "audio/meeting.m4a")
```

## Configuration

Pass options with `--option key=value` on the CLI, or as keywords to `registry.create`. Every
option is recorded in the run manifest, and an option the backend does not recognize is
**rejected**, not ignored — so the manifest cannot claim a setting that was never applied.

### Model and device

| Option | Default | Meaning |
| ------ | ------- | ------- |
| `model` | `base` | A size from `KNOWN_MODELS`, or a path to converted weights |
| `device` | `auto` | `auto`, `cpu`, or `cuda` |
| `compute_type` | `default` | CTranslate2 compute type, e.g. `int8`, `float16` |

### Decoding

| Option | Default | Meaning |
| ------ | ------- | ------- |
| `language` | `None` | Force a language code instead of detecting one |
| `beam_size` | `5` | Beam search width. `1` is greedy and faster |
| `vad_filter` | `True` | Drop non-speech with Silero VAD before decoding |
| `temperature` | `[0.0, 0.2, …, 1.0]` | A single temperature, or a fallback ladder: a decode that trips the thresholds below is retried one step up |
| `seed` | `0` | Seeds CTranslate2's RNG so the ladder is reproducible. `None` disables it |
| `condition_on_previous_text` | `True` | Feed each window the previous text. Turning it off can break a repetition loop |
| `repetition_penalty` | `1.0` | Penalize repeated tokens; `1.0` is none |
| `no_repeat_ngram_size` | `0` | Forbid repeating an n-gram of this size; `0` is no restriction |
| `best_of`, `patience`, `length_penalty` | `5`, `1.0`, `1.0` | Beam search shape |
| `compression_ratio_threshold` | `2.4` | Above this, a decode is judged repetitive |
| `log_prob_threshold` | `-1.0` | Below this average log-probability, a decode is judged unreliable |
| `no_speech_threshold` | `0.6` | Above this no-speech probability (with low log-probability), a window is judged silent |

**Keep the temperature ladder.** Removing it (`temperature=0.0`) was measured on the reference
corpus: 14.5% of segments on one recording fall back, some all the way to 1.0 to escape repetition
loops, and greedy-only decoding cost 0.185 WER and 15 points of coverage.

**`seed` makes runs reproducible, not better.** The same input, config and seed produce
byte-identical transcripts. A different seed produces a different, equally valid one — on one
recording seeds 42 and 7 scored WER 0.259 and 0.394. Do not tune it. Applying the seed means
rebuilding the model for each file, about 1.7 s for `large-v3`; `seed=None` skips that, at the
cost of reproducibility and of a file's transcript depending on which files preceded it.

### Language detection

| Option | Default | Meaning |
| ------ | ------- | ------- |
| `language_detection` | `file` | `file`: one language for the whole recording, as Whisper does natively. `chunked`: detect per speech chunk |
| `languages` | `None` | Languages detection may choose between, e.g. `["en","es"]` or `en,es`. `None` allows all |
| `language_threshold` | `0.5` | In `chunked` mode, a chunk whose best allowed language scores below this keeps the file-level language |
| `chunk_seconds` | `25` | In `chunked` mode, the maximum length of a detection chunk. Under Whisper's 30 s window, so a chunk is one decode |

`language` forces one language for the whole file and is **rejected** with
`language_detection=chunked`, which would otherwise overwrite it per chunk. Likewise `languages`,
`language_threshold` and `chunk_seconds` are rejected in `file` mode, which never reads them.

**How `chunked` works.** Silero VAD finds speech and groups it into chunks of at most
`chunk_seconds`, split at the best silence available. Each chunk's language is detected among
`languages`, taking the best *allowed* language rather than the model's overall first choice — on
Spanish-English audio the model sometimes ranks Portuguese or Galician first. A chunk scoring
below `language_threshold` keeps the file-level language. Consecutive chunks that agree are then
decoded as one span: every decode boundary costs content, because a word cut across two decodes
is transcribed badly in both. Segment times are shifted back onto the recording's timeline, so
they mean the same in either mode.

**The defaults are reasoned, not tuned.** At 0.5 the threshold rarely acts — 3.7% of chunks on
the evaluation corpus score below it — and raising it falls back more often to the file-level
language, which is decided from the recording's first 30 s. Shorter chunks roughly double the
decode boundaries, and each costs content. Neither was measured against alternatives by a full
evaluation.

**When to use it.** On the Spanish-English evaluation corpus, `chunked` reduced language-mix
error on all five recordings and reported both languages on every one, where `file` reported
one. It improved WER only where `file` had guessed the recording's language wrong, and cost a
little where it guessed right. It stays opt-in: use it for audio you know is bilingual. See
[`docs/evaluation.md`](../evaluation.md) and the track plan for the numbers.

### Diarization

| Option | Default | Meaning |
| ------ | ------- | ------- |
| `diarization` | `False` | Label each segment's `speaker` |
| `diarization_model` | `pyannote/speaker-diarization-community-1` | The pyannote pipeline to use |
| `hf_token` | `None` | Hugging Face token. Falls back to `HF_TOKEN` in the environment, then a `.env` file. Redacted in the manifest |
| `num_speakers` | `None` | Exact speaker count, when known |
| `min_speakers`, `max_speakers` | `None` | Bounds on the speaker count |

Each segment takes the speaker whose turns overlap most of it. A segment overlapping no turn keeps
`speaker: null` rather than being assigned a guess. Labels (`SPEAKER_00`, …) are stable within one
transcript, not across files. The pipeline runs on the device transcription resolved to. The
pyannote version is recorded in the manifest as `diarization_version`.

### How `device: auto` resolves

CTranslate2 reports a CUDA device whenever the driver is visible, even when the CUDA runtime
libraries are missing — and inference then fails on the *first segment*, which would abort a
batch part-way through. So `check_requirements()` does not trust the report: it runs a real
inference over a one-second silent buffer on each candidate device.

- `auto` tries CUDA first when a device is reported, and falls back to CPU if that inference fails
- An explicit `cuda` or `cpu` is **not** second-guessed; a failure is reported, not worked around

The device actually used is written to the manifest's `config.device`, so a run that fell back to
CPU is reproducible as a CPU run. Resolution costs about two seconds on CPU with `tiny`, once per
run.

## Models

Downloaded from the `Systran/faster-whisper-*` repositories on the Hugging Face Hub on first use,
and cached in `~/.cache/huggingface/hub`. Sizes are the repository totals:

| Model | Download | Notes |
| ----- | -------- | ----- |
| `tiny`, `tiny.en` | 78 MB | Fast, low accuracy. Good for smoke tests |
| `base`, `base.en` | 148 MB | The default |
| `small`, `small.en` | 486 MB | |
| `medium`, `medium.en` | 1.5 GB | |
| `large-v1`, `large-v2` | 3.1 GB | |
| `large-v3` | 3.1 GB | Best accuracy |

The `.en` variants are English-only and slightly more accurate on English than their multilingual
counterparts at the same size.

## Hardware

**CPU** works for every model size. Available compute types on CPU are `float32`, `int8`,
`int8_float32`, and `int16`; `int8` is the fastest and uses the least memory.

**GPU** requires an NVIDIA device plus the CUDA runtime libraries — the driver alone is not
enough. A missing `libcublas` is the usual symptom, reported at validation time by the device
check above. `float16` is the usual compute type on GPU.

Install the libraries into the venv:

```bash
.venv/bin/pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

No `LD_LIBRARY_PATH` is needed. CTranslate2 loads cuBLAS and cuDNN by name through the dynamic
linker, which does not search site-packages, and `LD_LIBRARY_PATH` is read at process start so
setting it from inside Python is too late. The backend instead preloads those objects with
`RTLD_GLOBAL` before trying CUDA, which puts them where CTranslate2's own `dlopen` finds them. A
system-wide CUDA install needs none of this and is unaffected.

### Measured VRAM

On an RTX 3060 (8 GB, WSL2), CUDA 12 libraries from the wheels above, measured with
`nvidia-smi` while transcribing:

| Model | `float16` | `int8_float16` |
| ----- | --------- | -------------- |
| `tiny` | 245 MB | |
| `base` | 373 MB | |
| `small` | 790 MB | |
| `medium` | 2.0 GB | |
| `large-v3` | 3.8 GB | 1.9 GB |

An 8 GB card runs `large-v3` at `float16` with room to spare. Use `int8_float16` to halve the
memory when sharing the GPU with something else.

**With diarization**, pyannote shares the device: `large-v3` plus `speaker-diarization-community-1`
peaked at 6.7 GB on a full 21.8-minute recording. That fits an 8 GB card; on a smaller one, use
`int8_float16` or a smaller Whisper model.

### Measured throughput

`large-v3`, RTX 3060, over the 150 minutes of real bilingual conversation in the evaluation
corpus (five recordings, 22–33 min each):

| Mode | Wall-clock | Realtime factor | Per recording |
| ---- | ---------- | --------------- | ------------- |
| `language_detection=file` | 20.5 min | 7.3x | 5.5x–21.9x |
| `language_detection=chunked` | 23.0 min | 6.5x | 5.4x–11.0x |
| Diarization alone, CUDA | — | 22.9x | measured on one 21.8-min recording |
| Diarization alone, CPU | — | 1.8x | same recording |

- **`chunked` costs about 12% overall.** The per-chunk detection pass took 136 s summed across
  the corpus at 25 s chunks — most of the difference. Per recording it varies both ways: a
  recording that decodes more cleanly under the right language can come out faster.
- **Diarization on GPU adds about 4% of the audio's duration** — 57 s for 21.8 minutes. On CPU it
  took 724 s for the same recording, longer than transcribing it, so a GPU matters more for
  diarization than for anything else here. Speaker turns were identical on both devices.
- **Silence is cheap.** `maria20` is mostly silence, which VAD skips, so it runs 21x realtime in
  `file` mode.

**Memory** scales with model size. Plan for roughly the download size in RAM with `int8`, more
with `float32`.

**Measured on this project's development machine** (WSL2, CPU, `tiny`, `int8` via `default`):
5 seconds of 16 kHz audio transcribed in 0.28 s, about 18x realtime. Treat it as an order of
magnitude, not a benchmark — it scales down substantially with larger models.

## Known limits

- **Diarization quality is unmeasured.** It runs end to end and labels every segment it can, but
  no evaluation corpus with speaker labels was available for code-switched speech, so there is no
  accuracy figure for this project's audio. pyannote's published benchmarks describe
  `speaker-diarization-3.1`, not the `community-1` pipeline used by default, and are not claimed
  here.
- **Diarization needs a gated model and a token.** See [For diarization](#for-diarization).
- **One language per file, by default.** In `file` mode the model detects a single language and
  every segment carries it, so `spoken_language` will not vary even when the audio switches. Use
  `language_detection=chunked` for bilingual audio.
- **No switching below chunk granularity.** In `chunked` mode each chunk — up to `chunk_seconds`
  of speech, several utterances — gets one language. A switch mid-sentence, or between two
  utterances in one chunk, is labelled and decoded as whichever language dominated the chunk. On
  the evaluation corpus at least 5.7% of utterances, holding 8.7% of words, mix both languages.
- **`chunked` is not a free win.** On a recording whose language `file` mode already guessed
  right, it adds overhead and can cost a little WER. That is why it is opt-in.
- **Word timestamps and translation are not exposed.** The engine supports both and the
  capability flags say so, but the pipeline has no config for either yet.
- **Hallucination on silence and noise.** Whisper can emit plausible text for audio containing no
  speech. `vad_filter` is on by default because it removes most of this; turning it off makes the
  problem noticeably worse.
- **First load of a large model is slow.** `large-v3` took about 70 s to download and load on
  first use; subsequent loads come from cache in a few seconds.
- **First run is slow and needs network.** Weights download on first use. In an offline or
  air-gapped environment, pre-populate the Hugging Face cache or pass a path to converted weights
  as `model`.
- **Timestamps are approximate.** Segment boundaries come from the decoder and drift on long
  audio. Overlaps of a few milliseconds are clamped during normalization. Timestamps also collapse
  at times into uniform one-second slots whatever the segment's length; the text is usually right
  and the times are not.
- **Implausibly dense segments are dropped.** A segment of four or more words claiming more than
  50 words per second is a hallucination squashed by clamping — observed as a stock YouTube outro
  in the last 15 ms of a recording — and is discarded during normalization.
- **Repeated short segments.** Whisper occasionally emits a segment of one to three words at the
  minimum 20 ms duration, repeating its neighbour ("No." seven times in a row). Too short for the
  density check to judge; they reach the transcript.
- **Rate limits.** Unauthenticated Hub downloads are rate-limited; set `HF_TOKEN` if you hit them.

## Conformance

```bash
.venv/bin/pytest -k conformance
```

Passing as of 2026-09-23 with `faster-whisper` 1.2.1, `ctranslate2` 4.8.2 and `pyannote.audio` 4.0.7, in four
configurations: the default, `language_detection=chunked`, `diarization=true`, and both together.
