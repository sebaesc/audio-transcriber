# Backend: `faster-whisper`

Whisper inference through [CTranslate2](https://github.com/OpenNMT/CTranslate2). This is the
reference backend: it exists to prove the interface is implementable and to give the pipeline a
real engine to run against.

| | |
| --- | --- |
| **Name** | `faster-whisper` |
| **Extra** | `.[faster-whisper]` |
| **Credentials** | None |
| **Network** | Only on first use, to download model weights |
| **Diarization** | No |
| **Language detection** | Yes |
| **Word timestamps** | Yes (not surfaced by the pipeline yet) |
| **Translation** | Yes (not surfaced by the pipeline yet) |

## Install

```bash
.venv/bin/pip install -e ".[faster-whisper]"
```

This pulls in `ctranslate2`, `onnxruntime`, `av`, `tokenizers`, and `huggingface_hub`. It is not
part of the core install: the pipeline does not depend on any engine.

## Usage

```bash
.venv/bin/python -m audio_transcriber --input ./audio --backend faster-whisper --out ./out
```

Programmatically:

```python
from audio_transcriber import registry

backend = registry.create("faster-whisper", model="small", language="es")
backend.check_requirements()
transcript = backend.transcribe(decoded_wav_path, "audio/meeting.m4a")
```

## Configuration

Every option is recorded in the run manifest.

| Option | Default | Meaning |
| ------ | ------- | ------- |
| `model` | `base` | A size from `KNOWN_MODELS`, or a path to converted weights |
| `device` | `auto` | `auto`, `cpu`, or `cuda` |
| `compute_type` | `default` | CTranslate2 compute type, e.g. `int8`, `float16` |
| `language` | `None` | Force a language code instead of detecting one |
| `beam_size` | `5` | Beam search width. `1` is greedy and faster |
| `vad_filter` | `True` | Drop non-speech with Silero VAD before decoding |

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

Throughput is not tabulated here on purpose: a meaningful figure needs real speech, and the
project has no licensed audio fixture yet. Expect roughly an order of magnitude over CPU for the
same model.

**Memory** scales with model size. Plan for roughly the download size in RAM with `int8`, more
with `float32`.

**Measured on this project's development machine** (WSL2, CPU, `tiny`, `int8` via `default`):
5 seconds of 16 kHz audio transcribed in 0.28 s, about 18x realtime. Treat it as an order of
magnitude, not a benchmark — it scales down substantially with larger models.

## Known limits

- **No diarization.** `speaker` is always `null`. Pair with a diarization pass, or use a backend
  that provides it, if you need speaker labels.
- **One language per file.** The model detects a single language and every segment carries it, so
  `spoken_language` will not vary within a transcript even when the audio switches language.
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
  audio. Overlaps of a few milliseconds are clamped during normalization.
- **Rate limits.** Unauthenticated Hub downloads are rate-limited; set `HF_TOKEN` if you hit them.

## Conformance

```bash
.venv/bin/pytest -k conformance
```

Passing as of 2026-09-22 with `faster-whisper` 1.2.1 and `ctranslate2` 4.8.2.
