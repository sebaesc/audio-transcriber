# The Backend Interface

A transcription engine becomes usable by implementing one interface and registering itself by
name. Pipeline code holds only this interface: it never imports an engine's library and never
branches on a backend name.

This document is the contract. Per `conductor/workflow.md`, changing the interface, the
transcript schema, or the manifest format requires approval, because every backend and every
downstream consumer depends on them.

## What you implement

Subclass `TranscriptionBackend` from `audio_transcriber.backend`, set a `name`, and implement
three methods.

```python
from pathlib import Path

from audio_transcriber.backend import Capabilities, TranscriptionBackend
from audio_transcriber.errors import BackendRequirementError, TranscriptionError
from audio_transcriber.registry import register
from audio_transcriber.schema import Segment, Transcript


@register
class ExampleBackend(TranscriptionBackend):
    """One-line description of the engine."""

    name = "example"
    capabilities = Capabilities(language_detection=True)

    def check_requirements(self) -> None:
        if not _model_is_available():
            raise BackendRequirementError(
                self.name,
                "model weights not found.",
                "Download them: example-cli pull base",
            )

    def model_version(self) -> str:
        return "example-base-1.2.0"

    def transcribe(self, audio_path: Path, source: str) -> Transcript:
        try:
            native = _engine.run(str(audio_path))
        except _engine.Error as error:
            raise TranscriptionError(source, self.name, str(error)) from error

        return Transcript(
            source=source,
            segments=tuple(
                Segment(start=item.start, end=item.end, text=item.text.strip())
                for item in native.segments
            ),
            language=native.language,
            duration=native.duration,
            backend=self.name,
            model_version=self.model_version(),
        )
```

### `name`

The string the user passes to `--backend`. It must be unique across registered backends;
registering a duplicate name raises `ConfigError`.

### `capabilities`

A `Capabilities` instance declaring what the engine supports: `diarization`,
`language_detection`, `word_timestamps`, `translation`. Declare abilities here rather than
letting the pipeline infer them from the backend's type. A pipeline feature that only one
backend can support does not belong in the pipeline.

### `check_requirements()`

Called once during up-front validation, before any file is processed, so a missing model,
credential, binary, or device fails the run immediately rather than part-way through a batch.

Raise `BackendRequirementError(backend, message, remedy)` when something is missing. The remedy
is the command or config change that fixes it — see the error guidelines in
`conductor/product-guidelines.md`.

### `model_version()`

The version identifier recorded in the run manifest, so a transcript can be traced back to what
produced it. Never a moving tag: `"latest"` makes a run unreproducible and the conformance suite
rejects it.

### `transcribe(audio_path, source)`

Transcribe one file and return a `Transcript`.

- `audio_path` is audio the pipeline has **already decoded** to mono 16-bit PCM WAV at
  `TARGET_SAMPLE_RATE` (16 kHz). Backends do not decode audio; the pipeline handles that once,
  with ffmpeg, for every engine.
- `source` is the **original** input path. Record it on the transcript, not `audio_path` — a
  transcript pointing at a deleted temporary file cannot be traced back to its input.

Raise `TranscriptionError(source, name, message)` when one file cannot be transcribed. The run
collects the failure into `errors.json` and continues with the next file; it does not abort.

### `effective_config()` (optional)

Override when the backend applies its own defaults, so the manifest records the values actually
used rather than only those the user passed. The base implementation returns a copy of the
config given to `__init__`.

## Normalization is the backend's job

The backend owns the mapping from its engine's native output to the shared schema. Nothing
engine-shaped may reach the pipeline. In practice:

- Segment times are **seconds as floats**, measured from the start of the audio
- Segments are ordered and non-overlapping; a segment may not start before the previous ended
- Segment text is stripped of leading and trailing whitespace
- `confidence` is normalized to 0.0–1.0, or left `None` when the engine reports nothing usable
- `speaker` stays `None` unless the engine diarizes
- Audio with no detected speech yields a transcript with **no segments** — that is a success,
  not an error

## Registration

Decorate the class with `@register`, or call `register(ExampleBackend)`. Add the module to
`_BUILTIN_MODULES` in `audio_transcriber/registry.py` so it is discovered on lookup.

A backend whose optional extra is not installed is skipped at import rather than raising: a
missing engine only matters when it is the one selected. Keep engine imports at module scope so
that skip works.

## Dependencies

A backend's dependencies go in `pyproject.toml` as an optional extra named after the backend,
never in the core `dependencies`. Installing the pipeline must not pull in every engine.

```toml
[project.optional-dependencies]
example = ["example-engine>=1.2,<2.0"]
```

## Conformance

Every backend must pass the shared suite:

```bash
.venv/bin/pytest -k conformance
```

The assertions live in `audio_transcriber.conformance`, so a backend developed outside this repo
can run the same checks. The suite parameterizes over every registered backend, and skips one
whose `check_requirements()` reports something missing.

## Definition of done

A backend is not complete until all of these hold — gates 3 and 6 in `conductor/workflow.md`:

1. It passes `pytest -k conformance`
2. Its dependencies are an optional extra, not core dependencies
3. It reports a pinned `model_version()`
4. It has a doc page in `docs/backends/<name>.md` covering usage, dependencies, credentials,
   hardware requirements, and known limits
