"""The faster-whisper backend.

Wraps CTranslate2's Whisper implementation. This is the reference backend: it
exists to prove the interface is implementable and to give the pipeline something
real to run against.

Requires the optional extra:

    pip install -e ".[faster-whisper]"

Everything engine-specific stays in this module. The import below is at module
scope on purpose, so the registry skips this backend when the extra is not
installed rather than failing at import time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import ctranslate2
import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.version import __version__ as _FASTER_WHISPER_VERSION

from audio_transcriber.backend import Capabilities, TranscriptionBackend
from audio_transcriber.errors import BackendRequirementError, TranscriptionError
from audio_transcriber.registry import register
from audio_transcriber.schema import Segment, Transcript

#: Model sizes CTranslate2 publishes converted weights for.
KNOWN_MODELS = (
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
)

#: Length of the silent buffer the device smoke test decodes, in samples.
_SMOKE_TEST_SAMPLES = 16_000

_DEFAULTS: dict[str, Any] = {
    "model": "base",
    "device": "auto",
    "compute_type": "default",
    "language": None,
    "beam_size": 5,
    "vad_filter": True,
}


@register
class FasterWhisperBackend(TranscriptionBackend):
    """Whisper inference through CTranslate2.

    Config options, all recorded in the run manifest:

    - `model`: a size from `KNOWN_MODELS`, or a path to converted weights.
      Defaults to `base`.
    - `device`: `auto`, `cpu`, or `cuda`. Defaults to `auto`.
    - `compute_type`: CTranslate2 compute type, e.g. `int8`, `float16`.
      Defaults to `default`, which lets CTranslate2 choose for the device.
    - `language`: force a language code instead of detecting one.
    - `beam_size`: beam search width. Defaults to 5.
    - `vad_filter`: drop non-speech with Silero VAD before decoding.
      Defaults to True.
    """

    name = "faster-whisper"
    capabilities = Capabilities(
        diarization=False,
        language_detection=True,
        word_timestamps=True,
        translation=True,
    )

    def __init__(self, **config: Any) -> None:
        super().__init__(**{**_DEFAULTS, **config})
        self._model: WhisperModel | None = None
        self._resolved_device: str | None = None

    def effective_config(self) -> dict[str, Any]:
        """Return the config with defaults applied and `device` resolved.

        `device: auto` is recorded as the device actually used, so a run that fell
        back to CPU is reproducible as a CPU run.
        """
        config = dict(self.config)
        if self._resolved_device is not None:
            config["device"] = self._resolved_device
        return config

    def model_version(self) -> str:
        """Return a pinned identifier: the library version and the model in use.

        Both halves matter for reproducibility. The same weights decoded by a
        different CTranslate2 release can produce different text.
        """
        return f"faster-whisper-{_FASTER_WHISPER_VERSION}/{self.config['model']}"

    def check_requirements(self) -> None:
        """Verify the model loads and the device actually runs inference.

        A device that is merely present is not enough. CTranslate2 reports a CUDA
        device whenever the driver is visible, but inference still fails if the
        CUDA runtime libraries are missing — and it fails lazily, on the first
        segment, which would abort part-way through a batch. So the check runs a
        real inference over a short silent buffer.

        With `device: auto`, CUDA is tried first and CPU is the fallback. With an
        explicit device, a failure is reported rather than worked around.

        Raises:
            BackendRequirementError: If the model cannot be loaded, or no
                candidate device can run inference. The message says how to fix it.
        """
        model_name = self.config["model"]
        if model_name not in KNOWN_MODELS and not Path(model_name).exists():
            raise BackendRequirementError(
                self.name,
                f"unknown model {model_name!r}.",
                f"Use one of: {', '.join(KNOWN_MODELS)}, "
                "or a path to converted weights.",
            )

        failures: list[str] = []
        for device in self._candidate_devices():
            try:
                model = WhisperModel(
                    model_name,
                    device=device,
                    compute_type=self.config["compute_type"],
                )
                self._smoke_test(model)
            except Exception as error:  # noqa: BLE001 - collected and reported below
                failures.append(f"{device}: {error}")
                continue
            self._model = model
            self._resolved_device = device
            return

        raise BackendRequirementError(
            self.name,
            f"could not run model {model_name!r} on any device. " + "; ".join(failures),
            "Weights download on first use and need network access. For a CUDA "
            "failure, install the CUDA runtime libraries or set device=cpu.",
        )

    def _candidate_devices(self) -> tuple[str, ...]:
        """Return the devices to try, in order, for the configured `device`."""
        requested = self.config["device"]
        if requested != "auto":
            return (requested,)
        if ctranslate2.get_cuda_device_count() > 0:
            return ("cuda", "cpu")
        return ("cpu",)

    def _smoke_test(self, model: WhisperModel) -> None:
        """Run one short inference to prove the device works.

        faster-whisper returns a lazy generator, so the segments are consumed
        here: without that, nothing is computed and a broken device looks healthy.
        """
        silence = np.zeros(_SMOKE_TEST_SAMPLES, dtype=np.float32)
        segments, _ = model.transcribe(silence, beam_size=1, vad_filter=False)
        list(segments)

    def _load_model(self) -> WhisperModel:
        """Return the loaded model, resolving requirements first if needed.

        The pipeline always validates before a run, so this normally returns the
        model `check_requirements` already loaded. It self-heals for a caller that
        transcribes without validating, such as a test.
        """
        if self._model is None:
            self.check_requirements()
        if self._model is None:  # pragma: no cover - check_requirements raises
            raise BackendRequirementError(
                self.name, "model was not loaded.", "Run check_requirements first."
            )
        return self._model

    def transcribe(self, audio_path: Path, source: str) -> Transcript:
        """Transcribe one decoded file and normalize the result.

        Args:
            audio_path: Mono 16 kHz WAV, already decoded by the pipeline.
            source: The original input path, recorded on the transcript.

        Returns:
            A transcript conforming to the shared schema.

        Raises:
            TranscriptionError: If this file cannot be transcribed.
        """
        model = self._load_model()
        try:
            native_segments, info = model.transcribe(
                str(audio_path),
                language=self.config["language"],
                beam_size=self.config["beam_size"],
                vad_filter=self.config["vad_filter"],
            )
            segments = self._normalize(native_segments, info.language)
        except Exception as error:  # noqa: BLE001 - one file failing is not fatal
            raise TranscriptionError(source, self.name, str(error)) from error

        return Transcript(
            source=source,
            segments=segments,
            language=info.language,
            duration=info.duration,
            backend=self.name,
            model_version=self.model_version(),
        )

    def _normalize(
        self, native_segments: Any, language: str | None
    ) -> tuple[Segment, ...]:
        """Map faster-whisper's segments onto the shared schema.

        Whisper occasionally emits empty text, a segment ending before it starts,
        or one overlapping its predecessor by a few milliseconds. The schema
        forbids all three, so they are cleaned up here rather than reaching the
        pipeline.

        Args:
            native_segments: The generator faster-whisper returns.
            language: The language the model detected or was given.

        Returns:
            Segments in ascending, non-overlapping order.
        """
        segments: list[Segment] = []
        previous_end = 0.0
        for native in native_segments:
            text = native.text.strip()
            if not text:
                continue
            start = max(float(native.start), previous_end)
            end = max(float(native.end), start)
            segments.append(
                Segment(
                    start_time=start,
                    end_time=end,
                    text=text,
                    speaker=None,
                    spoken_language=language,
                )
            )
            previous_end = end
        return tuple(segments)
