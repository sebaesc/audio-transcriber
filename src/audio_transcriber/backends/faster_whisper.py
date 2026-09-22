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

import ctypes
import glob
import logging
import os
import site
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

#: CUDA libraries CTranslate2 loads by name, in dependency order. Installed by
#: the `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` wheels.
_CUDA_LIBRARIES = (
    "cublas/lib/libcublas.so*",
    "cublas/lib/libcublasLt.so*",
    "cudnn/lib/libcudnn.so*",
)

_logger = logging.getLogger(__name__)
_cuda_preloaded = False


def _preload_cuda_libraries() -> None:
    """Load pip-installed CUDA libraries into the process.

    CTranslate2 loads cuBLAS and cuDNN by name through the dynamic linker, which
    does not search site-packages. The usual fix is to export `LD_LIBRARY_PATH`
    before starting Python, but glibc reads that at process start, so setting it
    from inside the process is too late. Loading the objects here with
    `RTLD_GLOBAL` puts them in the global symbol namespace, where CTranslate2's
    own `dlopen` finds them already resident.

    A system-wide CUDA install needs none of this: the linker finds those
    libraries on its own and the loads below simply find nothing to do. Failures
    are logged and ignored, because the device smoke test is what decides whether
    CUDA actually works.
    """
    global _cuda_preloaded
    if _cuda_preloaded:
        return
    _cuda_preloaded = True

    for directory in site.getsitepackages():
        for pattern in _CUDA_LIBRARIES:
            for path in sorted(glob.glob(os.path.join(directory, "nvidia", pattern))):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                except OSError as error:
                    _logger.debug("could not preload %s: %s", path, error)


_DEFAULTS: dict[str, Any] = {
    "model": "base",
    "device": "auto",
    "compute_type": "default",
    "language": None,
    "beam_size": 5,
    "vad_filter": True,
    # faster-whisper defaults temperature to [0.0, 0.2, ..., 1.0] and falls back
    # up that ladder when a decode trips its quality thresholds. Above 0.0 it
    # samples, so the same audio produces different text on different runs —
    # measured at 689 vs 642 segments and WER 0.409 vs 0.441 for one recording.
    # A single 0.0 is greedy and reproducible, which the project requires. The
    # ladder is still available by passing a list.
    "temperature": 0.0,
    # Conditioning on previous text helps context, but it also carries any
    # divergence into every later segment. Exposed because it is the lever to
    # reach for when a recording falls into a repetition loop.
    "condition_on_previous_text": True,
    # Deterministic alternatives to temperature sampling for breaking repetition
    # loops. faster-whisper's own defaults, exposed so they can be tuned.
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
}

#: Options consumed when the model is loaded rather than passed to each decode.
_LOAD_TIME_OPTIONS = frozenset({"model", "device", "compute_type"})


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
    - `temperature`: decoding temperature. Defaults to 0.0, which is greedy and
      reproducible. Pass a list to restore faster-whisper's fallback ladder,
      accepting that the output is then not reproducible.
    - `condition_on_previous_text`: feed each segment the previous one as
      context. Defaults to True. Turning it off can break repetition loops.
    - `repetition_penalty`: penalize repeated tokens. Defaults to 1.0, no penalty.
    - `no_repeat_ngram_size`: forbid repeating an n-gram of this size. Defaults
      to 0, no restriction.

    An option this backend does not recognize is rejected rather than ignored,
    so the manifest cannot record a setting that was never applied.
    """

    name = "faster-whisper"
    capabilities = Capabilities(
        diarization=False,
        language_detection=True,
        word_timestamps=True,
        translation=True,
    )

    def __init__(self, **config: Any) -> None:
        unknown = sorted(set(config) - set(_DEFAULTS))
        if unknown:
            raise BackendRequirementError(
                self.name,
                f"unknown option(s): {', '.join(unknown)}.",
                f"Valid options are: {', '.join(sorted(_DEFAULTS))}.",
            )
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
        candidates = self._candidate_devices()
        if "cuda" in candidates:
            _preload_cuda_libraries()
        for device in candidates:
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
            "failure, install the runtime libraries "
            "(pip install nvidia-cublas-cu12 nvidia-cudnn-cu12) or set device=cpu.",
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
                str(audio_path), **self._decode_options()
            )
            segments = self._normalize(native_segments, info.language, info.duration)
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

    def _decode_options(self) -> dict[str, Any]:
        """Return every config option that is passed to a decode.

        Built from the config rather than listed inline, so an option cannot be
        recorded in the manifest while never reaching the decoder. That happened:
        `no_repeat_ngram_size` was accepted, recorded, and silently dropped, and
        two runs configured differently produced byte-identical output.
        """
        return {
            key: value
            for key, value in self.config.items()
            if key not in _LOAD_TIME_OPTIONS
        }

    def _normalize(
        self,
        native_segments: Any,
        language: str | None,
        duration: float | None = None,
    ) -> tuple[Segment, ...]:
        """Map faster-whisper's segments onto the shared schema.

        Whisper decodes in 30-second windows and pads short audio with silence.
        It regularly hallucinates in that padding: a 10-second recording can come
        back with segments running to 40 seconds, sometimes in another language.
        Anything starting at or after the end of the audio is therefore dropped,
        and a segment overhanging the end is clamped to it.

        It also emits empty text, segments ending before they start, and segments
        overlapping their predecessor by a few milliseconds. The schema forbids
        all of those, so they are cleaned up here rather than reaching the
        pipeline.

        Args:
            native_segments: The generator faster-whisper returns.
            language: The language the model detected or was given.
            duration: Length of the audio in seconds, used to discard
                hallucinations past its end. None disables that check.

        Returns:
            Segments in ascending, non-overlapping order, within the audio.
        """
        segments: list[Segment] = []
        previous_end = 0.0
        for native in native_segments:
            text = native.text.strip()
            if not text:
                continue

            start = max(float(native.start), previous_end)
            end = max(float(native.end), start)

            if duration is not None:
                if start >= duration:
                    continue
                end = min(end, duration)
                if end <= start:
                    continue

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
