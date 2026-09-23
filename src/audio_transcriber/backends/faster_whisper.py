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
from dataclasses import replace
from pathlib import Path
from typing import Any

import ctranslate2
import numpy as np
from faster_whisper import WhisperModel, decode_audio
from faster_whisper.version import __version__ as _FASTER_WHISPER_VERSION

from audio_transcriber.backend import Capabilities, TranscriptionBackend
from audio_transcriber.backends import diarization
from audio_transcriber.backends.chunking import (
    DEFAULT_TARGET_SECONDS,
    SAMPLE_RATE,
    speech_chunks,
)
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
    # The fallback ladder. When a decode trips Whisper's quality thresholds it is
    # re-decoded at the next temperature up. This is worth keeping: 14.5% of
    # segments on one recording fell back, 46 of them all the way to 1.0, and
    # removing the ladder cost 0.185 WER and 15 points of coverage.
    #
    # Sampling above 0.0 made runs unreproducible until `seed` was pinned below.
    "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    # Seeds CTranslate2's RNG, which is what makes the ladder reproducible: the
    # same seed yields byte-identical transcripts, a different one does not.
    #
    # The value is arbitrary and deliberately untuned. Different seeds score
    # differently on any given recording — 0.259 against 0.394 WER on one of
    # them — so choosing a seed by its score fits noise rather than improving
    # anything. Treat that spread as the measurement's noise floor.
    "seed": 0,
    # How language is decided. `file` detects once for the whole recording, which
    # is what Whisper does natively. `chunked` detects per speech chunk, so a
    # recording that switches language mid-way can say so.
    "language_detection": "file",
    # Languages detection may choose between. Restricting it removes the long
    # tail of false positives: a Spanish-English corpus has no business being
    # identified as Welsh. None allows every language the model knows.
    "languages": None,
    # Below this probability a chunk keeps the file-level language rather than
    # trusting its own detection.
    "language_threshold": 0.5,
    # Target chunk length for `chunked` detection, in seconds.
    "chunk_seconds": DEFAULT_TARGET_SECONDS,
    # Speaker diarization. Off by default: it pulls a second model, and its
    # weights are gated on the Hugging Face Hub.
    "diarization": False,
    "diarization_model": diarization.DEFAULT_MODEL,
    "hf_token": None,
    "num_speakers": None,
    "min_speakers": None,
    "max_speakers": None,
    # Conditioning on previous text helps context, but it also carries any
    # divergence into every later segment. Exposed because it is the lever to
    # reach for when a recording falls into a repetition loop.
    "condition_on_previous_text": True,
    # Deterministic alternatives to temperature sampling for breaking repetition
    # loops. faster-whisper's own defaults, exposed so they can be tuned.
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    # Beam search shape. Deterministic quality levers, as opposed to sampling.
    "best_of": 5,
    "patience": 1.0,
    "length_penalty": 1.0,
    # The thresholds that decide a decode was bad. With a temperature ladder,
    # tripping one triggers a re-decode; without it, the bad decode is kept.
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.6,
}

#: Options this backend consumes itself rather than passing to the decoder.
_PIPELINE_OPTIONS = frozenset(
    {
        "language_detection",
        "languages",
        "language_threshold",
        "chunk_seconds",
        "diarization",
        "diarization_model",
        "hf_token",
        "num_speakers",
        "min_speakers",
        "max_speakers",
    }
)

#: Options only `language_detection=chunked` reads.
_CHUNKED_ONLY_OPTIONS = frozenset({"languages", "language_threshold", "chunk_seconds"})

#: Options that are not decode keywords: they configure how the model is loaded,
#: or are applied some other way. `seed` goes to CTranslate2's RNG rather than to
#: `transcribe()`, which would reject it.
_NON_DECODE_OPTIONS = (
    frozenset({"model", "device", "compute_type", "seed"}) | _PIPELINE_OPTIONS
)


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
    - `temperature`: decoding temperature, or a list forming a fallback ladder.
      Defaults to the full ladder, which re-decodes segments that fail Whisper's
      quality thresholds. Reproducible because `seed` is pinned.
    - `language_detection`: `file` (default) detects one language for the whole
      recording, as Whisper does natively. `chunked` detects per speech chunk, so
      a bilingual recording can report both.
    - `languages`: the languages detection may choose between, e.g. `["en","es"]`.
      None allows all of them.
    - `language_threshold`: below this probability a chunk falls back to the
      file-level language. Defaults to 0.5.
    - `chunk_seconds`: target chunk length for `chunked` detection.
    - `diarization`: label segments by speaker. Off by default; needs the
      `diarization` extra and accepted Hugging Face gates.
    - `diarization_model`: the pyannote pipeline to use.
    - `hf_token`: Hugging Face token. Falls back to `HF_TOKEN` in the environment,
      and is redacted from the manifest.
    - `num_speakers`, `min_speakers`, `max_speakers`: constrain the speaker count
      when it is known.
    - `seed`: seeds CTranslate2's RNG, making sampled decoding reproducible.
      Defaults to 0. The model is rebuilt per file to apply it, costing about
      1.7 s per file for `large-v3`. `None` skips that and reuses the loaded
      model: faster for many short files, and not reproducible.
    - `condition_on_previous_text`: feed each segment the previous one as
      context. Defaults to True. Turning it off can break repetition loops.
    - `repetition_penalty`: penalize repeated tokens. Defaults to 1.0, no penalty.
    - `no_repeat_ngram_size`: forbid repeating an n-gram of this size. Defaults
      to 0, no restriction.

    An option this backend does not recognize is rejected rather than ignored,
    so the manifest cannot record a setting that was never applied. So is one
    the chosen `language_detection` mode would ignore: a forced `language` in
    chunked mode, or a chunked-only option in file mode.
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
        self._reject_contradictions(config)
        super().__init__(**{**_DEFAULTS, **config})
        self._model: WhisperModel | None = None
        self._resolved_device: str | None = None
        self._diarizer = None
        if self.config["diarization"]:
            # Declared per instance, not per class: what this backend can do
            # depends on how it was configured, and a consumer reading the
            # capability flag should see what is actually active.
            self.capabilities = replace(type(self).capabilities, diarization=True)

    def _reject_contradictions(self, config: dict[str, Any]) -> None:
        """Reject options the chosen detection mode would silently ignore.

        Checked against the options as given, before defaults are applied: a
        default is not a choice the caller made, so it cannot contradict one.

        Raises:
            BackendRequirementError: If `language` is forced in chunked mode,
                where detection would overwrite it, or a chunked-only option is
                given in file mode, which never reads it.
        """
        mode = config.get("language_detection", _DEFAULTS["language_detection"])
        if mode == "chunked" and config.get("language"):
            raise BackendRequirementError(
                self.name,
                "language is forced, but language_detection=chunked detects it "
                "per chunk and would override it.",
                "Drop language to detect per chunk, or drop "
                "language_detection=chunked to force one language.",
            )
        if mode != "chunked":
            ignored = sorted(
                key
                for key in set(config) & _CHUNKED_ONLY_OPTIONS
                if config[key] != _DEFAULTS[key]
            )
            if ignored:
                raise BackendRequirementError(
                    self.name,
                    f"{', '.join(ignored)} only apply with "
                    "language_detection=chunked, and would be ignored.",
                    "Add language_detection=chunked, or drop them.",
                )

    def effective_config(self) -> dict[str, Any]:
        """Return the config with defaults applied and resolved values filled in.

        `device: auto` is recorded as the device actually used, so a run that fell
        back to CPU is reproducible as a CPU run. The diarization library version
        is recorded too, since speaker labels are only traceable if what produced
        them is known.

        The Hugging Face token is redacted: a manifest is written to disk and
        read by anyone with the output directory.
        """
        config = dict(self.config)
        if self._resolved_device is not None:
            config["device"] = self._resolved_device
        if config.get("hf_token"):
            config["hf_token"] = "<redacted>"
        if config["diarization"]:
            config["diarization_version"] = diarization.version()
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

        if self.config["diarization"]:
            diarization.check_requirements(
                self.config["diarization_model"], self.config["hf_token"], self.name
            )

        failures: list[str] = []
        candidates = self._candidate_devices()
        if "cuda" in candidates:
            _preload_cuda_libraries()
        for device in candidates:
            try:
                model = self._build_model(device)
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

    def _build_model(self, device: str) -> WhisperModel:
        """Construct a model on `device`, seeding the RNG first.

        CTranslate2's seed binds when the model is constructed, not when it
        decodes: calling `set_random_seed` between two decodes on one model has
        no effect, which was measured. Seeding here is therefore the only point
        at which it does anything.
        """
        if self.config["seed"] is not None:
            ctranslate2.set_random_seed(self.config["seed"])
        return WhisperModel(
            self.config["model"],
            device=device,
            compute_type=self.config["compute_type"],
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

    def _model_for_file(self) -> WhisperModel:
        """Return a model to transcribe one file with.

        With a seed configured the model is rebuilt per file, because that is
        what re-seeds the RNG. It makes a file's transcript independent of how
        many files preceded it in the batch — without it, the same recording
        scored WER 0.375 alone and 0.476 as the fifth file of a run.

        Rebuilding `large-v3` from cache costs about 1.7 s against minutes of
        transcription. `seed: None` skips it and reuses the loaded model, which
        is faster for many short files and not reproducible.
        """
        if self._model is None:
            self.check_requirements()
        if self.config["seed"] is None:
            assert self._model is not None
            return self._model
        # Replacing the reference frees the previous model rather than holding
        # two copies of the weights on the device.
        self._model = None
        self._model = self._build_model(self._resolved_device or "cpu")
        return self._model

    def _allowed_languages(self) -> tuple[str, ...] | None:
        """Return the languages detection may choose between, or None for all."""
        languages = self.config["languages"]
        if not languages:
            return None
        if isinstance(languages, str):
            languages = [item.strip() for item in languages.split(",") if item.strip()]
        return tuple(languages)

    def _detect_language(
        self, model: WhisperModel, samples
    ) -> tuple[str | None, float]:
        """Detect the language of one chunk, within the allowed set.

        Taking the best allowed language rather than the global argmax matters:
        on a Spanish-English corpus the model will occasionally rank Portuguese
        or Galician first, and acting on that helps nobody.

        Returns:
            The language code and its probability, or (None, 0.0) if nothing in
            the allowed set was ranked.
        """
        _, _, probabilities = model.detect_language(audio=samples)
        allowed = self._allowed_languages()
        ranked = [
            (language, probability)
            for language, probability in probabilities
            if allowed is None or language in allowed
        ]
        if not ranked:
            return None, 0.0
        return max(ranked, key=lambda item: item[1])

    def _language_spans(
        self,
        model: WhisperModel,
        samples,
        chunks: list,
        file_language: str | None,
        threshold: float,
    ) -> list[tuple[int, int, str | None]]:
        """Detect a language per chunk, then merge neighbours that agree.

        Returns:
            (start_sample, end_sample, language) spans covering the chunks, each
            a contiguous run of audio sharing one detected language.
        """
        spans: list[tuple[int, int, str | None]] = []
        for chunk in chunks:
            language, probability = self._detect_language(
                model, samples[chunk.start : chunk.end]
            )
            if language is None or probability < threshold:
                language = file_language
            if spans and spans[-1][2] == language:
                spans[-1] = (spans[-1][0], chunk.end, language)
            else:
                spans.append((chunk.start, chunk.end, language))
        return spans

    def _transcribe_chunked(
        self, model: WhisperModel, audio_path: Path, source: str
    ) -> Transcript:
        """Transcribe by detecting the language of each speech chunk.

        Whisper decides one language per file and decodes everything under it,
        which for bilingual audio means the other language is mislabelled and
        transcribed in the wrong orthography. Detecting per chunk lets a
        recording say what it actually contains.

        Chunk-relative timings are shifted back onto the recording's own
        timeline, so segment times mean the same thing in either mode.
        """
        samples = decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
        duration = len(samples) / SAMPLE_RATE
        chunks = speech_chunks(samples, float(self.config["chunk_seconds"]))

        file_language, _ = self._detect_language(model, samples)
        threshold = float(self.config["language_threshold"])
        options = self._decode_options()
        # The chunks are already speech; running VAD again inside each one only
        # risks trimming its edges.
        options["vad_filter"] = False

        # Detect per chunk, then transcribe consecutive chunks that agree as one
        # span. Every boundary costs content — a word cut across two decodes is
        # transcribed badly in both — so the fewer boundaries the better, and
        # neighbouring chunks usually share a language. Detection stays
        # fine-grained; only decoding is coarsened.
        spans = self._language_spans(model, samples, chunks, file_language, threshold)

        segments: list[Segment] = []
        previous_end = 0.0
        for start_sample, end_sample, language in spans:
            piece = samples[start_sample:end_sample]
            span_start = start_sample / SAMPLE_RATE
            span_duration = len(piece) / SAMPLE_RATE
            native, _ = model.transcribe(piece, **{**options, "language": language})

            for segment in self._normalize(native, language, duration=span_duration):
                start = max(segment.start_time + span_start, previous_end)
                end = max(segment.end_time + span_start, start)
                if start >= duration:
                    continue
                segments.append(
                    Segment(
                        start_time=start,
                        end_time=min(end, duration),
                        text=segment.text,
                        speaker=None,
                        spoken_language=segment.spoken_language,
                    )
                )
                previous_end = segments[-1].end_time

        return Transcript(
            source=source,
            segments=tuple(segments),
            language=file_language,
            duration=duration,
            backend=self.name,
            model_version=self.model_version(),
        )

    def _diarize(self, audio_path: Path) -> list[diarization.SpeakerTurn]:
        """Return the speaker turns for one file, loading the pipeline once."""
        if self._diarizer is None:
            self._diarizer = diarization.load_pipeline(
                self.config["diarization_model"],
                diarization.find_token(self.config["hf_token"]),
            )
        return diarization.diarize(
            self._diarizer,
            audio_path,
            num_speakers=self.config["num_speakers"],
            min_speakers=self.config["min_speakers"],
            max_speakers=self.config["max_speakers"],
        )

    def _maybe_diarize(
        self, transcript: Transcript, audio_path: Path, source: str
    ) -> Transcript:
        """Attach speaker labels to a transcript's segments.

        Each segment takes the speaker holding most of its span. A segment that
        overlaps no turn keeps `None`: inventing a speaker would be worse than
        admitting there is not one.
        """
        if not self.config["diarization"]:
            return transcript
        try:
            turns = self._diarize(audio_path)
        except Exception as error:  # noqa: BLE001 - one file failing is not fatal
            raise TranscriptionError(
                source, self.name, f"diarization failed: {error}"
            ) from error
        if not turns:
            return transcript
        return replace(
            transcript,
            segments=tuple(
                replace(
                    segment,
                    speaker=diarization.speaker_for(
                        turns, segment.start_time, segment.end_time
                    ),
                )
                for segment in transcript.segments
            ),
        )

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
        model = self._model_for_file()
        if self.config["language_detection"] == "chunked":
            try:
                transcript = self._transcribe_chunked(model, audio_path, source)
                return self._maybe_diarize(transcript, audio_path, source)
            except TranscriptionError:
                raise
            except Exception as error:  # noqa: BLE001 - one file failing is not fatal
                raise TranscriptionError(source, self.name, str(error)) from error
        try:
            native_segments, info = model.transcribe(
                str(audio_path), **self._decode_options()
            )
            segments = self._normalize(native_segments, info.language, info.duration)
        except Exception as error:  # noqa: BLE001 - one file failing is not fatal
            raise TranscriptionError(source, self.name, str(error)) from error

        transcript = Transcript(
            source=source,
            segments=segments,
            language=info.language,
            duration=info.duration,
            backend=self.name,
            model_version=self.model_version(),
        )
        return self._maybe_diarize(transcript, audio_path, source)

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
            if key not in _NON_DECODE_OPTIONS
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
