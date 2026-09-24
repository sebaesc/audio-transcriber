"""The ElevenLabs Scribe v2 backend.

Transcribes through ElevenLabs' hosted speech-to-text API. It is the first hosted
backend: audio leaves the machine, is billed per audio hour, and is kept by
ElevenLabs under the account's retention settings.

Requires the optional extra and an API key:

    pip install -e ".[elevenlabs]"
    ELEVENLABS_API_KEY=... in the environment or a .env beside the project

Everything engine-specific stays in this module. The import below is at module
scope on purpose, so the registry skips this backend when the extra is not
installed rather than failing at import time.
"""

from __future__ import annotations

import json
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import httpx
from elevenlabs.client import ElevenLabs
from elevenlabs.core.api_error import ApiError
from elevenlabs.types import SpeechToTextChunkResponseModel
from elevenlabs.version import __version__ as SDK_VERSION

from audio_transcriber.backend import Capabilities, TranscriptionBackend
from audio_transcriber.backends import credentials
from audio_transcriber.errors import BackendRequirementError, TranscriptionError
from audio_transcriber.registry import register
from audio_transcriber.schema import Segment, Transcript

#: The model this backend runs. Hosted, so there is no finer version to pin than
#: the model id and the SDK that called it.
MODEL_ID = "scribe_v2"

#: Seconds httpx waits on any one network operation. The wait that matters is
#: for the response once the upload is sent, while Scribe transcribes. Measured
#: 2026-09-24: 47 s end to end for a 33 min recording (sastre8), so this leaves
#: ample headroom, and extrapolates past the API's 10 h file limit.
DEFAULT_TIMEOUT = 900.0

#: A silence longer than this between two words starts a new segment. Scribe
#: returns words, not segments, so the backend decides where segments break:
#: at a sentence end, a speaker change, or a pause. Within a phrase words sit
#: closer than this; a gap this long is a turn or a thought ending.
PAUSE_SECONDS = 1.0

#: A word ending in one of these closes its segment, so segments read as
#: sentences, the way Whisper's do.
_SENTENCE_ENDS = (".", "?", "!", "…")

#: Longest excerpt of an API error body carried into a failure message.
_ERROR_BODY_CHARS = 300

#: Where the API key is looked up when the config does not give one.
_KEY_VARIABLES = ("ELEVENLABS_API_KEY",)

#: Speaker counts the API accepts for `num_speakers`.
_MAX_SPEAKERS_RANGE = (1, 32)

#: Scribe's ISO 639-3 codes mapped to the ISO 639-1 codes faster-whisper emits,
#: so transcripts from either backend label a language the same way. Covers every
#: language Scribe v2 lists that has a 639-1 code. Asturian, Cebuano, Filipino,
#: Kabuverdianu, Luo, Northern Sotho, Umbundu and Cantonese have none, and pass
#: through as Scribe reports them.
_ISO_639_1 = {
    "afr": "af", "amh": "am", "ara": "ar", "hye": "hy", "asm": "as", "aze": "az",
    "bel": "be", "ben": "bn", "bos": "bs", "bul": "bg", "mya": "my", "cat": "ca",
    "nya": "ny", "hrv": "hr", "ces": "cs", "dan": "da", "nld": "nl", "eng": "en",
    "est": "et", "fin": "fi", "fra": "fr", "ful": "ff", "glg": "gl", "lug": "lg",
    "kat": "ka", "deu": "de", "ell": "el", "guj": "gu", "hau": "ha", "heb": "he",
    "hin": "hi", "hun": "hu", "isl": "is", "ibo": "ig", "ind": "id", "gle": "ga",
    "ita": "it", "jpn": "ja", "jav": "jv", "kan": "kn", "kaz": "kk", "khm": "km",
    "kor": "ko", "kur": "ku", "kir": "ky", "lao": "lo", "lav": "lv", "lin": "ln",
    "lit": "lt", "ltz": "lb", "mkd": "mk", "msa": "ms", "mal": "ml", "mlt": "mt",
    "zho": "zh", "mri": "mi", "mar": "mr", "mon": "mn", "nep": "ne", "nor": "no",
    "oci": "oc", "ori": "or", "pus": "ps", "fas": "fa", "pol": "pl", "por": "pt",
    "pan": "pa", "ron": "ro", "rus": "ru", "srp": "sr", "sna": "sn", "snd": "sd",
    "sin": "si", "slk": "sk", "slv": "sl", "som": "so", "spa": "es", "swa": "sw",
    "swe": "sv", "tam": "ta", "tgk": "tg", "tel": "te", "tha": "th", "tur": "tr",
    "ukr": "uk", "urd": "ur", "uzb": "uz", "vie": "vi", "cym": "cy", "wol": "wo",
    "xho": "xh", "zul": "zu",
}  # fmt: skip


def normalize_language(code: str | None) -> str | None:
    """Return Scribe's language code in the two-letter form, where one exists.

    The API documents ISO 639-3 (`eng`) but has also been shown returning 639-1
    (`en`), so both are accepted. A code with no 639-1 form passes through.
    """
    if code is None or not code.strip():
        return None
    code = code.strip().lower()
    return _ISO_639_1.get(code, code)


@dataclass
class _Draft:
    """A segment being assembled from Scribe's tokens."""

    speaker: str | None
    parts: list[str] = field(default_factory=list)
    start: float | None = None
    end: float | None = None

    def add(self, text: str, start: float | None, end: float | None) -> None:
        """Append a token, widening the span by whatever timing it has."""
        self.parts.append(text)
        if start is not None and self.start is None:
            self.start = start
        if end is not None:
            self.end = end if self.end is None else max(self.end, end)

    @property
    def text(self) -> str:
        return "".join(self.parts).strip()


_DEFAULTS: dict[str, Any] = {
    # Falls back to ELEVENLABS_API_KEY in the environment or a .env. Redacted
    # from the manifest.
    "api_key": None,
    # Force a language (ISO 639-1 or 639-3) instead of detecting one. Scribe
    # detects one language per file, never per segment.
    "language": None,
    # Speaker diarization. Off by default, matching faster-whisper.
    "diarization": False,
    # Scribe's `num_speakers` is an upper bound, not pyannote's exact count, so
    # it is exposed under the name that says so.
    "max_speakers": None,
    # Higher merges more; only valid with no speaker count set.
    "diarization_threshold": None,
    # Best effort on ElevenLabs' side: "determinism is not guaranteed".
    "seed": 0,
    # The API's own default is "usually 0"; sent explicitly so the manifest
    # records what was asked for.
    "temperature": 0.0,
    "timeout": DEFAULT_TIMEOUT,
}


@register
class ElevenLabsScribeBackend(TranscriptionBackend):
    """ElevenLabs Scribe v2, over the hosted speech-to-text API.

    Config options, all recorded in the run manifest:

    - `api_key`: ElevenLabs API key. Falls back to `ELEVENLABS_API_KEY` in the
      environment or a `.env`, and is redacted from the manifest.
    - `language`: force a language code (ISO 639-1 or 639-3) instead of
      detecting one.
    - `diarization`: label segments by speaker. Off by default.
    - `max_speakers`: the most speakers Scribe may assign, 1 to 32. An upper
      bound, unlike faster-whisper's `num_speakers`.
    - `diarization_threshold`: higher merges more speakers into one. Only
      valid with diarization on and no `max_speakers`.
    - `seed`: sent for best-effort determinism. Defaults to 0.
    - `temperature`: 0.0 to 2.0. Defaults to 0.0.
    - `timeout`: seconds to wait on any one network operation, in practice
      the wait for the transcript after the upload. Defaults to
      `DEFAULT_TIMEOUT`.

    An option this backend does not recognize is rejected rather than ignored,
    so the manifest cannot record a setting that was never applied. So is an
    option the API would reject or silently reinterpret.
    """

    name = "elevenlabs-scribe"
    capabilities = Capabilities(
        diarization=False,
        language_detection=True,
        word_timestamps=True,
        translation=False,
    )

    def __init__(self, **config: Any) -> None:
        self._reject_unsupported(config)
        super().__init__(**{**_DEFAULTS, **config})
        self._api_key: str | None = None
        # Set only by tests, to stand in for the SDK. A real client is opened
        # per file instead; see `_session`.
        self._client: Any = None
        if self.config["diarization"]:
            # Declared per instance, as faster-whisper does: a consumer reading
            # the capability flag should see what this run actually does.
            self.capabilities = replace(type(self).capabilities, diarization=True)

    def _reject_unsupported(self, config: dict[str, Any]) -> None:
        """Reject options that are unknown, out of range, or contradictory.

        Checked against the options as given, before defaults are applied.

        Raises:
            BackendRequirementError: With the option at fault and how to fix it.
        """
        speaker_counts = sorted({"num_speakers", "min_speakers"} & set(config))
        if speaker_counts:
            raise BackendRequirementError(
                self.name,
                f"{', '.join(speaker_counts)} is not supported: Scribe takes only an "
                "upper bound on the number of speakers.",
                "Use max_speakers instead.",
            )
        unknown = sorted(set(config) - set(_DEFAULTS))
        if unknown:
            raise BackendRequirementError(
                self.name,
                f"unknown option(s): {', '.join(unknown)}.",
                f"Valid options are: {', '.join(sorted(_DEFAULTS))}.",
            )

        diarize = bool(config.get("diarization"))
        max_speakers = config.get("max_speakers")
        threshold = config.get("diarization_threshold")
        for option, value in (
            ("max_speakers", max_speakers),
            ("diarization_threshold", threshold),
        ):
            if value is not None and not diarize:
                raise BackendRequirementError(
                    self.name,
                    f"{option} only applies with diarization=true, and would be "
                    "ignored.",
                    "Add diarization=true, or drop it.",
                )
        if max_speakers is not None:
            low, high = _MAX_SPEAKERS_RANGE
            if not low <= int(max_speakers) <= high:
                raise BackendRequirementError(
                    self.name,
                    f"max_speakers={max_speakers} is out of range.",
                    f"Use a value between {low} and {high}.",
                )
            if threshold is not None:
                raise BackendRequirementError(
                    self.name,
                    "diarization_threshold cannot be combined with max_speakers.",
                    "Drop one of them.",
                )
        timeout = config.get("timeout", DEFAULT_TIMEOUT)
        if timeout is None or float(timeout) <= 0:
            raise BackendRequirementError(
                self.name,
                f"timeout={timeout} must be a positive number of seconds.",
                f"Omit it to use the default of {DEFAULT_TIMEOUT:g} s.",
            )

    def effective_config(self) -> dict[str, Any]:
        """Return the config with defaults applied and the API key redacted.

        A manifest is written to disk and read by anyone with the output
        directory, so the key never reaches it.
        """
        config = dict(self.config)
        if config.get("api_key"):
            config["api_key"] = "<redacted>"
        return config

    def model_version(self) -> str:
        """Return the model id with the SDK version that called it."""
        return f"elevenlabs-{SDK_VERSION}/{MODEL_ID}"

    def check_requirements(self) -> None:
        """Verify an API key is available.

        The key is not tried against the API here. A wrong key fails every file
        with a 401 naming the key, which is the signal a systematic fault should
        give, and checking up front would spend a request on every run.

        Raises:
            BackendRequirementError: If no key is configured anywhere.
        """
        key = credentials.find_credential(self.config["api_key"], _KEY_VARIABLES)
        if not key:
            raise BackendRequirementError(
                self.name,
                "no ElevenLabs API key found.",
                "Set ELEVENLABS_API_KEY in the environment or in a .env beside the "
                "project (see .env.example), or pass --option api_key=... . "
                "Create a key at https://elevenlabs.io/app/settings/api-keys",
            )
        self._api_key = key

    def transcribe(self, audio_path: Path, source: str) -> Transcript:
        """Upload one decoded file to Scribe and normalize the result.

        The request is synchronous. The SDK retries 408, 409, 429 and 5xx twice,
        honoring `Retry-After`; whatever still fails, and any timeout or
        connection error, becomes a failure for this file only.

        Args:
            audio_path: Mono 16 kHz WAV, already decoded by the pipeline.
            source: The original input path, recorded on the transcript.

        Raises:
            TranscriptionError: If this file cannot be transcribed.
        """
        try:
            with self._session() as client, audio_path.open("rb") as audio:
                response = client.speech_to_text.convert(
                    file=(audio_path.name, audio, "audio/wav"), **self._request()
                )
        except ApiError as error:
            raise self._failure(source, self._describe(error)) from error
        except httpx.TimeoutException as error:
            raise self._failure(
                source,
                f"no response within timeout={self.config['timeout']:g} s. Raise it "
                "with --option timeout=SECONDS for long recordings.",
            ) from error
        except httpx.HTTPError as error:
            raise self._failure(
                source, f"could not reach the ElevenLabs API: {error}"
            ) from error
        return self._normalize(response, source, _wav_duration(audio_path))

    @contextmanager
    def _session(self) -> Iterator[Any]:
        """Yield an API client whose connection is closed when the file is done.

        The backend interface has no close hook, so a client kept for the whole
        run would leak its connection. One upload per file means reusing a
        connection saves nothing.
        """
        if self._client is not None:
            yield self._client
            return
        if self._api_key is None:
            self.check_requirements()
        timeout = float(self.config["timeout"])
        with httpx.Client(timeout=timeout) as http:
            yield ElevenLabs(api_key=self._api_key, timeout=timeout, httpx_client=http)

    def _request(self) -> dict[str, Any]:
        """Return the request parameters, built from the config.

        An option left as None is not sent, so the API applies its own default
        rather than receiving an explicit null.
        """
        config = self.config
        diarize = bool(config["diarization"])
        request: dict[str, Any] = {
            "model_id": MODEL_ID,
            "timestamps_granularity": "word",
            "tag_audio_events": False,
            "diarize": diarize,
        }
        optional = {
            "language_code": config["language"],
            "num_speakers": config["max_speakers"] if diarize else None,
            "diarization_threshold": (
                config["diarization_threshold"] if diarize else None
            ),
            "seed": config["seed"],
            "temperature": config["temperature"],
        }
        request.update(
            {key: value for key, value in optional.items() if value is not None}
        )
        return request

    def _describe(self, error: ApiError) -> str:
        """Say what an API error means and what to do about it."""
        status = error.status_code
        detail = _excerpt(error.body)
        if _error_field(error.body, "code") == "quota_exceeded":
            # Sent as 401, so it must be told apart from a rejected key.
            hint = (
                "the ElevenLabs account is out of credits. Wait for the quota to "
                "reset, add credits, or use another key."
            )
            detail = _excerpt(_error_field(error.body, "message")) or detail
        elif status in (401, 403):
            hint = (
                "ElevenLabs rejected the API key or its permissions. Check "
                "ELEVENLABS_API_KEY, and that the key may use speech-to-text."
            )
        elif status == 429:
            hint = (
                "rate limited or over the plan's concurrency, still after the SDK's "
                "retries. Try again later, or run fewer files at once."
            )
        elif status is not None and status >= 500:
            hint = "ElevenLabs had a server error, still after the SDK's retries."
        else:
            hint = "ElevenLabs rejected the request."
        message = f"HTTP {status}: {hint}"
        return f"{message} Detail: {detail}" if detail else message

    def _failure(self, source: str, message: str) -> TranscriptionError:
        """Build a per-file failure, with the API key scrubbed from it."""
        if self._api_key:
            message = message.replace(self._api_key, "<redacted>")
        return TranscriptionError(source, self.name, message)

    def _normalize(
        self, response: Any, source: str, audio_duration: float | None
    ) -> Transcript:
        """Map Scribe's response onto the shared schema.

        Scribe reports one language for the whole file and none per word, so
        every segment carries the file language. On code-switched audio that
        mislabels the other language's stretches; the backend doc says so.

        Args:
            response: What `speech_to_text.convert` returned.
            source: The original input path, recorded on the transcript.
            audio_duration: Length of the decoded audio, used when the response
                does not report one.

        Raises:
            TranscriptionError: If the response is not a single-channel
                transcript, which this backend never asks for.
        """
        if not isinstance(response, SpeechToTextChunkResponseModel):
            raise TranscriptionError(
                source,
                self.name,
                f"got {type(response).__name__} where a single-channel "
                "transcript was expected; multichannel and webhook responses are "
                "not supported.",
            )
        language = normalize_language(response.language_code)
        duration = response.audio_duration_secs
        if duration is None:
            duration = audio_duration
        return Transcript(
            source=source,
            segments=self._segments(response.words, language, duration),
            language=language,
            duration=duration,
            backend=self.name,
            model_version=self.model_version(),
        )

    def _segments(
        self, words: list[Any], language: str | None, duration: float | None
    ) -> tuple[Segment, ...]:
        """Group Scribe's tokens into segments.

        Spacing tokens are kept inside a segment, so the text reads with the
        spacing Scribe returned, and none is invented for languages written
        without spaces. Audio events are dropped: this backend does not request
        them, and a tag is not speech.

        Segments are made ascending and non-overlapping: overlapping speakers
        are pushed back to where the previous segment ended. A segment with no
        timed word cannot be placed and is dropped, as is anything starting at
        or past the end of the audio; one overhanging the end is clamped to it.
        """
        diarize = bool(self.config["diarization"])
        drafts: list[_Draft] = []
        current: _Draft | None = None
        for token in words:
            if token.type == "audio_event":
                continue
            if token.type == "spacing":
                if current is not None:
                    current.parts.append(token.text)
                continue
            speaker = token.speaker_id if diarize else None
            if current is None or self._breaks(current, token.start, speaker):
                current = _Draft(speaker=speaker)
                drafts.append(current)
            current.add(token.text, token.start, token.end)

        segments: list[Segment] = []
        previous_end = 0.0
        for draft in drafts:
            text = draft.text
            if not text or draft.start is None or draft.end is None:
                continue
            start = max(draft.start, previous_end)
            end = max(draft.end, start)
            if duration is not None:
                if start >= duration:
                    continue
                end = min(end, duration)
            segments.append(
                Segment(
                    start_time=start,
                    end_time=end,
                    text=text,
                    speaker=draft.speaker,
                    spoken_language=language,
                )
            )
            previous_end = end
        return tuple(segments)

    @staticmethod
    def _breaks(current: _Draft, start: float | None, speaker: str | None) -> bool:
        """Return whether the next word starts a new segment."""
        if speaker != current.speaker:
            return True
        if current.text.endswith(_SENTENCE_ENDS):
            return True
        return (
            start is not None
            and current.end is not None
            and start - current.end > PAUSE_SECONDS
        )


def _excerpt(body: Any) -> str:
    """Return a short, single-line rendering of an API error body."""
    if body is None:
        return ""
    text = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
    text = " ".join(text.split())
    if len(text) > _ERROR_BODY_CHARS:
        text = text[: _ERROR_BODY_CHARS - 1] + "…"
    return text


def _error_field(body: Any, field: str) -> Any:
    """Return `body["detail"][field]` from an API error body, or None."""
    detail = body.get("detail") if isinstance(body, dict) else None
    return detail.get(field) if isinstance(detail, dict) else None


def _wav_duration(path: Path) -> float | None:
    """Return a WAV file's length in seconds, or None if it cannot be read."""
    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getnframes() / handle.getframerate()
    except (OSError, wave.Error):
        return None
