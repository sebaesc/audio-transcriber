"""Transcribe one audio file with ElevenLabs Scribe v2, with speaker diarization.

A standalone snapshot of the Audio Transcriber `elevenlabs-scribe` backend: one
file, no dependency on the package. See README.md beside it for what the
pipeline does and how this script relates to it.

The audio is uploaded to ElevenLabs (api.elevenlabs.io), billed per audio
minute, and kept under the account's retention settings.

Requirements
------------
- Python 3.12
- ffmpeg on PATH (e.g. `apt install ffmpeg`)
- pip install "elevenlabs>=2.0,<3.0"
- An API key with speech-to-text access in ELEVENLABS_API_KEY
  (create one at https://elevenlabs.io/app/settings/api-keys)

Cost: about 67 credits per audio minute; a free account's 10,000 monthly
credits cover about 150 minutes. Diarization costs nothing extra.

Run
---
    export ELEVENLABS_API_KEY=...
    python transcribe_elevenlabs_scribe.py recording.mp3 > transcript.json

    python transcribe_elevenlabs_scribe.py recording.mp3 --max-speakers 2

Prints the transcript to stdout as a JSON array of segments, each with exactly
`start_time`, `end_time`, `speaker`, `spoken_language`, `text`. Progress goes to
stderr.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import wave

import httpx
from elevenlabs.client import ElevenLabs
from elevenlabs.core.api_error import ApiError

log = logging.getLogger("elevenlabs_scribe_standalone")

SAMPLE_RATE = 16_000
MODEL_ID = "scribe_v2"

# Seconds to wait on any one network operation, in practice for the transcript
# once the upload is sent. A 33-minute recording took 47 s end to end.
TIMEOUT = 900.0

# Scribe returns words, not segments, so segments are cut here: at a speaker
# change, after a sentence end, or at a pause longer than this.
PAUSE_SECONDS = 1.0
SENTENCE_ENDS = (".", "?", "!", "…")

# Scribe reports ISO 639-3 codes; the pipeline uses ISO 639-1 everywhere, as
# Whisper does. The full backend maps every language Scribe lists; this covers
# the ones the team's audio contains.
ISO_639_1 = {"eng": "en", "spa": "es"}


# --- Audio --------------------------------------------------------------------


def decode_to_wav(path: str) -> bytes:
    """Decode any audio file to mono 16 kHz 16-bit WAV, in memory, with ffmpeg.

    The pipeline uploads the same decoded WAV for every input format, so the
    service sees what the local backends see.
    """
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH. Install it: apt install ffmpeg")
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", path]
        + ["-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"],
        capture_output=True,
    )
    if result.returncode != 0:
        sys.exit(f"ffmpeg could not decode {path}: {result.stderr.decode().strip()}")
    if not result.stdout:
        sys.exit(f"ffmpeg decoded no audio from {path}")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(result.stdout)
    return buffer.getvalue()


# --- The API call -------------------------------------------------------------


def call_scribe(wav: bytes, api_key: str, max_speakers: int | None):
    """Upload the audio to Scribe v2 with diarization on, and return the response.

    The SDK retries 408, 409, 429 and 5xx twice, honouring Retry-After. What
    still fails ends the script with a message saying what it means.
    """
    request = {
        "model_id": MODEL_ID,
        "timestamps_granularity": "word",
        "tag_audio_events": False,
        "diarize": True,
        # Best effort on ElevenLabs' side: "determinism is not guaranteed".
        "seed": 0,
        "temperature": 0.0,
    }
    if max_speakers is not None:
        request["num_speakers"] = max_speakers  # an upper bound, not a count
    try:
        with httpx.Client(timeout=TIMEOUT) as http:
            client = ElevenLabs(api_key=api_key, timeout=TIMEOUT, httpx_client=http)
            return client.speech_to_text.convert(
                file=("audio.wav", wav, "audio/wav"), **request
            )
    except ApiError as error:
        sys.exit(describe(error).replace(api_key, "<redacted>"))
    except httpx.TimeoutException:
        sys.exit(f"No response from ElevenLabs within {TIMEOUT:g} s.")
    except httpx.HTTPError as error:
        sys.exit(f"Could not reach the ElevenLabs API: {error}")


def describe(error: ApiError) -> str:
    """Say what an API error means and what to do about it."""
    body = error.body if isinstance(error.body, dict) else {}
    detail = body.get("detail") if isinstance(body.get("detail"), dict) else {}
    if detail.get("code") == "quota_exceeded":
        # Sent as 401, so it has to be told apart from a rejected key.
        hint = (
            "the account is out of credits. Wait for the monthly reset, or use "
            "another key."
        )
    elif error.status_code in (401, 403):
        hint = (
            "ElevenLabs rejected the API key. Check ELEVENLABS_API_KEY, and that "
            "the key may use speech-to-text."
        )
    elif error.status_code == 429:
        hint = "rate limited, still after the SDK's retries. Try again later."
    else:
        hint = "ElevenLabs rejected the request."
    message = detail.get("message") or error.body
    return f"HTTP {error.status_code}: {hint} Detail: {message}"


# --- Normalization ------------------------------------------------------------


def segments(response, audio_seconds: float | None = None) -> list[dict]:
    """Group Scribe's word tokens into segments of the shared schema.

    Scribe reports one language per file, so every segment carries it. On
    code-switched audio that mislabels the other language's stretches.

    Spacing tokens stay inside a segment, so text reads with Scribe's own
    spacing. Segments are made ascending and non-overlapping: overlapping
    speech is pushed back to where the previous segment ended, and anything
    past the end of the audio is dropped or clamped.
    """
    code = (response.language_code or "").strip().lower() or None
    language = ISO_639_1.get(code, code)
    duration = response.audio_duration_secs
    if duration is None:
        duration = audio_seconds

    drafts: list[dict] = []
    for token in response.words:
        if token.type == "audio_event":
            continue
        if token.type == "spacing":
            if drafts:
                drafts[-1]["text"] += token.text
            continue
        current = drafts[-1] if drafts else None
        if current is None or breaks(current, token.start, token.speaker_id):
            current = {
                "speaker": token.speaker_id,
                "text": "",
                "start": None,
                "end": None,
            }
            drafts.append(current)
        current["text"] += token.text
        if token.start is not None and current["start"] is None:
            current["start"] = token.start
        if token.end is not None:
            current["end"] = (
                token.end if current["end"] is None else max(current["end"], token.end)
            )

    result: list[dict] = []
    previous_end = 0.0
    for draft in drafts:
        text = draft["text"].strip()
        if not text or draft["start"] is None or draft["end"] is None:
            continue
        start = max(draft["start"], previous_end)
        end = max(draft["end"], start)
        if duration is not None:
            if start >= duration:
                continue
            end = min(end, duration)
        result.append(
            {
                "start_time": start,
                "end_time": end,
                "speaker": draft["speaker"],
                "spoken_language": language,
                "text": text,
            }
        )
        previous_end = end
    return result


def breaks(current: dict, start: float | None, speaker: str | None) -> bool:
    """Return whether the next word starts a new segment."""
    if speaker != current["speaker"]:
        return True
    if current["text"].strip().endswith(SENTENCE_ENDS):
        return True
    return (
        start is not None
        and current["end"] is not None
        and start - current["end"] > PAUSE_SECONDS
    )


# --- Main ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("audio", help="audio or video file, any format ffmpeg reads")
    parser.add_argument(
        "--max-speakers",
        type=int,
        choices=range(1, 33),
        metavar="N",
        help="the most speakers Scribe may assign, 1 to 32 (default: up to 32)",
    )
    args = parser.parse_args()
    # This script's progress only; httpx logs every request at INFO.
    logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stderr)
    log.setLevel(logging.INFO)

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        sys.exit(
            "Set ELEVENLABS_API_KEY. Create a key with speech-to-text access at "
            "https://elevenlabs.io/app/settings/api-keys"
        )

    wav = decode_to_wav(args.audio)
    seconds = (len(wav) - 44) / (2 * SAMPLE_RATE)
    log.info("decoded %.1f s of audio; uploading to ElevenLabs Scribe v2", seconds)
    response = call_scribe(wav, api_key, args.max_speakers)
    transcript = segments(response, seconds)
    log.info(
        "language %s; %d segment(s); speakers: %s",
        response.language_code,
        len(transcript),
        ", ".join(sorted({str(s["speaker"]) for s in transcript})) or "none",
    )

    for segment in transcript:
        segment["start_time"] = round(segment["start_time"], 3)
        segment["end_time"] = round(segment["end_time"], 3)
    json.dump(transcript, sys.stdout, ensure_ascii=False, indent=2)
    print()


if __name__ == "__main__":
    main()
