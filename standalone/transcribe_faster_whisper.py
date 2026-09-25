"""Transcribe one audio file with faster-whisper, with speaker diarization.

A standalone snapshot of the Audio Transcriber `faster-whisper` backend: one
file, no dependency on the package. See README.md beside it for what the
pipeline does and how this script relates to it.

Requirements
------------
- Python 3.12
- ffmpeg on PATH (e.g. `apt install ffmpeg`)
- pip install "faster-whisper>=1.0,<2.0" "pyannote.audio>=4.0,<5.0"
- For diarization: a Hugging Face token in HF_TOKEN, after accepting the model's
  conditions at https://huggingface.co/pyannote/speaker-diarization-community-1
- Optional GPU: an NVIDIA driver, plus
  pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
  Without a GPU it runs on CPU, which is much slower for large-v3.

Run
---
    export HF_TOKEN=hf_...
    python transcribe_faster_whisper.py recording.mp3 > transcript.json

    python transcribe_faster_whisper.py recording.mp3 --no-diarization   # no HF_TOKEN needed

Prints the transcript to stdout as a JSON array of segments, each with exactly
`start_time`, `end_time`, `speaker`, `spoken_language`, `text`. Progress goes to
stderr. Model weights (about 3 GB for large-v3) download on the first run.
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import json
import logging
import os
import shutil
import site
import subprocess
import sys
import warnings

import ctranslate2
import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps

log = logging.getLogger("faster_whisper_standalone")

SAMPLE_RATE = 16_000
MODEL = "large-v3"
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"

# The team's audio is English with Spanish speakers. Restricting detection to
# these stops the model ranking Portuguese or Galician first on Spanish.
LANGUAGES = ("en", "es")
# Below this probability a chunk keeps the file-level language.
LANGUAGE_THRESHOLD = 0.5
# Longest chunk language is detected on. 15 s beat 25 s on 4 of 5 eval recordings.
CHUNK_SECONDS = 15.0
# A chunk shorter than this is merged into its neighbour: too little to detect on.
MIN_CHUNK_SECONDS = 1.0

# Whisper's decode settings, as the backend sets them. The temperature ladder
# re-decodes segments that fail Whisper's quality checks; the seed makes the
# sampling it falls back to reproducible.
DECODE_OPTIONS = {
    "beam_size": 5,
    "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "condition_on_previous_text": True,
    # Chunks are already speech; VAD inside them would only trim their edges.
    "vad_filter": False,
}
SEED = 0

# A segment of 4+ words packed faster than 50 words/s is a hallucination that
# clamping squashed into a sliver of time, not speech.
MAX_WORDS_PER_SECOND = 50.0
DENSITY_MIN_WORDS = 4


# --- Audio --------------------------------------------------------------------


def decode_audio(path: str) -> np.ndarray:
    """Decode any audio file to mono 16 kHz float32 samples with ffmpeg.

    Goes through 16-bit PCM, as the pipeline's decode does. Detection sits near
    its threshold on some chunks, so float32 straight from ffmpeg, which differs
    slightly, can flip a chunk's language.
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
    samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768
    if samples.size == 0:
        sys.exit(f"ffmpeg decoded no audio from {path}")
    return samples


# --- Model --------------------------------------------------------------------


def preload_cuda_libraries() -> None:
    """Load pip-installed cuBLAS/cuDNN so CTranslate2 finds them.

    The dynamic linker does not search site-packages, and LD_LIBRARY_PATH is
    read only at process start. Loading them here with RTLD_GLOBAL puts them
    where CTranslate2's own dlopen finds them. A system CUDA install needs none
    of this, and then nothing is found to load.
    """
    patterns = (
        "cublas/lib/libcublas.so*",
        "cublas/lib/libcublasLt.so*",
        "cudnn/lib/libcudnn.so*",
    )
    for directory in site.getsitepackages():
        for pattern in patterns:
            for path in sorted(glob.glob(os.path.join(directory, "nvidia", pattern))):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def load_model() -> tuple[WhisperModel, str]:
    """Load the model on CUDA if it really works there, else on CPU.

    A visible CUDA device is not proof: inference fails lazily if the runtime
    libraries are missing. So each device is tried with a real decode.
    """
    devices = ["cuda", "cpu"] if ctranslate2.get_cuda_device_count() > 0 else ["cpu"]
    if "cuda" in devices:
        preload_cuda_libraries()
    for device in devices:
        try:
            # The seed binds when the model is built, not when it decodes.
            ctranslate2.set_random_seed(SEED)
            model = WhisperModel(MODEL, device=device, compute_type="default")
            segments, _ = model.transcribe(
                np.zeros(SAMPLE_RATE, dtype=np.float32), beam_size=1
            )
            list(segments)  # the generator is lazy; consume it to run inference
        except Exception as error:  # noqa: BLE001 - fall back to the next device
            log.warning("could not run %s on %s: %s", MODEL, device, error)
            continue
        log.info("loaded %s on %s", MODEL, device)
        return model, device
    sys.exit(f"could not run {MODEL} on any device; see the warnings above")


# --- Language, per chunk ------------------------------------------------------


def speech_chunks(samples: np.ndarray) -> list[tuple[int, int]]:
    """Split speech into chunks of at most CHUNK_SECONDS, as sample ranges.

    Silero VAD finds speech and, given a maximum length, splits long regions at
    the best silence it can find. Short neighbours are then grouped up to the
    maximum, and fragments too short to detect a language on are merged back.
    """
    target = int(CHUNK_SECONDS * SAMPLE_RATE)
    minimum = int(MIN_CHUNK_SECONDS * SAMPLE_RATE)
    regions = get_speech_timestamps(
        samples, VadOptions(max_speech_duration_s=CHUNK_SECONDS)
    )

    pieces: list[tuple[int, int]] = []
    for region in regions:
        start, end = region["start"], region["end"]
        while end - start > target:
            pieces.append((start, start + target))
            start += target
        if end > start:
            pieces.append((start, end))

    chunks: list[tuple[int, int]] = []
    for start, end in pieces:
        if chunks and end - chunks[-1][0] <= target:
            chunks[-1] = (chunks[-1][0], end)
        else:
            chunks.append((start, end))

    merged: list[tuple[int, int]] = []
    for start, end in chunks:
        if merged and end - start < minimum and end - merged[-1][0] <= target:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def detect_language(model: WhisperModel, samples: np.ndarray) -> tuple[str, float]:
    """Return the most likely allowed language and its probability."""
    _, _, probabilities = model.detect_language(audio=samples)
    return max(
        ((code, p) for code, p in probabilities if code in LANGUAGES),
        key=lambda item: item[1],
        default=(LANGUAGES[0], 0.0),
    )


def language_spans(
    model: WhisperModel, samples: np.ndarray
) -> tuple[str, list[tuple[int, int, str]]]:
    """Detect a language per chunk and merge neighbours that agree.

    Returns the file-level language and (start, end, language) spans. Decoding
    each span as one piece keeps decode boundaries, which cost words, few.
    """
    file_language, _ = detect_language(model, samples)
    spans: list[tuple[int, int, str]] = []
    for start, end in speech_chunks(samples):
        language, probability = detect_language(model, samples[start:end])
        if probability < LANGUAGE_THRESHOLD:
            language = file_language
        if spans and spans[-1][2] == language:
            spans[-1] = (spans[-1][0], end, language)
        else:
            spans.append((start, end, language))
    return file_language, spans


# --- Transcription ------------------------------------------------------------


def transcribe(model: WhisperModel, samples: np.ndarray, with_words: bool) -> list:
    """Transcribe span by span, each decoded in the language detected for it.

    Returns segments as dicts, each with its words as (start, end, text) when
    `with_words` is set, all on the recording's own timeline.
    """
    duration = len(samples) / SAMPLE_RATE
    file_language, spans = language_spans(model, samples)
    log.info(
        "file language %s; %d language span(s): %s",
        file_language,
        len(spans),
        " ".join(language for _, _, language in spans),
    )

    segments: list[dict] = []
    previous_end = 0.0
    for start_sample, end_sample, language in spans:
        offset = start_sample / SAMPLE_RATE
        native, _ = model.transcribe(
            samples[start_sample:end_sample],
            language=language,
            word_timestamps=with_words,
            **DECODE_OPTIONS,
        )
        for piece in native:
            text = piece.text.strip()
            # Whisper pads short audio with silence and hallucinates into it:
            # drop what starts past the end, clamp what overhangs it, and keep
            # segments ascending and non-overlapping.
            start = max(piece.start + offset, previous_end)
            end = min(max(piece.end + offset, start), duration)
            if not text or start >= duration or end <= start:
                continue
            words = len(text.split())
            if words >= DENSITY_MIN_WORDS and words > MAX_WORDS_PER_SECOND * (
                end - start
            ):
                continue
            segments.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "speaker": None,
                    "spoken_language": language,
                    "text": text,
                    "words": clamp_words(piece.words or [], offset, start, end),
                }
            )
            previous_end = end
    return segments


def clamp_words(words: list, offset: float, start: float, end: float) -> list:
    """Move a segment's words onto the recording's timeline, inside the segment."""
    clamped = []
    for word in words:
        word_start = word.start + offset
        if word_start > end:
            continue
        word_start = min(max(word_start, start), end)
        word_end = min(max(word.end + offset, word_start), end)
        clamped.append((word_start, word_end, word.word))
    return clamped


# --- Diarization --------------------------------------------------------------


def hf_token() -> str:
    """Return HF_TOKEN, or exit saying how to get one."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit(
            "Diarization needs HF_TOKEN. Accept the conditions at "
            f"https://huggingface.co/{DIARIZATION_MODEL}, create a token at "
            "https://huggingface.co/settings/tokens, and export it as HF_TOKEN. "
            "Or pass --no-diarization."
        )
    return token


def diarize(samples: np.ndarray, device: str) -> list[tuple[float, float, str]]:
    """Return (start, end, speaker) turns from pyannote, in time order."""
    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=hf_token())
    if pipeline is None:  # what an unaccepted gate looks like
        sys.exit(
            f"Hugging Face returned no pipeline for {DIARIZATION_MODEL}: accept "
            f"its conditions at https://huggingface.co/{DIARIZATION_MODEL}"
        )
    pipeline.to(torch.device(device))
    waveform = {
        "waveform": torch.from_numpy(samples.copy()).unsqueeze(0),
        "sample_rate": SAMPLE_RATE,
    }
    # pyannote warns on every run that it disabled TF32, and with little speech
    # its clustering averages empty sets and numpy warns. Neither needs action.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        output = pipeline(waveform)
    turns = [
        (float(turn["start"]), float(turn["end"]), str(turn["speaker"]))
        for turn in output.serialize().get("diarization", [])
    ]
    return sorted(turns)


def speaker_for(turns: list, start: float, end: float) -> str | None:
    """Return the speaker holding most of a span, or None if none overlaps it."""
    best, best_overlap = None, 0.0
    for turn_start, turn_end, speaker in turns:
        if turn_start >= end:
            break
        overlap = min(turn_end, end) - max(turn_start, start)
        if overlap > best_overlap:
            best, best_overlap = speaker, overlap
    if best is None and end == start:
        # Whisper collapses some words to an instant: take the turn around it.
        for turn_start, turn_end, speaker in turns:
            if turn_start <= start < turn_end:
                return speaker
    return best


def word_speakers(turns: list, words: list) -> list[str | None]:
    """Give each word a speaker, then fold one-word flips back.

    A word in a gap between turns takes its neighbour's speaker. A single word
    flanked on both sides by one other speaker is folded into that speaker: word
    timings drift by tens of milliseconds, so a lone flip at a turn boundary is
    more often timing error than a one-word turn.
    """
    found = [speaker_for(turns, start, end) for start, end, _ in words]
    known = next((speaker for speaker in found if speaker), None)
    filled = []
    for speaker in found:
        known = speaker or known
        filled.append(known)
    smoothed = list(filled)
    for index in range(1, len(filled) - 1):
        before, after = filled[index - 1], filled[index + 1]
        if before == after != filled[index]:
            smoothed[index] = before
    return smoothed


def assign_speakers(segments: list, turns: list) -> list:
    """Label each segment by speaker, splitting it where the speaker changes."""
    labelled = []
    for segment in segments:
        words = [word for word in segment["words"] if word[2].strip()]
        speakers = word_speakers(turns, words)
        if not speakers or speakers[0] is None:
            speaker = speaker_for(turns, segment["start_time"], segment["end_time"])
            labelled.append({**segment, "speaker": speaker})
            continue

        groups: list[tuple[str, list]] = []
        for word, speaker in zip(words, speakers, strict=True):
            if groups and groups[-1][0] == speaker:
                groups[-1][1].append(word)
            else:
                groups.append((speaker, [word]))
        if len(groups) == 1:
            labelled.append({**segment, "speaker": groups[0][0]})
            continue

        # Pieces are contiguous: each ends where the next speaker's first word
        # starts, and together they cover exactly the segment's span.
        start = segment["start_time"]
        for index, (speaker, group) in enumerate(groups):
            if index + 1 < len(groups):
                end = min(max(groups[index + 1][1][0][0], start), segment["end_time"])
            else:
                end = segment["end_time"]
            labelled.append(
                {
                    **segment,
                    "start_time": start,
                    "end_time": end,
                    "speaker": speaker,
                    "text": "".join(word[2] for word in group).strip(),
                }
            )
            start = end
    return labelled


# --- Main ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("audio", help="audio or video file, any format ffmpeg reads")
    parser.add_argument(
        "--no-diarization",
        action="store_true",
        help="skip speaker diarization; speaker is null and HF_TOKEN is not needed",
    )
    args = parser.parse_args()
    # This script's progress only; libraries log HTTP requests at INFO.
    logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stderr)
    log.setLevel(logging.INFO)

    diarization = not args.no_diarization
    if diarization:
        hf_token()  # checked before the long transcription, not after it

    samples = decode_audio(args.audio)
    log.info("decoded %.1f s of audio", len(samples) / SAMPLE_RATE)
    model, device = load_model()
    segments = transcribe(model, samples, with_words=diarization)

    if diarization:
        # Free the GPU for pyannote. Dropping the reference is not enough to
        # release the weights, and two large models at once can page to RAM.
        model.model.unload_model()
        log.info("diarizing on %s", device)
        turns = diarize(samples, device)
        log.info("%d speaker turn(s)", len(turns))
        if turns:
            segments = assign_speakers(segments, turns)

    keys = ("start_time", "end_time", "speaker", "spoken_language", "text")
    transcript = [
        {key: round(s[key], 3) if key.endswith("_time") else s[key] for key in keys}
        for s in segments
    ]
    json.dump(transcript, sys.stdout, ensure_ascii=False, indent=2)
    print()


if __name__ == "__main__":
    main()
