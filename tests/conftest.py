"""Shared test fixtures.

The audio fixture is generated with the standard library rather than committed as
a binary, so the suite runs without ffmpeg and without a large file in git.
"""

import math
import struct
import wave
from pathlib import Path

import pytest

from audio_transcriber.backend import TARGET_SAMPLE_RATE

pytest_plugins = ["tests.api_fixtures"]


def write_wav(path: Path, seconds: float = 1.0, frequency: float = 440.0) -> Path:
    """Write a mono 16-bit PCM WAV tone at the pipeline's target sample rate.

    Args:
        path: Where to write the file.
        seconds: Length of the tone.
        frequency: Tone frequency in Hz. Zero writes silence.

    Returns:
        The path written, for convenience.
    """
    frame_count = int(TARGET_SAMPLE_RATE * seconds)
    amplitude = 16000 if frequency else 0
    frames = bytearray()
    for index in range(frame_count):
        value = int(
            amplitude * math.sin(2 * math.pi * frequency * index / TARGET_SAMPLE_RATE)
        )
        frames += struct.pack("<h", value)

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(TARGET_SAMPLE_RATE)
        handle.writeframes(bytes(frames))
    return path


@pytest.fixture
def wav_file(tmp_path: Path) -> Path:
    """A one-second mono 16 kHz WAV tone."""
    return write_wav(tmp_path / "sample.wav")


@pytest.fixture
def silent_wav_file(tmp_path: Path) -> Path:
    """A one-second silent mono 16 kHz WAV, for the no-speech case."""
    return write_wav(tmp_path / "silence.wav", frequency=0.0)
