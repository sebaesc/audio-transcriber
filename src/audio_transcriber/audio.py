"""Audio decoding, via ffmpeg.

The pipeline decodes once, for every backend, so no engine has to deal with
container formats or sample rates. Backends receive mono 16-bit PCM WAV at
`TARGET_SAMPLE_RATE` and nothing else.

ffmpeg is a system dependency, invoked as a subprocess rather than through a
binding: it is the same tool the docs tell users to install, and its absence is
reported with the command that fixes it.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from audio_transcriber.backend import TARGET_SAMPLE_RATE
from audio_transcriber.errors import AudioDecodeError, ConfigError

FFMPEG = "ffmpeg"

#: How long a single file may take to decode before it is treated as stuck.
DECODE_TIMEOUT_SECONDS = 600

_INSTALL_HINT = "Install it: apt install ffmpeg"


def check_ffmpeg() -> None:
    """Verify ffmpeg is installed and runnable.

    Called during up-front validation, so a missing ffmpeg fails the run before
    any file is touched.

    Raises:
        ConfigError: If ffmpeg is absent or cannot be executed.
    """
    if shutil.which(FFMPEG) is None:
        raise ConfigError(f"ffmpeg not found on PATH. {_INSTALL_HINT}")
    try:
        subprocess.run(
            [FFMPEG, "-version"],
            capture_output=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ConfigError(
            f"ffmpeg is on PATH but would not run: {error}. {_INSTALL_HINT}"
        ) from error


def decode(source: Path, destination: Path) -> Path:
    """Decode any audio file to mono 16-bit PCM WAV at the target sample rate.

    Args:
        source: The input file, in any container ffmpeg can read.
        destination: Where to write the decoded WAV.

    Returns:
        `destination`, for convenience.

    Raises:
        AudioDecodeError: If ffmpeg cannot decode the file. The message carries
            ffmpeg's own last line, which names the real problem.
    """
    command = [
        FFMPEG,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        str(destination),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=DECODE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise AudioDecodeError(
            str(source),
            f"decoding timed out after {DECODE_TIMEOUT_SECONDS}s.",
        ) from error
    except OSError as error:
        raise AudioDecodeError(str(source), f"could not run ffmpeg: {error}") from error

    if result.returncode != 0:
        raise AudioDecodeError(str(source), _explain(result.stderr))

    if not destination.exists() or destination.stat().st_size == 0:
        raise AudioDecodeError(
            str(source), "ffmpeg reported success but produced no audio."
        )
    return destination


def _explain(stderr: str) -> str:
    """Turn ffmpeg's stderr into one actionable line."""
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    if not lines:
        return "ffmpeg failed without reporting why."
    detail = lines[-1]
    if "Invalid data found" in detail or "moov atom not found" in detail:
        return f"{detail} The file may be truncated or not audio."
    return detail


@contextmanager
def decoded(source: Path) -> Iterator[Path]:
    """Decode a file to a temporary WAV, cleaned up on exit.

    Args:
        source: The input file.

    Yields:
        Path to the decoded WAV, valid for the duration of the context.

    Raises:
        AudioDecodeError: If the file cannot be decoded.
    """
    with TemporaryDirectory(prefix="audio-transcriber-") as directory:
        yield decode(source, Path(directory) / "decoded.wav")
