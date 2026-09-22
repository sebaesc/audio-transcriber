"""Finding the audio files a run will process.

Discovery is deterministic: the same directory and pattern always yield the same
files in the same order, so two runs over unchanged input do the same work in the
same sequence.
"""

from __future__ import annotations

from pathlib import Path

from audio_transcriber.errors import ConfigError

#: Extensions treated as audio when no explicit pattern is given. Anything ffmpeg
#: can decode will work if named explicitly with `--glob`.
AUDIO_EXTENSIONS = frozenset(
    {
        ".aac",
        ".flac",
        ".m4a",
        ".mkv",
        ".mp3",
        ".mp4",
        ".oga",
        ".ogg",
        ".opus",
        ".wav",
        ".webm",
        ".wma",
    }
)

DEFAULT_GLOB = "**/*"


def discover(input_path: Path, pattern: str = DEFAULT_GLOB) -> list[Path]:
    """Return the audio files to transcribe, in a stable order.

    Args:
        input_path: A directory to search, or a single audio file.
        pattern: Glob pattern applied within a directory, relative to it.
            Defaults to every file at any depth, filtered by `AUDIO_EXTENSIONS`.

    Returns:
        Matching files sorted by path. A single file is returned as-is, without
        extension filtering, since naming it is explicit enough.

    Raises:
        ConfigError: If `input_path` does not exist.
    """
    if not input_path.exists():
        raise ConfigError(
            f"input path not found: {input_path}. Check the path, or create it."
        )

    if input_path.is_file():
        return [input_path]

    explicit_pattern = pattern != DEFAULT_GLOB
    found = [
        path
        for path in input_path.glob(pattern)
        if path.is_file() and (explicit_pattern or is_audio(path))
    ]
    return sorted(found)


def is_audio(path: Path) -> bool:
    """Report whether a path looks like audio, by extension."""
    return path.suffix.lower() in AUDIO_EXTENSIONS
