"""The API service's settings, read from the environment.

Environment variables rather than flags, so the same image runs anywhere with
a different `.env`. `serve` overrides the ones it has flags for.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

from audio_transcriber.errors import ConfigError

#: Largest upload accepted by default: an hour of uncompressed 16-bit stereo
#: 44.1 kHz audio is about 600 MB.
DEFAULT_MAX_UPLOAD_MB = 1024

ENV_MAX_UPLOAD_MB = "AUDIO_TRANSCRIBER_MAX_UPLOAD_MB"
ENV_CONCURRENCY = "AUDIO_TRANSCRIBER_CONCURRENCY"
ENV_WORK_DIR = "AUDIO_TRANSCRIBER_WORK_DIR"


def _default_work_dir() -> Path:
    return Path(tempfile.gettempdir()) / "audio-transcriber-api"


@dataclass(frozen=True, slots=True)
class ApiSettings:
    """How the service runs.

    Attributes:
        max_upload_bytes: Uploads larger than this are refused with 413.
        concurrency: Jobs run at once. One by default, so a local model does
            not overcommit the CPU or GPU; raise it for hosted backends.
        work_dir: Holds uploads while their job runs, and each job's run
            directory (transcript and manifest) for later evaluation.
    """

    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_MB * 1024 * 1024
    concurrency: int = 1
    work_dir: Path = field(default_factory=_default_work_dir)

    def __post_init__(self) -> None:
        if self.max_upload_bytes < 1:
            raise ConfigError("max upload size must be at least 1 byte.")
        if self.concurrency < 1:
            raise ConfigError(
                f"concurrency must be at least 1, got {self.concurrency}."
            )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Self:
        """Read the settings from environment variables, defaulting the rest.

        Raises:
            ConfigError: If a variable is set to something unusable.
        """
        environ = os.environ if environ is None else environ
        settings = cls()
        max_upload_mb = _positive_int(environ, ENV_MAX_UPLOAD_MB)
        concurrency = _positive_int(environ, ENV_CONCURRENCY)
        work_dir = environ.get(ENV_WORK_DIR)
        return cls(
            max_upload_bytes=(
                max_upload_mb * 1024 * 1024
                if max_upload_mb is not None
                else settings.max_upload_bytes
            ),
            concurrency=concurrency or settings.concurrency,
            work_dir=Path(work_dir) if work_dir else settings.work_dir,
        )

    @property
    def uploads_dir(self) -> Path:
        """Where uploads wait for their job."""
        return self.work_dir / "uploads"

    @property
    def runs_dir(self) -> Path:
        """Where each job's run directory is written."""
        return self.work_dir / "runs"


def _positive_int(environ: Mapping[str, str], name: str) -> int | None:
    raw = environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise ConfigError(f"{name} must be a whole number of at least 1, got {raw!r}.")
    return value
