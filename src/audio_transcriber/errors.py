"""Error types raised across the pipeline.

Messages follow `conductor/product-guidelines.md`: what happened, why, and how to
fix it. Each error carries the context a caller needs to report it without
re-deriving it.
"""

from __future__ import annotations


class AudioTranscriberError(Exception):
    """Base class for every error this package raises."""


class ConfigError(AudioTranscriberError):
    """Raised when a run's configuration is invalid.

    Raised during up-front validation, before any file is processed.
    """


class UnknownBackendError(ConfigError):
    """Raised when a backend name is not registered.

    Attributes:
        name: The requested backend name.
        known: The registered backend names.
    """

    def __init__(self, name: str, known: tuple[str, ...]) -> None:
        self.name = name
        self.known = known
        available = ", ".join(known) if known else "none registered"
        super().__init__(f"unknown backend {name!r}. Available backends: {available}")


class BackendNotInstalledError(UnknownBackendError):
    """Raised when a built-in backend is selected but could not be imported.

    Usually its optional extra is not installed. A subclass of
    `UnknownBackendError`, since the backend is not registered either way; only
    the message differs, saying what to install.

    Attributes:
        name: The requested backend name.
        known: The registered backend names.
        detail: Why it is unavailable, and the fix when there is one.
    """

    def __init__(self, name: str, detail: str, known: tuple[str, ...]) -> None:
        self.name = name
        self.known = known
        self.detail = detail
        ConfigError.__init__(self, f"backend {name!r} is {detail}")


class BackendRequirementError(AudioTranscriberError):
    """Raised when a backend's requirements are not met.

    Raised at validation time, before a run starts, so a missing model or
    credential fails fast rather than part-way through a batch.

    Attributes:
        backend: Name of the backend whose requirement is unmet.
        remedy: The command or config change that fixes it.
    """

    def __init__(self, backend: str, message: str, remedy: str = "") -> None:
        self.backend = backend
        self.remedy = remedy
        detail = f"{backend}: {message}"
        super().__init__(f"{detail} {remedy}".rstrip() if remedy else detail)


class TranscriptionError(AudioTranscriberError):
    """Raised when a backend fails to transcribe one file.

    A per-file failure. The run collects it and continues with the next file.

    Attributes:
        source: Path of the file that failed.
        backend: Name of the backend that raised it.
    """

    def __init__(self, source: str, backend: str, message: str) -> None:
        self.source = source
        self.backend = backend
        super().__init__(f"{source}: {backend} failed to transcribe it. {message}")


class AudioDecodeError(AudioTranscriberError):
    """Raised when an audio file cannot be decoded.

    A per-file failure, handled like a `TranscriptionError`.

    Attributes:
        source: Path of the file that could not be decoded.
    """

    def __init__(self, source: str, message: str) -> None:
        self.source = source
        super().__init__(f"{source}: {message}")
