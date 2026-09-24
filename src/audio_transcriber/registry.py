"""The backend registry.

Backends are selected by name, so choosing one is configuration rather than code.
An explicit dict keeps registration visible and import-time cheap; entry-point
discovery can replace the internals later without changing this interface.
"""

from __future__ import annotations

import importlib
from typing import Any, NamedTuple

from audio_transcriber.backend import TranscriptionBackend
from audio_transcriber.errors import (
    BackendNotInstalledError,
    ConfigError,
    UnknownBackendError,
)

_REGISTRY: dict[str, type[TranscriptionBackend]] = {}

_PACKAGE = __name__.partition(".")[0]


class _Builtin(NamedTuple):
    """A backend shipped with this package, behind an optional extra."""

    name: str
    module: str
    extra: str


#: Built-in backends, imported on first lookup. A backend whose optional extra
#: is not installed is skipped, not an error: a missing engine only matters when
#: it is the one selected. The name and extra are declared here because the
#: module cannot be imported to ask; a test checks they match.
_BUILTINS = (
    _Builtin(
        "faster-whisper", "audio_transcriber.backends.faster_whisper", "faster-whisper"
    ),
    _Builtin(
        "elevenlabs-scribe",
        "audio_transcriber.backends.elevenlabs_scribe",
        "elevenlabs",
    ),
)

#: Built-ins that failed to import: name to why, with the fix when there is one.
_unavailable: dict[str, str] = {}

_builtins_loaded = False


def register(backend_class: type[TranscriptionBackend]) -> type[TranscriptionBackend]:
    """Register a backend class under its `name`.

    Usable as a decorator on the class definition.

    Args:
        backend_class: The backend to register.

    Returns:
        The same class, so this can decorate it.

    Raises:
        ConfigError: If the class has no name, or the name is already taken by a
            different class.
    """
    name = backend_class.name
    if not name:
        raise ConfigError(
            f"{backend_class.__name__} has no name. Set a class-level "
            "`name` so it can be selected with --backend."
        )
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not backend_class:
        raise ConfigError(
            f"backend name {name!r} is already registered to "
            f"{existing.__name__}. Pick a different name."
        )
    _REGISTRY[name] = backend_class
    return backend_class


def _load_builtins() -> None:
    """Import the built-in backend modules once, skipping uninstalled extras."""
    global _builtins_loaded
    if _builtins_loaded:
        return
    _builtins_loaded = True
    for builtin in _BUILTINS:
        try:
            importlib.import_module(builtin.module)
        except ImportError as error:
            _unavailable[builtin.name] = _why_unavailable(builtin, error)


def _why_unavailable(builtin: _Builtin, error: ImportError) -> str:
    """Say why a built-in failed to import, completing "backend X is ...".

    A missing third-party module means the extra is not installed. Anything else,
    including a missing module of this package, is a bug, and telling the user
    to install the extra would send them the wrong way.
    """
    missing = error.name if isinstance(error, ModuleNotFoundError) else None
    if missing and missing.partition(".")[0] != _PACKAGE:
        return (
            f"not installed: it needs {missing!r}. Install its extra: "
            f'pip install -e ".[{builtin.extra}]"'
        )
    return (
        f"broken: importing {builtin.module} failed ({error}). "
        "This is not a missing extra."
    )


def available() -> tuple[str, ...]:
    """Return the names of registered backends, sorted."""
    _load_builtins()
    return tuple(sorted(_REGISTRY))


def unavailable() -> dict[str, str]:
    """Return built-ins that failed to import, by name, with why and the fix."""
    _load_builtins()
    return dict(sorted(_unavailable.items()))


def get(name: str) -> type[TranscriptionBackend]:
    """Return the backend class registered under `name`.

    Args:
        name: The backend name, as passed to `--backend`.

    Raises:
        UnknownBackendError: If no backend is registered under that name. The
            error lists the names that are available. It is a
            `BackendNotInstalledError` when the name is a built-in that could
            not be imported, saying what to install.
    """
    _load_builtins()
    try:
        return _REGISTRY[name]
    except KeyError:
        if name in _unavailable:
            raise BackendNotInstalledError(
                name, _unavailable[name], available()
            ) from None
        raise UnknownBackendError(name, available()) from None


def create(name: str, **config: Any) -> TranscriptionBackend:
    """Instantiate the backend registered under `name`.

    Args:
        name: The backend name.
        **config: Backend-specific options, recorded in the run manifest.

    Raises:
        UnknownBackendError: If no backend is registered under that name.
    """
    return get(name)(**config)


def _clear() -> None:
    """Reset the registry. For tests only."""
    global _builtins_loaded
    _REGISTRY.clear()
    _unavailable.clear()
    _builtins_loaded = False
