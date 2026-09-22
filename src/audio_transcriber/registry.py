"""The backend registry.

Backends are selected by name, so choosing one is configuration rather than code.
An explicit dict keeps registration visible and import-time cheap; entry-point
discovery can replace the internals later without changing this interface.
"""

from __future__ import annotations

import importlib
from typing import Any

from audio_transcriber.backend import TranscriptionBackend
from audio_transcriber.errors import ConfigError, UnknownBackendError

_REGISTRY: dict[str, type[TranscriptionBackend]] = {}

#: Modules registering a built-in backend, imported on first lookup. A backend
#: whose optional extra is not installed is skipped, not an error: a missing
#: engine only matters when it is the one selected.
_BUILTIN_MODULES = ("audio_transcriber.backends.faster_whisper",)

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
    for module in _BUILTIN_MODULES:
        try:
            importlib.import_module(module)
        except ImportError:
            continue


def available() -> tuple[str, ...]:
    """Return the names of registered backends, sorted."""
    _load_builtins()
    return tuple(sorted(_REGISTRY))


def get(name: str) -> type[TranscriptionBackend]:
    """Return the backend class registered under `name`.

    Args:
        name: The backend name, as passed to `--backend`.

    Raises:
        UnknownBackendError: If no backend is registered under that name. The
            error lists the names that are available.
    """
    _load_builtins()
    try:
        return _REGISTRY[name]
    except KeyError:
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
    _builtins_loaded = False
