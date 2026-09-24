"""Tests for the backend registry.

Backend selection is configuration, not code, so the registry's error messages
are part of the CLI's contract with the user.
"""

import importlib
import sys
import tomllib
from pathlib import Path

import pytest

from audio_transcriber import registry
from audio_transcriber.backend import TranscriptionBackend
from audio_transcriber.errors import (
    BackendNotInstalledError,
    ConfigError,
    UnknownBackendError,
)
from audio_transcriber.schema import Transcript


class _StubBackend(TranscriptionBackend):
    """A minimal backend, standing in for a real engine."""

    name = "stub"

    def check_requirements(self) -> None:
        return None

    def model_version(self) -> str:
        return "stub-1.0"

    def transcribe(self, audio_path, source) -> Transcript:
        return Transcript(source=source, backend=self.name)


@pytest.fixture
def empty_registry(monkeypatch):
    """Give each test its own registry, isolated from the built-ins."""
    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.setattr(registry, "_BUILTINS", ())
    monkeypatch.setattr(registry, "_unavailable", {})
    monkeypatch.setattr(registry, "_builtins_loaded", False)
    return registry


def test_registered_backend_is_retrievable_by_name(empty_registry):
    empty_registry.register(_StubBackend)

    assert empty_registry.get("stub") is _StubBackend


def test_register_returns_the_class_so_it_works_as_a_decorator(empty_registry):
    assert empty_registry.register(_StubBackend) is _StubBackend


def test_create_instantiates_with_config(empty_registry):
    empty_registry.register(_StubBackend)

    backend = empty_registry.create("stub", language="en")

    assert isinstance(backend, _StubBackend)
    assert backend.effective_config() == {"language": "en"}


def test_unknown_backend_lists_available_names(empty_registry):
    """The error tells the user what they can pick instead."""
    empty_registry.register(_StubBackend)

    with pytest.raises(UnknownBackendError, match="unknown backend 'nope'") as caught:
        empty_registry.get("nope")

    assert "stub" in str(caught.value)
    assert caught.value.known == ("stub",)


def test_unknown_backend_with_empty_registry_says_none_registered(empty_registry):
    with pytest.raises(UnknownBackendError, match="none registered"):
        empty_registry.get("nope")


def test_available_returns_sorted_names(empty_registry):
    class _Zeta(_StubBackend):
        name = "zeta"

    class _Alpha(_StubBackend):
        name = "alpha"

    empty_registry.register(_Zeta)
    empty_registry.register(_Alpha)

    assert empty_registry.available() == ("alpha", "zeta")


def test_registering_a_nameless_backend_is_rejected(empty_registry):
    class _Nameless(_StubBackend):
        name = ""

    with pytest.raises(ConfigError, match="has no name"):
        empty_registry.register(_Nameless)


def test_registering_a_duplicate_name_is_rejected(empty_registry):
    class _Other(_StubBackend):
        name = "stub"

    empty_registry.register(_StubBackend)

    with pytest.raises(ConfigError, match="already registered"):
        empty_registry.register(_Other)


def test_registering_the_same_class_twice_is_allowed(empty_registry):
    """Re-importing a backend module must not break the registry."""
    empty_registry.register(_StubBackend)
    empty_registry.register(_StubBackend)

    assert empty_registry.available() == ("stub",)


# --- Built-ins whose extra is not installed ---


@pytest.fixture
def ghost_builtin(empty_registry, monkeypatch, tmp_path):
    """Declare one built-in, `ghost`, whose module is written by each test."""
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        registry,
        "_BUILTINS",
        (registry._Builtin("ghost", "ghost_backend", "ghost-extra"),),
    )

    def write(source: str):
        (tmp_path / "ghost_backend.py").write_text(source)
        return empty_registry

    yield write
    sys.modules.pop("ghost_backend", None)


def test_missing_builtin_extra_is_skipped_not_raised(ghost_builtin):
    """A backend whose extra is not installed only matters if it is selected."""
    reg = ghost_builtin("import ghost_engine_not_installed\n")

    assert reg.available() == ()


def test_selecting_a_builtin_without_its_extra_names_the_extra(ghost_builtin):
    reg = ghost_builtin("import ghost_engine_not_installed\n")

    with pytest.raises(BackendNotInstalledError) as caught:
        reg.get("ghost")

    message = str(caught.value)
    assert "ghost_engine_not_installed" in message
    assert 'pip install -e ".[ghost-extra]"' in message
    assert "unknown backend" not in message


def test_a_missing_extra_is_still_an_unknown_backend_to_callers(ghost_builtin):
    """`get` documents UnknownBackendError; the subclass keeps that true."""
    reg = ghost_builtin("import ghost_engine_not_installed\n")

    with pytest.raises(UnknownBackendError):
        reg.create("ghost")


def test_a_broken_backend_module_is_not_blamed_on_the_extra(ghost_builtin):
    """An import error inside this package is a bug, not a missing install."""
    reg = ghost_builtin("import audio_transcriber.no_such_module\n")

    with pytest.raises(BackendNotInstalledError) as caught:
        reg.get("ghost")

    message = str(caught.value)
    assert "audio_transcriber.no_such_module" in message
    assert "pip install" not in message


def test_unavailable_lists_what_is_missing_and_why(ghost_builtin):
    reg = ghost_builtin("import ghost_engine_not_installed\n")

    missing = reg.unavailable()

    assert list(missing) == ["ghost"]
    assert 'pip install -e ".[ghost-extra]"' in missing["ghost"]


def test_an_unrelated_name_is_still_simply_unknown(ghost_builtin):
    reg = ghost_builtin("import ghost_engine_not_installed\n")

    with pytest.raises(UnknownBackendError) as caught:
        reg.get("nope")

    assert not isinstance(caught.value, BackendNotInstalledError)


def test_builtins_declare_their_real_names_and_extras():
    """The declared name and extra are what the error tells a user to install."""
    extras = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())[
        "project"
    ]["optional-dependencies"]

    for builtin in registry._BUILTINS:
        assert builtin.extra in extras, builtin
        try:
            module = importlib.import_module(builtin.module)
        except ImportError:
            continue  # Extra not installed here; nothing to compare against.
        defined = {
            value.name
            for value in vars(module).values()
            if isinstance(value, type)
            and issubclass(value, TranscriptionBackend)
            and value.__module__ == builtin.module
        }
        assert builtin.name in defined, builtin
