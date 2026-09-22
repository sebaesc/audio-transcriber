"""Tests for the backend registry.

Backend selection is configuration, not code, so the registry's error messages
are part of the CLI's contract with the user.
"""

import pytest

from audio_transcriber import registry
from audio_transcriber.backend import TranscriptionBackend
from audio_transcriber.errors import ConfigError, UnknownBackendError
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
    monkeypatch.setattr(registry, "_BUILTIN_MODULES", ())
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


def test_missing_builtin_extra_is_skipped_not_raised(monkeypatch):
    """A backend whose extra is not installed only matters if it is selected."""
    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.setattr(registry, "_BUILTIN_MODULES", ("audio_transcriber.nope",))
    monkeypatch.setattr(registry, "_builtins_loaded", False)

    assert registry.available() == ()
