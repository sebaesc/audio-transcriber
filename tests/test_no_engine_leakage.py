"""The pipeline must not know which engine it is running.

This is the project's first design principle, and the thing that makes backends
swappable. It is easy to violate by accident — one `if backend == "whisper"` in
the CLI is all it takes — so it is asserted structurally rather than trusted to
review.

The check reads the source of every pipeline module and fails on an engine
import or a mention of a registered backend's name.
"""

import ast
from pathlib import Path

import pytest

import audio_transcriber
from audio_transcriber import registry

PACKAGE_ROOT = Path(audio_transcriber.__file__).parent
BACKENDS_DIR = PACKAGE_ROOT / "backends"

#: Libraries that belong to a specific engine and nowhere else.
ENGINE_LIBRARIES = frozenset(
    {
        "faster_whisper",
        "whisper",
        "ctranslate2",
        "torch",
        "transformers",
        "openai",
        "vosk",
        "speechbrain",
    }
)

#: registry.py names the built-in backend modules so it can import them. That is
#: registration wiring, not engine-specific behavior: the registry never calls an
#: engine, and a module listed here that fails to import is skipped. Everything
#: else in the pipeline must be free of engine names.
REGISTRATION_WIRING = {PACKAGE_ROOT / "registry.py"}


def pipeline_modules() -> list[Path]:
    """Return every module outside the backends package."""
    return sorted(
        path for path in PACKAGE_ROOT.rglob("*.py") if BACKENDS_DIR not in path.parents
    )


def imported_modules(source: str) -> set[str]:
    """Return the top-level names imported by a module."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_there_are_pipeline_modules_to_check():
    """Guards against the check passing because it found nothing."""
    assert len(pipeline_modules()) >= 5


@pytest.mark.parametrize("module", pipeline_modules(), ids=lambda p: p.name)
def test_pipeline_module_imports_no_engine_library(module):
    leaked = imported_modules(module.read_text()) & ENGINE_LIBRARIES

    assert not leaked, (
        f"{module.name} imports {sorted(leaked)}. Engine libraries belong in "
        "a backend module, behind the interface."
    )


@pytest.mark.parametrize("module", pipeline_modules(), ids=lambda p: p.name)
def test_pipeline_module_does_not_mention_a_backend_by_name(module):
    if module in REGISTRATION_WIRING:
        pytest.skip("registration wiring, not engine-specific behavior")

    source = module.read_text()
    mentioned = [name for name in registry.available() if name in source]

    assert not mentioned, (
        f"{module.name} mentions {mentioned}. Selecting a backend is "
        "configuration; the pipeline resolves it through the registry."
    )


def test_registration_wiring_only_imports_backend_modules():
    """The one allowed exception stays an import list, not a branch."""
    source = (PACKAGE_ROOT / "registry.py").read_text()

    assert not imported_modules(source) & ENGINE_LIBRARIES


def test_backends_are_reachable_only_through_the_registry():
    """Pipeline modules import the interface, never a concrete backend."""
    for module in pipeline_modules():
        imports = imported_modules(module.read_text())
        assert "audio_transcriber.backends" not in imports
