"""The HTTP service is an optional extra, so the core must not need it.

The `api` extra pulls in a web framework and a server. A core install without it
has to import and run the CLI, so no core module may import them, even
indirectly. Only `serve` reaches for them, at call time.

Checked in a fresh interpreter: this test process may already have imported
them for the API tests.
"""

import subprocess
import sys

import pytest

SERVER_LIBRARIES = ("fastapi", "starlette", "uvicorn", "multipart")


@pytest.mark.parametrize(
    "module",
    ["audio_transcriber", "audio_transcriber.cli", "audio_transcriber.pipeline"],
)
def test_core_module_does_not_import_the_server(module):
    code = (
        f"import sys, {module}\n"
        f"print(','.join(m for m in {SERVER_LIBRARIES!r} if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "", (
        f"importing {module} imported {result.stdout.strip()}. Server libraries "
        "belong to the api extra; import them only inside audio_transcriber.api."
    )
