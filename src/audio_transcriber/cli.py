"""The command-line interface.

Output follows `conductor/product-guidelines.md`: progress and errors go to
stderr, transcript data goes to files, and every error says what happened and
how to fix it. Exit codes distinguish a clean run from a partial failure and
from a run that never started.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
import time
from enum import IntEnum
from pathlib import Path
from typing import Annotated, Any

import typer

from audio_transcriber import __version__, discovery, pipeline, registry
from audio_transcriber.errors import AudioTranscriberError
from audio_transcriber.pipeline import RunResult, RunStatus


class ExitCode(IntEnum):
    """What the process returns, so a caller can branch without parsing output."""

    SUCCESS = 0
    PARTIAL = 1
    ABORTED = 2
    CONFIG_ERROR = 3
    INTERRUPTED = 130


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Transcribe audio files with a swappable model backend.",
)


def _echo(message: str) -> None:
    """Write a line to stderr, keeping stdout free for data."""
    print(message, file=sys.stderr)


def _fail(error: Exception) -> None:
    """Report an error on stderr with the `error:` prefix."""
    _echo(f"error: {error}")


def _parse_option(raw: str) -> tuple[str, Any]:
    """Parse one `key=value` backend option.

    Values are read as JSON when possible, so numbers, booleans, and null keep
    their types instead of arriving as strings the backend has to reinterpret.

    Raises:
        typer.BadParameter: If the option is not `key=value`.
    """
    key, separator, value = raw.partition("=")
    if not separator or not key:
        raise typer.BadParameter(
            f"expected key=value, got {raw!r}. Example: --option beam_size=1"
        )
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


def _format_duration(seconds: float) -> str:
    """Format a duration the way the guidelines' example does: `4m18s`."""
    minutes, remainder = divmod(int(seconds), 60)
    return f"{minutes}m{remainder:02d}s" if minutes else f"{remainder}s"


def _report(result: RunResult, elapsed: float) -> None:
    """Summarize a finished run on stderr."""
    total = len(result.transcripts) + len(result.failures)
    if result.status is RunStatus.SUCCEEDED:
        _echo(
            f"Transcribed {total} files in {_format_duration(elapsed)} "
            f"→ {result.run_directory}/"
        )
        return

    _echo(
        f"warning: {len(result.failures)} of {total} files failed. "
        f"See {result.run_directory / pipeline.ERRORS_FILENAME}"
    )
    if result.transcripts:
        _echo(
            f"Transcribed {len(result.transcripts)} files in "
            f"{_format_duration(elapsed)} → {result.run_directory}/"
        )


def _version_callback(requested: bool) -> None:
    if requested:
        print(f"audio-transcriber {__version__}")
        raise typer.Exit(ExitCode.SUCCESS)


def _list_backends_callback(requested: bool) -> None:
    """Print the installed backends and exit.

    A flag rather than a subcommand, so the run form stays
    `audio-transcriber --input ... --backend ...` with no verb to remember.
    """
    if not requested:
        return
    names = registry.available()
    for name, detail in registry.unavailable().items():
        _echo(f"{name} is {detail}")
    if not names:
        _echo(
            "No backends installed. Install one as an extra: "
            'pip install -e ".[<backend>]". '
            "See docs/backends/ for the available ones."
        )
        raise typer.Exit(ExitCode.CONFIG_ERROR)
    for name in names:
        print(name)
    raise typer.Exit(ExitCode.SUCCESS)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    input_path: Annotated[
        Path | None,
        typer.Option(
            "--input",
            "-i",
            help="Directory of audio files, or a single file.",
        ),
    ] = None,
    backend: Annotated[
        str | None,
        typer.Option(
            "--backend",
            "-b",
            help="Backend to transcribe with. Use --list-backends to see them.",
        ),
    ] = None,
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Directory to write the run into."),
    ] = Path("out"),
    glob: Annotated[
        str,
        typer.Option(
            "--glob",
            help="Pattern for finding inputs, relative to --input.",
        ),
    ] = discovery.DEFAULT_GLOB,
    option: Annotated[
        list[str] | None,
        typer.Option(
            "--option",
            metavar="KEY=VALUE",
            help="Backend option. Repeatable, e.g. --option model=large-v3.",
        ),
    ] = None,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Report only the final summary."),
    ] = False,
    list_backends: Annotated[
        bool,
        typer.Option(
            "--list-backends",
            callback=_list_backends_callback,
            is_eager=True,
            help="List the installed backends and exit.",
        ),
    ] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Print the version and exit.",
        ),
    ] = False,
) -> None:
    """Transcribe audio files with a swappable model backend.

    Run with --input and --backend, or use a command below.
    """
    if ctx.invoked_subcommand is not None:
        # The run form is the callback, so `audio-transcriber --input ...`
        # needs no verb; a command such as `serve` runs instead of it.
        return
    logging.basicConfig(
        level=logging.ERROR if quiet else logging.WARNING,
        format="warning: %(message)s",
        stream=sys.stderr,
    )

    missing = [
        name
        for name, value in (("--input", input_path), ("--backend", backend))
        if value is None
    ]
    if missing:
        _echo(f"error: missing required option(s): {', '.join(missing)}")
        _echo("Run with --help for usage, or --list-backends to see backends.")
        raise typer.Exit(ExitCode.CONFIG_ERROR)

    config = dict(_parse_option(item) for item in option or [])

    def _progress(index: int, total: int, source: Path) -> None:
        if not quiet:
            _echo(f"[{index}/{total}] {source}")

    started = time.monotonic()
    try:
        result = pipeline.run(
            input_path=input_path,
            backend_name=backend,
            output_root=out,
            pattern=glob,
            backend_config=config,
            on_progress=_progress,
        )
    except KeyboardInterrupt:
        _echo("error: interrupted. Partial results are kept.")
        raise typer.Exit(ExitCode.INTERRUPTED) from None
    except AudioTranscriberError as error:
        _fail(error)
        raise typer.Exit(ExitCode.CONFIG_ERROR) from None

    _report(result, time.monotonic() - started)

    if result.status is RunStatus.SUCCEEDED:
        raise typer.Exit(ExitCode.SUCCESS)
    if result.status is RunStatus.PARTIAL:
        raise typer.Exit(ExitCode.PARTIAL)
    raise typer.Exit(ExitCode.ABORTED)


@app.command()
def serve(
    host: Annotated[
        str,
        typer.Option(help="Address to listen on. 0.0.0.0 listens on every one."),
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000,
    workers_concurrency: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Transcription jobs run at once. Overrides "
            "AUDIO_TRANSCRIBER_CONCURRENCY; defaults to 1.",
        ),
    ] = None,
) -> None:
    """Serve the HTTP API. Needs the api extra."""
    try:
        import uvicorn

        from audio_transcriber.api.app import create_app
        from audio_transcriber.api.settings import ApiSettings
    except ModuleNotFoundError as error:
        _echo(
            f"error: serving the HTTP API needs {error.name!r}, which is not "
            'installed. Install the api extra: pip install -e ".[api]"'
        )
        raise typer.Exit(ExitCode.CONFIG_ERROR) from None

    try:
        settings = ApiSettings.from_env()
        if workers_concurrency is not None:
            settings = dataclasses.replace(settings, concurrency=workers_concurrency)
        application = create_app(settings)
    except AudioTranscriberError as error:
        _fail(error)
        raise typer.Exit(ExitCode.CONFIG_ERROR) from None

    uvicorn.run(application, host=host, port=port)


def run() -> None:
    """Entry point for the console script and `python -m audio_transcriber`."""
    app()
