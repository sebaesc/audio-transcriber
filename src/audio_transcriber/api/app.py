"""The HTTP application: routes, and errors mapped to statuses.

Only this module imports the web framework, so the rest of the package stays
importable without the `api` extra.

A request is checked before anything is queued: an unknown or uninstalled
backend, invalid options or an oversize upload are refused with a 4xx and a JSON
error. What can only fail while transcribing, such as undecodable audio or a
missing credential, is reported on the job instead.

Every error body has one shape: `{"error": {"type", "message", "remedy"}}`.
"""

from __future__ import annotations

import json
import shutil
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, BinaryIO

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from audio_transcriber import __version__, registry
from audio_transcriber.api.jobs import InMemoryJobStore, Job, JobError, JobStore
from audio_transcriber.api.runner import JobRunner
from audio_transcriber.api.settings import ApiSettings
from audio_transcriber.errors import (
    AudioTranscriberError,
    BackendRequirementError,
    ConfigError,
)

#: Room for the multipart framing and form fields around the file itself, when
#: refusing a request by its declared length before reading it.
_FORM_OVERHEAD_BYTES = 64 * 1024

_CHUNK_BYTES = 1024 * 1024


class ApiError(Exception):
    """A request refused with an HTTP status and a structured error.

    Attributes:
        status: The HTTP status code.
        error: The error body.
    """

    def __init__(self, status: int, error: JobError) -> None:
        super().__init__(error.message)
        self.status = status
        self.error = error


class UploadTooLargeError(AudioTranscriberError):
    """Raised when an upload exceeds the configured maximum size."""

    def __init__(self, limit: int) -> None:
        self.remedy = (
            "Send a smaller file, or raise AUDIO_TRANSCRIBER_MAX_UPLOAD_MB "
            "on the server."
        )
        super().__init__(f"the upload is larger than the {limit}-byte limit.")


def error_response(status: int, error: JobError) -> JSONResponse:
    """Return an error in the API's one error shape."""
    return JSONResponse(status_code=status, content={"error": error.to_dict()})


def create_app(
    settings: ApiSettings | None = None, store: JobStore | None = None
) -> FastAPI:
    """Build the application.

    Args:
        settings: How the service runs. Read from the environment if omitted.
        store: Where jobs are kept. In memory if omitted.

    Returns:
        The app. Its job runner starts with it and, on shutdown, waits for the
        jobs already submitted.
    """
    settings = settings or ApiSettings.from_env()
    store = store or InMemoryJobStore()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.runs_dir.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.runner = JobRunner(store, settings)
        try:
            yield
        finally:
            app.state.runner.shutdown()

    app = FastAPI(
        title="Audio Transcriber",
        version=__version__,
        summary="Transcribe audio with a swappable model backend, as async jobs.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.store = store

    _add_error_handlers(app)
    _add_upload_limit(app, settings)
    _add_transcription_routes(app)
    _add_service_routes(app)
    return app


def _add_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, error: ApiError) -> JSONResponse:
        return error_response(error.status, error.error)

    @app.exception_handler(RequestValidationError)
    async def _invalid_request(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in problem['loc'])}: {problem['msg']}"
            for problem in error.errors()
        )
        return error_response(
            422,
            JobError(
                type="RequestValidationError",
                message=f"the request is invalid: {problems}.",
                remedy="See /docs for the request format.",
            ),
        )


def _add_upload_limit(app: FastAPI, settings: ApiSettings) -> None:
    """Refuse a request whose declared length is over the limit, unread.

    The multipart parser spools a whole upload before a route sees it, so a
    route alone cannot stop a client filling the disk. Requests without a
    length are still caught, byte-exact, while the upload is copied.
    """
    limit = settings.max_upload_bytes + _FORM_OVERHEAD_BYTES

    @app.middleware("http")
    async def _limit(request: Request, call_next):
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > limit:
            return error_response(
                413,
                JobError.from_exception(UploadTooLargeError(settings.max_upload_bytes)),
            )
        return await call_next(request)


def _add_transcription_routes(app: FastAPI) -> None:
    @app.post("/transcriptions", status_code=202)
    def submit_transcription(
        request: Request,
        file: Annotated[UploadFile, File(description="The audio to transcribe.")],
        backend: Annotated[str, Form(description="Backend name; see /backends.")],
        options: Annotated[
            str,
            Form(
                description='Backend options as a JSON object, e.g. {"beam_size": 1}.'
            ),
        ] = "{}",
    ) -> JSONResponse:
        """Queue a transcription. Poll the job at the returned `Location`."""
        settings: ApiSettings = request.app.state.settings
        config = _parse_options(options)
        _check_backend(backend, config)

        job = Job.new(backend=backend, options=config, filename=_safe_name(file))
        upload = _save_upload(
            file.file,
            settings.uploads_dir / uuid.uuid4().hex / job.filename,
            settings.max_upload_bytes,
        )
        request.app.state.runner.submit(job, upload)
        location = f"/transcriptions/{job.id}"
        return JSONResponse(
            status_code=202,
            content={"id": job.id, "status": job.status.value},
            headers={"Location": location},
        )

    @app.get("/transcriptions/{job_id}")
    def get_transcription(request: Request, job_id: str) -> dict[str, Any]:
        """Return a job: its status, and its transcript or error once finished."""
        return find_job(request, job_id).to_dict()


def find_job(request: Request, job_id: str) -> Job:
    """Return the job with this id.

    Raises:
        ApiError: 404 if there is no such job.
    """
    job = request.app.state.store.get(job_id)
    if job is None:
        raise ApiError(
            404,
            JobError(
                type="UnknownJobError",
                message=f"no transcription job {job_id!r}.",
                remedy=(
                    "Check the id. Jobs are kept in memory and are lost when "
                    "the server restarts."
                ),
            ),
        )
    return job


def _add_service_routes(app: FastAPI) -> None:
    @app.get("/backends")
    def list_backends() -> dict[str, Any]:
        """List the installed backends, and why the others are unavailable."""
        return {
            "available": list(registry.available()),
            "unavailable": registry.unavailable(),
        }

    @app.get("/health")
    def health() -> dict[str, str]:
        """Report that the service is up."""
        return {"status": "ok"}


def _parse_options(raw: str) -> dict[str, Any]:
    """Read the backend options form field.

    Raises:
        ApiError: 400 if it is not a JSON object.
    """
    try:
        options = json.loads(raw)
    except json.JSONDecodeError:
        options = None
    if not isinstance(options, dict):
        raise ApiError(
            400,
            JobError.from_exception(
                ConfigError(
                    f"options must be a JSON object, got {raw!r}. "
                    'Example: {"beam_size": 1}.'
                )
            ),
        )
    return options


def _check_backend(name: str, config: dict[str, Any]) -> None:
    """Refuse a request no job could run: the backend or its options are wrong.

    Constructing a backend only checks its options; models load and
    credentials are read when it runs, in the job.

    Raises:
        ApiError: 400 for an unknown or uninstalled backend, or bad options.
    """
    try:
        registry.create(name, **config)
    except (ConfigError, BackendRequirementError) as error:
        raise ApiError(400, JobError.from_exception(error)) from None


def _safe_name(file: UploadFile) -> str:
    """Return the upload's base name, so it cannot point outside its directory."""
    name = Path(file.filename or "").name
    return name if name not in ("", ".", "..") else "upload"


def _save_upload(source: BinaryIO, destination: Path, limit: int) -> Path:
    """Copy an upload to disk, refusing it past `limit` bytes.

    Raises:
        ApiError: 413 if the upload is too large. Nothing is kept.
    """
    destination.parent.mkdir(parents=True)
    written = 0
    try:
        with destination.open("wb") as handle:
            while chunk := source.read(_CHUNK_BYTES):
                written += len(chunk)
                if written > limit:
                    raise ApiError(
                        413, JobError.from_exception(UploadTooLargeError(limit))
                    )
                handle.write(chunk)
    except BaseException:
        shutil.rmtree(destination.parent, ignore_errors=True)
        raise
    return destination
