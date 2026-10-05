"""The HTTP API service: the pipeline over HTTP, as asynchronous jobs.

A consumer of the pipeline, installed with the optional `api` extra. Only
`audio_transcriber.api.app` imports the web framework, so the job model and
settings stay importable, and testable, without it.
"""
