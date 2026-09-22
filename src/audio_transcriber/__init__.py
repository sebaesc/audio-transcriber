"""Audio Transcriber: a modular audio transcription pipeline.

The transcription model is a plug-in behind a stable interface. Pipeline code
contains no engine-specific imports or branches; every backend normalizes its
own output to the shared transcript schema.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
