"""Transcription backends.

Each backend is one module in this package implementing the backend interface
and registering itself by name. Nothing outside a backend module may import an
engine's library or branch on a backend name.
"""
