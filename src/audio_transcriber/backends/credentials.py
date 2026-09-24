"""Credentials shared by backends: the config, then the environment, then `.env`.

Putting a key in a `.env` is the obvious thing to do, and Python does not read
one on its own, so it would otherwise be ignored in silence with the credential
sitting right there. Every backend that needs a secret looks it up here, so they
all resolve it the same way.
"""

from __future__ import annotations

import os

from dotenv import find_dotenv, load_dotenv

_dotenv_loaded = False


def find_credential(explicit: str | None, variables: tuple[str, ...]) -> str | None:
    """Return a credential, or None if none is set.

    Looked up in order: the value from the backend config, then each of
    `variables` in the environment, then the same variables from a `.env`
    beside the project.

    Args:
        explicit: The value from the backend config, if any.
        variables: Environment variable names to try, in order of preference.
    """
    if explicit:
        return explicit
    load_dotenv_once()
    for name in variables:
        value = os.environ.get(name)
        if value:
            return value
    return None


def load_dotenv_once() -> None:
    """Load a `.env` into the environment, at most once per process.

    `override=False` keeps an exported variable winning over the file, so
    setting one for a single run does what it looks like it does.
    """
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path, override=False)
