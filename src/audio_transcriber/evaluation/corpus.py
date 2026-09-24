"""The Bangor Miami Spanish-English reference corpus.

A naturalistic Spanish-English code-switching corpus recorded in Miami, published
as `BrunoHays/Bangor-Miami-Spanish-English-Corpus` on the Hugging Face Hub under
CC BY-SA 3.0.

The `mixed` configuration is the one worth evaluating against: five recordings,
about 2.5 hours, each between 20% and 80% Spanish. A file that is 98% English
tells us nothing about code-switching.

Audio is downloaded on demand into a cache directory and never committed. It is
116 MB, it is not ours, and a corpus does not belong in source control.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from audio_transcriber.errors import ConfigError

DATASET = "BrunoHays/Bangor-Miami-Spanish-English-Corpus"
LICENSE = "CC BY-SA 3.0"

_MIXED_URL = (
    "https://huggingface.co/datasets/BrunoHays/"
    "Bangor-Miami-Spanish-English-Corpus/resolve/main/"
    "mixed/train-00000-of-00001.parquet"
)

#: Smallest plausible size for the parquet, as a guard against a truncated
#: download or an HTML error page saved under the right filename.
_MIN_PARQUET_BYTES = 100 * 1024 * 1024

_PARQUET_MAGIC = b"PAR1"


@dataclass(frozen=True, slots=True)
class Recording:
    """One recording from the corpus.

    Attributes:
        name: Recording identifier, e.g. `zeledon4`.
        audio_path: Path to the extracted MP3.
        transcript: Ground-truth transcript, still carrying CHAT markup.
        spanish_share: Fraction of Spanish content words, from the corpus's own
            word-level language annotations.
        duration_s: Length of the recording in seconds.
    """

    name: str
    audio_path: Path
    transcript: str
    spanish_share: float
    duration_s: float


def cache_directory() -> Path:
    """Return the directory holding downloaded corpus data.

    Honors `AUDIO_TRANSCRIBER_EVAL_CACHE` so a shared machine can point several
    checkouts at one copy.
    """
    override = os.environ.get("AUDIO_TRANSCRIBER_EVAL_CACHE")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "audio-transcriber" / "bangor-miami"


def download(destination: Path | None = None, force: bool = False) -> Path:
    """Download the `mixed` parquet, unless it is already cached.

    Args:
        destination: Cache directory. Defaults to `cache_directory()`.
        force: Re-download even when a valid cached copy exists.

    Returns:
        Path to the parquet file.

    Raises:
        ConfigError: If the download fails or produces something that is not a
            parquet file.
    """
    directory = destination or cache_directory()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "mixed.parquet"

    if path.exists() and not force and _is_valid_parquet(path):
        return path

    try:
        urllib.request.urlretrieve(_MIXED_URL, path)
    except (urllib.error.URLError, OSError) as error:
        raise ConfigError(
            f"could not download the {DATASET} corpus: {error}. "
            "Check network access to huggingface.co, or set "
            "AUDIO_TRANSCRIBER_EVAL_CACHE to a directory holding mixed.parquet."
        ) from error

    if not _is_valid_parquet(path):
        size = path.stat().st_size if path.exists() else 0
        path.unlink(missing_ok=True)
        raise ConfigError(
            f"the download from huggingface.co was not a parquet file ({size} bytes). "
            "The dataset may have moved, or a proxy returned an error page."
        )
    return path


def _is_valid_parquet(path: Path) -> bool:
    """Report whether a file looks like the expected parquet.

    Checks the magic bytes as well as the size: a captive portal or auth page
    saved under this filename would otherwise pass a size check on a slow link
    and fail much later with a confusing parquet error.
    """
    if not path.exists() or path.stat().st_size < _MIN_PARQUET_BYTES:
        return False
    with path.open("rb") as handle:
        return handle.read(4) == _PARQUET_MAGIC


def load(destination: Path | None = None) -> list[Recording]:
    """Return every recording in the `mixed` subset, extracting audio on demand.

    Audio is written out beside the parquet once and reused, since decoding it
    from the parquet on every run wastes time for no benefit.

    Args:
        destination: Cache directory. Defaults to `cache_directory()`.

    Returns:
        The recordings, ordered by name.

    Raises:
        ConfigError: If the corpus cannot be downloaded or read.
    """
    import pyarrow.parquet as pq

    directory = destination or cache_directory()
    parquet = download(directory)
    table = pq.ParquetFile(parquet).read().to_pydict()

    recordings: list[Recording] = []
    for index, name in enumerate(table["file_name"]):
        audio_path = directory / f"{name}.mp3"
        if not audio_path.exists():
            audio_path.write_bytes(table["audio"][index]["bytes"])
        recordings.append(
            Recording(
                name=name,
                audio_path=audio_path,
                transcript=table["transcript"][index],
                spanish_share=float(table["spanish_share"][index]),
                duration_s=float(table["duration_s"][index]),
            )
        )
    return sorted(recordings, key=lambda item: item.name)
