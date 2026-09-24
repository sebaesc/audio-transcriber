"""Tests for corpus fetching and caching.

No network: what is tested is the validation that keeps a bad download from
looking like a good one, which is how the TalkBank auth page would have arrived.
"""

from pathlib import Path

import pytest

from audio_transcriber.errors import ConfigError
from audio_transcriber.evaluation import corpus


def _write_parquet(path: Path, size: int = 200 * 1024 * 1024) -> Path:
    """Write a file that passes both the magic-byte and size checks."""
    with path.open("wb") as handle:
        handle.write(b"PAR1")
        handle.seek(size - 1)
        handle.write(b"\0")
    return path


def test_cache_directory_honors_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("AUDIO_TRANSCRIBER_EVAL_CACHE", str(tmp_path / "shared"))

    assert corpus.cache_directory() == tmp_path / "shared"


def test_cache_directory_defaults_under_home(monkeypatch):
    monkeypatch.delenv("AUDIO_TRANSCRIBER_EVAL_CACHE", raising=False)

    assert "audio-transcriber" in str(corpus.cache_directory())


def test_a_cached_parquet_is_not_downloaded_again(monkeypatch, tmp_path):
    _write_parquet(tmp_path / "mixed.parquet")

    def _fail(*args, **kwargs):
        raise AssertionError("should not download when a valid copy is cached")

    monkeypatch.setattr(corpus.urllib.request, "urlretrieve", _fail)

    assert corpus.download(tmp_path) == tmp_path / "mixed.parquet"


def test_an_html_error_page_is_rejected(monkeypatch, tmp_path):
    """TalkBank returns an auth page with HTTP 200; a size check alone is not enough."""

    def _write_html(url, path):
        Path(path).write_bytes(b"<html><body>sign in</body></html>")

    monkeypatch.setattr(corpus.urllib.request, "urlretrieve", _write_html)

    with pytest.raises(ConfigError, match="not a parquet file"):
        corpus.download(tmp_path)


def test_a_truncated_download_is_rejected(monkeypatch, tmp_path):
    def _write_short(url, path):
        Path(path).write_bytes(b"PAR1" + b"\0" * 1000)

    monkeypatch.setattr(corpus.urllib.request, "urlretrieve", _write_short)

    with pytest.raises(ConfigError, match="not a parquet file"):
        corpus.download(tmp_path)


def test_a_rejected_download_is_not_left_on_disk(monkeypatch, tmp_path):
    """A bad file left behind would be found by the next run's cache check."""

    def _write_html(url, path):
        Path(path).write_bytes(b"<html>nope</html>")

    monkeypatch.setattr(corpus.urllib.request, "urlretrieve", _write_html)

    with pytest.raises(ConfigError):
        corpus.download(tmp_path)

    assert not (tmp_path / "mixed.parquet").exists()


def test_a_network_failure_says_what_to_check(monkeypatch, tmp_path):
    def _explode(url, path):
        raise OSError("Name or service not known")

    monkeypatch.setattr(corpus.urllib.request, "urlretrieve", _explode)

    with pytest.raises(ConfigError, match="huggingface.co"):
        corpus.download(tmp_path)


def test_force_redownloads_over_a_valid_cache(monkeypatch, tmp_path):
    _write_parquet(tmp_path / "mixed.parquet")
    called = []

    def _record(url, path):
        called.append(url)
        _write_parquet(Path(path))

    monkeypatch.setattr(corpus.urllib.request, "urlretrieve", _record)
    corpus.download(tmp_path, force=True)

    assert len(called) == 1
