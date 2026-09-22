"""Tests for audio decoding.

The decode path shells out to ffmpeg, so the tests that need a real decode are
skipped when it is absent. Error reporting is tested without it, since a clear
message matters most on the machine where ffmpeg is missing.
"""

import shutil
import subprocess

import pytest

from audio_transcriber import audio
from audio_transcriber.errors import AudioDecodeError, ConfigError

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is not installed"
)


def test_missing_ffmpeg_reports_how_to_install_it(monkeypatch):
    monkeypatch.setattr(audio.shutil, "which", lambda name: None)

    with pytest.raises(ConfigError, match="apt install ffmpeg"):
        audio.check_ffmpeg()


def test_unusable_ffmpeg_is_distinguished_from_a_missing_one(monkeypatch):
    """On PATH but broken is a different problem, and says so."""
    monkeypatch.setattr(audio.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    def _explode(*args, **kwargs):
        raise OSError("Exec format error")

    monkeypatch.setattr(audio.subprocess, "run", _explode)

    with pytest.raises(ConfigError, match="would not run"):
        audio.check_ffmpeg()


def test_decode_failure_reports_ffmpegs_own_message(monkeypatch, tmp_path):
    def _fail(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="a.m4a: Invalid data found\n"
        )

    monkeypatch.setattr(audio.subprocess, "run", _fail)

    with pytest.raises(AudioDecodeError, match="Invalid data found") as caught:
        audio.decode(tmp_path / "a.m4a", tmp_path / "out.wav")

    assert "truncated or not audio" in str(caught.value)


def test_decode_failure_without_stderr_still_says_something(monkeypatch, tmp_path):
    def _fail(*args, **kwargs):
        return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="")

    monkeypatch.setattr(audio.subprocess, "run", _fail)

    with pytest.raises(AudioDecodeError, match="without reporting why"):
        audio.decode(tmp_path / "a.m4a", tmp_path / "out.wav")


def test_decode_timeout_is_reported_as_such(monkeypatch, tmp_path):
    def _timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=600)

    monkeypatch.setattr(audio.subprocess, "run", _timeout)

    with pytest.raises(AudioDecodeError, match="timed out"):
        audio.decode(tmp_path / "a.m4a", tmp_path / "out.wav")


def test_silent_success_producing_no_audio_is_an_error(monkeypatch, tmp_path):
    """ffmpeg exiting 0 with an empty file must not look like a valid decode."""

    def _empty(*args, **kwargs):
        (tmp_path / "out.wav").write_bytes(b"")
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(audio.subprocess, "run", _empty)

    with pytest.raises(AudioDecodeError, match="produced no audio"):
        audio.decode(tmp_path / "a.m4a", tmp_path / "out.wav")


@needs_ffmpeg
def test_check_ffmpeg_passes_when_installed():
    audio.check_ffmpeg()


@needs_ffmpeg
def test_decode_produces_mono_16k_pcm(tmp_path, wav_file):
    import wave

    destination = audio.decode(wav_file, tmp_path / "out.wav")

    with wave.open(str(destination)) as handle:
        assert handle.getnchannels() == 1
        assert handle.getframerate() == 16_000
        assert handle.getsampwidth() == 2


@needs_ffmpeg
def test_decoded_context_cleans_up_after_itself(wav_file):
    with audio.decoded(wav_file) as path:
        assert path.exists()

    assert not path.exists()
