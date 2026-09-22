"""Tests for input discovery.

Discovery must be deterministic: reproducibility starts with two runs over the
same directory doing the same work in the same order.
"""

import pytest

from audio_transcriber.discovery import discover, is_audio
from audio_transcriber.errors import ConfigError


def _touch(directory, *names):
    for name in names:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")


def test_finds_audio_files_recursively(tmp_path):
    _touch(tmp_path, "a.wav", "nested/b.mp3", "nested/deep/c.flac")

    found = discover(tmp_path)

    assert [p.name for p in found] == ["a.wav", "b.mp3", "c.flac"]


def test_ignores_non_audio_files(tmp_path):
    _touch(tmp_path, "a.wav", "notes.txt", "cover.jpg", "script.py")

    assert [p.name for p in discover(tmp_path)] == ["a.wav"]


def test_order_is_stable_across_calls(tmp_path):
    _touch(tmp_path, "c.wav", "a.wav", "b.wav", "nested/a.wav")

    assert discover(tmp_path) == discover(tmp_path)


def test_order_is_sorted_by_path(tmp_path):
    _touch(tmp_path, "c.wav", "a.wav", "b.wav")

    assert [p.name for p in discover(tmp_path)] == ["a.wav", "b.wav", "c.wav"]


def test_extension_matching_is_case_insensitive(tmp_path):
    _touch(tmp_path, "LOUD.WAV", "quiet.Mp3")

    assert len(discover(tmp_path)) == 2


def test_explicit_pattern_overrides_extension_filtering(tmp_path):
    """Naming a pattern is explicit enough; ffmpeg decides what it can read."""
    _touch(tmp_path, "a.weird", "b.wav")

    assert [p.name for p in discover(tmp_path, "*.weird")] == ["a.weird"]


def test_pattern_can_narrow_to_a_subdirectory(tmp_path):
    _touch(tmp_path, "top.wav", "keep/a.wav", "keep/b.wav")

    assert [p.name for p in discover(tmp_path, "keep/*.wav")] == ["a.wav", "b.wav"]


def test_a_single_file_is_returned_as_is(tmp_path):
    _touch(tmp_path, "a.wav")

    assert discover(tmp_path / "a.wav") == [tmp_path / "a.wav"]


def test_a_named_file_skips_extension_filtering(tmp_path):
    """Naming one file explicitly means transcribing it, whatever it is called."""
    _touch(tmp_path, "recording.dat")

    assert discover(tmp_path / "recording.dat") == [tmp_path / "recording.dat"]


def test_directories_are_not_returned(tmp_path):
    _touch(tmp_path, "nested/a.wav")
    (tmp_path / "empty.wav").mkdir()

    assert [p.name for p in discover(tmp_path)] == ["a.wav"]


def test_empty_directory_yields_nothing(tmp_path):
    assert discover(tmp_path) == []


def test_missing_input_path_says_so(tmp_path):
    with pytest.raises(ConfigError, match="input path not found"):
        discover(tmp_path / "nope")


@pytest.mark.parametrize("name", ["a.wav", "a.MP3", "a.m4a", "a.opus", "a.flac"])
def test_is_audio_accepts_known_extensions(name, tmp_path):
    assert is_audio(tmp_path / name)


@pytest.mark.parametrize("name", ["a.txt", "a.json", "a", "a.py"])
def test_is_audio_rejects_other_extensions(name, tmp_path):
    assert not is_audio(tmp_path / name)
