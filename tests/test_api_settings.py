"""Tests for the API settings read from the environment."""

from pathlib import Path

import pytest

from audio_transcriber.api.settings import ApiSettings
from audio_transcriber.errors import ConfigError


def test_defaults_run_one_job_at_a_time():
    settings = ApiSettings.from_env({})

    assert settings.concurrency == 1
    assert settings.max_upload_bytes == 1024 * 1024 * 1024
    assert settings.uploads_dir.parent == settings.work_dir
    assert settings.runs_dir.parent == settings.work_dir


def test_the_environment_overrides_the_defaults(tmp_path):
    settings = ApiSettings.from_env(
        {
            "AUDIO_TRANSCRIBER_MAX_UPLOAD_MB": "5",
            "AUDIO_TRANSCRIBER_CONCURRENCY": "3",
            "AUDIO_TRANSCRIBER_WORK_DIR": str(tmp_path),
        }
    )

    assert settings.max_upload_bytes == 5 * 1024 * 1024
    assert settings.concurrency == 3
    assert settings.work_dir == Path(tmp_path)


@pytest.mark.parametrize("value", ["0", "-2", "many", "1.5"])
@pytest.mark.parametrize(
    "name", ["AUDIO_TRANSCRIBER_MAX_UPLOAD_MB", "AUDIO_TRANSCRIBER_CONCURRENCY"]
)
def test_an_unusable_number_is_a_config_error_naming_the_variable(name, value):
    with pytest.raises(ConfigError, match=name):
        ApiSettings.from_env({name: value})


def test_concurrency_below_one_is_refused_when_set_directly():
    with pytest.raises(ConfigError, match="concurrency"):
        ApiSettings(concurrency=0)
