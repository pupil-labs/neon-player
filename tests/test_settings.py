import json

from pathlib import Path
from unittest.mock import patch, MagicMock

from pupil_labs.neon_player.settings import GeneralSettings, load_recording_settings


def mock_path(contents: dict, corrupt: bool = False):
    class MockPath:
        def exists(self):
            return True

        def read_text(self):
            result = json.dumps(contents)
            return result[:10] if corrupt else result

    return MockPath()


def test_load_recording_settings__valid(mock_neon_recording):
    contents = {
        "export_window": [2, 5],
        "enabled_plugins": {"Plugin": True}
    }
    rec = mock_neon_recording(start_time=0, stop_time=10)
    settings = load_recording_settings(mock_path(contents), rec)

    assert settings.export_window == (2, 5)
    assert settings.enabled_plugins["Plugin"]


def test_load_recording_settings__invalid_export_window(mock_neon_recording):
    contents = {
        "export_window": [0],
        "enabled_plugins": {"Plugin": True}
    }
    rec = mock_neon_recording(start_time=0, stop_time=10)
    settings = load_recording_settings(mock_path(contents), rec)

    assert settings.export_window == (0, 10)
    assert settings.enabled_plugins["Plugin"]


def test_load_recording_settings__corrupted_settings(mock_neon_recording):
    contents = {
        "export_window": [0],
        "enabled_plugins": {"Plugin": True}
    }
    rec = mock_neon_recording(start_time=0, stop_time=10)
    settings = load_recording_settings(mock_path(contents, corrupt=True), rec)

    # Set export window to the recording's [start_time, stop_time]
    assert settings.enabled_plugins["DefaultPlugin"]
