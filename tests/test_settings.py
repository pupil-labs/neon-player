import json
from collections.abc import Callable

from pupil_labs.neon_recording import NeonRecording

from pupil_labs.neon_player.settings import load_recording_settings


class MockSettingsPath:
    """A minimal mock of a Path object that returns settings-like text contents."""

    def __init__(self, contents: dict, corrupt: bool = False) -> None:
        self.contents = contents
        self.corrupt = corrupt

    def exists(self) -> bool:
        return True

    def read_text(self) -> str:
        result = json.dumps(self.contents)
        return result[:10] if self.corrupt else result


def mock_settings_path(contents: dict, corrupt: bool = False) -> MockSettingsPath:
    return MockSettingsPath(contents, corrupt)


def test_load_recording_settings__valid(
    mock_neon_recording: Callable[..., NeonRecording],
) -> None:
    contents = {"export_window": [2, 5], "enabled_plugins": {"Plugin": True}}
    rec = mock_neon_recording(start_time=0, stop_time=10)
    settings = load_recording_settings(mock_settings_path(contents), rec)

    assert settings.export_window == (2, 5)
    assert settings.enabled_plugins["Plugin"]


def test_load_recording_settings__invalid_export_window(
    mock_neon_recording: Callable[..., NeonRecording],
) -> None:
    contents = {"export_window": [0], "enabled_plugins": {"Plugin": True}}
    rec = mock_neon_recording(start_time=0, stop_time=10)
    settings = load_recording_settings(mock_settings_path(contents), rec)

    assert settings.export_window == (0, 10)
    assert settings.enabled_plugins["Plugin"]


def test_load_recording_settings__corrupted_settings(
    mock_neon_recording: Callable[..., NeonRecording],
) -> None:
    contents = {"export_window": [0], "enabled_plugins": {"Plugin": True}}
    rec = mock_neon_recording(start_time=0, stop_time=10)
    settings = load_recording_settings(mock_settings_path(contents, corrupt=True), rec)

    # Set export window to the recording's [start_time, stop_time]
    assert settings.enabled_plugins["DefaultPlugin"]
