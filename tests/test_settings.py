import json
from collections.abc import Callable

import pytest
from pupil_labs.neon_recording import NeonRecording
from PySide6.QtWidgets import QApplication

from pupil_labs.neon_player.settings import RecordingSettings, load_recording_settings


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
    qapp: QApplication,
    mock_neon_recording: Callable[..., NeonRecording],
) -> None:
    contents = {"export_window": [0], "enabled_plugins": {"Plugin": True}}
    rec = mock_neon_recording(start_time=0, stop_time=10)
    settings = load_recording_settings(mock_settings_path(contents, corrupt=True), rec)

    # Set export window to the recording's [start_time, stop_time]
    assert (
        settings.enabled_plugins["DefaultPlugin"]
        == qapp.settings.default_plugins["DefaultPlugin"]
    )


@pytest.mark.parametrize(("window", "expected"), [([8, 2], (0, 10)), ([5, 5], (5, 5))])
def test_export_window_order(
    mock_neon_recording: Callable[..., NeonRecording],
    window: list[int],
    expected: tuple[int, int],
) -> None:
    rec = mock_neon_recording(start_time=0, stop_time=10)
    settings = load_recording_settings(
        mock_settings_path({"export_window": window, "enabled_plugins": {}}), rec
    )
    assert settings.export_window == expected


def test_recording_settings_restore_opaque_plugin_states() -> None:
    state = {
        "enabled_plugins": {"ExamplePlugin": True},
        "plugin_states": {"ExamplePlugin": {"enabled": True, "values": [1, 2]}},
        "export_window": [2, 5],
    }
    settings = RecordingSettings.from_dict(state)
    assert settings.plugin_states == state["plugin_states"]
    assert settings.export_window == (2, 5)
