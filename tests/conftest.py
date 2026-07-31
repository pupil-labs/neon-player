import pytest

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication
from unittest.mock import PropertyMock

from pupil_labs.neon_recording import NeonRecording
from pupil_labs.neon_recording.timeseries import (
    EventTimeseries,
    GazeTimeseries,
    SceneVideoTimeseries,
    WornTimeseries
)

from pupil_labs.neon_player.settings import GeneralSettings


FIELD_CLASS_MAPPING = {
    "events": EventTimeseries,
    "gaze": GazeTimeseries,
    "scene": SceneVideoTimeseries,
    "worn": WornTimeseries,
}


@pytest.fixture(autouse=False)
def mock_neon_recording(tmp_path):
    def inner(**kwargs):
        # Use a temporary folder to initialize mock NeonRecording
        rec = NeonRecording(tmp_path)

        # Add some mock properties by default if not explicitly provided
        if "info" not in kwargs:
            kwargs["info"] = {"recording_id": "mock"}

        # Mock properties of the recording as needed
        for key, value in kwargs.items():
            mock_value = value.copy() if hasattr(value, "copy") else value
            if key in FIELD_CLASS_MAPPING:
                mock_class = FIELD_CLASS_MAPPING[key]
                mock_value = mock_class(recording=rec, data=value)

            # Setup mock video resolution            
            if key == "scene":
                type(mock_value).width = 1600
                type(mock_value).height = 1200

            setattr(type(rec), key, PropertyMock(return_value=mock_value))

        return rec

    return inner


class MockNeonPlayerApp(QApplication):
    """
    Mock NeonPlayerApp to be used in tests that rely on the presence of an application
    instance. Properties are replaced with ordinary fields that can be set directly
    before executing the respective test.
    """
    export_window_changed = Signal(tuple[int, int])

    def __init__(self, *args):
        super().__init__(*args)
        self.headless = True
        self.plugins_by_class = {}
        self.recording = None
        self.settings = GeneralSettings()
        self.settings.default_plugins = {"DefaultPlugin": True}


@pytest.fixture(scope="session")
def qapp_cls():
    return MockNeonPlayerApp
