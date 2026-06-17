import pytest

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication
from unittest.mock import PropertyMock

from pupil_labs.neon_recording import NeonRecording
from pupil_labs.neon_recording.timeseries.events import EventTimeseries


@pytest.fixture(autouse=False)
def mock_neon_recording(tmp_path):
    def inner(**kwargs):
        # Use a temporary folder to initialize mock NeonRecording
        rec = NeonRecording(tmp_path)

        # Mock properties of the recording as needed
        for key, value in kwargs.items():
            mock_value = value.copy()
            if key == "events":
                mock_value = EventTimeseries(recording=rec, data=value)

            setattr(type(rec), key, PropertyMock(return_value=mock_value))

        return rec

    return inner


class MockNeonPlayerApp(QApplication):
    export_window_changed = Signal(tuple[int, int])

    def __init__(self, *args):
        super().__init__(*args)
        self.headless = True
        self.plugins_by_class = {}


@pytest.fixture(scope="session")
def qapp_cls():
    return MockNeonPlayerApp
