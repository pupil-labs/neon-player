import numpy as np
import pytest

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication
from unittest.mock import PropertyMock

from pupil_labs.neon_recording import NeonRecording
from pupil_labs.neon_recording.timeseries.events import EventArray, EventArray, EventTimeseries


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


@pytest.fixture(autouse=False)
def mock_event_timeseries():
    def inner(events_dict):
        all_events = []
        for event_name, timestamps in events_dict.items():
            for ts in timestamps:
                all_events.append((ts, event_name))
        all_events = sorted(all_events, key=lambda x: x[0])

        data = np.array([
            np.void(
                (ts, event_name),
                dtype=[("time", np.int64), ("event", np.str_, 50)]
            )
            for ts, event_name in all_events
        ])
        data = data.view(EventArray)

        return data

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


@pytest.fixture(scope="session")
def qapp_cls():
    return MockNeonPlayerApp
