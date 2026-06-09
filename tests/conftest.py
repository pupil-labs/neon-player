import pytest

from unittest.mock import PropertyMock

from pupil_labs.neon_recording import NeonRecording
from pupil_labs.neon_recording.timeseries.events import EventTimeseries


@pytest.fixture(autouse=False)
def mock_neon_recording(tmp_path):
    def inner(**kwargs):
        # Use test folder to initialize mock NeonRecording
        rec = NeonRecording(tmp_path)

        # Mock properties of the recording as needed
        for key, value in kwargs.items():
            mock_value = value.copy()
            if key == "events":
                mock_value = EventTimeseries(recording=rec, data=value)

            setattr(type(rec), key, PropertyMock(return_value=mock_value))

        return rec

    return inner
