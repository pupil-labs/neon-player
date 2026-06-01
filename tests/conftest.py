import pytest

from pathlib import Path
from unittest.mock import PropertyMock

from pupil_labs.neon_recording import NeonRecording


@pytest.fixture(autouse=False)
def mock_neon_recording(tmp_path):
    def inner(**kwargs):
        # Use test folder to initialize mock NeonRecording
        rec = NeonRecording(tmp_path)

        # Mock properties of the recording as needed
        for key, value in kwargs.items():
            setattr(type(rec), key, PropertyMock(return_value=value))

        return rec

    return inner
