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


@pytest.fixture(scope="session")
def qapp_cls():
    from pupil_labs.neon_player.app import NeonPlayerApp

    MockNeonPlayerApp.toggle_plugin = NeonPlayerApp.toggle_plugin

    return MockNeonPlayerApp

import sysconfig
import subprocess
from pathlib import Path

@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

@pytest.fixture(scope="session")
def native_extension(repo_root: Path) -> Path:
    src = repo_root / "tests" / "fixtures" / "plugin_site" / "demo_pkg" / "native_missing.c"
    out_dir = src.parent
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not suffix:
        pytest.skip("Python EXT_SUFFIX is unavailable")
    out = out_dir / f"native_missing{suffix}"
    include = sysconfig.get_paths()["include"]
    cmd = ["cc", "-shared", "-fPIC", f"-I{include}", str(src), "-o", str(out)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"C compiler/native headers unavailable: {exc}")
    return out
