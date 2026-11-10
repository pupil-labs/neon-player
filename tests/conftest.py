import subprocess
import sysconfig
from collections.abc import Callable
from pathlib import Path
from unittest.mock import PropertyMock

import pytest
from pupil_labs.neon_recording import NeonRecording
from pupil_labs.neon_recording.timeseries import (
    EventTimeseries,
    GazeTimeseries,
    SceneVideoTimeseries,
    WornTimeseries,
)
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication

from pupil_labs.neon_player.app import NeonPlayerApp
from pupil_labs.neon_player.history import RecordingHistory
from pupil_labs.neon_player.job_manager import JobManager
from pupil_labs.neon_player.managers.update import UpdateManager
from pupil_labs.neon_player.settings import GeneralSettings

FIELD_CLASS_MAPPING = {
    "events": EventTimeseries,
    "gaze": GazeTimeseries,
    "scene": SceneVideoTimeseries,
    "worn": WornTimeseries,
}


@pytest.fixture(autouse=False)
def mock_neon_recording(tmp_path: Path) -> Callable[..., NeonRecording]:
    def inner(**kwargs: object) -> NeonRecording:
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


class MockNeonPlayerApp(NeonPlayerApp):
    """
    Mock NeonPlayerApp to be used in tests that rely on the presence of an application
    instance. Properties are replaced with ordinary fields that can be set directly
    before executing the respective test.
    """

    export_window_changed = Signal(tuple[int, int])
    playback_state_changed = Signal(bool)
    position_changed = Signal(object)
    seeked = Signal(object)
    speed_changed = Signal(float)
    recording_loaded = Signal(object)
    recording_unloaded = Signal()

    def __init__(self, argv: list[str]) -> None:
        QApplication.__init__(self, argv)
        self.args = type("Args", (), {"job": ["test"]})()
        self.recording_settings = None
        self.plugins_by_class = {}
        self.recording = None
        self.recording_history = RecordingHistory()
        self.job_manager = JobManager()
        self.settings = GeneralSettings()
        self.settings.default_plugins = {"DefaultPlugin": True}
        self.update_manager = UpdateManager(self)
        self.playback_speed_options = []


@pytest.fixture(scope="session")
def qapp_cls() -> type[MockNeonPlayerApp]:
    return MockNeonPlayerApp


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def native_extension(repo_root: Path) -> Path:
    src = (
        repo_root
        / "tests"
        / "fixtures"
        / "plugin_site"
        / "demo_pkg"
        / "native_missing.c"
    )
    out_dir = src.parent
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not suffix:
        pytest.skip("Python EXT_SUFFIX is unavailable")
    out = out_dir / f"native_missing{suffix}"
    include = sysconfig.get_paths()["include"]
    cmd = ["cc", "-shared", "-fPIC", f"-I{include}", str(src), "-o", str(out)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"C compiler/native headers unavailable: {exc}")
    return out
