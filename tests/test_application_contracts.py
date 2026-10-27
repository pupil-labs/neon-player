from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from qt_property_widgets.utilities import create_action_object

from pupil_labs import neon_player
from pupil_labs.neon_player import Plugin, action
from pupil_labs.neon_player import app as app_module
from pupil_labs.neon_player import job_manager as job_module
from pupil_labs.neon_player.app import NeonPlayerApp
from pupil_labs.neon_player.job_manager import JobManager
from pupil_labs.neon_player.settings import GeneralSettings, RecordingSettings


def test_optional_application_before_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(NeonPlayerApp, "instance", staticmethod(lambda: None))
    assert neon_player.instance(required=False) is None
    with pytest.raises(RuntimeError, match="not initialized"):
        Plugin.get_instance_by_name("missing")
    with pytest.raises(RuntimeError, match="not initialized"):
        neon_player.instance()
    plugin = Plugin()
    with pytest.raises(RuntimeError, match="not initialized"):
        _ = plugin.recording
    with pytest.raises(RuntimeError, match="not initialized"):
        _ = plugin.headless


def test_optional_application_rejects_other_qt_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(NeonPlayerApp, "instance", staticmethod(object))
    with pytest.raises(RuntimeError, match="not initialized"):
        neon_player.instance()


def test_initialized_application_and_plugin_lookup(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert neon_player.instance() is qapp
    plugin = Plugin()
    monkeypatch.setattr(qapp, "plugins_by_class", {"Plugin": plugin})
    assert Plugin.get_instance_by_name("Plugin") is plugin
    assert Plugin.get_instance_by_name("missing") is None


def test_action_binding_and_property_framework_introspection() -> None:
    class Actions:
        @action
        def scale(self, value: int = 3) -> int:
            return value * 2

    instance = Actions()
    assert instance.scale(4) == 8
    action_object = create_action_object(Actions._actions["scale"], instance)
    assert action_object.args == {"value": 3}
    assert action_object() == 6
    action_object.__setstate__({"value": "5"})
    assert action_object() == 10


def test_no_recording_controls(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(qapp, "recording", None)
    monkeypatch.setattr(qapp, "recording_settings", None)
    monkeypatch.setattr(qapp, "playback_speed", 1.0, raising=False)
    plugin = Plugin()
    assert plugin.recording is None
    assert plugin.headless == qapp.headless
    assert plugin.get_cache_path() is None
    assert plugin.load_cached_json("missing.json") is None
    plugin.save_cached_json("unused.json", {})
    with pytest.raises(RuntimeError, match="loaded recording"):
        plugin.require_recording()
    with pytest.raises(AssertionError, match="Recording must be loaded"):
        qapp._reset_start_anchor()
    with pytest.raises(AssertionError, match="Recording must be loaded"):
        qapp.get_scene_idx_for_time(123)
    with pytest.raises(AssertionError, match="Recording must be loaded"):
        qapp.get_export_window()
    with pytest.raises(AssertionError, match="Recording must be loaded"):
        qapp.set_export_window((123, 456))
    assert JobManager().run_background_action("test", "Plugin.export") is None


def test_unload_clears_recording_settings(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording = Mock()
    monkeypatch.setattr(qapp, "recording", recording)
    monkeypatch.setattr(qapp, "recording_settings", RecordingSettings())
    monkeypatch.setattr(qapp, "plugins_by_class", {})
    monkeypatch.setattr(qapp, "save_settings", Mock())
    monkeypatch.setattr(qapp, "set_playback_state", Mock())
    monkeypatch.setattr(
        qapp, "main_window", SimpleNamespace(timeline=Mock()), raising=False
    )
    qapp.unload()
    recording.close.assert_called_once()
    assert qapp.recording is None
    assert qapp.recording_settings is None
    with pytest.raises(AssertionError, match="Recording must be loaded"):
        qapp.get_export_window()


@pytest.mark.parametrize("window", [(123, 456), (123, 123)])
def test_export_window_preserves_timestamps(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch, window: tuple[int, int]
) -> None:
    settings = RecordingSettings()
    monkeypatch.setattr(qapp, "recording", Mock())
    monkeypatch.setattr(qapp, "recording_settings", settings)
    timeline = Mock()
    monkeypatch.setattr(
        qapp, "main_window", SimpleNamespace(timeline=timeline), raising=False
    )
    qapp.set_export_window(window)
    assert qapp.get_export_window() == window
    timeline.set_export_window.assert_called_once_with(window)
    with pytest.raises(ValueError, match="cannot follow"):
        qapp.set_export_window((456, 123))
    assert qapp.get_export_window() == window


def test_load_switch_and_delayed_callbacks_after_unload(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recordings = [
        Mock(start_time=100, stop_time=200),
        Mock(start_time=300, stop_time=400),
    ]
    settings = [RecordingSettings(), RecordingSettings()]
    settings[0].export_window = (100, 200)
    settings[1].export_window = (300, 400)
    loader = Mock(side_effect=recordings)
    monkeypatch.setattr(app_module.nr, "load", loader)
    monkeypatch.setattr(
        app_module, "load_recording_settings", Mock(side_effect=settings)
    )
    monkeypatch.setattr(app_module.os, "chdir", Mock())
    monkeypatch.setattr(QTimer, "singleShot", Mock())
    monkeypatch.setattr(qapp, "save_settings", Mock())
    monkeypatch.setattr(qapp, "set_playback_state", Mock())
    monkeypatch.setattr(qapp, "seek_to", Mock())
    monkeypatch.setattr(qapp, "recording_history", Mock(), raising=False)
    general = GeneralSettings()
    general.skip_gray_frames_on_load = False
    monkeypatch.setattr(qapp, "settings", general)
    monkeypatch.setattr(qapp, "recording", None)
    monkeypatch.setattr(qapp, "recording_settings", None)
    monkeypatch.setattr(qapp, "plugins_by_class", {})
    monkeypatch.setattr(qapp, "loading_recording", False, raising=False)
    monkeypatch.setattr(qapp, "_initializing", False, raising=False)
    monkeypatch.setattr(qapp, "playback_start_anchor", 0, raising=False)
    monkeypatch.setattr(qapp, "main_window", Mock(), raising=False)
    qapp.load(tmp_path)
    assert qapp.recording is recordings[0]
    assert qapp.get_export_window() == (100, 200)
    qapp.load(tmp_path)
    recordings[0].close.assert_called_once()
    assert qapp.recording is recordings[1]
    assert qapp.get_export_window() == (300, 400)
    qapp.unload()
    recordings[1].close.assert_called_once()
    with pytest.raises(AssertionError, match="Recording must be loaded"):
        qapp.on_recording_load_complete()
    with pytest.raises(AssertionError, match="Recording settings must be loaded"):
        qapp.toggle_plugins_by_settings()
    assert qapp.recording_settings is None


def test_recordingless_dependency_job_keeps_command_contract(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    server_class = Mock()
    process = Mock()
    monkeypatch.setattr(job_module, "QLocalServer", server_class)
    monkeypatch.setattr(job_module.subprocess, "Popen", process)
    monkeypatch.setattr(qapp, "args", SimpleNamespace(job=None))
    monkeypatch.setattr(qapp, "recording", None)
    monkeypatch.setattr(qapp, "save_settings", Mock())
    job = JobManager().run_background_action(
        "Install dependencies", "install_packages", "pkg"
    )
    assert job is not None
    command = process.call_args.args[0]
    assert command[3] == "None"
    assert command[-2:] == ["install_packages", "pkg"]
