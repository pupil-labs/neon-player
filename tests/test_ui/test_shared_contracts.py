from collections.abc import Callable

import pytest
from pupil_labs.neon_recording import NeonRecording
from PySide6.QtCore import QObject, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QMainWindow, QMenu
from pytestqt.qtbot import QtBot
from qt_property_widgets.utilities import ActionObject

from pupil_labs.neon_player.job_manager import BackgroundJob
from pupil_labs.neon_player.settings import RecordingSettings
from pupil_labs.neon_player.ui import GUIEventNotifier
from pupil_labs.neon_player.ui.main_window import MainWindow
from pupil_labs.neon_player.ui.progress_action_form import ProgressActionForm
from pupil_labs.neon_player.ui.timeline_dock import TimeLineDock
from pupil_labs.neon_player.ui.timeline_dock_components import TrimEndMarker
from tests.conftest import MockNeonPlayerApp


def test_event_notifier_emits_mouse_click(qtbot: QtBot) -> None:
    widget = GUIEventNotifier()
    qtbot.addWidget(widget)
    event = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(1.25, 2.5),
        QPointF(1.25, 2.5),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    with qtbot.waitSignal(widget.mouse_pressed) as signal:
        widget.mousePressEvent(event)
    assert signal.args[0].position() == QPointF(1.25, 2.5)
    with qtbot.waitSignal(widget.mouse_clicked):
        widget.mouseReleaseEvent(event)


def test_timeline_callbacks_without_recording(
    qtbot: QtBot, qapp: MockNeonPlayerApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(qapp, "recording", None)
    monkeypatch.setattr(qapp, "playback_speed_options", [-1.0, 1.0], raising=False)
    timeline = TimeLineDock()
    qtbot.addWidget(timeline)
    timeline.on_position_changed(123)
    timeline.reset_view()
    timeline.on_chart_area_mouse_moved(QPointF(0, 0))
    assert timeline.playhead.t == 0
    assert timeline.trim_markers == []


def test_trim_markers_preserve_timestamps_and_empty_export_window(
    qtbot: QtBot, qapp: MockNeonPlayerApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(qapp, "recording", None)
    monkeypatch.setattr(qapp, "playback_speed_options", [-1.0, 1.0], raising=False)
    timeline = TimeLineDock()
    qtbot.addWidget(timeline)
    timestamp = 1_700_000_000_000_000_123
    timeline.trim_markers = [
        TrimEndMarker(timestamp, timeline.timestamps_plot),
        TrimEndMarker(timestamp, timeline.timestamps_plot),
    ]
    assert timeline.get_export_window() == (timestamp, timestamp)
    timeline.trim_markers[0].time = 123.75
    assert timeline.trim_markers[0].time == 123.75
    timeline.set_export_window((timestamp, timestamp + 1))
    assert timeline.get_export_window() == (timestamp, timestamp + 1)


def test_menu_lookup_keeps_missing_and_existing_menus_distinct(qtbot: QtBot) -> None:
    window = MainWindow.__new__(MainWindow)
    QMainWindow.__init__(window)
    qtbot.addWidget(window)
    assert window.get_menu("Timeline", auto_create=False) is None
    menu = window.get_menu("Timeline")
    assert isinstance(menu, QMenu)
    assert window.get_menu("Timeline", auto_create=False) is menu


def test_timeline_unload_and_reload_replace_recording_markers(
    qtbot: QtBot,
    qapp: MockNeonPlayerApp,
    monkeypatch: pytest.MonkeyPatch,
    mock_neon_recording: Callable[..., NeonRecording],
) -> None:
    monkeypatch.setattr(qapp, "recording", None)
    monkeypatch.setattr(qapp, "playback_speed_options", [-1.0, 1.0], raising=False)
    timeline = TimeLineDock()
    qtbot.addWidget(timeline)
    settings = RecordingSettings()
    monkeypatch.setattr(qapp, "recording_settings", settings)
    recording = mock_neon_recording(start_time=100, stop_time=200)
    monkeypatch.setattr(qapp, "recording", recording)
    settings.export_window = (120, 180)
    timeline.on_recording_loaded(recording)
    assert timeline.get_export_window() == (120, 180)
    timeline.dragging = timeline.trim_markers[0]
    monkeypatch.setattr(qapp, "recording", None)
    timeline.on_recording_unloaded()
    timeline.on_chart_area_mouse_moved(QPointF(0, 0))
    assert timeline.trim_markers == []
    assert timeline.dragging is None
    recording = mock_neon_recording(start_time=300, stop_time=400)
    monkeypatch.setattr(qapp, "recording", recording)
    settings.export_window = (350, 350)
    timeline.on_recording_loaded(recording)
    assert timeline.get_export_window() == (350, 350)


def test_progress_action_invokes_framework_action_and_tracks_job(qtbot: QtBot) -> None:
    job = BackgroundJob.__new__(BackgroundJob)
    QObject.__init__(job)
    calls: list[bool] = []

    def start_job() -> BackgroundJob:
        calls.append(True)
        return job

    form = ProgressActionForm(ActionObject(start_job, None))
    qtbot.addWidget(form)
    form.action_button.click()
    assert calls == [True]
    assert not form.progress_bar.isHidden()
    job.progress_changed.emit(0.375)
    assert form.progress_bar.value() == 37
    job.finished.emit()
    assert form.progress_bar.isHidden()
    assert not form.action_button.isHidden()
