from collections.abc import Callable

import pytest
from pupil_labs.neon_recording import NeonRecording
from PySide6.QtCore import QPointF
from pytestqt.qtbot import QtBot

from pupil_labs.neon_player.settings import RecordingSettings
from pupil_labs.neon_player.ui.timeline_dock import TimeLineDock
from pupil_labs.neon_player.ui.timeline_dock_components import (
    TimelinePlotItem,
    TrimEndMarker,
)
from tests.conftest import MockNeonPlayerApp


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


def test_y_limit_labels_are_visible_only_for_hovered_timeline_row(
    qtbot: QtBot, qapp: MockNeonPlayerApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(qapp, "recording", None)
    monkeypatch.setattr(qapp, "playback_speed_options", [-1.0, 1.0], raising=False)
    timeline = TimeLineDock()
    qtbot.addWidget(timeline)
    first_row = timeline.get_timeline_plot("First", create_if_missing=True)
    second_row = timeline.get_timeline_plot("Second", create_if_missing=True)
    first_row.configure_y_limit_labels(True)
    second_row.configure_y_limit_labels(True)
    timeline.resize(800, 500)
    timeline.show()
    qtbot.waitExposed(timeline)

    def row_center(row: TimelinePlotItem) -> QPointF:
        return row.sceneBoundingRect().center()

    timeline.on_chart_area_mouse_moved(row_center(first_row))
    assert all(label.isVisible() for label in first_row._y_range_labels)
    assert not any(label.isVisible() for label in second_row._y_range_labels)

    timeline.on_chart_area_mouse_moved(row_center(second_row))
    assert not any(label.isVisible() for label in first_row._y_range_labels)
    assert all(label.isVisible() for label in second_row._y_range_labels)

    timeline.on_chart_area_mouse_moved(QPointF(-1000, -1000))
    assert not any(label.isVisible() for label in first_row._y_range_labels)
    assert not any(label.isVisible() for label in second_row._y_range_labels)


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
