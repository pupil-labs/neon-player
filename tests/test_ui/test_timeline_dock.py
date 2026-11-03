from collections.abc import Callable

import pytest
from pupil_labs.neon_recording import NeonRecording
from PySide6.QtCore import QPoint, QPointF
from pytestqt.qtbot import QtBot

from pupil_labs.neon_player.settings import RecordingSettings
from pupil_labs.neon_player.ui.timeline_dock import TimeLineDock
from pupil_labs.neon_player.ui.timeline_dock_components import (
    TimelinePinnedAxis,
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


def test_pinned_time_axis_tracks_range_and_recording_lifecycle(
    qtbot: QtBot,
    qapp: MockNeonPlayerApp,
    monkeypatch: pytest.MonkeyPatch,
    mock_neon_recording: Callable[..., NeonRecording],
) -> None:
    monkeypatch.setattr(qapp, "recording", None)
    monkeypatch.setattr(qapp, "playback_speed_options", [-1.0, 1.0], raising=False)
    timeline = TimeLineDock()
    qtbot.addWidget(timeline)

    assert timeline.timestamps_plot.getAxis("top").isVisible()
    assert isinstance(timeline.pinned_axis, TimelinePinnedAxis)
    assert not timeline.pinned_axis.isHidden()
    assert (
        timeline.pinned_axis.axis_layout.getItem(0, 0) is timeline.pinned_axis.time_axis
    )
    assert (
        timeline.pinned_axis.time_axis.linkedView()
        is timeline.timestamps_plot.getViewBox()
    )

    recording = mock_neon_recording(start_time=100, stop_time=200)
    monkeypatch.setattr(qapp, "recording", recording)
    monkeypatch.setattr(qapp, "recording_settings", RecordingSettings())
    timeline.on_recording_loaded(recording)
    timeline.timestamps_plot.setXRange(120, 180, padding=0)

    assert timeline.pinned_axis.time_axis.recording_start_time_ns == 100
    assert timeline.pinned_axis.time_axis.recording_stop_time_ns == 200
    assert timeline.pinned_axis.time_axis.linkedView().viewRange()[0] == pytest.approx(
        timeline.timestamps_plot.viewRange()[0]
    )

    timeline.on_recording_unloaded()
    assert timeline.pinned_axis.time_axis.recording_start_time_ns == 0
    assert timeline.pinned_axis.time_axis.recording_stop_time_ns == 0


def test_pinned_time_axis_stays_in_place_when_timeline_scrolls(
    qtbot: QtBot, qapp: MockNeonPlayerApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(qapp, "recording", None)
    monkeypatch.setattr(qapp, "playback_speed_options", [-1.0, 1.0], raising=False)
    timeline = TimeLineDock()
    qtbot.addWidget(timeline)
    timeline.resize(500, 180)
    timeline.show()
    qtbot.wait(50)
    assert timeline.graphics_view.width() == timeline.scroll_area.viewport().width()
    assert timeline.pinned_axis.width() == (
        timeline.scroll_area.viewport().width() + timeline.scroll_area.frameWidth()
    )
    assert (
        timeline.pinned_axis.mapTo(
            timeline,
            QPoint(timeline.scroll_area.frameWidth(), 0),
        ).x()
        == timeline.scroll_area.viewport().mapTo(timeline, QPoint(0, 0)).x()
    )
    for row in range(10):
        timeline.get_timeline_plot(f"row {row}", create_if_missing=True)
    timeline.fix_scroll_size()
    qtbot.wait(50)

    main_plot_item = timeline.get_timeline_plot("row 0")
    assert main_plot_item is not None
    main_plot = main_plot_item.getViewBox().sceneBoundingRect()
    pinned_plot = timeline.pinned_axis.time_axis.sceneBoundingRect()
    main_plot_left = timeline.graphics_view.mapTo(
        timeline, timeline.graphics_view.mapFromScene(main_plot.topLeft())
    ).x()
    pinned_plot_left = timeline.pinned_axis.axis_view.mapTo(
        timeline, timeline.pinned_axis.axis_view.mapFromScene(pinned_plot.topLeft())
    ).x()
    main_plot_right = timeline.graphics_view.mapTo(
        timeline,
        timeline.graphics_view.mapFromScene(main_plot.topRight()),
    ).x()
    pinned_plot_right = timeline.pinned_axis.axis_view.mapTo(
        timeline,
        timeline.pinned_axis.axis_view.mapFromScene(pinned_plot.topRight()),
    ).x()
    assert pinned_plot_left == main_plot_left
    assert pinned_plot_right == main_plot_right

    timeline.resize(650, 180)
    qtbot.wait(50)
    main_plot = main_plot_item.getViewBox().sceneBoundingRect()
    pinned_plot = timeline.pinned_axis.time_axis.sceneBoundingRect()
    main_plot_left = timeline.graphics_view.mapTo(
        timeline, timeline.graphics_view.mapFromScene(main_plot.topLeft())
    ).x()
    pinned_plot_left = timeline.pinned_axis.axis_view.mapTo(
        timeline, timeline.pinned_axis.axis_view.mapFromScene(pinned_plot.topLeft())
    ).x()
    main_plot_right = timeline.graphics_view.mapTo(
        timeline,
        timeline.graphics_view.mapFromScene(main_plot.topRight()),
    ).x()
    pinned_plot_right = timeline.pinned_axis.axis_view.mapTo(
        timeline,
        timeline.pinned_axis.axis_view.mapFromScene(pinned_plot.topRight()),
    ).x()
    assert pinned_plot_left == main_plot_left
    assert pinned_plot_right == main_plot_right

    scrollbar = timeline.scroll_area.verticalScrollBar()
    assert scrollbar.maximum() > 0
    axis_y_before = timeline.pinned_axis.axis_view.mapTo(timeline, QPoint(0, 0)).y()
    scrollbar.setValue(scrollbar.maximum())
    qtbot.wait(50)
    axis_y_after = timeline.pinned_axis.axis_view.mapTo(timeline, QPoint(0, 0)).y()

    assert scrollbar.value() == scrollbar.maximum()
    assert axis_y_after == axis_y_before
