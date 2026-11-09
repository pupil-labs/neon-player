from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from pupil_labs.neon_recording import NeonRecording
from PySide6.QtCore import QPointF
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu
from pytestqt.qtbot import QtBot

from pupil_labs.neon_player.settings import RecordingSettings
from pupil_labs.neon_player.ui.timeline_dock import TimeLineDock
from pupil_labs.neon_player.ui.timeline_dock_components import (
    TimelineRowKey,
    TimelineRowRole,
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


@pytest.fixture
def timeline(
    qtbot: QtBot, qapp: MockNeonPlayerApp, monkeypatch: pytest.MonkeyPatch
) -> TimeLineDock:
    monkeypatch.setattr(qapp, "recording", None)
    monkeypatch.setattr(qapp, "playback_speed_options", [-1.0, 1.0], raising=False)
    dock = TimeLineDock()
    qtbot.addWidget(dock)
    return dock


@pytest.mark.parametrize("name", ["Ordinary", "Search", "HEADER", ""])
def test_global_row_identity_preserves_literal_names(
    timeline: TimeLineDock, name: str
) -> None:
    plot = timeline.get_timeline_plot(name, True)
    key = TimelineRowKey(None, name, TimelineRowRole.CUSTOM)
    assert isinstance(plot.row_key, TimelineRowKey)
    assert plot.row_key == key
    assert plot.row_key.group_name is None
    assert plot.row_key.row_name == name
    assert plot.row_key.role is TimelineRowRole.CUSTOM
    assert plot.row_name == name
    assert timeline.timeline_plots[key] is plot
    assert key in timeline.timeline_legends
    assert timeline.get_timeline_plot(name, True) is plot
    assert timeline.get_timeline_plot(name) is plot
    assert timeline.get_timeline_plot("Missing") is None


@pytest.mark.parametrize("kind", ["plot", "line", "scatter", "bar"])
def test_data_additions_use_custom_identity_and_keep_series_accessible(
    timeline: TimeLineDock,
    qapp: MockNeonPlayerApp,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    monkeypatch.setattr(
        qapp, "recording", SimpleNamespace(start_time=0, stop_time=1000)
    )
    if kind == "bar":
        plot = timeline.add_timeline_broken_bar("Search", [(100, 200)], "Series")
    else:
        add = {
            "plot": timeline.add_timeline_plot,
            "line": timeline.add_timeline_line,
            "scatter": timeline.add_timeline_scatter,
        }[kind]
        plot = add("Search", np.array([[100, 1], [200, 2]]), "Series")
    assert plot is not None
    key = TimelineRowKey(None, "Search", TimelineRowRole.CUSTOM)
    assert plot.row_key == key
    assert timeline.timeline_plots[key] is plot
    assert timeline.timeline_legends[key].items[0][1].text == "Series"
    if kind != "bar":
        series = timeline.get_timeline_series("Search", "Series")
        assert series is plot.items[0]
        timeline.remove_timeline_series("Search", "Series")
    else:
        timeline.remove_timeline_plot("Search")
    assert timeline.get_timeline_plot("Search") is None
    assert key not in timeline.timeline_legends


def test_series_removal_preserves_other_series_and_literal_row_names(
    timeline: TimeLineDock,
    qapp: MockNeonPlayerApp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qapp, "recording", SimpleNamespace(start_time=0, stop_time=1000)
    )
    data = np.array([[100, 1], [200, 2]])
    plot = timeline.add_timeline_scatter("", data, "First")
    assert timeline.add_timeline_scatter("", data, "Second") is plot
    timeline.add_timeline_scatter("Search", data, "First")
    second = timeline.get_timeline_series("", "Second")
    timeline.remove_timeline_series("", "Missing")
    timeline.remove_timeline_series("", "First")
    assert timeline.get_timeline_plot("") is plot
    assert timeline.get_timeline_series("", "Second") is second
    assert timeline.get_timeline_series("", "First") is None
    assert timeline.get_timeline_series("Search", "First") is not None


def test_point_actions_dispatch_by_complete_identity_and_preserve_payload(
    timeline: TimeLineDock, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[int, float]] = []
    other = Mock()
    point = (1_700_000_000_000_000_123, 3.5)
    key = TimelineRowKey(None, "", TimelineRowRole.CUSTOM)
    timeline.register_data_point_action("", "Seek", calls.append)
    timeline.register_data_point_action("Search", "Seek", other)
    assert key in timeline.data_point_actions

    actions: list[QAction] = []

    def add_action(name: str) -> QAction:
        action = QAction(name, timeline)
        actions.append(action)
        return action

    menu = Mock(spec=QMenu)
    menu.addAction.side_effect = add_action
    menu.exec.side_effect = lambda _: actions[-1].trigger()
    monkeypatch.setattr("pupil_labs.neon_player.ui.timeline_dock.QMenu", lambda: menu)
    timeline._show_data_point_actions(key, point)
    timeline.on_data_point_clicked("", point)
    timeline._show_data_point_actions(
        TimelineRowKey(None, "", TimelineRowRole.SEARCH), point
    )
    timeline._show_data_point_actions(
        TimelineRowKey(None, "", TimelineRowRole.HEADER), point
    )
    timeline._show_data_point_actions(
        TimelineRowKey("Other", "", TimelineRowRole.CUSTOM), point
    )
    assert calls == [point, point]
    other.assert_not_called()
    timeline.unregister_data_point_action("", "Seek")
    assert not timeline.data_point_actions[key]
    assert calls == [point, point]


def test_chart_point_click_forwards_the_plots_complete_identity(
    timeline: TimeLineDock, monkeypatch: pytest.MonkeyPatch
) -> None:
    plot = timeline.get_timeline_plot("", True)
    point = (123, 4.5)
    scene = timeline.graphics_layout.scene()
    assert scene is not None
    monkeypatch.setattr(scene, "itemsNearEvent", lambda _: [plot])
    monkeypatch.setattr(
        "pupil_labs.neon_player.ui.timeline_dock.get_clicked_data_point",
        lambda _plot, _event: point,
    )
    dispatch = Mock()
    monkeypatch.setattr(timeline, "_show_data_point_actions", dispatch)
    event = Mock()
    event.isAccepted.return_value = False
    timeline.check_for_data_item_click(event)
    dispatch.assert_called_once_with(plot.row_key, point)


def test_row_removal_clears_scoped_actions_before_name_reuse(
    timeline: TimeLineDock,
) -> None:
    callback = Mock()
    timeline.get_timeline_plot("", True)
    timeline.register_data_point_action("", "Old action", callback)
    timeline.remove_timeline_plot("")
    key = TimelineRowKey(None, "", TimelineRowRole.CUSTOM)
    assert key not in timeline.data_point_actions
    timeline.get_timeline_plot("", True)
    timeline.on_data_point_clicked("", (123, 4.5))
    callback.assert_not_called()


def test_sorting_preserves_equal_normalized_names_and_plot_type_order(
    timeline: TimeLineDock,
    qapp: MockNeonPlayerApp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qapp, "recording", SimpleNamespace(start_time=0, stop_time=1000)
    )
    timeline.get_timeline_plot("Ordinary", True)
    data = np.array([[100, 1], [200, 2]])
    timeline.add_timeline_line("Line", data)
    timeline.add_timeline_scatter("row", data)
    timeline.add_timeline_scatter("Row", data)
    timeline.add_timeline_scatter("", data)
    timeline.add_timeline_broken_bar("Bar", [(100, 200)])
    timeline.sort_plots()
    plots = [
        timeline.graphics_layout.getItem(row, 1)
        for row in sorted(timeline.graphics_layout.rows)
        if timeline.graphics_layout.getItem(row, 1) is not None
    ]
    assert plots[0] is timeline.timestamps_plot
    assert [plot.row_name for plot in plots[1:]] == [
        "Bar",
        "",
        "Ordinary",
        "row",
        "Row",
        "Line",
    ]
    assert {plot.row_key for plot in plots} == set(timeline.timeline_plots)
    assert len(plots) == len(timeline.timeline_plots)


def test_no_recording_keeps_existing_line_and_scatter_return_behavior(
    timeline: TimeLineDock,
) -> None:
    assert timeline.add_timeline_plot("Plot", []) is None
    assert timeline.add_timeline_line("Line", []) is None
    assert timeline.add_timeline_scatter("Scatter", []) is None
    assert list(timeline.timeline_plots) == [
        TimelineRowKey(None, "Export window", TimelineRowRole.CUSTOM)
    ]
