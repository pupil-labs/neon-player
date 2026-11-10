from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from pupil_labs.neon_recording import NeonRecording
from PySide6.QtCore import QCoreApplication, QEvent, QPointF, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu
from pytestqt.qtbot import QtBot
from shiboken6 import isValid

from pupil_labs.neon_player.settings import RecordingSettings
from pupil_labs.neon_player.ui.timeline_dock import TimeLineDock
from pupil_labs.neon_player.ui.timeline_dock_components import (
    TimelinePlotGroup,
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
        timeline.graphics_layout.getItem(row, 2)
        for row in sorted(timeline.graphics_layout.rows)
        if timeline.graphics_layout.getItem(row, 2) is not None
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


@pytest.mark.parametrize("use_name", [False, True])
@pytest.mark.parametrize("kind", ["plot", "line", "scatter", "bar"])
def test_grouped_contributions_use_names_or_handles(
    timeline: TimeLineDock,
    qapp: MockNeonPlayerApp,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    use_name: bool,
) -> None:
    group = timeline.add_timeline_group("Group")
    scope = group.name if use_name else group
    monkeypatch.setattr(
        qapp, "recording", SimpleNamespace(start_time=0, stop_time=1000)
    )
    if kind == "bar":
        plot = timeline.add_timeline_broken_bar("", [(100, 200)], "Series", group=scope)
    else:
        add = {
            "plot": timeline.add_timeline_plot,
            "line": timeline.add_timeline_line,
            "scatter": timeline.add_timeline_scatter,
        }[kind]
        plot = add("", np.array([[100, 1], [200, 2]]), "Series", group=scope)
    assert plot is not None
    assert plot.row_key == TimelineRowKey("Group", "", TimelineRowRole.CUSTOM)
    assert group.child_row_names == {""}
    assert timeline.get_timeline_plot("", group=group.name) is plot
    assert timeline.get_timeline_plot("", group=group) is plot
    assert timeline.get_timeline_plot("") is None
    assert timeline.get_timeline_plot("Header", group=scope) is None
    if kind != "bar":
        assert timeline.get_timeline_series("", "Series", group=scope) is plot.items[0]
        timeline.remove_timeline_series("", "Series", group=scope)
    else:
        timeline.remove_timeline_plot("", group=scope)
    assert timeline.get_timeline_plot("", group=scope) is None
    assert not group.child_row_names
    assert timeline.get_timeline_group("Group") is group


def test_group_and_main_row_share_a_global_name_but_children_can_repeat_it(
    timeline: TimeLineDock,
) -> None:
    first = timeline.add_timeline_group("First")
    second = timeline.add_timeline_group("Second")
    first_row = timeline.get_timeline_plot("First", True, group=first)
    second_row = timeline.get_timeline_plot("First", True, group=second.name)
    assert first_row is not second_row
    header = timeline.timeline_plots[
        TimelineRowKey(first.name, first.name, TimelineRowRole.HEADER)
    ]
    assert header.row_name == first.name
    assert header is not first_row
    assert first.child_row_names == {"First"}
    assert second.child_row_names == {"First"}
    assert timeline.get_timeline_group("Missing") is None
    with pytest.raises(ValueError, match="already exists"):
        timeline.add_timeline_group("First")
    with pytest.raises(ValueError, match="already exists"):
        timeline.get_timeline_plot("First", True)


def test_global_rows_and_groups_reserve_names_in_both_directions(
    timeline: TimeLineDock,
) -> None:
    global_row = timeline.get_timeline_plot("Taken", True)
    with pytest.raises(ValueError, match="already exists"):
        timeline.add_timeline_group("Taken")
    assert timeline.get_timeline_plot("Taken") is global_row
    assert timeline.get_timeline_group("Taken") is None
    with pytest.raises(ValueError, match="already exists"):
        timeline.add_timeline_group("Export window")
    group = timeline.add_timeline_group("Group")
    with pytest.raises(ValueError, match="already exists"):
        timeline.get_timeline_plot(group.name, True)
    timeline.remove_timeline_group(group)
    assert timeline.get_timeline_plot("Group", True) is not None
    timeline.remove_timeline_plot("Taken")
    assert timeline.add_timeline_group("Taken").name == "Taken"


@pytest.mark.parametrize("kind", ["plot", "line", "scatter", "bar"])
def test_global_contributions_cannot_reuse_group_names(
    timeline: TimeLineDock,
    kind: str,
) -> None:
    group = timeline.add_timeline_group("Group")
    add = {
        "plot": timeline.add_timeline_plot,
        "line": timeline.add_timeline_line,
        "scatter": timeline.add_timeline_scatter,
        "bar": timeline.add_timeline_broken_bar,
    }[kind]
    with pytest.raises(ValueError, match="already exists"):
        add("Group", [])
    assert add("Group", [], group=group) is None
    assert not group.child_row_names


@pytest.mark.parametrize("collapsed", [False, True])
def test_group_expander_controls_child_visibility_and_scroll_height(
    timeline: TimeLineDock, collapsed: bool
) -> None:
    group = timeline.add_timeline_group("Group", collapsed=collapsed)
    child = timeline.get_timeline_plot("Child", True, group=group)
    assert child is not None
    timeline.sort_plots()
    button = timeline.timeline_group_controls[group.name]
    legend = timeline.timeline_legends[child.row_key].parentItem()
    header = timeline.timeline_plots[
        TimelineRowKey(group.name, group.name, TimelineRowRole.HEADER)
    ]
    assert group.collapsed == collapsed
    assert child.isVisible() == (not collapsed)
    assert legend.isVisible() == (not collapsed)
    assert header.isVisible()
    assert button.text() == ("▸" if collapsed else "▾")
    assert button.toolTip() == ("Expand Group" if collapsed else "Collapse Group")
    assert button.accessibleName() == button.toolTip()
    assert button.cursor().shape() == Qt.CursorShape.PointingHandCursor
    assert timeline.graphics_view.height() == (75 if collapsed else 100)
    button.click()
    assert child.isVisible() == collapsed
    assert legend.isVisible() == collapsed
    assert header.isVisible()
    assert timeline.graphics_view.height() == (100 if collapsed else 75)
    button.click()
    assert group.collapsed == collapsed
    assert child.isVisible() == (not collapsed)


def test_late_children_and_empty_group_collapse(timeline: TimeLineDock) -> None:
    group = timeline.add_timeline_group("Group", collapsed=True)
    assert not group.child_row_names
    button = timeline.timeline_group_controls[group.name]
    button.click()
    button.click()
    timeline.disable_plot_sorting()
    first = timeline.get_timeline_plot("", True, group=group)
    second = timeline.get_timeline_plot("Search", True, group=group.name)
    assert first is not None
    assert second is not None
    assert not first.isVisible()
    assert not second.isVisible()
    assert timeline.graphics_view.height() == 75
    timeline.enable_plot_sorting()
    assert not first.isVisible()
    assert not second.isVisible()
    button.click()
    assert first.isVisible()
    assert second.isVisible()
    assert timeline.graphics_view.height() == 125


@pytest.mark.parametrize("kind", ["plot", "line", "scatter", "bar"])
def test_group_validation_precedes_no_recording_returns(
    timeline: TimeLineDock, kind: str
) -> None:
    group = timeline.add_timeline_group("Group")
    add = {
        "plot": timeline.add_timeline_plot,
        "line": timeline.add_timeline_line,
        "scatter": timeline.add_timeline_scatter,
        "bar": timeline.add_timeline_broken_bar,
    }[kind]
    assert add("Child", [], group=group.name) is None
    assert not group.child_row_names
    with pytest.raises(KeyError):
        add("Child", [], group="Missing")
    with pytest.raises(ValueError, match="does not belong"):
        add("Child", [], group=TimelinePlotGroup("Group"))


@pytest.mark.parametrize("use_name", [False, True])
def test_group_removal_cleans_all_state_and_invalidates_old_handles(
    timeline: TimeLineDock, use_name: bool
) -> None:
    group = timeline.add_timeline_group("Group", collapsed=True)
    child = timeline.get_timeline_plot("Child", True, group=group)
    global_row = timeline.get_timeline_plot("Child", True)
    callback = Mock()
    timeline.register_data_point_action("Child", "Action", callback, group=group.name)
    timeline.register_data_point_action("Not plotted", "Action", callback, group=group)
    button = timeline.timeline_group_controls[group.name]
    proxy = timeline.timeline_group_control_proxies[group.name]
    timeline.remove_timeline_group(group.name if use_name else group)
    assert timeline.get_timeline_group("Group") is None
    assert not group.child_row_names
    assert not any(key.group_name == group.name for key in timeline.timeline_plots)
    assert not any(key.group_name == group.name for key in timeline.timeline_legends)
    assert not any(key.group_name == group.name for key in timeline.data_point_actions)
    assert not any(
        key.group_name == group.name for key in timeline.hidden_timeline_rows
    )
    assert timeline.get_timeline_plot("Child") is global_row
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not isValid(button)
    assert not isValid(proxy)
    assert not isValid(child)
    fresh = timeline.add_timeline_group("Group")
    assert fresh is not group
    assert not fresh.collapsed
    assert timeline.get_timeline_plot("Child", group="Group") is None
    with pytest.raises(ValueError, match="does not belong"):
        timeline.get_timeline_plot("Child", group=group)
    with pytest.raises(ValueError, match="does not belong"):
        timeline.remove_timeline_group(group)
    with pytest.raises(KeyError):
        timeline.remove_timeline_group("Missing")


def test_scoped_point_actions_dispatch_only_the_requested_group(
    timeline: TimeLineDock, monkeypatch: pytest.MonkeyPatch
) -> None:
    group = timeline.add_timeline_group("Group")
    other = timeline.add_timeline_group("Other")
    global_callback, group_callback, other_callback = Mock(), Mock(), Mock()
    timeline.register_data_point_action("", "Seek", global_callback)
    timeline.register_data_point_action("", "Seek", group_callback, group=group)
    timeline.register_data_point_action("", "Seek", other_callback, group=other.name)
    actions: list[QAction] = []

    def add_action(name: str) -> QAction:
        action = QAction(name, timeline)
        actions.append(action)
        return action

    menu = Mock(spec=QMenu)
    menu.addAction.side_effect = add_action
    menu.exec.side_effect = lambda _: actions[-1].trigger()
    monkeypatch.setattr("pupil_labs.neon_player.ui.timeline_dock.QMenu", lambda: menu)
    point = (123, 4.5)
    timeline.on_data_point_clicked("", point, group=group.name)
    group_callback.assert_called_once_with(point)
    global_callback.assert_not_called()
    other_callback.assert_not_called()
    timeline.unregister_data_point_action("", "Seek", group=group.name)
    assert not timeline.data_point_actions[
        TimelineRowKey(group.name, "", TimelineRowRole.CUSTOM)
    ]
    timeline.on_data_point_clicked("", point, group=other)
    other_callback.assert_called_once_with(point)


def test_sorting_keeps_groups_contiguous_and_sorts_only_custom_rows_by_type(
    timeline: TimeLineDock, qapp: MockNeonPlayerApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        qapp, "recording", SimpleNamespace(start_time=0, stop_time=1000)
    )
    beta = timeline.add_timeline_group("Beta")
    alpha = timeline.add_timeline_group("Alpha")
    data = np.array([[100, 1], [200, 2]])
    timeline.add_timeline_line("Line", data)
    timeline.add_timeline_scatter("Aardvark", data)
    timeline.add_timeline_broken_bar("Bar", [(100, 200)])
    timeline.add_timeline_scatter("Point", data, group=beta)
    timeline.add_timeline_line("A line", data, group=alpha)
    timeline.add_timeline_broken_bar("Z bar", [(100, 200)], group=alpha.name)
    timeline.add_timeline_scatter("Z point", data, group=alpha)
    header_key = TimelineRowKey(alpha.name, "Alpha", TimelineRowRole.HEADER)
    timeline.timeline_plots[header_key].plot(data)
    timeline.sort_plots()
    keys = [
        timeline.graphics_layout.getItem(row, 2).row_key
        for row in sorted(timeline.graphics_layout.rows)
        if timeline.graphics_layout.getItem(row, 2) is not None
    ]
    assert [(key.group_name, key.row_name) for key in keys] == [
        (None, "Export window"),
        (None, "Bar"),
        (None, "Aardvark"),
        (alpha.name, "Alpha"),
        (alpha.name, "Z bar"),
        (alpha.name, "Z point"),
        (alpha.name, "A line"),
        (beta.name, "Beta"),
        (beta.name, "Point"),
        (None, "Line"),
    ]
    for group in (alpha, beta):
        header_row = (
            keys.index(TimelineRowKey(group.name, group.name, TimelineRowRole.HEADER))
            + 1
        )
        assert (
            timeline.graphics_layout.getItem(header_row, 0)
            is timeline.timeline_group_control_proxies[group.name]
        )


def test_grouped_rows_follow_the_shared_time_axis_and_clean_child_membership(
    timeline: TimeLineDock,
    qtbot: QtBot,
) -> None:
    group = timeline.add_timeline_group("Group")
    child = timeline.get_timeline_plot("Child", True, group=group)
    assert child is not None
    timeline.sort_plots()
    timeline.resize(1000, 500)
    timeline.show()
    QCoreApplication.processEvents()
    timeline.timestamps_plot.setXRange(100, 500, padding=0)
    qtbot.waitUntil(lambda: child.viewRange()[0] == pytest.approx([100, 500]))
    timeline.remove_timeline_plot("Child", group=group.name)
    assert not group.child_row_names
    assert timeline.get_timeline_group(group.name) is group
