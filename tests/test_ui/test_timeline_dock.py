from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from pupil_labs.neon_recording import NeonRecording
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QAction, QCursor
from PySide6.QtWidgets import (
    QGraphicsProxyWidget,
    QLineEdit,
    QMenu,
    QToolButton,
    QWidget,
)
from pytestqt.qtbot import QtBot
from shiboken6 import isValid

from pupil_labs.neon_player.settings import RecordingSettings
from pupil_labs.neon_player.ui.timeline_dock import TimeLineDock
from pupil_labs.neon_player.ui.timeline_dock_components import (
    TimelinePinnedAxis,
    TimelinePlotGroup,
    TimelinePlotItem,
    TimelineRowKey,
    TimelineRowRole,
    TrimEndMarker,
)
from pupil_labs.neon_player.ui.timeline_dock_tooltips import (
    TimelineTooltipContext,
    TimelineTooltipWidget,
)
from pupil_labs.neon_player.utilities import SlotDebouncer
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

    timeline.on_chart_area_mouse_moved(row_center(first_row))
    timeline.eventFilter(timeline.graphics_view.viewport(), QEvent(QEvent.Type.Leave))
    assert not any(label.isVisible() for label in first_row._y_range_labels)
    assert not any(label.isVisible() for label in second_row._y_range_labels)


def test_vertical_scroll_refreshes_hover_at_stationary_pointer(
    qtbot: QtBot,
    qapp: MockNeonPlayerApp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qapp, "recording", None)
    monkeypatch.setattr(qapp, "playback_speed_options", [-1.0, 1.0], raising=False)
    timeline = TimeLineDock()
    qtbot.addWidget(timeline)
    timeline.resize(600, 180)
    timeline.show()
    for row in range(10):
        timeline.get_timeline_plot(f"row {row}", create_if_missing=True)
    timeline.fix_scroll_size()
    qtbot.wait(50)

    debounce_calls = []
    assert timeline._on_scroll_or_playback in SlotDebouncer._connections
    debouncer = SlotDebouncer._connections[timeline._on_scroll_or_playback]
    monkeypatch.setattr(debouncer, "slot", lambda *_args: debounce_calls.append(1))

    viewport = timeline.graphics_view.viewport()
    QCursor.setPos(viewport.mapToGlobal(QPoint(20, 20)))

    scrollbar = timeline.scroll_area.verticalScrollBar()
    assert scrollbar.maximum() > 0

    with qtbot.waitSignal(scrollbar.valueChanged):
        scrollbar.setValue(scrollbar.maximum())

    qtbot.waitUntil(lambda: bool(debounce_calls))


def test_x_axis_range_change_refreshes_hover_under_stationary_cursor(
    qtbot: QtBot, qapp: MockNeonPlayerApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(qapp, "recording", None)
    monkeypatch.setattr(qapp, "playback_speed_options", [-1.0, 1.0], raising=False)
    timeline = TimeLineDock()
    qtbot.addWidget(timeline)
    timeline.resize(600, 300)
    timeline.show()
    qtbot.waitExposed(timeline)

    viewport = timeline.graphics_view.viewport()
    cursor_pos = viewport.mapToGlobal(viewport.rect().center())
    QCursor.setPos(cursor_pos)
    refreshed_positions: list[QPointF] = []
    monkeypatch.setattr(
        timeline,
        "_on_scroll_or_playback",
        refreshed_positions.append,
    )

    timeline.timestamps_plot.setXRange(10, 20, padding=0)

    assert refreshed_positions


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
    assert timeline.graphics_view.height() == (55 if collapsed else 80)
    button.click()
    assert child.isVisible() == collapsed
    assert legend.isVisible() == collapsed
    assert header.isVisible()
    assert timeline.graphics_view.height() == (80 if collapsed else 55)
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
    assert timeline.graphics_view.height() == 55
    timeline.enable_plot_sorting()
    assert not first.isVisible()
    assert not second.isVisible()
    button.click()
    assert first.isVisible()
    assert second.isVisible()
    assert timeline.graphics_view.height() == 105


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


def test_searchable_declaration_filters_only_custom_children(
    timeline: TimeLineDock, qtbot: QtBot
) -> None:
    group = timeline.add_timeline_group("Group", searchable=True)
    plain = timeline.add_timeline_group("Plain")
    assert group.searchable
    assert not plain.searchable
    names = ["Search", "", "Group", "Trial Begin", "trial end"]
    plots = {
        name: timeline.get_timeline_plot(name, True, group=group) for name in names
    }
    global_plot = timeline.get_timeline_plot("Global", True)
    other = timeline.get_timeline_plot("Other", True, group=plain)
    timeline.sort_plots()
    fields = search_fields(timeline)
    assert len(fields) == 1
    field = fields[0]
    assert field.placeholderText() == "Search"
    assert field.isClearButtonEnabled()
    assert "Group" in field.accessibleName()
    qtbot.keyClicks(field, "trial")
    assert [name for name, plot in plots.items() if plot.isVisible()] == [
        "Trial Begin",
        "trial end",
    ]
    assert global_plot.isVisible()
    assert other.isVisible()
    search_key = TimelineRowKey("Group", "Search", TimelineRowRole.SEARCH)
    search = timeline.timeline_plots[search_key]
    assert search.items == []
    assert search.isVisible()
    assert group.child_row_names == set(names)
    timeline.timeline_group_controls[group.name].click()
    assert not search.isVisible()
    field.clear()
    assert not any(plot.isVisible() for plot in plots.values())
    timeline.timeline_group_controls[group.name].click()
    assert all(plot.isVisible() for plot in plots.values())
    field.setText("g")
    assert [name for name, plot in plots.items() if plot.isVisible()] == ["Group"]


def search_fields(timeline: TimeLineDock) -> list[QLineEdit]:
    return [
        field
        for item in timeline.graphics_view.scene().items()
        if isinstance(item, QGraphicsProxyWidget) and item.widget() is not None
        for field in item.widget().findChildren(QLineEdit)
    ]


def test_search_updates_new_rows_and_preserves_data_on_refresh(
    timeline: TimeLineDock,
    qtbot: QtBot,
    qapp: MockNeonPlayerApp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qapp, "recording", SimpleNamespace(start_time=0, stop_time=1000)
    )
    group = timeline.add_timeline_group("Trials", searchable=True)
    timeline.resize(1000, 500)
    timeline.show()
    timeline.activateWindow()
    QCoreApplication.processEvents()
    field = search_fields(timeline)[0]
    proxy = field.parentWidget().graphicsProxyWidget()
    qtbot.mouseClick(
        timeline.graphics_view.viewport(),
        Qt.MouseButton.LeftButton,
        pos=timeline.graphics_view.mapFromScene(
            proxy.mapToScene(QPointF(field.geometry().center()))
        ),
    )
    qtbot.keyClicks(field, "begin")
    plot = timeline.add_timeline_scatter(
        "Begin", np.array([[100, 1]]), "First", group=group
    )
    assert plot is not None
    assert plot.isVisible()
    series = timeline.get_timeline_series("Begin", "First", group=group)
    timeline.add_timeline_scatter("End", np.array([[200, 2]]), group=group)
    timeline.add_timeline_scatter("Begin", np.array([[300, 3]]), "Second", group=group)
    assert field.text() == "begin"
    assert search_fields(timeline) == [field]
    assert timeline.get_timeline_series("Begin", "First", group=group) is series
    assert list(series.getOriginalDataset()[0]) == [100]
    assert not timeline.get_timeline_plot("End", group=group).isVisible()
    field.setText("nothing matches")
    assert not plot.isVisible()
    search_key = TimelineRowKey("Trials", "Search", TimelineRowRole.SEARCH)
    assert timeline.timeline_plots[search_key].isVisible()
    timeline.remove_timeline_plot("Begin", group=group)
    assert not timeline.get_timeline_plot("", True, group=group).isVisible()
    field.clear()
    assert timeline.get_timeline_plot("", group=group).isVisible()


def test_search_controls_cannot_retarget_recreated_groups(
    timeline: TimeLineDock,
) -> None:
    old = timeline.add_timeline_group("Group", searchable=True)
    old_field = search_fields(timeline)[0]
    timeline.remove_timeline_group(old)
    fresh = timeline.add_timeline_group("Group", searchable=True)
    plot = timeline.get_timeline_plot("Child", True, group=fresh)
    old_field.setText("no match")
    assert plot.isVisible()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not isValid(old_field)
    assert len(search_fields(timeline)) == 1
    assert search_fields(timeline)[0].text() == ""


@pytest.mark.parametrize("target", ["bar", "field", "proxy"])
def test_unexpected_search_destruction_restores_fresh_control(
    timeline: TimeLineDock, qtbot: QtBot, target: str
) -> None:
    group = timeline.add_timeline_group("Group", searchable=True)
    plot = timeline.get_timeline_plot("Child", True, group=group)
    field = search_fields(timeline)[0]
    field.setText("missing")
    bar = field.parentWidget()
    if target == "field":
        field.deleteLater()
    elif target == "proxy":
        bar.graphicsProxyWidget().deleteLater()
    else:
        bar.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qtbot.waitUntil(lambda: len(search_fields(timeline)) == 1)
    fresh = search_fields(timeline)[0]
    assert fresh is not field
    assert fresh.text() == ""
    assert plot.isVisible()


def test_pending_search_restoration_cannot_touch_replacement_group(
    timeline: TimeLineDock,
) -> None:
    group = timeline.add_timeline_group("Group", searchable=True)
    field = search_fields(timeline)[0]
    field.parentWidget().deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    timeline.remove_timeline_group(group)
    fresh = timeline.add_timeline_group("Group", searchable=True)
    replacement = search_fields(timeline)[0]
    replacement.setText("Child")
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QCoreApplication.processEvents()
    assert timeline.get_timeline_group("Group") is fresh
    assert search_fields(timeline) == [replacement]
    assert replacement.text() == "Child"


def test_search_editing_blocks_shortcuts(timeline: TimeLineDock, qtbot: QtBot) -> None:
    timeline.add_timeline_group("Group", searchable=True)
    timeline.resize(1000, 500)
    timeline.show()
    timeline.activateWindow()
    QCoreApplication.processEvents()
    shortcuts = Mock()
    for shortcut in ("Space", "Right", "Left", "Shift+Right", "Shift+Left"):
        action = QAction(timeline)
        action.setShortcut(shortcut)
        action.triggered.connect(shortcuts)
        timeline.addAction(action)
    forwarded = Mock()
    timeline.key_pressed.connect(forwarded)
    field = search_fields(timeline)[0]
    qtbot.mouseClick(field, Qt.MouseButton.LeftButton)
    qtbot.keyClicks(field, "trial")
    qtbot.keyClick(field, Qt.Key.Key_Space)
    assert field.text() == "trial "
    qtbot.keyClick(field, Qt.Key.Key_Left)
    qtbot.keyClick(field, Qt.Key.Key_Left, Qt.KeyboardModifier.ShiftModifier)
    assert field.selectedText() == "l"
    qtbot.keyClick(field, Qt.Key.Key_Right)
    qtbot.keyClick(field, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
    assert field.selectedText() == " "
    shortcuts.assert_not_called()
    forwarded.assert_not_called()


def test_unload_disposes_search_and_reload_restores_surviving_declaration(
    timeline: TimeLineDock,
    qapp: MockNeonPlayerApp,
    monkeypatch: pytest.MonkeyPatch,
    mock_neon_recording: Callable[..., NeonRecording],
) -> None:
    group = timeline.add_timeline_group("Group", searchable=True)
    plot = timeline.get_timeline_plot("Child", True, group=group)
    field = search_fields(timeline)[0]
    field.setText("missing")
    timeline.timeline_group_controls[group.name].click()
    timeline.on_recording_unloaded()
    # Even before Qt deletes the detached field, its signals are inert.
    field.setText("Child")
    recording = mock_neon_recording(start_time=100, stop_time=200)
    monkeypatch.setattr(qapp, "recording", recording)
    monkeypatch.setattr(qapp, "recording_settings", RecordingSettings())
    timeline.on_recording_loaded(recording)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not isValid(field)
    fresh = search_fields(timeline)[0]
    assert fresh.text() == ""
    assert group.collapsed
    assert not plot.isVisible()
    timeline.timeline_group_controls[group.name].click()
    assert plot.isVisible()
    assert timeline.get_timeline_group("Group") is group


def test_search_rows_sort_before_children_and_keep_independent_queries(
    timeline: TimeLineDock,
) -> None:
    beta = timeline.add_timeline_group("Beta", searchable=True)
    alpha = timeline.add_timeline_group("Alpha", searchable=True)
    beta_plot = timeline.get_timeline_plot("Child", True, group=beta)
    alpha_plot = timeline.get_timeline_plot("Child", True, group=alpha)
    fields = {field.accessibleName(): field for field in search_fields(timeline)}
    fields["Search Alpha"].setText("missing")
    timeline.sort_plots()
    keys = [
        timeline.graphics_layout.getItem(row, 2).row_key
        for row in sorted(timeline.graphics_layout.rows)
        if timeline.graphics_layout.getItem(row, 2) is not None
    ]
    assert [(key.group_name, key.role) for key in keys[1:]] == [
        ("Alpha", TimelineRowRole.HEADER),
        ("Alpha", TimelineRowRole.SEARCH),
        ("Alpha", TimelineRowRole.CUSTOM),
        ("Beta", TimelineRowRole.HEADER),
        ("Beta", TimelineRowRole.SEARCH),
        ("Beta", TimelineRowRole.CUSTOM),
    ]
    assert not alpha_plot.isVisible()
    assert beta_plot.isVisible()
    fields["Search Beta"].setText("Child")
    assert fields["Search Alpha"].text() == "missing"


def test_search_height_tracks_font_and_clear_button_restores_rows(
    timeline: TimeLineDock, qtbot: QtBot
) -> None:
    group = timeline.add_timeline_group("Group", searchable=True)
    plot = timeline.get_timeline_plot("Child", True, group=group)
    timeline.sort_plots()
    timeline.resize(1000, 500)
    timeline.show()
    timeline.activateWindow()
    QCoreApplication.processEvents()
    field = search_fields(timeline)[0]
    search_key = TimelineRowKey("Group", "Search", TimelineRowRole.SEARCH)
    search_plot = timeline.timeline_plots[search_key]
    font = field.font()
    font.setPointSize(30)
    field.setFont(font)
    qtbot.waitUntil(lambda: search_plot.height() >= field.sizeHint().height())
    host = timeline.timeline_legends[search_key].parentItem()
    assert host.height() == search_plot.height()
    assert timeline.graphics_view.height() >= 80 + search_plot.height()
    qtbot.keyClicks(field, "missing")
    assert not plot.isVisible()
    QCoreApplication.processEvents()
    clear = field.findChild(QToolButton)
    assert clear is not None
    qtbot.mouseClick(clear, Qt.MouseButton.LeftButton)
    assert field.text() == ""
    assert plot.isVisible()


def test_empty_search_plot_keeps_shared_axis_and_mouse_seek(
    timeline: TimeLineDock,
    qtbot: QtBot,
    qapp: MockNeonPlayerApp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qapp, "recording", SimpleNamespace(start_time=0, stop_time=1000)
    )
    seek = Mock()
    monkeypatch.setattr(qapp, "seek_to", seek)
    timeline.add_timeline_group("Group", searchable=True)
    search = timeline.timeline_plots[
        TimelineRowKey("Group", "Search", TimelineRowRole.SEARCH)
    ]
    timeline.resize(1000, 500)
    timeline.show()
    timeline.activateWindow()
    QCoreApplication.processEvents()
    timeline.timestamps_plot.setXRange(100, 500, padding=0)
    qtbot.waitUntil(lambda: search.viewRange()[0] == pytest.approx([100, 500]))
    scene_position = search.getViewBox().mapViewToScene(QPointF(300, 0))
    qtbot.mouseClick(
        timeline.graphics_view.viewport(),
        Qt.MouseButton.LeftButton,
        pos=timeline.graphics_view.mapFromScene(scene_position),
    )
    assert search.items == []
    assert seek.call_count >= 1
    assert seek.call_args.args[0] == pytest.approx(300, abs=2)


def test_removing_a_series_shrinks_row_and_scroll_height(
    timeline: TimeLineDock,
    qapp: MockNeonPlayerApp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qapp, "recording", SimpleNamespace(start_time=0, stop_time=1000)
    )
    group = timeline.add_timeline_group("Group", searchable=True)
    field = search_fields(timeline)[0]
    data = np.array([[100, 1], [200, 2]])
    plot = timeline.add_timeline_scatter("Child", data, "First", group=group)
    timeline.add_timeline_scatter("Child", data, "Second", group=group)
    old_height = plot.preferredHeight()
    old_scroll_height = timeline.graphics_view.height()
    timeline.remove_timeline_series("Child", "Second", group=group)
    assert plot.preferredHeight() < old_height
    assert timeline.graphics_view.height() < old_scroll_height
    assert search_fields(timeline) == [field]
    assert timeline.get_timeline_series("Child", "First", group=group) is not None


def test_expand_collapse_all_groups_preserves_queries_and_global_rows(
    timeline: TimeLineDock,
) -> None:
    first = timeline.add_timeline_group("First", searchable=True)
    second = timeline.add_timeline_group("Second", collapsed=True)
    empty = timeline.add_timeline_group("Empty", collapsed=True)
    field = search_fields(timeline)[0]
    match = timeline.get_timeline_plot("Match", True, group=first)
    hidden = timeline.get_timeline_plot("Other", True, group=first)
    child = timeline.get_timeline_plot("Child", True, group=second)
    global_plot = timeline.get_timeline_plot("Global", True)
    field.setText("match")
    timeline.expand_all_groups()
    assert not any(group.collapsed for group in (first, second, empty))
    assert match.isVisible()
    assert not hidden.isVisible()
    assert child.isVisible()
    expanded_height = timeline.graphics_view.height()
    timeline.collapse_all_groups()
    assert all(group.collapsed for group in (first, second, empty))
    assert not match.isVisible()
    assert not child.isVisible()
    assert global_plot.isVisible()
    assert timeline.graphics_view.height() < expanded_height
    timeline.collapse_all_groups()
    assert all(group.collapsed for group in (first, second, empty))
    timeline.expand_all_groups()
    timeline.expand_all_groups()
    assert not any(group.collapsed for group in (first, second, empty))
    assert field.text() == "match"
    assert match.isVisible()
    assert not hidden.isVisible()
    assert child.isVisible()
    for group in (first, second, empty):
        control = timeline.timeline_group_controls[group.name]
        assert control.toolTip().startswith("Collapse")
        timeline.remove_timeline_group(group)
    timeline.expand_all_groups()
    timeline.collapse_all_groups()
    assert global_plot.isVisible()


def test_pinned_time_axis_tracks_recording_and_shared_view_range(
    qtbot: QtBot,
    qapp: MockNeonPlayerApp,
    monkeypatch: pytest.MonkeyPatch,
    mock_neon_recording: Callable[..., NeonRecording],
) -> None:
    monkeypatch.setattr(qapp, "recording", None)
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


class MutableTimelineTooltip(TimelineTooltipWidget):
    def __init__(self) -> None:
        super().__init__()
        self.context: TimelineTooltipContext | None = None

    def update_tooltip(self, context: TimelineTooltipContext) -> None:
        self.context = context


@pytest.mark.parametrize("plot_type", ["line", "scatter", "broken_bar"])
def test_grouped_tooltips_keep_row_identity_and_release_widgets(
    qtbot: QtBot,
    qapp: MockNeonPlayerApp,
    monkeypatch: pytest.MonkeyPatch,
    mock_neon_recording: Callable[..., NeonRecording],
    plot_type: str,
) -> None:
    monkeypatch.setattr(qapp, "recording", None)
    timeline = TimeLineDock()
    qtbot.addWidget(timeline)
    recording = mock_neon_recording(start_time=100, stop_time=300)
    monkeypatch.setattr(qapp, "recording", recording)
    timeline.recording = recording
    group = timeline.add_timeline_group("Group")
    other_group = timeline.add_timeline_group("Other group")
    add_plot = getattr(timeline, f"add_timeline_{plot_type}")
    # A grouped row may share the export row's name without losing its tooltip.
    row_name = "Export window"
    data = np.array([[150, 200]])
    global_row = add_plot(row_name, data, tooltip=lambda _: "global")
    grouped_tooltip = MutableTimelineTooltip()
    grouped_row = add_plot(row_name, data, tooltip=grouped_tooltip, group=group)
    other_row = add_plot(row_name, data, tooltip=lambda _: "other", group=other_group)
    assert global_row is not None
    assert grouped_row is not None
    assert other_row is not None
    assert timeline._tooltip_content_for_hovered_row(global_row.row_key, 150) is None
    assert (
        timeline._tooltip_content_for_hovered_row(grouped_row.row_key, 150)
        is grouped_tooltip
    )
    assert grouped_tooltip.context.row_name == row_name
    assert timeline._tooltip_content_for_hovered_row(other_row.row_key, 150) == "other"

    timeline.set_timeline_tooltip(row_name, None, group=other_group)
    assert timeline._tooltip_content_for_hovered_row(other_row.row_key, 150) is None
    timeline.remove_timeline_group(group)
    assert grouped_row.row_key not in timeline.timeline_tooltips
    assert id(grouped_tooltip) not in timeline._tooltip_widget_rows
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not isValid(grouped_tooltip)


def test_adding_line_series_preserves_explicit_y_limit_preference(
    qtbot: QtBot,
    qapp: MockNeonPlayerApp,
    monkeypatch: pytest.MonkeyPatch,
    mock_neon_recording: Callable[..., NeonRecording],
) -> None:
    monkeypatch.setattr(qapp, "recording", None)
    timeline = TimeLineDock()
    qtbot.addWidget(timeline)
    monkeypatch.setattr(qapp, "recording", mock_neon_recording())
    group = timeline.add_timeline_group("Group")
    for scope in (None, group):
        row = timeline.add_timeline_line(
            "Signal", np.array([[150, 1]]), show_y_limit_labels=False, group=scope
        )
        assert row is not None
        timeline.add_timeline_line("Signal", np.array([[200, 2]]), group=scope)
        assert not row.show_y_limit_labels


def test_timeline_tooltip_callback_receives_hovered_series_and_row_context(
    qtbot: QtBot,
    qapp: MockNeonPlayerApp,
    monkeypatch: pytest.MonkeyPatch,
    mock_neon_recording: Callable[..., NeonRecording],
) -> None:
    monkeypatch.setattr(qapp, "recording", None)
    monkeypatch.setattr(qapp, "playback_speed_options", [-1.0, 1.0], raising=False)
    timeline = TimeLineDock()
    qtbot.addWidget(timeline)
    timeline.recording = mock_neon_recording(start_time=100, stop_time=300)
    row = timeline.get_timeline_plot("Signal", create_if_missing=True)
    assert row is not None
    row.plot([150, 200], [1.0, 2.0], name="samples")
    contexts: list[TimelineTooltipContext] = []
    timeline.set_timeline_tooltip(
        "Signal", lambda context: contexts.append(context) or "ok"
    )

    assert timeline._tooltip_content_for_hovered_row(row.row_key, 200) == "ok"
    assert len(contexts) == 1
    assert contexts[0].row_name == "Signal"
    assert contexts[0].recording_start_time_ns == 100
    assert contexts[0].series[0].timestamp_ns == 200
    assert contexts[0].series[0].value == 2.0
    assert timeline._tooltip_content_for_hovered_row(None, 200) is None
    assert timeline._tooltip_content_for_hovered_row(row.row_key, 250) is None


def test_timeline_tooltip_custom_widget_is_updated_and_owned_by_one_row(
    qtbot: QtBot,
    qapp: MockNeonPlayerApp,
    monkeypatch: pytest.MonkeyPatch,
    mock_neon_recording: Callable[..., NeonRecording],
) -> None:
    monkeypatch.setattr(qapp, "recording", None)
    monkeypatch.setattr(qapp, "playback_speed_options", [-1.0, 1.0], raising=False)
    timeline = TimeLineDock()
    qtbot.addWidget(timeline)
    timeline.recording = mock_neon_recording(start_time=100, stop_time=300)
    row = timeline.get_timeline_plot("Signal", create_if_missing=True)
    assert row is not None
    row.plot([150], [1.0], name="sample")
    tooltip = MutableTimelineTooltip()
    timeline.set_timeline_tooltip("Signal", tooltip)

    result = timeline._tooltip_content_for_hovered_row(row.row_key, 150)

    assert result is tooltip
    assert tooltip.parentWidget() is timeline.tooltip_widget
    assert tooltip.context is not None
    assert tooltip.context.series[0].timestamp_ns == 150
    with pytest.raises(ValueError, match="must be parentless"):
        timeline.set_timeline_tooltip("Another row", tooltip)


def test_timeline_tooltip_widget_must_be_parentless(
    qtbot: QtBot, qapp: MockNeonPlayerApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(qapp, "recording", None)
    monkeypatch.setattr(qapp, "playback_speed_options", [-1.0, 1.0], raising=False)
    timeline = TimeLineDock()
    qtbot.addWidget(timeline)
    parent = QWidget()
    tooltip = MutableTimelineTooltip()
    tooltip.setParent(parent)

    with pytest.raises(ValueError, match="must be parentless"):
        timeline.set_timeline_tooltip("Signal", tooltip)

    parent.close()


def test_timeline_tooltip_callback_error_is_logged_and_ignored(
    qtbot: QtBot,
    qapp: MockNeonPlayerApp,
    monkeypatch: pytest.MonkeyPatch,
    mock_neon_recording: Callable[..., NeonRecording],
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(qapp, "recording", None)
    monkeypatch.setattr(qapp, "playback_speed_options", [-1.0, 1.0], raising=False)
    timeline = TimeLineDock()
    qtbot.addWidget(timeline)
    timeline.recording = mock_neon_recording(start_time=100, stop_time=300)
    row = timeline.get_timeline_plot("Signal", create_if_missing=True)
    assert row is not None
    row.plot([150], [1.0], name="sample")

    def fail(_context: TimelineTooltipContext) -> str:
        raise RuntimeError("tooltip failed")

    timeline.set_timeline_tooltip("Signal", fail)

    with caplog.at_level("ERROR"):
        assert timeline._tooltip_content_for_hovered_row(row.row_key, 150) is None

    assert "Timeline tooltip callback failed for Signal" in caplog.text


def test_hover_timestamp_is_displayed_on_pinned_axis(
    qtbot: QtBot,
    qapp: MockNeonPlayerApp,
    monkeypatch: pytest.MonkeyPatch,
    mock_neon_recording: Callable[..., NeonRecording],
) -> None:
    monkeypatch.setattr(qapp, "recording", None)
    monkeypatch.setattr(qapp, "playback_speed_options", [-1.0, 1.0], raising=False)
    timeline = TimeLineDock()
    qtbot.addWidget(timeline)
    timeline.resize(700, 300)
    timeline.show()
    qtbot.waitExposed(timeline)
    qtbot.wait(50)

    recording = mock_neon_recording(start_time=100, stop_time=200)
    monkeypatch.setattr(qapp, "recording", recording)
    monkeypatch.setattr(qapp, "recording_settings", RecordingSettings())
    timeline.on_recording_loaded(recording)
    timeline.timestamps_plot.setXRange(100, 200, padding=0)
    timeline.hover_marker.set_cursor_time(150)

    assert timeline.hover_marker.isVisible()
    assert timeline.hover_marker.label.isVisible()
    assert timeline.hover_marker.label.parentWidget() is timeline.pinned_axis

    data_rect = timeline.pinned_axis.data_rect()
    expected_x = round(data_rect.width() / 2)
    timeline_x = round(timeline.hover_marker.get_x_pixel_for_x_value(150))
    assert abs(expected_x - timeline_x) <= 2

    timeline.hover_marker.set_cursor_time(None)
    assert not timeline.hover_marker.isVisible()
    assert not timeline.hover_marker.label.isVisible()

    timeline.hover_marker.set_cursor_time(201)
    assert timeline.hover_marker.cursor_time is None
    assert not timeline.hover_marker.label.isVisible()
