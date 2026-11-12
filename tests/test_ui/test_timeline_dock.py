from collections.abc import Callable

import numpy as np
import pyqtgraph as pg
import pytest
from pupil_labs.neon_recording import NeonRecording
from PySide6.QtCore import QEvent, QPoint, QPointF
from PySide6.QtGui import QBrush, QColor, QCursor, QPen
from PySide6.QtWidgets import QWidget
from pytestqt.qtbot import QtBot

from pupil_labs.neon_player.settings import RecordingSettings
from pupil_labs.neon_player.ui.timeline_dock import (
    TimeLineDock,
    _get_normalized_values,
    _resolve_bar_colors,
)
from pupil_labs.neon_player.ui.timeline_dock_components import (
    TimelinePinnedAxis,
    TimelinePlotItem,
    TrimEndMarker,
)
from pupil_labs.neon_player.ui.timeline_dock_tooltips import (
    TimelineTooltipContext,
    TimelineTooltipWidget,
)
from pupil_labs.neon_player.utilities import SlotDebouncer
from tests.conftest import MockNeonPlayerApp


@pytest.mark.parametrize(
    ("values", "limits", "expected"),
    [
        pytest.param([-4.0, -1.0, 8.0], None, [0.0, 0.25, 1.0], id="data-range"),
        pytest.param([8.0, -4.0, -1.0], None, [1.0, 0.0, 0.25], id="input-order"),
        pytest.param([3.0, 3.0], None, [0.5, 0.5], id="constant-values"),
        pytest.param([3.0], None, [0.5], id="single-value"),
        pytest.param([2.0, 4.0, 6.0], (0.0, 8.0), [0.25, 0.5, 0.75], id="limits"),
        pytest.param([-2.0, 6.0], (0.0, 4.0), [-0.5, 1.5], id="no-clipping"),
        pytest.param([1.0, 9.0], (4.0, 4.0), [0.5, 0.5], id="equal-limits"),
    ],
)
def test_get_normalized_values(
    values: list[float], limits: tuple[float, float] | None, expected: list[float]
) -> None:
    result = _get_normalized_values(np.array(values), limits)

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("values", [[], [np.nan, np.nan]], ids=["empty", "all-nan"])
@pytest.mark.parametrize("count", [0, 1, 3])
def test_resolve_bar_colors_uses_solid_color_without_values(
    values: list[float], count: int
) -> None:
    pens, brushes = _resolve_bar_colors(np.array(values), color_arg="red", count=count)

    assert len(pens) == len(brushes) == count
    assert all(isinstance(pen, QPen) and pen.color() == QColor("red") for pen in pens)
    assert all(
        isinstance(brush, QBrush) and brush.color() == QColor("red")
        for brush in brushes
    )


def test_resolve_bar_colors_defaults_to_one_white_bar_without_values() -> None:
    pens, brushes = _resolve_bar_colors(np.array([]))

    assert len(pens) == len(brushes) == 1
    assert pens[0].color() == brushes[0].color() == QColor("white")


@pytest.mark.parametrize(
    ("values", "limits", "positions"),
    [
        pytest.param([8.0, -4.0, 2.0], None, [1.0, 0.0, 0.5], id="data-range"),
        pytest.param([2.0, 4.0, 6.0], (0.0, 8.0), [0.25, 0.5, 0.75], id="limits"),
        pytest.param([3.0, 3.0], None, [0.5, 0.5], id="constant-values"),
        pytest.param([-2.0, 6.0], (0.0, 4.0), [0.0, 1.0], id="colormap-clipping"),
    ],
)
def test_resolve_bar_colors_maps_each_value_to_pen_and_brush(
    values: list[float], limits: tuple[float, float] | None, positions: list[float]
) -> None:
    pens, brushes = _resolve_bar_colors(
        np.array(values), limits, color_arg="plasma", count=7
    )
    expected = pg.colormap.get("plasma").map(np.array(positions), mode="qcolor")

    assert len(pens) == len(brushes) == len(values)
    assert all(isinstance(pen, QPen) for pen in pens)
    assert all(isinstance(brush, QBrush) for brush in brushes)
    assert [pen.color() for pen in pens] == expected
    assert [brush.color() for brush in brushes] == expected


@pytest.mark.parametrize("color_arg", ["white", "nonexistent-timeline-colormap"])
def test_resolve_bar_colors_falls_back_to_viridis(color_arg: str) -> None:
    pens, brushes = _resolve_bar_colors(
        np.array([10.0, 15.0, 20.0]), color_arg=color_arg
    )
    expected = pg.colormap.get("viridis").map(np.array([0.0, 0.5, 1.0]), mode="qcolor")

    assert [pen.color() for pen in pens] == expected
    assert [brush.color() for brush in brushes] == expected


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

    assert timeline._tooltip_content_for_hovered_row("Signal", 200) == "ok"
    assert len(contexts) == 1
    assert contexts[0].row_name == "Signal"
    assert contexts[0].recording_start_time_ns == 100
    assert contexts[0].series[0].timestamp_ns == 200
    assert contexts[0].series[0].value == 2.0
    assert timeline._tooltip_content_for_hovered_row("missing", 200) is None
    assert timeline._tooltip_content_for_hovered_row("Signal", 250) is None


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

    result = timeline._tooltip_content_for_hovered_row("Signal", 150)

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
        assert timeline._tooltip_content_for_hovered_row("Signal", 150) is None

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
