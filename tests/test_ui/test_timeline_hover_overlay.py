from collections.abc import Generator

import pyqtgraph as pg
import pytest
from PySide6.QtCore import QEvent

from pupil_labs.neon_player.ui.timeline_dock import TimeLineDock
from pupil_labs.neon_player.ui.timeline_dock_components import (
    TimeAxisItem,
    TimelineHoverMarker,
    TimelinePinnedAxis,
    TimestampLabel,
    format_timestamp_ns,
)
from tests.conftest import MockNeonPlayerApp

type TimelineHoverState = tuple[
    TimeLineDock,
    pg.PlotItem,
    pg.PlotItem,
    TimelinePinnedAxis,
    TimelineHoverMarker,
]


@pytest.fixture
def timeline_hover_marker(
    qapp: MockNeonPlayerApp,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[TimelineHoverState, None, None]:
    monkeypatch.setattr(qapp, "playback_speed_options", [-1.0, 1.0], raising=False)
    timeline = TimeLineDock()
    timeline.resize(800, 400)
    timeline.show()
    qapp.processEvents()

    first_plot = timeline.timestamps_plot
    second_plot = timeline.get_timeline_plot("second", create_if_missing=True)
    assert second_plot is not None
    timeline.pinned_axis.set_time_frame(0, 100)
    first_plot.setXRange(0, 100, padding=0)
    timeline._sync_pinned_axis_geometry()
    marker = timeline.hover_marker
    pinned_axis = timeline.pinned_axis
    qapp.processEvents()

    yield timeline, first_plot, second_plot, pinned_axis, marker

    timeline.close()


def test_timeline_hover_marker_tracks_time_range_and_spans_rows(
    qapp: MockNeonPlayerApp, timeline_hover_marker: TimelineHoverState
) -> None:
    timeline, plot, _, _, marker = timeline_hover_marker

    marker.set_cursor_time(50)
    qapp.processEvents()

    assert marker.isVisible()
    assert marker.height() == timeline.graphics_view.height()
    assert marker.y() == 0
    first_x = marker.get_x_pixel_for_x_value(50)
    assert first_x == pytest.approx(marker.width() / 2 - 1, abs=2)
    first_image = marker.grab().toImage()
    assert all(
        first_image.pixelColor(round(first_x), y).alpha() > 0
        for y in range(first_image.height())
    )

    plot.setXRange(0, 200, padding=0)
    qapp.processEvents()

    second_x = marker.get_x_pixel_for_x_value(50)
    assert second_x < first_x
    second_image = marker.grab().toImage()
    assert any(
        second_image.pixelColor(round(second_x), y).alpha() > 0
        for y in range(second_image.height())
    )


def test_timeline_hover_marker_hides_when_pointer_leaves_viewport(
    qapp: MockNeonPlayerApp, timeline_hover_marker: TimelineHoverState
) -> None:
    timeline, _, _, _, marker = timeline_hover_marker
    marker.set_cursor_time(25)
    qapp.processEvents()
    assert marker.isVisible()

    qapp.sendEvent(timeline.graphics_view.viewport(), QEvent(QEvent.Type.Leave))

    assert not marker.isVisible()
    assert marker.cursor_time is None


def test_timeline_hover_marker_refreshes_geometry_after_resize(
    qapp: MockNeonPlayerApp, timeline_hover_marker: TimelineHoverState
) -> None:
    timeline, _, _, _, marker = timeline_hover_marker
    initial_width = marker.width()

    timeline.resize(1000, 500)
    qapp.processEvents()

    assert marker.height() == timeline.graphics_view.height()
    assert marker.width() > initial_width


def test_timestamp_label_and_time_axis_ticks_share_timestamp_formatting() -> None:
    time_axis = TimeAxisItem(orientation="top")
    recording_start = 1_700_000_000_000_000_000
    time_axis.set_time_frame(recording_start, recording_start + 5_000_000_000_000)

    label = TimestampLabel()
    label.set_time(3_723_456_789_000)
    assert label.text() == "01:02:03.457"

    time_axis.interval = 1
    assert time_axis.tickStrings(
        [recording_start + 1_500_000_000, recording_start + 3_723_000_000_000],
        1,
        1,
    ) == ["00:01", "01:02:03"]

    time_axis.interval = 0.5
    assert time_axis.tickStrings(
        [recording_start + 3_723_456_789_000],
        1,
        1,
    ) == ["01:02:03.457"]

    assert (
        format_timestamp_ns(
            recording_start + 59_999_600_000,
            origin_ns=recording_start,
            always_show_hours=True,
            show_milliseconds=True,
        )
        == "00:01:00.000"
    )


def test_timeline_hover_marker_hides_outside_recording_interval(
    timeline_hover_marker: TimelineHoverState,
) -> None:
    _, _, _, _, marker = timeline_hover_marker

    marker.set_cursor_time(-1)
    assert not marker.isVisible()
    assert marker.cursor_time is None

    marker.set_cursor_time(0)
    assert marker.isVisible()
    assert marker.cursor_time == 0

    marker.set_cursor_time(100)
    assert marker.isVisible()
    assert marker.cursor_time == 100

    marker.set_cursor_time(101)
    assert not marker.isVisible()
    assert marker.cursor_time is None


def test_timeline_hover_marker_draws_clamped_white_on_grey_badge(
    qapp: MockNeonPlayerApp, timeline_hover_marker: TimelineHoverState
) -> None:
    _, _, _, pinned_axis, marker = timeline_hover_marker
    assert marker.label.parentWidget() is pinned_axis
    marker.set_cursor_time(1)
    qapp.processEvents()

    marker.label.refresh_geometry()
    image = marker.label.grab().toImage()
    badge_edge_pixel = image.pixelColor(1, image.height() // 2)

    assert badge_edge_pixel.red() == 102
    assert badge_edge_pixel.green() == 102
    assert badge_edge_pixel.blue() == 102
