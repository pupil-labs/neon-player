import logging

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import QLabel, QScrollArea

from pupil_labs.neon_player.ui.timeline_dock_tooltips import (
    TimelineBarSeriesHover,
    TimelineLineSeriesHover,
    TimelineScatterSeriesHover,
    TimelineTooltipContext,
    TimelineTooltipOverlay,
    _get_hovered_timeline_series,
    default_broken_bar_tooltip,
    default_line_tooltip,
    default_scatter_tooltip,
)
from tests.conftest import MockNeonPlayerApp


def test_get_hovered_timeline_series__line_and_scatter() -> None:
    line = pg.PlotDataItem(
        [100, 200, 300], [1.0, 2.0, 3.0], pen=QPen(QColor("red")), name="Line"
    )
    scatter = pg.PlotDataItem(
        [100, 200, 300],
        [4.0, 5.0, 6.0],
        pen=None,
        symbol="o",
        symbolPen=QPen(QColor("blue")),
        name="Points",
    )

    hovered = _get_hovered_timeline_series([line, scatter], 205, 10)

    assert len(hovered) == 2
    assert isinstance(hovered[0], TimelineLineSeriesHover)
    assert (hovered[0].name, hovered[0].timestamp_ns, hovered[0].index) == (
        "Line",
        200,
        1,
    )
    assert hovered[0].value == 2.0
    assert isinstance(hovered[1], TimelineScatterSeriesHover)
    assert (hovered[1].name, hovered[1].timestamp_ns, hovered[1].index) == (
        "Points",
        200,
        1,
    )

    assert _get_hovered_timeline_series([line], 211, 10) == []


def test_get_hovered_timeline_series__bar_containment_and_values() -> None:
    bars = pg.BarGraphItem(
        x0=np.array([100, 300]),
        x1=np.array([200, 500]),
        y0=-0.4,
        y1=0.4,
        brushes=[QColor("red"), QColor("blue")],
        name="Intervals",
    )
    bars.bar_values = [2, 7]

    hovered = _get_hovered_timeline_series([bars], 400, tolerance_ns=0)

    assert len(hovered) == 1
    bar = hovered[0]
    assert isinstance(bar, TimelineBarSeriesHover)
    assert (
        bar.name,
        bar.index,
        bar.start_ts,
        bar.stop_ts,
        bar.value,
        bar.color.name(),
    ) == ("Intervals", 1, 300, 500, 7, "#0000ff")


def test_default_line_tooltip_single_series() -> None:
    line_context = TimelineTooltipContext(
        120,
        [TimelineLineSeriesHover("<signal>", 120, 0, 1.23456, QColor("red"))],
    )
    assert default_line_tooltip(precision=2, units="mV")(line_context) == "1.23 mV"


def test_default_line_tooltip_multiple_series() -> None:
    multi_line_context = TimelineTooltipContext(
        120,
        [
            TimelineLineSeriesHover("<signal>", 120, 0, 1.23456, QColor("red")),
            TimelineLineSeriesHover("second", 120, 0, 2.0, QColor("blue")),
        ],
    )
    multi_line = default_line_tooltip(precision=1)(multi_line_context)
    assert "&lt;signal&gt;" in multi_line
    assert "1.2" in multi_line
    assert "second" in multi_line


@pytest.mark.parametrize(
    ("timestamp", "expected"),
    [
        (1_500_000_000, "00:00:00.500"),
        (2_000_000_000, "00:00:01.000"),
        (11_000_000_000, "00:00:10.000"),
    ],
)
def test_default_scatter_tooltip(timestamp: int, expected: str) -> None:
    scatter_context = TimelineTooltipContext(
        timestamp,
        [TimelineScatterSeriesHover("", timestamp, 2, 1.0, None)],
        recording_start_time_ns=1_000_000_000,
    )
    scatter = default_scatter_tooltip(name="<event>", total=lambda: 5)(scatter_context)
    assert "&lt;event&gt;" in scatter
    assert "(3/5)" in scatter
    assert expected in scatter


@pytest.mark.parametrize(
    ("start_ts", "stop_ts", "expected"),
    [(0, 500_000_000, "500ms"), (0, 10_000_000_000, "10s")],
)
def test_default_broken_bar_tooltip(start_ts: int, stop_ts: int, expected: str) -> None:
    bar_context = TimelineTooltipContext(
        1_000_000_000,
        [TimelineBarSeriesHover("", 1_000_000_000, 0, None, None, start_ts, stop_ts)],
    )
    assert (
        default_broken_bar_tooltip(prefix="Aligned")(bar_context)
        == f"Aligned {expected}"
    )
    assert default_broken_bar_tooltip()(bar_context) == f"#1 {expected}"


def test_tooltip_callbacks_reject_mismatched_series_types(
    caplog: pytest.LogCaptureFixture,
) -> None:
    context = TimelineTooltipContext(
        100, [TimelineScatterSeriesHover("", 100, 0, 1.0, None)]
    )

    with caplog.at_level(logging.WARNING):
        assert default_line_tooltip()(context) == ""

    assert "only line series" in caplog.text


@pytest.mark.usefixtures("qapp")
def test_timeline_tooltip_overlay_adopts_and_releases_custom_widget(
    qapp: MockNeonPlayerApp,
) -> None:
    parent = QScrollArea()
    parent.resize(300, 200)
    parent.show()
    qapp.processEvents()
    overlay = TimelineTooltipOverlay(parent)
    content = QLabel("Tooltip")

    overlay.own_content_widget(content)
    assert content.parentWidget() is overlay
    overlay.show_widget(content, parent.mapToGlobal(parent.rect().center()))
    assert overlay.isVisible()
    assert overlay._content_widget is content

    overlay.release_content_widget(content)
    assert content.parentWidget() is None
    assert overlay._content_widget is overlay._text_label
    parent.close()
