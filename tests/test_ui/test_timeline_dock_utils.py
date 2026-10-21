import inspect
from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF

from pupil_labs import neon_player
from pupil_labs.neon_player.ui.timeline_dock import TimeLineDock
from pupil_labs.neon_player.ui.timeline_dock_utils import (
    TimelineTooltipContext,
    get_clicked_data_point,
    get_item_value_at_timestamp,
    get_timeline_row_values,
)


def test_get_clicked_data_point__plot_item_none():
    assert get_clicked_data_point(None, None) is None


def test_timeline_row_values_uses_nearest_point_within_tolerance():
    line = pg.PlotDataItem([100, 200], [1.5, 2.5], name="speed")
    scatter = pg.PlotDataItem([110, 210], [3, 4], name="speed")

    assert {
        name: series.value
        for name, series in get_timeline_row_values([line, scatter], 108, 10).items()
    } == {
        "speed": 1.5,
        "speed (2)": 3,
    }
    assert get_item_value_at_timestamp(line, 111, 10) is None


def test_timeline_row_values_includes_only_active_broken_bar_values():
    bars = pg.BarGraphItem(
        x0=np.array([10, 40]),
        x1=np.array([30, 60]),
        y0=-0.4,
        y1=0.4,
        name="interval",
    )
    bars.bar_values = np.array([7.0, np.nan])

    assert get_item_value_at_timestamp(bars, 20, 0) == 7.0
    assert get_item_value_at_timestamp(bars, 50, 0) == "Active"
    assert get_item_value_at_timestamp(bars, 35, 0) is None
    interval = get_timeline_row_values([bars], 50, 0)["interval"]
    assert interval.index == 1
    assert (interval.start_ns, interval.stop_ns) == (40, 60)


def test_timeline_row_values_assigns_stable_names_for_unnamed_series():
    first = pg.PlotDataItem([100], [1])
    second = pg.PlotDataItem([100], [2])

    assert {
        name: series.value
        for name, series in get_timeline_row_values([first, second], 100, 0).items()
    } == {
        "Series 1": 1,
        "Series 2": 2,
    }


def test_tooltip_registration_replaces_callback_with_latest_non_none():
    first = lambda context: "first"
    second = lambda context: "second"
    owner = SimpleNamespace(timeline_tooltips={})

    TimeLineDock._register_timeline_tooltip(owner, "row", first)
    TimeLineDock._register_timeline_tooltip(owner, "row", second)

    assert owner.timeline_tooltips == {"row": second}


def test_none_tooltip_clears_registered_callback():
    owner = SimpleNamespace(timeline_tooltips={"row": lambda context: "shown"})

    TimeLineDock._register_timeline_tooltip(owner, "row", None)

    assert owner.timeline_tooltips == {}


def test_all_timeline_add_apis_expose_optional_tooltip():
    for name in (
        "add_timeline_plot",
        "add_timeline_line",
        "add_timeline_scatter",
        "add_timeline_broken_bar",
    ):
        parameter = inspect.signature(getattr(TimeLineDock, name)).parameters["tooltip"]
        assert parameter.default is None


def test_callback_receives_integral_timestamp_and_row_values():
    plot = SimpleNamespace(
        getViewBox=lambda: SimpleNamespace(viewPixelSize=lambda: (2.0, 1.0)),
        items=[pg.PlotDataItem([100], [5], name="value")],
    )
    received = []
    owner = SimpleNamespace(
        timeline_tooltips={
            "row": lambda context: received.append(context) or "<b>ok</b>"
        },
        timeline_plots={"row": plot},
    )

    assert (
        TimeLineDock._tooltip_text_for_hovered_row(owner, "row", 100.4)
        == "<b>ok</b>"
    )
    assert isinstance(received[0], TimelineTooltipContext)
    assert received[0].timestamp_ns == 100
    assert received[0].values == {"value": 5}
    assert received[0].series["value"].index == 0


def test_empty_callback_result_hides_tooltip():
    plot = SimpleNamespace(
        getViewBox=lambda: SimpleNamespace(viewPixelSize=lambda: (1.0, 1.0)),
        items=[],
    )
    owner = SimpleNamespace(
        timeline_tooltips={"row": lambda context: ""},
        timeline_plots={"row": plot},
    )

    assert TimeLineDock._tooltip_text_for_hovered_row(owner, "row", 1) == ""


def test_update_tooltip_hides_outside_recording_bounds(monkeypatch):
    hidden = []
    owner = SimpleNamespace(
        tooltip_widget=SimpleNamespace(hide=lambda: hidden.append(True)),
        timeline_plots={},
    )
    monkeypatch.setattr(
        neon_player,
        "instance",
        lambda: SimpleNamespace(
            recording=SimpleNamespace(start_time=100, stop_time=200)
        ),
    )

    TimeLineDock._update_tooltip(owner, QPointF(), 201)

    assert hidden == [True]
