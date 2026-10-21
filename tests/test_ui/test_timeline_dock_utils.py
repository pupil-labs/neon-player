import inspect
from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QLabel, QWidget

from pupil_labs import neon_player
from pupil_labs.neon_player.ui.timeline_dock import TimeLineDock
from pupil_labs.neon_player.ui.timeline_dock_components import TimelineTooltipWidget
from pupil_labs.neon_player.ui.timeline_dock_utils import (
    TimelineTooltipContext,
    TimelineTooltipSeries,
    default_alt_timeline_tooltip,
    default_timeline_tooltip,
    get_clicked_data_point,
    get_item_tooltip_series,
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
    assert get_item_tooltip_series(line, 111, 10).value is None


def test_timeline_row_values_includes_only_active_broken_bar_values():
    bars = pg.BarGraphItem(
        x0=np.array([10, 40]),
        x1=np.array([30, 60]),
        y0=-0.4,
        y1=0.4,
        name="interval",
    )
    bars.bar_values = np.array([7.0, np.nan])

    assert get_item_tooltip_series(bars, 20, 0).value == 7.0
    assert get_item_tooltip_series(bars, 50, 0).value == "Active"
    assert get_item_tooltip_series(bars, 35, 0).value is None
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

    TimeLineDock._register_timeline_tooltip(owner, "row", first, None)
    TimeLineDock._register_timeline_tooltip(owner, "row", second, None)

    assert owner.timeline_tooltips == {"row": (second, None)}


def test_none_tooltip_clears_registered_callback():
    owner = SimpleNamespace(timeline_tooltips={"row": lambda context: "shown"})

    TimeLineDock._register_timeline_tooltip(owner, "row", None, None)

    assert owner.timeline_tooltips == {"row": (None, None)}


def test_omitted_tooltip_registers_default_callback():
    owner = SimpleNamespace(timeline_tooltips={})
    default = inspect.signature(TimeLineDock.add_timeline_plot).parameters[
        "tooltip"
    ].default

    alt_default = inspect.signature(TimeLineDock.add_timeline_plot).parameters[
        "alt_tooltip"
    ].default
    TimeLineDock._register_timeline_tooltip(owner, "row", default, alt_default)

    assert owner.timeline_tooltips == {
        "row": (default_timeline_tooltip, default_alt_timeline_tooltip)
    }


def test_custom_alternative_tooltip_is_registered_and_none_can_hide_it():
    custom_alt = lambda context: "alternate"
    owner = SimpleNamespace(timeline_tooltips={})
    normal_default = inspect.signature(TimeLineDock.add_timeline_plot).parameters[
        "tooltip"
    ].default

    TimeLineDock._register_timeline_tooltip(owner, "row", normal_default, custom_alt)
    assert owner.timeline_tooltips["row"] == (default_timeline_tooltip, custom_alt)

    TimeLineDock._register_timeline_tooltip(owner, "row", normal_default, None)
    assert owner.timeline_tooltips["row"] == (default_timeline_tooltip, None)


def test_custom_widget_is_adopted_and_released_when_replaced(qapp):
    class CustomTooltip(QLabel):
        def update_tooltip(self, timestamp_ns, values):
            pass

    parent = QWidget()
    host = TimelineTooltipWidget(parent)
    owner = SimpleNamespace(
        timeline_tooltips={},
        _tooltip_widget_rows={},
        tooltip_widget=host,
    )
    widget = CustomTooltip()

    TimeLineDock._register_timeline_tooltip(owner, "row", widget, widget)
    assert widget.parent() is host
    assert owner.timeline_tooltips["row"] == (widget, widget)

    TimeLineDock._register_timeline_tooltip(owner, "row", lambda context: "text", None)
    assert widget.parent() is None
    assert id(widget) not in owner._tooltip_widget_rows


def test_invalid_custom_widgets_are_rejected_without_taking_ownership():
    owner = SimpleNamespace(
        timeline_tooltips={},
        _tooltip_widget_rows={},
        tooltip_widget=TimelineTooltipWidget(QWidget()),
    )
    widget_without_update = QLabel()

    with pytest.raises(TypeError, match="update_tooltip"):
        TimeLineDock._register_timeline_tooltip(
            owner, "row", widget_without_update, None
        )

    assert widget_without_update.parent() is None


def test_parented_custom_widget_is_rejected_and_stays_caller_owned():
    parent = QWidget()

    class CustomTooltip(QLabel):
        def update_tooltip(self, timestamp_ns, values):
            pass

    widget = CustomTooltip(parent)
    owner = SimpleNamespace(
        timeline_tooltips={},
        _tooltip_widget_rows={},
        tooltip_widget=TimelineTooltipWidget(QWidget()),
    )

    with pytest.raises(ValueError, match="parentless"):
        TimeLineDock._register_timeline_tooltip(owner, "row", widget, None)

    assert widget.parent() is parent


def test_default_timeline_tooltip_formats_single_series_value():
    context = TimelineTooltipContext(
        100, {"speed": TimelineTooltipSeries(value=3.5, kind="point")}
    )

    assert default_timeline_tooltip(context) == "3.500"


def test_default_timeline_tooltip_formats_multiple_series_and_omits_none():
    context = TimelineTooltipContext(
        100,
        {
            "speed": TimelineTooltipSeries(value=3.5, kind="point"),
            "temperature": TimelineTooltipSeries(value=None, kind="point"),
            "altitude": TimelineTooltipSeries(value=42, kind="point"),
        },
    )

    assert (
        default_timeline_tooltip(context)
        == '<table cellspacing="0" cellpadding="0">'
        '<tr><td style="padding-right: 6px"><b>speed</b>:</td>'
        '<td style="text-align: right">3.500</td></tr>'
        '<tr><td style="padding-right: 6px"><b>altitude</b>:</td>'
        '<td style="text-align: right">42</td></tr>'
        "</table>"
    )


def test_default_timeline_tooltip_formats_broken_bar_index_and_duration():
    context = TimelineTooltipContext(
        100,
        {
            "interval": TimelineTooltipSeries(
                value=7.0,
                kind="interval",
                index=2,
                start_ns=100_000_000,
                stop_ns=350_000_000,
            )
        },
    )

    assert default_timeline_tooltip(context) == "#3 250ms"


def test_default_timeline_tooltip_rounds_broken_bar_duration_to_whole_ms():
    context = TimelineTooltipContext(
        100,
        {
            "interval": TimelineTooltipSeries(
                value=7.0,
                kind="interval",
                index=2,
                start_ns=100_000_000,
                stop_ns=350_700_000,
            )
        },
    )

    assert default_timeline_tooltip(context) == "#3 251ms"


def test_default_alt_timeline_tooltip_keeps_line_value_unchanged():
    context = TimelineTooltipContext(
        100, {"speed": TimelineTooltipSeries(value=3.5, kind="point")}
    )

    assert default_alt_timeline_tooltip(context) == "3.500"


def test_default_alt_timeline_tooltip_adds_exact_interval_timestamps():
    context = TimelineTooltipContext(
        100,
        {
            "interval": TimelineTooltipSeries(
                value=7.0,
                kind="interval",
                index=2,
                start_ns=123_456_789,
                stop_ns=987_654_321,
            )
        },
    )

    tooltip = default_alt_timeline_tooltip(context)
    assert tooltip.startswith("#3 864ms")
    assert "Start:</td>" in tooltip
    assert "Stop:</td>" in tooltip
    assert "interval" not in tooltip
    assert "123456789 ns" in tooltip
    assert "987654321 ns" in tooltip


def test_default_alt_timeline_tooltip_omits_names_for_multiple_broken_bar_series():
    context = TimelineTooltipContext(
        100,
        {
            "left bar": TimelineTooltipSeries(
                value=1,
                kind="interval",
                index=0,
                start_ns=10,
                stop_ns=20,
            ),
            "right bar": TimelineTooltipSeries(
                value=1,
                kind="interval",
                index=0,
                start_ns=30,
                stop_ns=50,
            ),
        },
    )

    assert default_alt_timeline_tooltip(context) == ""


def test_alt_tooltip_key_events_switch_mode_and_refresh():
    refreshed = []
    owner = SimpleNamespace(
        _alt_tooltip_active=False,
        _refresh_tooltip_for_hover=lambda: refreshed.append(owner._alt_tooltip_active),
    )

    TimeLineDock._on_alt_tooltip_key_event(
        owner,
        QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_Alt, Qt.KeyboardModifier.AltModifier
        ),
    )
    TimeLineDock._on_alt_tooltip_key_event(
        owner,
        QKeyEvent(
            QEvent.Type.KeyRelease, Qt.Key.Key_Alt, Qt.KeyboardModifier.NoModifier
        ),
    )

    assert refreshed == [True, False]


def test_default_timeline_tooltip_hides_when_all_values_are_none():
    context = TimelineTooltipContext(
        100,
        {
            "speed": TimelineTooltipSeries(value=None, kind="point"),
            "temperature": TimelineTooltipSeries(value=None, kind="point"),
        },
    )

    assert default_timeline_tooltip(context) == ""


def test_all_timeline_add_apis_expose_optional_tooltip():
    for name in (
        "add_timeline_plot",
        "add_timeline_line",
        "add_timeline_scatter",
        "add_timeline_broken_bar",
    ):
        parameter = inspect.signature(getattr(TimeLineDock, name)).parameters["tooltip"]
        assert parameter.default is not None
        assert inspect.signature(getattr(TimeLineDock, name)).parameters[
            "alt_tooltip"
        ].default is not None


def test_callback_receives_integral_timestamp_and_row_values():
    plot = SimpleNamespace(
        getViewBox=lambda: SimpleNamespace(viewPixelSize=lambda: (2.0, 1.0)),
        items=[pg.PlotDataItem([100], [5], name="value")],
    )
    received = []
    owner = SimpleNamespace(
        timeline_tooltips={
            "row": (lambda context: received.append(context) or "<b>ok</b>", None)
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


def test_custom_widget_receives_timestamp_and_row_values_in_alt_mode(qapp):
    class CustomTooltip(QLabel):
        def __init__(self):
            super().__init__()
            self.received = None

        def update_tooltip(self, timestamp_ns, values):
            self.received = (timestamp_ns, values)

    widget = CustomTooltip()
    plot = SimpleNamespace(
        getViewBox=lambda: SimpleNamespace(viewPixelSize=lambda: (2.0, 1.0)),
        items=[pg.PlotDataItem([100], [5], name="value")],
    )
    owner = SimpleNamespace(
        _alt_tooltip_active=True,
        timeline_tooltips={"row": (None, widget)},
        timeline_plots={"row": plot},
    )

    assert TimeLineDock._tooltip_content_for_hovered_row(owner, "row", 100.4) is widget
    assert widget.received == (100, {"value": 5})


def test_custom_widget_is_released_when_timeline_row_is_removed(qapp):
    class CustomTooltip(QLabel):
        def update_tooltip(self, timestamp_ns, values):
            pass

    parent = QWidget()
    host = TimelineTooltipWidget(parent)
    widget = CustomTooltip()
    owner = SimpleNamespace(
        timeline_tooltips={},
        _tooltip_widget_rows={},
        tooltip_widget=host,
        timeline_plots={"row": object()},
        timeline_legends={"row": SimpleNamespace(parentItem=lambda: object())},
        graphics_layout=SimpleNamespace(removeItem=lambda item: None),
        get_timeline_plot=lambda name: object(),
        sort_plots=lambda: None,
    )
    TimeLineDock._register_timeline_tooltip(owner, "row", widget, widget)

    TimeLineDock.remove_timeline_plot(owner, "row")

    assert widget.parent() is None
    assert id(widget) not in owner._tooltip_widget_rows


def test_alt_mode_uses_alternative_callback():
    plot = SimpleNamespace(
        getViewBox=lambda: SimpleNamespace(viewPixelSize=lambda: (1.0, 1.0)),
        items=[pg.PlotDataItem([100], [5], name="value")],
    )
    owner = SimpleNamespace(
        _alt_tooltip_active=True,
        timeline_tooltips={
            "row": (lambda context: "normal", lambda context: "alternative")
        },
        timeline_plots={"row": plot},
    )

    assert TimeLineDock._tooltip_text_for_hovered_row(owner, "row", 100) == "alternative"


def test_empty_callback_result_hides_tooltip():
    plot = SimpleNamespace(
        getViewBox=lambda: SimpleNamespace(viewPixelSize=lambda: (1.0, 1.0)),
        items=[],
    )
    owner = SimpleNamespace(
        timeline_tooltips={"row": (lambda context: "", None)},
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
