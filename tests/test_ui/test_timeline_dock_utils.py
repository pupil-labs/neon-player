import inspect
from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QLabel, QWidget

from pupil_labs import neon_player
from pupil_labs.neon_player.ui.timeline_dock import TimeLineDock
from pupil_labs.neon_player.ui.timeline_dock_utils import get_clicked_data_point
from pupil_labs.neon_player.ui.timeline_dock_tooltips import (
    TimelineTooltipContext,
    TimelineBarSeriesHover,
    TimelineLineSeriesHover,
    TimelineScatterSeriesHover,
    TimelineTooltipWidget,
    default_broken_bar_tooltip,
    default_line_tooltip,
    default_scatter_tooltip,
    get_bar_graph_series_hover,
    get_hovered_timeline_series,
    get_series_hover,
)


def test_get_clicked_data_point__plot_item_none():
    assert get_clicked_data_point(None, None) is None


def test_timeline_row_values_uses_nearest_point_within_tolerance():
    line = pg.PlotDataItem([100, 200], [1.5, 2.5], name="speed")
    scatter = pg.PlotDataItem([110, 210], [3, 4], name="speed", symbol="o")

    hovered = get_hovered_timeline_series([line, scatter], 108, 10)
    assert [(series.name, series.value) for series in hovered] == [
        ("speed", 1.5),
        ("speed", 3),
    ]
    assert get_series_hover(line, "speed", 111, 10) is None


def test_bar_graph_series_hover_returns_hovered_interval_and_value():
    bars = pg.BarGraphItem(
        x0=np.array([10, 40]),
        x1=np.array([30, 60]),
        y0=-0.4,
        y1=0.4,
        name="interval",
    )
    bars.bar_values = np.array([7.0, np.nan])

    assert get_bar_graph_series_hover(bars, "interval", 20).value == 7.0
    assert np.isnan(get_bar_graph_series_hover(bars, "interval", 50).value)
    assert get_bar_graph_series_hover(bars, "interval", 35) is None
    interval = get_hovered_timeline_series([bars], 50, 0)[0]
    assert interval.index == 1
    assert (interval.start_ns, interval.stop_ns) == (40, 60)


def test_bar_graph_series_hover_without_bar_values_returns_none_value():
    bars = pg.BarGraphItem(
        x0=np.array([10]),
        x1=np.array([30]),
        y0=-0.4,
        y1=0.4,
        name="interval",
    )

    hover = get_bar_graph_series_hover(bars, "interval", 20)

    assert hover is not None
    assert hover.value is None


def test_timeline_row_values_assigns_stable_names_for_unnamed_series():
    first = pg.PlotDataItem([100], [1])
    second = pg.PlotDataItem([100], [2])

    assert [
        (series.name, series.value)
        for series in get_hovered_timeline_series([first, second], 100, 0)
    ] == [("", 1), ("", 2)]


def test_set_timeline_tooltip_replaces_callback():
    first = lambda context: "first"
    second = lambda context: "second"
    owner = SimpleNamespace(timeline_tooltips={})

    TimeLineDock.set_timeline_tooltip(owner, "row", first)
    TimeLineDock.set_timeline_tooltip(owner, "row", second)

    assert owner.timeline_tooltips == {"row": second}


def test_none_tooltip_clears_registered_callback():
    owner = SimpleNamespace(timeline_tooltips={"row": lambda context: "shown"})

    TimeLineDock.set_timeline_tooltip(owner, "row", None)

    assert owner.timeline_tooltips == {"row": None}


def test_adding_series_without_tooltip_preserves_row_tooltip(monkeypatch):
    callback = lambda context: "configured"

    class PlotItem:
        items = []

        def plot(self, *args, **kwargs):
            return SimpleNamespace(name="")

    plot_item = PlotItem()
    owner = SimpleNamespace(
        timeline_tooltips={"row": callback},
        get_timeline_plot=lambda *args: plot_item,
        timeline_legends={"row": SimpleNamespace(addItem=lambda *args: None)},
        plot_sorting_enabled=False,
    )
    monkeypatch.setattr(
        neon_player,
        "instance",
        lambda: SimpleNamespace(recording=object()),
    )

    TimeLineDock.add_timeline_plot(
        owner, "row", [], color=pg.mkColor("white"), tooltip=None
    )

    assert owner.timeline_tooltips["row"] is callback


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

    TimeLineDock.set_timeline_tooltip(owner, "row", widget)
    assert widget.parent() is host
    assert owner.timeline_tooltips["row"] is widget

    TimeLineDock.set_timeline_tooltip(owner, "row", lambda context: "text")
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
        TimeLineDock.set_timeline_tooltip(
            owner, "row", widget_without_update
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
        TimeLineDock.set_timeline_tooltip(owner, "row", widget)

    assert widget.parent() is parent


def test_line_tooltip_factory_formats_multiple_line_series():
    context = TimelineTooltipContext(
        100,
        [
            TimelineLineSeriesHover(name="speed", timestamp_ns=None, index=0, value=3.5),
            TimelineLineSeriesHover(name="altitude", timestamp_ns=None, index=0, value=42),
        ],
    )

    assert (
        default_line_tooltip()(context)
        == '<table cellspacing="0" cellpadding="0">'
        '<tr><td style="padding-right: 6px"><b>speed</b>:</td>'
        '<td style="text-align: right">3.500</td></tr>'
        '<tr><td style="padding-right: 6px"><b>altitude</b>:</td>'
        '<td style="text-align: right">42.000</td></tr>'
        "</table>"
    )


def test_broken_bar_tooltip_formats_index_and_duration():
    context = TimelineTooltipContext(
        100,
        [
            TimelineBarSeriesHover(
                name="interval",
                timestamp_ns=100,
                index=2,
                value=7.0,
                start_ns=100_000_000,
                stop_ns=350_000_000,
            ),
        ],
    )

    assert default_broken_bar_tooltip(context) == "#3 250ms"


def test_broken_bar_tooltip_rounds_duration_to_whole_ms():
    context = TimelineTooltipContext(
        100,
        [
            TimelineBarSeriesHover(
                name="interval",
                timestamp_ns=100,
                index=2,
                value=7.0,
                start_ns=100_000_000,
                stop_ns=350_700_000,
            ),
        ],
    )

    assert default_broken_bar_tooltip(context) == "#3 251ms"


def test_broken_bar_tooltip_formats_long_duration_as_seconds():
    context = TimelineTooltipContext(
        100,
        [
            TimelineBarSeriesHover(
                name="interval",
                timestamp_ns=100,
                index=2,
                value=7.0,
                start_ns=0,
                stop_ns=1_234_000_000,
            ),
        ],
    )

    assert default_broken_bar_tooltip(context) == "#3 1.23s"


def test_broken_bar_tooltip_warns_and_returns_empty_for_multiple_series(caplog):
    context = TimelineTooltipContext(
        100,
        [
            TimelineBarSeriesHover(name="first", timestamp_ns=100, index=0, value=7.0, start_ns=0, stop_ns=1),
            TimelineBarSeriesHover(name="second", timestamp_ns=100, index=0, value=8.0, start_ns=0, stop_ns=1),
        ],
    )

    assert default_broken_bar_tooltip(context) == ""
    assert "requires exactly one broken-bar series" in caplog.text


def test_line_tooltip_factory_warns_for_non_line_series(caplog):
    context = TimelineTooltipContext(
        100,
        [TimelineScatterSeriesHover(name="speed", timestamp_ns=100, index=0, value=3.5)],
    )

    assert default_line_tooltip()(context) == ""
    assert "only line series" in caplog.text


def test_line_tooltip_factory_formats_precision_and_units():
    context = TimelineTooltipContext(
        100, [TimelineLineSeriesHover(name="speed", timestamp_ns=None, index=0, value=3.5678)]
    )

    assert default_line_tooltip()(context) == "3.568"
    assert default_line_tooltip(precision=1, units="m/s")(context) == "3.6 m/s"


def test_line_tooltip_factory_formats_values_and_appends_units():
    integer_context = TimelineTooltipContext(
        100, [TimelineLineSeriesHover(name="count", timestamp_ns=None, index=0, value=42)]
    )
    float_context = TimelineTooltipContext(
        100, [TimelineLineSeriesHover(name="state", timestamp_ns=None, index=0, value=1.25)]
    )

    assert default_line_tooltip(precision=2, units="items")(integer_context) == (
        "42.00 items"
    )
    assert default_line_tooltip(precision=2, units="m")(float_context) == "1.25 m"


def test_scatter_tooltip_shows_sample_time():
    context = TimelineTooltipContext(
        200_000_000,
        [
            TimelineScatterSeriesHover(
                name="sample",
                timestamp_ns=123_456_789,
                index=0,
                value=3.5678,
            ),
        ],
        origin_ns=100_000_000,
    )

    assert default_scatter_tooltip(context) == "00:00:00.023"


def test_scatter_tooltip_ignores_value():
    context = TimelineTooltipContext(
        100,
        [
            TimelineScatterSeriesHover(
                name="sample", timestamp_ns=123_456_789, index=0, value="Active"
            ),
        ],
    )

    assert default_scatter_tooltip(context) == "00:00:00.123"


def test_scatter_tooltip_keeps_integral_values_integral():
    context = TimelineTooltipContext(
        200_000_000,
        [
            TimelineScatterSeriesHover(
                name="sample", timestamp_ns=123_456_789, index=0, value=42
            ),
        ],
    )

    assert default_scatter_tooltip(context) == "00:00:00.123"


@pytest.mark.parametrize(
    "series",
    [
        [],
        [TimelineLineSeriesHover(name="line", timestamp_ns=123, index=0, value=1.5)],
        [
            TimelineScatterSeriesHover(
                name="first", timestamp_ns=123, index=0, value=1.5
            ),
            TimelineScatterSeriesHover(
                name="second", timestamp_ns=456, index=1, value=2.5
            ),
        ],
    ],
    ids=["empty", "non-scatter", "multiple-scatter"],
)
def test_scatter_tooltip_warns_and_returns_empty_for_unsupported_series(
    series, caplog
):
    context = TimelineTooltipContext(100, series)

    assert default_scatter_tooltip(context) == ""
    assert "requires exactly one scatter series" in caplog.text


def test_broken_bar_tooltip_ignores_value():
    context = TimelineTooltipContext(
        100,
        [
            TimelineBarSeriesHover(
                name="interval",
                timestamp_ns=100,
                index=2,
                value=7.256,
                start_ns=100_000_000,
                stop_ns=350_700_000,
            ),
        ],
    )

    assert default_broken_bar_tooltip(context) == "#3 251ms"


def test_broken_bar_tooltip_ignores_nonnumeric_value():
    context = TimelineTooltipContext(
        100,
        [
            TimelineBarSeriesHover(
                name="interval",
                timestamp_ns=100,
                index=0,
                value="Active",
                start_ns=10,
                stop_ns=20,
            ),
        ],
    )

    assert default_broken_bar_tooltip(context) == "#1 0ms"


def test_all_timeline_add_apis_expose_optional_tooltip():
    for name in (
        "add_timeline_plot",
        "add_timeline_line",
        "add_timeline_scatter",
        "add_timeline_broken_bar",
    ):
        parameter = inspect.signature(getattr(TimeLineDock, name)).parameters["tooltip"]
        assert parameter.default is None
        assert "tooltip" in inspect.signature(getattr(TimeLineDock, name)).parameters


def test_timeline_tooltip_setter_is_public():
    assert callable(TimeLineDock.set_timeline_tooltip)


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
        TimeLineDock._tooltip_content_for_hovered_row(owner, "row", 100)
        == "<b>ok</b>"
    )
    assert isinstance(received[0], TimelineTooltipContext)
    assert received[0].timestamp_ns == 100
    assert received[0].values == {"value": 5}
    assert received[0].series[0].index == 0


def test_custom_widget_receives_timestamp_and_row_values(qapp):
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
        timeline_tooltips={"row": widget},
        timeline_plots={"row": plot},
    )

    assert TimeLineDock._tooltip_content_for_hovered_row(owner, "row", 100) is widget
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
        plot_sorting_enabled=True,
        get_timeline_plot=lambda name: object(),
        sort_plots=lambda: None,
    )
    TimeLineDock.set_timeline_tooltip(owner, "row", widget)

    TimeLineDock.remove_timeline_plot(owner, "row")

    assert widget.parent() is None
    assert id(widget) not in owner._tooltip_widget_rows


def test_empty_callback_result_hides_tooltip():
    plot = SimpleNamespace(
        getViewBox=lambda: SimpleNamespace(viewPixelSize=lambda: (1.0, 1.0)),
        items=[pg.PlotDataItem([1], [2], name="value")],
    )
    owner = SimpleNamespace(
        timeline_tooltips={"row": lambda context: ""},
        timeline_plots={"row": plot},
    )

    assert TimeLineDock._tooltip_content_for_hovered_row(owner, "row", 1) == ""


def test_update_tooltip_hides_outside_recording_bounds():
    hidden = []
    owner = SimpleNamespace(
        recording=SimpleNamespace(start_time=100, stop_time=200),
        tooltip_widget=SimpleNamespace(hide=lambda: hidden.append(True)),
        timeline_plots={},
    )

    TimeLineDock._update_tooltip(owner, QPointF(), 201)

    assert hidden == [True]
