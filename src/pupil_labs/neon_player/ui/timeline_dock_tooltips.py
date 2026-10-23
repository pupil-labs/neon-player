import dataclasses
import html
import logging
import numpy as np
import pyqtgraph as pg
import typing as T

from bisect import bisect_left
from functools import partial
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from pupil_labs.neon_player.ui.timeline_dock_components import format_timestamp_ns


@dataclasses.dataclass(frozen=True)
class TimelineSeriesHover:
    name: str
    """The name of the time series, empty if not set."""
    timestamp_ns: int
    """The timestamp of the hovered data point."""
    index: int
    """The index of the hovered data point."""
    value: float
    """The y-axis value that corresponds to the data point."""


@dataclasses.dataclass(frozen=True)
class TimelineBarSeriesHover(TimelineSeriesHover):
    """Value and source details for one plotted series at the hovered time."""
    start_ts: int
    """The start timestamp of the hovered bar."""
    stop_ts: int
    """The stop timestamp of the hovered bar."""
    value: float | None
    """The bar value that corresponds to the bar, if set."""


@dataclasses.dataclass(frozen=True)
class TimelineLineSeriesHover(TimelineSeriesHover):
    """The closest line point to the current hover timestamp."""


@dataclasses.dataclass(frozen=True)
class TimelineScatterSeriesHover(TimelineSeriesHover):
    """The closest scatter point to the current hover timestamp"""


@dataclasses.dataclass(frozen=True)
class TimelineTooltipContext:
    """Stable callback input with row values and optional per-series details."""

    timestamp_ns: int
    """The timestamp of the hover marker."""
    series: list[TimelineSeriesHover]
    origin_ns: int = 0
    row_name: str = ""
    """The name of the hovered timeline row."""

    @property
    def values(self) -> dict[str, T.Any | None]:
        return {entry.name: entry.value for entry in self.series}


TimelineTooltipCallback: T.TypeAlias = T.Callable[[TimelineTooltipContext], str]
TimelineTooltipConfig: T.TypeAlias = TimelineTooltipCallback | QWidget


def format_value(value: float, precision: int, units: str) -> str:
    formatted = f"{value:.{precision}f}"
    return f"{formatted} {units}" if units else formatted


def format_duration(start_ts: int, stop_ts: int) -> str:
    duration_ms = (stop_ts - start_ts) / 1_000_000
    if duration_ms > 1_000:
        duration = f"{duration_ms / 1_000:.2f}".rstrip(".0") + "s"
    else:
        duration = f"{round(duration_ms)}ms"

    return duration


def _default_line_tooltip(
    context: TimelineTooltipContext,
    precision: int = 3,
    units: str = ""
) -> str:
    if not all(isinstance(s, TimelineLineSeriesHover) for s in context.series):
        logging.warning("Default line tooltip requires the plot to have only line series")
        return ""

    formatted_values = [format_value(s.value, precision, units) for s in context.series]
    if len(formatted_values) == 1:
        return formatted_values[0]

    names = [s.name or f"Series {index + 1}" for index, s in enumerate(context.series)]
    rows = "".join(
        "<tr>"
        f'<td style="padding-right: 6px"><b>{html.escape(name)}</b>:</td>'
        f'<td style="text-align: right">{value}</td>'
        "</tr>"
        for name, value in zip(names, formatted_values)
    )
    return f'<table cellspacing="0" cellpadding="0">{rows}</table>'


def default_line_tooltip(*, precision: int = 3, units: str = "") -> TimelineTooltipCallback:
    """Create a line tooltip formatter with optional numeric precision and units.

    The tooltip is formatted as `{value} {units}` if the plot contains one data series.
    For multiple series, a 2-column table (series name, `{value} {units}`) is shown."""
    return partial(_default_line_tooltip, precision=precision, units=units)


def default_scatter_tooltip(context: TimelineTooltipContext) -> str:
    """Format the sample time and value for exactly one scatter series.

    The tooltip shows the timestamp of the hovered scatter point."""
    if len(context.series) != 1 or not isinstance(
        context.series[0], TimelineScatterSeriesHover
    ):
        logging.warning("Default scatter tooltip requires exactly one scatter series")
        return ""

    entry = context.series[0]
    timestamp = format_timestamp_ns(
        entry.timestamp_ns,
        origin_ns=context.origin_ns,
        always_show_hours=True,
        show_milliseconds=True,
    )
    return timestamp


def default_broken_bar_tooltip(context: TimelineTooltipContext) -> str:
    """Default tooltip format for a row containing exactly one broken-bar series.

    The tooltip is formatted as `{bar index} {duration}. Bar index is 1-based.
    Duration is displayed in milliseconds for bars shorter than one second, or
    in seconds otherwise."""

    if len(context.series) != 1 or not isinstance(
        context.series[0], TimelineBarSeriesHover
    ):
        logging.warning(
            "Default broken-bar tooltip requires exactly one broken-bar series"
        )
        return ""

    entry = context.series[0]
    duration = format_duration(entry.start_ts, entry.stop_ts)
    return f"#{entry.index + 1} {duration}"


def _get_series_hover(
    item: T.Any, series_name: str, timestamp_ns: int, tolerance_ns: int
) -> TimelineSeriesHover | None:
    kind = "scatter" if item.opts.get("symbol") is not None else "point"
    timestamps, values = item.getData()
    if timestamps is None or values is None or len(timestamps) == 0:
        return None

    index = bisect_left(timestamps, timestamp_ns)
    candidates = [i for i in (index - 1, index) if 0 <= i < len(timestamps)]
    closest = min(candidates, key=lambda i: abs(timestamps[i] - timestamp_ns))
    if abs(timestamps[closest] - timestamp_ns) <= tolerance_ns:
        hover_type = (
            TimelineScatterSeriesHover
            if kind == "scatter"
            else TimelineLineSeriesHover
        )
        return hover_type(
            value=float(values[closest]),
            name=series_name,
            index=closest,
            timestamp_ns=int(timestamps[closest]) if kind == "scatter" else None,
        )
    return None


def _get_bar_graph_series_hover(
    item: pg.BarGraphItem, series_name: str, timestamp_ns: int
) -> TimelineBarSeriesHover | None:
    starts, stops = item.opts.get("x0"), item.opts.get("x1")
    if starts is None or stops is None:
        return None

    starts = np.asarray(starts)
    stops = np.asarray(stops)
    if starts.size != stops.size:
        logging.warning("Bar graph starts and stops must have equal lengths")
        return None

    matches = np.flatnonzero((starts <= timestamp_ns) & (timestamp_ns <= stops))
    if not matches.size:
        return None

    index = int(matches[0])
    start, stop = starts[index], stops[index]
    value = None
    values = getattr(item, "bar_values", None)
    if values is not None and index < len(values):
        value = values[index]

    return TimelineBarSeriesHover(
        value=value,
        name=series_name,
        timestamp_ns=timestamp_ns,
        index=index,
        start_ts=int(start),
        stop_ts=int(stop),
    )


def _get_item_name(item: pg.BarGraphItem | pg.PlotDataItem) -> str:
    name = getattr(item, "name", "")
    if callable(name):
        name = name()
    if not name:
        name = item.opts.get("name", "")

    return name


def _get_hovered_timeline_series(
    items: T.Iterable[T.Any], timestamp_ns: int, tolerance_ns: int
) -> list[TimelineSeriesHover]:
    values: list[TimelineSeriesHover] = []
    for item in items:
        if not isinstance(item, (pg.BarGraphItem, pg.PlotDataItem)):
            continue

        series_name = _get_item_name(item)
        if isinstance(item, pg.BarGraphItem):
            series = _get_bar_graph_series_hover(item, series_name, timestamp_ns)
        else:
            series = _get_series_hover(item, series_name, timestamp_ns, tolerance_ns)

        if series is None:
            continue

        values.append(series)

    return values


class TimelineTooltipWidget(QFrame):
    """Persistent, mouse-transparent tooltip surface for timeline hover content."""

    CURSOR_OFFSET = QPoint(14, 18)

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("timelineTooltip")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(
            "QFrame#timelineTooltip { background: #2b2b2b; color: #e0e0e0; "
            "border-radius: 4px; }"
        )
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(7, 4, 7, 4)
        self._text_label = QLabel(self)
        self._text_label.setTextFormat(Qt.TextFormat.RichText)
        self._text_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._text_label.setWordWrap(True)
        self._text_label.setMaximumWidth(420)
        self._content_widget: QWidget = self._text_label
        self._owned_content_widgets: dict[int, QWidget] = {}
        self._layout.addWidget(self._content_widget)
        self.hide()

    def show_text(self, text: str, global_cursor_position: QPoint) -> None:
        if self._content_widget is not self._text_label:
            self.set_content_widget(self._text_label)
        self._text_label.setText(text)
        self.show_at(global_cursor_position)

    def show_widget(self, widget: QWidget, global_cursor_position: QPoint) -> None:
        self.set_content_widget(widget)
        self.show_at(global_cursor_position)

    def own_content_widget(self, widget: QWidget) -> None:
        """Adopt a parentless tooltip widget until its row releases it."""
        widget.setParent(self)
        widget.hide()
        self._owned_content_widgets[id(widget)] = widget

    def release_content_widget(self, widget: QWidget) -> None:
        """Release and schedule disposal of an adopted tooltip widget."""
        if self._content_widget is widget:
            self.set_content_widget(self._text_label)
        self._owned_content_widgets.pop(id(widget), None)
        self._layout.removeWidget(widget)
        widget.hide()
        widget.setParent(None)
        widget.deleteLater()

    def show_at(self, global_cursor_position: QPoint) -> None:
        self.adjustSize()
        parent = self.parentWidget()
        if parent is None:
            return
        available = parent.rect().adjusted(4, 4, -4, -4)
        self.resize(min(self.width(), available.width()), min(self.height(), available.height()))
        position = parent.mapFromGlobal(global_cursor_position) + self.CURSOR_OFFSET
        position.setX(min(position.x(), available.right() - self.width() + 1))
        position.setY(min(position.y(), available.bottom() - self.height() + 1))
        position.setX(max(position.x(), available.left()))
        position.setY(max(position.y(), available.top()))
        self.move(position)
        self.raise_()
        self.show()

    def set_content_widget(self, widget: QWidget) -> None:
        """Replace the text child with custom content while retaining this surface."""
        if widget is self._content_widget:
            return
        self._layout.removeWidget(self._content_widget)
        self._content_widget.hide()
        self._content_widget = widget
        if widget is not self._text_label and id(widget) not in self._owned_content_widgets:
            self.own_content_widget(widget)
        widget.setParent(self)
        self._layout.addWidget(widget)
        widget.show()
