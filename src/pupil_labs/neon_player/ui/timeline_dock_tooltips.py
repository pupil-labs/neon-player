import dataclasses
import html
import logging
import typing as T
from bisect import bisect_left
from functools import partial

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from pupil_labs.neon_player.ui.timeline_dock_components import format_timestamp_ns

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class TimelineSeriesHover:
    """Details for a time series whose data is under the timeline pointer.

    The tooltip context only contains subclasses of this class, depending on
    the series type:

      * line plots: [`TimelineLineSeriesHover`][]
      * scatter plots: [`TimelineScatterSeriesHover`][]
      * bar graph plots: [`TimelineBarSeriesHover`][]
    """

    name: str
    """The name of the time series, empty if not set."""
    timestamp_ns: int
    """The timestamp of the hovered data point."""
    index: int
    """The index of the hovered data point / bar."""
    value: float | None
    """The y-axis value that corresponds to the data point."""
    color: QColor | None
    """The color of line / scatter / bar elements."""


@dataclasses.dataclass(frozen=True)
class TimelineBarSeriesHover(TimelineSeriesHover):
    """Details for the first bar in a series that contains the hover time."""

    start_ts: int
    """The start timestamp of the hovered bar (in nanoseconds)."""
    stop_ts: int
    """The stop timestamp of the hovered bar (in nanoseconds)."""
    value: float | None
    """The bar value that corresponds to the bar, if set."""


@dataclasses.dataclass(frozen=True)
class TimelineLineSeriesHover(TimelineSeriesHover):
    """The closest line sample within the hover tolerance."""

    value: float
    """The y-axis value that corresponds to the data point."""


@dataclasses.dataclass(frozen=True)
class TimelineScatterSeriesHover(TimelineSeriesHover):
    """The closest scatter sample within the hover tolerance."""

    value: float
    """The y-axis value that corresponds to the data point."""


@dataclasses.dataclass(frozen=True)
class TimelineTooltipContext:
    """Hover context passed to a timeline row's tooltip callback or widget."""

    timestamp_ns: int
    """The timestamp of the hover marker (in nanoseconds)."""
    series: list[TimelineSeriesHover]
    """Details about the hovered data point / bar for each series in the plot."""
    recording_start_time_ns: int = 0
    """Recording start time (in nanoseconds).

    The start time is provided to allow presenting relative timestamps."""
    row_name: str = ""
    """The name of the hovered timeline row."""

    @property
    def values(self) -> dict[str, T.Any | None]:
        """Map each hovered series name to its value."""
        return {entry.name: entry.value for entry in self.series}


type TimelineTooltipCallback = T.Callable[[TimelineTooltipContext], str]
"""Callback returning the timeline tooltip string (optionally, as HTML)."""


class TimelineTooltipWidget(QWidget):
    """Base class for custom widgets used as timeline row tooltips."""

    def update_tooltip(self, context: TimelineTooltipContext) -> None:
        """Update this widget for the current pointer position.

        Args:
            context: The current hover context for the tooltip, provided by
                the timeline object.
        """
        raise NotImplementedError


type TimelineTooltipConfig = TimelineTooltipCallback | TimelineTooltipWidget
"""A tooltip callback or a custom timeline tooltip widget.

Custom widgets must be parentless when registered and can belong to only one
timeline row. The timeline adopts and disposes of the widget when its
configuration is replaced or the row is removed.
"""


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
    context: TimelineTooltipContext, precision: int = 3, units: str = ""
) -> str:
    formatted_values = []
    for s in context.series:
        if not isinstance(s, TimelineLineSeriesHover):
            logger.warning(
                "Default line tooltip requires the plot to have only line series"
            )
            return ""
        formatted_values.append(format_value(s.value, precision, units))

    if len(formatted_values) == 1:
        return formatted_values[0]

    names = [s.name or f"Series {index + 1}" for index, s in enumerate(context.series)]
    colors = [s.color.name() if s.color else "#1c2021" for s in context.series]
    rows = "".join(
        "<tr>"
        f'<td><span style="color: {color}">―</span></td>'
        '<td style="padding-left: 3px; padding-right: 6px">'
        f"<b>{html.escape(name)}</b>:"
        "</td>"
        f'<td style="text-align: right">{value}</td>'
        "</tr>"
        for name, value, color in zip(names, formatted_values, colors, strict=True)
    )
    return f'<table cellspacing="0" cellpadding="0">{rows}</table>'


def default_line_tooltip(
    *, precision: int = 3, units: str = ""
) -> TimelineTooltipCallback:
    """Create a line tooltip formatter with optional numeric precision and units.

    The tooltip is formatted as `{value} {units}` if the plot contains one data series.
    For multiple series, a 2-column table (series name, `{value} {units}`) is shown.

    Args:
        precision: How many digits to display after the floating point.
        units: Append units to the displayed value.
    """
    return partial(_default_line_tooltip, precision=precision, units=units)


def _default_scatter_tooltip(
    context: TimelineTooltipContext,
    name: str,
    total: int | T.Callable[[], int],
) -> str:
    if len(context.series) != 1:
        logger.warning("Default scatter tooltip requires exactly one scatter series")
        return ""

    entry = context.series[0]
    if not isinstance(entry, TimelineScatterSeriesHover):
        logger.warning("Default scatter tooltip requires exactly one scatter series")
        return ""

    total_count = total() if callable(total) else total
    timestamp = format_timestamp_ns(
        entry.timestamp_ns,
        origin_ns=context.recording_start_time_ns,
        always_show_hours=True,
        show_milliseconds=True,
    )
    return (
        f"<b>{html.escape(name)}</b> ({entry.index + 1}/{total_count})<br>{timestamp}"
    )


def default_scatter_tooltip(
    *, name: str, total: int | T.Callable[[], int]
) -> TimelineTooltipCallback:
    """Create a scatter tooltip showing its name, index, total, and sample time.

    Args:
        name: Descriptive name (e.g., of the event marker).
        total: The total number of scatter points in this row. It may be a
            callable when the number of samples can change over time.
    """
    return partial(_default_scatter_tooltip, name=name, total=total)


def _default_broken_bar_tooltip(context: TimelineTooltipContext, prefix: str) -> str:
    if len(context.series) != 1:
        logger.warning(
            "Default broken-bar tooltip requires exactly one broken-bar series"
        )
        return ""

    entry = context.series[0]
    if not isinstance(entry, TimelineBarSeriesHover):
        logger.warning(
            "Default broken-bar tooltip requires exactly one broken-bar series"
        )
        return ""

    duration = format_duration(entry.start_ts, entry.stop_ts)
    if prefix:
        return f"{prefix} {duration}"

    return f"#{entry.index + 1} {duration}"


def default_broken_bar_tooltip(*, prefix: str = "") -> TimelineTooltipCallback:
    """Default tooltip format for a row containing exactly one broken-bar series.

    The tooltip is formatted as `{bar index} {duration}. Bar index is 1-based.
    Duration is displayed in milliseconds for bars shorter than one second, or
    in seconds otherwise.

    Args:
        prefix: Optional prefix to display instead of the bar index.
    """
    return partial(_default_broken_bar_tooltip, prefix=prefix)


def _get_series_hover(
    item: pg.PlotDataItem,
    series_name: str,
    timestamp_ns: int,
    tolerance_ns: int,
) -> TimelineLineSeriesHover | TimelineScatterSeriesHover | None:
    kind = "scatter" if item.opts.get("symbol") is not None else "point"
    timestamps, values = item.getData()
    if timestamps is None or values is None or len(timestamps) == 0:
        return None

    index = bisect_left(timestamps, timestamp_ns)
    candidates = [i for i in (index - 1, index) if 0 <= i < len(timestamps)]
    closest = min(candidates, key=lambda i: abs(timestamps[i] - timestamp_ns))
    if abs(timestamps[closest] - timestamp_ns) > tolerance_ns:
        return None

    pen_key = "symbolPen" if kind == "scatter" else "pen"
    pen = item.opts.get(pen_key)
    color = pen.color() if isinstance(pen, QPen) else None

    return_type = (
        TimelineScatterSeriesHover if kind == "scatter" else TimelineLineSeriesHover
    )

    return return_type(
        value=float(values[closest]),
        name=series_name,
        index=closest,
        timestamp_ns=int(timestamps[closest]),
        color=color,
    )


def _get_bar_graph_series_hover(
    item: pg.BarGraphItem, series_name: str, timestamp_ns: int
) -> TimelineBarSeriesHover | None:
    starts, stops = item.opts.get("x0"), item.opts.get("x1")
    if starts is None or stops is None:
        return None

    starts = np.asarray(starts)
    stops = np.asarray(stops)
    if starts.size != stops.size:
        logger.warning("Bar graph starts and stops must have equal lengths")
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

    color = None
    brushes = getattr(item, "brushes", None)
    if (
        brushes is not None
        and index < len(brushes)
        and isinstance(brushes[index], QBrush)
    ):
        color = brushes[index].color()

    return TimelineBarSeriesHover(
        value=value,
        name=series_name,
        timestamp_ns=timestamp_ns,
        index=index,
        start_ts=int(start),
        stop_ts=int(stop),
        color=color,
    )


def _get_item_name(item: pg.BarGraphItem | pg.PlotDataItem) -> str:
    name = getattr(item, "name", "")
    if callable(name):
        name = name()
    if not name:
        name = item.opts.get("name", "")

    return name


def _get_hovered_timeline_series(
    items: T.Iterable[pg.BarGraphItem | pg.PlotDataItem],
    timestamp_ns: int,
    tolerance_ns: int,
) -> list[TimelineSeriesHover]:
    values: list[TimelineSeriesHover] = []
    for item in items:
        if not isinstance(item, (pg.BarGraphItem, pg.PlotDataItem)):
            continue

        series_name = _get_item_name(item)
        series: TimelineSeriesHover | None
        if isinstance(item, pg.BarGraphItem):
            series = _get_bar_graph_series_hover(item, series_name, timestamp_ns)
        else:
            series = _get_series_hover(item, series_name, timestamp_ns, tolerance_ns)

        if series is None:
            continue

        values.append(series)

    return values


class TimelineTooltipOverlay(QFrame):
    """Persistent mouse-transparent surface used to display row tooltip content.

    Text content is rendered as rich text. Custom tooltip widgets are adopted
    while registered and released when their row configuration changes.
    """

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
        self._text_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
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
        self.resize(
            min(self.width(), available.width()), min(self.height(), available.height())
        )
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
        if (
            widget is not self._text_label
            and id(widget) not in self._owned_content_widgets
        ):
            self.own_content_widget(widget)
        widget.setParent(self)
        self._layout.addWidget(widget)
        widget.show()


def _normalize_timeline_tooltip(
    tooltip: TimelineTooltipConfig | None,
) -> TimelineTooltipConfig | None:
    if tooltip is None:
        return None

    if isinstance(tooltip, QWidget):
        if not isinstance(tooltip, TimelineTooltipWidget):
            raise TypeError(
                "Timeline tooltip widgets must subclass TimelineTooltipWidget"
            )
        return tooltip

    if callable(tooltip):
        return tooltip

    raise TypeError(
        "Timeline tooltip must be a callback, TimelineTooltipWidget, or None"
    )
