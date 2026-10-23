"""Tooltip formatters, data extraction, and the timeline tooltip surface."""

from bisect import bisect_left
import dataclasses
import html
import numbers
import typing as T

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from pupil_labs.neon_player.ui.timeline_dock_components import format_timestamp_ns


@dataclasses.dataclass(frozen=True)
class TimelineTooltipSeries:
    """Value and source details for one plotted series at the hovered time."""

    value: T.Any | None
    kind: str
    index: int | None = None
    start_ns: int | None = None
    stop_ns: int | None = None
    timestamp_ns: int | None = None


@dataclasses.dataclass(frozen=True)
class TimelineTooltipContext:
    """Stable callback input with row values and optional per-series details."""

    timestamp_ns: int
    series: dict[str, TimelineTooltipSeries]
    origin_ns: int = 0

    @property
    def values(self) -> dict[str, T.Any | None]:
        return {name: entry.value for name, entry in self.series.items()}


TimelineTooltipCallback: T.TypeAlias = T.Callable[[TimelineTooltipContext], str]
TimelineTooltipConfig: T.TypeAlias = TimelineTooltipCallback | QWidget


def _format_value(value: T.Any, *, precision: int, units: str) -> str:
    is_numeric = isinstance(value, numbers.Real) and not isinstance(value, bool)
    formatted = f"{value:.{precision}f}" if is_numeric and not isinstance(value, numbers.Integral) else str(value)
    return f"{formatted} {units}" if is_numeric and units else formatted


def _format_series_tooltip(
    context: TimelineTooltipContext,
    format_series: T.Callable[[TimelineTooltipContext, TimelineTooltipSeries], str],
) -> str:
    available = [
        (name, format_series(context, entry))
        for name, entry in context.series.items()
        if entry.value is not None
    ]
    if not available:
        return ""
    if len(context.series) == 1:
        return html.escape(available[0][1])
    rows = "".join(
        "<tr>"
        f'<td style="padding-right: 6px"><b>{html.escape(name)}</b>:</td>'
        f'<td style="text-align: right">{html.escape(value)}</td>'
        "</tr>"
        for name, value in available
    )
    return f'<table cellspacing="0" cellpadding="0">{rows}</table>'


def default_line_tooltip(*, precision: int = 3, units: str = "") -> TimelineTooltipCallback:
    """Create a line tooltip formatter with optional numeric precision and units."""
    return lambda context: _format_series_tooltip(
        context,
        lambda _context, entry: _format_value(entry.value, precision=precision, units=units),
    )


def default_scatter_tooltip(*, precision: int = 3, units: str = "") -> TimelineTooltipCallback:
    """Create a scatter tooltip formatter showing sample time and value."""
    def format_scatter(context: TimelineTooltipContext, entry: TimelineTooltipSeries) -> str:
        value = _format_value(entry.value, precision=precision, units=units)
        if entry.timestamp_ns is None:
            return value
        timestamp = format_timestamp_ns(
            entry.timestamp_ns, origin_ns=context.origin_ns,
            always_show_hours=True, show_milliseconds=True,
        )
        return f"{timestamp} ({value})"
    return lambda context: _format_series_tooltip(context, format_scatter)


def default_broken_bar_tooltip(*, precision: int = 3, units: str = "") -> TimelineTooltipCallback:
    """Create a broken-bar tooltip formatter with index, duration, and value."""
    def format_interval(_context: TimelineTooltipContext, entry: TimelineTooltipSeries) -> str:
        value = _format_value(entry.value, precision=precision, units=units)
        if entry.index is None or entry.start_ns is None or entry.stop_ns is None:
            return value
        duration_ms = round((entry.stop_ns - entry.start_ns) / 1_000_000)
        return f"#{entry.index + 1} {duration_ms}ms ({value})"
    return lambda context: _format_series_tooltip(context, format_interval)


def get_item_tooltip_series(item: T.Any, timestamp_ns: int, tolerance_ns: int) -> TimelineTooltipSeries:
    if isinstance(item, pg.PlotDataItem):
        kind = "scatter" if item.opts.get("symbol") is not None else "point"
        timestamps, values = item.getData()
        if timestamps is None or values is None or len(timestamps) == 0:
            return TimelineTooltipSeries(value=None, kind=kind)
        index = bisect_left(timestamps, timestamp_ns)
        candidates = [i for i in (index - 1, index) if 0 <= i < len(timestamps)]
        closest = min(candidates, key=lambda i: abs(timestamps[i] - timestamp_ns))
        if abs(timestamps[closest] - timestamp_ns) <= tolerance_ns:
            return TimelineTooltipSeries(
                value=values[closest], kind=kind, index=closest,
                timestamp_ns=int(timestamps[closest]) if kind == "scatter" else None,
            )
        return TimelineTooltipSeries(value=None, kind=kind)

    if isinstance(item, pg.BarGraphItem):
        starts, stops = item.opts.get("x0"), item.opts.get("x1")
        if starts is None or stops is None:
            return TimelineTooltipSeries(value=None, kind="interval")
        for index, (start, stop) in enumerate(zip(starts, stops, strict=True)):
            if start <= timestamp_ns <= stop:
                values = getattr(item, "bar_values", None)
                value = "Active"
                if values is not None and index < len(values) and not np.isnan(values[index]):
                    value = values[index]
                return TimelineTooltipSeries(
                    value=value, kind="interval", index=index,
                    start_ns=int(start), stop_ns=int(stop),
                )
        return TimelineTooltipSeries(value=None, kind="interval")
    return TimelineTooltipSeries(value=None, kind="unknown")


def get_timeline_row_values(
    items: T.Iterable[T.Any], timestamp_ns: int, tolerance_ns: int
) -> dict[str, TimelineTooltipSeries]:
    values: dict[str, TimelineTooltipSeries] = {}
    for index, item in enumerate(items):
        if not isinstance(item, (pg.PlotDataItem, pg.BarGraphItem)):
            continue
        name = getattr(item, "name", "")
        if callable(name):
            name = name()
        if not name:
            name = item.opts.get("name", "")
        name = str(name).strip() or f"Series {index + 1}"
        key, suffix = name, 2
        while key in values:
            key, suffix = f"{name} ({suffix})", suffix + 1
        series = get_item_tooltip_series(item, timestamp_ns, tolerance_ns)
        if isinstance(series.value, np.generic):
            series = dataclasses.replace(series, value=series.value.item())
        values[key] = series
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
