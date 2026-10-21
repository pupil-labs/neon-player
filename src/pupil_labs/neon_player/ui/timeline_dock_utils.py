from bisect import bisect_left
import dataclasses
import html
import numbers
import typing as T

import numpy as np
import pyqtgraph as pg

from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent

from pupil_labs.neon_player.ui.timeline_dock_components import SmartSizePlotItem


@dataclasses.dataclass(frozen=True)
class TimelineTooltipSeries:
    """Value and source details for one plotted series at the hovered time."""

    value: T.Any | None
    kind: str
    index: int | None = None
    start_ns: int | None = None
    stop_ns: int | None = None


@dataclasses.dataclass(frozen=True)
class TimelineTooltipContext:
    """Stable callback input with row values and optional per-series details."""

    timestamp_ns: int
    series: dict[str, TimelineTooltipSeries]

    @property
    def values(self) -> dict[str, T.Any | None]:
        return {name: entry.value for name, entry in self.series.items()}


TimelineTooltipCallback: T.TypeAlias = T.Callable[[TimelineTooltipContext], str]


def default_timeline_tooltip(context: TimelineTooltipContext) -> str:
    """Format available row values for the timeline's default tooltip."""
    available = [
        (name, entry, _format_default_series_value(entry))
        for name, entry in context.series.items()
        if entry.value is not None
    ]
    if not available:
        return ""
    if len(context.series) == 1:
        return available[0][2]
    rows = "".join(
        "<tr>"
        f'<td style="padding-right: 6px"><b>{html.escape(name)}</b>:</td>'
        f'<td style="text-align: right">{html.escape(value)}</td>'
        "</tr>"
        for name, _entry, value in available
    )
    return f'<table cellspacing="0" cellpadding="0">{rows}</table>'


def default_alt_timeline_tooltip(context: TimelineTooltipContext) -> str:
    """Show default values and exact bounds for active intervals."""
    interval_series = [
        (name, entry)
        for name, entry in context.series.items()
        if entry.kind == "interval"
    ]
    if len(interval_series) > 1:
        return ""
    if not interval_series:
        return default_timeline_tooltip(context)

    intervals = [
        (name, entry)
        for name, entry in interval_series
        if entry.value is not None
        and entry.start_ns is not None
        and entry.stop_ns is not None
    ]
    if not intervals:
        return ""

    available_values = [
        _format_default_series_value(entry)
        for entry in context.series.values()
        if entry.value is not None
    ]
    if len(available_values) == 1:
        values = available_values[0]
    else:
        values = "<table cellspacing=\"0\" cellpadding=\"0\">" + "".join(
            f"<tr><td>{html.escape(value)}</td></tr>" for value in available_values
        ) + "</table>"

    details = "".join(
        "<tr>"
        '<td style="padding-right: 6px">Start:</td>'
        f'<td style="text-align: right">{entry.start_ns} ns</td>'
        "</tr><tr>"
        '<td style="padding-right: 6px">Stop:</td>'
        f'<td style="text-align: right">{entry.stop_ns} ns</td>'
        "</tr>"
        for _, entry in intervals
    )
    return f'{values}<table cellspacing="0" cellpadding="0">{details}</table>'


def _format_default_series_value(entry: TimelineTooltipSeries) -> str:
    if (
        entry.kind == "interval"
        and entry.index is not None
        and entry.start_ns is not None
        and entry.stop_ns is not None
    ):
        duration_ms = round((entry.stop_ns - entry.start_ns) / 1_000_000)
        return f"#{entry.index+1} {duration_ms}ms"
    value = entry.value
    if isinstance(value, numbers.Real) and not isinstance(value, numbers.Integral):
        return f"{value:.3f}"
    return str(value)


def get_clicked_plot_item(items: list[T.Any]) -> SmartSizePlotItem | None:
    for item in items:
        if isinstance(item, SmartSizePlotItem):
            return item
    return None


def get_clicked_data_point(
    plot_item: SmartSizePlotItem | None, event: MouseClickEvent
) -> tuple[int, T.SupportsFloat] | None:
    if plot_item is None:
        return None

    for item in plot_item.items:
        if not isinstance(item, pg.PlotDataItem):
            continue

        if not hasattr(item, "scatter") or not isinstance(item.scatter, pg.ScatterPlotItem):
            continue

        scatter = item.scatter
        clicked_data_pos = scatter.mapFromScene(event.scenePos())
        candidate_points = scatter.pointsAt(clicked_data_pos)
        if not candidate_points.size:
            continue

        # ScatterPlotItem.pointsAt() return all points that overlap with the
        # provided position, pick the closest point if multiple candidates
        closest_point = candidate_points[0]
        if candidate_points.size > 1:
            clicked_ts = clicked_data_pos.x()
            candidate_ts = np.array([p.pos().x() for p in candidate_points])
            closest_point_index = np.argmin(np.abs(candidate_ts - clicked_ts))
            closest_point = candidate_points[closest_point_index]

        # ScatterPlotItem stores timestamps as floats internally, thereby losing
        # precision. Original timestamps can be restored from the parent PlotDataItem
        original_ts, original_y = item.getData()
        selected_ts = closest_point.pos().x()
        ts_index = np.argmin(np.abs(original_ts.astype(float) - selected_ts))
        return original_ts[ts_index], T.cast(T.SupportsFloat, original_y[ts_index])

    return None


def get_item_tooltip_series(
    item: T.Any, timestamp_ns: int, tolerance_ns: int
) -> TimelineTooltipSeries:
    if isinstance(item, pg.PlotDataItem):
        timestamps, values = item.getData()
        if timestamps is None or values is None or len(timestamps) == 0:
            return TimelineTooltipSeries(value=None, kind="point")

        # TODO: ensure that the timestamps are sorted
        index = bisect_left(timestamps, timestamp_ns)
        candidates = [i for i in (index - 1, index) if 0 <= i < len(timestamps)]
        closest = min(candidates, key=lambda i: abs(timestamps[i] - timestamp_ns))
        if abs(timestamps[closest] - timestamp_ns) <= tolerance_ns:
            return TimelineTooltipSeries(
                value=values[closest], kind="point", index=closest
            )
        return TimelineTooltipSeries(value=None, kind="point")

    if isinstance(item, pg.BarGraphItem):
        options = item.opts
        starts = options.get("x0")
        stops = options.get("x1")
        if starts is None or stops is None:
            return TimelineTooltipSeries(value=None, kind="interval")
        for index, (start, stop) in enumerate(zip(starts, stops, strict=True)):
            if start <= timestamp_ns <= stop:
                values = getattr(item, "bar_values", None)
                value = "Active"
                if values is not None and index < len(values) and not np.isnan(values[index]):
                    value = values[index]
                return TimelineTooltipSeries(
                    value=value,
                    kind="interval",
                    index=index,
                    start_ns=int(start),
                    stop_ns=int(stop),
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
        key = name
        suffix = 2
        while key in values:
            key = f"{name} ({suffix})"
            suffix += 1
        series = get_item_tooltip_series(item, timestamp_ns, tolerance_ns)
        if isinstance(series.value, np.generic):
            series = dataclasses.replace(series, value=series.value.item())
        values[key] = series
    return values
