import numpy as np
import pyqtgraph as pg
import typing as T

from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent

from pupil_labs.neon_player.ui.timeline_dock_components import SmartSizePlotItem



def get_clicked_plot_item(items: list[T.Any]) -> SmartSizePlotItem | None:
    for item in items:
        if isinstance(item, SmartSizePlotItem):
            return item
    return None


def get_clicked_data_point(
    plot_item: SmartSizePlotItem | None, event: MouseClickEvent
) -> tuple[int, float] | None:
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
        return original_ts[ts_index], original_y[ts_index]

    return None
