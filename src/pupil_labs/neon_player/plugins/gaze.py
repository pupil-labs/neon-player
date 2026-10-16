import enum
import logging
import typing as T
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from PySide6.QtCore import QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen
from qt_property_widgets.utilities import (
    PersistentPropertiesMixin,
    action_params,
    property_params,
)

from pupil_labs import neon_player
from pupil_labs.neon_player import action
from pupil_labs.neon_player.managers.playback import PlaybackPosition
from pupil_labs.neon_player.overlays import InteractivePanelOverlay2D
from pupil_labs.neon_player.ui import ListPropertyAppenderAction
from pupil_labs.neon_player.ui.panel_widget import PanelWidget
from pupil_labs.neon_player.utilities import (
    cart_to_spherical,
    find_ranged_index,
    get_scene_intrinsics,
    unproject_points,
)
from pupil_labs.neon_recording import NeonRecording
from pupil_labs.neon_recording.timeseries import GazeTimeseries, WornTimeseries
from pupil_labs.neon_recording.sample import match_ts


class Aggregation(enum.Enum):
    Raw = "Raw"
    Mean = "Mean"
    Median = "Median"
    First = "First"
    Last = "Last"

    def apply(self, samples):
        if self is Aggregation.Raw or len(samples) == 0:
            v = samples

        elif self is Aggregation.Mean:
            v = samples.mean(axis=0)

        elif self is Aggregation.Median:
            v = np.median(samples, axis=0)

        elif self is Aggregation.First:
            v = samples[0]

        elif self is Aggregation.Last:
            v = samples[-1]

        if len(samples.shape) > 1:
            return v.reshape(-1, samples.shape[-1])

        else:
            return v.reshape(-1)


class GazeDataPlugin(neon_player.Plugin):
    label = "Gaze Data"
    offset_changed = Signal()

    def __init__(self) -> None:
        super().__init__()

        self.render_layer = 10
        self._offset_x = 0.0
        self._offset_y = 0.0
        self.worn_data: WornTimeseries | None = None

        self._visualizations: list[GazeVisualization] = [
            CircleViz(),
        ]
        self._connected_visualizations: set[GazeVisualization] = set()
        self._visualizations_attached = False
        for viz in self._visualizations:
            viz.changed.connect(self.changed.emit)
            self._connected_visualizations.add(viz)

        self.header_action = ListPropertyAppenderAction("visualizations", "+ Add viz")

    def on_recording_loaded(self, recording: NeonRecording) -> None:
        self._attach_visualizations(recording)
        self._visualizations_attached = True

        try:
            worn_data = recording.worn
            if worn_data is None:
                raise RuntimeError("Worn data is unavailable")
            self.worn_data = worn_data

            worn_on = worn_data.worn // 255
            worn_on = np.concatenate([[0], worn_on])
            state_diff = np.diff(worn_on.astype(int))

            start_times = worn_data.time[state_diff == 1].tolist()
            stop_times = worn_data.time[state_diff == -1].tolist()

            if len(stop_times) < len(start_times):
                stop_times.append(worn_data[-1][0])

            self.get_timeline().add_timeline_broken_bar(
                "Worn",
                list(zip(start_times, stop_times, strict=False)),
            )

        except Exception:
            self.worn_data = None
            logging.warning("Failed to load worn data")

    def on_disabled(self) -> None:
        if self.worn_data:
            self.get_timeline().remove_timeline_plot("Worn")
            self.worn_data = None

    def _get_gazes_for_visualization(
        self, viz: "GazeVisualization", timestamp: int
    ) -> npt.NDArray[np.float64]:
        return self._get_visualization_gazes(timestamp).get(
            viz, np.empty((0, 2), dtype=np.float64)
        )

    def _get_visualization_gazes(
        self, time_in_recording: int
    ) -> dict["GazeVisualization", npt.NDArray[np.float64]]:
        empty = np.empty((0, 2), dtype=np.float64)
        recording = self.recording
        if recording is None:
            return {viz: empty for viz in self._visualizations}

        scene_idx = self.get_scene_idx_for_time(time_in_recording)
        if scene_idx >= len(recording.scene) or scene_idx < 0:
            return {viz: empty for viz in self._visualizations}

        samples = self.get_gazes_for_scene(scene_idx)
        gazes = samples.point
        offset_gazes = apply_offset(recording, gazes, self.offset)

        worn_data = self.worn_data
        if worn_data is None:
            worns = np.full(len(gazes), 255)
        else:
            try:
                worns = worn_data.sample(
                    np.asarray(samples.time, dtype=np.int64)
                ).worn
            except Exception:
                worns = np.full(len(gazes), 255)

        aggregations = {}
        offset_aggregations = {}
        worn_aggregations = {}
        renderable_gazes = {}
        for viz in self._visualizations:
            if viz._aggregation not in aggregations:
                aggregations[viz._aggregation] = viz._aggregation.apply(gazes)
                offset_aggregations[viz._aggregation] = viz._aggregation.apply(
                    offset_gazes
                )
                worn_aggregations[viz._aggregation] = viz._aggregation.apply(worns)

            aggregation_dict = offset_aggregations if viz.use_offset else aggregations
            aggregated_gazes = aggregation_dict[viz._aggregation]
            if len(aggregated_gazes) == 0:
                renderable_gazes[viz] = empty
                continue

            renderable_mask = np.array([False] * len(aggregated_gazes))
            if viz.show_when_worn:
                renderable_mask |= worn_aggregations[viz._aggregation] >= 128

            if viz.show_when_not_worn:
                renderable_mask |= worn_aggregations[viz._aggregation] < 128

            renderable_gazes[viz] = aggregated_gazes[renderable_mask]

        return renderable_gazes

    def _attach_visualizations(self, recording: NeonRecording) -> None:
        panel = self.app.main_window.scene_panel
        for viz in self._visualizations:
            if not viz._attached_once:
                viz._bind_playback(self.panel_manager.playback)
            def provide_gazes(
                timestamp: int, visualization: GazeVisualization = viz
            ) -> npt.NDArray[np.float64]:
                return self._get_gazes_for_visualization(visualization, timestamp)

            viz._set_gaze_provider(provide_gazes)
            viz.on_recording_loaded(recording)
            if viz not in self._connected_visualizations:
                viz.changed.connect(self.changed.emit)
                self._connected_visualizations.add(viz)
            if not viz._attached_once:
                self.add_overlay(panel, viz, layer=self.render_layer)

    def get_gazes_for_scene(self, scene_idx: int = -1) -> GazeTimeseries:
        recording = self.recording
        if recording is None:
            raise RuntimeError("Gaze data requires a loaded recording")
        if scene_idx < 0:
            scene_idx = self.get_scene_idx_for_time()

        gaze_start_time = recording.scene[scene_idx].time
        after_mask = recording.gaze.time >= gaze_start_time

        if scene_idx < len(recording.scene) - 1:
            gaze_end_ts = recording.scene[scene_idx + 1].time
            before_mask = recording.gaze.time < gaze_end_ts
            time_mask = after_mask & before_mask
        else:
            time_mask = after_mask

        gazes: GazeTimeseries = recording.gaze[time_mask]
        return gazes

    @action
    @action_params(compact=True, icon=QIcon(str(neon_player.asset_path("export.svg"))))
    def export(self, destination: Path = Path()) -> None:
        if self.recording is None:
            return

        gaze = _prepare_gaze_export(
            self.recording,
            self.worn_data,
            self.app.get_export_window(),
            self.offset
        )

        if gaze.empty:
            logging.warning("No gaze data are present in the export window")
            return

        export_file = destination / "gaze.csv"
        gaze.to_csv(export_file, index=False)

        logging.info(f"Wrote {export_file}")

    @property
    @property_params(min=-1, max=1, step=0.01, decimals=3)
    def offset_x(self) -> float:
        return self._offset_x

    @offset_x.setter
    def offset_x(self, value: float) -> None:
        self._offset_x = value
        self.offset_changed.emit()

    @property
    @property_params(min=-1, max=1, step=0.01, decimals=3)
    def offset_y(self) -> float:
        return self._offset_y

    @offset_y.setter
    def offset_y(self, value: float) -> None:
        self._offset_y = value
        self.offset_changed.emit()

    @property
    @property_params(widget=None, dont_encode=True)
    def offset(self) -> tuple[float, float]:
        return (self._offset_x, self._offset_y)

    @property
    @property_params(
        use_subclass_selector=True,
        prevent_add=True,
        item_params={"label_field": "label"},
        primary=True,
    )
    def visualizations(self) -> list["GazeVisualization"]:
        return self._visualizations

    @visualizations.setter
    def visualizations(self, value: list["GazeVisualization"]) -> None:
        previous = self._visualizations
        self._visualizations = value

        for viz in value:
            if viz not in self._connected_visualizations:
                viz.changed.connect(self.changed.emit)
                self._connected_visualizations.add(viz)

        for viz in previous:
            if viz in value:
                continue
            if viz in self._connected_visualizations:
                viz.changed.disconnect(self.changed.emit)
                self._connected_visualizations.remove(viz)
            if self.recording is not None and viz._attached_once:
                panel = self.app.main_window.scene_panel
                host = panel.widget
                if isinstance(host, PanelWidget) and viz in host.overlays:
                    self.remove_overlay(panel, viz)

        recording = self.recording
        if self._visualizations_attached and recording is not None:
            self._attach_visualizations(recording)


class GazeVisualization(InteractivePanelOverlay2D, PersistentPropertiesMixin):
    changed = Signal()

    _known_types: T.ClassVar[list] = []

    def __init__(self, playback: PlaybackPosition | None = None) -> None:
        if playback is None:
            app = neon_player.instance()
            playback = getattr(app, "playback", None)
            if not isinstance(playback, PlaybackPosition):
                playback = PlaybackPosition()

        display_name = getattr(type(self), "label", type(self).__name__)
        InteractivePanelOverlay2D.__init__(self, playback, display_name)
        PersistentPropertiesMixin.__init__(self)
        self._use_offset = True
        self._aggregation = Aggregation.Mean
        self._show_when_worn = True
        self._show_when_not_worn = False

        self.recording: NeonRecording | None = None
        self._gaze_provider: T.Callable[
            [int], npt.NDArray[np.float64]
        ] | None = None

    @property
    @property_params(widget=None, dont_encode=True)
    def playback(self) -> PlaybackPosition:
        return self._playback

    @property
    @property_params(widget=None, dont_encode=True)
    def display_name(self) -> str:
        return self._display_name

    @property
    @property_params(widget=None, dont_encode=True)
    def hitbox(self) -> QRect | None:
        return None if self._hitbox is None else QRect(self._hitbox)

    @hitbox.setter
    def hitbox(self, value: QRect | None) -> None:
        if value is not None and not isinstance(value, QRect):
            raise TypeError("Overlay hitbox must be a QRect or None")
        self._hitbox = None if value is None else QRect(value)

    def render(
        self,
        painter: QPainter,
        gazes: npt.NDArray[np.float64],
    ) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    def paint(self, painter: QPainter) -> None:
        gazes = self._current_gazes()
        self._update_hitbox(gazes)
        self.render(painter, gazes)

    def on_click(self, point: QPointF) -> None:
        pass

    def hit_test(self, point: QPointF) -> bool:
        gazes = self._current_gazes()
        self._update_hitbox(gazes)
        hitbox = self.hitbox
        return hitbox is not None and QRectF(hitbox).contains(point)

    def render_hover_highlight(self, painter: QPainter) -> None:
        gazes = self._current_gazes()
        self._update_hitbox(gazes)
        hitbox = self.hitbox
        if hitbox is None:
            return
        painter.setPen(QPen(QColor(255, 255, 0, 220), 3))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRectF(hitbox))

    def _current_gazes(self) -> npt.NDArray[np.float64]:
        timestamp = self.playback.time
        if timestamp is None or self._gaze_provider is None:
            return np.empty((0, 2), dtype=np.float64)
        gazes = self._gaze_provider(timestamp)
        return gazes[np.isfinite(gazes).all(axis=1)]

    def _update_hitbox(self, gazes: npt.NDArray[np.float64]) -> None:
        self.hitbox = None

    def _bind_playback(self, playback: PlaybackPosition) -> None:
        if self.parent() is not None or self._attached_once:
            raise RuntimeError("Cannot change playback after overlay attachment")
        self._playback = playback

    def _set_gaze_provider(
        self,
        provider: T.Callable[[int], npt.NDArray[np.float64]] | None,
    ) -> None:
        self._gaze_provider = provider

    def on_removed(self) -> None:
        self._gaze_provider = None
        self.hitbox = None

    def __init_subclass__(cls: type["GazeVisualization"], **kwargs: dict) -> None:
        GazeVisualization._known_types.append(cls)

    def on_recording_loaded(self, recording: NeonRecording) -> None:
        self.recording = recording

    def to_dict(
        self,
        include_class_name: bool = False,
        condition: T.Callable[[dict], bool] | None = None,
        recursive: bool = False
    ) -> dict:
        state = super().to_dict(
            include_class_name=include_class_name,
            condition=condition,
            recursive=recursive
        )

        # NOTE: Gaze visualizations require the class name to be loaded correctly
        state["__class__"] = self.__class__.__name__
        return state

    @property
    def use_offset(self) -> bool:
        return self._use_offset

    @use_offset.setter
    def use_offset(self, value: bool) -> None:
        self._use_offset = value

    @property
    def aggregation(self) -> Aggregation:
        return self._aggregation

    @aggregation.setter
    def aggregation(self, value: Aggregation) -> None:
        self._aggregation = value

    @property
    def show_when_worn(self) -> bool:
        return self._show_when_worn

    @show_when_worn.setter
    def show_when_worn(self, value: bool) -> None:
        self._show_when_worn = value

    @property
    def show_when_not_worn(self) -> bool:
        return self._show_when_not_worn

    @show_when_not_worn.setter
    def show_when_not_worn(self, value: bool) -> None:
        self._show_when_not_worn = value


class CircleViz(GazeVisualization):
    label = "Circle"

    def __init__(self, playback: PlaybackPosition | None = None) -> None:
        super().__init__(playback)
        self._color = QColor(255, 0, 0, 128)
        self._radius = 30
        self._stroke_width = 10
        self._applied_radius: float = self._radius
        self._applied_stroke_width: float = self._stroke_width

    def cap_radius_stroke(self) -> None:
        """
        If stroke width exceeds 2 * radius (i.e., the user tries to fill the whole
        circle), the circle is rendered incorrectly on exported frames. To prevent this
        we cap the stroke width to be at most 2 * radius, and adjust the radius
        accordingly to maintain the overall size of the rendered circle.
        """
        self._applied_radius = self.radius
        self._applied_stroke_width = self.stroke_width

        if self.stroke_width <= 2 * self.radius:
            return

        self._applied_radius = self.radius / 2 + self.stroke_width / 4
        self._applied_stroke_width = 2 * self._applied_radius

    def render(self, painter: QPainter, gazes: npt.NDArray[np.float64]) -> None:
        pen = painter.pen()
        pen.setWidthF(self._applied_stroke_width)
        pen.setColor(self._color)
        painter.setPen(pen)

        for gaze in gazes:
            center = QPointF(gaze[0], gaze[1])
            painter.drawEllipse(center, self._applied_radius, self._applied_radius)

    def hit_test(self, point: QPointF) -> bool:
        gazes = self._current_gazes()
        self._update_hitbox(gazes)
        radius = self._applied_radius + self._applied_stroke_width / 2
        return any(
            (point.x() - gaze[0]) ** 2 + (point.y() - gaze[1]) ** 2 <= radius**2
            for gaze in gazes
        )

    def render_hover_highlight(self, painter: QPainter) -> None:
        gazes = self._current_gazes()
        self._update_hitbox(gazes)
        radius = self._applied_radius + self._applied_stroke_width / 2 + 2
        painter.setPen(QPen(QColor(255, 255, 0, 220), 3))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for gaze in gazes:
            painter.drawEllipse(QPointF(gaze[0], gaze[1]), radius, radius)

    def _update_hitbox(self, gazes: npt.NDArray[np.float64]) -> None:
        if len(gazes) == 0:
            self.hitbox = None
            return
        extent = self._applied_radius + self._applied_stroke_width / 2
        left = min(gaze[0] for gaze in gazes) - extent
        top = min(gaze[1] for gaze in gazes) - extent
        right = max(gaze[0] for gaze in gazes) + extent
        bottom = max(gaze[1] for gaze in gazes) + extent
        self.hitbox = QRectF(left, top, right - left, bottom - top).toAlignedRect()

    @property
    def color(self) -> QColor:
        return self._color

    @color.setter
    def color(self, value: QColor) -> None:
        self._color = value

    @property
    @property_params(min=1, max=999)
    def radius(self) -> int:
        return self._radius

    @radius.setter
    def radius(self, value: int) -> None:
        self._radius = value
        self.cap_radius_stroke()

    @property
    @property_params(min=1, max=999)
    def stroke_width(self) -> int:
        return self._stroke_width

    @stroke_width.setter
    def stroke_width(self, value: int) -> None:
        self._stroke_width = value
        self.cap_radius_stroke()

class CrosshairViz(GazeVisualization):
    label = "Crosshair"

    def __init__(self, playback: PlaybackPosition | None = None) -> None:
        super().__init__(playback)
        self._color = QColor(0, 255, 0, 128)
        self._size = 40
        self._gap_size = 20
        self._stroke_width = 5
        self._draw_dot = True

    def render(
        self,
        painter: QPainter,
        gazes: npt.NDArray[np.float64],
    ) -> None:
        self._draw_crosshairs(painter, gazes, self._color, self._stroke_width)

    def hit_test(self, point: QPointF) -> bool:
        gazes = self._current_gazes()
        self._update_hitbox(gazes)
        half_stroke = self._stroke_width / 2
        for gaze in gazes:
            dx = abs(point.x() - gaze[0])
            dy = abs(point.y() - gaze[1])
            if self._draw_dot and dx**2 + dy**2 <= half_stroke**2:
                return True
            horizontal_arm = (
                self._gap_size <= dx <= self._gap_size + self._size
                and dy <= half_stroke
            )
            vertical_arm = (
                self._gap_size <= dy <= self._gap_size + self._size
                and dx <= half_stroke
            )
            if horizontal_arm or vertical_arm:
                return True
        return False

    def render_hover_highlight(self, painter: QPainter) -> None:
        gazes = self._current_gazes()
        self._update_hitbox(gazes)
        self._draw_crosshairs(
            painter, gazes, QColor(255, 255, 0, 220), self._stroke_width + 5
        )

    def _update_hitbox(self, gazes: npt.NDArray[np.float64]) -> None:
        if len(gazes) == 0:
            self.hitbox = None
            return
        extent = self._gap_size + self._size + self._stroke_width / 2
        left = min(gaze[0] for gaze in gazes) - extent
        top = min(gaze[1] for gaze in gazes) - extent
        right = max(gaze[0] for gaze in gazes) + extent
        bottom = max(gaze[1] for gaze in gazes) + extent
        self.hitbox = QRectF(left, top, right - left, bottom - top).toAlignedRect()

    def _draw_crosshairs(
        self,
        painter: QPainter,
        gazes: npt.NDArray[np.float64],
        color: QColor,
        stroke_width: int,
    ) -> None:
        brush = painter.brush()
        pen = painter.pen()
        pen.setColor(Qt.GlobalColor.transparent)
        painter.setPen(pen)
        painter.setBrush(color)

        if self._draw_dot:
            for gaze in gazes:
                painter.drawEllipse(
                    QPointF(gaze[0], gaze[1]),
                    stroke_width / 2,
                    stroke_width / 2,
                )

        pen.setWidth(stroke_width)
        pen.setColor(color)
        painter.setPen(pen)

        for gaze in gazes:
            center = QPointF(gaze[0], gaze[1])
            painter.drawLine(
                QPointF(center.x() - self._gap_size, center.y()),
                QPointF(center.x() - self._gap_size - self._size, center.y()),
            )
            painter.drawLine(
                QPointF(center.x() + self._gap_size + self._size, center.y()),
                QPointF(center.x() + self._gap_size, center.y()),
            )
            painter.drawLine(
                QPointF(center.x(), center.y() - self._gap_size),
                QPointF(center.x(), center.y() - self._gap_size - self._size),
            )
            painter.drawLine(
                QPointF(center.x(), center.y() + self._gap_size),
                QPointF(center.x(), center.y() + self._gap_size + self._size),
            )

        painter.setBrush(brush)

    @property
    def color(self) -> QColor:
        return self._color

    @color.setter
    def color(self, value: QColor) -> None:
        self._color = value

    @property
    @property_params(min=0, max=1920)
    def size(self) -> int:
        return self._size

    @size.setter
    def size(self, value: int) -> None:
        self._size = value

    @property
    @property_params(min=0, max=1920)
    def gap_size(self) -> int:
        return self._gap_size

    @gap_size.setter
    def gap_size(self, value: int) -> None:
        self._gap_size = value

    @property
    @property_params(min=1, max=100)
    def stroke_width(self) -> int:
        return self._stroke_width

    @stroke_width.setter
    def stroke_width(self, value: int) -> None:
        self._stroke_width = value

    @property
    def draw_dot(self) -> bool:
        return self._draw_dot

    @draw_dot.setter
    def draw_dot(self, value: bool) -> None:
        self._draw_dot = value


def apply_offset(
    recording: NeonRecording,
    gazes: np.ndarray,
    offset: tuple[float, float]
) -> np.ndarray:
    exit_conditions = [
        recording is None,
        recording.scene.width is None,
        recording.scene.height is None
    ]
    if any(exit_conditions):
        return gazes

    offset_x, offset_y = offset
    offset_pixels = np.array([
        offset_x * recording.scene.width,
        offset_y * recording.scene.height,
    ])
    return gazes + offset_pixels


def _prepare_gaze_export(
    recording: NeonRecording,
    worn_data: WornTimeseries | None,
    export_window: tuple[int, int],
    gaze_offset: tuple[float, float],
) -> pd.DataFrame:
    start_time, stop_time = export_window
    start_mask = recording.gaze.time >= start_time
    stop_mask = recording.gaze.time <= stop_time

    export_gazes = recording.gaze[start_mask & stop_mask]
    if not len(export_gazes):
        return pd.DataFrame()

    export_gaze_points = apply_offset(recording, export_gazes.point, gaze_offset)

    scene_camera_matrix, scene_distortion_coefficients = get_scene_intrinsics(recording)
    spherical_coords = cart_to_spherical(
        unproject_points(
            export_gaze_points,
            scene_camera_matrix,
            scene_distortion_coefficients,
        )
    )

    gaze = pd.DataFrame({
        "recording id": recording.info["recording_id"],
        "timestamp [ns]": export_gazes.time,
        "gaze x [px]": export_gaze_points[:, 0],
        "gaze y [px]": export_gaze_points[:, 1],
        "azimuth [deg]": spherical_coords[2],
        "elevation [deg]": spherical_coords[1],
    })
    if worn_data:
        gaze["worn"] = _prepare_worn_export(export_gazes, worn_data)
    else:
        logging.warning("No worn data to export")

    try:
        matched_fixation_ids = (
            find_ranged_index(
                export_gazes.time,
                recording.fixations.start_time,
                recording.fixations.stop_time,
            )
            + 1
        )
        gaze["fixation id"] = matched_fixation_ids
        gaze["fixation id"] = gaze["fixation id"].replace(0, None)
    except Exception:
        logging.warning("Failed to match fixations")

    try:
        matched_blink_ids = (
            find_ranged_index(
                export_gazes.time,
                recording.blinks.start_time,
                recording.blinks.stop_time,
            )
            + 1
        )
        gaze["blink id"] = matched_blink_ids
        gaze["blink id"] = gaze["blink id"].replace(0, None)
    except Exception:
        logging.warning("Failed to match blinks")

    return gaze


def _prepare_worn_export(gazes: GazeTimeseries, worn: WornTimeseries) -> np.ndarray:
    gaze_in_worn_idx = match_ts(gazes.time, worn.time)
    matched = gazes.time == worn.time[gaze_in_worn_idx]
    matched_indices = gaze_in_worn_idx[matched]
    export_worn = np.full_like(gazes.time, np.nan, dtype=np.float64)
    export_worn[matched] = worn.worn[matched_indices] / 255.0
    return export_worn
