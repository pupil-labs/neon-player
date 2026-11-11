import logging
import typing as T
from collections import OrderedDict
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
from pupil_labs.neon_recording import NeonRecording
from pupil_labs.neon_recording.timeseries.fixations import FixationArray
from PySide6.QtCore import QKeyCombination, QObject, QPointF, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter
from qt_property_widgets.utilities import (
    PersistentPropertiesMixin,
    action_params,
    property_params,
)

from pupil_labs import neon_player
from pupil_labs.neon_player import action
from pupil_labs.neon_player.job_manager import ProgressUpdate
from pupil_labs.neon_player.plugins.gaze import GazeDataPlugin, apply_offset
from pupil_labs.neon_player.ui import ListPropertyAppenderAction
from pupil_labs.neon_player.ui.timeline_dock_tooltips import default_broken_bar_tooltip
from pupil_labs.neon_player.utilities import (
    cart_to_spherical,
    get_scene_intrinsics,
    unproject_points,
)

logger = logging.getLogger(__name__)


class FixationsPlugin(neon_player.Plugin):
    label = "Fixations & Saccades"

    def __init__(self) -> None:
        super().__init__()

        self._visualizations: list[FixationVisualization] = [
            ScanpathViz(),
            FixationCircleViz(),
        ]

        self.flow_dict: dict[int, dict[int, npt.NDArray[np.float64]]] = {}
        self.header_action = ListPropertyAppenderAction("visualizations", "+ Add viz")

    def seek_by_fixation(self, direction: int) -> None:
        recording = self.require_recording()
        if len(recording.fixations) == 0:
            return

        fixations_up_to_now = recording.fixations[
            recording.fixations.start_time <= self.app.current_ts
        ]

        current_idx = len(fixations_up_to_now) - 1
        idx = max(0, min(len(recording.fixations) - 1, current_idx + direction))
        self.app.seek_to(int(recording.fixations.start_time[idx]))

    def on_recording_loaded(self, recording: NeonRecording) -> None:
        for viz in self._visualizations:
            viz.on_recording_loaded(recording)

        if len(recording.fixations) == 0:
            return

        self._load_optic_flow()
        if not self.flow_dict and not self.app.headless:
            job = self.job_manager.run_background_action(
                "Calculate optic flow", "FixationsPlugin.bg_optic_flow"
            )
            job.finished.connect(self._load_optic_flow)

        self.fixations = recording.fixations

        # TODO(type-checking): Neon Recording mistypes column selection.
        self.get_timeline().add_timeline_broken_bar(
            "Fixations",
            self.fixations[["start_time", "stop_time"]],  # type: ignore[arg-type]
            tooltip=default_broken_bar_tooltip(),
        )

        self.register_action(
            "Playback/Next Fixation",
            QKeyCombination(Qt.Key.Key_S),
            lambda: self.seek_by_fixation(1),
        )
        self.register_action(
            "Playback/Previous Fixation",
            QKeyCombination(Qt.Key.Key_A),
            lambda: self.seek_by_fixation(-1),
        )

    def _load_optic_flow(self) -> None:
        cache_path = self.get_cache_path()
        if cache_path is None:
            return
        optic_flow_file = cache_path / "optic_flow_offsets.npz"
        if optic_flow_file.exists():
            data = np.load(optic_flow_file)
            self.flow_dict = {}

            if "flow_data" in data:
                flow_array = data["flow_data"]
                for row in flow_array:
                    fidx, frame_idx = int(row[0]), int(row[1])
                    if frame_idx not in self.flow_dict:
                        self.flow_dict[frame_idx] = {}
                    self.flow_dict[frame_idx][fidx] = np.array([row[2], row[3]])

    def render(self, painter: QPainter, time_in_recording: int) -> None:
        recording = self.recording
        if recording is None or not hasattr(self, "fixations"):
            return

        after_mask = self.fixations.start_time <= time_in_recording
        before_mask = self.fixations.stop_time > time_in_recording
        active_mask = after_mask & before_mask

        requires_history = any(
            getattr(viz, "requires_history", False) for viz in self._visualizations
        )

        if requires_history:
            past_mask = (self.fixations.stop_time <= time_in_recording) & (
                self.fixations.stop_time >= time_in_recording - 60_000_000_000
            )
            past_idx = np.where(past_mask)[0]

            if len(past_idx) > 7:
                past_idx = past_idx[-7:]

            active_idx = np.where(active_mask)[0]
            render_idx = np.concatenate((past_idx, active_idx))
            render_idx = np.sort(render_idx)
        else:
            render_idx = np.where(active_mask)[0]

        if len(render_idx) == 0:
            return

        fixations = self.fixations[render_idx]
        fixation_ids = render_idx + 1

        scene_idx = self.get_scene_idx_for_time(time_in_recording)
        frame_offsets = self.flow_dict.get(scene_idx, {})

        # TODO(type-checking): Neon Recording mistypes masked array selection.
        for viz in self._visualizations:
            viz.render(
                painter,
                fixations,  # type: ignore[arg-type]
                fixation_ids,
                frame_offsets,
                self.get_gaze_offset(),
                time_in_recording,
            )

    def get_gaze_offset(self) -> tuple[float, float]:
        gaze_plugin = neon_player.Plugin.get_instance_by_name("GazeDataPlugin")

        if not isinstance(gaze_plugin, GazeDataPlugin):
            return (0.0, 0.0)

        return gaze_plugin.offset

    def on_disabled(self) -> None:
        self.get_timeline().remove_timeline_plot("Fixations")
        self.unregister_action("Playback/Next Fixation")
        self.unregister_action("Playback/Previous Fixation")

    def get_export_fixations(self) -> pd.DataFrame:
        recording = self.require_recording()
        start_time, stop_time = self.app.get_export_window()
        start_mask = recording.fixations.stop_time > start_time
        stop_mask = recording.fixations.start_time < stop_time

        fixations_ids = np.arange(len(recording.fixations)) + 1

        fixations = recording.fixations[start_mask & stop_mask]
        fixation_ids = fixations_ids[start_mask & stop_mask]

        # TODO(type-checking): Neon Recording mistypes numerical field arrays.
        offset_means = apply_offset(
            recording,
            fixations.mean_gaze_point,  # type: ignore[arg-type]
            self.get_gaze_offset(),
        )

        scene_camera_matrix, scene_distortion_coefficients = get_scene_intrinsics(
            recording
        )
        spherical_coords = cart_to_spherical(
            unproject_points(
                offset_means,
                scene_camera_matrix,
                scene_distortion_coefficients,
            )
        )

        export_data = pd.DataFrame(
            {
                "recording id": recording.info["recording_id"],
                "fixation id": fixation_ids,
                "start timestamp [ns]": fixations.start_time,
                "end timestamp [ns]": fixations.stop_time,
                "duration [ms]": (fixations.stop_time - fixations.start_time) / 1e6,
                "fixation x [px]": offset_means[:, 0],
                "fixation y [px]": offset_means[:, 1],
                "azimuth [deg]": spherical_coords[2],
                "elevation [deg]": spherical_coords[1],
            }
        )

        return export_data

    def get_export_saccades(self) -> pd.DataFrame:
        recording = self.require_recording()
        start_time, stop_time = self.app.get_export_window()
        start_mask = recording.saccades.stop_time > start_time
        stop_mask = recording.saccades.start_time < stop_time

        saccades_ids = np.arange(len(recording.saccades)) + 1

        saccades = recording.saccades[start_mask & stop_mask]
        saccade_ids = saccades_ids[start_mask & stop_mask]

        export_data = pd.DataFrame(
            {
                "recording id": recording.info["recording_id"],
                "saccade id": saccade_ids,
                "start timestamp [ns]": saccades.start_time,
                "end timestamp [ns]": saccades.stop_time,
                "duration [ms]": (saccades.stop_time - saccades.start_time) / 1e6,
                "amplitude [deg]": saccades.amplitude,
                "mean velocity [px/s]": saccades.mean_velocity,
                "peak velocity [px/s]": saccades.max_velocity,
            }
        )

        return export_data

    @action
    @action_params(compact=True, icon=QIcon(str(neon_player.asset_path("export.svg"))))
    def export(self, destination: Path = Path()) -> None:
        export_fixations = self.get_export_fixations()

        export_file = destination / "fixations.csv"
        export_fixations.to_csv(export_file, index=False)
        logger.info(f"Exported fixations to '{export_file}'")

        export_saccades = self.get_export_saccades()

        export_file = destination / "saccades.csv"
        export_saccades.to_csv(export_file, index=False)
        logger.info(f"Exported saccades to '{export_file}'")

    @property
    @property_params(
        use_subclass_selector=True,
        prevent_add=True,
        item_params={"label_field": "label"},
        primary=True,
    )
    def visualizations(self) -> list["FixationVisualization"]:
        return self._visualizations

    @visualizations.setter
    def visualizations(self, value: list["FixationVisualization"]) -> None:
        recording = self.recording
        self._visualizations = value

        for viz in self._visualizations:
            viz.changed.connect(self.changed.emit)
            if recording is not None:
                viz.on_recording_loaded(recording)

    def bg_optic_flow(self) -> T.Generator[ProgressUpdate, None, None]:
        recording = self.require_recording()
        cache_path = self.require_cache_path()
        flow_array = yield from _OpticFlowTracker(recording).calculate()

        save_file = cache_path / "optic_flow_offsets.npz"
        logger.info(f"Saving optic flow offsets to {save_file}")
        save_file.parent.mkdir(parents=True, exist_ok=True)

        with save_file.open("wb") as file_handle:
            np.savez(file_handle, flow_data=flow_array)

        yield ProgressUpdate(1.0)


class FixationVisualization(PersistentPropertiesMixin, QObject):
    changed = Signal()
    requires_history: bool = False
    _max_radius: float = 80.0

    _known_types: T.ClassVar[list[type["FixationVisualization"]]] = []

    def __init__(self) -> None:
        super().__init__()
        self._use_offset = True
        self._adjust_for_optic_flow = True
        self.recording: NeonRecording | None = None

    def render(
        self,
        painter: QPainter,
        fixations: FixationArray,
        fixation_ids: npt.NDArray[np.intp],
        frame_offsets: dict[int, npt.NDArray[np.float64]],
        gaze_offset: tuple[float, float],
        time_in_recording: int,
    ) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    def __init_subclass__(cls: type["FixationVisualization"], **kwargs: object) -> None:
        FixationVisualization._known_types.append(cls)

    def on_recording_loaded(self, recording: NeonRecording) -> None:
        self.recording = recording

    def to_dict(
        self,
        include_class_name: bool = False,
        condition: T.Callable[[dict[str, T.Any]], bool] | None = None,
        recursive: bool = False,
    ) -> dict[str, T.Any]:
        state = super().to_dict(
            include_class_name=include_class_name,
            condition=condition,
            recursive=recursive,
        )

        # NOTE: Fixation visualizations require the class name to be loaded correctly
        state["__class__"] = self.__class__.__name__
        return state

    @property
    def use_offset(self) -> bool:
        return self._use_offset

    @use_offset.setter
    def use_offset(self, value: bool) -> None:
        self._use_offset = value

    @property
    def adjust_for_optic_flow(self) -> bool:
        return self._adjust_for_optic_flow

    @adjust_for_optic_flow.setter
    def adjust_for_optic_flow(self, value: bool) -> None:
        self._adjust_for_optic_flow = value


class FixationCircleViz(FixationVisualization):
    label = "Circle"

    def __init__(self) -> None:
        super().__init__()
        self._color = QColor(255, 255, 0, 196)
        self._base_radius = 10
        self._stroke_width = 5
        self._font_size = 20

    def render(
        self,
        painter: QPainter,
        fixations: FixationArray,
        fixation_ids: npt.NDArray[np.intp],
        frame_offsets: dict[int, npt.NDArray[np.float64]],
        gaze_offset: tuple[float, float],
        time_in_recording: int,
    ) -> None:
        recording = self.recording
        if recording is None:
            return

        pen = painter.pen()
        pen.setWidth(self._stroke_width)
        painter.setPen(pen)

        font = painter.font()
        font.setPointSize(self._font_size)
        painter.setFont(font)

        offset_mean_gaze_points = apply_offset(
            recording,
            fixations.mean_gaze_point,
            gaze_offset if self._use_offset else (0.0, 0.0),
        )

        width = recording.scene.width
        height = recording.scene.height
        if width is None or height is None:
            return

        for fixation_id, fixation, offset_mean_gaze_point in zip(
            fixation_ids, fixations, offset_mean_gaze_points, strict=False
        ):
            is_active = (fixation.start_time <= time_in_recording) and (
                fixation.stop_time > time_in_recording
            )
            if not is_active:
                continue

            internal_fidx = fixation_id - 1

            if self._adjust_for_optic_flow:
                if internal_fidx not in frame_offsets:
                    continue

                of_x, of_y = frame_offsets[internal_fidx]
                cx = offset_mean_gaze_point[0] - of_x
                cy = offset_mean_gaze_point[1] - of_y
            else:
                cx = offset_mean_gaze_point[0]
                cy = offset_mean_gaze_point[1]

            if not (0 <= cx <= width and 0 <= cy <= height):
                continue

            center = QPointF(cx, cy)

            circle_pen = painter.pen()
            circle_pen.setColor(self._color)
            painter.setPen(circle_pen)

            dur_ms = (fixation.stop_time - fixation.start_time) / 1e6
            radius = min(
                self._max_radius, max(5.0, self._base_radius * (dur_ms / 100.0))
            )

            painter.drawEllipse(center, radius, radius)
            painter.drawText(center, str(fixation_id))

    @property
    def color(self) -> QColor:
        return self._color

    @color.setter
    def color(self, value: QColor) -> None:
        self._color = value

    @property
    @property_params(min=1, max=999)
    def base_radius(self) -> int:
        return self._base_radius

    @base_radius.setter
    def base_radius(self, value: int) -> None:
        self._base_radius = value

    @property
    @property_params(min=1, max=999)
    def stroke_width(self) -> int:
        return self._stroke_width

    @stroke_width.setter
    def stroke_width(self, value: int) -> None:
        self._stroke_width = value

    @property
    @property_params(min=1, max=200)
    def font_size(self) -> int:
        return self._font_size

    @font_size.setter
    def font_size(self, value: int) -> None:
        self._font_size = value


class ScanpathViz(FixationVisualization):
    label = "Scanpath"
    requires_history = True

    def __init__(self) -> None:
        super().__init__()
        self._plot_line = True
        self._current_fixation_color = QColor(18, 99, 204, 191)
        self._circle_color = QColor(18, 99, 204, 191)
        self._line_color = QColor(144, 164, 174, 128)
        self._base_radius = 10
        self._stroke_width = 5
        self._font_size = 20

    def render(
        self,
        painter: QPainter,
        fixations: FixationArray,
        fixation_ids: npt.NDArray[np.intp],
        frame_offsets: dict[int, npt.NDArray[np.float64]],
        gaze_offset: tuple[float, float],
        _time_in_recording: int,
    ) -> None:
        recording = self.recording
        if recording is None:
            return

        pen = painter.pen()
        pen.setWidth(self._stroke_width)
        painter.setPen(pen)

        font = painter.font()
        font.setPointSize(self._font_size)
        painter.setFont(font)

        offset_mean_gaze_points = apply_offset(
            recording,
            fixations.mean_gaze_point,
            gaze_offset if self._use_offset else (0.0, 0.0),
        )

        width = recording.scene.width
        height = recording.scene.height
        if width is None or height is None:
            return

        previous_center = None

        for fixation_id, fixation, offset_mean_gaze_point in zip(
            fixation_ids, fixations, offset_mean_gaze_points, strict=False
        ):
            internal_fidx = fixation_id - 1

            if self._adjust_for_optic_flow:
                if internal_fidx not in frame_offsets:
                    continue

                of_x, of_y = frame_offsets[internal_fidx]
                cx = offset_mean_gaze_point[0] - of_x
                cy = offset_mean_gaze_point[1] - of_y
            else:
                cx = offset_mean_gaze_point[0]
                cy = offset_mean_gaze_point[1]

            center = QPointF(cx, cy)

            if self._plot_line and previous_center is not None:
                line_pen = painter.pen()
                line_pen.setColor(self._line_color)
                painter.setPen(line_pen)
                painter.drawLine(previous_center, center)

            previous_center = center

            if not (0 <= cx <= width and 0 <= cy <= height):
                continue

            circle_pen = painter.pen()
            circle_pen.setColor(self._circle_color)
            painter.setPen(circle_pen)

            dur_ms = (fixation.stop_time - fixation.start_time) / 1e6
            radius = min(
                self._max_radius, max(5.0, self._base_radius * (dur_ms / 100.0))
            )

            painter.drawEllipse(center, radius, radius)
            painter.drawText(center, str(fixation_id))

    @property
    def plot_line(self) -> bool:
        return self._plot_line

    @plot_line.setter
    def plot_line(self, value: bool) -> None:
        self._plot_line = value

    @property
    def circle_color(self) -> QColor:
        return self._circle_color

    @circle_color.setter
    def circle_color(self, value: QColor) -> None:
        self._circle_color = value

    @property
    def line_color(self) -> QColor:
        return self._line_color

    @line_color.setter
    def line_color(self, value: QColor) -> None:
        self._line_color = value

    @property
    @property_params(min=1, max=999)
    def base_radius(self) -> int:
        return self._base_radius

    @base_radius.setter
    def base_radius(self, value: int) -> None:
        self._base_radius = value

    @property
    @property_params(min=1, max=999)
    def stroke_width(self) -> int:
        return self._stroke_width

    @stroke_width.setter
    def stroke_width(self, value: int) -> None:
        self._stroke_width = value

    @property
    @property_params(min=1, max=200)
    def font_size(self) -> int:
        return self._font_size

    @font_size.setter
    def font_size(self, value: int) -> None:
        self._font_size = value


@dataclass
class _FixationFlowPlan:
    """Optic flow data for a single fixation.

    Scene video frames with indices in [`start`, `stop`) are considered when
    calculating the optic flow, with `reference` being the closest scene
    frame to the reference gaze point. `rows` store the output flow data.
    """

    start: int
    reference: int
    stop: int
    reference_point: npt.NDArray[np.float32]
    rows: npt.NDArray[np.float64]

    def has_frames_to_track(self, current: int, direction: int) -> bool:
        if direction < 0:
            return current > self.start

        return current < self.stop - 1

    def set_row_data(
        self,
        current: int,
        frame_index: int,
        offset: npt.NDArray[np.float32 | np.float64],
    ) -> None:
        direction = -1 if current < self.reference else 1

        # NOTE: preserves legacy relative order of scene frames within flow data for
        # one fixation (ref-1, ref-2, ..., start, ref+1, ref+2, ... stop)
        row = (
            self.reference - current - 1 if direction < 0 else current - self.start - 1
        )
        self.rows[row, 1] = frame_index
        self.rows[row, 2:] = offset


class _OpticFlowTracker:
    def __init__(self, recording: NeonRecording, *, chunk_size: int = 32) -> None:
        self._recording = recording
        self._gaze = recording.gaze
        self._scene_timestamps: npt.NDArray[np.int64] | None = None
        self._gray_cache: OrderedDict[int, tuple[int, npt.NDArray[np.uint8]]] = (
            OrderedDict()
        )
        self._chunk_size = chunk_size

    def calculate(
        self,
    ) -> T.Generator[ProgressUpdate, None, npt.NDArray[np.float64]]:
        fixations = self._recording.fixations
        chunks: list[npt.NDArray[np.float64]] = []
        last_progress = -1

        # NOTE: processing fixations in batches, optical flow is calculated for
        # each pair of frames only once per direction within the batch
        for first in range(0, len(fixations), self._chunk_size):
            progress = int(99 * first / len(fixations))
            if progress > last_progress:
                yield ProgressUpdate(progress / 100)
                last_progress = progress

            stop = min(first + self._chunk_size, len(fixations))
            plans = []
            for fidx in range(first, stop):
                fixation = fixations[fidx]
                plan = self._prepare(fidx, fixation.start_time, fixation.stop_time)
                if plan is not None:
                    plans.append(plan)

            total_steps = sum(len(plan.rows) for plan in plans)
            completed_steps = 0
            for direction in (-1, 1):
                for steps in self._track(plans, direction):
                    completed_steps += steps
                    fraction = completed_steps / total_steps
                    progress = int(
                        99 * (first + (stop - first) * fraction) / len(fixations)
                    )
                    if progress > last_progress:
                        yield ProgressUpdate(progress / 100)
                        last_progress = progress

            chunks.extend(plan.rows for plan in plans)

        return np.concatenate(chunks) if chunks else np.zeros((0, 4))

    def _prepare(
        self, fidx: int, start_time: np.int64, stop_time: np.int64
    ) -> _FixationFlowPlan | None:
        gaze_samples = self._gaze[
            (start_time <= self._gaze.time) & (self._gaze.time <= stop_time)
        ]
        if not gaze_samples.size:
            return None

        ref_gaze = gaze_samples[len(gaze_samples) // 2]

        if self._scene_timestamps is None:
            self._scene_timestamps = self._recording.scene.time
        start, stop = np.searchsorted(
            self._scene_timestamps, [start_time, stop_time + 2e9]
        )
        if start == stop:
            return None

        diff = np.abs(self._scene_timestamps[start:stop] - ref_gaze.time)
        if np.min(diff) > 100 * 1e6:
            return None

        reference = int(start + np.argmin(diff))
        rows = np.empty((stop - start - 1, 4), dtype=np.float64)
        rows[:, 0] = fidx

        return _FixationFlowPlan(
            int(start),
            reference,
            int(stop),
            np.asarray(ref_gaze.point, dtype=np.float32),
            rows,
        )

    def _track(self, plans: list[_FixationFlowPlan], direction: int) -> T.Iterator[int]:
        points = np.array([plan.reference_point for plan in plans], dtype=np.float32)
        reference_points = points.copy()

        # Using a priority queue with (current frame index, fixation index) as key
        # to traverse scene frames for all fixations within a batch in one direction
        frontier = [
            (direction * plan.reference, index)
            for index, plan in enumerate(plans)
            if plan.has_frames_to_track(plan.reference, direction)
        ]
        heapify(frontier)

        while frontier:
            previous, index = heappop(frontier)

            # Collect all fixations which require the calculation of optical flow
            # for the currently considered pair of frames
            indices = [index]
            while frontier and frontier[0][0] == previous:
                indices.append(heappop(frontier)[1])

            current = direction * previous + direction
            _, previous_img = self._frame(direction * previous)
            frame_index, current_img = self._frame(current)
            next_points = calc_optic_flow(previous_img, current_img, points[indices])
            del previous_img, current_img

            offsets = reference_points[indices] - next_points
            points[indices] = next_points
            for index, offset in zip(indices, offsets, strict=True):
                plan = plans[index]
                plan.set_row_data(current, frame_index, offset)

                if plan.has_frames_to_track(current, direction):
                    heappush(frontier, (direction * current, index))

            yield len(indices)

    def _frame(self, index: int) -> tuple[int, npt.NDArray[np.uint8]]:
        if index in self._gray_cache:
            self._gray_cache.move_to_end(index)
            return self._gray_cache[index]

        if len(self._gray_cache) == 2:
            self._gray_cache.popitem(last=False)

        frame = self._recording.scene[index]
        value = (frame.index, frame.gray)
        self._gray_cache[index] = value

        return value


def calc_optic_flow(
    previous_frame: npt.NDArray[np.uint8],
    current_frame: npt.NDArray[np.uint8],
    points: npt.NDArray[np.float32 | np.float64],
    lk_winSize: tuple[int, int] | None = (90, 90),
    lk_maxLevel: int = 3,
    lk_criteria: tuple[int, int, float] = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        20,
        0.03,
    ),
    lk_min_eig_threshold: float = 0.005,
) -> npt.NDArray[np.float32 | np.float64]:

    corrected_points, _status, _err = cv2.calcOpticalFlowPyrLK(
        previous_frame,
        current_frame,
        points.reshape([-1, 1, 2]),
        None,
        winSize=lk_winSize,
        maxLevel=lk_maxLevel,
        criteria=lk_criteria,
        minEigThreshold=lk_min_eig_threshold,
    )
    return np.asarray(corrected_points).reshape(points.shape)


class OpticFlow(T.NamedTuple):
    ts: npt.NDArray[np.int64]
    x: npt.NDArray[np.float32]
    y: npt.NDArray[np.float32]
