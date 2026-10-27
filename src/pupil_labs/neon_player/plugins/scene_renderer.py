import typing as T
from pathlib import Path

import cv2
from PySide6.QtGui import QColorConstants, QIcon, QPainter
from qt_property_widgets.utilities import action, action_params, property_params

from pupil_labs import neon_player
from pupil_labs.neon_player import Plugin, asset_path
from pupil_labs.neon_player.job_manager import BackgroundJob, ProgressUpdate
from pupil_labs.neon_player.plugins.shared import bg_export_video
from pupil_labs.neon_player.utilities import qimage_from_frame


class SceneRendererPlugin(Plugin):
    label = "Scene Renderer"

    DEFAULT_BRIGHTNESS = 0
    DEFAULT_CONTRAST = 1.0

    def __init__(self) -> None:
        super().__init__()
        self.render_layer = 0
        self.gray = QColorConstants.Gray

        self._show_frame_index = False
        self._brightness = self.DEFAULT_BRIGHTNESS
        self._contrast = self.DEFAULT_CONTRAST

    def render(self, painter: QPainter, time_in_recording: int) -> None:
        recording = self.recording
        assert recording is not None, (
            "Recording must be loaded before rendering the scene"
        )

        width = recording.scene.width
        height = recording.scene.height
        assert width is not None, "Scene width must be available before rendering"
        assert height is not None, "Scene height must be available before rendering"

        if self.is_time_gray(time_in_recording):
            painter.fillRect(
                0,
                0,
                width,
                height,
                self.gray,
            )
            return

        scene_frame = recording.scene.sample([time_in_recording], method="backward")[0]
        frame_img = cv2.convertScaleAbs(
            scene_frame.bgr, alpha=self._contrast, beta=self._brightness
        )

        painter.drawImage(0, 0, qimage_from_frame(frame_img))

        if self.show_frame_index:
            font = painter.font()
            font.setPointSize(font.pointSize() * 2)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(0, scene_frame.height, str(scene_frame.idx))

    @property
    def show_frame_index(self) -> bool:
        return self._show_frame_index

    @show_frame_index.setter
    def show_frame_index(self, value: bool) -> None:
        self._show_frame_index = value
        self.changed.emit()

    @property
    @property_params(min=0, max=100)
    def brightness(self) -> int:
        return self._brightness

    @brightness.setter
    def brightness(self, value: int) -> None:
        self._brightness = value

    @property
    @property_params(min=0.0, max=3.0)
    def contrast(self) -> float:
        return self._contrast

    @contrast.setter
    def contrast(self, value: float) -> None:
        self._contrast = value

    @action
    @action_params(compact=True, icon=QIcon(str(asset_path("reset_settings.svg"))))
    def reset_settings(self) -> None:
        self.show_frame_index = False
        self.brightness = self.DEFAULT_BRIGHTNESS
        self.contrast = self.DEFAULT_CONTRAST

    @action
    @action_params(compact=True, icon=QIcon(str(asset_path("export.svg"))))
    def export_raw_scene_video(
        self, destination: Path = Path()
    ) -> BackgroundJob | T.Generator[ProgressUpdate, None, None] | None:
        app = neon_player.instance()

        if not app.headless:
            return self.job_manager.run_background_action(
                "Scene Video Export", "SceneRendererPlugin.bg_export", destination
            )

        return self.bg_export(destination)

    def render_for_export(self, painter: QPainter, time_in_recording: int) -> None:
        self.render(painter, time_in_recording)

    def bg_export(
        self, destination: Path = Path()
    ) -> T.Generator[ProgressUpdate, None, None]:
        recording = self.recording
        assert recording is not None, (
            "Recording must be loaded before exporting scene video"
        )
        export_window = self.app.get_export_window()
        assert export_window is not None, (
            "Export window must be available before exporting scene video"
        )
        yield from bg_export_video(
            recording=recording,
            export_window=export_window,
            render_fn=self.render_for_export,
            destination=destination,
            output_video_filename="scene.mp4",
            output_timestamps_filename="scene_timestamps.csv",
        )
