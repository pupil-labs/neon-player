import typing as T
from pathlib import Path

from PySide6.QtCore import QSize
from PySide6.QtGui import QColorConstants, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QFileDialog
from qt_property_widgets.utilities import action_params

from pupil_labs import neon_player
from pupil_labs.neon_player import action
from pupil_labs.neon_player.job_manager import BackgroundJob, ProgressUpdate
from pupil_labs.neon_player.plugins.shared import bg_export_video


class VideoExporter(neon_player.Plugin):
    label = "Video Exporter"

    def __init__(self) -> None:
        super().__init__()
        self.render_layer = 0
        self.gray = QColorConstants.Gray
        self.is_exporting = False

    @action
    @action_params(compact=True, icon=QIcon(str(neon_player.asset_path("export.svg"))))
    def export(
        self, destination: Path = Path()
    ) -> BackgroundJob | T.Generator[ProgressUpdate, None, None] | None:
        app = neon_player.instance()

        if not app.headless:
            return self.job_manager.run_background_action(
                "Video Export", "VideoExporter.export", destination
            )

        return self.bg_export(destination)

    def render_for_export(self, painter: QPainter, time_in_recording: int) -> None:
        self.app.render_to(painter, time_in_recording)

    def bg_export(
        self, destination: Path = Path()
    ) -> T.Generator[ProgressUpdate, None, None]:
        recording = self.require_recording()
        export_window = self.app.get_export_window()
        assert export_window is not None, (
            "Export window must be available before exporting video"
        )
        yield from bg_export_video(
            recording=recording,
            export_window=export_window,
            render_fn=self.render_for_export,
            destination=destination,
            output_video_filename="world.mp4",
            output_timestamps_filename="world_timestamps.csv",
        )

    @action
    @action_params(compact=True, icon=QIcon(str(neon_player.asset_path("export.svg"))))
    def export_current_frame(self) -> None:
        recording = self.require_recording()
        file_path_str, _type_selection = QFileDialog.getSaveFileName(
            None, "Export frame", "", "PNG Images (*.png)"
        )
        if not file_path_str:
            return

        if not file_path_str.endswith(".png"):
            file_path_str += ".png"

        width, height = recording.scene.width, recording.scene.height
        if width is None or height is None:
            raise ValueError("Scene video dimensions are unavailable")
        frame_size = QSize(width, height)
        frame = QImage(frame_size, QImage.Format.Format_RGB32)
        painter = QPainter(frame)

        self.app.render_to(painter)
        painter.end()
        frame.save(str(file_path_str))

    @action
    @action_params(
        compact=True,
        icon=QIcon(str(neon_player.asset_path("duplicate.svg"))),
    )
    def copy_frame_to_clipboard(self) -> None:
        recording = self.require_recording()
        width, height = recording.scene.width, recording.scene.height
        if width is None or height is None:
            raise ValueError("Scene video dimensions are unavailable")
        frame_size = QSize(width, height)
        frame = QImage(frame_size, QImage.Format.Format_RGB32)
        painter = QPainter(frame)

        self.app.render_to(painter)
        painter.end()

        clipboard = neon_player.instance().clipboard()
        clipboard.setPixmap(QPixmap.fromImage(frame))
