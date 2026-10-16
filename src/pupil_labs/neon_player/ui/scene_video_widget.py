import logging

from pupil_labs.neon_recording import NeonRecording
from PySide6.QtCore import QSize
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QWidget

from pupil_labs import neon_player
from pupil_labs.neon_player.managers.playback import PlaybackPosition
from pupil_labs.neon_player.ui.video_render_widget import VideoRenderWidget


class SceneVideoWidget(VideoRenderWidget):
    """Main scene visualization using the application's legacy scene renderer."""

    def __init__(
        self, playback: PlaybackPosition, parent: QWidget | None = None
    ) -> None:
        super().__init__(playback, parent)
        self._scene_available = False

    def on_recording_loaded(self, recording: NeonRecording) -> None:
        self._scene_available = False
        try:
            scene = recording.scene
            width, height = scene.width, scene.height
        except NeonRecording.SensorError:
            logging.getLogger(__name__).warning(
                "Scene video unavailable; displaying a gray frame", exc_info=True
            )
            self.update()
            return
        if width is None or height is None or width <= 0 or height <= 0:
            logging.getLogger(__name__).warning(
                "Scene video dimensions unavailable; displaying a gray frame"
            )
            self.update()
            return
        self._scene_available = True
        self.source_size = QSize(width, height)
        self.adjust_size()
        self.refresh()

    def on_recording_unloaded(self) -> None:
        """Release recording-specific children while retaining the scene widget."""
        self._scene_available = False
        self.scaled_children_positions.clear()
        for child in self.children():
            if isinstance(child, QWidget) and child is not self.fps_label:
                child.hide()
                child.deleteLater()
        self.update()

    def _paint_frame(self, painter: QPainter, timestamp: int) -> None:
        if not self._scene_available:
            self._paint_unavailable_frame(painter)
            return
        neon_player.instance().render_to(painter, timestamp)
