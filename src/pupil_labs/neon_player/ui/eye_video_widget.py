from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QImage, QPainter, QPaintEvent
from PySide6.QtWidgets import QWidget

from pupil_labs.neon_player.utilities import qimage_from_frame
from pupil_labs.neon_recording import NeonRecording


class EyeVideoWidget(QWidget):
    """Aspect-fit eye frames with an explicit empty state for recording gaps."""

    def __init__(self) -> None:
        super().__init__()
        self._image = QImage()

    def set_frame(self, recording: NeonRecording | None, ts: int) -> None:
        self._image = QImage()
        if recording is not None and recording.eye is not None:
            frames = recording.eye.sample([ts])
            if len(frames) and frames[0] is not None:
                frame = frames[0]
                if abs(ts - frame.time) / 1e9 <= 1 / 30:
                    self._image = qimage_from_frame(frame.gray).copy()
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        if self._image.isNull():
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No eye frame")
        else:
            size = self._image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
            target = QRectF(
                (self.width() - size.width()) / 2,
                (self.height() - size.height()) / 2,
                size.width(), size.height(),
            )
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawImage(target, self._image)
        painter.end()
