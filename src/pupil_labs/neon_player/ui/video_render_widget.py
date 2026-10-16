import time

from PySide6.QtCore import QPoint, QPointF, QPropertyAnimation, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColorConstants,
    QMouseEvent,
    QPainter,
    QResizeEvent,
)
from PySide6.QtWidgets import (
    QGraphicsOpacityEffect,
    QLabel,
    QWidget,
)

from pupil_labs import neon_player
from pupil_labs.neon_player.managers.playback import PlaybackPosition
from pupil_labs.neon_player.ui.panel_widget_2d import PanelWidget2D


class ScalingWidget(PanelWidget2D):
    mouse_pressed = Signal(QMouseEvent)
    mouse_released = Signal(QMouseEvent)
    mouse_clicked = Signal(QMouseEvent)
    mouse_moved = Signal(QMouseEvent)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setMouseTracking(True)

        self.source_size = QSize(100, 100)
        self.scaled_children_positions: dict[QWidget, tuple[float, float, bool]] = {}
        self._mouse_down = False

        self.offset = QPointF(0, 0)
        self.scale = 1.0

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._mouse_down = True

        self.mouse_pressed.emit(event)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._mouse_down:
            self.mouse_clicked.emit(event)
            self._mouse_down = False

        self.mouse_released.emit(event)
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self.mouse_moved.emit(event)
        super().mouseMoveEvent(event)

    def map_point(self, point: QPoint | QPointF) -> QPointF:
        if self.width() <= 0 or self.height() <= 0 or self.scale <= 0:
            raise RuntimeError("Cannot map points without a usable visualization size")
        point = QPointF(point) - self.offset
        return QPointF(point.x() / self.scale, point.y() / self.scale)

    def _has_valid_mapping(self) -> bool:
        return super()._has_valid_mapping() and self.scale > 0

    def transform_painter(self, painter: QPainter) -> None:
        painter.translate(self.offset)
        painter.scale(self.scale, self.scale)

    def content_rect(self) -> QRectF:
        return QRectF(0, 0, self.source_size.width(), self.source_size.height())

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self.adjust_size()
        for child in self.children():
            if isinstance(child, QWidget):
                self.update_child_position(child)

    def adjust_size(self) -> None:
        self.fit_rect()
        self.update()

    def fit_rect(self, source_size: QSize | None = None) -> None:
        size = self.source_size if source_size is None else source_size
        if size.width() <= 0 or size.height() <= 0:
            raise ValueError("Visualization source dimensions must be positive")
        if source_size is not None:
            self.source_size = QSize(source_size)

        if self.width() <= 0 or self.height() <= 0:
            self.scale = 0.0
            self.offset = QPointF()
            return

        source_aspect = self.source_size.width() / self.source_size.height()
        target_aspect = self.width() / self.height()

        if source_aspect > target_aspect:
            self.scale = self.width() / self.source_size.width()
            self.offset = QPointF(
                0, int((self.height() - self.source_size.height() * self.scale) / 2.0)
            )

        else:
            self.scale = self.height() / self.source_size.height()
            self.offset = QPointF(
                int((self.width() - self.source_size.width() * self.scale) / 2.0), 0
            )

    def set_child_scaled_pos(self, child: QWidget, x: float, y: float) -> None:
        self.scaled_children_positions[child] = (x, y, False)
        self.update_child_position(child)

    def set_child_scaled_center(self, child: QWidget, x: float, y: float) -> None:
        self.scaled_children_positions[child] = (x, y, True)
        self.update_child_position(child)

    def update_child_position(self, child: QWidget) -> None:
        if child not in self.scaled_children_positions:
            return

        x, y, centered = self.scaled_children_positions[child]

        x *= self.scale
        y *= self.scale

        if centered:
            x -= child.width() / 2
            y -= child.height() / 2

        child.move(QPoint(int(x + self.offset.x()), int(y + self.offset.y())))


class VideoRenderWidget(ScalingWidget):
    """Playback-driven visualization with preparation and content-painting hooks."""

    def __init__(
        self, playback: PlaybackPosition, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._last_frame_time: float | None = None
        self._fps = 0.0
        self.fps_label = QLabel(self)
        self.fps_label.resize(1024, 24)
        self.opacity_effect = QGraphicsOpacityEffect()
        self.fps_label.setGraphicsEffect(self.opacity_effect)
        self.opacity_effect.setOpacity(0.0)
        self.fade_anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_anim.setDuration(1000)
        self.fade_anim.setStartValue(1.0)
        self.fade_anim.setEndValue(0.0)
        self.fade_anim.finished.connect(self.on_fade_finished)
        self.playback = playback
        self.playback.changed.connect(self._on_position_changed)

    def _on_position_changed(self, ts: int) -> None:
        self._prepare_frame(ts)
        self.update()

    def refresh(self) -> None:
        """Prepare current playback state before scheduling a coalescible paint."""
        self._prepare_frame(self.playback.time)
        self.update()

    def _prepare_frame(self, timestamp: int | None) -> None:
        """Prepare visualization-specific state without painting synchronously."""

    def on_fade_finished(self) -> None:
        self._last_frame_time = None

    def adjust_size(self) -> None:
        if neon_player.instance().recording is not None:
            super().adjust_size()

    def _paint_background(self, painter: QPainter) -> None:
        super()._paint_background(painter)
        if neon_player.instance().settings.show_fps:
            now = time.monotonic()
            if self._last_frame_time is not None:
                delta = now - self._last_frame_time
                self.fade_anim.stop()
                instant_fps = 1.0 / delta
                self._fps = self._fps * 0.98 + instant_fps * 0.02
                self.fps_label.setText(f"{self._fps:.2f} fps")
                self.opacity_effect.setOpacity(1.0)
                self.fade_anim.start()
            self._last_frame_time = now

    def _paint_content(self, painter: QPainter) -> None:
        # Legacy video content used a fresh painter without these render hints.
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        timestamp = self.playback.time
        if timestamp is None:
            self._paint_unavailable_frame(painter)
        else:
            self._paint_frame(painter, timestamp)

    def _paint_frame(self, painter: QPainter, timestamp: int) -> None:
        """Paint in visualization coordinates using the borrowed painter."""
        raise NotImplementedError("Video widgets must implement frame painting")

    def _paint_unavailable_frame(self, painter: QPainter) -> None:
        painter.save()
        try:
            painter.resetTransform()
            painter.fillRect(self.rect(), QColorConstants.Gray)
        finally:
            painter.restore()
