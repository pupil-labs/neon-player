from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColorConstants, QPainter, QPaintEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget
from pupil_labs.neon_player.overlays import PanelOverlay, PanelOverlay2D
from pupil_labs.neon_player.ui.panel_widget import PanelWidget, _require_gui_thread


class PanelWidget2D(PanelWidget, QOpenGLWidget):
    """Optional OpenGL host; the default visualization uses widget coordinates.

    Override _paint_content, transform_painter and content_rect to specialize
    rendering. Overlay layers are fixed at attachment; ties use insertion order.
    """

    overlay_removed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        _require_gui_thread()
        super().__init__(parent)
        surface_format = self.format()
        surface_format.setSamples(10)
        self.setFormat(surface_format)
        self._initialize_overlays()

    def _notify_overlay_removed(self, overlay: PanelOverlay) -> None:
        self.overlay_removed.emit(overlay)

    def map_point(self, point: QPoint | QPointF) -> QPointF:
        return QPointF(point)

    def transform_painter(self, painter: QPainter) -> None:
        """Apply visualization geometry; the unscaled host uses identity."""

    def content_rect(self) -> QRectF:
        return QRectF(self.rect())

    def _paint_content(self, painter: QPainter) -> None:
        """Paint content before overlays, without preparing data here."""

    def _paint_background(self, painter: QPainter) -> None:
        painter.fillRect(self.rect(), QColorConstants.Black)

    def paint(self, painter: QPainter) -> None:
        """Render into a borrowed painter, preserving its state.

        This image-rendering seam never prepares content or changes video export.
        """
        _require_gui_thread()
        if not painter.isActive():
            raise ValueError("Panel painting requires an active painter")
        if self._painting:
            raise RuntimeError("Recursive panel painting is not supported")
        self._painting = True
        painter.save()
        try:
            self._paint_background(painter)
            painter.save()
            try:
                self.transform_painter(painter)
                self._paint_content(painter)
            finally:
                painter.restore()
            for overlay in self.overlays:
                if not isinstance(overlay, PanelOverlay2D):
                    raise TypeError("2D painting requires a PanelOverlay2D")
                painter.save()
                try:
                    self.transform_painter(painter)
                    painter.setClipRect(
                        self.content_rect(), Qt.ClipOperation.IntersectClip
                    )
                    overlay.paint(painter)
                finally:
                    painter.restore()
        finally:
            painter.restore()
            self._painting = False

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        try:
            painter.setRenderHints(
                painter.RenderHint.Antialiasing
                | painter.RenderHint.SmoothPixmapTransform
            )
            self.paint(painter)
        finally:
            painter.end()
