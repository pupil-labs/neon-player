from PySide6.QtCore import QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import QColorConstants, QPainter, QPaintEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget
from shiboken6 import isValid

from pupil_labs.neon_player.managers.panel import _require_gui_thread
from pupil_labs.neon_player.overlays import PanelOverlay2D


class PanelWidget2D(QOpenGLWidget):
    """Optional OpenGL host; the default visualization uses widget coordinates.

    Override _paint_content, transform_painter and content_rect to specialize
    rendering. Overlay layers are fixed at attachment; ties use insertion order.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        _require_gui_thread()
        super().__init__(parent)
        surface_format = self.format()
        surface_format.setSamples(10)
        self.setFormat(surface_format)
        self._overlays: dict[PanelOverlay2D, int] = {}
        self._painting = False
        self._mutating = False

    @property
    def overlays(self) -> tuple[PanelOverlay2D, ...]:
        _require_gui_thread()
        return tuple(sorted(self._overlays, key=self._overlays.__getitem__))

    def _require_mutation(self) -> None:
        _require_gui_thread()
        if self._painting or self._mutating:
            raise RuntimeError("Cannot mutate overlays during painting or removal")

    def add_overlay(self, overlay: PanelOverlay2D, *, layer: int = 0) -> None:
        self._require_mutation()
        if not isinstance(overlay, PanelOverlay2D):
            raise TypeError("Host requires a PanelOverlay2D")
        if not isinstance(layer, int):
            raise TypeError("Overlay layer must be an integer")
        if not isValid(overlay):
            raise ValueError("Overlay has been destroyed")
        if overlay.thread() != self.thread():
            raise ValueError("Overlay must live on the GUI thread")
        if overlay.parent() is not None or overlay._attached_once:
            raise ValueError("Overlay is already owned or was previously attached")
        overlay.changed.connect(self._overlay_changed)
        overlay.setParent(self)
        overlay._attached_once = True
        self._overlays[overlay] = layer
        self.update()

    def remove_overlay(self, overlay: PanelOverlay2D) -> None:
        self._require_mutation()
        if overlay not in self._overlays or not isValid(overlay):
            raise ValueError("Overlay is not attached to this host")
        self._mutating = True
        try:
            overlay.on_removed()
        finally:
            self._mutating = False
        overlay.changed.disconnect(self._overlay_changed)
        del self._overlays[overlay]
        overlay.deleteLater()
        self.update()

    def _overlay_changed(self) -> None:
        self.update()

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
