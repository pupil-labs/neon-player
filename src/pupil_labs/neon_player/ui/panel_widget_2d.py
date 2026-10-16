from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColorConstants, QMouseEvent, QPainter, QPaintEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget
from pupil_labs.neon_player.overlays import (
    InteractivePanelOverlay2D,
    PanelOverlay,
    PanelOverlay2D,
)
from pupil_labs.neon_player.ui.panel_widget import PanelWidget, _require_gui_thread


class PanelWidget2D(PanelWidget, QOpenGLWidget):
    """Optional OpenGL host; the default visualization uses widget coordinates.

    Override _paint_content, transform_painter and content_rect to specialize
    rendering. Overlay layers are fixed at attachment; ties use insertion order.
    """

    overlay_removed = Signal(object)
    overlay_clicked = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        _require_gui_thread()
        super().__init__(parent)
        surface_format = self.format()
        surface_format.setSamples(10)
        self.setFormat(surface_format)
        self._initialize_overlays()
        self._pressed_interactive_overlay: InteractivePanelOverlay2D | None = None
        self._hovered_interactive_overlay: InteractivePanelOverlay2D | None = None
        self._hovered_point: QPointF | None = None
        self.setMouseTracking(True)

    def _notify_overlay_removed(self, overlay: PanelOverlay) -> None:
        if overlay is self._hovered_interactive_overlay:
            self._hovered_interactive_overlay = None
            self._hovered_point = None
        if overlay is self._pressed_interactive_overlay:
            self._pressed_interactive_overlay = None
        self.overlay_removed.emit(overlay)

    def map_point(self, point: QPoint | QPointF) -> QPointF:
        return QPointF(point)

    def transform_painter(self, painter: QPainter) -> None:
        """Apply visualization geometry; the unscaled host uses identity."""

    def content_rect(self) -> QRectF:
        return QRectF(self.rect())

    def _has_valid_mapping(self) -> bool:
        content = self.content_rect()
        return (
            self.width() > 0
            and self.height() > 0
            and content.width() > 0
            and content.height() > 0
        )

    def _paint_content(self, painter: QPainter) -> None:
        """Paint content before overlays, without preparing data here."""

    def _paint_background(self, painter: QPainter) -> None:
        painter.fillRect(self.rect(), QColorConstants.Black)

    def _interactive_overlay_at(
        self, widget_point: QPointF
    ) -> tuple[InteractivePanelOverlay2D, QPointF] | None:
        point = self._point_in_content(widget_point)
        if point is None:
            return None
        for overlay in reversed(self.overlays):
            if isinstance(overlay, InteractivePanelOverlay2D) and overlay.hit_test(point):
                return overlay, point
        return None

    def _point_in_content(self, widget_point: QPointF) -> QPointF | None:
        if not self._has_valid_mapping():
            return None
        try:
            point = self.map_point(widget_point)
        except RuntimeError:
            return None
        return point if self.content_rect().contains(point) else None

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            target = self._interactive_overlay_at(event.position())
            self._pressed_interactive_overlay = (
                None if target is None else target[0]
            )
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pressed_overlay = self._pressed_interactive_overlay
            self._pressed_interactive_overlay = None
            target = self._interactive_overlay_at(event.position())
            if target is not None and target[0] is pressed_overlay:
                self.overlay_clicked.emit(target[0])
                target[0].on_click(target[1])
        super().mouseReleaseEvent(event)

    def _update_hover(
        self, target: tuple[InteractivePanelOverlay2D, QPointF] | None
    ) -> None:
        if target is None:
            overlay = None
            point = None
        else:
            overlay, point = target
        if overlay is self._hovered_interactive_overlay:
            self._hovered_point = point
            return
        previous_overlay = self._hovered_interactive_overlay
        if previous_overlay is not None:
            previous_point = self._hovered_point
            if previous_point is not None:
                previous_overlay.on_hover_leave(previous_point)
        self._hovered_interactive_overlay = overlay
        self._hovered_point = point
        if overlay is not None and point is not None:
            overlay.on_hover_enter(point)
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self._update_hover(self._interactive_overlay_at(event.position()))
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        self._update_hover(None)
        super().leaveEvent(event)

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
            hovered = self._hovered_interactive_overlay
            if hovered is not None:
                painter.save()
                try:
                    self.transform_painter(painter)
                    painter.setClipRect(
                        self.content_rect(), Qt.ClipOperation.IntersectClip
                    )
                    hovered.render_hover_highlight(painter)
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
