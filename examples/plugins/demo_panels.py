"""Copy into the custom-plugin directory and enable Panels Demo for a recording.

The static widget never subscribes to playback. The position widget subscribes
only after registration supplies playback. In the image panel, hover over or
click the bright yellow-and-green "CLICK HERE" box near the center to highlight
it or increment its counter.
Discovery uses synchronous GUI-thread connect-then-snapshot; observers must not
mutate panels during notifications.
"""

import logging
from contextlib import ExitStack

from pupil_labs.neon_recording import NeonRecording
from PySide6.QtCore import QPointF, QRect, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColorConstants, QImage, QPainter
from PySide6.QtWidgets import QLabel

from pupil_labs.neon_player import (
    InteractivePanelOverlay2D,
    Panel,
    PanelOverlay2D,
    PanelWidget2D,
    Plugin,
    action,
)
from pupil_labs.neon_player.managers.playback import PlaybackPosition
from pupil_labs.neon_player.ui.video_render_widget import ScalingWidget


class StaticImage(ScalingWidget):
    def __init__(self) -> None:
        super().__init__()
        self.preparation_count = 0
        self.image = self._prepare_image()
        self.fit_rect(self.image.size())

    def _prepare_image(self) -> QImage:
        self.preparation_count += 1
        image = QImage(QSize(320, 160), QImage.Format.Format_RGB32)
        image.fill(QColorConstants.DarkBlue)
        painter = QPainter(image)
        try:
            painter.fillRect(40, 40, 240, 80, QColorConstants.DarkCyan)
        finally:
            painter.end()
        return image

    def _paint_content(self, painter: QPainter) -> None:
        painter.drawImage(0, 0, self.image)


class PlaybackCursor(InteractivePanelOverlay2D):
    def __init__(self, playback: PlaybackPosition) -> None:
        super().__init__(
            playback,
            "Playback cursor and click counter",
            hitbox=QRect(95, 55, 130, 50),
        )
        self.timestamp = playback.time
        self.click_count = 0
        self._subscribed = True
        playback.changed.connect(self.show_position)

    def on_click(self, _point: QPointF) -> None:
        self.click_count += 1
        self.changed.emit()

    def show_position(self, timestamp: int | None) -> None:
        self.timestamp = timestamp
        self.changed.emit()

    def paint(self, painter: QPainter) -> None:
        painter.setPen(QColorConstants.Yellow)
        if self.timestamp is None:
            painter.drawText(8, 24, "No recording")
        else:
            # Sweep once per recording-axis second, without reducing ns precision.
            x = int((self.timestamp % 1_000_000_000) * 319 // 1_000_000_000)
            painter.drawLine(x, 0, x, 159)
            painter.drawText(8, 24, f"{self.timestamp} ns")
        painter.setPen(QColorConstants.Yellow)
        painter.setBrush(QColorConstants.DarkGreen)
        hitbox = self.hitbox
        if hitbox is None:
            raise RuntimeError("Playback cursor hitbox is required for painting")
        painter.drawRect(QRectF(hitbox))
        painter.drawText(
            QRectF(hitbox),
            Qt.AlignmentFlag.AlignCenter,
            f"CLICK HERE\nClicks: {self.click_count}",
        )

    def on_removed(self) -> None:
        if self._subscribed:
            self.playback.changed.disconnect(self.show_position)
            self._subscribed = False


class PlaybackLabel(QLabel):
    def bind_playback(self, playback: PlaybackPosition) -> None:
        playback.changed.connect(self.show_position)
        self.show_position(playback.time)

    def show_position(self, timestamp: int | None) -> None:
        self.setText(
            "No recording" if timestamp is None else f"Recording time: {timestamp} ns"
        )


class PanelsDemo(Plugin):
    label = "Panels Demo"

    def __init__(self) -> None:
        super().__init__()
        self._owned_panels: dict[str, Panel] = {}
        self._known_panels: dict[str, Panel] = {}
        self._observing = False

    def on_recording_loaded(self, recording: NeonRecording) -> None:
        if not self._observing:
            self.panel_manager.panel_added.connect(self._on_panel_added)
            self.panel_manager.panel_removed.connect(self._on_panel_removed)
            self._observing = True
            for panel in self.panel_manager.panels:
                self._on_panel_added(panel)
        self.open_panels()

    @action
    def open_panels(self) -> None:
        """Create missing demo panels; closed panels reopen as fresh instances."""
        if "demo.static" not in self._owned_panels:
            self._add_panel(
                Panel(
                    "demo.static",
                    "Static analysis",
                    QLabel("Static content: playback does not recompute this widget."),
                )
            )
        if "demo.playback" not in self._owned_panels:
            widget = PlaybackLabel()
            panel = Panel("demo.playback", "Playback position", widget)
            self._add_panel(panel)
            widget.bind_playback(panel.playback)
        if "demo.overlay" not in self._owned_panels:
            host = StaticImage()
            panel = Panel("demo.overlay", "Static image with playback overlay", host)
            with ExitStack() as rollback:
                rollback.callback(panel.deleteLater)
                rollback.callback(host.deleteLater)
                if isinstance(panel.widget, PanelWidget2D):
                    overlay = PlaybackCursor(self.panel_manager.playback)
                    rollback.callback(overlay.deleteLater)
                    panel.widget.add_overlay(overlay, layer=10)
                self.add_panel(panel)
                rollback.pop_all()
            self._owned_panels[panel.panel_id] = panel

    def _add_panel(self, panel: Panel) -> None:
        with ExitStack() as rollback:
            rollback.callback(panel.deleteLater)
            rollback.callback(panel.widget.deleteLater)
            self.add_panel(panel)
            rollback.pop_all()
        self._owned_panels[panel.panel_id] = panel

    def _on_panel_added(self, panel: Panel) -> None:
        self._known_panels[panel.panel_id] = panel

    def _on_panel_removed(self, panel: Panel) -> None:
        self._known_panels.pop(panel.panel_id, None)
        if self._owned_panels.get(panel.panel_id) is panel:
            del self._owned_panels[panel.panel_id]

    def on_disabled(self) -> None:
        # Optional explicit removal while discovery is still connected.
        for panel in tuple(self._owned_panels.values()):
            self.remove_panel(panel)
        if self._observing:
            self.panel_manager.panel_added.disconnect(self._on_panel_added)
            self.panel_manager.panel_removed.disconnect(self._on_panel_removed)
            self._observing = False
        self._known_panels.clear()


class SharedTimeLabel(PanelOverlay2D):
    def __init__(self, playback: PlaybackPosition) -> None:
        super().__init__(playback, "Shared playback time")
        self.timestamp = playback.time
        self._subscribed = True
        playback.changed.connect(self.show_position)

    def show_position(self, timestamp: int | None) -> None:
        self.timestamp = timestamp
        self.changed.emit()

    def paint(self, painter: QPainter) -> None:
        painter.setPen(QColorConstants.White)
        text = "No recording" if self.timestamp is None else f"Shared: {self.timestamp} ns"
        painter.drawText(8, 150, text)

    def on_removed(self) -> None:
        if self._subscribed:
            self.playback.changed.disconnect(self.show_position)
            self._subscribed = False


class PanelOverlaysDemo(Plugin):
    """Enable before or after Panels Demo; disable independently."""

    label = "Panel Overlays Demo"

    def __init__(self) -> None:
        super().__init__()
        self._observing = False
        self._pending: dict[str, Panel] = {}
        self._attached: dict[Panel, SharedTimeLabel] = {}
        self._attach_timer = QTimer(self)
        self._attach_timer.setSingleShot(True)
        self._attach_timer.timeout.connect(self._attach_pending)

    def on_recording_loaded(self, recording: NeonRecording) -> None:
        if self._observing:
            return
        self._observing = True
        self.panel_manager.panel_added.connect(self._queue_panel)
        self.panel_manager.panel_removed.connect(self._panel_removed)
        for panel in self.panel_manager.panels:
            self._queue_panel(panel)

    def _queue_panel(self, panel: Panel) -> None:
        if self._observing and panel.panel_id == "demo.overlay":
            self._pending[panel.panel_id] = panel
            self._attach_timer.start(0)

    def _panel_removed(self, panel: Panel) -> None:
        if self._pending.get(panel.panel_id) is panel:
            self._pending.pop(panel.panel_id)
        self._attached.pop(panel, None)

    def _attach_pending(self) -> None:
        pending, self._pending = self._pending, {}
        if not self._observing:
            return
        for panel_id, panel in pending.items():
            if self.panel_manager.get_panel(panel_id) is not panel:
                continue
            if panel in self._attached or not isinstance(panel.widget, PanelWidget2D):
                continue
            overlay = SharedTimeLabel(self.panel_manager.playback)
            try:
                self.add_overlay(panel, overlay, layer=20)
            except Exception:
                overlay.on_removed()
                overlay.deleteLater()
                logging.getLogger(__name__).exception("Failed to attach demo overlay")
            else:
                self._attached[panel] = overlay

    def on_disabled(self) -> None:
        if not self._observing:
            return
        self._observing = False
        self._attach_timer.stop()
        self._pending.clear()
        self.panel_manager.panel_added.disconnect(self._queue_panel)
        self.panel_manager.panel_removed.disconnect(self._panel_removed)
        self._attached.clear()
