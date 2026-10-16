from PySide6.QtCore import QObject, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPalette
from PySide6.QtWidgets import QFrame, QVBoxLayout, QWidget
from qt_property_widgets.utilities import property_params
from shiboken6 import isValid

from pupil_labs.neon_player.managers.playback import PlaybackPosition


class PanelOverlayTooltip:
    """Host-managed tooltip content backed by a parentless QWidget.

    Attaching an overlay transfers the content widget to its host, which deletes
    it when the overlay is removed.
    """

    def __init__(self, widget: QWidget) -> None:
        if not isinstance(widget, QWidget):
            raise TypeError("Overlay tooltip content must be a QWidget")
        if widget.parent() is not None:
            raise ValueError("Overlay tooltip content must be parentless")
        self._widget = widget
        self._display_widget: QFrame | None = None
        self._host: QWidget | None = None

    @property
    def widget(self) -> QWidget:
        return self._widget

    def _require_attachable(self) -> None:
        if self._display_widget is not None:
            raise ValueError("Overlay tooltip is already attached")
        if not isValid(self._widget):
            raise ValueError("Overlay tooltip content has been destroyed")
        if self._widget.parent() is not None:
            raise ValueError("Overlay tooltip content must be parentless")

    def _attach(self, host: QWidget) -> None:
        self._require_attachable()
        display_widget = QFrame(host)
        display_widget.setFrameShape(QFrame.Shape.StyledPanel)
        display_widget.setFrameShadow(QFrame.Shadow.Raised)
        display_widget.setPalette(QPalette(host.palette()))
        display_widget.setAutoFillBackground(True)
        display_widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout = QVBoxLayout(display_widget)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.addWidget(self._widget)
        self._display_widget = display_widget
        self._host = host
        display_widget.hide()

    def _show_at(self, point: QPointF) -> None:
        host = self._host
        display_widget = self._display_widget
        if host is None or display_widget is None or not isValid(display_widget):
            return
        self._widget.show()
        display_widget.adjustSize()
        size = display_widget.sizeHint()
        if size.width() > 0 and size.height() > 0:
            display_widget.resize(size)
        offset = 12
        x = int(point.x()) + offset
        y = int(point.y()) + offset
        if x + display_widget.width() > host.width():
            x = int(point.x()) - display_widget.width() - offset
        if y + display_widget.height() > host.height():
            y = int(point.y()) - display_widget.height() - offset
        x = max(0, min(x, max(0, host.width() - display_widget.width())))
        y = max(0, min(y, max(0, host.height() - display_widget.height())))
        display_widget.move(x, y)
        display_widget.show()
        display_widget.raise_()

    def _hide(self) -> None:
        if self._display_widget is not None and isValid(self._display_widget):
            self._display_widget.hide()
        if isValid(self._widget):
            self._widget.hide()

    def _dispose(self) -> None:
        self._hide()
        if self._display_widget is not None and isValid(self._display_widget):
            self._display_widget.deleteLater()
        self._display_widget = None
        self._host = None


class PanelOverlay(QObject):
    """Named generic overlay state with borrowed playback and invalidation.

    Subscriptions are optional. Use QObject-bound receivers so destruction
    disconnects them, and release subscriptions in on_removed for immediate
    disconnection before deferred deletion. Hook failures are logged and do not
    prevent disposal.
    """

    changed = Signal()

    def __init__(
        self,
        playback: PlaybackPosition,
        display_name: str,
        *,
        tooltip: PanelOverlayTooltip | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(display_name, str):
            raise TypeError("Overlay display name must be a string")
        display_name = display_name.strip()
        if not display_name:
            raise ValueError("Overlay display name cannot be empty")
        if tooltip is not None and not isinstance(tooltip, PanelOverlayTooltip):
            raise TypeError("Overlay tooltip must be a PanelOverlayTooltip or None")
        self._playback = playback
        self._display_name = display_name
        self._tooltip = tooltip
        self._attached_once = False

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def playback(self) -> PlaybackPosition:
        return self._playback

    @property
    @property_params(widget=None, dont_encode=True)
    def tooltip(self) -> PanelOverlayTooltip | None:
        return self._tooltip

    def on_removed(self) -> None:
        """Release subscriptions before removal; failures are logged by the host."""


class PanelOverlay2D(PanelOverlay):
    def paint(self, painter: QPainter) -> None:
        """Paint using the host's borrowed painter in visualization coordinates."""
        raise NotImplementedError("2D overlays must implement painting")


class InteractivePanelOverlay2D(PanelOverlay2D):
    """A 2D overlay with hit-tested click and hover interaction."""

    def __init__(
        self,
        playback: PlaybackPosition,
        display_name: str,
        *,
        hitbox: QRect | None = None,
        tooltip: PanelOverlayTooltip | None = None,
    ) -> None:
        super().__init__(playback, display_name, tooltip=tooltip)
        self._hitbox: QRect | None = None
        self.hitbox = hitbox

    @property
    def hitbox(self) -> QRect | None:
        return None if self._hitbox is None else QRect(self._hitbox)

    @hitbox.setter
    def hitbox(self, value: QRect | None) -> None:
        if value is not None and not isinstance(value, QRect):
            raise TypeError("Overlay hitbox must be a QRect or None")
        self._hitbox = None if value is None else QRect(value)

    def hit_test(self, point: QPointF) -> bool:
        """Hit-test in visualization coordinates using the configured hitbox."""
        return self._hitbox is not None and QRectF(self._hitbox).contains(point)

    def on_click(self, point: QPointF) -> None:
        """Handle a completed left click in visualization coordinates."""
        raise NotImplementedError("Interactive overlays must implement click handling")

    def on_hover_enter(self, point: QPointF) -> None:
        """Handle entry into the topmost hit region at a visualization point."""

    def on_hover_leave(self, point: QPointF) -> None:
        """Handle leaving the hit region, with the last hovered visualization point."""

    def on_hover_move(self, point: QPointF) -> None:
        """Handle pointer movement within the hit region."""

    def render_hover_highlight(self, painter: QPainter) -> None:
        """Render the default highlight; subclasses can customize its appearance."""
        hitbox = self.hitbox
        if hitbox is None:
            return
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(128, 128, 128, 40))
        painter.drawRoundedRect(QRectF(hitbox).adjusted(-5, -5, 5, 5), 5, 5)
