from PySide6.QtCore import QObject, QPointF, QRect, QRectF, Signal
from PySide6.QtGui import QPainter

from pupil_labs.neon_player.managers.playback import PlaybackPosition


class PanelOverlay(QObject):
    """Named generic overlay state with borrowed playback and invalidation.

    Subscriptions are optional. Use QObject-bound receivers so destruction
    disconnects them, and release subscriptions in on_removed for immediate
    disconnection before deferred deletion. Hook failures are logged and do not
    prevent disposal.
    """

    changed = Signal()

    def __init__(self, playback: PlaybackPosition, display_name: str) -> None:
        super().__init__()
        if not isinstance(display_name, str):
            raise TypeError("Overlay display name must be a string")
        display_name = display_name.strip()
        if not display_name:
            raise ValueError("Overlay display name cannot be empty")
        self._playback = playback
        self._display_name = display_name
        self._attached_once = False

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def playback(self) -> PlaybackPosition:
        return self._playback

    def on_removed(self) -> None:
        """Release subscriptions before removal; failures are logged by the host."""


class PanelOverlay2D(PanelOverlay):
    def paint(self, painter: QPainter) -> None:
        """Paint using the host's borrowed painter in visualization coordinates."""
        raise NotImplementedError("2D overlays must implement painting")


class ClickablePanelOverlay2D(PanelOverlay2D):
    """A 2D overlay with optional QRect-based click targeting."""

    def __init__(
        self,
        playback: PlaybackPosition,
        display_name: str,
        *,
        hitbox: QRect | None = None,
    ) -> None:
        super().__init__(playback, display_name)
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
        raise NotImplementedError("Clickable overlays must implement click handling")
