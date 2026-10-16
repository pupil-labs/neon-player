from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QPainter

from pupil_labs.neon_player.managers.playback import PlaybackPosition


class PanelOverlay(QObject):
    """Generic overlay state with borrowed playback and opt-in invalidation.

    Subscriptions are optional. Use QObject-bound receivers so destruction
    disconnects them, and release subscriptions in on_removed for immediate
    disconnection before deferred deletion. Hook failures are logged and do not
    prevent disposal.
    """

    changed = Signal()

    def __init__(self, playback: PlaybackPosition) -> None:
        super().__init__()
        self._playback = playback
        self._attached_once = False

    @property
    def playback(self) -> PlaybackPosition:
        return self._playback

    def on_removed(self) -> None:
        """Release subscriptions before removal; failures are logged by the host."""


class PanelOverlay2D(PanelOverlay):
    def paint(self, painter: QPainter) -> None:
        """Paint using the host's borrowed painter in visualization coordinates."""
        raise NotImplementedError("2D overlays must implement painting")
