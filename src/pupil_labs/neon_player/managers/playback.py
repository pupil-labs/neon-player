from PySide6.QtCore import QObject, Signal


class PlaybackPosition(QObject):
    """Read-only recording timestamp shared by playback consumers.

    Time is integer nanoseconds on the recording axis, or None without a
    recording. Reading it does not subscribe or schedule work. Each seek and
    playback tick notifies subscribers, even if the timestamp repeats.
    Unload resets time silently after recording cleanup.
    """

    changed = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._time: int | None = None

    @property
    def time(self) -> int | None:
        return self._time

    def _publish(self, timestamp: int) -> None:
        self._time = timestamp
        self.changed.emit(timestamp)

    def _reset(self) -> None:
        self._time = None
