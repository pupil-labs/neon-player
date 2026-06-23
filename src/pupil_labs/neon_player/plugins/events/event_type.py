from PySide6.QtCore import QObject, Signal
from qt_property_widgets.utilities import PersistentPropertiesMixin, property_params


IMMUTABLE_EVENTS = ["recording.begin", "recording.end"]


class EventType(PersistentPropertiesMixin, QObject):
    changed = Signal()
    name_changed = Signal(str, str)

    def __init__(self) -> None:
        super().__init__()
        self._name = ""
        self._shortcut = ""
        self._uid = ""
        self._source = "recording"

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if self._name == value:
            return

        old_name = self._name
        self._name = value
        self.name_changed.emit(old_name, value)

    @property
    @property_params(max_length=1)
    def shortcut(self) -> str:
        return self._shortcut

    @shortcut.setter
    def shortcut(self, value: str) -> None:
        self._shortcut = value

    @property
    @property_params(widget=None, dont_encode=True)
    def source(self) -> str:
        return self._source

    @source.setter
    def source(self, value: str) -> None:
        if value not in ["recording", "workspace"]:
            raise ValueError(f"Invalid event type source: {value}")
        self._source = value

    @property
    @property_params(widget=None)
    def uid(self) -> str:
        """
        This field should always have the same value as `name`. It is kept for compatibility
        with the original implementation of the plugin that used UIDs as the primary identifier
        for event types.
        """
        return self._uid

    @uid.setter
    def uid(self, value: str) -> None:
        self._uid = value

    @staticmethod
    def from_name(name: str) -> "EventType":
        et = EventType()
        et._name = name
        et._uid = name
        return et
