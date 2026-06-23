import typing as T

from PySide6.QtCore import QObject, Signal


class WorkspaceEventIndex(QObject):
    """
    Maintains an index of all events and the number of their occurrences
    across all recordings in the workspace to ensure correct renaming and deletion.
    """
    changed = Signal()

    def __init__(self) -> None:
        # {event_name: {recording_name: event_count}}
        self.events: dict[str, dict[str, int]] = {}

        # Names of recording that the index is based on
        self.recording_names: set[str] = set()

    def load(self, data: dict[str, T.Any]) -> None:
        if data is None:
            return

        self.events = data.get("events", {})
        self.recording_names = set(data.get("recording_names", []))

    def save(self) -> dict[str, T.Any]:
        data = {
            "events": self.events,
            "recording_names": list(self.recording_names)
        }
        return data

    def update(self) -> None:
        recording_name = self.plugin.recording._rec_dir.name
        if recording_name not in self.recording_names:
            self.recording_names.add(recording_name)

        events_to_process = self.plugin.events
        for event_name, timestamps in events_to_process.items():
            # if event_name in IMMUTABLE_EVENTS:
            #     continue

            if event_name not in self.events:
                self.events[event_name] = {}

            # Clean up the entries if:
            #  - all events of this type were deleted from the current recording
            #  - this event type was deleted across all recordings
            if not timestamps and recording_name in self.events[event_name]:
                del self.events[event_name][recording_name]
                if not self.events[event_name]:
                    del self.events[event_name]
                continue

            if timestamps:
                self.events[event_name][recording_name] = len(timestamps)

    def add_event(self, event_name: str, recording_name: str) -> None:
        if event_name not in self.events:
            self.events[event_name] = {}

        if recording_name not in self.events[event_name]:
            self.events[event_name][recording_name] = 0

        self.changed.emit()

    def delete_event(self, event_name: str, recording_name: str) -> None:
        if event_name not in self.events or recording_name not in self.events[event_name]:
            return

        self.events[event_name][recording_name] -= 1

        if self.events[event_name][recording_name] <= 0:
            del self.events[event_name][recording_name]
        if not self.events[event_name]:
            del self.events[event_name]
        self.changed.emit()

    def rename_event(self, old_name: str, new_name: str, recording_name: str) -> None:
        if old_name not in self.events:
            return

        if new_name not in self.events:
            self.events[new_name] = {}

        self.events[new_name][recording_name] = self.events[old_name].pop(recording_name)
        if not self.events[old_name]:
            del self.events[old_name]

        self.changed.emit()

    def delete_event_type(self, event_name: str) -> None:
        if event_name in self.events:
            del self.events[event_name]
            self.changed.emit()
