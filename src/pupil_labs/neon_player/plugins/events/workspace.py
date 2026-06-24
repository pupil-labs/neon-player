import typing as T

from pupil_labs.neon_player.plugins.events.event_type import IMMUTABLE_EVENTS


class WorkspaceEventIndex():
    """
    Maintains an index of all events and the number of their occurrences
    across all recordings in the workspace to support renaming and deletion.
    """
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

    def to_dict(self) -> dict[str, T.Any]:
        data = {
            "events": self.events,
            "recording_names": list(self.recording_names)
        }
        return data

    def update(self, recording_name: str, recording_events: dict[str, list[int]]) -> None:
        """
        Batch index update that goes through all events in the index and all
        recording events.
        """
        # First, check if existing index entries for this recording are up-to-date
        for event_name, recording_counts in self.events.items():
            event_in_recording = event_name in recording_events
            recording_in_index = recording_name in recording_counts

            if recording_in_index and not event_in_recording:
                del recording_counts[recording_name]

            if event_in_recording:
                timestamps = recording_events[event_name]
                self.events[event_name][recording_name] = len(timestamps)

        # Process other events from the recording that are not yet in the index
        for event_name, timestamps in recording_events.items():
            if event_name in IMMUTABLE_EVENTS:
                continue

            if event_name in self.events:
                continue

            self.events[event_name] = {}
            self.events[event_name][recording_name] = len(timestamps)

        # Mark recording as processed in the index
        if recording_name not in self.recording_names:
            self.recording_names.add(recording_name)

    def add_event(self, event_name: str, recording_name: str) -> None:
        if event_name not in self.events:
            self.events[event_name] = {}

        if recording_name not in self.events[event_name]:
            self.events[event_name][recording_name] = 0

    def delete_event(self, event_name: str, recording_name: str) -> None:
        if event_name not in self.events or recording_name not in self.events[event_name]:
            return

        self.events[event_name][recording_name] -= 1

        if self.events[event_name][recording_name] <= 0:
            del self.events[event_name][recording_name]
        if not self.events[event_name]:
            del self.events[event_name]

    def rename_event(self, old_name: str, new_name: str, recording_name: str) -> None:
        if old_name not in self.events:
            return

        if new_name not in self.events:
            self.events[new_name] = {}

        self.events[new_name][recording_name] = self.events[old_name].pop(recording_name)
        if not self.events[old_name]:
            del self.events[old_name]

    def delete_event_type(self, event_name: str) -> None:
        if event_name in self.events:
            del self.events[event_name]
