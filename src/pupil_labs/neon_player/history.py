from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from pupil_labs.neon_recording import NeonRecording
from PySide6.QtCore import QObject, Signal


def get_recording_metadata(path: Path, recording: NeonRecording) -> dict[str, str]:
    """Short description that is displayed for recently opened recordings."""
    start_time = datetime.fromtimestamp(recording.info["start_time"] / 1e9)
    recorded_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    last_opened_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    wearer = recording.wearer["name"]

    return {
        "name": path.name,
        "wearer": wearer,
        "recorded": recorded_str,
        "last_opened": last_opened_str,
    }


class RecordingHistory(QObject):
    changed = Signal()

    def __init__(self, capacity: int = 10) -> None:
        super().__init__()
        self._recordings : OrderedDict[str, dict[str, str]] = OrderedDict()
        self.capacity : int = capacity

    def _cleanup(self) -> None:
        """Remove non-existing recordings and enforce capacity limit."""
        self._recordings = OrderedDict({
            path: meta for path, meta in self._recordings.items() if Path(path).exists()
        })

        while len(self._recordings) > self.capacity:
            self._recordings.popitem()

    def add_recording(self, path: Path, recording: NeonRecording):
        metadata = get_recording_metadata(path, recording)
        key = str(path.resolve())

        # NOTE: always update metadata since the last opened timestamp changes if
        # even the key already existed
        self._recordings[key] = metadata
        self._recordings.move_to_end(key, last=False)
        self._cleanup()

        self.changed.emit()

    @property
    def recordings(self):
        return self._recordings

    @classmethod
    def from_dict(cls, recordings):
        instance = cls()
        instance._recordings = OrderedDict(recordings)
        instance._cleanup()
        instance.changed.emit()
        return instance
