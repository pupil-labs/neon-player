import logging

from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from pupil_labs.neon_recording import NeonRecording
from PySide6.QtCore import QObject, Signal


def create_recording_metadata(path: Path, recording: NeonRecording) -> dict[str, str]:
    """Short description that is displayed for recently opened recordings."""
    last_opened_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result = {"name": path.name, "last_opened": last_opened_str}

    try:
        start_time = datetime.fromtimestamp(recording.info["start_time"] / 1e9)
        result["recorded"] = start_time.strftime("%Y-%m-%d %H:%M:%S")
        result["wearer"] = recording.wearer["name"]
    except FileNotFoundError:
        logging.warning(f"Could not extract metadata from the recording")
    except Exception as e:
        logging.exception(f"Error occurred while extracting recording metadata: {str(e)}")

    return result


class RecordingHistory(QObject):
    changed = Signal()

    def __init__(self, capacity: int = 10) -> None:
        super().__init__()
        self._recent_recordings : OrderedDict[str, dict[str, str]] = OrderedDict()
        self.capacity : int = capacity

    def _cleanup(self) -> None:
        """Remove non-existing recent_recordings and enforce capacity limit."""
        self._recent_recordings = OrderedDict({
            path: meta
            for path, meta in self._recent_recordings.items()
            if Path(path).exists()
        })

        while len(self._recent_recordings) > self.capacity:
            self._recent_recordings.popitem()

    def add_recording(self, path: Path, recording: NeonRecording):
        metadata = create_recording_metadata(path, recording)
        key = str(path.resolve())

        # NOTE: always update metadata since the last opened timestamp changes if
        # even the key already existed
        self._recent_recordings[key] = metadata
        self._recent_recordings.move_to_end(key, last=False)
        self._cleanup()

        self.changed.emit()

    @property
    def recent_recordings(self):
        return self._recent_recordings

    @classmethod
    def from_dict(cls, recent_recordings):
        instance = cls()
        instance._recent_recordings = OrderedDict(recent_recordings)
        instance._cleanup()
        instance.changed.emit()
        return instance
