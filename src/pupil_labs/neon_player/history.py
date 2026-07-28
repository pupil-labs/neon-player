import copy
import logging
import typing as T

from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from pupil_labs.neon_recording import NeonRecording
from PySide6.QtCore import QObject, Signal
from qt_property_widgets.utilities import PersistentPropertiesMixin


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


def create_workspace_metadata(path: Path) -> dict[str, str]:
    """Short description that is displayed for recently opened workspaces."""
    last_opened_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result = {"name": path.name, "last_opened": last_opened_str}
    return result


def _ensure_correct_format(history: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    """
    Ensure that the history dictionary has the correct format. Originally, it contained only
    recent recordings, so the format is extended to include recent workspaces as well.
    """
    if "recent_recordings" not in history:
        history = {"recent_recordings": history, "recent_workspaces": {}}

    return history


class LoadHistory(PersistentPropertiesMixin, QObject):
    changed = Signal()

    def __init__(self, capacity: int = 10) -> None:
        super().__init__()
        self._recent_recordings : OrderedDict[str, dict[str, str]] = OrderedDict()
        self._recent_workspaces : OrderedDict[str, dict[str, str]] = OrderedDict()
        self.capacity : int = capacity

    def _cleanup_recordings(self) -> None:
        self._recent_recordings = self._cleanup_dict(self._recent_recordings)

    def _cleanup_workspaces(self) -> None:
        self._recent_workspaces = self._cleanup_dict(self._recent_workspaces)

    def _cleanup_dict(
        self, recent_dict: OrderedDict[str, dict[str, str]]
    ) -> OrderedDict[str, dict[str, str]]:
        """
        Remove entries from the recent_dict that no longer exist on disk and
        ensure that the number of entries does not exceed the capacity.
        """
        recent_dict = OrderedDict({
            path: meta
            for path, meta in recent_dict.items()
            if Path(path).exists()
        })

        while len(recent_dict) > self.capacity:
            recent_dict.popitem()

        return recent_dict

    def add_recording(self, path: Path, recording: NeonRecording):
        metadata = create_recording_metadata(path, recording)
        self._add_entry("recording", path, metadata)

    def add_workspace(self, path: Path):
        metadata = create_workspace_metadata(path)
        self._add_entry("workspace", path, metadata)

    def _add_entry(self, kind: str, path: Path, metadata: dict[str, str]) -> None:
        # NOTE: always update metadata since the last opened timestamp changes if
        # even the key already existed
        is_recording_entry = kind == "recording"
        dest = self._recent_recordings if is_recording_entry else self._recent_workspaces
        key = str(path.resolve())

        dest[key] = metadata
        dest.move_to_end(key, last=False)
        if is_recording_entry:
            self._cleanup_recordings()
        else:
            self._cleanup_workspaces()

        self.changed.emit()

    @property
    def recent_recordings(self):
        return copy.deepcopy(self._recent_recordings)

    @recent_recordings.setter
    def recent_recordings(self, value: OrderedDict[str, dict[str, str]]):
        self._recent_recordings = value
        self._cleanup_recordings()
        self.changed.emit()

    @property
    def recent_workspaces(self):
        return copy.deepcopy(self._recent_workspaces)

    @recent_workspaces.setter
    def recent_workspaces(self, value: OrderedDict[str, dict[str, str]]):
        self._recent_workspaces = value
        self._cleanup_workspaces()
        self.changed.emit()

    @classmethod
    def from_dict(cls: type["LoadHistory"], state: dict[str, T.Any]) -> "LoadHistory":
        state = _ensure_correct_format(state)

        instance = cls()
        instance.__setstate__(state)
        return instance
