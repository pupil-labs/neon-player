import logging

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from pupil_labs import neon_recording as nr


@dataclass
class RecordingDescription:
    name: str
    path: Path
    recorded: datetime
    duration: timedelta
    wearer: str


def get_recording_description(path: Path) -> RecordingDescription | None:
    """
    Extracts recording name, duration, and wearer name to provide a brief
    description of the recording.
    """
    try:
        rec = nr.load(path)
        duration = timedelta(seconds=rec.duration / 1e9)
        recorded = datetime.fromtimestamp(rec.start_time / 1e9)
        return RecordingDescription(
            name=path.name,
            path=path,
            duration=duration,
            wearer=rec.wearer["name"],
            recorded=recorded
        )
    except FileNotFoundError:  # path / info.json / wearer.json
        return None


def get_recording_list(path: Path) -> list[RecordingDescription]:
    """
    Get a list of recordings present in a folder.
    Only a subset of fields is extracted to provide a brief description.
    """
    recordings = []
    folders = sorted([p for p in path.iterdir() if p.is_dir()])
    for folder in folders:
        if desc := get_recording_description(folder):
            recordings.append(desc)

    return recordings


class Workspace(QObject):
    recording_list_loaded = Signal(object)

    def __init__(self):
        super().__init__()

        self.recording_dict : dict[str, RecordingDescription] = {}
        self.initialized : bool = False

    @property
    def recordings(self) -> list[RecordingDescription]:
        return list(self.recording_dict.values())

    def get_recording_path(self, recording_name: str) -> Path | None:
        """
        Get the file path of a recording by its name.
        """
        if recording_name not in self.recording_dict:
            return None

        return self.recording_dict[recording_name].path

    def update_recording_list(self, path: Path):
        logging.info(f"Scanning for recordings in: {path}")
        self.initialized = False
        recordings = get_recording_list(path)
        self.recording_dict = {rec.name: rec for rec in recordings}

        logging.info(f"Found {len(self.recording_dict)} recordings in the provided folder")
        self.initialized = True
        self.recording_list_loaded.emit(self.recordings)
