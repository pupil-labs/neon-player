import logging

from dataclasses import dataclass, fields
from datetime import timedelta
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from pupil_labs import neon_recording as nr


@dataclass
class RecordingDescription:
    name: str
    path: Path
    duration: timedelta
    wearer: str


def get_recording_description(path: Path) -> RecordingDescription | None:
    """
    Extracts recording name, duration, and wearer name to provide a brief
    description of the recording.
    """
    try:
        rec = nr.load(path)
        return RecordingDescription(
            name=path.name,
            path=path,
            duration=timedelta(seconds=rec.duration / 1e9),
            wearer=rec.wearer["name"]
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


class Project(QObject):
    recording_list_loaded = Signal(object)

    def __init__(self):
        super().__init__()

        self.recording_list : list[RecordingDescription] = []
        self.initialized : bool = False

    def load_recording_list(self, path: Path):
        logging.info(f"Loading recording list from: {path}")
        self.initialized = False
        self.recording_list = get_recording_list(path)

        logging.info(f"Found {len(self.recording_list)} recordings in the provided folder")
        self.initialized = True
        self.recording_list_loaded.emit(self.recording_list)
