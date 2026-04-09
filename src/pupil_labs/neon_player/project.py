from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from pupil_labs import neon_recording as nr


@dataclass
class RecordingDescription:
    name: str
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
    folders = [p for p in path.iterdir() if p.is_dir()]
    for folder in folders:
        if desc := get_recording_description(folder):
            recordings.append(desc)

    return recordings
