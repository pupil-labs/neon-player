import typing as T

from pathlib import Path
from PySide6.QtGui import QIcon
from qt_property_widgets.utilities import action_params

from pupil_labs import neon_player
from pupil_labs.neon_player import action
from pupil_labs.neon_recording import NeonRecording


def get_batch_export_destination_gen(destination: Path) -> Path:
    def destination_generator(rec: NeonRecording) -> Path:
        save_path = destination / rec._rec_dir.name
        save_path.mkdir(exist_ok=True)
        return [save_path]
    return destination_generator
