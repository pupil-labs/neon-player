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


def run_export_across_recordings(
    plugin: neon_player.Plugin, 
    destination: Path, 
    action_name: str = "export"
) -> None:
    if plugin.workspace is None:
        return

    if not plugin.app.headless:
        plugin.job_manager.run_background_batch_action(
            f"Export all recordings",
            f"{plugin.__class__.__name__}.{action_name}",
            get_batch_export_destination_gen(destination),
        )
