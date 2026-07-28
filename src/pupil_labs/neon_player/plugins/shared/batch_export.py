import typing as T

from pathlib import Path

from pupil_labs import neon_player
from pupil_labs.neon_recording import NeonRecording


def get_batch_export_destination_gen(
    destination: Path
) -> T.Callable[[NeonRecording], list[Path]]:
    """
    Collect the export for each recording in a subfolder with the same name
    as the recording directory.
    """
    def destination_generator(rec: NeonRecording) -> list[Path]:
        save_path = destination / rec._rec_dir.name
        save_path.mkdir(exist_ok=True)
        return [save_path]

    return destination_generator


def run_export_across_recordings(
    plugin: neon_player.Plugin,
    destination: Path,
    name: str = "Export all recordings",
    action_name: str = "export"
) -> None:
    if plugin.workspace is None:
        return

    if not plugin.app.headless:
        plugin.job_manager.run_background_batch_action(
            name=name,
            action_name=f"{plugin.__class__.__name__}.{action_name}",
            args_generator=get_batch_export_destination_gen(destination),
        )
