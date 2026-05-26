import typing as T

from pathlib import Path
from PySide6.QtGui import QIcon
from qt_property_widgets.utilities import action_params

from pupil_labs import neon_player
from pupil_labs.neon_player import action
from pupil_labs.neon_recording import NeonRecording


class BackgroundBatchExportMixin:
    export_fn: str = "export"

    @action
    @action_params(compact=True, icon=QIcon(str(neon_player.asset_path("export.svg"))))
    def export_all_recordings(self, destination: Path = Path(".")):
        if self.workspace is None:
            return

        def destination_generator(rec: NeonRecording) -> Path:
            return [destination / rec._rec_dir.name]

        self.job_manager.run_background_batch_action(
            f"Export all recordings",
            f"{self.__class__.__name__}.{self.export_fn}",
            destination_generator,
        )
