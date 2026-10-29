import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pupil_labs.neon_recording import NeonRecording
from PySide6.QtGui import QIcon
from qt_property_widgets.utilities import action_params

from pupil_labs import neon_player
from pupil_labs.neon_player import action

logger = logging.getLogger(__name__)


class BlinksPlugin(neon_player.Plugin):
    label = "Blinks"

    def on_recording_loaded(self, recording: NeonRecording) -> None:
        if len(recording.blinks) == 0:
            return

        # TODO(type-checking): Neon Recording mistypes column selection.
        self.get_timeline().add_timeline_broken_bar(
            "Blinks",
            recording.blinks[["start_time", "stop_time"]],  # type: ignore[arg-type]
        )

    def on_disabled(self) -> None:
        self.get_timeline().remove_timeline_plot("Blinks")

    @action
    @action_params(compact=True, icon=QIcon(str(neon_player.asset_path("export.svg"))))
    def export(self, destination: Path = Path()) -> None:
        recording = self.require_recording()
        blink_ids = 1 + np.arange(len(recording.blinks))
        blinks = recording.blinks

        start_time, stop_time = self.app.get_export_window()
        start_mask = blinks.stop_time > start_time
        stop_mask = blinks.start_time < stop_time

        # TODO(type-checking): Neon Recording mistypes masked array selection.
        blinks = blinks[start_mask & stop_mask]  # type: ignore[assignment]
        blink_ids = blink_ids[start_mask & stop_mask]

        export_data = pd.DataFrame(
            {
                "recording id": recording.info["recording_id"],
                "blink id": blink_ids,
                "start timestamp [ns]": blinks.start_time,
                "end timestamp [ns]": blinks.stop_time,
                "duration [ms]": (blinks.stop_time - blinks.start_time) / 1e6,
            }
        )

        export_file = destination / "blinks.csv"
        export_data.to_csv(export_file, index=False)
        logger.info(f"Exported blinks to {export_file}")
