"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging

from video_export.plugin_base.isolated_frame_exporter import IsolatedFrameExporter

logger = logging.getLogger(__name__)


class Eye_Video_Exporter(IsolatedFrameExporter):
    """
    Exports eye videos in the selected time range together with their timestamps.
    Optionally (via a switch button), pupil detections are rendered on the video.
    """

    icon_chr = "EV"

    def __init__(self, g_pool):
        super().__init__(g_pool, max_concurrent_tasks=2)  # export 2 eyes at once
        self.logger = logger
        self.logger.info("Eye Video Exporter has been launched.")

    def customize_menu(self):
        self.menu.label = "Eye Video Exporter"
        super().customize_menu()

    def export_data(self, export_range, export_dir):
        process_frame = _no_change

        eye_name = "eye0"
        try:
            self.add_export_job(
                export_range,
                export_dir,
                input_name=eye_name,
                output_name=eye_name,
                process_frame=process_frame,
                timestamp_export_format="all",
            )
        except FileNotFoundError:
            # happens if there is no such eye video
            pass


def _no_change(_, frame):
    """
    Processing function for IsolatedFrameExporter.
    Just leaves all frames unchanged.
    """
    return frame.img

