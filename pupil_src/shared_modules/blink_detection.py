"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import csv
import logging
import os
from collections import deque
from pathlib import Path

import pandas as pd

import file_methods as fm
import gl_utils
import numpy as np
import OpenGL.GL as gl
import player_methods as pm
import pyglui.cygl.utils as cygl_utils
from plugin import Plugin
from pyglui import ui
from pyglui.pyfontstash import fontstash as fs

from pupil_recording.info import recording_info_utils

logger = logging.getLogger(__name__)

blink_color = cygl_utils.RGBA(0.9961, 0.3789, 0.5313, 0.8)


class Blink_Detection(Plugin):
    """
    This plugin loads blinks from the recording
    """

    order = 0.8
    icon_chr = chr(0xE81A)
    icon_font = "pupil_icons"

    def __init__(self, g_pool, hide_gaze_during_blinks=True):
        super().__init__(g_pool)
        self.menu = None

        g_pool.blinks = pm.Affiliator()
        g_pool.hide_gaze_during_blinks = hide_gaze_during_blinks

        self.status = ""
        self.cache = {"class_points": ()}

        self.bg_task = None
        self.load_blinks()

    def init_ui(self):
        super().init_ui()
        self.add_menu()
        self.menu.label = "Blink Detection"

        self.menu.append(ui.Switch(
            "hide_gaze_during_blinks", self.g_pool, label="Hide gaze during blinks", setter=self.set_hide_gaze_during_blinks
        ))

        self.glfont = fs.Context()
        self.glfont.add_font("opensans", ui.get_opensans_font_path())
        self.glfont.set_font("opensans")
        self.timeline = ui.Timeline(
            "Blink Detection", self.draw_activation, self.draw_legend
        )
        self.g_pool.user_timelines.append(self.timeline)

    def deinit_ui(self):
        super().deinit_ui()
        self.remove_menu()
        self.g_pool.user_timelines.remove(self.timeline)
        self.timeline = None

    def set_hide_gaze_during_blinks(self, value):
        self.g_pool.hide_gaze_during_blinks = value

        if len(self.g_pool.blinks) > 0:
            self.notify_all({"subject": "blinks_changed", "delay": 0.2})

    def on_notify(self, notification):
        if notification["subject"] == "blinks_changed":
            self.cache_activation()
            try:
                self.timeline.refresh()
            except AttributeError:
                pass
        if notification["subject"] == "should_export":
            self.export(notification["ts_window"], notification["export_dir"])

    def export(self, export_window, export_dir):
        """
        Between in and out mark

            blink_detection_report.csv:
                - history length
                - onset threshold
                - offset threshold

            blinks.csv:
                blink id | start timestamp [ns] | end timestamp [ns] | duration [ms]
        """
        if not self.g_pool.blinks:
            logger.warning(
                "No blinks were detected in this recording. Nothing to export."
            )
            return

        header = (
            "blink id",
            "start timestamp [ns]",
            "end timestamp [ns]",
            "duration [ms]",
        )

        blinks_in_section = self.g_pool.blinks.by_ts_window(export_window)

        with open(
            os.path.join(export_dir, "blinks.csv"), "w", encoding="utf-8", newline=""
        ) as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=header)
            csv_writer.writeheader()
            for b in blinks_in_section:
                csv_writer.writerow({
                    "blink id": b["id"],
                    "start timestamp [ns]": f'{b["start timestamp [ns]"]:0.0f}',
                    "end timestamp [ns]": f'{b["end timestamp [ns]"]:0.0f}',
                    "duration [ms]": (b["end timestamp [ns]"] - b["start timestamp [ns]"]) * 1e-6,
                })
            logger.info("Created 'blinks.csv' file.")

    def load_blinks(self):
        rec_path = Path(self.g_pool.rec_dir)
        neon_rec_path = rec_path.parent
        data_path = rec_path / "offline_data"
        cache_file = data_path / "blinks.csv"

        if not cache_file.exists():
            data = pd.DataFrame({
                'start timestamp [ns]': self.g_pool.recording_api.blinks['start_timestamp_ns'],
                'end timestamp [ns]': self.g_pool.recording_api.blinks['end_timestamp_ns'],
            })
            data['duration [ms]'] = (data['end timestamp [ns]'] - data['start timestamp [ns]']) * 1e-6
            data['blink id'] = data.index + 1
            data.to_csv(cache_file, index=False)

        blink_data = deque()
        blink_start_ts = deque()
        blink_stop_ts = deque()

        info_json = recording_info_utils.read_neon_info_file(str(neon_rec_path))
        start_time_synced_ns = int(info_json["start_time"])

        with cache_file.open() as csvfile:
            reader = csv.DictReader(csvfile)
            for blink_meta in reader:
                blink = {
                    "id": blink_meta["blink id"],
                    "start_timestamp": (float(blink_meta["start timestamp [ns]"]) - start_time_synced_ns) * 1e-9,
                    "end_timestamp": (float(blink_meta["end timestamp [ns]"]) - start_time_synced_ns) * 1e-9,
                    "start timestamp [ns]": float(blink_meta["start timestamp [ns]"]),
                    "end timestamp [ns]": float(blink_meta["end timestamp [ns]"]),
                }
                blink["duration"] = blink["end_timestamp"] - blink["start_timestamp"]

                idx_start, idx_end = np.searchsorted(
                    self.g_pool.timestamps, [blink["start_timestamp"], blink["end_timestamp"]]
                )
                idx_end = min(idx_end, len(self.g_pool.timestamps) - 1)
                blink["start_frame_index"] = int(idx_start)
                blink["end_frame_index"] = int(idx_end)
                blink["index"] = int((idx_start + idx_end) // 2)

                blink_data.append(fm.Serialized_Dict(python_dict=blink))
                blink_start_ts.append(blink["start_timestamp"])
                blink_stop_ts.append(blink["end_timestamp"])

        self.g_pool.blinks = pm.Affiliator(blink_data, blink_start_ts, blink_stop_ts)
        self.notify_all({"subject": "blinks_changed", "delay": 0.2})
        self.status = f"{len(self.g_pool.blinks)} blinks detected"

    def recent_events(self, events):
        if self.bg_task:
            while not self.mp_queue.empty():
                current_progress = self.mp_queue.get_nowait()
                self.menu_icon.indicator_stop = current_progress
                self.status = f'Detecting blinks ({round(100*current_progress)}%)'

            for status in self.bg_task.fetch():
                self.status = status

            if self.bg_task.completed:
                self.bg_task = None
                self.menu_icon.indicator_stop = 0.0
                self.load_blinks()

    def cache_activation(self):
        t0, t1 = self.g_pool.timestamps[0], self.g_pool.timestamps[-1]

        class_points = deque([(t0, -0.9)])
        for b in self.g_pool.blinks:
            class_points.append((b["start_timestamp"], -0.9))
            class_points.append((b["start_timestamp"], 0.9))
            class_points.append((b["end_timestamp"], 0.9))
            class_points.append((b["end_timestamp"], -0.9))
        class_points.append((t1, -0.9))
        self.cache["class_points"] = tuple(class_points)

    def draw_activation(self, width, height, scale):
        t0, t1 = self.g_pool.timestamps[0], self.g_pool.timestamps[-1]
        with gl_utils.Coord_System(t0, t1, -1, 1):
            cygl_utils.draw_polyline(
                self.cache["class_points"],
                color=blink_color,
                line_type=gl.GL_LINE_STRIP,
                thickness=scale,
            )

    def draw_legend(self, width, height, scale):
        self.glfont.push_state()
        self.glfont.set_align_string(v_align="right", h_align="top")
        self.glfont.set_size(15.0 * scale)
        self.glfont.draw_text(width, 0, self.timeline.label)

        legend_height = 13.0 * scale
        pad = 10 * scale

        self.glfont.draw_text(width, legend_height, "Blinks")
        cygl_utils.draw_polyline(
            [
                (pad, legend_height + pad * 2 / 3),
                (width / 2, legend_height + pad * 2 / 3),
            ],
            color=blink_color,
            line_type=gl.GL_LINES,
            thickness=4.0 * scale,
        )
