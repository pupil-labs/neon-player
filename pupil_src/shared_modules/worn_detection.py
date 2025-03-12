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

import gl_utils
import numpy as np
import OpenGL.GL as gl
import pyglui.cygl.utils as cygl_utils
from plugin import Plugin
from pyglui import ui
from pyglui.pyfontstash import fontstash as fs

logger = logging.getLogger(__name__)

worn_color = cygl_utils.RGBA(0.88, 0.88, 0.18, 0.8)


class Worn_Detection(Plugin):
    """
    This plugin loads "worn" sensor data from the recording
    """

    order = 0.8
    icon_chr = chr(0xE81A)
    icon_font = "pupil_icons"

    def __init__(self, g_pool, hide_gaze_when_not_worn=True):
        super().__init__(g_pool)
        self.menu = None
        self.graph_points = None

    def init_ui(self):
        super().init_ui()
        self.add_menu()
        self.menu.label = "Worn Detection"

        self.glfont = fs.Context()
        self.glfont.add_font("opensans", ui.get_opensans_font_path())
        self.glfont.set_font("opensans")
        self.timeline = ui.Timeline(
            "Worn Detection", self.draw_timeline, self.draw_legend
        )
        self.g_pool.user_timelines.append(self.timeline)

    def deinit_ui(self):
        super().deinit_ui()
        self.remove_menu()
        self.g_pool.user_timelines.remove(self.timeline)
        self.timeline = None

    def on_notify(self, notification):
        if notification["subject"] == "should_export":
            self.export(notification["ts_window"], notification["export_dir"])

    def export(self, export_window, export_dir):
        """
        Between in and out mark
            worn.csv:
                timestamp [ns] | worn
        """
        header = ("timestamp [ns]", "worn",)
        t0, t1 = [self.g_pool.capture.ts_to_ns(v) for v in export_window]
        timestamps = self.g_pool.recording_api.worn.ts
        timestamps = timestamps[(timestamps >= t0) & (timestamps <= t1)]

        worn_in_section = self.g_pool.recording_api.worn.sample(timestamps)

        with open(
            os.path.join(export_dir, "worn.csv"), "w", encoding="utf-8", newline=""
        ) as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=header)
            csv_writer.writeheader()
            for w in worn_in_section:
                csv_writer.writerow({
                    "timestamp [ns]": f"{w.ts:0.0f}",
                    "worn": w.worn,
                })
            logger.info("Created 'worn.csv' file.")

    def draw_timeline(self, width, height, scale):
        t0, t1 = [self.g_pool.capture.ts_to_ns(self.g_pool.timestamps[v]) for v in [0, -1]]

        if self.graph_points is None or width != len(self.graph_points):
            samples = self.g_pool.recording_api.worn.sample(
                np.linspace(t0, t1, width),
                "linear"
            )
            self.graph_points = [(idx, sample[1]) for idx, sample in enumerate(samples)]

        with gl_utils.Coord_System(0, len(self.graph_points), -1, 300):
            cygl_utils.draw_polyline(
                self.graph_points,
                color=worn_color,
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

        self.glfont.draw_text(width, legend_height, "Worn")
        cygl_utils.draw_polyline(
            [
                (pad, legend_height + pad * 2 / 3),
                (width / 2, legend_height + pad * 2 / 3),
            ],
            color=worn_color,
            line_type=gl.GL_LINES,
            thickness=4.0 * scale,
        )
