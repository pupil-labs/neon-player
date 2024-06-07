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

import cv2
import numpy as np
from data_changed import Listener
from methods import denormalize
from observable import Observable
from plugin import Plugin
from pyglui import ui

from scan_path import ScanPathController
from scan_path.utils import np_denormalize
from gaze_producer.gaze_from_recording import GazeFromRecording

logger = logging.getLogger(__name__)


class Vis_Light_Points(Plugin, Observable):
    """docstring
    show gaze dots at light dots on numpy.

    """

    uniqueness = "not_unique"
    icon_chr = chr(0xE3A5)
    icon_font = "pupil_icons"

    def __init__(self, g_pool, falloff=20, scan_path_init_dict={}):
        super().__init__(g_pool)
        self.order = 0.8
        self.menu = None

        self.falloff = falloff

        self.scan_path_controller = ScanPathController(g_pool, **scan_path_init_dict)
        self.scan_path_controller.add_observer(
            "on_update_ui", self._update_scan_path_ui
        )

    def recent_events(self, events):
        self.scan_path_controller.process()

        frame = events.get("frame")
        if not frame:
            return

        pts = self._polyline_points(
            image_size=frame.img.shape[:-1][::-1],
            base_gaze_data=events.get("gaze", []),
            scan_path_gaze_data=self.scan_path_controller.scan_path_gaze_for_frame(
                frame
            ),
        )

        if not pts:
            return

        pts = np.array(pts, dtype=np.int32)

        if self.scan_path_controller.timeframe > 0:
            for plugin in self.g_pool.plugins:
                if isinstance(plugin, GazeFromRecording):
                    manual_correction = plugin.get_manual_correction_for_frame(frame.index)
                    pts[:,0] += int(manual_correction[0]*frame.img.shape[1])
                    pts[:,1] -= int(manual_correction[1]*frame.img.shape[0])

                    break

        img = frame.img
        overlay = np.ones(img.shape[:-1], dtype=img.dtype)

        # draw recent gaze postions as black dots on an overlay image.
        for gaze_point in pts:
            try:
                overlay[int(gaze_point[1]), int(gaze_point[0])] = 0
            except Exception:
                pass

        out = cv2.distanceTransform(overlay, cv2.DIST_L2, 5)

        # fix for opencv binding inconsitency
        if type(out) == tuple:
            out = out[0]

        overlay = 1 / (out / self.falloff + 1)

        img[:] = np.multiply(
            img, cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB), casting="unsafe"
        )

    def cleanup(self):
        self.scan_path_controller.cleanup()

    def _update_scan_path_ui(self):
        if self.menu_icon:
            self.menu_icon.indicator_stop = self.scan_path_controller.progress
        if self.scan_path_status:
            self.scan_path_status.text = self.scan_path_controller.status_string

    def _polyline_points(self, image_size, base_gaze_data, scan_path_gaze_data):
        if scan_path_gaze_data is not None:
            points_fields = ["norm_x", "norm_y"]
            gaze_points = scan_path_gaze_data[points_fields]
            gaze_points = np.array(
                gaze_points.tolist(), dtype=gaze_points.dtype[0]
            )  # FIXME: This is a workaround
            gaze_points = gaze_points.reshape((-1, len(points_fields)))
            gaze_points = np_denormalize(gaze_points, image_size, flip_y=True)
            return gaze_points.tolist()
        else:
            return [
                denormalize(datum["norm_pos"], image_size, flip_y=True)
                for datum in base_gaze_data
                if datum["confidence"] >= self.g_pool.min_data_confidence
            ]

    def init_ui(self):
        self.add_menu()
        self.menu.label = self.pretty_class_name
        self.falloff_slider = ui.Slider("falloff", self, min=1, step=1, max=1000, label="Falloff")
        self.menu.append(self.falloff_slider)

        self.scan_path_timeframe_range = ui.Slider(
            "timeframe",
            self.scan_path_controller,
            min=self.scan_path_controller.min_timeframe,
            max=self.scan_path_controller.max_timeframe,
            step=self.scan_path_controller.timeframe_step,
            label="Duration",
        )

        scan_path_doc = ui.Info_Text("Duration of past gaze to include in polyline.")
        self.scan_path_status = ui.Info_Text("")

        scan_path_menu = ui.Growing_Menu("Gaze History")
        scan_path_menu.collapsed = False
        scan_path_menu.append(scan_path_doc)
        scan_path_menu.append(self.scan_path_timeframe_range)
        scan_path_menu.append(self.scan_path_status)

        self.menu.append(scan_path_menu)

    def deinit_ui(self):
        self.remove_menu()
        self.scan_path_timeframe_range = None
        self.scan_path_status = None

    def get_init_dict(self):
        return {
            "falloff": self.falloff,
            "scan_path_init_dict": self.scan_path_controller.get_init_dict(),
        }
