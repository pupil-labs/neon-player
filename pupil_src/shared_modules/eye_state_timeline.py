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
import typing

import csv_utils
import gl_utils
import numpy as np
import OpenGL.GL as gl
from plugin import Plugin
from pyglui import pyfontstash, ui
from pyglui.cygl import utils as cygl_utils
from raw_data_exporter import _Base_Positions_Exporter


logger = logging.getLogger(__name__)


def glfont_generator():
    glfont = pyfontstash.fontstash.Context()
    glfont.add_font("opensans", ui.get_opensans_font_path())
    glfont.set_align_string(v_align="right", h_align="top")
    return glfont


def get_limits(data, keys):
    finite_data = data[np.isfinite(data)]
    limits = (np.min(finite_data), np.max(finite_data))

    # If the difference between the lower and upper bound is too small,
    # OpenGL will start throwing errors.
    limit_min_diff = 0.001
    if limits[1] - limits[0] < limit_min_diff:
        limits = limits[0] - limit_min_diff / 2, limits[0] + limit_min_diff / 2

    return limits


class EyeStateTimeline(Plugin):
    """
    Plot and export eye state data
    export: 3d_eye_states.csv
    keys:
        timestamp [ns]: timestamp of the eye state data
        pupil diameter left [mm]: the diameter of the left pupil in mm
        pupil diameter right [mm]: the diameter of the right eye in mm
        eyeball center left x [mm]: location of the left eye center on the x axis in mm
        eyeball center left y [mm]: location of the left eye center on the y axis in mm
        eyeball center left z [mm]: location of the left eye center on the z axis in mm
        eyeball center right x [mm]: location of the right eye center on the x axis in mm
        eyeball center right y [mm]: location of the right eye center on the y axis in mm
        eyeball center right z [mm]: location of the right eye center on the z axis in mm
        optical axis left x: direction of the left eye optical axis
        optical axis left y: direction of the left eye optical axis
        optical axis left z: direction of the left eye optical axis
        optical axis right x: direction of the right eye optical axis
        optical axis right y: direction of the right eye optical axis
        optical axis right z: direction of the right eye optical axis
        eyelid_angle_top_left: angle of the top eyelid of the left eye,
        eyelid_angle_bottom_left: angle of the bottom eyelid of the left eye,
        eyelid_angle_top_right: angle of the top eyelid of the right eye,
        eyelid_angle_bottom_right: angle of the bottom eyelid of the right eye,
        eyelid_aperture_left_mm: aperture of the left eyelid in mm,
        eyelid_aperture_right_mm: aperture of the left eyelid in mm,
    """

    icon_chr = chr(0xEC02)
    icon_font = "pupil_icons"

    CMAP = {
        "pupil_diameter_left_mm": cygl_utils.RGBA(0.12156, 0.46666, 0.70588, 1.0),
        "pupil_diameter_right_mm": cygl_utils.RGBA(1.0, 0.49803, 0.05490, 1.0),

        "eyeball_center_left_x": cygl_utils.RGBA(0.83921, 0.15294, 0.15686, 1.0),
        "eyeball_center_left_y": cygl_utils.RGBA(0.58039, 0.40392, 0.74117, 1.0),
        "eyeball_center_left_z": cygl_utils.RGBA(0.54901, 0.33725, 0.29411, 1.0),

        "eyeball_center_right_x": cygl_utils.RGBA(0.12156, 0.46666, 0.70588, 1.0),
        "eyeball_center_right_y": cygl_utils.RGBA(0.49803, 0.70588, 0.05490, 1.0),
        "eyeball_center_right_z": cygl_utils.RGBA(1.0, 0.49803, 0.05490, 1.0),

        "optical_axis_left_x": cygl_utils.RGBA(0.83921, 0.15294, 0.15686, 1.0),
        "optical_axis_left_y": cygl_utils.RGBA(0.58039, 0.40392, 0.74117, 1.0),
        "optical_axis_left_z": cygl_utils.RGBA(0.54901, 0.33725, 0.29411, 1.0),

        "optical_axis_right_x": cygl_utils.RGBA(0.12156, 0.46666, 0.70588, 1.0),
        "optical_axis_right_y": cygl_utils.RGBA(0.49803, 0.70588, 0.05490, 1.0),
        "optical_axis_right_z": cygl_utils.RGBA(1.0, 0.49803, 0.05490, 1.0),

        "eyelid_angle_top_left": cygl_utils.RGBA(0.12156, 0.46666, 0.70588, 1.0),
        "eyelid_angle_bottom_left": cygl_utils.RGBA(0.83921, 0.15294, 0.15686, 1.0),

        "eyelid_angle_top_right": cygl_utils.RGBA(1.0, 0.49803, 0.05490, 1.0),
        "eyelid_angle_bottom_right": cygl_utils.RGBA(0.54901, 0.33725, 0.29411, 1.0),

        "eyelid_aperture_left_mm": cygl_utils.RGBA(0.12156, 0.46666, 0.70588, 1.0),
        "eyelid_aperture_right_mm": cygl_utils.RGBA(1.0, 0.49803, 0.05490, 1.0),
    }
    NUMBER_SAMPLES_TIMELINE = 4000
    TIMELINE_LINE_HEIGHT = 16

    @classmethod
    def parse_pretty_class_name(cls) -> str:
        return "Eye State Timeline"

    def __init__(
        self,
        g_pool,
        show_pupil_diameters=True,
        show_eyeball_centers=True,
        show_optical_axes=True,
        show_eyelid_angles=True,
        show_eyelid_apertures=True,
    ):
        super().__init__(g_pool)

        self.toggle_values = {
            "pupil_diameters": show_pupil_diameters,
            "eyeball_centers": show_eyeball_centers,
            "optical_axes": show_optical_axes,
            "eyelid_angles": show_eyelid_angles,
            "eyelid_apertures": show_eyelid_apertures,
        }

        self.legend_keys = {
            "pupil_diameters": [f"pupil_diameter_{d}_mm" for d in ["left", "right"]],
            "eyeball_centers": [f"eyeball_center_{d}_{a}" for d in ["left", "right"] for a in "xyz"],
            "optical_axes": [f"optical_axis_{d}_{a}" for d in ["left", "right"] for a in "xyz"],
            "eyelid_angles": [f"eyelid_angle_{a}_{d}" for d in ["left", "right"] for a in ["top", "bottom"]],
            "eyelid_apertures": [f"eyelid_aperture_{d}_mm" for d in ["left", "right"]],
        }

        self.timelines = {}
        self.glfont_raw = None

        self.resampled_data = None

    def get_init_dict(self):
        return {f"show_{k}": v for k, v in self.toggle_values.items()}

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Eye State Timelines"

        for toggle_name, enabled in self.toggle_values.items():
            self.menu.append(ui.Switch(
                toggle_name,
                self.toggle_values,
                label=toggle_name.replace("_", " ").title(),
                setter=lambda v, n=toggle_name: self.on_timeline_toggled(n, v),
            ))

            self.on_timeline_toggled(toggle_name, enabled)

        self.glfont_raw = glfont_generator()

    def deinit_ui(self):
        for toggle_name, enabled in self.toggle_values.items():
            if enabled:
                self.g_pool.user_timelines.remove(self.timelines[toggle_name])
                del self.timelines[toggle_name]

        del self.glfont_raw

        self.cleanup()
        self.remove_menu()

    def on_timeline_toggled(self, toggle_name, new_value):
        self.toggle_values[toggle_name] = new_value
        if new_value:

            if toggle_name.startswith("pupil_diameter"):
                height = self.TIMELINE_LINE_HEIGHT * 3
            elif toggle_name.startswith("eyelid_aperture"):
                height = self.TIMELINE_LINE_HEIGHT * 3
            elif toggle_name.startswith("eyelid_angle"):
                height = self.TIMELINE_LINE_HEIGHT * 5
            else:
                height = self.TIMELINE_LINE_HEIGHT * 7

            self.timelines[toggle_name] = ui.Timeline(
                toggle_name,
                lambda *args, **kwargs: self.draw_raw(toggle_name, *args, **kwargs),
                lambda *args, **kwargs: self.draw_legend(toggle_name, *args, **kwargs),
                height,
            )

            self.g_pool.user_timelines.append(self.timelines[toggle_name])

        elif toggle_name in self.timelines:
            self.g_pool.user_timelines.remove(self.timelines[toggle_name])
            del self.timelines[toggle_name]

    def draw_raw(self, name, width, height, scale):
        if self.g_pool.recording_api.eye_state.data.size < 1:
            return

        ts_min = self.g_pool.recording_api.scene.ts[0]
        ts_max = self.g_pool.recording_api.scene.ts[-1]
        timestamps = np.linspace(ts_min, ts_max, width)

        if self.resampled_data is None or self.resampled_data.data.shape[0] != width:
            self.resampled_data = self.g_pool.recording_api.eye_state.sample(timestamps)

        keys = self.legend_keys[name]
        try:
            data = self.resampled_data[keys]
            y_limits = get_limits(data, keys)

            gl.glTranslatef(0, self.TIMELINE_LINE_HEIGHT * scale, 0)
            with gl_utils.Coord_System(0, width, *y_limits):
                for key_idx, key in enumerate(keys):
                    data_keyed = data[:, key_idx]
                    if data_keyed.shape[0] == 0:
                        continue

                    points = list(zip(range(width), data_keyed))
                    cygl_utils.draw_points(points, size=1.5 * scale, color=self.CMAP[key])
        except KeyError:
            pass

    def draw_legend(self, name, width, height, scale):
        self._draw_legend_grouped(self.legend_keys[name], width, height, scale, self.glfont_raw)

    def _draw_legend_grouped(self, labels, width, height, scale, glfont):
        friendly_labels = {}
        glfont.set_size(self.TIMELINE_LINE_HEIGHT * scale)
        glfont.set_align_string(v_align="left", h_align="top")
        for prefix in ["pupil_diameter", "eyeball_center", "optical_axis", "eyelid_angle", "eyelid_aperture"]:
            if labels[0].startswith(prefix):
                friendly_labels = {
                    label: label.replace(f"{prefix}_", "").replace("_", " ").replace("_mm", "").title()
                    for label in labels
                }

                glfont.draw_text(10, 0, prefix.replace("_", " ").title())
                gl.glTranslatef(0, self.TIMELINE_LINE_HEIGHT * scale, 0)
                break

        glfont.set_size(self.TIMELINE_LINE_HEIGHT * 0.8 * scale)
        glfont.set_align_string(v_align="right", h_align="top")
        for label in labels:
            color = self.CMAP[label]
            cygl_utils.draw_polyline(
                [
                    (20, 8),
                    (60, 8),
                ],
                color=color,
                line_type=gl.GL_LINES,
                thickness=4.0 * scale,
            )
            friendly_label = friendly_labels.get(label, label)
            if friendly_label.endswith(" Mm"):
                friendly_label = friendly_label[:-3]
            glfont.draw_text(width, 0, friendly_label)
            gl.glTranslatef(0, self.TIMELINE_LINE_HEIGHT * scale, 0)

    def on_notify(self, notification):
        if notification["subject"] == "should_export":
            self.export_data(notification["ts_window"], notification["export_dir"])

    def export_data(self, export_window, export_dir):
        export_window_unix = [self.g_pool.capture.ts_to_ns(ts) for ts in export_window]
        all_ts = self.g_pool.recording_api.eye_state.data.ts

        window_filter = (export_window_unix[0] <= all_ts) & (all_ts <= export_window_unix[1])
        windowed_ts = all_ts[window_filter]

        if len(self.g_pool.recording_api.eye_state) == 0:
            data = []
        else:
            data = self.g_pool.recording_api.eye_state.sample(windowed_ts)

        eye_state_exporter = EyeStateExporter()
        eye_state_exporter.csv_export_write(data, export_dir)


class EyeStateExporter(_Base_Positions_Exporter):
    field_map = {
        "ts": "timestamp [ns]",
        "pupil_diameter_left_mm": "pupil diameter left [mm]",
        "pupil_diameter_right_mm": "pupil diameter right [mm]",
        "eyeball_center_left_x": "eyeball center left x [mm]",
        "eyeball_center_left_y": "eyeball center left y [mm]",
        "eyeball_center_left_z": "eyeball center left z [mm]",
        "eyeball_center_right_x": "eyeball center right x [mm]",
        "eyeball_center_right_y": "eyeball center right y [mm]",
        "eyeball_center_right_z": "eyeball center right z [mm]",
        "optical_axis_left_x": "optical axis left x",
        "optical_axis_left_y": "optical axis left y",
        "optical_axis_left_z": "optical axis left z",
        "optical_axis_right_x": "optical axis right x",
        "optical_axis_right_y": "optical axis right y",
        "optical_axis_right_z": "optical axis right z",
        "eyelid_angle_top_left": "eyelid angle top left [deg]",
        "eyelid_angle_bottom_left": "eyelid angle bottom left [deg]",
        "eyelid_angle_top_right": "eyelid angle top right [deg]",
        "eyelid_angle_bottom_right": "eyelid angle bottom right [deg]",
        "eyelid_aperture_left_mm": "eyelid aperture left [mm]",
        "eyelid_aperture_right_mm": "eyelid aperture right [mm]",
    }

    @classmethod
    def csv_export_filename(cls) -> str:
        return "3d_eye_states.csv"

    @classmethod
    def csv_export_labels(cls) -> typing.Tuple[csv_utils.CSV_EXPORT_LABEL_TYPE, ...]:
        return EyeStateExporter.field_map.values()

    @classmethod
    def dict_export(
        cls, raw_value: csv_utils.CSV_EXPORT_RAW_TYPE, world_ts: float = None, world_index: int = None
    ) -> dict:
        data = {}
        for k, v in EyeStateExporter.field_map.items():
            if hasattr(raw_value, k):
                data[v] = raw_value[k]

        data["timestamp [ns]"] = f"{data['timestamp [ns]']:0.0f}"

        return data

    def csv_export_write(self, data, export_dir):
        export_file = type(self).csv_export_filename()
        export_path = os.path.join(export_dir, export_file)

        with open(export_path, "w", encoding="utf-8", newline="") as csvfile:
            csv_header = type(self).csv_export_labels()
            dict_writer = csv.DictWriter(csvfile, fieldnames=csv_header)
            dict_writer.writeheader()

            for raw_record in data:
                csv_record = type(self).dict_export(raw_record)
                dict_writer.writerow(csv_record)

        logger.info(f"Created '{export_file}' file.")
