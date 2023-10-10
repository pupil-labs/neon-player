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
import pathlib
import typing
import typing as T

import csv_utils
import gl_utils
import numpy as np
import OpenGL.GL as gl
import player_methods as pm
from plugin import Plugin
from pupil_recording import PupilRecording
from pyglui import pyfontstash, ui
from pyglui.cygl import utils as cygl_utils
from raw_data_exporter import _Base_Positions_Exporter
from scipy.spatial.transform import Rotation

from . import imu_pb2

logger = logging.getLogger(__name__)

def parse_neon_imu_raw_packets(buffer):
    index = 0
    packet_sizes = []
    while True:
        nums = np.frombuffer(buffer[index : index + 2], np.uint16)
        if not nums:
            break
        index += 2
        packet_size = nums[0]
        packet_sizes.append(packet_size)
        packet_bytes = buffer[index : index + packet_size]
        index += packet_size
        packet = imu_pb2.ImuPacket()
        packet.ParseFromString(packet_bytes)
        yield packet


def glfont_generator():
    glfont = pyfontstash.fontstash.Context()
    glfont.add_font("opensans", ui.get_opensans_font_path())
    glfont.set_color_float((1.0, 1.0, 1.0, 0.8))
    glfont.set_align_string(v_align="right", h_align="top")
    return glfont


def get_limits(data, keys):
    limits = (
        min(data[key].min() if data[key].shape[0] else 0 for key in keys),
        max(data[key].max() if data[key].shape[0] else 1 for key in keys),
    )
    # If the difference between the lower and upper bound is too small,
    # OpenGL will start throwing errors.
    limit_min_diff = 0.001
    if limits[1] - limits[0] < limit_min_diff:
        limits = limits[0] - limit_min_diff / 2, limits[0] + limit_min_diff / 2
    return limits


class IMURecording:
    DTYPE_RAW = np.dtype(
        [
            ("gyro_x", "<f4"),
            ("gyro_y", "<f4"),
            ("gyro_z", "<f4"),
            ("accel_x", "<f4"),
            ("accel_y", "<f4"),
            ("accel_z", "<f4"),
            ("yaw", "<f4"),
            ("pitch", "<f4"),
            ("roll", "<f4"),
        ]
    )

    def __init__(self, path_to_imu_raw: pathlib.Path):
        stem = path_to_imu_raw.stem
        self.path_raw = path_to_imu_raw
        self.path_ts = path_to_imu_raw.with_name(stem + "_timestamps.npy")
        self.load()

    def load(self):
        if not self.path_raw.exists() and self.path_ts.exists():
            self.ts = np.empty(0, dtype=np.float64)
            self.raw = []
            return

        self.ts = np.load(str(self.path_ts))
        with self.path_raw.open('rb') as raw_file:
            raw_data = raw_file.read()
            imu_packets = parse_neon_imu_raw_packets(raw_data)
            imu_data = []
            for packet in imu_packets:
                rotation = Rotation.from_quat([packet.rotVecData.x, packet.rotVecData.y, packet.rotVecData.z, packet.rotVecData.w])
                euler = rotation.as_euler(seq='XYZ', degrees=True)
                imu_data.append((
                    packet.accelData.x, packet.accelData.y, packet.accelData.z,
                    packet.gyroData.x, packet.gyroData.y, packet.gyroData.z,
                    *euler
                ))

            self.raw = np.array(imu_data, dtype=IMURecording.DTYPE_RAW).view(
                np.recarray
            )

        num_ts_during_init = self.ts.size - len(self.raw)
        if num_ts_during_init > 0:
            self.ts = self.ts[num_ts_during_init:]


class IMUTimeline(Plugin):
    """
    plot and export imu data
    export: imu_timeline.csv
    keys:
        imu_timestamp: timestamp of the source image frame
        world_index: associated_frame: closest world video frame
        gyro_x: angular velocity about the x axis in degrees/s
        gyro_y: angular velocity about the y axis in degrees/s
        gyro_z: angular velocity about the z axis in degrees/s
        accel_x: linear acceleration along the x axis in G (9.80665 m/s^2)
        accel_y: linear acceleration along the y axis in G (9.80665 m/s^2)
        accel_z: linear acceleration along the z axis in G (9.80665 m/s^2)
        yaw: orientation expressed as Euler angles
        pitch: orientation expressed as Euler angles
        roll: orientation expressed as Euler angles
    See Pupil docs for relevant coordinate systems
    """

    IMU_PATTERN_RAW = r"^extimu ps(\d+).raw"

    CMAP = {
        "gyro_x": cygl_utils.RGBA(0.12156, 0.46666, 0.70588, 1.0),
        "gyro_y": cygl_utils.RGBA(1.0, 0.49803, 0.05490, 1.0),
        "gyro_z": cygl_utils.RGBA(0.17254, 0.62745, 0.1725, 1.0),
        "accel_x": cygl_utils.RGBA(0.83921, 0.15294, 0.15686, 1.0),
        "accel_y": cygl_utils.RGBA(0.58039, 0.40392, 0.74117, 1.0),
        "accel_z": cygl_utils.RGBA(0.54901, 0.33725, 0.29411, 1.0),
        "yaw": cygl_utils.RGBA(0.49803, 0.70588, 0.05490, 1.0),
        "pitch": cygl_utils.RGBA(0.12156, 0.46666, 0.70588, 1.0),
        "roll": cygl_utils.RGBA(1.0, 0.49803, 0.05490, 1.0),
    }
    NUMBER_SAMPLES_TIMELINE = 4000
    TIMELINE_LINE_HEIGHT = 16
    icon_chr = chr(0xEC22)
    icon_font = "pupil_icons"

    @classmethod
    def parse_pretty_class_name(cls) -> str:
        return "IMU Timeline"

    @classmethod
    def is_available_within_context(cls, g_pool) -> bool:
        imu_recs = cls._imu_recordings(g_pool)

        if not len(imu_recs):
            # Plugin not available if recording doesn't have IMU files (due to hardware failure, for example)
            logger.debug(f"{cls.__name__} unavailable because there are no IMU files")
            return False

        return True

    @classmethod
    def _imu_recordings(cls, g_pool) -> T.List[IMURecording]:
        rec = PupilRecording(g_pool.rec_dir)
        imu_files: T.List[pathlib.Path] = sorted(
            rec.files().filter_patterns(cls.IMU_PATTERN_RAW)
        )
        return [IMURecording(imu_file) for imu_file in imu_files]

    def __init__(
        self,
        g_pool,
        should_draw=True,
    ):
        super().__init__(g_pool)
        imu_recs = self._imu_recordings(g_pool)

        self.should_draw = should_draw

        self.gyro_timeline = None
        self.accel_timeline = None
        self.orient_timeline = None
        self.glfont_raw = None

        self.data_raw = np.concatenate([rec.raw for rec in imu_recs])
        self.data_ts = np.concatenate([rec.ts for rec in imu_recs])
        self.data_len = len(self.data_raw)
        self.gyro_keys = ["gyro_x", "gyro_y", "gyro_z"]
        self.accel_keys = ["accel_x", "accel_y", "accel_z"]
        self.orient_keys = ["yaw", "pitch", "roll"]

    def get_init_dict(self):
        return {
            "should_draw": self.should_draw,
        }

    def init_ui(self):
        self.add_menu()
        self.menu.label = "IMU Timeline"
        self.menu.append(ui.Info_Text("Visualize IMU data and export to .csv file"))
        self.menu.append(
            ui.Info_Text(
                "This plugin visualizes accelerometer, gyroscope and "
                " orientation data from Neon recordings. Results are "
                " exported in 'imu_timeline.csv' "
            )
        )

        self.menu.append(
            ui.Switch(
                "should_draw",
                self,
                label="View the IMU timeline",
                setter=self.on_draw_toggled,
            )
        )
        if self.should_draw:
            self.append_timeline_raw()

    def deinit_ui(self):
        if self.should_draw:
            self.g_pool.user_timelines.remove(self.gyro_timeline)
            self.g_pool.user_timelines.remove(self.accel_timeline)
            self.g_pool.user_timelines.remove(self.orient_timeline)
            del self.gyro_timeline
            del self.accel_timeline
            del self.orient_timeline
            del self.glfont_raw

        self.cleanup()
        self.remove_menu()

    def on_draw_toggled(self, new_value):
        self.should_draw = new_value
        if self.should_draw:
            self.append_timeline_raw()
        else:
            self.remove_timeline_raw()

    def append_timeline_raw(self):
        self.gyro_timeline = ui.Timeline(
            "gyro",
            self.draw_raw_gyro,
            self.draw_legend_gyro,
            self.TIMELINE_LINE_HEIGHT * 3,
        )
        self.accel_timeline = ui.Timeline(
            "accel",
            self.draw_raw_accel,
            self.draw_legend_accel,
            self.TIMELINE_LINE_HEIGHT * 3,
        )
        self.orient_timeline = ui.Timeline(
            "orientation",
            self.draw_orient,
            self.draw_legend_orient,
            self.TIMELINE_LINE_HEIGHT * 3,
        )
        self.g_pool.user_timelines.append(self.gyro_timeline)
        self.g_pool.user_timelines.append(self.accel_timeline)
        self.g_pool.user_timelines.append(self.orient_timeline)
        self.glfont_raw = glfont_generator()

    def remove_timeline_raw(self):
        self.g_pool.user_timelines.remove(self.gyro_timeline)
        self.g_pool.user_timelines.remove(self.accel_timeline)
        self.g_pool.user_timelines.remove(self.orient_timeline)
        del self.gyro_timeline
        del self.accel_timeline
        del self.orient_timeline
        del self.glfont_raw

    def draw_raw_gyro(self, width, height, scale):
        y_limits = get_limits(self.data_raw, self.gyro_keys)
        self._draw_grouped(
            self.data_raw, self.gyro_keys, y_limits, width, height, scale
        )

    def draw_raw_accel(self, width, height, scale):
        y_limits = get_limits(self.data_raw, self.accel_keys)
        self._draw_grouped(
            self.data_raw, self.accel_keys, y_limits, width, height, scale
        )

    def draw_orient(self, width, height, scale):
        y_limits = get_limits(self.data_raw, self.orient_keys)
        self._draw_grouped(
            self.data_raw, self.orient_keys, y_limits, width, height, scale
        )

    def _draw_grouped(self, data, keys, y_limits, width, height, scale):
        ts_min = self.g_pool.timestamps[0]
        ts_max = self.g_pool.timestamps[-1]
        data_raw = data[keys]
        sub_samples = np.linspace(
            0,
            self.data_len - 1,
            min(self.NUMBER_SAMPLES_TIMELINE, self.data_len),
            dtype=int,
        )
        with gl_utils.Coord_System(ts_min, ts_max, *y_limits):
            for key in keys:
                data_keyed = data_raw[key]
                if data_keyed.shape[0] == 0:
                    continue
                points = list(zip(self.data_ts[sub_samples], data_keyed[sub_samples]))
                cygl_utils.draw_points(points, size=1.5 * scale, color=self.CMAP[key])

    def draw_legend_gyro(self, width, height, scale):
        self._draw_legend_grouped(self.gyro_keys, width, height, scale, self.glfont_raw)

    def draw_legend_accel(self, width, height, scale):
        self._draw_legend_grouped(
            self.accel_keys, width, height, scale, self.glfont_raw
        )

    def draw_legend_orient(self, width, height, scale):
        self._draw_legend_grouped(
            self.orient_keys, width, height, scale, self.glfont_raw
        )

    def _draw_legend_grouped(self, labels, width, height, scale, glfont):
        glfont.set_size(self.TIMELINE_LINE_HEIGHT * 0.8 * scale)
        pad = width * 2 / 3
        for label in labels:
            color = self.CMAP[label]
            glfont.draw_text(width, 0, label)

            cygl_utils.draw_polyline(
                [
                    (pad, self.TIMELINE_LINE_HEIGHT / 2),
                    (width / 4, self.TIMELINE_LINE_HEIGHT / 2),
                ],
                color=color,
                line_type=gl.GL_LINES,
                thickness=4.0 * scale,
            )
            gl.glTranslatef(0, self.TIMELINE_LINE_HEIGHT * scale, 0)

    def on_notify(self, notification):
        if notification["subject"] == "should_export":
            self.export_data(notification["ts_window"], notification["export_dir"])

    def export_data(self, export_window, export_dir):
        imu_bisector = Imu_Bisector(self.data_raw, self.data_ts)
        imu_exporter = Imu_Exporter()
        imu_exporter.csv_export_write(
            imu_bisector=imu_bisector,
            timestamps=self.g_pool.timestamps,
            export_window=export_window,
            export_dir=export_dir,
        )


class Imu_Bisector(pm.Bisector):
    """Stores data with associated timestamps, both sorted by the timestamp;
    subclassed to avoid casting to object and losing dtypes for recarrays"""

    def __init__(self, data=(), data_ts=()):
        if len(data) != len(data_ts):
            raise ValueError(
                "Each element in 'data' requires a corresponding"
                " timestamp in `data_ts`"
            )

        elif not len(data):
            self.data = np.array([], dtype=object)
            self.data_ts = np.array([])
            self.sorted_idc = []

        else:
            self.data_ts = data_ts
            self.data = data

            # Find correct order once and reorder both lists in-place
            self.sorted_idc = np.argsort(self.data_ts)
            self.data_ts = self.data_ts[self.sorted_idc]
            self.data = self.data[self.sorted_idc]


class Imu_Exporter(_Base_Positions_Exporter):
    @classmethod
    def csv_export_filename(cls) -> str:
        return "imu_data.csv"

    @classmethod
    def csv_export_labels(cls) -> typing.Tuple[csv_utils.CSV_EXPORT_LABEL_TYPE, ...]:
        return (
            "imu_timestamp",
            "world_index",
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "accel_x",
            "accel_y",
            "accel_z",
            "yaw",
            "pitch",
            "roll",
        )

    @classmethod
    def dict_export(
        cls, raw_value: csv_utils.CSV_EXPORT_RAW_TYPE, world_ts: float, world_index: int
    ) -> dict:
        try:
            imu_timestamp = str(world_ts)
            gyro_x = raw_value["gyro_x"]
            gyro_y = raw_value["gyro_y"]
            gyro_z = raw_value["gyro_z"]
            accel_x = raw_value["accel_x"]
            accel_y = raw_value["accel_y"]
            accel_z = raw_value["accel_z"]
            yaw = raw_value["yaw"]
            pitch = raw_value["pitch"]
            roll = raw_value["roll"]
        except KeyError:
            imu_timestamp = None
            gyro_x = None
            gyro_y = None
            gyro_z = None
            accel_x = None
            accel_y = None
            accel_z = None
            yaw = None
            pitch = None
            roll = None

        return {
            "imu_timestamp": imu_timestamp,
            "world_index": world_index,
            "gyro_x": gyro_x,
            "gyro_y": gyro_y,
            "gyro_z": gyro_z,
            "accel_x": accel_x,
            "accel_y": accel_y,
            "accel_z": accel_z,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
        }

    def csv_export_write(self, imu_bisector, timestamps, export_window, export_dir):
        export_file = type(self).csv_export_filename()
        export_path = os.path.join(export_dir, export_file)

        export_section = imu_bisector.init_dict_for_window(export_window)
        export_world_idc = pm.find_closest(timestamps, export_section["data_ts"])

        with open(export_path, "w", encoding="utf-8", newline="") as csvfile:
            csv_header = type(self).csv_export_labels()
            dict_writer = csv.DictWriter(csvfile, fieldnames=csv_header)
            dict_writer.writeheader()

            for d_raw, wts, idx in zip(
                export_section["data"], export_section["data_ts"], export_world_idc
            ):
                dict_row = type(self).dict_export(
                    raw_value=d_raw, world_ts=wts, world_index=idx
                )
                dict_writer.writerow(dict_row)

        logger.info(f"Created '{export_file}' file.")
