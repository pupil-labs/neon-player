"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import abc
import csv
import logging
import os
import typing

import csv_utils
import player_methods as pm
from plugin import Plugin
from pyglui import ui
from rich.progress import track

# logging
logger = logging.getLogger(__name__)


class Raw_Data_Exporter(Plugin):
    """
    gaze_positions.csv
    keys:
        timestamp - timestamp of the source image frame
        index - associated_frame: closest world video frame
        confidence - computed confidence between 0 (not confident) -1 (confident)
        norm_pos_x - x position in the world image frame in normalized coordinates
        norm_pos_y - y position in the world image frame in normalized coordinates
        base_data - "timestamp-id timestamp-id ..." of pupil data that this gaze position is computed from

        #data made available by the 3d vector gaze mappers
        gaze_point_3d_x - x position of the 3d gaze point (the point the sublejct lookes at) in the world camera coordinate system
        gaze_point_3d_y - y position of the 3d gaze point
        gaze_point_3d_z - z position of the 3d gaze point
        eye_center0_3d_x - x center of eye-ball 0 in the world camera coordinate system (of camera 0 for binocular systems or any eye camera for monocular system)
        eye_center0_3d_y - y center of eye-ball 0
        eye_center0_3d_z - z center of eye-ball 0
        gaze_normal0_x - x normal of the visual axis for eye 0 in the world camera coordinate system (of eye 0 for binocular systems or any eye for monocular system). The visual axis goes through the eye ball center and the object thats looked at.
        gaze_normal0_y - y normal of the visual axis for eye 0
        gaze_normal0_z - z normal of the visual axis for eye 0
        eye_center1_3d_x - x center of eye-ball 1 in the world camera coordinate system (not avaible for monocular setups.)
        eye_center1_3d_y - y center of eye-ball 1
        eye_center1_3d_z - z center of eye-ball 1
        gaze_normal1_x - x normal of the visual axis for eye 1 in the world camera coordinate system (not avaible for monocular setups.). The visual axis goes through the eye ball center and the object thats looked at.
        gaze_normal1_y - y normal of the visual axis for eye 1
        gaze_normal1_z - z normal of the visual axis for eye 1
    """

    icon_chr = chr(0xE873)
    icon_font = "pupil_icons"

    def __init__(
        self,
        g_pool,
        should_export_field_info=True,
        should_export_gaze_positions=True,
        should_include_low_confidence_data=True,
    ):
        super().__init__(g_pool)

        self.should_export_field_info = should_export_field_info
        self.should_export_gaze_positions = should_export_gaze_positions

    def get_init_dict(self):
        return {
            "should_export_field_info": self.should_export_field_info,
            "should_export_gaze_positions": self.should_export_gaze_positions,
        }

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Raw Data Exporter"
        self.menu.append(ui.Info_Text("Export Raw Pupil Capture data into .csv files."))
        self.menu.append(
            ui.Info_Text(
                "Select your export frame range using the trim marks in the seek bar. This will affect all exporting plugins."
            )
        )

        self.menu.append(
            ui.Switch(
                "should_export_field_info",
                self,
                label="Export Pupil Gaze Positions Info",
            )
        )
        self.menu.append(
            ui.Switch(
                "should_export_gaze_positions", self, label="Export Gaze Positions"
            )
        )
        self.menu.append(
            ui.Info_Text("Press the export button or type 'e' to start the export.")
        )

    def deinit_ui(self):
        self.remove_menu()

    def on_notify(self, notification):
        if notification["subject"] == "should_export":
            self.export_data(notification["ts_window"], notification["export_dir"])

    def export_data(self, export_window, export_dir):
        if self.should_export_gaze_positions:
            gaze_positions_exporter = Gaze_Positions_Exporter()
            gaze_positions_exporter.csv_export_write(
                positions_bisector=self.g_pool.gaze_positions,
                timestamps=self.g_pool.timestamps,
                export_window=export_window,
                export_dir=export_dir,
                min_confidence_threshold=0.0
            )

        if self.should_export_field_info:
            field_info_name = "pupil_gaze_positions_info.txt"
            field_info_path = os.path.join(export_dir, field_info_name)
            with open(field_info_path, "w", encoding="utf-8", newline="") as info_file:
                info_file.write(self.__doc__)


class _Base_Positions_Exporter(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def csv_export_filename(cls) -> str:
        pass

    @classmethod
    @abc.abstractmethod
    def csv_export_labels(cls) -> typing.Tuple[csv_utils.CSV_EXPORT_LABEL_TYPE, ...]:
        pass

    @classmethod
    @abc.abstractmethod
    def dict_export(
        cls, raw_value: csv_utils.CSV_EXPORT_RAW_TYPE, world_index: int
    ) -> dict:
        pass

    def csv_export_write(
        self,
        positions_bisector,
        timestamps,
        export_window,
        export_dir,
        min_confidence_threshold=0.0,
    ):
        export_file = type(self).csv_export_filename()
        export_path = os.path.join(export_dir, export_file)

        export_section = positions_bisector.init_dict_for_window(export_window)
        export_world_idc = pm.find_closest(timestamps, export_section["data_ts"])

        with open(export_path, "w", encoding="utf-8", newline="") as csvfile:
            csv_header = type(self).csv_export_labels()
            dict_writer = csv.DictWriter(csvfile, fieldnames=csv_header)
            dict_writer.writeheader()

            for g, idx in track(
                zip(export_section["data"], export_world_idc),
                description=f"Exporting {export_file}",
                total=len(export_world_idc),
            ):
                if g["confidence"] < min_confidence_threshold:
                    continue
                dict_row = type(self).dict_export(raw_value=g, world_index=idx)
                dict_writer.writerow(dict_row)

        logger.info(f"Created '{export_file}' file.")


class Gaze_Positions_Exporter(_Base_Positions_Exporter):
    @classmethod
    def csv_export_filename(cls) -> str:
        return "gaze_positions.csv"

    @classmethod
    def csv_export_labels(cls) -> typing.Tuple[csv_utils.CSV_EXPORT_LABEL_TYPE, ...]:
        return (
            "gaze_timestamp",
            "world_index",
            "confidence",
            "norm_pos_x",
            "norm_pos_y",
            "base_data",
            "gaze_point_3d_x",
            "gaze_point_3d_y",
            "gaze_point_3d_z",
            "eye_center0_3d_x",
            "eye_center0_3d_y",
            "eye_center0_3d_z",
            "gaze_normal0_x",
            "gaze_normal0_y",
            "gaze_normal0_z",
            "eye_center1_3d_x",
            "eye_center1_3d_y",
            "eye_center1_3d_z",
            "gaze_normal1_x",
            "gaze_normal1_y",
            "gaze_normal1_z",
        )

    @classmethod
    def dict_export(
        cls, raw_value: csv_utils.CSV_EXPORT_RAW_TYPE, world_index: int
    ) -> dict:
        gaze_timestamp = str(raw_value["timestamp"])
        confidence = raw_value["confidence"]
        norm_pos = raw_value["norm_pos"]
        base_data = None
        gaze_points_3d = [None, None, None]
        eye_centers0_3d = [None, None, None]
        eye_centers1_3d = [None, None, None]
        gaze_normals0_3d = [None, None, None]
        gaze_normals1_3d = [None, None, None]

        if raw_value.get("base_data", None) is not None:
            base_data = raw_value["base_data"]
            base_data = " ".join(
                "{}-{}".format(b["timestamp"], b["id"]) for b in base_data
            )

        # add 3d data if avaiblable
        if raw_value.get("gaze_point_3d", None) is not None:
            gaze_points_3d = raw_value["gaze_point_3d"]
            # binocular
            if raw_value.get("eye_centers_3d", None) is not None:
                eye_centers_3d = raw_value["eye_centers_3d"]
                gaze_normals_3d = raw_value["gaze_normals_3d"]

                eye_centers0_3d = (
                    eye_centers_3d.get("0", None)
                    or eye_centers_3d.get(0, None)  # backwards compatibility
                    or [None, None, None]
                )
                eye_centers1_3d = (
                    eye_centers_3d.get("1", None)
                    or eye_centers_3d.get(1, None)  # backwards compatibility
                    or [None, None, None]
                )
                gaze_normals0_3d = (
                    gaze_normals_3d.get("0", None)
                    or gaze_normals_3d.get(0, None)  # backwards compatibility
                    or [None, None, None]
                )
                gaze_normals1_3d = (
                    gaze_normals_3d.get("1", None)
                    or gaze_normals_3d.get(1, None)  # backwards compatibility
                    or [None, None, None]
                )
            # monocular
            elif raw_value.get("eye_center_3d", None) is not None:
                try:
                    eye_id = raw_value["base_data"][0]["id"]
                except (KeyError, IndexError):
                    logger.warning(
                        f"Unexpected raw base_data for monocular gaze!"
                        f" Data: {raw_value.get('base_data', None)}"
                    )
                else:
                    if str(eye_id) == "0":
                        eye_centers0_3d = raw_value["eye_center_3d"]
                        gaze_normals0_3d = raw_value["gaze_normal_3d"]
                    elif str(eye_id) == "1":
                        eye_centers1_3d = raw_value["eye_center_3d"]
                        gaze_normals1_3d = raw_value["gaze_normal_3d"]

        return {
            "gaze_timestamp": gaze_timestamp,
            "world_index": world_index,
            "confidence": confidence,
            "norm_pos_x": norm_pos[0],
            "norm_pos_y": norm_pos[1],
            "base_data": base_data,
            "gaze_point_3d_x": gaze_points_3d[0],
            "gaze_point_3d_y": gaze_points_3d[1],
            "gaze_point_3d_z": gaze_points_3d[2],
            "eye_center0_3d_x": eye_centers0_3d[0],
            "eye_center0_3d_y": eye_centers0_3d[1],
            "eye_center0_3d_z": eye_centers0_3d[2],
            "gaze_normal0_x": gaze_normals0_3d[0],
            "gaze_normal0_y": gaze_normals0_3d[1],
            "gaze_normal0_z": gaze_normals0_3d[2],
            "eye_center1_3d_x": eye_centers1_3d[0],
            "eye_center1_3d_y": eye_centers1_3d[1],
            "eye_center1_3d_z": eye_centers1_3d[2],
            "gaze_normal1_x": gaze_normals1_3d[0],
            "gaze_normal1_y": gaze_normals1_3d[1],
            "gaze_normal1_z": gaze_normals1_3d[2],
        }
