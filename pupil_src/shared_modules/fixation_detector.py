"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
"""
Fixations general knowledge from literature review
    + Goldberg et al. - fixations rarely < 100ms and range between 200ms and 400ms in
      duration (Irwin, 1992 - fixations dependent on task between 150ms - 600ms)
    + Very short fixations are considered not meaningful for studying behavior
        - eye+brain require time for info to be registered (see Munn et al. APGV, 2008)
    + Fixations are rarely longer than 800ms in duration
        + Smooth Pursuit is exception and different motif
        + If we do not set a maximum duration, we will also detect smooth pursuit (which
          is acceptable since we compensate for VOR)
Terms
    + dispersion (spatial) = how much spatial movement is allowed within one fixation
      (in visual angular degrees or pixels)
    + duration (temporal) = what is the minimum time required for gaze data to be within
      dispersion threshold?
"""

import csv
import logging
import os
from bisect import bisect_left, bisect_right
from pathlib import Path
import multiprocessing as mp
import pandas as pd

import background_helper as bh
import cv2
import data_changed
import file_methods as fm
import msgpack
import numpy as np
import player_methods as pm
from hotkey import Hotkey
from observable import Observable
from plugin import Plugin
from gaze_producer.gaze_from_recording import GazeFromRecording
from pyglui import ui
from progress_reporter import ProgressReporter

from pupil_labs import neon_recording as nr
from pupil_labs.rec_export.explib.fixation_detector.optic_flow_correction import (
    calculate_optic_flow_vectors,
    save_optic_flow_vectors,
    load_optic_flow_vectors,
)
from pupil_recording.info import recording_info_utils

logger = logging.getLogger(__name__)


def fixation_from_data(info, timestamps, world_start_time, frame_size):
    if info["fixation x [px]"] == "" or info["fixation y [px]"] == "":
        norm_pos = None
        pos_2d = None
    else:
        norm_pos = (
            float(info["fixation x [px]"]) / frame_size[0],
            1.0 - float(info["fixation y [px]"]) / frame_size[1]
        )
        pos_2d = [float(info[f"fixation {axis} [px]"]) for axis in "xy"]

    start_time = (float(info["start timestamp [ns]"]) - world_start_time) * 1e-9
    end_time = (float(info["end timestamp [ns]"]) - world_start_time) * 1e-9

    start_frame, end_frame = np.searchsorted(timestamps, [start_time, end_time])
    end_frame = min(end_frame, len(timestamps) - 1)

    start_time_ns = int(info["start timestamp [ns]"])
    end_time_ns = int(info["end timestamp [ns]"])

    datum = {
        "topic": "fixations",
        "norm_pos": norm_pos,
        "timestamp": start_time,
        "duration": (end_time - start_time),
        "id": int(info["fixation id"]),
        "start_frame_index": int(start_frame),
        "end_frame_index": int(end_frame),
        "mid_frame_index": int((start_frame + end_frame) // 2),
        "gaze_point_2d": pos_2d,
        "start timestamp [ns]": start_time_ns,
        "end timestamp [ns]": end_time_ns,
        "duration [ms]": (end_time_ns - start_time_ns) * 1e-6,
    }

    return (datum, start_time, end_time)


def detect_fixations(rec_dir, data_dir, timestamps, frame_size, queue):
    yield "Processing fixations...", ()

    fixations_csv = Path(data_dir) / "fixations.csv"
    saccades_csv = Path(data_dir) / "saccades.csv"
    if not fixations_csv.exists() or not saccades_csv.exists():
        rec = nr.load(Path(rec_dir).parent)
        df = pd.DataFrame({n: rec.fixations[n] for n in rec.fixations.data.dtype.names})
        if len(df) == 0:
            with fixations_csv.open("w") as csvfile:
                csv.writer(csvfile).writerow([
                    "fixation id",
                    "start timestamp [ns]",
                    "end timestamp [ns]",
                    "duration [ms]",
                    "fixation x [px]",
                    "fixation y [px]",
                ])
            with saccades_csv.open("w") as csvfile:
                csv.writer(csvfile).writerow([
                    "saccade id",
                    "start timestamp [ns]",
                    "end timestamp [ns]",
                    "duration [ms]",
                    "amplitude [px]",
                    "amplitude [deg]",
                    "mean velocity [px/s]",
                    "peak velocity [px/s]",
                ])
        else:
            df.rename(columns={
                "start_timestamp_ns": "start timestamp [ns]",
                "end_timestamp_ns": "end timestamp [ns]",

                # fixation data
                "mean_gaze_x": "fixation x [px]",
                "mean_gaze_y": "fixation y [px]",

                # saccade data
                "amplitude_pixels": "amplitude [px]",
                "amplitude_angle_deg": "amplitude [deg]",
                "mean_velocity": "mean velocity [px/s]",
                "max_velocity": "peak velocity [px/s]",
            }, inplace=True)

            df["duration [ms]"] = (df["end timestamp [ns]"] - df["start timestamp [ns]"]) * 1e-6

            fixations_df = df[df["event_type"] == 1].copy()
            fixations_df["fixation id"] = range(len(fixations_df))
            fixations_df["fixation id"] += 1
            fixations_df.drop(columns=["ts", "event_type"], inplace=True)
            fixations_df.to_csv(fixations_csv, index=False)

            saccades_df = df[df["event_type"] == 0].copy()
            saccades_df["saccade id"] = range(len(saccades_df))
            saccades_df["saccade id"] += 1
            saccades_df.drop(columns=["ts", "event_type"], inplace=True)
            saccades_df.to_csv(saccades_csv, index=False)

    info_json = recording_info_utils.read_neon_info_file(str(Path(rec_dir).parent))
    start_time_synced_ns = int(info_json["start_time"])

    with fixations_csv.open() as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            fixation = fixation_from_data(row, timestamps, start_time_synced_ns, frame_size)
            yield f"Processing fixations... {idx}", fixation

    if not (Path(rec_dir) / "offline_data" / "optic_flow_vectors.npz").exists():
        with ProgressReporter(queue) as progress:
            optic_flow_vectors = calculate_optic_flow_vectors(Path(rec_dir).parent, progress)
            save_optic_flow_vectors(data_dir, optic_flow_vectors)

    return "Fixation processing complete", ()


class Fixation_Detector(Observable, Plugin):
    """Fixation Detector
    """

    CACHE_VERSION = 1
    icon_chr = chr(0xEC03)
    icon_font = "pupil_icons"

    class VersionMismatchError(ValueError):
        pass

    class ConfigMismatchError(ValueError):
        pass

    class DataMismatchError(ValueError):
        pass

    def __init__(
        self,
        g_pool,
        show_fixations=True,
        radius=25.0, color=(1.0, 1.0, 0.0, 1.0), thickness=3, fill=False
    ):
        super().__init__(g_pool)
        self.show_fixations = show_fixations
        self.current_fixation_details = None
        self.fixation_data = []
        self.prev_index = -1
        self.bg_task = None
        self.optic_flow_vectors = None
        self.status = "Please wait..."
        self.data_dir = os.path.join(g_pool.rec_dir, "offline_data")
        self._gaze_changed_listener = data_changed.Listener(
            "gaze_positions", g_pool.rec_dir, plugin=self
        )
        self._gaze_changed_listener.add_observer("on_data_changed", self._classify)

        self._fixations_changed_announcer = data_changed.Announcer(
            "fixations", g_pool.rec_dir, plugin=self
        )
        try:
            self.load_offline_data()
        except (
            FileNotFoundError,
            self.VersionMismatchError,
            self.ConfigMismatchError,
            self.DataMismatchError,
        ) as err:
            logger.debug(f"Offline data not loaded: {err} ({type(err).__name__})")
            self.notify_all(
                {"subject": "fixation_detector.should_recalculate", "delay": 0.5}
            )

        self.r = color[0]
        self.g = color[1]
        self.b = color[2]
        self.a = color[3]
        self.radius = radius
        self.thickness = thickness
        self.fill = fill

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Fixation Detector"

        def jump_next_fixation(_):
            cur_idx = self.last_frame_idx
            all_idc = [f["mid_frame_index"] for f in self.g_pool.fixations]
            if not all_idc:
                logger.warning("No fixations available")
                return
            # wrap-around index
            tar_fix = bisect_right(all_idc, cur_idx) % len(all_idc)
            self.notify_all(
                {
                    "subject": "seek_control.should_seek",
                    "index": int(self.g_pool.fixations[tar_fix]["mid_frame_index"]),
                }
            )

        def jump_prev_fixation(_):
            cur_idx = self.last_frame_idx
            all_idc = [f["mid_frame_index"] for f in self.g_pool.fixations]
            if not all_idc:
                logger.warning("No fixations available")
                return
            # wrap-around index
            tar_fix = (bisect_left(all_idc, cur_idx) - 1) % len(all_idc)
            self.notify_all(
                {
                    "subject": "seek_control.should_seek",
                    "index": int(self.g_pool.fixations[tar_fix]["mid_frame_index"]),
                }
            )

        for help_block in self.__doc__.split("\n\n"):
            help_str = help_block.replace("\n", " ").replace("  ", "").strip()
            self.menu.append(ui.Info_Text(help_str))

        self.menu.append(
            ui.Info_Text(
                "To start the export, wait until the detection has finished and press "
                "the export button or type \"e\"."
            )
        )

        self.menu.append(
            ui.Text_Input(
                "status", self, label="Detection progress:", setter=lambda x: None
            )
        )
        self.menu.append(ui.Switch("show_fixations", self, label="Show fixations"))

        self.next_fix_button = ui.Thumb(
            "jump_next_fixation",
            setter=jump_next_fixation,
            getter=lambda: False,
            label=chr(0xE044),
            hotkey=Hotkey.FIXATION_NEXT_PLAYER_HOTKEY(),
            label_font="pupil_icons",
        )
        self.next_fix_button.status_text = "Next Fixation"
        self.g_pool.quickbar.append(self.next_fix_button)

        self.prev_fix_button = ui.Thumb(
            "jump_prev_fixation",
            setter=jump_prev_fixation,
            getter=lambda: False,
            label=chr(0xE045),
            hotkey=Hotkey.FIXATION_PREV_PLAYER_HOTKEY(),
            label_font="pupil_icons",
        )
        self.prev_fix_button.status_text = "Previous Fixation"
        self.g_pool.quickbar.append(self.prev_fix_button)

        self.menu.append(
            ui.Slider("radius", self, min=1, step=1, max=100, label="Radius")
        )
        self.menu.append(
            ui.Slider("thickness", self, min=1, step=1, max=15, label="Stroke width")
        )
        self.menu.append(ui.Switch("fill", self, label="Fill"))

        color_menu = ui.Growing_Menu("Color")
        color_menu.append(
            ui.Info_Text("Set RGB color components and alpha (opacity) values.")
        )
        color_menu.append(
            ui.Slider("r", self, min=0.0, step=0.05, max=1.0, label="Red")
        )
        color_menu.append(
            ui.Slider("g", self, min=0.0, step=0.05, max=1.0, label="Green")
        )
        color_menu.append(
            ui.Slider("b", self, min=0.0, step=0.05, max=1.0, label="Blue")
        )
        color_menu.append(
            ui.Slider("a", self, min=0.0, step=0.05, max=1.0, label="Alpha")
        )
        self.menu.append(color_menu)

        self.current_fixation_details = ui.Info_Text("")
        self.menu.append(self.current_fixation_details)

    def deinit_ui(self):
        self.remove_menu()
        self.current_fixation_details = None
        self.g_pool.quickbar.remove(self.next_fix_button)
        self.g_pool.quickbar.remove(self.prev_fix_button)
        self.next_fix_button = None
        self.prev_fix_button = None

    def cleanup(self):
        if self.bg_task:
            self.bg_task.cancel()
            self.bg_task = None

    def get_init_dict(self):
        return {
            "show_fixations": self.show_fixations,
            "radius": self.radius,
            "color": (self.r, self.g, self.b, self.a),
            "thickness": self.thickness,
            "fill": self.fill,
        }

    def on_notify(self, notification):
        if notification["subject"] == "fixation_detector.should_recalculate":
            self._classify()

        elif notification["subject"] == "should_export":
            self.export_fixations(notification["ts_window"], notification["export_dir"])
            self.export_saccades(notification["ts_window"], notification["export_dir"])

    def _classify(self):
        """
        classify fixations
        """
        if self.g_pool.app == "exporter":
            return

        if self.bg_task:
            self.bg_task.cancel()

        self.fixation_data = []
        self.fixation_start_ts = []
        self.fixation_stop_ts = []
        self.mp_queue = mp.Queue()
        args = (
            self.g_pool.rec_dir,
            self.data_dir,
            self.g_pool.timestamps,
            self.g_pool.capture.video_stream.frame_size,
            self.mp_queue
        )
        self.bg_task = bh.IPC_Logging_Task_Proxy(
            "Fixation processing", detect_fixations, args=args
        )
        self.publish_empty()

    def recent_events(self, events):
        if self.bg_task:
            while not self.mp_queue.empty():
                current_progress = self.mp_queue.get_nowait()
                self.menu_icon.indicator_stop = current_progress
                self.status = f"Processing fixations ({round(100*current_progress)}%)"

            for progress, fixation_result in self.bg_task.fetch():
                self.status = progress
                if fixation_result:
                    datum, start_ts, stop_ts = fixation_result
                    serialized = msgpack.packb(datum, use_bin_type=True, default=fm.Serialized_Dict.packing_hook)

                    self.fixation_data.append(fm.Serialized_Dict(msgpack_bytes=serialized))
                    self.fixation_start_ts.append(start_ts)
                    self.fixation_stop_ts.append(stop_ts)

            if self.bg_task.completed:
                self.status = f"{len(self.fixation_data)} fixations detected"
                self.correlate_and_publish_new()
                self.bg_task = None
                self.menu_icon.indicator_stop = 0.0

        frame = events.get("frame")
        if not frame:
            return

        self.last_frame_idx = frame.index
        frame_window = pm.enclosing_window(self.g_pool.timestamps, frame.index)
        fixations = self.g_pool.fixations.by_ts_window(frame_window)
        events["fixations"] = fixations
        if self.show_fixations:
            manual_correction = [0, 0]
            for plugin in self.g_pool.plugins:
                if isinstance(plugin, GazeFromRecording):
                    manual_correction = plugin.get_manual_correction_for_frame(frame.index)
                    break

            for f in fixations:
                if f["norm_pos"] is None:
                    continue

                # calculate cumulative offset to current timestamp
                optic_flow_offset = [0, 0]
                start_offset = (0, 0)

                if self.optic_flow_vectors is None:
                    self.optic_flow_vectors = load_optic_flow_vectors(self.data_dir)

                gaze_points = self.g_pool.gaze_positions.by_ts_window((f["timestamp"], f["timestamp"]+1/30))
                first_gaze_point = f
                for gp in gaze_points:
                    if gp["norm_pos"] is not None:
                        first_gaze_point = gp
                        break

                start_offset = (
                    f["norm_pos"][0] - first_gaze_point["norm_pos"][0],
                    f["norm_pos"][1] - first_gaze_point["norm_pos"][1],
                )

                for frame_idx in range(f["start_frame_index"], frame.index):
                    optic_flow_idx = frame_idx - 1
                    optic_frame_duration = self.optic_flow_vectors.ts[optic_flow_idx] - self.optic_flow_vectors.ts[optic_flow_idx-1]
                    optic_flow_offset[0] += self.optic_flow_vectors.x[optic_flow_idx] * optic_frame_duration
                    optic_flow_offset[1] += self.optic_flow_vectors.y[optic_flow_idx] * optic_frame_duration

                x = int((f["norm_pos"][0] - start_offset[0]) * frame.width + optic_flow_offset[0])
                y = int((1.0 - f["norm_pos"][1] + start_offset[1]) * frame.height + optic_flow_offset[1])

                x += int(manual_correction[0] * frame.width)
                y -= int(manual_correction[1] * frame.height)

                if self.fill:
                    thickness = -1
                else:
                    thickness = self.thickness

                pm.transparent_circle(
                    frame.img,
                    (x, y),
                    radius=self.radius,
                    color=(self.b, self.g, self.r, self.a),
                    thickness=thickness,
                )

                cv2.putText(
                    frame.img,
                    "{}".format(f["id"]),
                    (x + 30, y),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8,
                    (255, 150, 100),
                )

        if self.current_fixation_details and self.prev_index != frame.index:
            info = ""
            for f in fixations:
                info += "Fixation #{} of {}\n".format(
                    f["id"], len(self.g_pool.fixations)
                )
                info += "    Duration: {:.2f} milliseconds\n".format(f["duration [ms]"])
                info += "    Frame range: {}-{}\n".format(
                    f["start_frame_index"] + 1, f["end_frame_index"] + 1
                )
                if f["id"] > 1:
                    prev_f = self.g_pool.fixations[f["id"] - 2]
                    time_lapsed = (
                        f["timestamp"] - prev_f["timestamp"] + prev_f["duration"] / 1000
                    )
                    info += "    Time since prev. fixation: {:.2f} seconds\n".format(
                        time_lapsed
                    )
                else:
                    info += "    Time since prev. fixation: N/A\n"

                if f["id"] < len(self.g_pool.fixations):
                    next_f = self.g_pool.fixations[f["id"]]
                    time_lapsed = (
                        next_f["timestamp"] - f["timestamp"] + f["duration"] / 1000
                    )
                    info += "    Time to next fixation: {:.2f} seconds\n ".format(
                        time_lapsed
                    )
                else:
                    info += "    Time to next fixation: N/A\n"

            self.current_fixation_details.text = info
            self.prev_index = frame.index

    def correlate_and_publish_new(self):
        self.g_pool.fixations = pm.Affiliator(
            self.fixation_data, self.fixation_start_ts, self.fixation_stop_ts
        )
        self._fixations_changed_announcer.announce_new(
            delay=0.3,
            token_data=(
                self._gaze_changed_listener._current_token,
            ),
        )
        self.save_offline_data()

    def publish_empty(self):
        self.g_pool.fixations = pm.Affiliator([], [], [])
        self._fixations_changed_announcer.announce_new(token_data=())

    def correlate_and_publish_existing(self):
        self.g_pool.fixations = pm.Affiliator(
            self.fixation_data, self.fixation_start_ts, self.fixation_stop_ts
        )
        self._fixations_changed_announcer.announce_existing()
        self.status = f"{len(self.fixation_data)} fixations detected"

    def save_offline_data(self):
        with fm.PLData_Writer(self.data_dir, "fixations") as writer:
            for timestamp, datum in zip(self.fixation_start_ts, self.fixation_data):
                writer.append_serialized(timestamp, "fixation", datum.serialized)
        path_stop_ts = os.path.join(self.data_dir, "fixations_stop_timestamps.npy")
        np.save(path_stop_ts, self.fixation_stop_ts)
        path_meta = os.path.join(self.data_dir, "fixations.meta")
        fm.save_object(
            {
                "version": self.CACHE_VERSION,
                "config": self._cache_config(),
            },
            path_meta,
        )

    def load_offline_data(self):
        if not os.path.isfile(os.path.join(self.data_dir, "fixations.pldata")):
            raise FileNotFoundError()

        path_stop_ts = os.path.join(self.data_dir, "fixations_stop_timestamps.npy")
        fixation_stop_ts = np.load(path_stop_ts)
        path_meta = os.path.join(self.data_dir, "fixations.meta")
        meta = fm.load_object(path_meta)
        version_loaded = meta.get("version", -1)
        if version_loaded != self.CACHE_VERSION:
            raise self.VersionMismatchError(
                f"Expected version {self.CACHE_VERSION}, got {version_loaded}"
            )
        config_loaded = meta.get("config", None)
        config_expected = self._cache_config()
        if config_loaded != config_expected:
            raise self.ConfigMismatchError(
                f"Expected config {config_expected}, got {config_loaded}"
            )
        fixations = fm.load_pldata_file(self.data_dir, "fixations")
        if not (
            len(fixations.data) == len(fixations.timestamps) == len(fixation_stop_ts)
        ):
            raise self.DataMismatchError(
                f"Data inconsistent:\n"
                f"\tlen(fixations.data)={len(fixations.data)}\n"
                f"\tlen(fixations.timestamps)={len(fixations.timestamps)}\n"
                f"\tlen(fixation_stop_ts)={len(fixation_stop_ts)}"
            )
        self.fixation_data = fixations.data
        self.fixation_start_ts = fixations.timestamps
        self.fixation_stop_ts = fixation_stop_ts
        self.correlate_and_publish_existing()

    def _cache_config(self):
        return {}

    @classmethod
    def csv_representation_keys(self):
        return (
            "fixation id",
            "start timestamp [ns]",
            "end timestamp [ns]",
            "duration [ms]",
            "fixation x [px]",
            "fixation y [px]",
        )

    @classmethod
    def csv_representation_for_fixation(self, fixation):
        return (
            fixation["id"],
            fixation["start timestamp [ns]"],
            fixation["end timestamp [ns]"],
            fixation["duration [ms]"],
            *fixation["gaze_point_2d"]
        )

    def export_fixations(self, export_window, export_dir):
        """
        between in and out mark

            fixation report:
                - fixation detection method and parameters
                - fixation count

            fixation list:
                fixation id | start timestamp [ns] | end timestamp [ns] | duration [ms] |
                fixation x [px] | fixation y [px]
        """
        if not self.fixation_data:
            logger.warning("No fixations in this recording nothing to export")
            return

        if export_window[0] == self.g_pool.timestamps[0]:
            # If a fixation starts and ends before the first world frame, this will ensure its included
            export_window[0] = 0

        fixations_in_section = self.g_pool.fixations.by_ts_window(export_window)
        gaze_offset_plugin = None
        for plugin in self.g_pool.plugins:
            if isinstance(plugin, GazeFromRecording):
                gaze_offset_plugin = plugin
                break

        with open(
            os.path.join(export_dir, "fixations.csv"), "w", encoding="utf-8", newline=""
        ) as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(self.csv_representation_keys())
            for f in fixations_in_section:

                if gaze_offset_plugin is not None and f["norm_pos"] is not None:
                    f = f.copy()
                    manual_correction = gaze_offset_plugin.get_manual_correction_for_frame(f["start_frame_index"])
                    f["norm_pos"] = (
                        f["norm_pos"][0] + manual_correction[0],
                        f["norm_pos"][1] + manual_correction[1],
                    )
                    f["gaze_point_2d"] = (
                        f["gaze_point_2d"][0] + manual_correction[0] * self.g_pool.capture.frame_size[0],
                        f["gaze_point_2d"][1] - manual_correction[1] * self.g_pool.capture.frame_size[1],
                    )

                csv_writer.writerow(self.csv_representation_for_fixation(f))
            logger.info("Created \"fixations.csv\" file.")

    def export_saccades(self, export_window, export_dir):
        export_window[0] = max(export_window[0], self.g_pool.timestamps[0])
        export_window_ns = [self.g_pool.capture.ts_to_ns(v) for v in export_window]

        input_file_path = os.path.join(self.data_dir, "saccades.csv")
        with open(input_file_path, mode="r") as input_csv_file:
            csv_reader = csv.DictReader(input_csv_file)

            with open(
                os.path.join(export_dir, "saccades.csv"), "w", encoding="utf-8", newline=""
            ) as csvfile:
                csv_writer = csv.DictWriter(csvfile, extrasaction="ignore", fieldnames=[
                    "saccade id",
                    "start timestamp [ns]",
                    "end timestamp [ns]",
                    "duration [ms]",
                    "amplitude [px]",
                    "amplitude [deg]",
                    "mean velocity [px/s]",
                    "peak velocity [px/s]",
                ])
                csv_writer.writeheader()

                for saccade in csv_reader:
                    if float(saccade["end timestamp [ns]"]) < export_window_ns[0]:
                        continue
                    if float(saccade["start timestamp [ns]"]) > export_window_ns[1]:
                        break

                    csv_writer.writerow(saccade)

                logger.info("Created \"saccades.csv\" file.")
