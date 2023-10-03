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

import time
import csv
import enum
import logging
import os
from bisect import bisect_left, bisect_right
from collections import deque
from pathlib import Path

import background_helper as bh
import cv2
import data_changed
import file_methods as fm
import msgpack
import numpy as np
import player_methods as pm
from hotkey import Hotkey
from methods import denormalize
from observable import Observable
from plugin import Plugin
from pupil_recording import PupilRecording, RecordingInfo
from pyglui import ui
from pyglui.cygl.utils import RGBA, draw_circle
from pyglui.pyfontstash import fontstash
from scipy.spatial.distance import pdist

from pupil_labs.rec_export.explib.fixation_detector.neon import detect_fixations_neon
from pupil_labs.rec_export.export import *
from pupil_recording.info import recording_info_utils

logger = logging.getLogger(__name__)

NS_TO_S = 1e-9

class FixationDetectionMethod(enum.Enum):
    GAZE_2D = "2d gaze"
    GAZE_3D = "3d gaze"


class Fixation_Detector_Base(Plugin):
    icon_chr = chr(0xEC03)
    icon_font = "pupil_icons"

    @classmethod
    def parse_pretty_class_name(cls) -> str:
        return "Fixation Detector"

def fixation_from_data(info, timestamps, world_start_time, frame_size):
    norm_pos = (
        float(info['fixation x [px]']) / frame_size[0],
        1.0 - float(info['fixation y [px]']) / frame_size[1]
    )

    start_time = (float(info['start timestamp [ns]']) - world_start_time) * NS_TO_S
    end_time = (float(info['end timestamp [ns]']) - world_start_time) * NS_TO_S

    start_frame, end_frame = np.searchsorted(timestamps, [start_time, end_time])
    end_frame = min(end_frame, len(timestamps) - 1)

    datum = {
        "topic": "fixations",
        "norm_pos": norm_pos,
        "timestamp": start_time,
        "duration": (end_time - start_time),
        "id": int(info["fixation id"]),
        "start_frame_index": int(start_frame),
        "end_frame_index": int(end_frame),
        "mid_frame_index": int((start_frame + end_frame) // 2),
        "gaze_point_2d": [info[f'fixation {axis} [px]'] for axis in 'xy'],
    }

    serialization_hook = fm.Serialized_Dict.packing_hook
    datum = msgpack.packb(
        datum, use_bin_type=True, default=serialization_hook
    )

    return (datum, start_time, end_time)


def detect_fixations(rec_dir, data_dir, timestamps, frame_size):
    yield "Detecting fixations...", ()

    progress = Progress(disable=True)
    process_recording(Path(rec_dir).parent, data_dir, force=True, blinks=False, fixations=True, progress=progress)
    while not progress.finished:
        time.sleep(1.0)

    fixations_csv = Path(data_dir) / 'fixations.csv'
    while not fixations_csv.exists():
        # @TODO: is there a better way to check for completion than polling for the file?
        time.sleep(1.0)

    info_json = recording_info_utils.read_neon_info_file(str(Path(rec_dir).parent))
    start_time_synced_ns = int(info_json["start_time"])

    with fixations_csv.open() as csvfile:
        reader = csv.DictReader(csvfile)
        for idx,row in enumerate(reader):
            if '' in row.values():
                continue

            fixation = fixation_from_data(row, timestamps, start_time_synced_ns, frame_size)
            yield f"Processing fixations... {idx}", fixation

    return "Fixation detection complete", ()



class Offline_Fixation_Detector(Observable, Fixation_Detector_Base):
    """pl-rec-export
    """

    CACHE_VERSION = 1

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
    ):
        super().__init__(g_pool)
        self.show_fixations = show_fixations
        self.current_fixation_details = None
        self.fixation_data = []
        self.prev_index = -1
        self.bg_task = None
        self.status = ""
        self.data_dir = os.path.join(g_pool.rec_dir, "offline_data")
        self._gaze_changed_listener = data_changed.Listener( # only need ths for the token?
            "gaze_positions", g_pool.rec_dir, plugin=self
        )
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
                "the export button or type 'e'."
            )
        )

        self.menu.append(
            ui.Text_Input(
                "status", self, label="Detection progress:", setter=lambda x: None
            )
        )
        self.menu.append(ui.Switch("show_fixations", self, label="Show fixations"))
        self.current_fixation_details = ui.Info_Text("")
        self.menu.append(self.current_fixation_details)

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
        }

    def on_notify(self, notification):
        if notification["subject"] == "fixation_detector.should_recalculate":
            self._classify()
        elif notification["subject"] == "should_export":
            self.export_fixations(notification["ts_window"], notification["export_dir"])

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
        args = (
            self.g_pool.rec_dir,
            self.data_dir,
            self.g_pool.timestamps,
            self.g_pool.capture.video_stream.frame_size
        )
        self.bg_task = bh.IPC_Logging_Task_Proxy(
            "Fixation detection", detect_fixations, args=args
        )
        self.publish_empty()

    def recent_events(self, events):
        if self.bg_task:
            for progress, fixation_result in self.bg_task.fetch():
                self.status = progress
                if fixation_result:
                    serialized, start_ts, stop_ts = fixation_result
                    self.fixation_data.append(
                        fm.Serialized_Dict(msgpack_bytes=serialized)
                    )
                    self.fixation_start_ts.append(start_ts)
                    self.fixation_stop_ts.append(stop_ts)

                if self.fixation_data:
                    current_ts = self.fixation_stop_ts[-1]
                    progress = (current_ts - self.g_pool.timestamps[0]) / (
                        self.g_pool.timestamps[-1] - self.g_pool.timestamps[0]
                    )
                    self.menu_icon.indicator_stop = progress
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
            for f in fixations:
                x = int(f["norm_pos"][0] * frame.width)
                y = int((1.0 - f["norm_pos"][1]) * frame.height)
                pm.transparent_circle(
                    frame.img,
                    (x, y),
                    radius=25.0,
                    color=(0.0, 1.0, 1.0, 1.0),
                    thickness=3,
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
                info += "Current fixation, {} of {}\n".format(
                    f["id"], len(self.g_pool.fixations)
                )
                info += "    Duration: {:.2f} milliseconds\n".format(f["duration"])
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
                    info += "    Time to next fixation: {:.2f} seconds\n".format(
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
            "id",
            "start_timestamp",
            "duration",
            "start_frame_index",
            "end_frame_index",
            "norm_pos_x",
            "norm_pos_y",
            "gaze_point_2d_x",
            "gaze_point_2d_y",
        )

    @classmethod
    def csv_representation_for_fixation(self, fixation):
        return (
            fixation["id"],
            fixation["timestamp"],
            fixation["duration"],
            fixation["start_frame_index"],
            fixation["end_frame_index"],
            *fixation["norm_pos"],
            *fixation["gaze_point_2d"],
        )

    def export_fixations(self, export_window, export_dir):
        """
        between in and out mark

            fixation report:
                - fixation detection method and parameters
                - fixation count

            fixation list:
                id | start_timestamp | duration | start_frame_index | end_frame_index |
                norm_pos_x | norm_pos_y | method | gaze_point_2d_x | gaze_point_2d_y
        """
        if not self.fixation_data:
            logger.warning("No fixations in this recording nothing to export")
            return

        fixations_in_section = self.g_pool.fixations.by_ts_window(export_window)

        with open(
            os.path.join(export_dir, "fixations.csv"), "w", encoding="utf-8", newline=""
        ) as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(self.csv_representation_keys())
            for f in fixations_in_section:
                csv_writer.writerow(self.csv_representation_for_fixation(f))
            logger.info("Created 'fixations.csv' file.")
