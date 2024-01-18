"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from gaze_producer import controller, model
from gaze_producer import ui as plugin_ui
from gaze_producer.gaze_producer_base import GazeProducerBase
from plugin_timeline import PluginTimeline
from tasklib.manager import UniqueTaskManager

import data_changed
import file_methods as fm
import player_methods as pm

class GazeFromRecording(GazeProducerBase):
    @classmethod
    def plugin_menu_label(cls) -> str:
        return "Gaze Offset Correction"

    def __init__(self, g_pool):
        super().__init__(g_pool)

        self.inject_plugin_dependencies()
        self._task_manager = UniqueTaskManager(plugin=self)
        self._correction_changed_announcer = data_changed.Announcer(
            "gaze_positions", g_pool.rec_dir, plugin=self
        )

        self._setup_storages()
        self._setup_controllers()
        self._setup_ui()

        if self._gaze_mapper_menu.allow_multiple:
            self._setup_timelines()

    def _setup_timelines(self):
        self._plugin_timeline = PluginTimeline(
            plugin=self,
            title="Gaze Mappers",
            timeline_ui_parent=self.g_pool.user_timelines,
            all_timestamps=self.g_pool.timestamps,
        )
        self._gaze_mapper_timeline = plugin_ui.GazeMapperTimeline(
            self._gaze_mapper_storage,
            self._gaze_mapper_controller,
        )
        self._gaze_mapper_timeline.render_parent_timeline = self._refresh_timeline

    def inject_plugin_dependencies(self):
        from gaze_producer.worker import map_gaze
        map_gaze.g_pool = self.g_pool

    def init_ui(self):
        super().init_ui()
        self._gaze_mapper_menu.render()
        self.menu.extend(self._gaze_mapper_menu.menu)

        if self._gaze_mapper_menu.allow_multiple:
            self._refresh_timeline()

    def _refresh_timeline(self):
        self._plugin_timeline.clear_rows()
        for row in self._gaze_mapper_timeline.create_rows():
            self._plugin_timeline.add_row(row)
        self._plugin_timeline.refresh()

    def _publish_gaze(self, gaze_bisector):
        self.g_pool.gaze_positions = gaze_bisector
        self._gaze_changed_announcer.announce_new(delay=1)

    def _setup_storages(self):
        self._gaze_mapper_storage = model.GazeMapperStorage(
            rec_dir=self.g_pool.rec_dir,
            get_recording_index_range=self._recording_index_range,
        )

    def cleanup(self):
        super().cleanup()
        self._gaze_mapper_storage.save_to_disk()

    def _setup_controllers(self):
        self._gaze_mapper_controller = controller.GazeMapperController(
            self._gaze_mapper_storage,
            task_manager=self._task_manager,
            get_current_trim_mark_range=self._current_trim_mark_range,
            publish_gaze_bisector=self._publish_gaze,
        )

    def _setup_ui(self):
        self._gaze_mapper_menu = plugin_ui.GazeMapperMenu(
            self._gaze_mapper_controller,
            self._gaze_mapper_storage,
            index_range_as_str=self._index_range_as_str,
            correction_changed_announcer=self._correction_changed_announcer
        )

    def _recording_index_range(self):
        left_index = 0
        right_index = len(self.g_pool.timestamps) - 1
        return left_index, right_index

    def _current_trim_mark_range(self):
        right_idx = self.g_pool.seek_control.trim_right
        left_idx = self.g_pool.seek_control.trim_left
        return left_idx, right_idx

    def _index_range_as_str(self, index_range):
        from_index, to_index = index_range
        return (
            f"{self._index_time_as_str(from_index)} - "
            f"{self._index_time_as_str(to_index)}"
        )

    def _index_time_as_str(self, index):
        ts = self.g_pool.timestamps[index]
        min_ts = self.g_pool.timestamps[0]
        time = ts - min_ts
        minutes = abs(time // 60)  # abs because it's sometimes -0
        seconds = round(time % 60)
        return f"{minutes:02.0f}:{seconds:02.0f}"

    def get_manual_correction_for_frame(self, frame_idx):
        for mapper in self._gaze_mapper_storage:
            if frame_idx >= mapper.mapping_index_range[0] and frame_idx < mapper.mapping_index_range[1]:
                return (mapper.manual_correction_x, mapper.manual_correction_y)
        return (0, 0)

    def get_manual_correction_for_ts(self, ts):
        for mapper in self._gaze_mapper_storage:
            if ts >= mapper.gaze_ts[0] and ts < mapper.gaze_ts[-1]:
                return (mapper.manual_correction_x, mapper.manual_correction_y)
        return (0, 0)

    def on_notify(self, notification):
        if notification["subject"] == "blinks_changed":
            gaze_data = [datum.copy() for datum in self.g_pool.gaze_positions.data]
            gaze_ts = self.g_pool.gaze_positions.data_ts

            for blink in self.g_pool.blinks:
                window = (blink["start_timestamp"], blink["end_timestamp"], )
                gaze_idx_range = self.g_pool.gaze_positions._start_stop_idc_for_window(window)
                for gaze_idx in range(*gaze_idx_range):
                    gaze_data[gaze_idx]["confidence"] = 0.0

            gaze_data = [fm.Serialized_Dict(gaze) for gaze in gaze_data]
            self._publish_gaze(pm.Bisector(gaze_data, gaze_ts))
