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

import data_changed
import player_methods as pm
from observable import Observable
from plugin import System_Plugin_Base


class GazeProducerBase(Observable, System_Plugin_Base):
    uniqueness = "by_base_class"
    order = 0.02
    icon_chr = chr(0xEC14)
    icon_font = "pupil_icons"

    @classmethod
    @abc.abstractmethod
    def plugin_menu_label(cls) -> str:
        raise NotImplementedError()

    @classmethod
    def gaze_data_source_selection_label(cls) -> str:
        return cls.plugin_menu_label()

    @classmethod
    def gaze_data_source_selection_order(cls) -> float:
        return float("inf")

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self._gaze_changed_announcer = data_changed.Announcer(
            "gaze_positions", g_pool.rec_dir, plugin=self
        )

    def init_ui(self):
        self.add_menu()
        self.menu.label = self.plugin_menu_label()
        self.menu_icon.order = 0.3
        self.menu_icon.tooltip = "Gaze Data"

    def deinit_ui(self):
        self.remove_menu()

    def recent_events(self, events):
        # TODO: comments or method extraction
        if "frame" in events:
            frame_idx = events["frame"].index
            window = pm.enclosing_window(self.g_pool.timestamps, frame_idx)
            events["gaze"] = self.g_pool.gaze_positions.by_ts_window(window)
