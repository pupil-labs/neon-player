"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import file_methods as fm
import player_methods as pm
from gaze_producer.gaze_producer_base import GazeProducerBase


class GazeFromRecording(GazeProducerBase):
    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.g_pool.gaze_positions = self._load_gaze_data()
        self._gaze_changed_announcer.announce_existing()

    def _load_gaze_data(self):
        gaze = fm.load_pldata_file(self.g_pool.rec_dir, "gaze")
        return pm.Bisector(gaze.data, gaze.timestamps)
