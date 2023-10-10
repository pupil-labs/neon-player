"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from storage import StorageItem


class GazeMapper(StorageItem):
    version = 1

    def __init__(
        self,
        unique_id,
        name,
        mapping_index_range,
        manual_correction_x=0.0,
        manual_correction_y=0.0,
        activate_gaze=True,
        status="Not calculated yet",
        gaze=None,
        gaze_ts=None,
    ):
        self.unique_id = unique_id
        self.name = name
        self.mapping_index_range = tuple(mapping_index_range)
        self.manual_correction_x = manual_correction_x
        self.manual_correction_y = manual_correction_y
        self.activate_gaze = activate_gaze
        self.status = status
        self.gaze = gaze if gaze is not None else []
        self.gaze_ts = gaze_ts if gaze_ts is not None else []

    def empty(self):
        return len(self.gaze) == 0 and len(self.gaze_ts) == 0

    @staticmethod
    def from_tuple(tuple_):
        return GazeMapper(*tuple_)

    @property
    def as_tuple(self):
        return (
            self.unique_id,
            self.name,
            self.mapping_index_range,
            self.manual_correction_x,
            self.manual_correction_y,
            self.activate_gaze,
            self.status,
        )
