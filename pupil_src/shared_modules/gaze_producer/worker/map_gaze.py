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
import tasklib

from .fake_gpool import FakeGPool

g_pool = None  # set by the plugin


def create_task(gaze_mapper):
    assert g_pool, "You forgot to set g_pool by the plugin"

    mapping_window = pm.exact_window(g_pool.timestamps, gaze_mapper.mapping_index_range)

    fake_gpool = FakeGPool.from_g_pool(g_pool)

    args = (
        fake_gpool,
        mapping_window,
        gaze_mapper.manual_correction_x,
        gaze_mapper.manual_correction_y,
    )
    name = f"Create gaze mapper {gaze_mapper.name}"
    return tasklib.background.create(
        name,
        _map_gaze,
        args=args,
        pass_shared_memory=True,
    )


def time_filter(mapping_window, datum):
    t = datum["timestamp"]
    return t >= mapping_window[0] and t <= mapping_window[1]


def _map_gaze(
    fake_gpool,
    mapping_window,
    manual_correction_x,
    manual_correction_y,
    shared_memory,
):
    raw_gaze_data = fm.load_pldata_file(fake_gpool.rec_dir, "gaze").data
    raw_gaze_data = filter(lambda d: time_filter(mapping_window, d), raw_gaze_data)

    fake_gpool.import_runtime_plugins()

    for gaze_datum in raw_gaze_data:
        result = (gaze_datum["timestamp"], gaze_datum)
        yield [result]
