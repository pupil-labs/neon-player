import numpy as np

from pupil_labs.neon_player.plugins.shared.video_export import _prepare_timestamps

from tests.mocks import mock_scene_timeseries


def test_prepare_timestamps_integer(mock_neon_recording):
    scene = mock_scene_timeseries([10, 20, 30, 40, 50])
    rec = mock_neon_recording(
        start_time=1,
        stop_time=100,
        scene=scene
    )
    assert _prepare_timestamps(rec, (1, 100)).dtype == np.int64
