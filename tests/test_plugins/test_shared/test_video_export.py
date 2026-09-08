import numpy as np
import pytest

from pupil_labs.neon_player.plugins.shared.video_export import _prepare_timestamps

from tests.mocks import mock_scene_timeseries


@pytest.mark.parametrize(
    "export_window", [(0, 100), (10, 100), (50, 100)]
)
def test_prepare_timestamps__export_window(export_window, mock_neon_recording):
    scene = mock_scene_timeseries(np.arange(100, dtype=np.int64))
    rec = mock_neon_recording(
        start_time=0,
        stop_time=100,
        scene=scene
    )

    export_timestamps = _prepare_timestamps(rec, export_window)
    assert export_timestamps.dtype == np.int64, \
        "Expected timestamps to be integer"

    start_ts, end_ts = export_window
    assert np.all(start_ts <= export_timestamps), \
        "Expected timestamps to be within the export window"
    assert np.all(export_timestamps <= end_ts), \
        "Expected timestamps to be within the export window"


@pytest.mark.parametrize(
    "export_window", [(0, 15), (85, 100)]
)
def test_prepare_timestamps__grey_filler_frames(export_window, mock_neon_recording):
    # Scene timestamps do not cover the whole range between start and stop time
    scene = mock_scene_timeseries(np.arange(20, 80, dtype=np.int64))
    rec = mock_neon_recording(
        start_time=0,
        stop_time=100,
        scene=scene
    )

    export_timestamps = _prepare_timestamps(rec, export_window, fps=1e9)
    assert export_timestamps.size, \
        "Expected timestamps to be filled in gaps"
    assert np.all(np.diff(export_timestamps) == 1), \
        "Expected timestamps in gaps to match the provided fps"
    assert export_timestamps.dtype == np.int64, \
        "Expected timestamps to be integer"

    start_ts, end_ts = export_window
    assert np.all(start_ts <= export_timestamps), \
        "Expected timestamps to be within the export window"
    assert np.all(export_timestamps <= end_ts), \
        "Expected timestamps to be within the export window"


def test_prepare_timestamps__gaps(mock_neon_recording):
    # Simulate gaps in scene video
    # NOTE: gaps have to be at least 0.5 s, using 20 fps for nicer numbers
    scene_timestamps = np.concatenate([
        np.arange(20, 40, dtype=np.int64) * 1e9 // 20,
        np.arange(70, 90, dtype=np.int64) * 1e9 // 20,
    ])
    scene = mock_scene_timeseries(scene_timestamps)
    rec = mock_neon_recording(
        start_time=0,
        stop_time=100,
        scene=scene
    )

    export_window = (0, 100 * 1e9 // 20)
    export_timestamps = _prepare_timestamps(rec, export_window, fps=20)
    assert export_timestamps.size, \
        "Expected timestamps to be filled in gaps"
    assert np.all(np.diff(export_timestamps) == 1e9 // 20), \
        "Expected timestamps in gaps to match the provided fps"
    assert export_timestamps.dtype == np.int64, \
        "Expected timestamps to be integer"
