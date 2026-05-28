import numpy as np
import pandas as pd
import pytest

from unittest.mock import MagicMock

from pupil_labs.neon_player.plugins.surface_tracking.tracked_surface import TrackedSurface
from pupil_labs.neon_recording.timeseries.gaze import GazeTimeseries, GazeArray


def mock_gaze_timeseries(time, point_x, point_y):
    data = np.array([
        np.void(
            (t, x, y),
            dtype=[("time", np.int64), ("point_x", np.float32), ("point_y", np.float32)]
        )
        for t, x, y in zip(time, point_x, point_y)
    ])
    data = data.view(GazeArray)
    return GazeTimeseries("", data=data)


def _prepare_test_data():
    """
    Four fixations with two gaze samples each, only the first two are mapped to
    the surface (see side effect of the mocked method).
    """

    surface = TrackedSurface()
    gazes = mock_gaze_timeseries(
        time=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
        point_x=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        point_y=[15, 25, 35, 45, 55, 65, 75, 85, 95, 105],
    )
    fixation_data = pd.DataFrame({
        "recording id": ["test"] * 5,
        "fixation id": [1, 2, 3, 4, 5],
        "start timestamp [ns]": [500, 2500, 4500, 6500, 8500],
        "end timestamp [ns]": [2500, 4500, 6500, 8500, 10500],
        "duration [ms]": [2000, 2000, 2000, 2000, 2000],
        "fixation x [px]": [20, 40, 60, 80, 100],
        "fixation y [px]": [25, 45, 65, 85, 105],
        "azimuth [deg]": [0, 0, 0, 0, 0],
        "elevation [deg]": [0, 0, 0, 0, 0],
    })

    surface.apply_offset_and_map_gazes = MagicMock(
        side_effect=[
            # Fixation 1: mapped, average normalized coords = (0.5, 0.5)
            np.array([[0.2, 0.2], [0.8, 0.8]]),
            # Fixation 2: mapped, but some gaze samples are not mapped, same average coords
            np.array([[0.5, 0.5], [np.nan, np.nan]]),
            # Fixation 3: not mapped, x-coords are outside
            np.array([[1.2, 0.2], [-0.2, 0.8]]),
            # Fixation 4: not mapped, y-coords are outside
            np.array([[0.2, -0.2], [0.8, 1.2]]),
            # Fixation 5: not mapped, no gaze samples available
            np.array([[np.nan, np.nan], [np.nan, np.nan]]),
        ]
    )

    return surface, gazes, fixation_data


def test_tracked_surface_append_mapped_fixation_data_gaze_samples_forwarded():
    surface, gazes, fixation_data = _prepare_test_data()
    result = surface._append_mapped_fixation_data(gazes, fixation_data.copy())

    for idx, mock_call in enumerate(surface.apply_offset_and_map_gazes.call_args_list):
        start_time = fixation_data.loc[idx, "start timestamp [ns]"]
        end_time = fixation_data.loc[idx, "end timestamp [ns]"]

        expected_mask = (gazes.time >= start_time) & (gazes.time <= end_time)
        expected_gazes = gazes[expected_mask]

        assert np.array_equal(mock_call[0][0].time, expected_gazes.time)


@pytest.mark.parametrize("fixation_id", [1, 2])
def test_tracked_surface_append_mapped_fixation_data_fixation_should_be_mapped(fixation_id):
    surface, gazes, fixation_data = _prepare_test_data()
    result = surface._append_mapped_fixation_data(gazes, fixation_data.copy())

    fixation_row = result[result["fixation id"] == fixation_id]
    assert fixation_row["fixation detected on surface"].values[0] == True
    assert fixation_row["fixation x [normalized]"].values[0] == 0.5
    assert fixation_row["fixation y [normalized]"].values[0] == 0.5


@pytest.mark.parametrize("fixation_id", [3, 4, 5])
def test_tracked_surface_append_mapped_fixation_data_fixation_should_not_be_mapped(fixation_id):
    surface, gazes, fixation_data = _prepare_test_data()
    result = surface._append_mapped_fixation_data(gazes, fixation_data.copy())

    fixation_row = result[result["fixation id"] == fixation_id]
    assert fixation_row["fixation detected on surface"].values[0] == False
    assert np.isnan(fixation_row["fixation x [normalized]"].values[0])
    assert np.isnan(fixation_row["fixation y [normalized]"].values[0])
