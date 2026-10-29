from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from pupil_labs.marker_mapper.surface import Surface
from pupil_labs.neon_recording.timeseries.gaze import GazeTimeseries

from pupil_labs.neon_player.plugins.gaze import GazeDataPlugin
from pupil_labs.neon_player.plugins.surface_tracking.tracked_surface import (
    TrackedSurface,
)
from pupil_labs.neon_player.plugins.surface_tracking.ui import MarkerEditWidget
from tests.mocks import mock_gaze_timeseries

if TYPE_CHECKING:
    from tests.conftest import MockNeonPlayerApp


def _prepare_test_data(
    qapp: "MockNeonPlayerApp", monkeypatch: pytest.MonkeyPatch
) -> tuple[TrackedSurface, GazeTimeseries, pd.DataFrame]:
    """
    Five fixations with two gaze samples each, only the first two are mapped to
    the surface (see side effect of the mocked method).
    """
    # Tracked surface requires gaze plugin to be present for tracking changes in offset
    monkeypatch.setattr(qapp, "plugins_by_class", {"GazeDataPlugin": GazeDataPlugin()})

    surface = TrackedSurface()
    gazes = mock_gaze_timeseries(
        time=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
        point_x=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        point_y=[15, 25, 35, 45, 55, 65, 75, 85, 95, 105],
    )
    fixation_data = pd.DataFrame(
        {
            "fixation id": [1, 2, 3, 4, 5],
            "start timestamp [ns]": [500, 2500, 4500, 6500, 8500],
            "end timestamp [ns]": [2500, 4500, 6500, 8500, 10500],
            "duration [ms]": [2000, 2000, 2000, 2000, 2000],
        }
    )

    surface.apply_offset_and_map_gazes = MagicMock(
        side_effect=[
            # Fixation 1: mapped, average normalized coords = (0.5, 0.5)
            np.array([[0.2, 0.2], [0.8, 0.8]]),
            # Fixation 2: partially mapped gaze samples, same average coordinates
            np.array([[0.5, 0.5], [np.nan, np.nan]]),
            # Fixation 3: not mapped, x-coords are outside
            np.array([[1.3, 0.3], [-0.1, 0.9]]),
            # Fixation 4: not mapped, y-coords are outside
            np.array([[0.3, -0.1], [0.9, 1.3]]),
            # Fixation 5: not mapped, no gaze samples available
            np.array([[np.nan, np.nan], [np.nan, np.nan]]),
        ]
    )

    return surface, gazes, fixation_data


def test_tracked_surface_append_mapped_fixation_data_gaze_samples_forwarded(
    qapp: "MockNeonPlayerApp",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surface, gazes, fixation_data = _prepare_test_data(qapp, monkeypatch)
    surface._append_mapped_fixation_data(gazes, fixation_data.copy())

    for idx, mock_call in enumerate(surface.apply_offset_and_map_gazes.call_args_list):
        start_time = fixation_data.loc[idx, "start timestamp [ns]"]
        end_time = fixation_data.loc[idx, "end timestamp [ns]"]

        expected_mask = (gazes.time >= start_time) & (gazes.time <= end_time)
        expected_gazes = gazes[expected_mask]

        assert np.array_equal(mock_call[0][0].time, expected_gazes.time)


@pytest.mark.parametrize("fixation_id", [1, 2])
def test_tracked_surface_append_mapped_fixation_data_fixation_should_be_mapped(
    qapp: "MockNeonPlayerApp", monkeypatch: pytest.MonkeyPatch, fixation_id: int
) -> None:
    surface, gazes, fixation_data = _prepare_test_data(qapp, monkeypatch)
    result = surface._append_mapped_fixation_data(gazes, fixation_data.copy())

    fixation_row = result[result["fixation id"] == fixation_id]
    assert fixation_row["fixation detected on surface"].to_numpy()[0]
    assert fixation_row["fixation x [normalized]"].to_numpy()[0] == 0.5
    assert fixation_row["fixation y [normalized]"].to_numpy()[0] == 0.5


@pytest.mark.parametrize("fixation_id", [3, 4])
def test_tracked_surface_append_mapped_fixation_data_fixation_should_not_be_mapped(
    qapp: "MockNeonPlayerApp", monkeypatch: pytest.MonkeyPatch, fixation_id: int
) -> None:
    surface, gazes, fixation_data = _prepare_test_data(qapp, monkeypatch)
    result = surface._append_mapped_fixation_data(gazes, fixation_data.copy())

    fixation_row = result[result["fixation id"] == fixation_id]
    assert not fixation_row["fixation detected on surface"].to_numpy()[0]
    assert fixation_row["fixation x [normalized]"].to_numpy()[0] == 0.6
    assert fixation_row["fixation y [normalized]"].to_numpy()[0] == 0.6


@pytest.mark.parametrize("fixation_id", [5])
def test_tracked_surface_append_mapped_fixation_data_fixation_no_mapped_gazes(
    qapp: "MockNeonPlayerApp", monkeypatch: pytest.MonkeyPatch, fixation_id: int
) -> None:
    surface, gazes, fixation_data = _prepare_test_data(qapp, monkeypatch)
    result = surface._append_mapped_fixation_data(gazes, fixation_data.copy())

    fixation_row = result[result["fixation id"] == fixation_id]
    assert not fixation_row["fixation detected on surface"].to_numpy()[0]
    assert np.isnan(fixation_row["fixation x [normalized]"].to_numpy()[0])
    assert np.isnan(fixation_row["fixation y [normalized]"].to_numpy()[0])


def test_surface_export_requires_gaze_plugin(
    qapp: "MockNeonPlayerApp", tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(qapp, "plugins_by_class", {})
    surface = TrackedSurface()
    surface.name = "test"
    gazes = mock_gaze_timeseries([1], [20], [30])

    assert surface.apply_offset_and_map_gazes(gazes) is None

    surface.export_gazes(gazes, tmp_path)
    surface.export_fixations(gazes, tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_surface_requires_tracker_plugin(
    qapp: "MockNeonPlayerApp", monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(qapp, "plugins_by_class", {})
    surface = TrackedSurface()
    with pytest.raises(RuntimeError, match="Surface tracking plugin must be available"):
        _ = surface.tracker_plugin


def test_surface_marker_id_from_widget_uses_library_integer_id(
    qapp: "MockNeonPlayerApp", monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(qapp, "plugins_by_class", {})
    surface = TrackedSurface()
    surface.tracker_surface = Surface("Desk", OrderedDict([(42, np.zeros((4, 2)))]))
    widget = MarkerEditWidget("42")

    widget.set_surface(surface)

    assert widget.isChecked()
    surface.remove_marker(widget.marker_uid)
    assert not surface.tracker_surface.markers
