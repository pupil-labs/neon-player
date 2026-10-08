import numpy as np
import pytest

from pupil_labs.camera import Camera
from pupil_labs.marker_mapper.surface import normalized_corners

from pupil_labs.neon_player.plugins.gaze import GazeDataPlugin
from pupil_labs.neon_player.plugins.surface_tracking import SurfaceTrackingPlugin
from pupil_labs.neon_player.plugins.surface_tracking.surface_tracking import (
    _get_position_for_export,
    _prepare_surface_positions_export,
    DetectedMarker
)
from pupil_labs.neon_player.plugins.surface_tracking.tracked_surface import TrackedSurface

from tests.mocks import mock_scene_timeseries


def mock_transform(x0, y0, w, h):
    """Mock surface location (translation + scaling only).

    Transforms normalized corners to the corners of a wxh rectangle
    centered at (x0, y0).
    """
    return np.array([[w, 0, x0 - w / 2], [0, h, y0 - h / 2], [0, 0, 1]], dtype=float)


def test_surface_tracking_plugin__to_dict__recursive(qapp, mock_neon_recording):
    qapp.recording = mock_neon_recording()
    qapp.plugins_by_class["GazeDataPlugin"] = GazeDataPlugin()

    plugin = SurfaceTrackingPlugin()
    # Set the surfaces directly to a private field to skip surface initialization
    plugin._surfaces = [TrackedSurface(), TrackedSurface()]
    plugin_dict = plugin.to_dict(recursive=True)

    for el in plugin_dict["surfaces"]:
        assert isinstance(el, dict), \
            f"Expected surface to be serialized to dict, but got {type(el)}"
        assert not el["edit"], \
            f"Expected surface to be serialized with edit=False, but got {el['edit']}"

        assert isinstance(el["preview_options"], dict), \
            f"Expected surface preview_options to be serialized to dict"


@pytest.mark.parametrize(
    "x0,y0,w,h,expected_nan_row_mask", [
        (800, 600, 400, 300, np.array([False, False, False, False])),
        (1500, 100, 300, 300, np.array([True, True, True, False])),
    ]
)
def test_get_position_for_export(x0, y0, w, h, expected_nan_row_mask):
    # No distortion to simplify the calculations
    scene_size = (1600, 1200)
    camera = Camera(*scene_size, np.eye(3))

    # Only the second element is used (surface coords -> image coords)
    location = [None, mock_transform(x0, y0, w, h)]
    corners = _get_position_for_export(location, camera, scene_size)
    assert np.all(np.isnan(corners[expected_nan_row_mask, :]))
    assert np.all(~np.isnan(corners[~expected_nan_row_mask, :]))


def test_prepare_surface_positions_export_respects_export_window(mock_neon_recording):
    recording = mock_neon_recording(
        scene=mock_scene_timeseries([1, 2, 3, 4, 5])
    )
    scene_size = (1600, 1200)
    camera = Camera(*scene_size, np.eye(3))
    positions_df = _prepare_surface_positions_export(
        recording,
        (6, 8),  # no scene timestamps satisfy the window
        None,
        None,
        camera
    )
    assert positions_df.empty


def test_prepare_surface_positions_export_drops_rows_if_no_markers(mock_neon_recording):
    recording = mock_neon_recording(
        scene=mock_scene_timeseries([1])
    )
    scene_size = (1600, 1200)
    camera = Camera(*scene_size, np.eye(3))
    location = [None, mock_transform(800, 600, 400, 300)]
    positions_df = _prepare_surface_positions_export(
        recording,
        (0, 3),
        [[]],
        [location],
        camera
    )
    assert positions_df.empty


def test_prepare_surface_positions_export_drops_rows_if_no_location(mock_neon_recording):
    recording = mock_neon_recording(
        scene=mock_scene_timeseries([1])
    )
    scene_size = (1600, 1200)
    camera = Camera(*scene_size, np.eye(3))
    positions_df = _prepare_surface_positions_export(
        recording,
        (0, 3),
        [[DetectedMarker(0, normalized_corners())]],
        [None],
        camera
    )
    assert positions_df.empty


def test_prepare_surface_positions_export(mock_neon_recording):
    recording = mock_neon_recording(
        scene=mock_scene_timeseries([1, 2, 3])
    )
    scene_size = (1600, 1200)
    camera = Camera(*scene_size, np.eye(3))
    location = [None, mock_transform(800, 600, 400, 300)]
    positions_df = _prepare_surface_positions_export(
        recording,
        (1, 3),
        [[], [], [DetectedMarker(tag_id, normalized_corners()) for tag_id in [0, 1, 2, 3]]],
        [None, None, location],
        camera
    )
    assert len(positions_df) == 1
    row = positions_df.iloc[0, :]

    assert row["recording id"] == "mock"
    assert row["timestamp [ns]"] == 3
    assert row["detected marker IDs"] == "0;1;2;3"
    assert np.isclose(row["tr x [px]"], 1000)
    assert np.isclose(row["br y [px]"], 750)
