import numpy as np
import pytest

from pupil_labs.neon_player.plugins.gaze import (
    CircleViz,
    CrosshairViz,
    GazeDataPlugin,
    apply_offset,
    _prepare_gaze_export
)

from tests.mocks import mock_gaze_timeseries, mock_scene_timeseries, mock_worn_timeseries


def test_circle_viz_parameter_capping_not_required():
    viz = CircleViz()
    viz.radius = 30
    viz.stroke_width = 10   # < 2 * radius
    assert viz._applied_radius == viz.radius
    assert viz._applied_stroke_width == viz.stroke_width


def test_circle_viz_parameter_capping_required():
    viz = CircleViz()
    viz.radius = 30
    viz.stroke_width = 80   # > 2 * radius
    assert viz._applied_radius == 35
    assert viz._applied_stroke_width == 70


def test_gaze_plugin_to_dict_viz_have_class_names(qapp, mock_neon_recording):
    qapp.recording = mock_neon_recording()

    plugin = GazeDataPlugin()
    plugin.visualizations = [CircleViz(), CrosshairViz()]
    state = plugin.to_dict(recursive=True)

    assert state["visualizations"][0]["__class__"] == "CircleViz"
    assert state["visualizations"][1]["__class__"] == "CrosshairViz"


@pytest.mark.parametrize(
    "offset, expected_gaze",
    [
        ((0.0, 0.0), (300.0, 400.0)),
        ((0.01, 0.0), (316.0, 400.0)),
        ((0.0, 0.01), (300.0, 412.0)),
        ((0.01, 0.01), (316.0, 412.0)),
    ]
)
def test_apply_offset(offset, expected_gaze, mock_neon_recording):
    timestamps = np.array([1, 2, 3])
    gaze_x = np.array([300, 300, 300], dtype=np.float64)
    gaze_y = np.array([400, 400, 400], dtype=np.float64)
    gaze = mock_gaze_timeseries(timestamps, gaze_x, gaze_y)
    recording = mock_neon_recording(
        scene=mock_scene_timeseries([1, 2, 3]),
    )
    corrected_gazes = apply_offset(recording, gaze.point, offset)

    # NOTE: assuming 1600x1200 resolution of the scene video
    expected_x, expected_y = expected_gaze
    assert np.allclose(corrected_gazes[:, 0], expected_x), \
        "Offset correction for gaze x-coordinate is not applied correctly"
    assert np.allclose(corrected_gazes[:, 1], expected_y), \
        "Offset correction for gaze x-coordinate is not applied correctly"


def test_prepare_gaze_export_respects_export_window(mock_neon_recording):
    timestamps = np.array([1, 2, 3])
    gaze_x = np.array([300, 300, 300], dtype=np.float64)
    gaze_y = np.array([400, 400, 400], dtype=np.float64)
    recording = mock_neon_recording(
        gaze=mock_gaze_timeseries(timestamps, gaze_x, gaze_y),
        scene=mock_scene_timeseries([1, 2, 3]),
        info={"recording_id": "mock"}
    )
    export_window = (5, 7)  # contains no gaze timestamps
    gaze_offset = (0.0, 0.0)

    gaze_df = _prepare_gaze_export(recording, None, export_window, gaze_offset)
    assert gaze_df.empty


def test_prepare_gaze_export_applies_gaze_offset(mock_neon_recording):
    timestamps = np.array([1, 2, 3])
    gaze_x = np.array([300, 300, 300], dtype=np.float64)
    gaze_y = np.array([400, 400, 400], dtype=np.float64)
    recording = mock_neon_recording(
        gaze=mock_gaze_timeseries(timestamps, gaze_x, gaze_y),
        scene=mock_scene_timeseries([1, 2, 3]),
        info={"recording_id": "mock"}
    )

    gaze_offset = (0.01, 0.01)
    gaze_df = _prepare_gaze_export(recording, None, (0, 4), gaze_offset)

    # NOTE: assuming 1600x1200 resolution of the scene video
    assert np.allclose(gaze_df["gaze x [px]"].values, 316.0)
    assert np.allclose(gaze_df["gaze y [px]"].values, 412.0)


def test_prepare_export_data_no_worn_data(mock_neon_recording):
    mock_gaze = mock_gaze_timeseries(
        timestamps=np.arange(5),
        xs=np.arange(5),
        ys=np.arange(5)
    )
    recording = mock_neon_recording(gaze=mock_gaze)
    gaze = _prepare_gaze_export(recording, None, (0.5, 3.5), (0.0, 0.0))
    assert len(gaze) == 3
    assert "worn" not in gaze.columns


def test_prepare_export_data_with_worn_data(mock_neon_recording):
    # Simulate the case of mismatch between gaze and worn length
    timestamps = np.arange(5)
    mock_gaze = mock_gaze_timeseries(timestamps, xs=np.arange(5), ys=np.arange(5))
    mock_worn = mock_worn_timeseries(timestamps[:-1], worn=[255, 255, 0, 255])

    recording = mock_neon_recording(gaze=mock_gaze, worn=mock_worn)
    gaze = _prepare_gaze_export(recording, recording.worn, (0.5, 4.5), (0.0, 0.0))
    assert len(gaze) == 4

    assert np.allclose(gaze["worn"].values, np.array([1, 0, 1, np.nan]), equal_nan=True)
