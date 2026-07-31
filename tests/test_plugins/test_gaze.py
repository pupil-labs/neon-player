import numpy as np

from pupil_labs.neon_recording.timeseries.gaze import GazeArray
from pupil_labs.neon_recording.timeseries.worn import WornArray

from pupil_labs.neon_player.plugins.gaze import GazeDataPlugin, CircleViz, CrosshairViz


def mock_gaze_timeseries(timestamps: np.ndarray, gaze_data: np.ndarray) -> np.recarray:
    data = np.array([
        np.void(
            (ts, gaze_datum[0], gaze_datum[1]),
            dtype=[
                ("time", np.int64),
                ("point_x", np.float64),
                ("point_y", np.float64)
            ]
        )
        for ts, gaze_datum in zip(timestamps, gaze_data)
    ])
    data = data.view(GazeArray)

    return data


def mock_worn_timeseries(timestamps: np.ndarray, worn_data: np.ndarray) -> np.recarray:
    data = np.array([
        np.void(
            (ts, worn_datum),
            dtype=[
                ("time", np.int64),
                ("worn", np.float64)
            ]
        )
        for ts, worn_datum in zip(timestamps, worn_data)
    ])
    data = data.view(WornArray)

    return data

def mock_test_data():
    mock_gaze = mock_gaze_timeseries(
        timestamps=np.array([0, 1, 2, 3, 4]),
        gaze_data=np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]),
    )
    mock_worn = mock_worn_timeseries(
        timestamps=np.array([0, 1, 2, 3, 4]),
        worn_data=np.array([255.0, 255.0, 0.0, 255.0, 0.0]),
    )
    return mock_gaze, mock_worn


def test_prepare_export_data__no_recording():
    export_gazes, export_worn = GazeDataPlugin._prepare_export_data(None, (0, 1))
    assert export_gazes is None
    assert export_worn is None


def test_prepare_export_data__no_worn_data(mock_neon_recording):
    mock_gaze, _ = mock_test_data()
    recording = mock_neon_recording(gaze=mock_gaze)
    plugin = GazeDataPlugin()
    export_gazes, export_worn = plugin._prepare_export_data(recording, (0.5, 3.5))
    assert len(export_gazes) == 3
    assert export_worn is None


def test_prepare_export_data__with_worn_data__all_match(mock_neon_recording):
    mock_gaze, mock_worn = mock_test_data()
    recording = mock_neon_recording(gaze=mock_gaze, worn=mock_worn)
    plugin = GazeDataPlugin()
    export_gazes, export_worn = plugin._prepare_export_data(recording, (0.5, 3.5))
    assert len(export_gazes) == 3
    assert len(export_worn) == 3

    assert np.allclose(export_worn, np.array([1, 0, 1]))


def test_prepare_export_data__with_worn_data__partial_match(mock_neon_recording):
    mock_gaze, mock_worn = mock_test_data()
    recording = mock_neon_recording(gaze=mock_gaze, worn=mock_worn[:3])
    plugin = GazeDataPlugin()
    export_gazes, export_worn = plugin._prepare_export_data(recording, (0.5, 3.5))
    assert len(export_gazes) == 3
    assert len(export_worn) == 3

    # The last worn value is NaN since it doesn't match any worn timestamp
    assert np.allclose(export_worn, np.array([1, 0, np.nan]), equal_nan=True)


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


def test_gaze_plugin__to_dict__viz_have_class_names(qapp, mock_neon_recording):
    qapp.recording = mock_neon_recording()

    plugin = GazeDataPlugin()
    plugin.visualizations = [CircleViz(), CrosshairViz()]
    state = plugin.to_dict(recursive=True)

    assert state["visualizations"][0]["__class__"] == "CircleViz"
    assert state["visualizations"][1]["__class__"] == "CrosshairViz"
