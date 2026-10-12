import numpy as np

from pupil_labs.neon_recording.timeseries.events import EventArray
from pupil_labs.neon_recording.timeseries.gaze import GazeArray
from pupil_labs.neon_recording.timeseries.worn import WornArray


def mock_event_timeseries(events_dict):
    all_events = []
    for event_name, timestamps in events_dict.items():
        for ts in timestamps:
             all_events.append((ts, event_name))
    all_events = sorted(all_events, key=lambda x: x[0])

    data = np.array([
        np.void(
            (ts, event_name),
            dtype=[("time", np.int64), ("event", np.str_, 50)]
        )
        for ts, event_name in all_events
    ])
    data = data.view(EventArray)

    return data


def mock_gaze_timeseries(timestamps, xs, ys):
    data = np.array([
        np.void(
            (ts, x, y),
            dtype=[("time", np.int64), ("point_x", np.float64), ("point_y", np.float64)]
        )
        for ts, x, y in zip(timestamps, xs, ys)
    ])
    data = data.view(GazeArray)

    return data


def mock_scene_timeseries(timestamps):
    data = np.array([
        np.void(
            (ts, idx),
            dtype=[("time", np.int64), ("idx", np.int64)]
        )
        for idx, ts in enumerate(timestamps)
    ])

    return data


def mock_worn_timeseries(timestamps: np.ndarray, worn: np.ndarray) -> np.recarray:
    data = np.array([
        np.void(
            (ts, worn_datum),
            dtype=[
                ("time", np.int64),
                ("worn", np.float64)
            ]
        )
        for ts, worn_datum in zip(timestamps, worn)
    ])
    data = data.view(WornArray)

    return data
