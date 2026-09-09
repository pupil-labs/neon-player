import numpy as np

from pupil_labs.neon_recording.timeseries.events import EventArray


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


def mock_scene_timeseries(timestamps):
    data = np.array([
        np.void(
            (ts, idx),
            dtype=[("time", np.int64), ("idx", np.int64)]
        )
        for idx, ts in enumerate(timestamps)
    ])

    return data
