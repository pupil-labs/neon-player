from unittest.mock import MagicMock

import numpy as np

from pupil_labs.neon_player.plugins.events import (
    _load_events_from_cache, _load_events_from_recording, EventType, EventsPlugin
)
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


def test_events__load_events_from_recording(mock_neon_recording):
    events_dict = {
        "recording.begin": [0],
        "trial.begin": [100, 400, 700],
        "trial.end": [200, 500, 800],
        "recording.end": [1000],
    }
    mock_events = mock_event_timeseries(events_dict)
    recording = mock_neon_recording(events=mock_events)

    event_types, events = _load_events_from_recording(recording)
    assert len(event_types) == 4
    assert len(events) == 4

    for event_name in ["recording.begin", "trial.begin", "trial.end", "recording.end"]:
        et = [et for et in event_types if et.name == event_name]
        assert len(et) == 1, f"Expected exactly one event type for {event_name}, got {len(et)}"
        et = et[0]

        assert et.name == event_name
        if "recording" in event_name:
            assert et.uid == event_name, "Expected uid to match name for immutable events"

        assert events[et.uid] == events_dict[event_name], \
            f"Timestamps for {event_name} do not match input data"


def test_events__load_events_from_cache():
    # Should not contain recording.begin / recording.end as these types are
    # not stored in the settings file
    event_names = [
        "trial.begin", "trial.end", "extra.event"
    ]
    event_types = []
    for idx, event_name in enumerate(event_names):
        et = EventType()
        et.name = event_name
        et.uid = event_name if "recording" in event_name else f"uid{idx}"
        event_types.append(et)

    cached_events = {
        "recording.begin": [0],
        "uid0": [100, 400, 700],  # trial.begin
        "uid1": [200, 500, 800],  # trial.end
        "recording.end": [1000],
    }

    event_types, events = _load_events_from_cache(cached_events, event_types)

    assert len(event_types) == 4
    for et in event_types:
        assert et in event_types, f"Expected known event types to be used"
        assert et.name != "extra.event", "Only present event types should be returned"
    assert events == cached_events, "Expected events to match cached events"


def test_events_plugin__add_event__new_type(mock_neon_recording):
    plugin = EventsPlugin()
    type(plugin).recording = mock_neon_recording()
    plugin.save_cached_json = MagicMock()
    et = EventType.from_name("test.event")
    plugin.event_types = [et]
    plugin.events = {}

    plugin.add_event(et, ts=1)

    assert et.uid in plugin.events, "Expected new event type to be added to events dict"
    assert plugin.events[et.uid] == [1], "Expected timestamp to be added to events dict"
    plugin.save_cached_json.assert_called_once()


def test_events_plugin__add_event__existing_type(mock_neon_recording):
    plugin = EventsPlugin()
    type(plugin).recording = mock_neon_recording()
    plugin.save_cached_json = MagicMock()
    et = EventType.from_name("test.event")
    plugin.event_types = [et]
    plugin.events = {et.uid: [1, 2, 3]}

    plugin.add_event(et, ts=4)

    assert plugin.events[et.uid] == [1, 2, 3, 4], \
        "Expected new timestamp to be appended to existing list"
    plugin.save_cached_json.assert_called_once()
