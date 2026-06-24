import pandas as pd

from pupil_labs.neon_player.plugins.events.event_type import EventType
from pupil_labs.neon_player.plugins.events.load import (
    _load_events_from_recording,
    _load_events_from_cache,
    _load_events_from_dataframe
)


def test_events__load_events_from_recording(mock_neon_recording, mock_event_timeseries):
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


def test_events__load_events_from_cache__old_format():
    # Should not contain recording.begin / recording.end as these types are
    # not stored in the settings file
    event_names = [
        "trial.begin", "trial.end", "extra.event"
    ]
    event_types = []
    for idx, event_name in enumerate(event_names):
        et = EventType()
        et._name = event_name
        et._uid = event_name if "recording" in event_name else f"uid{idx}"
        event_types.append(et)

    # Old cache file uses event type UIDs as keys, they should be replaced
    # with event names
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
    for et in event_types:
        assert events[et.name] == cached_events[et.uid], \
            f"Timestamps for {et.name} do not match cached data, or event was incorrectly mapped"


def test_events__load_events_from_cache__new_format():
    # Should not contain recording.begin / recording.end as these types are
    # not stored in the settings file
    event_names = [
        "trial.begin", "trial.end", "extra.event"
    ]
    event_types = []
    for idx, event_name in enumerate(event_names):
        et = EventType()
        et._name = event_name
        et._uid = event_name
        event_types.append(et)

    cached_events = {
        "recording.begin": [0],
        "trial.begin": [100, 400, 700],
        "trial.end": [200, 500, 800],
        "recording.end": [1000],
    }

    event_types, events = _load_events_from_cache(cached_events, event_types)

    assert len(event_types) == 4
    for et in event_types:
        assert et in event_types, f"Expected known event types to be used"
        assert et.name != "extra.event", "Only present event types should be returned"
    assert events == cached_events, "Expected events to match cached data"


def test_events__load_events_from_dataframe():
    events_df = pd.DataFrame({
        "name": [
            "recording.begin", "trial.begin", "trial.end",
            "trial.begin", "trial.end", "recording.end"
        ],
        "timestamp [ns]": [0, 100, 200, 500, 600, 1000],
    })
    et_begin = EventType.from_name("trial.begin")
    events = _load_events_from_dataframe(events_df)

    assert len(events) == 2, "Expected events dict to only contain event of mutable type"
    for event_name, ts in zip(["trial.begin", "trial.end"], [[100, 500], [200, 600]]):
        assert events[event_name] == ts, f"Wrong timestamp for {event_name}"
