"""
Tests below are designed for the case when a single recording is opened.
"""

import pytest
from unittest.mock import MagicMock

from pupil_labs.neon_player.plugins.events import EventsPlugin
from pupil_labs.neon_player.plugins.events.event_type import EventType


def test_events_plugin__on_recording_loaded__from_recording(mock_neon_recording):
    et = EventType.from_name("test.event")

    plugin = EventsPlugin()
    plugin._load_events = MagicMock(
        return_value=([et], {"test.event": [100, 200, 300]}, "recording")
    )
    plugin.save_cached_json = MagicMock()
    plugin.on_recording_loaded(mock_neon_recording())

    assert plugin.event_types == [et], "Expected event types to be loaded from recording"
    assert plugin.events == {"test.event": [100, 200, 300]}, "Expected events to be loaded from recording"
    plugin.save_cached_json.assert_called_once()


def test_events_plugin__on_recording_loaded__from_cache__old_format(mock_neon_recording):
    et = EventType.from_name("test.event")
    et._uid = "uid0"

    plugin = EventsPlugin()
    plugin._event_types_by_name = {"test.event": et}
    plugin._load_events = MagicMock(
        return_value=([et], {"test.event": [100, 200, 300]}, "cache")
    )
    plugin.save_cached_json = MagicMock()
    plugin.on_recording_loaded(mock_neon_recording())

    assert plugin.event_types[0].uid == "test.event", "Expected event ID to match its name"
    assert plugin.events == {"test.event": [100, 200, 300]}, "Expected events to be loaded from cache"
    # NOTE: The save_cached_json method should be called to update the cache with the new event type names
    plugin.save_cached_json.assert_called_once()


def test_events_plugin__on_recording_loaded__from_cache__new_format(mock_neon_recording):
    et = EventType.from_name("test.event")

    plugin = EventsPlugin()
    plugin._event_types_by_name = {"test.event": et}
    plugin._load_events = MagicMock(
        return_value=([et], {"test.event": [100, 200, 300]}, "cache")
    )
    plugin.save_cached_json = MagicMock()
    plugin.on_recording_loaded(mock_neon_recording())

    assert plugin.event_types == [et], "Expected event types not to be modified"
    assert plugin.events == {"test.event": [100, 200, 300]}, "Expected events to be loaded from cache"
    # No need to call save_cached_json since the cache format is already correct
    plugin.save_cached_json.assert_not_called()


def test_events_plugin__get_unique_event_name():
    plugin = EventsPlugin()
    et = EventType.from_name("event-1")
    plugin.event_types = [et]
    new_event_name = plugin._get_unique_event_name()

    assert new_event_name == "event-2", "Expected new event name to follow the pattern"


def test_events_plugin__add_event_type(qtbot):
    plugin = EventsPlugin()
    et = EventType.from_name("test.event")

    with qtbot.waitSignal(plugin.changed):
        plugin.add_event_type(et)

    assert et in plugin.event_types, "Expected new event type to be added to event_types list"


def test_events_plugin__add_event_type__type_already_exists():
    plugin = EventsPlugin()
    et = EventType.from_name("recording.begin")
    plugin.event_types = [et]

    with pytest.raises(ValueError, match="already exists"):
        plugin.add_event_type(et)


def test_events_plugin__delete_event_type__no_events(qtbot):
    plugin = EventsPlugin()
    et = EventType.from_name("test.event")
    plugin.event_types = [et]

    with qtbot.waitSignal(plugin.changed):
        plugin.delete_event_type(et)

    assert et not in plugin.event_types, "Expected event type to be removed from event_types list"


def test_events_plugin__delete_event_type__with_events(qtbot):
    plugin = EventsPlugin()
    et = EventType.from_name("test.event")
    plugin.event_types = [et]
    plugin._events = {et.uid: [100, 200, 300]}
    plugin.save_cached_json = MagicMock()

    with qtbot.waitSignal(plugin.changed):
        plugin.delete_event_type(et)

    assert et not in plugin.event_types, "Expected event type to be removed from event_types list"
    assert et.uid not in plugin._events, "Expected events for deleted type to be removed from events dict"
    plugin.save_cached_json.assert_called_once()


def test_events_plugin__delete_event_type__immutable_type():
    plugin = EventsPlugin()
    et = EventType.from_name("recording.begin")
    plugin.event_types = [et]

    with pytest.raises(ValueError, match="cannot be deleted"):
        plugin.delete_event_type(et)


def test_events_plugin__on_event_name_changed():
    plugin = EventsPlugin()
    et = EventType.from_name("test.event")
    plugin.event_types = [et]

    et._name = "renamed.event"

    # Simulate the name_changed signal being emitted
    plugin._on_event_name_changed("test.event", "renamed.event", et)

    assert "test.event" not in plugin._event_types_by_name
    assert "renamed.event" in plugin._event_types_by_name


def test_events_plugin__add_events__adds_new_event_types(qtbot):
    plugin = EventsPlugin()
    plugin.save_cached_json = MagicMock()

    # Load function returns mock output so using an empty dataframe
    with qtbot.waitSignal(plugin.changed):
        plugin.add_events({"new.event": [100, 200, 300]})

    assert len(plugin.event_types) == 1, "Expected new event type to be added"
    et = plugin.event_types[0]
    assert et.name == "new.event", "Expected event type name to match the data"
    assert plugin._events == {et.uid: [100, 200, 300]}, "Expected events to be stored"
    plugin.save_cached_json.assert_called_once()


def test_events_plugin__add_events__reuses_existing_event_types():
    plugin = EventsPlugin()
    existing_et = EventType.from_name("existing.event")
    plugin.event_types = [existing_et]
    plugin._events = {existing_et.uid: [0]}
    plugin.save_cached_json = MagicMock()

    plugin.add_events({"existing.event": [100, 200, 300]})

    assert len(plugin.event_types) == 1, "Expected no new event type to be added"
    assert plugin.event_types[0] is existing_et, "Expected existing event type to be reused"
    assert plugin._events == {existing_et.uid: [0, 100, 200, 300]}, \
        "Expected events to be appended to the existing ones"
    plugin.save_cached_json.assert_called_once()


def test_events_plugin__add_events__raises_on_immutable_type():
    plugin = EventsPlugin()
    et = EventType.from_name("recording.begin")
    plugin.event_types = [et]

    with pytest.raises(ValueError, match="cannot be added"):
        plugin.add_events({"recording.begin": [100, 200]})


def test_events_plugin__delete_events__deletes_events():
    plugin = EventsPlugin()
    et = EventType.from_name("test.event")
    plugin.event_types = [et]
    plugin._events = {et.uid: [100, 200, 300, 400]}
    plugin.save_cached_json = MagicMock()

    plugin.delete_events({"test.event": [200, 300]})

    assert sorted(plugin._events[et.uid]) == [100, 400], \
        "Expected specified events to be deleted"
    plugin.save_cached_json.assert_called_once()


def test_events_plugin__delete_events__removes_event_type_if_no_events_left(qtbot):
    plugin = EventsPlugin()
    et = EventType.from_name("test.event")
    plugin.event_types = [et]
    plugin._events = {et.uid: [100, 200]}
    plugin.save_cached_json = MagicMock()

    with qtbot.waitSignal(plugin.changed):
        plugin.delete_events({"test.event": [100, 200]}, remove_empty_types=True)

    assert et.name not in plugin._event_types_by_name, \
        "Expected event type to be removed from event_types_by_name dict"
    assert et.uid not in plugin._events, \
        "Expected events for deleted type to be removed from events dict"
    plugin.save_cached_json.assert_called_once()


def test_events_plugin__delete_events__raises_on_immutable_type():
    plugin = EventsPlugin()
    et = EventType.from_name("recording.begin")
    plugin.event_types = [et]
    plugin._events = {"recording.begin": [0]}

    with pytest.raises(ValueError, match="cannot be deleted"):
        plugin.delete_events({"recording.begin": [0]})


@pytest.fixture(autouse=False)
def test_event_data(mock_neon_recording, mock_event_timeseries):
    plugin = EventsPlugin()
    mock_events = mock_event_timeseries({
        "recording.begin": [0],
        "event.from.rec": [100, 400, 700],
        "event.same.ts": [100],
        "recording.end": [1000],
    })
    et_rec = EventType.from_name("event.from.rec")
    et_player = EventType.from_name("event.from.player")
    et_same_ts = EventType.from_name("event.same.ts")
    plugin.event_types = [et_rec, et_player, et_same_ts]
    plugin.events = {
        "recording.begin": [0],
        "event.from.rec": [100, 400, 700],
        "event.from.player": [200, 500, 800],
        "event.same.ts": [100],
        "recording.end": [1000],
    }
    mock_recording = mock_neon_recording(events=mock_events, info={"recording_id": "test.recording"})
    return plugin, mock_recording


def test_events_plugin__find_closest_event(test_event_data):
    plugin, _ = test_event_data
    et = plugin._event_types_by_name["event.from.rec"]
    assert plugin._find_closest_event(et, 100) == 100


def test_events_plugin__find_closest_event__missing_type(test_event_data):
    plugin, _ = test_event_data
    et = EventType.from_name("non.existent.event")
    assert plugin._find_closest_event(et, 100) is None


def test_events_plugin__find_closest_event__tolerance(test_event_data):
    plugin, _ = test_event_data
    et = plugin._event_types_by_name["event.from.rec"]
    assert plugin._find_closest_event(et, 110, tolerance_ns=15) == 100


def test_events_plugin__events__uses_names_not_ids(test_event_data):
    plugin, _ = test_event_data
    event_names = ["recording.begin", "event.from.rec", "event.from.player", "event.same.ts", "recording.end"]
    for event_name in event_names:
        assert event_name in plugin.events, f"Expected {event_name} to be present in events property"


def test_events_plugin__prepare_events_export__whole_time_range(test_event_data):
    plugin, mock_recording = test_event_data
    events_df = plugin._prepare_events_export(mock_recording, export_window=(0, 1000))
    assert len(events_df) == 9, "Expected all events to be included in export"


def test_events_plugin__prepare_events_export__custom_time_range(test_event_data):
    plugin, mock_recording = test_event_data
    events_df = plugin._prepare_events_export(mock_recording, export_window=(200, 750))
    assert len(events_df) == 4, "Expected only events within the custom time range to be included in export"
    for event_to_include in ["event.from.rec", "event.from.player"]:
        assert len(events_df[events_df["name"] == event_to_include]) == 2, \
            f"Expected two {event_to_include} events in export"


def test_events_plugin__prepare_events_export__timestamps_are_sorted(test_event_data):
    plugin, mock_recording = test_event_data
    events_df = plugin._prepare_events_export(mock_recording, export_window=(150, 450))
    assert events_df["name"].tolist() == ["event.from.player", "event.from.rec"], \
        "Expected events to be sorted by timestamp in export"


def test_events_plugin__prepare_events_export__source(test_event_data):
    plugin, mock_recording = test_event_data
    events_df = plugin._prepare_events_export(mock_recording, export_window=(0, 1000))
    for event_name, source in zip(
        ["event.from.rec", "event.from.player", "event.same.ts"],
        ["recording", "player", "recording"]
    ):
        assert all(events_df[events_df["name"] == event_name]["type"] == source), \
            f"Expected all {event_name} events to be labeled as {source} in export"
