from collections.abc import Callable, Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest
from pupil_labs.neon_recording import NeonRecording
from PySide6.QtCore import QCoreApplication, QEvent
from pytestqt.qtbot import QtBot

from pupil_labs.neon_player.plugins.events import (
    EventsPlugin,
    EventType,
    _load_events_from_cache,
    _load_events_from_dataframe,
    _load_events_from_recording,
)
from pupil_labs.neon_player.ui.timeline_dock import TimeLineDock
from pupil_labs.neon_player.ui.timeline_dock_components import (
    TimelineRowKey,
    TimelineRowRole,
)
from tests.conftest import MockNeonPlayerApp
from tests.mocks import mock_event_timeseries


def test_events__load_events_from_recording(
    mock_neon_recording: Callable[..., NeonRecording],
) -> None:
    events_dict = {
        "recording.begin": [0],
        "trial.begin": [400, 100, 700],
        "trial.end": [200, 500, 800],
        "recording.end": [1000],
    }
    mock_events = mock_event_timeseries(events_dict, sort=False)
    recording = mock_neon_recording(events=mock_events)

    event_types, events = _load_events_from_recording(recording)
    assert len(event_types) == 4
    assert len(events) == 4

    for event_name in ["recording.begin", "trial.begin", "trial.end", "recording.end"]:
        et = [et for et in event_types if et.name == event_name]
        assert len(et) == 1, (
            f"Expected exactly one event type for {event_name}, got {len(et)}"
        )
        et = et[0]

        assert et.name == event_name
        if "recording" in event_name:
            assert et.uid == event_name, (
                "Expected uid to match name for immutable events"
            )

        assert events[et.uid] == sorted(events_dict[event_name]), (
            f"Timestamps for {event_name} do not match input data"
        )


def test_events__load_events_from_cache() -> None:
    # Should not contain recording.begin / recording.end as these types are
    # not stored in the settings file
    event_names = ["trial.begin", "trial.end", "extra.event"]
    event_types = []
    for idx, event_name in enumerate(event_names):
        et = EventType()
        et._name = event_name
        et._uid = event_name if "recording" in event_name else f"uid{idx}"
        event_types.append(et)

    cached_events = {
        "recording.begin": [0],
        "uid0": [400, 100, 700],  # trial.begin, unsorted
        "uid1": [200, 500, 800],  # trial.end
        "recording.end": [1000],
    }

    event_types, events = _load_events_from_cache(cached_events, event_types)

    assert len(event_types) == 4
    for et in event_types:
        assert et in event_types, "Expected known event types to be used"
        assert et.name != "extra.event", "Only present event types should be returned"
        assert events[et.uid] == sorted(cached_events[et.uid]), (
            "Expected event timestamps to match cache and be sorted during load"
        )


def test_events__load_events_from_dataframe() -> None:
    events_df = pd.DataFrame(
        {
            "name": [
                "recording.begin",
                "trial.begin",
                "trial.end",
                "trial.begin",
                "trial.end",
                "recording.end",
            ],
            "timestamp [ns]": [0, 100, 200, 500, 600, 1000],
        }
    )
    EventType.from_name("trial.begin")
    events = _load_events_from_dataframe(events_df)

    assert len(events) == 2, (
        "Expected events dict to only contain event of mutable type"
    )
    for event_name, ts in zip(
        ["trial.begin", "trial.end"], [[100, 500], [200, 600]], strict=False
    ):
        assert events[event_name] == ts, f"Wrong timestamp for {event_name}"


@pytest.mark.usefixtures("qapp")
def test_events_plugin__on_recording_loaded__from_recording(
    mock_neon_recording: Callable[..., NeonRecording],
) -> None:
    et = EventType.from_name("test.event")

    plugin = EventsPlugin()
    plugin._load_events = MagicMock(
        return_value=([et], {et.uid: [100, 200, 300]}, "recording")
    )
    plugin.save_cached_json = MagicMock()
    plugin.on_recording_loaded(mock_neon_recording())

    assert plugin.event_types == [et], (
        "Expected event types to be loaded from recording"
    )
    assert plugin._events == {et.uid: [100, 200, 300]}, (
        "Expected events to be loaded from recording"
    )
    plugin.save_cached_json.assert_called_once()


@pytest.mark.usefixtures("qapp")
def test_events_plugin__on_recording_loaded__from_cache(
    mock_neon_recording: Callable[..., NeonRecording],
) -> None:
    et = EventType.from_name("test.event")

    plugin = EventsPlugin()
    plugin._load_events = MagicMock(
        return_value=([et], {et.uid: [100, 200, 300]}, "cache")
    )
    plugin.save_cached_json = MagicMock()
    plugin.on_recording_loaded(mock_neon_recording())

    assert plugin.event_types == [], "Expected event types not to be modified"
    assert plugin._events == {et.uid: [100, 200, 300]}, (
        "Expected events to be loaded from cache"
    )
    plugin.save_cached_json.assert_not_called()


@pytest.mark.usefixtures("qapp")
def test_events_plugin__create_event_type() -> None:
    plugin = EventsPlugin()
    et = EventType.from_name("event-1")
    plugin.event_types = [et]
    new_et = plugin.create_event_type()

    assert new_et.name == "event-2", "Expected event type to have a unique name"


def test_events_plugin__add_event_type(qtbot: QtBot) -> None:
    plugin = EventsPlugin()
    et = EventType.from_name("test.event")

    with qtbot.waitSignal(plugin.changed):
        plugin.add_event_type(et)

    assert et in plugin.event_types, (
        "Expected new event type to be added to event_types list"
    )


def test_events_plugin__add_event_type__type_already_exists() -> None:
    plugin = EventsPlugin()
    et = EventType.from_name("recording.begin")
    plugin.event_types = [et]

    with pytest.raises(ValueError, match="already exists"):
        plugin.add_event_type(et)


def test_events_plugin__delete_event_type__no_events(qtbot: QtBot) -> None:
    plugin = EventsPlugin()
    et = EventType.from_name("test.event")
    plugin.event_types = [et]

    with qtbot.waitSignal(plugin.changed):
        plugin.delete_event_type(et)

    assert et not in plugin.event_types, (
        "Expected event type to be removed from event_types list"
    )


def test_events_plugin__delete_event_type__with_events(qtbot: QtBot) -> None:
    plugin = EventsPlugin()
    et = EventType.from_name("test.event")
    plugin.event_types = [et]
    plugin._events = {et.uid: [100, 200, 300]}
    plugin.save_cached_json = MagicMock()

    with qtbot.waitSignal(plugin.changed):
        plugin.delete_event_type(et)

    assert et not in plugin.event_types, (
        "Expected event type to be removed from event_types list"
    )
    assert et.uid not in plugin._events, (
        "Expected events for deleted type to be removed from events dict"
    )
    plugin.save_cached_json.assert_called_once()


def test_events_plugin__delete_event_type__immutable_type() -> None:
    plugin = EventsPlugin()
    et = EventType.from_name("recording.begin")
    plugin.event_types = [et]

    with pytest.raises(ValueError, match="cannot be deleted"):
        plugin.delete_event_type(et)


def test_events_plugin__on_event_name_changed() -> None:
    plugin = EventsPlugin()
    et = EventType.from_name("test.event")
    plugin.event_types = [et]

    et._name = "renamed.event"

    # Simulate the name_changed signal being emitted
    plugin._on_event_name_changed("test.event", "renamed.event", et)

    assert "test.event" not in plugin._event_types_by_name
    assert "renamed.event" in plugin._event_types_by_name


def test_events_plugin__add_events__adds_new_event_types(qtbot: QtBot) -> None:
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


def test_events_plugin__add_events__reuses_existing_event_types() -> None:
    plugin = EventsPlugin()
    existing_et = EventType.from_name("existing.event")
    plugin.event_types = [existing_et]
    plugin._events = {existing_et.uid: [0]}
    plugin.save_cached_json = MagicMock()

    plugin.add_events({"existing.event": [100, 200, 300]})

    assert len(plugin.event_types) == 1, "Expected no new event type to be added"
    assert plugin.event_types[0] is existing_et, (
        "Expected existing event type to be reused"
    )
    assert plugin._events == {existing_et.uid: [0, 100, 200, 300]}, (
        "Expected events to be appended to the existing ones"
    )
    plugin.save_cached_json.assert_called_once()


def test_events_plugin__add_events__raises_on_immutable_type() -> None:
    plugin = EventsPlugin()
    et = EventType.from_name("recording.begin")
    plugin.event_types = [et]

    with pytest.raises(ValueError, match="cannot be added"):
        plugin.add_events({"recording.begin": [100, 200]})


def test_events_plugin__add_events__sorts_timestamps() -> None:
    plugin = EventsPlugin()
    et = EventType.from_name("test.event")
    plugin.event_types = [et]
    plugin._events = {et.uid: [100, 200, 300]}
    plugin.save_cached_json = MagicMock()

    plugin.add_events({"test.event": [150, 250]})

    assert plugin._events == {et.uid: [100, 150, 200, 250, 300]}, (
        "Expected events to be sorted"
    )
    plugin.save_cached_json.assert_called_once()


def test_events_plugin__delete_events__deletes_events() -> None:
    plugin = EventsPlugin()
    et = EventType.from_name("test.event")
    plugin.event_types = [et]
    plugin._events = {et.uid: [100, 200, 300, 400]}
    plugin.save_cached_json = MagicMock()

    plugin.delete_events({"test.event": [200, 300]})

    assert sorted(plugin._events[et.uid]) == [100, 400], (
        "Expected specified events to be deleted"
    )
    plugin.save_cached_json.assert_called_once()


def test_events_plugin__delete_events__removes_event_type_if_no_events_left(
    qtbot: QtBot,
) -> None:
    plugin = EventsPlugin()
    et = EventType.from_name("test.event")
    plugin.event_types = [et]
    plugin._events = {et.uid: [100, 200]}
    plugin.save_cached_json = MagicMock()

    with qtbot.waitSignal(plugin.changed):
        plugin.delete_events({"test.event": [100, 200]}, remove_empty_types=True)

    assert et.name not in plugin._event_types_by_name, (
        "Expected event type to be removed from event_types_by_name dict"
    )
    assert et.uid not in plugin._events, (
        "Expected events for deleted type to be removed from events dict"
    )
    plugin.save_cached_json.assert_called_once()


def test_events_plugin__delete_events__raises_on_immutable_type() -> None:
    plugin = EventsPlugin()
    et = EventType.from_name("recording.begin")
    plugin.event_types = [et]
    plugin._events = {"recording.begin": [0]}

    with pytest.raises(ValueError, match="cannot be deleted"):
        plugin.delete_events({"recording.begin": [0]})


def test_events_plugin__delete_events__sorts_timestamps() -> None:
    plugin = EventsPlugin()
    et = EventType.from_name("test.event")
    plugin.event_types = [et]
    plugin._events = {et.uid: [100, 200, 300, 400]}
    plugin.save_cached_json = MagicMock()

    plugin.delete_events({"test.event": [200, 300]})

    assert plugin._events == {et.uid: [100, 400]}, "Expected events to be stored"
    plugin.save_cached_json.assert_called_once()


def _prepare_test_data(
    mock_neon_recording: Callable[..., NeonRecording],
) -> tuple[EventsPlugin, NeonRecording]:
    plugin = EventsPlugin()
    mock_events = mock_event_timeseries(
        {
            "recording.begin": [0],
            "event.from.rec": [100, 400, 700],
            "event.same.ts": [100],
            "recording.end": [1000],
        }
    )
    et_rec = EventType.from_name("event.from.rec")
    et_player = EventType.from_name("event.from.player")
    et_same_ts = EventType.from_name("event.same.ts")
    plugin.event_types = [et_rec, et_player, et_same_ts]
    plugin._events = {
        "recording.begin": [0],
        et_rec.uid: [100, 400, 700],
        et_player.uid: [200, 500, 800],
        et_same_ts.uid: [100],
        "recording.end": [1000],
    }
    mock_recording = mock_neon_recording(
        events=mock_events, info={"recording_id": "test.recording"}
    )
    return plugin, mock_recording


def test_events_plugin__find_closest_event(
    mock_neon_recording: Callable[..., NeonRecording],
) -> None:
    plugin, _ = _prepare_test_data(mock_neon_recording)
    et = plugin._event_types_by_name["event.from.rec"]
    assert plugin._find_closest_event(et, 100) == 100


def test_events_plugin__find_closest_event__missing_type(
    mock_neon_recording: Callable[..., NeonRecording],
) -> None:
    plugin, _ = _prepare_test_data(mock_neon_recording)
    et = EventType.from_name("non.existent.event")
    assert plugin._find_closest_event(et, 100) is None


def test_events_plugin__find_closest_event__tolerance(
    mock_neon_recording: Callable[..., NeonRecording],
) -> None:
    plugin, _ = _prepare_test_data(mock_neon_recording)
    et = plugin._event_types_by_name["event.from.rec"]
    assert plugin._find_closest_event(et, 110, tolerance_ns=15) == 100


def test_events_plugin__events__uses_names_not_ids(
    mock_neon_recording: Callable[..., NeonRecording],
) -> None:
    plugin, _ = _prepare_test_data(mock_neon_recording)
    event_names = [
        "recording.begin",
        "event.from.rec",
        "event.from.player",
        "event.same.ts",
        "recording.end",
    ]
    for event_name in event_names:
        assert event_name in plugin.events, (
            f"Expected {event_name} to be present in events property"
        )


def test_events_plugin__prepare_events_export__whole_time_range(
    mock_neon_recording: Callable[..., NeonRecording],
) -> None:
    plugin, mock_recording = _prepare_test_data(mock_neon_recording)
    events_df = plugin._prepare_events_export(mock_recording, export_window=(0, 1000))
    assert len(events_df) == 9, "Expected all events to be included in export"


def test_events_plugin__prepare_events_export__custom_time_range(
    mock_neon_recording: Callable[..., NeonRecording],
) -> None:
    plugin, mock_recording = _prepare_test_data(mock_neon_recording)
    events_df = plugin._prepare_events_export(mock_recording, export_window=(200, 750))
    assert len(events_df) == 4, (
        "Expected only events within the custom time range to be included in export"
    )
    for event_to_include in ["event.from.rec", "event.from.player"]:
        assert len(events_df[events_df["name"] == event_to_include]) == 2, (
            f"Expected two {event_to_include} events in export"
        )


def test_events_plugin__prepare_events_export__timestamps_are_sorted(
    mock_neon_recording: Callable[..., NeonRecording],
) -> None:
    plugin, mock_recording = _prepare_test_data(mock_neon_recording)
    events_df = plugin._prepare_events_export(mock_recording, export_window=(150, 450))
    assert events_df["name"].tolist() == ["event.from.player", "event.from.rec"], (
        "Expected events to be sorted by timestamp in export"
    )


def test_events_plugin__prepare_events_export__source(
    mock_neon_recording: Callable[..., NeonRecording],
) -> None:
    plugin, mock_recording = _prepare_test_data(mock_neon_recording)
    events_df = plugin._prepare_events_export(mock_recording, export_window=(0, 1000))
    for event_name, source in zip(
        ["event.from.rec", "event.from.player", "event.same.ts"],
        ["recording", "player", "recording"],
        strict=False,
    ):
        assert all(events_df[events_df["name"] == event_name]["type"] == source), (
            f"Expected all {event_name} events to be labeled as {source} in export"
        )


def test_events_plugin__to_dict__recursive(
    mock_neon_recording: Callable[..., NeonRecording],
) -> None:
    plugin, _ = _prepare_test_data(mock_neon_recording)
    plugin_dict = plugin.to_dict(recursive=True)
    plugin_rec = EventsPlugin.from_dict(plugin_dict)

    for et_dict in plugin_dict["event_types"]:
        assert isinstance(et_dict, dict), (
            "Expected event types to be serialized as dicts in recursive mode"
        )

        event_name = et_dict["name"]
        assert event_name in plugin._event_types_by_name, (
            "Expected event name in serialized dict to match plugin event types"
        )
        assert event_name in plugin_rec._event_types_by_name, (
            "Expected deserialized plugin to contain the same event types "
            "as the original"
        )

        et_rec = plugin_rec._event_types_by_name[event_name]
        assert et_dict["uid"] == et_rec.uid, (
            "Expected deserialized event type uid to match the original event type"
        )
        assert et_dict["shortcut"] == et_rec.shortcut, (
            "Expected deserialized event type shortcut to match the original event type"
        )


def test_events_plugin__global_properties__to_dict__includes_class_name() -> None:
    plugin = EventsPlugin()
    plugin_dict = plugin.global_properties.to_dict()
    assert plugin_dict["__class__"] == "EventsPluginGlobalProps"


@pytest.fixture
def events_gui(
    qapp: MockNeonPlayerApp, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[EventsPlugin, TimeLineDock]]:
    monkeypatch.setattr(qapp, "recording", None)
    monkeypatch.setattr(qapp, "playback_speed_options", [-1.0, 1.0], raising=False)
    timeline = TimeLineDock()
    monkeypatch.setattr(
        qapp, "recording", SimpleNamespace(start_time=0, stop_time=1000)
    )
    window = SimpleNamespace(
        timeline=timeline,
        register_action=Mock(),
        unregister_action=Mock(),
        sort_action_menu=Mock(),
        remove_menu_if_empty=Mock(),
    )
    monkeypatch.setattr(qapp, "main_window", window, raising=False)
    monkeypatch.setattr(EventsPlugin, "headless", property(lambda _: False))
    plugin = EventsPlugin()
    monkeypatch.setattr(qapp, "plugins_by_class", {"EventsPlugin": plugin})
    monkeypatch.setattr(plugin, "save_cached_json", Mock())
    yield plugin, timeline
    if plugin._timeline_group is not None:
        plugin.on_disabled()
    plugin.deleteLater()
    timeline.close()
    timeline.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_events_contributes_scoped_rows_actions_and_data(
    events_gui: tuple[EventsPlugin, TimeLineDock],
) -> None:
    plugin, timeline = events_gui
    group = timeline.get_timeline_group("Events")
    assert group is not None
    assert not group.collapsed
    assert plugin._ensure_timeline_group() is group
    header_key = TimelineRowKey("Events", "Events", TimelineRowRole.HEADER)
    header = timeline.timeline_plots[header_key]
    for name in ("Events", "Search", ""):
        event_type = EventType.from_name(name)
        plugin.add_event_type(event_type)
        plugin.add_events({name: [100, 300]})
        plot = timeline.get_timeline_plot(name, group=group)
        assert plot is not None
        assert plot is not header
        assert plot.row_key == TimelineRowKey("Events", name, TimelineRowRole.CUSTOM)
        assert list(plot.items[0].getOriginalDataset()[0]) == [100, 300]
        assert f"Delete {name} instance" in timeline.data_point_actions[plot.row_key]
        assert timeline.get_timeline_plot(f"Events - {name}") is None
    assert group.child_row_names == {"Events", "Search", ""}
    assert header.items == []


def test_event_rename_and_deletion_preserve_group_and_other_scopes(
    events_gui: tuple[EventsPlugin, TimeLineDock],
) -> None:
    plugin, timeline = events_gui
    group = plugin._timeline_group
    assert group is not None
    event_type = EventType.from_name("Original")
    plugin.add_event_type(event_type)
    plugin.add_events({"Original": [100, 200]})
    other = timeline.add_timeline_group("Other")
    other_plot = timeline.add_timeline_scatter("Original", [], group=other)
    global_plot = timeline.add_timeline_scatter("Original", [])
    timeline.timeline_group_controls[group.name].click()
    event_type.name = ""
    assert timeline.get_timeline_plot("Original", group=group) is None
    renamed = timeline.get_timeline_plot("", group=group.name)
    assert renamed is not None
    assert not renamed.isVisible()
    assert list(renamed.items[0].getOriginalDataset()[0]) == [100, 200]
    old_key = TimelineRowKey(group.name, "Original", TimelineRowRole.CUSTOM)
    assert old_key not in timeline.data_point_actions
    assert timeline.get_timeline_plot("Original", group=other) is other_plot
    assert timeline.get_timeline_plot("Original") is global_plot
    event_type.name = "Events"
    assert timeline.get_timeline_plot("", group=group) is None
    plugin.delete_event_type(event_type)
    assert not group.child_row_names
    assert timeline.get_timeline_group("Events") is group
    timeline.remove_timeline_group(other)


def test_immutable_event_rows_keep_scoped_navigation_actions(
    events_gui: tuple[EventsPlugin, TimeLineDock],
) -> None:
    plugin, timeline = events_gui
    event_type = EventType.from_name("recording.begin")
    plugin._immutable_event_types = [event_type]
    plugin._update_gui_for_event_types(event_types_to_add=[event_type])
    plot = timeline.get_timeline_plot(event_type.name, group="Events")
    assert plot is not None
    actions = timeline.data_point_actions[plot.row_key]
    assert "Seek to this recording.begin" in actions
    assert "Set this recording.begin as the left export window boundary" in actions
    assert "Delete recording.begin instance" not in actions


def test_events_disable_without_recording_cleans_group_and_reenable_is_fresh(
    events_gui: tuple[EventsPlugin, TimeLineDock],
    qapp: MockNeonPlayerApp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin, timeline = events_gui
    event_type = EventType.from_name("Trial")
    plugin.add_event_type(event_type)
    old_group = plugin._timeline_group
    assert old_group is not None
    timeline.timeline_group_controls[old_group.name].click()
    global_plot = timeline.get_timeline_plot("Trial", True)
    monkeypatch.setattr(qapp, "recording", None)
    plugin.on_disabled()
    assert plugin._timeline_group is None
    assert timeline.get_timeline_group("Events") is None
    assert timeline.get_timeline_plot("Trial") is global_plot
    fresh = EventsPlugin()
    new_group = timeline.get_timeline_group("Events")
    assert new_group is not None
    assert new_group is not old_group
    assert not new_group.collapsed
    assert not new_group.child_row_names
    fresh.on_disabled()
    fresh.deleteLater()


@pytest.mark.usefixtures("qapp")
def test_headless_events_creates_no_group() -> None:
    plugin = EventsPlugin()
    assert plugin.headless
    assert plugin._timeline_group is None
    plugin.deleteLater()


@pytest.mark.parametrize("replace_group", [False, True])
def test_events_disable_never_reacquires_a_removed_group(
    events_gui: tuple[EventsPlugin, TimeLineDock],
    monkeypatch: pytest.MonkeyPatch,
    replace_group: bool,
) -> None:
    plugin, timeline = events_gui
    plugin.add_event_type(EventType.from_name("Trial"))
    original = plugin._timeline_group
    assert original is not None
    timeline.remove_timeline_group(original)
    replacement = None
    replacement_plot = None
    callback = Mock()
    if replace_group:
        replacement = timeline.add_timeline_group("Events")
        replacement_plot = timeline.add_timeline_scatter("Trial", [], group=replacement)
        timeline.register_data_point_action(
            "Trial", "Keep this action", callback, group=replacement
        )
    create_group = Mock(wraps=timeline.add_timeline_group)
    monkeypatch.setattr(timeline, "add_timeline_group", create_group)
    plugin.on_disabled()
    create_group.assert_not_called()
    assert plugin._timeline_group is None
    assert timeline.get_timeline_group("Events") is replacement
    if replacement is not None:
        assert (
            timeline.get_timeline_plot("Trial", group=replacement) is replacement_plot
        )
        key = TimelineRowKey("Events", "Trial", TimelineRowRole.CUSTOM)
        assert timeline.data_point_actions[key]["Keep this action"] is callback
        timeline.remove_timeline_group(replacement)
