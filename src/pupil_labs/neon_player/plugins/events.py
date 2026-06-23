import numpy as np
import pandas as pd
import logging
import typing as T

from pathlib import Path
from pupil_labs.neon_recording import NeonRecording
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QIcon, QKeyEvent
from PySide6.QtWidgets import QMessageBox
from qt_property_widgets.utilities import (
    FilePath,
    PersistentPropertiesMixin,
    action_params,
    property_params,
)
from qt_property_widgets.widgets import ValueListWidget, ValueListItemWidget

from pupil_labs import neon_player
from pupil_labs.neon_player import GlobalPluginProperties, action
from pupil_labs.neon_player.plugins.shared import run_export_across_recordings
from pupil_labs.neon_player.plugins import Plugin
from pupil_labs.neon_player.ui import ListPropertyAppenderAction

IMMUTABLE_EVENTS = ["recording.begin", "recording.end"]


class EventsPluginGlobalProps(GlobalPluginProperties):
    def __init__(self) -> None:
        super().__init__()
        self._global_event_types: list[str] = []

    @property
    def global_event_types(self) -> list[str]:
        return self._global_event_types

    @global_event_types.setter
    def global_event_types(self, value: list[str]) -> None:
        self._global_event_types = value


class EventType(PersistentPropertiesMixin, QObject):
    changed = Signal()
    name_changed = Signal(str, str)

    def __init__(self) -> None:
        super().__init__()
        self._name = ""
        self._shortcut = ""
        self._uid = ""
        self._source = "recording"

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if self._name == value:
            return

        plugin = EventsPlugin.instance()
        if plugin is not None and value in IMMUTABLE_EVENTS:
            QMessageBox.warning(
                None,
                "Invalid event name",
                f"Event cannot be renamed to '{value}' as this name is reserved. "
                f"Please choose a different name.",
            )
            return

        if plugin is not None and value in plugin._event_types_by_name:
            QMessageBox.warning(
                None,
                "Duplicate event name",
                f"Event '{value}' already exists. Please choose a different name.",
            )
            return

        old_name = self._name
        self._name = value
        self.name_changed.emit(old_name, value)

    @property
    @property_params(max_length=1)
    def shortcut(self) -> str:
        return self._shortcut

    @shortcut.setter
    def shortcut(self, value: str) -> None:
        self._shortcut = value

    @property
    @property_params(widget=None, dont_encode=True)
    def source(self) -> str:
        return self._source

    @source.setter
    def source(self, value: str) -> None:
        if value not in ["recording", "workspace"]:
            raise ValueError(f"Invalid event type source: {value}")
        self._source = value

    @property
    @property_params(widget=None)
    def uid(self) -> str:
        """
        This field should always have the same value as `name`. It is kept for compatibility
        with the original implementation of the plugin that used UIDs as the primary identifier
        for event types.
        """
        return self._uid

    @uid.setter
    def uid(self, value: str) -> None:
        self._uid = value

    @staticmethod
    def from_name(name: str) -> "EventType":
        et = EventType()
        et._name = name
        et._uid = name
        return et


class EventTypeListWidget(ValueListWidget):
    @staticmethod
    def from_property_impl(prop: property) -> "EventTypeListWidget":
        source_params: dict[str, T.Any] = {}
        if hasattr(prop, "fget") and prop.fget is not None:
            source_params = prop.fget.parameters if hasattr(prop.fget, "parameters") else {}

        return EventTypeListWidget(EventType, source_params)

    @staticmethod
    def from_type(cls: type) -> "EventTypeListWidget":
        return EventTypeListWidget(cls, {})

    def on_add_button_clicked(self) -> None:
        plugin = EventsPlugin.instance()
        if plugin is None:
            return

        new_event_name = plugin._get_unique_event_name()
        new_event_type = EventType.from_name(new_event_name)
        plugin.add_event_type(new_event_type)
        self.add_item(new_event_type)

    def remove_item(self, item_widget: ValueListItemWidget) -> None:
        plugin = EventsPlugin.instance()
        if plugin is None:
            return True

        if item_widget.item_widget is None:
            return

        event_type = item_widget.item_widget.value
        if not self._confirm_removal(plugin, event_type):
            return

        plugin.delete_event_type(event_type)
        super().remove_item(item_widget)

    def _confirm_removal(self, plugin: "EventsPlugin", event_type: str) -> bool:
        events_to_delete, affected_recordings = plugin._count_events_across_workspace(event_type)
        if events_to_delete == 0:
            return True

        suffix = "" if events_to_delete == 1 else "s"
        if len(affected_recordings) > 1:
            recordings_note = f" across {len(affected_recordings)} recordings"
        else:
            recordings_note = " in the current recording"
            affected_name = affected_recordings[0]
            if plugin.recording._rec_dir.name != affected_name:
                recordings_note = f" in recording {affected_name}"

        confirmed = plugin.user_confirm(
            "Confirm event type deletion",
            f"Deleting event type '{event_type.name}' will also delete its {events_to_delete}"
            f" instance{suffix}{recordings_note}. Do you want to proceed?",
        )
        return confirmed


def _load_events_from_recording(
    recording: NeonRecording, global_event_types: list[str] = []
) -> tuple[list[EventType], dict[str, list[int]]]:
    """
    Loads events from the recording and additionally creates event types that
    are defined in global settings.

    Returns all created event types and the events as {event name: list of timestamps}.
    """
    event_type_cache: dict[str, EventType] = {}
    events: dict[str, list[int]] = {}

    for event in recording.events:
        event_name = str(event.event)

        # Look up or create the event type
        et = event_type_cache.get(event_name, None)
        if et is None:
            et = EventType.from_name(event_name)
            event_type_cache[event_name] = et

        # Add event to the dictionary
        if et.name not in events:
            events[et.name] = []
        events[et.name].append(int(event.time))

    for event_name in global_event_types:
        if event_name in event_type_cache:
            continue

        event_type_cache[event_name] = EventType.from_name(event_name)

    return list(event_type_cache.values()), events


def _load_events_from_cache(
    cached_events: dict, known_event_types: list[EventType]
) -> tuple[list[EventType], dict]:
    """
    Load event data from a cache stored in events.json. All event types are expected
    to either be immutable (recording.begin, recording.end) or have been previously
    defined and stored in the known_event_types list.

    Returns event types that are present in the cache as well as events themselves.
    """
    known_event_types_by_uid = {et.uid: et for et in known_event_types}
    for event_name in IMMUTABLE_EVENTS:
        et = EventType.from_name(event_name)
        known_event_types_by_uid[et.uid] = et

    event_type_cache = {}
    events = {}
    for uid in cached_events:
        if uid in event_type_cache:
            continue

        if uid not in known_event_types_by_uid:
            raise ValueError(f"Event type with uid {uid} not found")

        et = known_event_types_by_uid[uid]
        event_type_cache[uid] = et
        events[et.name] = cached_events[uid]

    return list(event_type_cache.values()), events


def _load_events_from_dataframe(events_df: pd.DataFrame) -> dict[str, list[int]]:
    events = {}
    for name, group in events_df.groupby("name"):
        name = str(name)
        if name in IMMUTABLE_EVENTS:
            logging.warning(f"Skipping immutable event '{name}' from imported CSV")
            continue

        events[name] = group["timestamp [ns]"].astype(int).tolist()

    return events


class WorkspaceEventIndex:
    """
    Maintains an index of all events and the number of their occurrences
    across all recordings in the workspace to ensure correct renaming and deletion.
    """
    def __init__(self, plugin: "EventsPlugin") -> None:
        self.plugin = plugin

        # {event_name: {recording_name: event_count}}
        self.events: dict[str, dict[str, int]] = {}

        # Names of recording that the index is based on
        self.recording_names: set[str] = set()

    def load(self) -> None:
        data = self.plugin.load_cached_json("workspace-events.json", workspace=True)
        if data is None:
            return

        self.events = data.get("events", {})
        self.recording_names = set(data.get("recording_names", []))

    def save(self) -> None:
        data = {
            "events": self.events,
            "recording_names": list(self.recording_names)
        }
        self.plugin.save_cached_json("workspace-events.json", data, workspace=True)

    def update(self, load: bool = False, save: bool = True) -> None:
        if load:
            self.load()

        recording_name = self.plugin.recording._rec_dir.name
        if recording_name not in self.recording_names:
            self.recording_names.add(recording_name)

        events_to_process = self.plugin.events
        for event_name, timestamps in events_to_process.items():
            if event_name in IMMUTABLE_EVENTS:
                continue

            if event_name not in self.events:
                self.events[event_name] = {}

            # Clean up the entries if:
            #  - all events of this type were deleted from the current recording
            #  - this event type was deleted across all recordings
            if not timestamps and recording_name in self.events[event_name]:
                del self.events[event_name][recording_name]
                if not self.events[event_name]:
                    del self.events[event_name]
                continue

            if timestamps:
                self.events[event_name][recording_name] = len(timestamps)

        if save:
            self.save()

    def add_event(self, event_name: str, recording_name: str) -> None:
        if event_name not in self.events:
            self.events[event_name] = {}

        if recording_name not in self.events[event_name]:
            self.events[event_name][recording_name] = 0

        self.events[event_name][recording_name] += 1
        self.save()

    def delete_event(self, event_name: str, recording_name: str) -> None:
        if event_name not in self.events or recording_name not in self.events[event_name]:
            return

        self.events[event_name][recording_name] -= 1

        if self.events[event_name][recording_name] <= 0:
            del self.events[event_name][recording_name]
        if not self.events[event_name]:
            del self.events[event_name]

        self.save()

    def rename_event(self, old_name: str, new_name: str, recording_name: str) -> None:
        if old_name not in self.events:
            return

        if new_name not in self.events:
            self.events[new_name] = {}

        self.events[new_name][recording_name] = self.events[old_name].pop(recording_name)
        if not self.events[old_name]:
            del self.events[old_name]

        self.save()

    def delete_event_type(self, event_type: EventType) -> None:
        if event_type.name in self.events:
            del self.events[event_type.name]
            self.save()


class EventsPlugin(neon_player.Plugin):
    label = "Events"
    global_properties = EventsPluginGlobalProps()

    def __init__(self) -> None:
        super().__init__()
        self._event_types_by_name: dict[str, EventType] = {}
        self.events: dict[str, list[int]] = {}
        self._workspace_index: WorkspaceEventIndex = WorkspaceEventIndex(self)

        if self.headless:
            return

        self.get_timeline().key_pressed.connect(self._on_key_pressed)
        self.header_action = ListPropertyAppenderAction(
            "event_types", "+ Add event type"
        )

    @staticmethod
    def instance() -> T.Union["EventsPlugin", None]:
        if neon_player.instance() is None:
            return None

        plugin = Plugin.get_instance_by_name("EventsPlugin")
        if plugin is not None and not isinstance(plugin, EventsPlugin):
            raise RuntimeError("Invalid instance of EventsPlugin found")

        return plugin

    def _load_events(self, recording: NeonRecording) -> tuple[list[EventType], dict, str]:
        logging.debug(f"Loading events for recording {recording._rec_dir.name}")
        events: dict[str, list[int]] = {}
        event_types: list[EventType] = []

        try:
            cached_events = self.load_cached_json("events.json")
        except Exception:
            logging.exception("Failed to load events json")
            cached_events = None

        if cached_events is not None:
            event_types, events = _load_events_from_cache(
                cached_events, list(self._event_types_by_name.values())
            )
            return event_types, events, "cache"

        event_types, events = _load_events_from_recording(
            recording, self.global_properties.global_event_types
        )

        return event_types, events, "recording"

    def _scan_events_in_workspace(self):
        # Update the index for the current recording to account for any changes
        # outside of the workspace mode
        self._workspace_index.update(load=True)

        # If all recordings have already been scanned, update the UI immediately,
        # otherwise launch a batch background job
        workspace_recording_names = {rec._rec_dir.name for rec in self.workspace.recordings}
        scanned_recording_names = self._workspace_index.recording_names
        missing_recordings = workspace_recording_names - scanned_recording_names
        if not missing_recordings:
            self._on_workspace_event_scan_finished()
            return

        recordings_to_scan = [
            rec for rec in self.workspace.recordings
            if rec._rec_dir.name in missing_recordings
        ]
        batch_job = self.job_manager.run_background_batch_action(
            "Scan events in workspace",
            "EventsPlugin._bg_update_workspace_event_index",
            recordings=recordings_to_scan
        )
        batch_job.finished.connect(self._on_workspace_event_scan_finished)

    def _bg_update_workspace_event_index(self) -> None:
        self._workspace_index.update(load=True)

    def _on_workspace_event_scan_finished(self):
        self._workspace_index.load()

        types_to_add = []
        for event_name in self._workspace_index.events:
            if event_name in self._event_types_by_name:
                continue

            event_type = EventType.from_name(event_name)
            types_to_add.append(event_type)

        if types_to_add:
            self.event_types = self.event_types + types_to_add
            self._update_gui_for_event_types(event_types_to_add=types_to_add)

    def on_recording_loaded(self, recording: NeonRecording) -> None:
        event_types, events, source = self._load_events(recording)
        self.events = events

        if source == "recording":
            # When loading from recording, only mutable event types need to be saved,
            # but the UI needs to be set up for all event types
            mutable_event_types = filter(lambda et: et.name not in IMMUTABLE_EVENTS, event_types)
            event_types_to_setup_ui_for = event_types
            self.event_types = list(mutable_event_types)
            self.save_cached_json("events.json", events)
        elif source == "cache":
            # When loading from cache, all event types are expected to have been loaded
            # from plugin settings, so self.event_types is already correct, but the UI
            # still needs to be set up for stored and immutable event types
            immutable_event_types = filter(lambda et: et.name in IMMUTABLE_EVENTS, event_types)
            event_types_to_setup_ui_for = self.event_types + list(immutable_event_types)

            # Force the UIDs of all event types to match their names. This way, it becomes
            # easier to match events with the same name across recordings
            event_types_renamed = False
            for et in self._event_types_by_name.values():
                if et.name != et.uid:
                    et.uid = et.name
                    event_types_renamed = True
            if event_types_renamed:
                self.save_cached_json("events.json", events)

        logging.info(f"Loaded {sum(len(v) for v in self.events.values())} events")

        if self.headless:
            return

        need_to_scan_workspace = self.app.batch_mode_enabled and self.workspace.size > 1
        if need_to_scan_workspace:
            self._scan_events_in_workspace()

        self._update_gui_for_event_types(event_types_to_add=event_types_to_setup_ui_for)

    def on_disabled(self) -> None:
        if self.headless or self.recording is None:
            return

        event_types = list(self._event_types_by_name.values())
        self._update_gui_for_event_types(event_types_to_remove=event_types)

    def _update_gui_for_event_types(
        self,
        event_types_to_add: list[EventType] = [],
        event_types_to_remove: list[EventType] = []
    ) -> None:
        if self.headless:
            return

        if not event_types_to_add and not event_types_to_remove:
            return

        # Disable plot sorting to improve performance for recording with many events
        timeline = self.get_timeline()
        plot_sorting_was_enabled = timeline.disable_plot_sorting()

        for et in event_types_to_add:
            et.changed.connect(self.changed.emit)
            et.name_changed.connect(
                lambda old, new, et=et: self._on_event_name_changed(old, new, et)
            )
            self._setup_gui_for_event_type(et)
            self._update_timeline_data(et)

        for et in event_types_to_remove:
            self._remove_gui_for_event_name(et.name)

        # Enable plot sorting, sort rows if any were added or removed
        if plot_sorting_was_enabled:
            timeline.enable_plot_sorting()

    def _setup_gui_for_event_type(self, event_type: EventType) -> None:
        if self.headless:
            return

        timeline = self.get_timeline()
        existing_plot = timeline.get_timeline_plot(
            f"Events - {event_type.name}", create_if_missing=False
        )
        if existing_plot is not None:
            return

        plot_item = timeline.add_timeline_scatter(f"Events - {event_type.name}", [])
        if plot_item is None:
            return
        plot_item.getViewBox().allow_y_panning = False

        if event_type.name not in IMMUTABLE_EVENTS:
            self.register_timeline_action(
                f"Add Event/{event_type.name}", None, lambda: self._add_event(event_type)
            )
            self.app.main_window.sort_action_menu("Timeline/Add Event")

        self.register_data_actions(event_type)

    def register_data_actions(self, event_type: EventType) -> None:
        if event_type.name not in IMMUTABLE_EVENTS:
            self.register_data_point_action(
                f"Events - {event_type.name}",
                f"Delete {event_type.name} instance",
                lambda dp, et=event_type: self.delete_event_instance(dp, et),
            )

        self.register_data_point_action(
            f"Events - {event_type.name}",
            f"Seek to this {event_type.name}",
            self.seek_to_event_instance,
        )
        self.register_data_point_action(
            f"Events - {event_type.name}",
            f"Set this {event_type.name} as the left export window boundary",
            lambda dp, et=event_type: self.set_event_as_export_boundary(dp, et, left=True),
        )
        self.register_data_point_action(
            f"Events - {event_type.name}",
            f"Set this {event_type.name} as the right export window boundary",
            lambda dp, et=event_type: self.set_event_as_export_boundary(dp, et, left=False),
        )

    def _on_key_pressed(self, event: QKeyEvent) -> None:
        key_text = event.text().lower()
        if key_text == "":
            return

        for event_type in self._event_types_by_name.values():
            if event_type.shortcut.lower() == key_text:
                self._add_event(event_type)

    def _remove_gui_for_event_name(self, event_name: str) -> None:
        if self.headless:
            return

        timeline = self.get_timeline()
        timeline.remove_timeline_plot(f"Events - {event_name}")
        data_point_actions = [
            f"Seek to this {event_name}",
            f"Set this {event_name} as the left export window boundary",
            f"Set this {event_name} as the right export window boundary",
        ]
        for action_name in data_point_actions:
            timeline.unregister_data_point_action(
                f"Events - {event_name}", action_name
            )

        if event_name in IMMUTABLE_EVENTS:
            return

        timeline.unregister_data_point_action(
            f"Events - {event_name}", f"Delete {event_name} instance"
        )
        self.unregister_timeline_action(f"Add Event/{event_name}")
        self.app.main_window.remove_menu_if_empty("Timeline/Add Event")

    def _get_unique_event_name(self) -> str:
        event_name_counter = 1
        while True:
            candidate_name = f"event-{event_name_counter}"
            if candidate_name not in self._event_types_by_name:
                return candidate_name
            event_name_counter += 1

    def add_event_type(self, event_type: EventType) -> None:
        event_name = event_type.name
        if event_name in self._event_types_by_name:
            raise ValueError(
                f"Event type {event_name} already exists."
            )

        self._event_types_by_name[event_name] = event_type

        self._update_gui_for_event_types(event_types_to_add=[event_type])
        self.changed.emit()

    def delete_event_type(self, event_type: EventType) -> None:
        if event_type.name in IMMUTABLE_EVENTS:
            raise ValueError(
                f"Event type {event_type.name} cannot be deleted."
            )

        if event_type.name not in self._event_types_by_name:
            return

        if event_type.name in self.events:
            del self.events[event_type.name]
            self.save_cached_json("events.json", self.events)

        del self._event_types_by_name[event_type.name]
        if self.app.batch_mode_enabled and not self.headless:
            self._delete_event_type_across_workspace(event_type)
        self._update_gui_for_event_types(event_types_to_remove=[event_type])
        self.changed.emit()

    def _delete_event_type_across_workspace(self, event_type: EventType) -> None:
        affected_recording_names = self._workspace_index.events.get(event_type.name, {})
        if not affected_recording_names:
            self._workspace_index.delete_event_type(event_type)
            self._workspace_index.save()
            return

        affected_recordings = [
            rec for rec in self.workspace.recordings
            if rec._rec_dir.name in affected_recording_names
        ]
        self.job_manager.run_background_batch_action(
            f"Delete event type across workspace",
            "EventsPlugin.delete_event_type",
            args_generator=lambda _: [event_type],
            recordings=affected_recordings
        )

    def _bg_delete_event_type_by_name(self, event_name: str) -> None:
        if event_name not in self._event_types_by_name:
            return

        event_type = self._event_types_by_name[event_name]
        self.delete_event_type(event_type)

    def _add_event(self, event_type: EventType, ts: int | None = None) -> None:
        if self.recording is None:
            return

        if ts is None:
            ts = self.app.current_ts

        self.add_events({event_type.name: [ts]})

    def _find_closest_event(
        self, event_type: EventType, target_ts: int, tolerance_ns: int = 5
    ) -> int | None:
        if event_type.name not in self.events:
            return None

        events_list = self.events[event_type.name]
        if not events_list:
            return None

        closest_event = min(events_list, key=lambda x: abs(x - target_ts))
        if abs(closest_event - target_ts) > tolerance_ns:
            return None

        return closest_event

    def delete_event_instance(
        self, data_point: tuple[int, T.SupportsFloat], event_type: EventType
    ) -> None:
        if event_type.name in IMMUTABLE_EVENTS:
            raise ValueError(
                f"Instances of {event_type.name} event cannot be deleted."
            )

        closest_event = self._find_closest_event(event_type, data_point[0])
        if closest_event is None:
            return

        self.delete_events({event_type.name: [closest_event]})

    def seek_to_event_instance(self, data_point: tuple[int, T.SupportsFloat]) -> None:
        self.app.seek_to(data_point[0])

    def set_event_as_export_boundary(
        self, data_point: tuple[int, T.SupportsFloat], event_type: EventType, left: bool
    ) -> None:
        closest_event = self._find_closest_event(event_type, data_point[0])
        if closest_event is None:
            return

        current_export_window = self.app.get_export_window()
        if left:
            new_window = (closest_event, current_export_window[1])
        else:
            new_window = (current_export_window[0], closest_event)
        self.app.set_export_window(new_window)

    def _update_timeline_data(self, event_type: EventType) -> None:
        if self.headless:
            return

        timeline = self.get_timeline()
        event_name = event_type.name
        plot_item = timeline.get_timeline_plot(f"Events - {event_name}")
        if plot_item is None or not plot_item.items:
            return

        x = np.array(self.events.get(event_type.name, []))
        y = np.zeros_like(x)
        plot_item.items[0].setData(x, y)

    @property
    @property_params(
        widget=EventTypeListWidget,
        item_params={"label_field": "name"},
        prevent_add=True,
        primary=True,
        scope="recording",
    )
    def event_types(self) -> list[EventType]:
        return list(self._event_types_by_name.values())

    @event_types.setter
    def event_types(self, value: list[EventType]) -> None:
        self._event_types_by_name = {et.name: et for et in value}

    def event_types_for_scope(self, scope: str) -> list[EventType]:
        if scope == "workspace":
            return list(self._event_types_by_name.values())

        if scope != "recording":
            raise ValueError(f"Unsupported scope: {scope}")

        return [et for et in self._event_types_by_name.values() if et.source == "recording"]

    def add_events(self, events: dict[str, list[int]]) -> None:
        """
        Add events from the provided dictionary to the existing ones. The expected
        format of the dictionary is {event name: list of timestamps to add}.
        """
        event_types_to_add = []
        event_types_to_update = []
        for event_name, timestamps in events.items():
            event_type = self._event_types_by_name.get(event_name, None)
            if event_type is None:
                event_type = EventType.from_name(event_name)
                event_type.source = "recording"
                self._event_types_by_name[event_name] = event_type
                event_types_to_add.append(event_type)
            else:
                if event_type.name in IMMUTABLE_EVENTS:
                    raise ValueError(
                        f"Event type {event_type.name} is immutable, new instances "
                        f"of this event cannot be added."
                    )
                event_types_to_update.append(event_type)

            if event_type.name not in self.events:
                self.events[event_type.name] = []
            self.events[event_type.name].extend(timestamps)

        self.save_cached_json("events.json", self.events)
        self._update_gui_for_event_types(event_types_to_add=event_types_to_add)
        if event_types_to_add:
            self.changed.emit()
        for event_type in event_types_to_update:
            self._update_timeline_data(event_type)

    def delete_events(
        self, events: dict[str, list[int]], remove_empty_types: bool = False
    ) -> None:
        """
        Delete events specified in the provided dictionary from the existing ones.
        The expected format of the dictionary is {event name: list of timestamps to remove}.
        Event types, for which all events are removed, are not deleted by default,
        but can optionally be removed if `remove_empty_types` is set to True.
        """
        event_types_to_remove = []
        event_types_to_update = []
        for event_name, timestamps in events.items():
            if event_name not in self._event_types_by_name:
                logging.warning(f"Skipping unknown event '{event_name}' from deletion")
                continue

            if event_name in IMMUTABLE_EVENTS:
                raise ValueError(
                    f"Event type {event_name} is immutable, instances "
                    f"of this event cannot be deleted."
                )

            event_type = self._event_types_by_name[event_name]
            existing_timestamps = set(self.events[event_type.name])
            timestamps_to_remove = set(timestamps)
            remaining_timestamps = existing_timestamps - timestamps_to_remove
            if remaining_timestamps:
                self.events[event_type.name] = list(remaining_timestamps)
                event_types_to_update.append(event_type)
                continue

            del self.events[event_type.name]
            if remove_empty_types:
                del self._event_types_by_name[event_name]
                event_types_to_remove.append(event_type)
            else:
                event_types_to_update.append(event_type)

        self.save_cached_json("events.json", self.events)
        if event_types_to_remove:
            self._update_gui_for_event_types(event_types_to_remove=event_types_to_remove)
            self.changed.emit()
        for event_type in event_types_to_update:
            self._update_timeline_data(event_type)

    def _count_events_across_workspace(self, event_type: EventType) -> tuple[int, list[str]]:
        if not self.app.batch_mode_enabled:
            events = self.events.get(event_type.name, [])
            return len(events), [self.recording._rec_dir.name] if events else []

        event_occurrences = self._workspace_index.events.get(event_type.name, {})
        return sum(event_occurrences.values()), list(event_occurrences.keys())

    def _on_event_name_changed(
        self, old_name: str, new_name: str, event_type: EventType
    ) -> None:
        self._event_types_by_name[new_name] = event_type
        del self._event_types_by_name[old_name]

        if self.headless:
            return

        timeline = self.get_timeline()
        plot_sorting_was_enabled = timeline.disable_plot_sorting()

        self._remove_gui_for_event_name(old_name)
        self._setup_gui_for_event_type(event_type)
        self._update_timeline_data(event_type)

        if plot_sorting_was_enabled:
            timeline.enable_plot_sorting()

    @action
    @action_params(
        compact=True,
        icon=QIcon.fromTheme("window-new"),
    )
    def import_csv(self, source: FilePath) -> None:
        events_df = pd.read_csv(source)
        events = _load_events_from_dataframe(events_df)
        self.add_events(events)

    @action
    @action_params(compact=True, icon=QIcon(str(neon_player.asset_path("export.svg"))))
    def export(self, destination: Path = Path()) -> None:
        export_window = self.app.get_export_window()
        events_df = self._prepare_events_export(self.recording, export_window)

        destination_file = destination / "events.csv"
        events_df.to_csv(destination_file, index=False)

        logging.info(f"Exported events to {destination_file}")

    def _prepare_events_export(
        self, recording: NeonRecording, export_window: tuple[int, int]
    ) -> pd.DataFrame:
        start_time, stop_time = export_window
        events = self.events
        events_df = pd.DataFrame({
            "recording id": recording.info["recording_id"],
            "timestamp [ns]": list(events.values()),
            "name": list(events.keys()),
        })

        events_df = events_df.explode("timestamp [ns]").reset_index(drop=True).dropna()
        events_df["timestamp [ns]"] = events_df["timestamp [ns]"].astype(
            recording.events.time.dtype
        )
        events_df = events_df.sort_values("timestamp [ns]")
        start_mask = events_df["timestamp [ns]"] >= start_time
        stop_mask = events_df["timestamp [ns]"] <= stop_time
        events_df = events_df[start_mask & stop_mask]

        events_df["type"] = "player"
        for index, row in events_df.iterrows():
            mask = recording.events.time == row["timestamp [ns]"]
            matching = recording.events[mask]
            if any(row["name"] == matching.event):
                events_df.loc[index, "type"] = "recording"

        return events_df

    @action
    @action_params(compact=True, icon=QIcon(str(neon_player.asset_path("export.svg"))))
    def export_all_recordings(self, destination: Path = Path(".")) -> None:
        run_export_across_recordings(self, destination)
