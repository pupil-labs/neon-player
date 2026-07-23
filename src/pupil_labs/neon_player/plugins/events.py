import logging
import numpy as np
import pandas as pd
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
from pupil_labs.neon_player.job_manager import BatchBackgroundJob
from pupil_labs.neon_player.plugins.shared import run_export_across_recordings
from pupil_labs.neon_player.plugins import Plugin
from pupil_labs.neon_player.ui import ListPropertyAppenderAction
from pupil_labs.neon_player.utilities import SignalDebouncer

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

    def __repr__(self) -> str:
        shortcut_str = f", shortcut={self._shortcut}" if self._shortcut else ""
        return f"EventType({self._name}{shortcut_str})"

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if self._name == value:
            return

        plugin = EventsPlugin.instance()
        if plugin is not None and not plugin.validate_event_name(value):
            return

        old_name = self._name
        self._name = value
        self._uid = value

        # NOTE: if multiple name changes are made in quick succession, we need to,
        # on the one hand, debounce the signal to avoid launching multiple bg jobs,
        # but on the other hand, pass original name before the first change as
        # `old_name` to ensure correct modification, so we re-use the value from
        # the pending debouncer if it exists
        if plugin is not None and plugin.batch_mode_enabled:
            debouncer = SignalDebouncer.get_pending_debouncer(self.name_changed)
            original_name = debouncer.args[0] if debouncer is not None else old_name
            SignalDebouncer.debounce(self.name_changed, 1.0, original_name, value)
        else:
            self.name_changed.emit(old_name, value)

    @property
    @property_params(max_length=1)
    def shortcut(self) -> str:
        return self._shortcut

    @shortcut.setter
    def shortcut(self, value: str) -> None:
        self._shortcut = value

    @property
    @property_params(widget=None)
    def uid(self) -> str:
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

        new_event_type = plugin.create_event_type()
        plugin.add_event_type(new_event_type)
        self.add_item(new_event_type)

    def remove_item(self, item_widget: ValueListItemWidget) -> None:
        plugin = EventsPlugin.instance()
        if plugin is None:
            return

        if item_widget.item_widget is None:
            return

        event_type = item_widget.item_widget.value
        events = plugin._events.get(event_type.uid, [])
        if events:
            suffix = "" if len(events) == 1 else "s"
            confirmed = plugin.user_confirm(
                "Confirm event type deletion",
                f"Deleting event type '{event_type.name}' will also delete its {len(events)}"
                f" instance{suffix}. Do you want to proceed?",
            )
            if not confirmed:
                return

        plugin.delete_event_type(event_type)
        self.container_layout.removeWidget(item_widget)
        item_widget.deleteLater()


class WorkspaceEventIndex():
    """
    Maintains an index of all events and the number of their occurrences
    across all recordings in the workspace to support renaming and deletion.
    """
    def __init__(self) -> None:
        # {event_name: {recording_name: event_count}}
        self.events: dict[str, dict[str, int]] = {}

        # IDs of recording that the index is based on
        self.recording_ids: set[str] = set()

    def load(self, data: dict[str, T.Any]) -> None:
        if data is None:
            return

        self.events = data.get("events", {})
        self.recording_ids = set(data.get("recording_ids", []))

    def to_dict(self) -> dict[str, T.Any]:
        data = {
            "events": self.events,
            "recording_ids": list(self.recording_ids)
        }
        return data

    def _cleanup_events(self) -> None:
        """
        Clean up any events that no longer exist in any recordings.
        """
        self.events = {k: v for k, v in self.events.items() if v}

    def update(self, recording_id: str, recording_events: dict[str, list[int]]) -> None:
        """
        Batch index update that goes through all events in the index and all events
        that belong to a particular recording.
        """
        # First, check if existing index entries for this recording are up-to-date
        for event_name, recording_counts in self.events.items():
            event_in_recording = event_name in recording_events
            recording_in_index = recording_id in recording_counts

            if recording_in_index and not event_in_recording:
                del recording_counts[recording_id]

            if event_in_recording:
                timestamps = recording_events[event_name]
                self.events[event_name][recording_id] = len(timestamps)

        self._cleanup_events()

        # Process other events from the recording that are not yet in the index
        for event_name, timestamps in recording_events.items():
            if event_name in IMMUTABLE_EVENTS:
                continue

            if event_name in self.events:
                continue

            self.events[event_name] = {}
            self.events[event_name][recording_id] = len(timestamps)

        # Mark recording as processed in the index
        if recording_id not in self.recording_ids:
            self.recording_ids.add(recording_id)

    def drop(self, recording_id: str) -> None:
        """
        Remove all events from the index that belong to a particular recording.
        """
        if recording_id not in self.recording_ids:
            return

        for recording_counts in self.events.values():
            if recording_id in recording_counts:
                del recording_counts[recording_id]

        self._cleanup_events()
        self.recording_ids.remove(recording_id)


def _load_events_from_recording(recording: NeonRecording):
    events: dict[str, list[int]] = {}

    logging.debug("Loading events from Neon recording")
    for event_name in np.unique(recording.events.event):
        timestamps = recording.events[recording.events.event == event_name].time
        events[str(event_name)] = [int(t) for t in timestamps]

    return events


def _load_events_from_dataframe(events_df: pd.DataFrame) -> dict[str, list[int]]:
    events = {}
    for name, group in events_df.groupby("name"):
        name = str(name)
        if name in IMMUTABLE_EVENTS:
            logging.warning(f"Skipping immutable event '{name}' from imported CSV")
            continue

        events[name] = group["timestamp [ns]"].astype(int).tolist()

    return events


class EventsPlugin(neon_player.Plugin):
    label = "Events"
    global_properties = EventsPluginGlobalProps()

    def __init__(self) -> None:
        super().__init__()
        self._event_types_by_name: dict[str, EventType] = {}
        self._immutable_event_types: list[EventType] = []
        self._events: dict[str, list[int]] = {}

        self._consider_workspace = False
        self._batch_rename_job: dict[str, BatchBackgroundJob] = {}
        self._batch_delete_job: dict[str, BatchBackgroundJob] = {}
        self._batch_update_job: BatchBackgroundJob | None = None
        self._workspace_index: WorkspaceEventIndex = WorkspaceEventIndex()
        self._index_file = "workspace-events.json"

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

    def _on_key_pressed(self, event: QKeyEvent) -> None:
        key_text = event.text().lower()
        if key_text == "":
            return

        for event_type in self._event_types_by_name.values():
            if event_type.shortcut.lower() == key_text:
                self._add_event(event_type)

    def _get_uid_name_mapping(self) -> dict[str, str]:
        recording_settings = self.app.session_settings.recording_settings
        plugin_state = recording_settings.plugin_states.get(self.__class__.__name__, {})
        event_types_state = plugin_state.get("event_types", [])

        uid_name_mapping = {}
        for et in event_types_state:
            uid_name_mapping[et["uid"]] = et["name"]

        return uid_name_mapping

    def _load_events_from_cache(self) -> dict[str, list[int]] | None:
        """
        Load events from the cached JSON file. Ensure that the events dictionary has
        event names as keys, not unique IDs. If IDs are used as keys, the corresponding
        event names are derived from the recording settings.
        """
        try:
            events = self.load_cached_json("events.json")
        except Exception:
            logging.exception("Failed to load events.json")
            return None

        if events is None:
            return None

        uid_name_mapping = self._get_uid_name_mapping()
        corrected_events = {}
        events_changed = False
        for key, value in events.items():
            if key in IMMUTABLE_EVENTS:
                corrected_events[key] = value
                continue

            # New format: keys are event names, keep them as is
            if key in self._event_types_by_name:
                corrected_events[key] = value
                continue

            # Old format: keys are event type UIDs, replace with event names using
            # the mapping based on recording settings
            if key in uid_name_mapping:
                event_name = uid_name_mapping[key]
                corrected_events[event_name] = value
                events_changed = True
                continue

            # Otherwise, keep the key as is, a corresponding event type will be created
            corrected_events[key] = value

        if events_changed:
            logging.debug("Corrected event keys from UIDs to names in events.json")
            self.save_cached_json("events.json", corrected_events)

        return corrected_events

    def _load_workspace_index(self) -> None:
        data = self.load_cached_json(self._index_file, workspace=True)
        self._workspace_index.load(data)

    def _save_workspace_index(self) -> None:
        data = self._workspace_index.to_dict()
        self.save_cached_json(self._index_file, data, workspace=True)

    def _update_workspace_index(self, load: bool = True, save: bool = True) -> None:
        if load:
            self._load_workspace_index()
        self._workspace_index.update(self.recording.id, self.events)
        if save:
            self._save_workspace_index()

    def _scan_events_in_workspace(self):
        # Skip scanning if it is already in progress, connected slot should update the
        # event types in the UI once done
        if self._batch_update_job is not None:
            return

        # Update the index for the current recording to account for any changes
        # outside of the workspace mode
        self._update_workspace_index()

        if self.headless:
            self._on_workspace_event_scan_finished()
            return

        workspace_recording_ids = {rec.id for rec in self.workspace.recordings}
        scanned_recording_ids = self._workspace_index.recording_ids

        # Remove any recordings from the index that no longer exist in the workspace
        outdated_recording_ids = scanned_recording_ids - workspace_recording_ids
        for recording_id in outdated_recording_ids:
            self._workspace_index.drop(recording_id)

        # If all recordings have already been scanned, update the UI immediately,
        # otherwise launch a batch background job
        missing_recording_ids = workspace_recording_ids - scanned_recording_ids
        if not missing_recording_ids:
            self._on_workspace_event_scan_finished()
            return

        recordings_to_scan = self.workspace.get_recordings_by_id(missing_recording_ids)
        logging.info(f"Scanning events in {len(recordings_to_scan)} recordings of the workspace")
        batch_job = self.job_manager.run_background_batch_action(
            "Scan events in workspace",
            "EventsPlugin._update_workspace_index",
            recordings=recordings_to_scan
        )
        batch_job.finished.connect(self._on_workspace_event_scan_finished)
        self._batch_update_job = batch_job

    def _on_workspace_event_scan_finished(self):
        self._batch_update_job = None
        self._load_workspace_index()
        logging.info("Finished scanning events in workspace recordings")

        types_to_add = []
        for event_name in self._workspace_index.events:
            if event_name in self._event_types_by_name:
                continue

            # Hide event types that are being deleted across the workspace
            if event_name in self._batch_delete_job:
                continue

            types_to_add.append(EventType.from_name(event_name))

        if types_to_add:
            self.event_types = self.event_types + types_to_add
            self._update_gui_for_event_types(event_types_to_add=types_to_add)

    def on_recording_loaded(self, recording: NeonRecording) -> None:
        events = self._load_events_from_cache()
        if events is None:
            events = _load_events_from_recording(recording)
        self._events = events
        logging.info(f"Loaded {sum(len(v) for v in self._events.values())} events")

        # NOTE: event types are loaded from plugin settings before this method is called,
        # so existing event types need to be preserved while adding missing ones.
        for event_name in events:
            if event_name in IMMUTABLE_EVENTS:
                continue

            if event_name in self._event_types_by_name:
                continue

            new_event_type = EventType.from_name(event_name)
            self._event_types_by_name[event_name] = new_event_type

        # If event type ID is different from the event name, make them match to
        # align event types across recordings
        event_types_changed = False
        for event_type in self._event_types_by_name.values():
            if event_type.uid != event_type.name:
                event_type.uid = event_type.name
                event_types_changed = True
        if event_types_changed:
            self.changed.emit()

        # Immutable event types are not stored in plugin settings but require UI
        self._immutable_event_types = [
            EventType.from_name(event_name) for event_name in IMMUTABLE_EVENTS
        ]
        event_types_to_setup_ui_for = self._immutable_event_types + self.event_types

        self._consider_workspace = self.batch_mode_enabled and self.workspace.size > 1
        if self._consider_workspace:
            self._scan_events_in_workspace()

        self._update_gui_for_event_types(event_types_to_add=event_types_to_setup_ui_for)

    def on_disabled(self) -> None:
        if self.headless or self.recording is None:
            return

        event_types = list(self._event_types_by_name.values()) + self._immutable_event_types
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
            et.name_changed.connect(self._on_event_name_changed)
            self._setup_gui_for_event_type(et)
            self._update_timeline_data(et)

        for et in event_types_to_remove:
            et.name_changed.disconnect()
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

    def create_event_type(self) -> EventType:
        event_type_counter = 1
        candidate_name = f"event-{event_type_counter}"
        while candidate_name in self._event_types_by_name:
            event_type_counter += 1
            candidate_name = f"event-{event_type_counter}"
        return EventType.from_name(candidate_name)

    def validate_event_name(self, name: str) -> bool:
        if name in IMMUTABLE_EVENTS:
            QMessageBox.warning(
                None,
                "Invalid event name",
                f"Event cannot be renamed to '{name}' as this name is reserved. "
                f"Please choose a different name.",
            )
            return False

        if name in self._event_types_by_name:
            QMessageBox.warning(
                None,
                "Duplicate event name",
                f"Event '{name}' already exists. Please choose a different name.",
            )
            return False

        if self._batch_rename_job is not None:
            QMessageBox.warning(
                None,
                "Event rename in progress",
                "Please wait for the current event rename operation to complete "
                "before renaming the event again.",
            )
            return False

        return True

    def add_event_type(self, event_type: EventType) -> None:
        if event_type.name in self._event_types_by_name:
            raise ValueError(
                f"Event type {event_type.name} already exists."
            )

        self._event_types_by_name[event_type.name] = event_type
        self._update_gui_for_event_types(event_types_to_add=[event_type])
        self.changed.emit()

    def delete_event_type(self, event_type: EventType) -> None:
        if event_type.name in IMMUTABLE_EVENTS:
            raise ValueError(
                f"Event type {event_type.name} cannot be deleted."
            )

        if event_type.name not in self._event_types_by_name:
            return

        del self._event_types_by_name[event_type.name]
        self._delete_events_by_name(event_type.name)
        self.changed.emit()
        if not self.headless:
            self._update_gui_for_event_types(event_types_to_remove=[event_type])

        if not self._consider_workspace:
            return

        confirm_cancel = (
            f"Cancelling this operation will leave instances of event '{event_type.name}' "
            f"in an inconsistent state across recordings. Are you sure you want to cancel?"
        )
        batch_job = self.job_manager.run_background_batch_action(
            f"Delete events [{event_type.name}]",
            "EventsPlugin._delete_events_by_name",
            args_generator=lambda _: [event_type.name],
            confirm_cancel=confirm_cancel
        )
        batch_job.finished.connect(lambda: self._on_batch_delete_finished(event_type))
        batch_job.canceled.connect(lambda: self._on_batch_delete_finished(event_type))
        self._batch_delete_job[event_type.name] = batch_job

    def _on_batch_delete_finished(self, event_type: EventType) -> None:
        del self._batch_delete_job[event_type.name]

    def _delete_events_by_name(self, event_name: str) -> None:
        """
        Background job to delete an event type in sibling recordings of the workspace.
        The event type will be deleted from the current recording in the main process, so
        the background job only needs to delete events of this type if they exist.
        """
        if event_name in IMMUTABLE_EVENTS:
            return

        if event_name not in self.events:
            return

        logging.debug(f"Deleting all instances of event '{event_name}'")
        del self._events[event_name]
        self._update_workspace_index()
        self.save_cached_json("events.json", self._events)

    def _add_event(self, event_type: EventType, ts: int | None = None) -> None:
        if self.recording is None:
            return

        if ts is None:
            ts = self.app.current_ts

        self.add_events({event_type.name: [ts]})

    def _find_closest_event(
        self, event_type: EventType, target_ts: int, tolerance_ns: int = 5
    ) -> int | None:
        if event_type.uid not in self._events:
            return None

        events_list = self._events[event_type.uid]
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

        x = np.array(self._events.get(event_type.uid, []))
        y = np.zeros_like(x)
        plot_item.items[0].setData(x, y)

    @property
    @property_params(
        widget=EventTypeListWidget,
        item_params={"label_field": "name"},
        prevent_add=True,
        primary=True,
        scope=["workspace"],
    )
    def event_types(self) -> list[EventType]:
        return list(self._event_types_by_name.values())

    @event_types.setter
    def event_types(self, value: list[EventType]) -> None:
        self._event_types_by_name = {et.name: et for et in value}

    @property
    @property_params(widget=None, dont_encode=True)
    def events(self) -> dict[str, list[int]]:
        """
        Read-only property that returns the events as a dictionary with event names
        as keys and lists of all timestamps for each event as values. For modifying
        events from other plugins, use add_events() and delete_events() methods.
        """
        event_id_name_mapping = {et.uid: et.name for et in self.event_types}
        events_by_name = {}
        for event_id, timestamps in self._events.items():
            # Use IDs as fallback for immutable events that are not included in event_types
            event_name = event_id_name_mapping.get(event_id, event_id)
            events_by_name[event_name] = timestamps
        return events_by_name

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
                self._event_types_by_name[event_name] = event_type
                event_types_to_add.append(event_type)
            else:
                if event_type.name in IMMUTABLE_EVENTS:
                    raise ValueError(
                        f"Event type {event_type.name} is immutable, new instances "
                        f"of this event cannot be added."
                    )
                event_types_to_update.append(event_type)

            if event_type.uid not in self._events:
                self._events[event_type.uid] = []
            self._events[event_type.uid].extend(timestamps)

        self.save_cached_json("events.json", self._events)
        self._update_workspace_index()

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
            existing_timestamps = set(self._events[event_type.uid])
            timestamps_to_remove = set(timestamps)
            remaining_timestamps = existing_timestamps - timestamps_to_remove
            if remaining_timestamps:
                self._events[event_type.uid] = list(remaining_timestamps)
                event_types_to_update.append(event_type)
                continue

            del self._events[event_type.uid]
            if remove_empty_types:
                del self._event_types_by_name[event_name]
                event_types_to_remove.append(event_type)
            else:
                event_types_to_update.append(event_type)

        self.save_cached_json("events.json", self._events)
        self._update_workspace_index()

        if event_types_to_remove:
            self._update_gui_for_event_types(event_types_to_remove=event_types_to_remove)
            self.changed.emit()
        for event_type in event_types_to_update:
            self._update_timeline_data(event_type)

    def _on_event_name_changed(self, old_name: str, new_name: str) -> None:
        logging.info(f"Renaming event '{old_name}' to '{new_name}'")
        event_type = self._event_types_by_name.pop(old_name, None)
        if event_type is None:
            logging.warning(f"Event type '{old_name}' not found for renaming")
            return

        if self._batch_rename_job is not None:
            return

        # NOTE: While renaming the events, we need both the old and new names to be
        # present for events to load correctly. Once the renaming is complete, the
        # old name can be removed from the event types.
        self._event_types_by_name[new_name] = event_type
        self._event_types_by_name[old_name] = EventType.from_name(old_name)
        self._rename_all_events_by_name(old_name, new_name)
        if not self.headless:
            timeline = self.get_timeline()
            plot_sorting_was_enabled = timeline.disable_plot_sorting()

            self._remove_gui_for_event_name(old_name)
            self._setup_gui_for_event_type(event_type)
            self._update_timeline_data(event_type)

            if plot_sorting_was_enabled:
                timeline.enable_plot_sorting()

        if not self._consider_workspace:
            self._rename_all_events_by_name(old_name, new_name)
            self._finalize_event_rename(old_name)
            return

        self._batch_rename_job = self.job_manager.run_background_batch_action(
            f"Rename events [{old_name} -> {new_name}]",
            "EventsPlugin._rename_all_events_by_name",
            lambda _: [old_name, new_name]
        )
        self._batch_rename_job.finished.connect(lambda: self._finalize_event_rename(old_name))

    def _rename_all_events_by_name(self, old_name: str, new_name: str) -> None:
        if old_name not in self._events:
            return

        self._events[new_name] = self._events.pop(old_name)
        self.save_cached_json("events.json", self._events)
        self._update_workspace_index()

    def _finalize_event_rename(self, old_name: str) -> None:
        if old_name in self._event_types_by_name:
            del self._event_types_by_name[old_name]
        self.changed.emit()
        self._batch_rename_job = None

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
