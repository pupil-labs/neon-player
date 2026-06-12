import logging
import numpy as np
import pandas as pd
import typing as T
import uuid

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

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if self._name == value:
            return

        plugin = EventsPlugin.instance()
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
    @property_params(widget=None)
    def uid(self) -> str:
        return self._uid

    @uid.setter
    def uid(self, value: str):
        self._uid = value

    @staticmethod
    def from_name(name: str) -> "EventType":
        et = EventType()
        et._name = name
        et._uid = name if name in IMMUTABLE_EVENTS else str(uuid.uuid4())
        return et


class EventTypeListWidget(ValueListWidget):
    @staticmethod
    def from_property_impl(prop: property) -> "EventTypeListWidget":
        source_params = prop.fget.parameters if hasattr(prop.fget, "parameters") else {}

        return EventTypeListWidget(EventType, source_params)

    @staticmethod
    def from_type(cls: type) -> "EventTypeListWidget":
        return EventTypeListWidget(cls)

    def on_add_button_clicked(self):
        plugin = EventsPlugin.instance()
        if plugin is None:
            return

        new_event_type = plugin.create_event_type()
        plugin.add_event_type(new_event_type)
        self.add_item(new_event_type)

    def remove_item(self, item_widget: ValueListItemWidget):
        plugin = EventsPlugin.instance()
        if plugin is None:
            return

        event_type = item_widget.item_widget.value
        events = plugin._events.get(event_type.uid, [])
        if events:
            suffix = "" if len(events) == 1 else "s"
            reply = QMessageBox.question(
                None,
                "Confirm event type deletion",
                f"Deleting event type '{event_type.name}' will also delete its {len(events)}"
                f" instance{suffix}. Do you want to proceed?",
                buttons=QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                defaultButton=QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        plugin.delete_event_type(event_type)
        super().remove_item(item_widget)


def _load_events_from_recording(
    recording: NeonRecording, global_event_types: list[str] = []
) -> tuple[list[EventType], dict]:
    """
    Loads events from the recording and additionally creates event types that
    are defined in global settings.

    Returns all created event types and the events as {uid: list of timestamps}.
    """
    event_type_cache = {}
    events = {}

    for event in recording.events:
        event_name = str(event.event)

        # Look up or create the event type
        et = event_type_cache.get(event_name, None)
        if et is None:
            et = EventType.from_name(event_name)
            event_type_cache[event_name] = et

        # Add event to the dictionary
        if et.uid not in events:
            events[et.uid] = []
        events[et.uid].append(event.time)

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
    for uid in cached_events:
        if uid in event_type_cache:
            continue

        et = known_event_types_by_uid.get(uid, None)
        if et is not None:
            event_type_cache[uid] = et
            continue

        raise ValueError(f"Event type with uid {uid} not found")

    return list(event_type_cache.values()), cached_events


def _load_events_from_dataframe(events_df: pd.DataFrame) -> dict[str, list[int]]:
    events = {}
    for name, group in events_df.groupby("name"):
        if name in IMMUTABLE_EVENTS:
            logging.warning(f"Skipping immutable event '{name}' from imported CSV")
            continue

        events[name] = group["timestamp [ns]"].tolist()

    return events


def _filter_event_types(event_types: list[EventType], mutable: bool) -> list[EventType]:
    if mutable:
        return [et for et in event_types if et.name not in IMMUTABLE_EVENTS]
    else:
        return [et for et in event_types if et.name in IMMUTABLE_EVENTS]


class EventsPlugin(neon_player.Plugin):
    label = "Events"
    global_properties = EventsPluginGlobalProps()

    def __init__(self) -> None:
        super().__init__()
        self._event_types_by_name: dict[str, EventType] = {}
        self._events: dict[str, list[int]] = {}

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

        return Plugin.get_instance_by_name("EventsPlugin")

    def _on_key_pressed(self, event: QKeyEvent) -> None:
        key_text = event.text().lower()
        if key_text == "":
            return

        for event_type in self._event_types_by_name.values():
            if event_type.shortcut.lower() == key_text:
                self._add_event(event_type)

    def _load_events(self, recording: NeonRecording) -> tuple[list[EventType], dict, str]:
        events = {}
        event_types = []

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

    def on_recording_loaded(self, recording: NeonRecording) -> None:  # noqa: C901
        event_types, events, source = self._load_events(recording)
        self._events = events

        if source == "recording":
            # When loading from recording, only mutable event types need to be saved,
            # but the UI needs to be set up for all event types
            mutable_event_types = _filter_event_types(event_types, mutable=True)
            event_types_to_setup_ui_for = event_types
            self.event_types = mutable_event_types
            self.save_cached_json("events.json", events)
        elif source == "cache":
            # When loading from cache, all event types are expected to have been loaded
            # from plugin settings, so self.event_types is already correct, but the UI
            # still needs to be set up for stored and immutable event types
            immutable_event_types = _filter_event_types(event_types, mutable=False)
            event_types_to_setup_ui_for = self.event_types + immutable_event_types

        logging.info(f"Loaded {sum(len(v) for v in self._events.values())} events")

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
            f"Events - {event_type.name}", create_if_not_exists=False
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
        new_event_type = EventType()
        new_event_type.uid = str(uuid.uuid4())

        event_type_counter = 1
        while new_event_type.name == "":
            candidate_name = f"event-{event_type_counter}"
            if candidate_name not in self._event_types_by_name:
                new_event_type.name = candidate_name
            event_type_counter += 1

        return new_event_type

    def add_event_type(self, event_type: EventType):
        self._event_types_by_name[event_type.name] = event_type
        self._update_gui_for_event_types(event_types_to_add=[event_type])
        self.changed.emit()

    def delete_event_type(self, event_type: EventType) -> None:
        if event_type.uid in self._events:
            del self._events[event_type.uid]
            self.save_cached_json("events.json", self._events)

        del self._event_types_by_name[event_type.name]
        self._update_gui_for_event_types(event_types_to_remove=[event_type])
        self.changed.emit()

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
        closest_event = self._find_closest_event(event_type, data_point[0])
        if closest_event is None:
            return

        self._events[event_type.uid].remove(closest_event)
        self.save_cached_json("events.json", self._events)
        self._update_timeline_data(event_type)

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

        x = np.array(self.events.get(event_type.uid, []))
        y = np.zeros_like(x)
        plot_item.items[0].setData(x, y)

    @property
    @property_params(
        widget=EventTypeListWidget,
        item_params={"label_field": "name"},
        prevent_add=True,
        primary=True,
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
                event_types_to_update.append(event_type)

            if event_type.uid not in self._events:
                self._events[event_type.uid] = []
            self._events[event_type.uid].extend(timestamps)

        self.save_cached_json("events.json", self._events)
        self._update_gui_for_event_types(event_types_to_add=event_types_to_add)
        for event_type in event_types_to_update:
            self._update_timeline_data(event_type)

    def delete_events(self, events: dict[str, list[int]]) -> None:
        """
        Delete events specified in the provided dictionary from the existing ones.
        The expected format of the dictionary is {event name: list of timestamps to remove}.
        """
        event_types_to_remove = []
        event_types_to_update = []
        for event_name, timestamps in events.items():
            if event_name not in self._event_types_by_name:
                logging.warning(f"Skipping unknown event '{event_name}' from deletion")
                continue

            event_type = self._event_types_by_name[event_name]
            existing_timestamps = set(self._events[event_type.uid])
            timestamps_to_remove = set(timestamps)
            remaining_timestamps = existing_timestamps - timestamps_to_remove
            if not remaining_timestamps:
                del self._events[event_type.uid]
                event_types_to_remove.append(event_type)
            else:
                self._events[event_type.uid] = list(remaining_timestamps)
                event_types_to_update.append(event_type)

        self.save_cached_json("events.json", self._events)
        self._update_gui_for_event_types(event_types_to_remove=event_types_to_remove)
        for event_type in event_types_to_update:
            self._update_timeline_data(event_type)

    def _on_event_name_changed(self, old_name, new_name, event_type) -> None:
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
    def import_csv(self, source: FilePath):
        events_df = pd.read_csv(source)
        events = _load_events_from_dataframe(events_df)
        self.add_events(events)

    @action
    @action_params(compact=True, icon=QIcon(str(neon_player.asset_path("export.svg"))))
    def export(self, destination: Path = Path()):
        export_window = self.app.get_export_window()
        events_df = self._prepare_events_export(self.recording, export_window)

        destination_file = destination / "events.csv"
        events_df.to_csv(destination_file, index=False)

        logging.info(f"Exported events to {destination_file}")

    def _prepare_events_export(self, recording: NeonRecording, export_window: tuple[int, int]) -> pd.DataFrame:
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
