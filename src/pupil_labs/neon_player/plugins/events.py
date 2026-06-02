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

from pupil_labs import neon_player
from pupil_labs.neon_player import GlobalPluginProperties, action
from pupil_labs.neon_player.ui import HeaderAction

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
        self._plugin = None

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if self._name == value:
            return

        if self._plugin is not None and value in self._plugin._event_types_by_name:
            QMessageBox.warning(
                None,
                "Duplicate event type",
                f"Event type {value} already exists. Please choose a different name.",
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
        et.name = name
        et.uid = str(uuid.uuid4())
        return et


def _load_events_from_recording(
    recording: NeonRecording, global_event_types: list[str] = []
) -> tuple[list[EventType], dict]:
    event_type_cache = {}
    events = {}

    for event in recording.events:
        event_name = str(event.event)

        et = event_type_cache.get(event_name, None)
        if et is None:
            et = EventType.from_name(event_name)
            if event_name in IMMUTABLE_EVENTS:
                et.uid = event_name
            event_type_cache[event_name] = et

        # Add to memory
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
        et.uid = event_name
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


class EventsPlugin(neon_player.Plugin):
    label = "Events"
    global_properties = EventsPluginGlobalProps()

    def __init__(self) -> None:
        super().__init__()
        self._event_types_by_name: dict[str, EventType] = {}
        self._event_type_counter = 1
        self.events: dict[str, list[int]] = {}

        if self.headless:
            return

        self.get_timeline().key_pressed.connect(self._on_key_pressed)
        self.header_action = HeaderAction(
            self.add_event_type, "+ Add event type"
        )

    def _on_key_pressed(self, event: QKeyEvent) -> None:
        key_text = event.text().lower()
        if key_text == "":
            return

        for event_type in self._event_types_by_name.values():
            if event_type.shortcut.lower() == key_text:
                self.add_event(event_type)

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
        self.events = events

        # NOTE: The event types need to be updated only when loading from the
        # recording. When loading from cache, the event types are loaded
        # beforehand from plugin settings.
        if source == "recording":
            self.event_types = event_types
            self.save_cached_json("events.json", events)

        logging.info(f"Loaded {sum(len(v) for v in self.events.values())} events")

        if self.headless:
            return

        # Create timeline plots for each event type that is present among the events
        # Disable plot sorting to improve performance for recording with many events
        timeline = self.get_timeline()
        plot_sorting_was_enabled = timeline.disable_plot_sorting()
        for et in event_types:
            self._attach_event_type(et)
            self._setup_gui_for_event_type(et)
            self._update_timeline_data(et)
        if plot_sorting_was_enabled:
            timeline.enable_plot_sorting()

    def on_disabled(self) -> None:
        if self.recording is None or self.headless:
            return

        timeline = self.get_timeline()
        plot_sorting_was_enabled = timeline.disable_plot_sorting()
        timeline.remove_timeline_plot("Events")
        for event_name in self._event_types_by_name:
            self._remove_gui_for_event_name(event_name)
        if plot_sorting_was_enabled:
            timeline.enable_plot_sorting()

    def _attach_event_type(self, event_type: EventType) -> None:
        # Provide a reference to the plugin for checking whether the new name
        # of the event type is already taken
        event_type._plugin = self

        # Connect the required signals
        event_type.changed.connect(self.changed.emit)
        event_type.name_changed.connect(
            lambda old, new, et=event_type: self._on_event_name_changed(old, new, et)
        )

    def _setup_gui_for_event_type(self, event_type: EventType) -> None:
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
            action = self.register_timeline_action(
                f"Add Event/{event_type.name}", None, lambda: self.add_event(event_type)
            )
            self.app.main_window.sort_action_menu("Timeline/Add Event")
            event_type.name_changed.connect(lambda old, new: action.setText(new))

        self.register_data_actions(event_type)

    def register_data_actions(self, event_type: EventType) -> None:
        if event_type.name not in IMMUTABLE_EVENTS:
            self.register_data_point_action(
                f"Events - {event_type.name}",
                f"Delete {event_type.name} instance",
                lambda data_point, et=event_type: self.delete_event_instance(
                    f"Events - {event_type.name}", data_point, et
                ),
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

        self.register_data_point_action(
            f"Events - {event_type.name}",
            f"Seek to this {event_type.name}",
            self.seek_to_event_instance,
        )

    def _remove_gui_for_event_name(
        self,
        event_name: str,
        remove_add_action: bool = True
    ) -> None:
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
        if remove_add_action:
            self.unregister_timeline_action(f"Add Event/{event_name}")
            self.app.main_window.remove_menu_if_empty("Timeline/Add Event")

    def add_event_type(self):
        new_event_type = EventType()
        new_event_type.uid = str(uuid.uuid4())

        while new_event_type.name == "":
            candidate_name = f"event-{self._event_type_counter}"
            if candidate_name not in self._event_types_by_name:
                new_event_type.name = candidate_name
            self._event_type_counter += 1

        self._event_types_by_name[new_event_type.name] = new_event_type
        self._attach_event_type(new_event_type)
        self._setup_gui_for_event_type(new_event_type)
        self.changed.emit()

    def add_event(self, event_type: EventType, ts: int | None = None) -> None:
        if self.recording is None:
            return

        if ts is None:
            ts = self.app.current_ts

        if event_type.uid not in self.events:
            self.events[event_type.uid] = []

        self.events[event_type.uid].append(ts)
        self.save_cached_json("events.json", self.events)
        self._update_timeline_data(event_type)

    def _find_closest_event(
        self, event_type: EventType, target_ts: int, tolerance_ns: int = 5
    ) -> int | None:
        if event_type.uid not in self.events:
            return None

        events_list = self.events[event_type.uid]
        if not events_list:
            return None

        closest_event = min(events_list, key=lambda x: abs(x - target_ts))
        if abs(closest_event - target_ts) > tolerance_ns:
            return None

        return closest_event

    def delete_event_instance(
        self, data_point: tuple[int, T.SupportsFloat], event_type: EventType
    ) -> None:
        if event_type.uid not in self.events:
            return

        closest_event = self._find_closest_event(event_type, data_point[0])
        if closest_event is None:
            return

        self.events[event_type.uid].remove(closest_event)
        self.save_cached_json("events.json", self.events)
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
        add_button_text="Create new event type",
        item_params={"label_field": "name"},
        prevent_add=True,
        primary=True,
    )
    def event_types(self) -> list[EventType]:
        return list(self._event_types_by_name.values())

    @event_types.setter
    def event_types(self, value: list[EventType]) -> None:
        self._event_types_by_name = {et.name: et for et in value}

    def _on_event_name_changed(self, old_name, new_name, event_type) -> None:
        self._event_types_by_name[new_name] = event_type
        del self._event_types_by_name[old_name]

        self._remove_gui_for_event_name(old_name, remove_add_action=False)
        self.register_data_actions(event_type)
        self._update_timeline_data(event_type)

    @action
    @action_params(
        compact=True,
        icon=QIcon.fromTheme("window-new"),
    )
    def import_csv(self, source: FilePath):
        events_df = pd.read_csv(source)
        modified_types = set()

        for name, group in events_df.groupby("name"):
            if name in IMMUTABLE_EVENTS:
                continue

            event_type = self.get_event_type_by_name(name)
            if event_type is None:
                event_type = self.create_event_type(name)

            if event_type.uid not in self.events:
                self.events[event_type.uid] = []

            self.events[event_type.uid].extend(group["timestamp [ns]"].tolist())
            modified_types.add(event_type)

        self.save_cached_json("events.json", self.events)

        for et in modified_types:
            self._update_timeline_data(et)

    @action
    @action_params(compact=True, icon=QIcon(str(neon_player.asset_path("export.svg"))))
    def export(self, destination: Path = Path()):
        start_time, stop_time = self.app.get_export_window()
        event_types_by_uid = self._get_event_types_by_uid()
        event_names = []
        for uid in self.events:
            name = uid if uid in IMMUTABLE_EVENTS else event_types_by_uid[uid].name
            event_names.append(name)

        events_df = pd.DataFrame({
            "recording id": self.recording.info["recording_id"],
            "timestamp [ns]": list(self.events.values()),
            "name": event_names,
        })

        events_df = events_df.explode("timestamp [ns]").reset_index(drop=True).dropna()
        events_df["timestamp [ns]"] = events_df["timestamp [ns]"].astype(
            self.recording.events.time.dtype
        )
        events_df = events_df.sort_values("timestamp [ns]")
        start_mask = events_df["timestamp [ns]"] >= start_time
        stop_mask = events_df["timestamp [ns]"] <= stop_time
        events_df = events_df[start_mask & stop_mask]

        events_df["type"] = "player"
        for index, row in events_df.iterrows():
            matching = self.recording.events.sample([row["timestamp [ns]"]])
            if any(row["name"] == matching.event):
                events_df.loc[index, "type"] = "recording"

        destination_file = destination / "events.csv"
        events_df.to_csv(destination_file, index=False)

        logging.info(f"Exported events to {destination_file}")

    def _get_event_types_by_uid(self) -> dict[str, EventType]:
        return {et.uid: et for et in self._event_types_by_name.values()}
