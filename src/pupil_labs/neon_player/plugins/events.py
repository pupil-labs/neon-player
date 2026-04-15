import contextlib
import logging
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pupil_labs.neon_recording import NeonRecording
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QIcon, QKeyEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QToolButton,
    QWidget,
)
from qt_property_widgets.utilities import (
    FilePath,
    PersistentPropertiesMixin,
    action_params,
    property_params,
)

from pupil_labs import neon_player
from pupil_labs.neon_player import GlobalPluginProperties, action
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


class EventsPlugin(neon_player.Plugin):
    label = "Events"
    global_properties = EventsPluginGlobalProps()

    def __init__(self) -> None:
        super().__init__()
        self._event_types: list[EventType] = []
        self.events: dict[str, list[int]] = {}
        self.events_expanded = True
        self.events_search = ""
        self.get_timeline().key_pressed.connect(self._on_key_pressed)
        self.header_action = ListPropertyAppenderAction(
            "event_types", "+ Add event type"
        )

        self.register_data_point_action(
            "00_Events",
            "Seek to this event",
            self.seek_to_event_instance,
        )
        self.register_data_point_action(
            "00_Events",
            "Delete event instance",
            self.delete_event_from_aggregate,
        )

    def _on_key_pressed(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Q and hasattr(self, "search_input"):
            self.search_input.setFocus()
            self.search_input.selectAll()
            return

        key_text = event.text().lower()
        if key_text == "":
            return

        for event_type in self._event_types:
            if event_type.shortcut.lower() == key_text:
                self.add_event(event_type)

    def on_recording_loaded(self, recording: NeonRecording) -> None:  # noqa: C901
        self.events = {}

        try:
            cached_events = self.load_cached_json("events.json")
        except Exception:
            logging.exception("Failed to load events json")
            cached_events = None

        if cached_events is None:
            type_cache = {et.name: et for et in self._event_types}

            for event in self.recording.events:
                evt_name = str(event.event)

                if evt_name in type_cache:
                    et = type_cache[evt_name]
                else:
                    et = self.get_event_type_by_name(evt_name)
                    if et is None:
                        et = self.create_event_type(evt_name)

                    if evt_name in IMMUTABLE_EVENTS:
                        et.uid = evt_name

                    type_cache[evt_name] = et

                if et.uid not in self.events:
                    self.events[et.uid] = []

                if event.time not in self.events[et.uid]:
                    self.events[et.uid].append(event.time)

            recording_event_names = list(type_cache.keys())
            for event_name in self.global_properties.global_event_types:
                if event_name not in recording_event_names:
                    self.create_event_type(event_name)

            self.save_cached_json("events.json", self.events)

        else:
            self.events = cached_events
            for uid in self.events:
                if uid in IMMUTABLE_EVENTS:
                    et = self.get_event_type_by_name(uid)
                    if et is None:
                        et = self.create_event_type(uid)
                        et.uid = uid
                else:
                    et = self.get_event_type(uid)

        logging.info(f"Loaded {sum(len(v) for v in self.events.values())} events")

        timeline = self.get_timeline()
        timeline.setUpdatesEnabled(False)
        try:
            self._update_all_timeline_data()
        finally:
            timeline.setUpdatesEnabled(True)
            timeline.sort_plots()

    def get_event_type(self, uid: str) -> EventType:
        for event_type in self._event_types:
            if event_type.uid == uid:
                return event_type

        raise ValueError(f"Event type with uid {uid} not found")

    def on_disabled(self) -> None:
        if self.recording is None:
            return

        timeline = self.get_timeline()
        timeline.remove_timeline_plot("00_Events")
        timeline.remove_timeline_plot("01_Events_Search")
        for et in self._event_types:
            timeline.remove_timeline_plot(et.name)

        for plot_name in IMMUTABLE_EVENTS:
            timeline.remove_timeline_plot(plot_name)

    def _populate_add_menu(self, menu: QMenu):
        menu.clear()
        source_menu = self.app.main_window.get_menu("Timeline/Add Event")
        for action in source_menu.actions():
            menu.addAction(action)

    def _create_event_type_dialog(self):
        from PySide6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(
            None, "Create Event Type", "Enter event type name:"
        )
        if ok and name:
            self.create_event_type(name)

    def _setup_events_header(self):
        timeline = self.get_timeline()
        if timeline.get_timeline_plot("00_Events", create_if_missing=False) is not None:
            return

        header_widget = QWidget()
        layout = QHBoxLayout(header_widget)
        layout.setContentsMargins(5, 0, 0, 0)
        layout.setSpacing(4)

        self.btn_toggle = QToolButton()
        self.btn_toggle.setArrowType(
            Qt.ArrowType.DownArrow if self.events_expanded else Qt.ArrowType.RightArrow
        )
        self.btn_toggle.clicked.connect(self._toggle_events)
        self.btn_toggle.setStyleSheet(
            "QToolButton { border: none; padding: 0; color: #ccc; } QToolButton::menu-indicator { image: none; }"
        )

        lbl = QLabel("Events")
        lbl.setStyleSheet("color: #969696; font-weight: normal; font-size: 13px;")

        layout.addWidget(lbl)
        layout.addStretch()
        layout.addWidget(self.btn_toggle)

        self.btn_toggle.setPopupMode(QToolButton.ToolButtonPopupMode.DelayedPopup)
        menu = QMenu(self.btn_toggle)

        create_action = menu.addAction("Create Event Type...")
        create_action.triggered.connect(self._create_event_type_dialog)

        menu.addSeparator()

        add_menu = menu.addMenu("Add Event Instance")
        add_menu.aboutToShow.connect(lambda: self._populate_add_menu(add_menu))
        self.btn_toggle.setMenu(menu)

        plot_item = timeline.get_timeline_plot(
            "00_Events", True, legend_widget=header_widget, sort_name="00_events"
        )
        plot_item.getViewBox().allow_y_panning = False
        plot_item.setYRange(-0.5, 0.5)

        if not any(isinstance(i, pg.PlotDataItem) for i in plot_item.items):
            timeline.add_timeline_scatter("00_Events", [], is_event=True)

    def _toggle_events(self):
        self.events_expanded = not self.events_expanded
        timeline = self.get_timeline()
        timeline.setUpdatesEnabled(False)
        try:
            self._update_all_timeline_data()
        finally:
            timeline.setUpdatesEnabled(True)
            timeline.sort_plots()

    def _on_search_changed(self, text):
        self.events_search = text.lower()
        timeline = self.get_timeline()

        had_focus = self.search_input.hasFocus()
        cursor_pos = self.search_input.cursorPosition()

        timeline.setUpdatesEnabled(False)
        try:
            self._update_all_timeline_data()
        finally:
            timeline.setUpdatesEnabled(True)
            timeline.sort_plots()

        if had_focus and hasattr(self, "search_input"):
            self.search_input.setFocus()
            self.search_input.setCursorPosition(cursor_pos)

    def _update_all_timeline_data(self):
        self._setup_events_header()
        timeline = self.get_timeline()

        if hasattr(self, "btn_toggle"):
            self.btn_toggle.setArrowType(
                Qt.ArrowType.DownArrow
                if self.events_expanded
                else Qt.ArrowType.RightArrow
            )

        all_events_info = []
        for uid, times in self.events.items():
            if uid in IMMUTABLE_EVENTS:
                et_name = uid
            else:
                try:
                    et = self.get_event_type(uid)
                    et_name = et.name
                except ValueError:
                    continue

            sorted_times = sorted(times)
            total = len(sorted_times)
            for i, t in enumerate(sorted_times):
                all_events_info.append({
                    "ts": t,
                    "name": et_name,
                    "rep": f" ({i + 1}/{total})" if total > 1 else "",
                })

        all_events_info.sort(key=lambda x: x["ts"])
        all_events_data = (
            np.array([[x["ts"], 0] for x in all_events_info], dtype=np.float64)
            if all_events_info
            else np.empty((0, 2))
        )
        all_events_names = [x["name"] for x in all_events_info]
        all_events_reps = [x["rep"] for x in all_events_info]

        plot_item = timeline.get_timeline_plot("00_Events", False)
        if plot_item:
            scatter_item = next(
                (item for item in plot_item.items if isinstance(item, pg.PlotDataItem)),
                None,
            )
            if scatter_item is None:
                scatter_item = timeline.add_timeline_scatter(
                    "00_Events", all_events_data, is_event=True
                )
            else:
                scatter_item.setData(all_events_data)

            scatter_item.custom_names = all_events_names
            scatter_item.custom_reps = all_events_reps

        if not self.events_expanded:
            timeline.remove_timeline_plot("01_Events_Search")
        else:
            if not timeline.get_timeline_plot("01_Events_Search", False):
                search_widget = QWidget()
                search_layout = QHBoxLayout(search_widget)
                search_layout.setContentsMargins(5, 0, 0, 0)

                self.search_input = QLineEdit()
                self.search_input.setText(self.events_search)
                self.search_input.setPlaceholderText("Q Search")
                self.search_input.setStyleSheet(
                    "background: transparent; border: none; color: #888; font-size: 13px;"
                )
                self.search_input.textChanged.connect(self._on_search_changed)
                search_layout.addWidget(self.search_input)

                search_plot = timeline.get_timeline_plot(
                    "01_Events_Search",
                    True,
                    legend_widget=search_widget,
                    sort_name="01_events_search",
                )
                search_plot.getViewBox().allow_y_panning = False
                search_plot.hideAxis("top")
                search_plot.hideAxis("bottom")
                search_plot.hideAxis("left")

        for uid in list(self.events.keys()):
            try:
                et = self.get_event_type(uid)
            except ValueError:
                et = self.get_event_type_by_name(uid)

            if et is None:
                continue

            et_name = et.name
            matches_search = self.events_search in et_name.lower()

            if not self.events_expanded or not matches_search:
                timeline.remove_timeline_plot(et_name)
            else:
                self._setup_gui_for_event_type(et)
                self._update_timeline_data(et)

    def _setup_gui_for_event_type(self, event_type: EventType) -> None:
        timeline = self.get_timeline()

        def register_data_actions():
            self.unregister_data_point_actions(event_type.name)
            if event_type.name not in IMMUTABLE_EVENTS:
                self.register_data_point_action(
                    event_type.name,
                    f"Delete {event_type.name} instance",
                    lambda data_point, et=event_type: self.delete_event_instance(
                        event_type.name, data_point, et
                    ),
                )

            self.register_data_point_action(
                event_type.name,
                f"Seek to this {event_type.name}",
                self.seek_to_event_instance,
            )

        existing_plot = timeline.get_timeline_plot(
            event_type.name, create_if_missing=False
        )

        if existing_plot is not None:
            if any(isinstance(i, pg.PlotDataItem) for i in existing_plot.items):
                register_data_actions()
                return
            plot_item = existing_plot
        else:
            plot_item = timeline.get_timeline_plot(
                event_type.name, create_if_missing=True
            )
            plot_item.getViewBox().allow_y_panning = False
            plot_item.setYRange(-0.5, 0.5)

        timeline.add_timeline_scatter(event_type.name, [], is_event=True)

        if event_type.name not in IMMUTABLE_EVENTS:
            action = self.register_timeline_action(
                f"Add Event/{event_type.name}", None, lambda: self.add_event(event_type)
            )
            self.app.main_window.sort_action_menu("Timeline/Add Event")

            with contextlib.suppress(RuntimeError):
                event_type.name_changed.disconnect()
            event_type.name_changed.connect(lambda old, new: action.setText(new))

        register_data_actions()
        with contextlib.suppress(RuntimeError):
            event_type.changed.disconnect(self.changed.emit)
        event_type.changed.connect(self.changed.emit)

    def add_event(self, event_type: EventType, ts: int | None = None) -> None:
        if self.recording is None:
            return

        if ts is None:
            ts = self.app.current_ts

        if event_type.uid not in self.events:
            self.events[event_type.uid] = []

        if ts in self.events[event_type.uid]:
            return

        self.events[event_type.uid].append(ts)
        self.save_cached_json("events.json", self.events)
        timeline = self.get_timeline()
        timeline.setUpdatesEnabled(False)
        try:
            self._update_all_timeline_data()
        finally:
            timeline.setUpdatesEnabled(True)
            timeline.sort_plots()

    def delete_event_instance(self, timeline_name, data_point, event_type) -> None:
        if event_type.uid not in self.events:
            return

        events_list = self.events[event_type.uid]
        target_ts = data_point[0]

        if not events_list:
            return

        closest_event = min(events_list, key=lambda x: abs(x - target_ts))

        if abs(closest_event - target_ts) < 5:
            events_list.remove(closest_event)
            self.save_cached_json("events.json", self.events)
            timeline = self.get_timeline()
            timeline.setUpdatesEnabled(False)
            try:
                self._update_all_timeline_data()
            finally:
                timeline.setUpdatesEnabled(True)
                timeline.sort_plots()

    def delete_event_from_aggregate(self, data_point) -> None:
        target_ts = data_point[0]
        best_dist = float("inf")
        best_et = None
        best_ts = None

        for uid, times in self.events.items():
            if uid in IMMUTABLE_EVENTS:
                continue
            try:
                et = self.get_event_type(uid)
            except ValueError:
                continue

            if not times:
                continue

            closest = min(times, key=lambda x: abs(x - target_ts))
            dist = abs(closest - target_ts)
            if dist < best_dist:
                best_dist = dist
                best_et = et
                best_ts = closest

        if best_et and best_dist < 5:
            self.delete_event_instance("00_Events", (best_ts, 0), best_et)

    def seek_to_event_instance(self, data_point) -> None:
        self.app.seek_to(data_point[0])

    def _update_timeline_data(self, event_type: EventType) -> None:
        timeline = self.get_timeline()
        event_name = event_type.name
        plot_item = timeline.get_timeline_plot(event_name, True)

        raw_events = self.events.get(event_type.uid, [])
        if raw_events:
            data = np.array([[t, 0] for t in raw_events], dtype=np.float64)
        else:
            data = np.empty((0, 2))

        scatter_item = next(
            (item for item in plot_item.items if isinstance(item, pg.PlotDataItem)),
            None,
        )

        if scatter_item is None:
            timeline.add_timeline_scatter(
                event_name,
                data,
                is_event=True,
            )
        else:
            scatter_item.setData(data)

    @property
    @property_params(
        add_button_text="Create new event type",
        item_params={"label_field": "name"},
        prevent_add=True,
        primary=True,
    )
    def event_types(self) -> list[EventType]:
        return self._event_types

    @event_types.setter
    def event_types(self, value: list[EventType]) -> None:
        unique_value = []
        seen_names = set()
        for et in value:
            if et.name and et.name in seen_names:
                continue
            if et.name:
                seen_names.add(et.name)
            unique_value.append(et)

        value = unique_value

        new_event_types = [
            event_type for event_type in value if event_type not in self._event_types
        ]
        removed_event_types = [
            event_type for event_type in self._event_types if event_type not in value
        ]

        self._event_types = value

        for new_event_type in new_event_types:
            if new_event_type.uid == "":
                new_event_type.uid = str(uuid.uuid4())

            if new_event_type.uid not in self.events:
                self.events[new_event_type.uid] = []

            event_type_counter = 1
            while new_event_type.name == "":
                candidate_name = f"event-{event_type_counter}"
                if candidate_name not in [s.name for s in self._event_types]:
                    new_event_type.name = candidate_name
                event_type_counter += 1

            self._update_all_timeline_data()
            new_event_type.changed.connect(self.changed.emit)
            new_event_type.name_changed.connect(
                lambda old, new, et=new_event_type: self._on_event_name_changed(
                    old, new, et
                )
            )

        kept_uids = {et.uid for et in value}
        for removed_event_type in removed_event_types:
            if (
                removed_event_type.uid in self.events
                and removed_event_type.uid not in kept_uids
            ):
                del self.events[removed_event_type.uid]

            self.get_timeline().remove_timeline_plot(removed_event_type.name)
            self.save_cached_json("events.json", self.events)
            self.app.main_window.unregister_action(
                f"Timeline/Add Event/{removed_event_type.name}"
            )

    def _on_event_name_changed(self, old_name, new_name, event_type) -> None:
        timeline = self.get_timeline()
        plot = timeline.get_timeline_plot(old_name)
        if plot:
            timeline.timeline_plots[new_name] = timeline.timeline_plots.pop(old_name)
            if old_name in timeline.timeline_legends:
                timeline.timeline_legends[new_name] = timeline.timeline_legends.pop(
                    old_name
                )

            if old_name in timeline.data_point_actions:
                timeline.data_point_actions[new_name] = timeline.data_point_actions.pop(
                    old_name
                )

            if old_name in timeline.hover_lines:
                timeline.hover_lines[new_name] = timeline.hover_lines.pop(old_name)

            legend_container = timeline.timeline_legends[new_name].parentItem()
            for item in legend_container.items:
                if isinstance(item, pg.LabelItem):
                    item.setText(new_name)
                    break

        self.app.main_window.unregister_action(f"Timeline/Add Event/{old_name}")
        self._update_all_timeline_data()
        timeline.sort_plots()
        self.save_cached_json("events.json", self.events)

    def create_event_type(self, event_name: str) -> EventType:
        event_type = EventType()
        event_type.uid = str(uuid.uuid4())
        event_type._name = event_name

        self.event_types = [*self._event_types, event_type]

        return event_type

    def get_event_type_by_name(self, name: str) -> EventType | None:
        for event_type in self._event_types:
            if event_type.name == name:
                return event_type

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
        start_time, stop_time = neon_player.instance().recording_settings.export_window
        event_names = []
        for uid in self.events:
            name = uid if uid in IMMUTABLE_EVENTS else self.get_event_type(uid).name
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
