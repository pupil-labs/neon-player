import json
import logging
import numpy as np
import pandas as pd
import pyxdf

from pathlib import Path
from PySide6.QtCore import Signal
from PySide6.QtGui import QIcon
from qt_property_widgets.utilities import (
    FilePath,
    property_params,
    action_params,
)
from qt_property_widgets.widgets import DynamicComboWidget
from scipy.signal import butter, filtfilt
from typing import Optional

from pupil_labs import neon_player
from pupil_labs.neon_player import Plugin, action
from pupil_labs.neon_recording import NeonRecording


class XDFMultimodalPlugin(Plugin):
    label = "XDF Multimodal"
    streams_changed = Signal()
    sync_events_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._xdf_path: Path = Path("")
        self._available_stream_names: list[str] = []
        self._available_marker_stream_names: list[str] = []
        self._available_sync_events: list[str] = []
        self._data_stream_name: str = ""
        self._marker_stream_name: str = ""
        self._selected_sync_event: str = ""

        self._stream_data: Optional[np.ndarray] = None
        self._stream_ts: Optional[np.ndarray] = None
        self._stream_fs: float = 250.0
        self._apply_bandpass: bool = False
        self._xdf_markers: list[dict] = []
        self._channel_names: list[str] = []   # ordered channel names from XDF
        self._channels: dict[str, bool] = {}  # channel name -> enabled

        self._offset_s: float = 0.0
        self._is_aligned: bool = False

    @property
    @property_params(label="File Path (.xdf)")
    def file_path(self) -> FilePath:
        # Returning None keeps FilePathWidget visually empty.
        return FilePath(self._xdf_path) if self._xdf_path != Path("") else None

    @file_path.setter
    def file_path(self, value: FilePath | None) -> None:
        p = Path(str(value)) if value else Path("")

        # Allow clearing the field programmatically.
        if str(p) in ("", "."):
            if self._xdf_path != Path(""):
                self._xdf_path = Path("")
                self.changed.emit()
            return

        if p != self._xdf_path:
            self._xdf_path = p
            self.changed.emit()
            if self._xdf_path.exists() and self._xdf_path.is_file():
                self.load_xdf()

    @property
    @property_params(
        label="Data Stream",
        widget=DynamicComboWidget,
        options_source="_available_stream_names",
        options_changed_signal="streams_changed",
    )
    def data_stream(self) -> str:
        return self._data_stream_name

    @data_stream.setter
    def data_stream(self, value: str | None) -> None:
        clean_value = str(value or "").strip()
        if self._data_stream_name != clean_value:
            self._data_stream_name = clean_value
            self.changed.emit()
            if self._xdf_path.exists() and self._xdf_path.is_file():
                self.load_xdf()

    @property
    @property_params(
        label="Marker Stream",
        widget=DynamicComboWidget,
        options_source="_available_marker_stream_names",
        options_changed_signal="streams_changed",
    )
    def marker_stream(self) -> str:
        return self._marker_stream_name

    @marker_stream.setter
    def marker_stream(self, value: str | None) -> None:
        clean_value = str(value or "").strip()
        if self._marker_stream_name != clean_value:
            self._marker_stream_name = clean_value
            self.changed.emit()
            if self._xdf_path.exists() and self._xdf_path.is_file():
                self.load_xdf()

    @property
    @property_params(
        label="Sync Event",
        widget=DynamicComboWidget,
        options_source="_available_sync_events",
        options_changed_signal="sync_events_changed",
    )
    def common_sync_events(self) -> str:
        return self._selected_sync_event

    @common_sync_events.setter
    def common_sync_events(self, value: str | None) -> None:
        clean_value = str(value or "").strip()
        if self._selected_sync_event != clean_value:
            self._selected_sync_event = clean_value
            self._is_aligned = False
            self.align_with_recording()
            self.changed.emit()

    @property
    @property_params(label="Apply Bandpass 1-30 Hz (EEG-like streams)")
    def apply_bandpass(self) -> bool:
        return self._apply_bandpass

    @apply_bandpass.setter
    def apply_bandpass(self, value: bool) -> None:
        new_value = bool(value)
        if self._apply_bandpass != new_value:
            self._apply_bandpass = new_value
            self.update_timeline()

    @property
    @property_params(label="Channel Selection")
    def channels(self) -> dict[str, bool]:
        return self._channels

    @channels.setter
    def channels(self, value: dict[str, bool]) -> None:
        self._channels = value
        self.update_timeline()

    def on_recording_loaded(self, recording: NeonRecording) -> None:
        # Reset transient timeline/UI state for the newly opened recording.
        self._clear_timeline_tracks()

        # Keep persisted file_path from settings. If it is still valid, reload it
        # automatically so the XDF opens together with the recording.
        if self._xdf_path.exists() and self._xdf_path.is_file():
            self.load_xdf()
        else:
            self._available_stream_names = []
            self._available_marker_stream_names = []
            self._available_sync_events = []
            self.streams_changed.emit()
            self.sync_events_changed.emit()
            self._stream_data = None
            self._stream_ts = None
            self._xdf_markers = []
            self._channel_names = []
            self._channels = {}
            self._is_aligned = False

    def on_disabled(self) -> None:
        self._clear_timeline_tracks()

    def _set_common_sync_events(self, event_names: list[str]) -> None:
        unique_names = sorted({name for name in event_names if name})
        selected = self._selected_sync_event if self._selected_sync_event in unique_names else ""
        if not selected and unique_names:
            selected = unique_names[0]

        changed = (unique_names != self._available_sync_events) or (selected != self._selected_sync_event)
        self._available_sync_events = unique_names
        self._selected_sync_event = selected
        self.sync_events_changed.emit()

        if changed:
            self.changed.emit()

    def _clear_timeline_tracks(self) -> None:
        if self.headless:
            return

        timeline = self.get_timeline()
        was_sorting_enabled = timeline.disable_plot_sorting()

        for row_name in (*self._channel_names, "Data Stream", "XDF Markers"):
            timeline.remove_timeline_plot(row_name)

        if was_sorting_enabled:
            timeline.enable_plot_sorting()

    def _to_numeric_matrix(self, time_series) -> Optional[np.ndarray]:
        """Best-effort conversion from XDF time_series to a 2D numeric array."""
        if time_series is None:
            return None

        try:
            data = np.asarray(time_series, dtype=np.float32)
        except Exception:
            try:
                # Handles streams where samples are nested lists/tuples of numeric strings.
                data = np.array([[float(v) for v in sample] for sample in time_series], dtype=np.float32)
            except Exception:
                return None

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if data.ndim != 2:
            return None

        return data

    def load_xdf(self):
        try:
            logging.info(f"Loading XDF file: {self._xdf_path}")
            streams, header = pyxdf.load_xdf(str(self._xdf_path))
            data_stream = None
            marker_stream = None

            # Clear all existing timeline rows (channels + Data Stream + XDF Markers)
            # before loading new data, so stale rows from a previous XDF don't persist.
            self._clear_timeline_tracks()

            # Reset data so stream re-selection cannot leave stale content behind.
            self._stream_data = None
            self._stream_ts = None
            self._xdf_markers = []
            self._channel_names = []
            self._channels = {}

            stream_names: list[str] = []
            marker_stream_names: list[str] = []
            non_marker_stream_names: list[str] = []
            for stream in streams:
                info = stream.get("info", {})
                stream_name = str(info.get("name", [""])[0])
                if not stream_name:
                    continue
                stream_names.append(stream_name)

                stream_type = str(info.get("type", [""])[0]).strip().lower()
                if stream_type in ("markers", "event"):
                    marker_stream_names.append(stream_name)
                else:
                    non_marker_stream_names.append(stream_name)

            # Fallback: if no stream exposes type=Markers metadata, keep old behavior.
            if not marker_stream_names:
                marker_stream_names = list(stream_names)

            # Prefer non-marker streams for data selection, but keep a fallback when
            # type metadata is missing or all streams are marker-typed.
            self._available_stream_names = non_marker_stream_names or list(stream_names)
            self._available_marker_stream_names = marker_stream_names
            if self._data_stream_name not in self._available_stream_names:
                self._data_stream_name = ""
            if self._marker_stream_name not in self._available_marker_stream_names:
                self._marker_stream_name = ""
            self.streams_changed.emit()

            for stream in streams:
                name = stream["info"]["name"][0]
                if self._data_stream_name and name == self._data_stream_name:
                    data_stream = stream
                if self._marker_stream_name and name == self._marker_stream_name:
                    marker_stream = stream
            
            if data_stream:
                numeric_data = self._to_numeric_matrix(data_stream.get("time_series"))
                if numeric_data is None:
                    logging.error(
                        "Selected stream '%s' is not numeric. Choose a numeric stream to plot.",
                        self._data_stream_name,
                    )
                else:
                    self._stream_data = numeric_data
                    self._stream_ts = np.asarray(data_stream["time_stamps"], dtype=np.float64)

                try:
                    self._stream_fs = float(data_stream["info"]["nominal_srate"][0])
                except Exception:
                    self._stream_fs = 250.0

                # If nominal sample rate is unavailable or invalid, estimate from timestamps.
                if (not np.isfinite(self._stream_fs)) or self._stream_fs <= 0:
                    if self._stream_ts is not None and len(self._stream_ts) > 2:
                        dt = np.diff(self._stream_ts)
                        dt = dt[np.isfinite(dt) & (dt > 0)]
                        if len(dt) > 0:
                            self._stream_fs = 1.0 / float(np.median(dt))
                        else:
                            self._stream_fs = 0.0
                    else:
                        self._stream_fs = 0.0
                
                # Extract channel names properly
                channel_names = []
                try:
                    desc = data_stream.get('info', {}).get('desc', [{}])[0]
                    ch_list = desc.get('channels', [{}])[0].get('channel', [])
                    if not ch_list:
                        ch_list = desc.get('channel', [])
                    for i, ch in enumerate(ch_list):
                        label_value = ch.get('label', [ch.get('name', [f"Ch{i}"])[0]])[0]
                        channel_names.append(str(label_value))
                except Exception:
                    pass
                if (not channel_names) and self._stream_data is not None:
                    channel_names = [f"Ch{i}" for i in range(self._stream_data.shape[1])]
                
                # Populate channels dict (preserve previous enabled state)
                self._channel_names = channel_names
                new_channels: dict[str, bool] = {}
                for idx, name in enumerate(channel_names):
                    if not self._channels:
                        # Default to the first channel for unknown stream types.
                        new_channels[name] = idx == 0
                    else:
                        new_channels[name] = self._channels.get(name, False)
                
                self.channels = new_channels
                if self._stream_data is not None:
                    logging.info(
                        "Found data stream '%s' with %d channels: %s",
                        self._data_stream_name,
                        self._stream_data.shape[1],
                        channel_names,
                    )
            
            if marker_stream:
                self._xdf_markers = []
                for ts, marker in zip(marker_stream["time_stamps"], marker_stream["time_series"], strict=False):
                    m_text = str(marker[0])
                    m_name = self._parse_event_name(m_text)
                    self._xdf_markers.append({"timestamp": ts, "name": m_name, "raw": m_text})
                logging.info(f"Found {len(self._xdf_markers)} markers in XDF")
            
            self.align_with_recording()
            self.changed.emit()
        except Exception as e:
            logging.exception(f"Failed to load XDF: {e}")

    def _parse_event_name(self, text: str) -> str:
        if not text: return ""
        text = text.strip()
        if '{' in text and '}' in text:
            try:
                start = text.find('{')
                end = text.rfind('}') + 1
                data = json.loads(text[start:end])
                if isinstance(data, dict):
                    return str(data.get('name', text)).strip()
            except: pass
        return text

    def _get_neon_ev_info(self, ev) -> tuple[str, Optional[float]]:
        if isinstance(ev, dict):
            raw_name = str(ev.get("event", ev.get("name", ev.get("text", str(ev)))))
            ts_ns = ev.get("time", ev.get("timestamp", ev.get("timestamp_ns", None)))
            return self._parse_event_name(raw_name), ts_ns

        raw_name = str(getattr(ev, "event", getattr(ev, "name", getattr(ev, "text", str(ev)))))
        name = self._parse_event_name(raw_name)
        ts_ns = getattr(ev, "time", getattr(ev, "timestamp", getattr(ev, "timestamp_ns", None)))
        return name, ts_ns

    def align_with_recording(self) -> None:
        if not self.recording or not self._xdf_markers:
            self._set_common_sync_events([])
            self._is_aligned = False
            self.update_timeline()
            return

        neon_events = []
        sources = [
            ("recording.events", getattr(self.recording, "events", None)),
        ]
        ep = Plugin.get_instance_by_name("EventsPlugin")
        if ep:
            mapped_ep_events: list[dict[str, object]] = []
            try:
                event_id_name_mapping: dict[object, str] = {
                    et.uid: et.name
                    for et in getattr(ep, "event_types", [])
                    if getattr(et, "uid", None) is not None and getattr(et, "name", None)
                }

                ep_events = getattr(ep, "events", None)
                if isinstance(ep_events, dict):
                    for event_id, timestamps in ep_events.items():
                        event_name = event_id_name_mapping.get(event_id)
                        if event_name is None:
                            continue
                        for ts_ns in timestamps:
                            if ts_ns is None:
                                continue
                            mapped_ep_events.append(
                                {
                                    "name": str(event_name),
                                    "timestamp_ns": int(ts_ns),
                                }
                            )
            except Exception:
                logging.exception("Failed to decode EventsPlugin.events")

            if mapped_ep_events:
                sources.append(("EventsPlugin.events(mapped)", mapped_ep_events))
            else:
                sources.append(("EventsPlugin.events", getattr(ep, "events", None)))

        for src_name, src_val in sources:
            if src_val is None: continue
            try:
                extracted = list(src_val) if not hasattr(src_val, "samples") else list(src_val.samples)
                if len(extracted) > 0:
                    neon_events = extracted
                    break
            except: pass

        if not neon_events:
            self._set_common_sync_events([])
            self._is_aligned = False
            self.update_timeline()
            return

        # Find Recording Bounds from events as primary source
        self._rec_begin_ns = self.recording.start_time
        self._rec_end_ns = self._rec_begin_ns + getattr(self.recording, "duration_ns", 0)

        for n_ev in neon_events:
            n_name, n_ts = self._get_neon_ev_info(n_ev)
            if n_ts is None: continue
            if n_name.lower() == "recording.begin":
                self._rec_begin_ns = n_ts
            elif n_name.lower() == "recording.end":
                self._rec_end_ns = n_ts

        neon_name_map: dict[str, str] = {}
        for ev in neon_events:
            ev_name, _ = self._get_neon_ev_info(ev)
            if ev_name:
                neon_name_map.setdefault(ev_name.lower(), ev_name)

        xdf_name_map: dict[str, str] = {}
        for marker in self._xdf_markers:
            marker_name = marker.get("name")
            if marker_name:
                xdf_name_map.setdefault(marker_name.lower(), marker_name)

        common_names = sorted(
            xdf_name_map[name_lower]
            for name_lower in set(neon_name_map).intersection(xdf_name_map)
        )
        self._set_common_sync_events(common_names)

        target_sync = self._selected_sync_event.strip().lower()

        # If no common sync event is selected, don't try to align.
        if not target_sync:
            logging.warning("No common sync event selected. Skipping alignment.")
            self._is_aligned = False
            self.update_timeline()
            return

        for n_ev in neon_events:
            n_name, n_ts = self._get_neon_ev_info(n_ev)
            if n_ts is None: continue
            if n_name.lower() == target_sync:
                for x_m in self._xdf_markers:
                    if x_m["name"].lower() == target_sync:
                        self._offset_s = x_m["timestamp"] - (n_ts * 1e-9)
                        self._is_aligned = True
                        logging.info(
                            f"Aligned using first occurrence of '{target_sync}'. Offset: {self._offset_s:.4f}s. "
                            "Note: if multiple events with this name exist, only the first is used for sync."
                        )
                        self.update_timeline()
                        return

        for n_ev in neon_events:
            n_name, n_ts = self._get_neon_ev_info(n_ev)
            if n_ts is None or n_name.lower() in ["recording.begin", "recording.end"]: continue
            for x_m in self._xdf_markers:
                if x_m["name"].lower() == n_name.lower():
                    self._offset_s = x_m["timestamp"] - (n_ts * 1e-9)
                    self._is_aligned = True
                    logging.info(f"Aligned using common marker '{n_name}'. Offset: {self._offset_s:.4f}s")
                    self.update_timeline()
                    return

        self._is_aligned = False
        self.update_timeline()

    def update_timeline(self):
        timeline = self.get_timeline()
        if not timeline or not self.recording:
            return

        # Clear all previously drawn rows before potentially drawing new content.
        self._clear_timeline_tracks()

        if self._stream_data is not None and self._is_aligned:
            # 1. Convert XDF timestamps to Neon clock (nanoseconds)
            neon_ts = ((self._stream_ts - self._offset_s) * 1e9).astype(np.int64)

            # 2. Filter data to fit within the recording bounds
            mask = (neon_ts >= self._rec_begin_ns) & (neon_ts <= self._rec_end_ns)

            if not np.any(mask):
                logging.warning("No data found within recording bounds.")
                return

            plot_ts = neon_ts[mask]
            plot_data = self._stream_data[mask]

            selected_names = [n for n in self._channel_names if self._channels.get(n, False)]

            if selected_names:
                indices = [self._channel_names.index(n) for n in selected_names]
                names = selected_names

                # Extract and clean data
                data = plot_data[:, indices].astype(np.float32)
                data = np.nan_to_num(data, nan=0.0)

                # Optional EEG-style filter for streams where that makes sense.
                if self._apply_bandpass and butter is not None and filtfilt is not None and self._stream_fs > 0:
                    try:
                        nyq = 0.5 * self._stream_fs
                        low, high = 1.0 / nyq, 30.0 / nyq
                        # Simple bandpass 1-30Hz
                        b, a = butter(4, [max(0.001, low), min(0.999, high)], btype='band')
                        data = filtfilt(b, a, data, axis=0)
                    except Exception as e:
                        logging.error(f"Filtering failed: {e}")
                elif self._apply_bandpass and self._stream_fs <= 0:
                    logging.warning(
                        "Bandpass filtering is enabled, but sample rate is invalid (%.3f Hz). Skipping filter.",
                        self._stream_fs,
                    )

                # 4. Normalize (Center the data around 0)
                data = data - np.nanmean(data, axis=0)

                # Ensure data is 2D: (Samples, Channels)
                if data.ndim == 1:
                    data = data.reshape(-1, 1)

                # 6. Plot the raw, centered data
                try:
                    plot_item = None
                    for i, name in enumerate(names):
                        channel_data = data[:, i]
                        plot_data_matrix = np.column_stack((plot_ts, channel_data))

                        plot_item = timeline.add_timeline_plot(
                            timeline_row_name=name,
                            data=plot_data_matrix,
                            plot_name=""
                        )
                        if plot_item is not None:
                            plot_item.preferred_height_2d = 60
                            plot_item.adjust_size()

                    # 7. Let the graph auto-range to fit the raw data naturally!
                    if plot_item is not None:
                        plot_item.getViewBox().enableAutoRange(y=True)

                except Exception as e:
                    logging.error(f"Failed to add EEG plot: {e}")

            # Update the status bars on the timeline
            try:
                start_ns = int(plot_ts[0])
                end_ns = int(plot_ts[-1])
                timeline.add_timeline_broken_bar("Data Stream", [(start_ns, end_ns)], "", "#00FFFF")

                if self._xdf_markers:
                    # Filter and plot markers that fall within the recording
                    marker_segs = []
                    for m in self._xdf_markers:
                        m_ts_ns = int((m["timestamp"] - self._offset_s) * 1e9)
                        if self._rec_begin_ns <= m_ts_ns <= self._rec_end_ns:
                            # 100ms duration for visibility
                            marker_segs.append((m_ts_ns, m_ts_ns + 100_000_000))

                    if marker_segs:
                        timeline.add_timeline_broken_bar("XDF Markers", marker_segs, "", "#FFCC00")
            except Exception as e:
                logging.error(f"Failed to update tracks: {e}")

        self.changed.emit()

    @action
    @action_params(compact=True, icon=QIcon(str(neon_player.asset_path("export.svg"))))
    def export(self, destination: Path = Path()) -> None:
        if self._stream_data is None or not self._is_aligned:
            logging.warning("Cannot export: no aligned data available.")
            return

        neon_ts = ((self._stream_ts - self._offset_s) * 1e9).astype(np.int64)
        mask = (neon_ts >= self._rec_begin_ns) & (neon_ts <= self._rec_end_ns)

        start_time, stop_time = self.app.get_export_window()
        mask &= (neon_ts >= start_time) & (neon_ts <= stop_time)

        if not np.any(mask):
            logging.warning("No data in the export window.")
            return

        export_ts = neon_ts[mask]
        export_data = self._stream_data[mask]

        df = pd.DataFrame({"timestamp [ns]": export_ts})
        for i, name in enumerate(self._channel_names):
            df[name] = export_data[:, i].astype(np.float32)

        export_file = destination / "xdf_stream.csv"
        df.to_csv(export_file, index=False)
        logging.info(f"Exported {len(df)} samples to {export_file}")

    @action
    @action_params(compact=True, icon=QIcon.fromTheme("help-about"))
    def debug_events(self) -> None:
        if not self.recording:
            return

        logging.info("--- NEON EVENTS ---")
        evs = getattr(self.recording, "events", [])
        for e in evs:
            name, ts = self._get_neon_ev_info(e)
            logging.info(f"Neon: '{name}' @ {ts}")
        logging.info("--- XDF MARKERS ---")
        for m in self._xdf_markers:
            logging.info(f"XDF: '{m['name']}' @ {m['timestamp']}")

    @action
    @action_params(compact=True, icon=QIcon.fromTheme("edit-select-all"), label="Enable All")
    def enable_all_channels(self) -> None:
        self.channels = {name: True for name in self._channel_names}

    @action
    @action_params(compact=True, icon=QIcon.fromTheme("edit-clear"), label="Disable All")
    def disable_all_channels(self) -> None:
        self.channels = {name: False for name in self._channel_names}