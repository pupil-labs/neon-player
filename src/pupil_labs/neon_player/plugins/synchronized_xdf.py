import logging
import json
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
from collections.abc import Generator

from pupil_labs import neon_player
from pupil_labs.neon_player import Plugin, action
from pupil_labs.neon_player.job_manager import ProgressUpdate
from pupil_labs.neon_recording import NeonRecording


class XDFStream:
    """Parses and holds the data from a single XDF stream dict."""

    def __init__(self) -> None:
        self.name: str = ""
        self.type: str = ""
        self.type_display: str = ""
        self.channel_count: int = -1
        self.channel_format: str = ""
        self._is_marker_stream: bool = False

    @property
    def is_data_stream(self) -> bool:
        return not self._is_marker_stream

    @property
    def is_marker_stream(self) -> bool:
        return self._is_marker_stream

    @classmethod
    def from_dict(cls, xdf_dict: dict) -> "XDFStream":
        stream = cls()
        info = xdf_dict.get("info", {})
        stream.name = str(info.get("name", [""])[0])
        stream.type_display = str(info.get("type", [""])[0]).strip()
        stream.type = stream.type_display.lower()
        stream.channel_count = int(info.get("channel_count", [-1])[0])
        stream.channel_format = str(info.get("channel_format", [""])[0]).strip().lower()

        stream._is_marker_stream = (
            stream.channel_format == "string"
            or stream.type in ("markers", "event")
        )

        if stream.is_marker_stream:
            stream.markers = cls._parse_markers(xdf_dict)
            return stream

        stream.data, stream.timestamps, stream.fs = cls._parse_stream_data(xdf_dict)

        parsed_channel_names = cls._parse_channel_names(xdf_dict)
        if parsed_channel_names is not None:
            stream.channel_names = parsed_channel_names
        else:
            n_channels = stream.data.shape[1] if stream.data is not None else stream.channel_count
            stream.channel_names = cls._fallback_channel_names(n_channels)
        return stream

    @staticmethod
    def _parse_stream_data(
        xdf_dict: dict,
    ) -> "tuple[Optional[np.ndarray], Optional[np.ndarray], float]":
        time_series = xdf_dict.get("time_series")
        data = XDFStream._to_numeric(time_series)

        if data is None:
            return None, None, 0.0

        timestamps = np.asarray(xdf_dict["time_stamps"], dtype=np.float64)
        fs = float(xdf_dict["info"]["nominal_srate"][0])

        if (not np.isfinite(fs)) or fs <= 0:
            if len(timestamps) > 2:
                dt = np.diff(timestamps)
                dt = dt[np.isfinite(dt) & (dt > 0)]
                fs = 1.0 / float(np.median(dt)) if len(dt) > 0 else 0.0
            else:
                fs = 0.0

        return data, timestamps, fs

    @staticmethod
    def _to_numeric(time_series) -> "Optional[np.ndarray]":
        if time_series is None:
            return None
        try:
            data = np.asarray(time_series, dtype=np.float32)
        except Exception:
            try:
                data = np.array(
                    [[float(v) for v in sample] for sample in time_series],
                    dtype=np.float32,
                )
            except Exception:
                return None
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return data

    @staticmethod
    def _parse_channel_names(xdf_dict: dict) -> "Optional[list[str]]":
        desc = xdf_dict.get("info", {}).get("desc", [{}])[0]
        ch_list = desc.get("channels", [{}])[0].get("channel", [])
        if not ch_list:
            ch_list = desc.get("channel", [])
        if not ch_list:
            return None
        channel_names = []
        for i, ch in enumerate(ch_list):
            label_value = ch.get("label", [ch.get("name", [f"Ch{i+1}"])[0]])[0]
            channel_names.append(str(label_value))
        return channel_names

    @staticmethod
    def _fallback_channel_names(channel_count: int) -> list[str]:
        return [f"Ch{i+1}" for i in range(max(0, channel_count))]

    @staticmethod
    def _parse_markers(xdf_dict: dict) -> list[dict]:
        markers = []
        for ts, marker in zip(xdf_dict["time_stamps"], xdf_dict["time_series"], strict=False):
            m_text = str(marker[0])
            m_name = XDFStream._parse_event_name(m_text)
            markers.append({"timestamp": ts, "name": m_name, "raw": m_text})
        return markers

    @staticmethod
    def _parse_event_name(text: str) -> str:
        return text.strip() # For now, just strip whitespace. Future improvements could parse JSON or other structured formats.


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
        self._data_stream_type: str = ""
        self._marker_stream_name: str = ""
        self._selected_sync_event: str = ""

        self._stream_data: Optional[np.ndarray] = None
        self._stream_ts: Optional[np.ndarray] = None
        self._stream_fs: float = 0
        self._apply_bandpass: bool = False
        self._xdf_markers: list[dict] = []
        self._channel_names: list[str] = []   # ordered channel names from XDF
        self._channels: dict[str, bool] = {}  # channel name -> enabled

        self._offset_s: float = 0.0
        self._is_aligned: bool = False
        self._xdf_load_job = None
        self._reload_after_job = False
        self._active_channel_row_names: list[str] = []

    def _get_data_stream_group_title(self) -> str:
        stream_type = self._data_stream_type.strip()
        return f"XDF - {stream_type}" if stream_type else "XDF - Data Stream"

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
            return

        if p != self._xdf_path:
            self._xdf_path = p
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

    @property
    @property_params(label="Apply Bandpass 1-30 Hz (EEG-like streams)")
    def apply_bandpass(self) -> bool:
        return self._apply_bandpass

    @apply_bandpass.setter
    def apply_bandpass(self, value: bool) -> None:
        if self._apply_bandpass != value:
            self._apply_bandpass = value
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

        channel_row_names = [
            f"{self._get_data_stream_group_title()} - {channel_name}"
            for channel_name in self._channel_names
        ]

        for row_name in (
            *self._active_channel_row_names,
            *channel_row_names,
            "Data Stream",
            "XDF Markers",
        ):
            timeline.remove_timeline_plot(row_name)

        self._active_channel_row_names = []

        if was_sorting_enabled:
            timeline.enable_plot_sorting()

    def _get_selected_stream_info_cache_file(self) -> Path:
        safe_stream_name = "".join(
            c if c.isalnum() or c in ("-", "_") else "_"
            for c in self._data_stream_name
        )
        return self.get_cache_path() / f"xdf_stream_{safe_stream_name}.json"

    def _get_xdf_meta_cache_file(self) -> Path:
        return self.get_cache_path() / "xdf_selection_meta.json"

    def _reset_loaded_xdf_state(self) -> None:
        self._stream_data = None
        self._stream_ts = None
        self._stream_fs = 0
        self._xdf_markers = []
        self._data_stream_type = ""
        self._channel_names = []
        self._channels = {}

    def _restore_channel_selection(self, channel_names: list[str]) -> None:
        self._channel_names = list(channel_names)
        previous_channels = dict(self._channels)
        new_channels: dict[str, bool] = {}
        for idx, ch_name in enumerate(self._channel_names):
            if previous_channels:
                new_channels[ch_name] = previous_channels.get(ch_name, False)
            else:
                new_channels[ch_name] = idx == 0
        self.channels = new_channels

    def load_xdf(self):
        #Prevent overlapping loads
        if self._xdf_load_job is not None:
            self._reload_after_job = True
            return

        # Fast path: if this xdf stream is already cached, load it instantly.
        if self._load_xdf_from_cache(log_missing=False):
            return

        self._clear_timeline_tracks()
        self._reset_loaded_xdf_state()

        if self.app.headless:
            # pump background job updates until it is complete
            for _ in self._bg_load_xdf(
                str(self._xdf_path),
                self._data_stream_name,
                self._marker_stream_name,
            ):
                pass
            self._load_xdf_from_cache()
            return

        self._reload_after_job = False
        self._xdf_load_job = self.job_manager.run_background_action(
            "Loading XDF streams",
            "XDFMultimodalPlugin._bg_load_xdf",
            self._xdf_path,
            self._data_stream_name,
            self._marker_stream_name,
        )
        if self._xdf_load_job is not None:
            self._xdf_load_job.finished.connect(self._on_xdf_load_finished)

    def _on_xdf_load_finished(self) -> None:
        self._xdf_load_job = None
        if self._reload_after_job:
            self._reload_after_job = False
            self.load_xdf()
            return
        self._load_xdf_from_cache()

    def _bg_load_xdf(
        self,
        xdf_path: str,
        data_stream_name: str,
        marker_stream_name: str,
    ) -> Generator[ProgressUpdate, None, None]:
        try:
            xdf_file = Path(xdf_path)
            logging.info("Loading XDF in background: %s", xdf_file)
            yield ProgressUpdate(0.1)

            streams, _ = pyxdf.load_xdf(str(xdf_file))
            xdf_streams = [XDFStream.from_dict(s) for s in streams]
            xdf_streams = [s for s in xdf_streams if s.name]

            marker_stream_names = [s.name for s in xdf_streams if s.is_marker_stream]
            non_marker_stream_names = [s.name for s in xdf_streams if s.is_data_stream]
            all_stream_names = [s.name for s in xdf_streams]
            if not marker_stream_names:
                marker_stream_names = list(all_stream_names)

            selected_data_stream: dict[str, object] | None = None
            marker_stream_payloads: dict[str, list[dict]] = {}
            n_streams = len(xdf_streams)

            for idx, stream in enumerate(xdf_streams):
                if data_stream_name and stream.name == data_stream_name:
                    if stream.data is not None:
                        safe_stream_name = "".join(
                            c if c.isalnum() or c in ("-", "_") else "_"
                            for c in data_stream_name
                        )
                        data_cache_file = self.get_cache_path() / f"xdf_stream_{safe_stream_name}.npy"
                        data_cache_file.parent.mkdir(parents=True, exist_ok=True)

                        # One NPY per selected data stream. First column is timestamps.
                        stream_matrix = np.column_stack((stream.timestamps, stream.data))
                        np.save(str(data_cache_file), stream_matrix.astype(np.float32))

                        selected_data_stream = {
                            "name": stream.name,
                            "type_display": stream.type_display,
                            "fs": stream.fs,
                            "channel_names": stream.channel_names,
                            "data_file": data_cache_file.name,
                        }

                        stream_info_cache_file = self.get_cache_path() / f"xdf_stream_{safe_stream_name}.json"
                        with stream_info_cache_file.open("w", encoding="utf-8") as stream_meta_fp:
                            json.dump(
                                {
                                    "source_path": str(xdf_file.resolve()),
                                    **selected_data_stream,
                                },
                                stream_meta_fp,
                            )

                if stream.is_marker_stream:
                    marker_stream_payloads[stream.name] = stream.markers
                    
                yield ProgressUpdate(0.2 + (0.7 * (idx + 1) / n_streams))

            meta_payload = {
                "source_path": str(xdf_file.resolve()),
                "available_stream_names": non_marker_stream_names or list(all_stream_names),
                "available_marker_stream_names": marker_stream_names,
                "selected_data_stream": selected_data_stream,
                "marker_stream_payloads": marker_stream_payloads,
            }

            meta_cache_file = self.get_cache_path() / "xdf_selection_meta.json"
            meta_cache_file.parent.mkdir(parents=True, exist_ok=True)
            with meta_cache_file.open("w", encoding="utf-8") as meta_fp:
                json.dump(meta_payload, meta_fp)
        except Exception:
            logging.exception("Failed to load XDF in background")

        yield ProgressUpdate(1.0)

    def _load_xdf_from_cache(self, *, log_missing: bool = True) -> bool:
        meta_cache_file = self._get_xdf_meta_cache_file()
        if not meta_cache_file.exists():
            if log_missing:
                logging.error("XDF cache metadata file not found: %s", meta_cache_file)
            return False

        try:
            with meta_cache_file.open("r", encoding="utf-8") as meta_fp:
                meta_payload = json.load(meta_fp)

            source_path = meta_payload.get("source_path", "")
            if source_path != str(self._xdf_path.resolve()):
                logging.info("Ignoring stale XDF cache metadata for %s", source_path)
                return False

            self._available_stream_names = list(meta_payload.get("available_stream_names", []))
            self._available_marker_stream_names = list(meta_payload.get("available_marker_stream_names", []))
            if self._data_stream_name not in self._available_stream_names:
                self._data_stream_name = ""
            if self._marker_stream_name not in self._available_marker_stream_names:
                self._marker_stream_name = ""
            self.streams_changed.emit()

            self._stream_data = None
            self._stream_ts = None
            self._stream_fs = 0
            self._xdf_markers = []
            self._data_stream_type = ""
            self._channel_names = []

            loaded_data_from_cache = False
            selected_data_stream = meta_payload.get("selected_data_stream")
            if isinstance(selected_data_stream, dict) and selected_data_stream.get("name") == self._data_stream_name:
                data_file_name = selected_data_stream.get("data_file")
                if isinstance(data_file_name, str):
                    data_cache_file = self.get_cache_path() / data_file_name
                    if data_cache_file.exists():
                        stream_matrix = np.load(str(data_cache_file))
                        if stream_matrix.ndim == 2 and stream_matrix.shape[1] >= 2:
                            self._stream_ts = stream_matrix[:, 0].astype(np.float64)
                            self._stream_data = stream_matrix[:, 1:].astype(np.float32)
                            self._stream_fs = float(selected_data_stream.get("fs", 0.0))
                            self._data_stream_type = str(selected_data_stream.get("type_display", ""))
                            self._restore_channel_selection(
                                list(selected_data_stream.get("channel_names", []))
                            )
                            loaded_data_from_cache = True
                        else:
                            logging.error("Invalid cached stream matrix for '%s'", self._data_stream_name)
                    else:
                        logging.error("Missing cached stream file: %s", data_cache_file)

            # If main metadata does not match current selection, try stream-specific cache.
            if self._data_stream_name and not loaded_data_from_cache:
                stream_info_cache_file = self._get_selected_stream_info_cache_file()
                if stream_info_cache_file.exists():
                    with stream_info_cache_file.open("r", encoding="utf-8") as stream_meta_fp:
                        stream_meta = json.load(stream_meta_fp)

                    if (
                        stream_meta.get("source_path") == str(self._xdf_path.resolve())
                        and stream_meta.get("name") == self._data_stream_name
                    ):
                        data_file_name = stream_meta.get("data_file")
                        if isinstance(data_file_name, str):
                            data_cache_file = self.get_cache_path() / data_file_name
                            if data_cache_file.exists():
                                stream_matrix = np.load(str(data_cache_file))
                                if stream_matrix.ndim == 2 and stream_matrix.shape[1] >= 2:
                                    self._stream_ts = stream_matrix[:, 0].astype(np.float64)
                                    self._stream_data = stream_matrix[:, 1:].astype(np.float32)
                                    self._stream_fs = float(stream_meta.get("fs", 0.0))
                                    self._data_stream_type = str(stream_meta.get("type_display", ""))
                                    self._restore_channel_selection(
                                        list(stream_meta.get("channel_names", []))
                                    )
                                    loaded_data_from_cache = True

            marker_loaded_from_cache = False
            marker_stream_payloads = meta_payload.get("marker_stream_payloads")
            if isinstance(marker_stream_payloads, dict) and self._marker_stream_name:
                cached_markers = marker_stream_payloads.get(self._marker_stream_name)
                if isinstance(cached_markers, list):
                    self._xdf_markers = list(cached_markers)
                    marker_loaded_from_cache = True

            self.align_with_recording()
            self.changed.emit()
            marker_ready = (not self._marker_stream_name) or marker_loaded_from_cache
            return loaded_data_from_cache and marker_ready
        except Exception:
            logging.exception("Failed to load XDF from cache")
            return False

    def _get_neon_ev_info(self, ev) -> tuple[str, Optional[float]]:
        name_keys = ("event", "name", "text")
        time_keys = ("time", "timestamp", "timestamp_ns")

        if isinstance(ev, dict):
            raw_name = str(next((ev[k] for k in name_keys if k in ev), str(ev)))
            ts_ns = next((ev[k] for k in time_keys if k in ev), None)
        else:
            raw_name = str(next((getattr(ev, k) for k in name_keys if hasattr(ev, k)), str(ev)))
            ts_ns = next((getattr(ev, k) for k in time_keys if hasattr(ev, k)), None)

        return XDFStream._parse_event_name(raw_name), ts_ns

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
            if src_val is None:
                continue
            try:
                extracted = list(src_val) if not hasattr(src_val, "samples") else list(src_val.samples)
                if len(extracted) > 0:
                    neon_events = extracted
                    break
            except Exception as exc:
                logging.debug("Failed to extract events from source %s: %s", src_name, exc, exc_info=True)
                
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

                # Extract and clean data
                data = plot_data[:, indices].astype(np.float32)
                data = np.nan_to_num(data, nan=0.0)

                # Optional EEG-style filter for streams where that makes sense.
                if self._apply_bandpass and self._stream_fs > 0:
                    nyq = 0.5 * self._stream_fs
                    low, high = 1.0 / nyq, 30.0 / nyq
                    # Simple bandpass 1-30Hz
                    b, a = butter(4, [max(0.001, low), min(0.999, high)], btype='band')
                    data = filtfilt(b, a, data, axis=0)
                elif self._apply_bandpass and self._stream_fs <= 0:
                    logging.warning(
                        "Bandpass filtering is enabled, but sample rate is invalid (%.3f Hz). Skipping filter.",
                        self._stream_fs,
                    )

                # 4. Normalize (Center the data around 0)
                data = data - np.nanmean(data, axis=0)

                # 6. Plot each channel in its own subplot under the XDF group prefix.
                plotted_row_names: list[str] = []
                for i, name in enumerate(selected_names):
                    channel_data = data[:, i]
                    plot_data_matrix = np.column_stack((plot_ts, channel_data))
                    row_name = f"{self._get_data_stream_group_title()} - {name}"
                    plotted_row_names.append(row_name)

                    plot_item = timeline.add_timeline_plot(
                        timeline_row_name=row_name,
                        data=plot_data_matrix,
                        plot_name="",
                    )
                    if plot_item is not None:
                        plot_item.preferred_height_2d = 60
                        plot_item.adjust_size()
                        plot_item.getViewBox().enableAutoRange(y=True)

                self._active_channel_row_names = plotted_row_names

            # Update the status bars on the timeline
            start_ns = int(plot_ts[0])
            end_ns = int(plot_ts[-1])
            timeline.add_timeline_broken_bar(
                "Data Stream",
                [(start_ns, end_ns)],
                "",
                "#00FFFF",
            )

            if self._xdf_markers:
                # Filter and plot markers that fall within the recording
                marker_segs = []
                for m in self._xdf_markers:
                    m_ts_ns = int((m["timestamp"] - self._offset_s) * 1e9)
                    if self._rec_begin_ns <= m_ts_ns <= self._rec_end_ns:
                        # 100ms duration for visibility
                        marker_segs.append((m_ts_ns, m_ts_ns + 100_000_000))

                if marker_segs:
                    timeline.add_timeline_broken_bar(
                        "XDF Markers",
                        marker_segs,
                        "",
                        "#FFCC00",
                    )
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