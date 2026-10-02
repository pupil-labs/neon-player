import logging
import json
import numpy as np
import pandas as pd
import pyxdf
import shutil
import uuid

from pathlib import Path
from PySide6.QtCore import Signal
from PySide6.QtGui import QIcon
from qt_property_widgets.utilities import (
    FilePath,
    PersistentPropertiesMixin,
    property_params,
    action_params,
)
from qt_property_widgets.widgets import DynamicComboWidget
from scipy.signal import butter, filtfilt
from typing import Any, Callable, Optional
from collections.abc import Iterator

from pupil_labs import neon_player
from pupil_labs.neon_player import Plugin, action
from pupil_labs.neon_player.job_manager import ProgressUpdate
from pupil_labs.neon_player.plugins.events import _load_events_from_recording
from pupil_labs.neon_recording import NeonRecording


def first(xdf_dict: dict[str, list[Any]], key: str, default: Any) -> Any:
    """Pick the first value from a str-list mapping.

    Many fields in the parsed XDF data contain one value which is wrapped
    in a list. This function retrieves this value or returns a fallback one
    if the key is not present in the mapping.
    """
    value = xdf_dict.get(key)
    if value is None:
        return default

    if not isinstance(value, list):
        return value

    return value[0]


def apply_offset(xdf_timestamps: np.ndarray, offset_ns: int) -> np.ndarray:
    """Apply offset to convert XDF timestamps to Neon ones."""
    return (np.array(xdf_timestamps) * 1e9).astype(np.int64) + offset_ns


class XDFStreamError(Exception):
    """Indicates that the parsed or cached XDF data is not valid."""


class XDFStream(PersistentPropertiesMixin):
    """Parses and holds the data from a single XDF stream dict.

    The reference specification is available at
    https://github.com/sccn/xdf/wiki/Specifications.
    """

    def __init__(self) -> None:
        self.xdf_id: int = -1
        self._uid: str = ""
        self._name: str = ""
        self._type: str = ""
        self._fs: float = 0.0
        self.channel_count: int = -1
        self.channel_format: str = ""
        self._is_marker_stream: bool = False
        self._loaded: bool = False

    @property
    def uid(self) -> str:
        return self._uid

    @uid.setter
    def uid(self, value: str) -> None:
        self._uid = value

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    @property_params(dont_encode=True)
    def display_name(self) -> str:
        return self.name or self.uid

    @property
    def type(self) -> str:
        return self._type

    @type.setter
    def type(self, value: str) -> None:
        self._type = value

    @property
    def is_marker_stream(self) -> bool:
        return self._is_marker_stream

    @is_marker_stream.setter
    def is_marker_stream(self, value: bool) -> None:
        self._is_marker_stream = value

    @property
    def fs(self) -> float:
        return self._fs

    @fs.setter
    def fs(self, value: float) -> None:
        self._fs = value

    @property
    @property_params(dont_encode=True)
    def loaded(self) -> bool:
        return self._loaded

    def raise_not_valid(self) -> None:
        if self.channel_count < 0:
            raise XDFStreamError("Channel count is missing or negative")

        if not self.channel_format:
            raise XDFStreamError("Channel format is missing or empty")

    def to_dict(
        self,
        include_class_name: bool = False,
        condition: Callable[[dict], bool] | None = None,
        recursive: bool = False
    ) -> dict[str, Any]:
        state = super().to_dict(include_class_name, condition, recursive)

        # NOTE: class name is required to restore whether it is a marker or a data stream
        state["__class__"] = self.__class__.__name__
        return state

    def parse_info(self, info: dict[str, list[Any]]) -> None:
        self.xdf_id = int(first(info, "id", -1))
        self._uid = str(first(info, "uid", uuid.uuid4()))
        # NOTE: according to the specification, the 'name' attribute is not mandatory
        # but should be present most of the times for interpretability purposes
        self._name = str(first(info, "name", "<No name>"))
        self._type = str(first(info, "type", "")).strip()
        self._fs = int(first(info, "nominal_rate", 0))
        self.channel_count = int(first(info, "channel_count", -1))
        self.channel_format = str(first(info, "channel_format", "")).strip().lower()

    @classmethod
    def from_xdf_dict(cls, xdf_dict: dict) -> "XDFStream":
        info = xdf_dict.get("info", {})
        channel_format = str(first(info, "channel_format", "")).strip().lower()
        if channel_format == "string":
            return MarkerXDFStream.from_xdf_dict(xdf_dict)
        else:
            return DataXDFStream.from_xdf_dict(xdf_dict)


class MarkerXDFStream(XDFStream):
    def __init__(self) -> None:
        super().__init__()
        self._is_marker_stream = True
        self.markers: dict[str, list[float]] = {}

    def load_markers(self, marker_cache_file: Path) -> None:
        if not marker_cache_file.exists():
            raise XDFStreamError(f"Marker cache file does not exist")

        try:
            with open(marker_cache_file, "r") as f:
                self.markers = json.load(f)
            self._loaded = True
        except Exception as e:
            raise XDFStreamError("Failed to load markers from cache")

    @classmethod
    def from_xdf_dict(cls, xdf_dict: dict) -> "MarkerXDFStream":
        stream = cls()
        stream.parse_info(xdf_dict.get("info", {}))
        stream.markers = cls._parse_markers(xdf_dict)
        stream._loaded = True
        return stream

    @staticmethod
    def _parse_markers(xdf_dict: dict) -> dict[str, list[float]]:
        if "time_stamps" not in xdf_dict or "time_series" not in xdf_dict:
            return {}

        markers: dict[str, list[float]] = {}
        for ts, marker in zip(xdf_dict["time_stamps"], xdf_dict["time_series"]):
            name = MarkerXDFStream._parse_event_name(str(marker[0]))
            if name not in markers:
                markers[name] = []

            try:
                timestamp = float(ts)
                markers[name].append(timestamp)
            except ValueError:
                logging.warning(
                    f"Could not parse marker timestamp {ts}, skipping the marker {marker}"
                )
        return markers

    @staticmethod
    def _parse_event_name(text: str) -> str:
        """Parse event names from strings.

        For now, just strip whitespace. Future improvements could parse JSON
        or other structured formats.
        """
        return text.strip()


class DataXDFStream(XDFStream):
    def __init__(self) -> None:
        super().__init__()
        self._is_marker_stream = False
        self.data: np.ndarray | None = None
        self.timestamps: np.ndarray | None = None
        self._channel_names: list[str] = []
        self._loaded = False

    @property
    def channel_names(self) -> list[str]:
        return self._channel_names

    @channel_names.setter
    def channel_names(self, value: list[str]) -> None:
        self._channel_names = value

    def load_data(self, data_cache_file: Path) -> None:
        if not data_cache_file.exists():
            raise XDFStreamError(f"Data cache file does not exist")

        stream_matrix = np.load(str(data_cache_file))
        if not stream_matrix.ndim == 2 or stream_matrix.shape[1] < 2:
            raise XDFStreamError("Invalid cached stream matrix")

        self.timestamps = stream_matrix[:, 0].astype(np.float64)
        self.data = stream_matrix[:, 1:].astype(np.float32)
        self._loaded = True

    def raise_not_valid(self) -> None:
        super().raise_not_valid()
        if self.data is None:
            raise XDFStreamError("Failed to parse stream data")

    @classmethod
    def from_xdf_dict(cls, xdf_dict: dict) -> "DataXDFStream":
        info = xdf_dict.get("info", {})

        stream = cls()
        stream.parse_info(info)
        stream.data, stream.timestamps, stream.fs = cls._parse_stream_data(xdf_dict)
        if stream.timestamps is not None and np.any(np.diff(stream.timestamps) < 0):
            logging.warning(
                f"Timestamps of the XDF stream {stream.display_name} are not "
                f"increasingly monotonically, sorting the timestamps"
            )
            stream.timestamps.sort()

        parsed_channel_names = cls._parse_channel_names(info)
        if parsed_channel_names is not None:
            stream.channel_names = parsed_channel_names
        else:
            n_channels = stream.data.shape[1] if stream.data is not None else stream.channel_count
            stream.channel_names = cls._fallback_channel_names(n_channels)
        stream._loaded = True

        return stream

    @staticmethod
    def _parse_stream_data(
        xdf_dict: dict,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        time_series = xdf_dict.get("time_series")
        data = DataXDFStream._to_numeric(time_series)

        if data is None:
            return None, None, 0.0

        timestamps = np.asarray(xdf_dict["time_stamps"], dtype=np.float64)
        fs = float(first(xdf_dict["info"], "nominal_srate", 0.0))
        if not np.isfinite(fs) or fs <= 0:
            fs = DataXDFStream._get_fs_from_timestamps(timestamps)

        return data, timestamps, fs

    @staticmethod
    def _get_fs_from_timestamps(timestamps: np.ndarray) -> float:
        if len(timestamps) < 2:
            return 0.0

        dt = np.diff(timestamps)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        return 1.0 / float(np.median(dt)) if len(dt) > 0 else 0.0

    @staticmethod
    def _to_numeric(time_series) -> Optional[np.ndarray]:
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
    def _parse_channel_names(info: dict) -> Optional[list[str]]:
        desc = first(info, "desc", {})
        channels = first(desc, "channels", {})
        ch_list = channels.get("channel", [])
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


class XDFMultimodalPlugin(Plugin):
    label = "XDF Multimodal"
    _XDF_CACHE_VERSION = 3
    streams_changed = Signal()
    sync_events_changed = Signal()

    def __init__(self) -> None:
        super().__init__()

        self._state_initialized = False
        self._xdf_path: Path = Path("")
        self._available_data_streams: list[tuple[str, DataXDFStream]] = []
        self._available_marker_streams: list[tuple[str, MarkerXDFStream]] = []
        self._streams_by_uid: dict[str, XDFStream] = {}
        self._data_stream_uid: str | None = None
        self._data_stream: DataXDFStream | None = None
        self._marker_stream_uid: str | None = None
        self._marker_stream: MarkerXDFStream | None = None
        self._available_sync_events: list[str] = []
        self._neon_events: dict[str, list[int]] = {}
        self._sync_event: str = ""
        self._apply_bandpass: bool = False
        self._channels: dict[str, bool] = {}  # channel name -> enabled

        self._offset_ns: int = 0
        self._is_aligned: bool = False
        self._xdf_load_job = None
        self._active_timeline_row_names: set[str] = set()

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

        if p == self._xdf_path:
            return

        self._xdf_path = p
        if self.file_path_valid:
            self.load_xdf()

    @property
    @property_params(widget=None, dont_encode=True)
    def file_path_valid(self) -> bool:
        return self._xdf_path.exists() and self._xdf_path.is_file()

    def _rebuild_stream_uid_mapping(self) -> None:
        self._streams_by_uid = {}
        for s in [*self.available_data_streams, *self.available_marker_streams]:
            self._streams_by_uid[s.uid] = s

    @property
    @property_params(widget=None)
    def available_data_streams(self) -> list[DataXDFStream]:
        return self._available_data_streams

    @available_data_streams.setter
    def available_data_streams(self, value: list[DataXDFStream]) -> None:
        self._available_data_streams = value
        self._rebuild_stream_uid_mapping()
        self.streams_changed.emit()

    @property
    @property_params(widget=None, dont_encode=True)
    def available_data_stream_options(self) -> list[tuple[str, str]]:
        return [(s.display_name, s.uid) for s in self.available_data_streams]

    @property
    @property_params(
        label="Data Stream",
        widget=DynamicComboWidget,
        options_source="available_data_stream_options",
        options_changed_signal="streams_changed",
    )
    def data_stream_uid(self) -> str:
        return self._data_stream_uid

    @data_stream_uid.setter
    def data_stream_uid(self, value: str | None) -> None:
        if self._data_stream_uid == value:
            return

        self._data_stream_uid = value
        self._data_stream = self._streams_by_uid.get(self._data_stream_uid)
        if self._state_initialized and self.file_path_valid and self._data_stream:
            self._load_data_stream_from_cache()
            self._update_timeline_data()

    @property
    @property_params(widget=None, dont_encode=True)
    def data_stream(self) -> DataXDFStream | None:
        return self._data_stream

    @property
    @property_params(widget=None)
    def available_marker_streams(self) -> list[MarkerXDFStream]:
        return self._available_marker_streams

    @available_marker_streams.setter
    def available_marker_streams(self, value: list[MarkerXDFStream]) -> None:
        self._available_marker_streams = value
        self._rebuild_stream_uid_mapping()
        self.streams_changed.emit()

    @property
    @property_params(widget=None, dont_encode=True)
    def available_marker_stream_options(self) -> list[tuple[str, MarkerXDFStream]]:
        return [(s.display_name, s.uid) for s in self.available_marker_streams]

    @property
    @property_params(
        label="Marker Stream",
        widget=DynamicComboWidget,
        options_source="available_marker_stream_options",
        options_changed_signal="streams_changed",
    )
    def marker_stream_uid(self) -> str:
        return self._marker_stream_uid

    @marker_stream_uid.setter
    def marker_stream_uid(self, value: str | None) -> None:
        if self._marker_stream_uid == value:
            return

        self._marker_stream_uid = value
        self._marker_stream = self._streams_by_uid.get(self._marker_stream_uid)
        if self._state_initialized and self.file_path_valid and self._marker_stream:
            self._load_marker_stream_from_cache()
            self._update_events()
            self.align_with_recording()

    @property
    @property_params(widget=None, dont_encode=True)
    def marker_stream(self) -> MarkerXDFStream | None:
        return self._marker_stream

    @property
    @property_params(widget=None)
    def available_sync_events(self) -> list[str]:
        return self._available_sync_events

    @available_sync_events.setter
    def available_sync_events(self, value: list[str]) -> None:
        self._available_sync_events = value
        self.sync_events_changed.emit()

    @property
    @property_params(
        label="Sync Event",
        widget=DynamicComboWidget,
        options_source="available_sync_events",
        options_changed_signal="sync_events_changed",
    )
    def sync_event(self) -> str:
        return self._sync_event

    @sync_event.setter
    def sync_event(self, value: str | None) -> None:
        clean_value = str(value or "").strip()
        if self._sync_event == clean_value:
            return

        self._sync_event = clean_value
        self._is_aligned = False
        if self._state_initialized and self.marker_stream:
            self.align_with_recording()

    @property
    @property_params(label="Apply Bandpass 1-30 Hz")
    def apply_bandpass(self) -> bool:
        return self._apply_bandpass

    @apply_bandpass.setter
    def apply_bandpass(self, value: bool) -> None:
        if self._apply_bandpass == value:
            return

        self._apply_bandpass = value
        if self._state_initialized and self.data_stream:
            self._update_timeline_data()

    @property
    @property_params(label="Channel Selection")
    def channels(self) -> dict[str, bool]:
        return self._channels

    @channels.setter
    def channels(self, value: dict[str, bool]) -> None:
        self._channels = value
        if self._state_initialized:
            self.update_timeline()

    def on_recording_loaded(self, recording: NeonRecording) -> None:
        # State is fully initialized by now, allow loading XDF data from now on
        self._state_initialized = True

        # Keep persisted file_path from settings. If it is still valid, reload it
        # automatically so the XDF opens together with the recording.
        if self.file_path_valid:
            self.load_xdf()
            return

        self.file_path = None
        self._reset_loaded_xdf_state()
        # self._available_sync_events = []
        # self.streams_changed.emit()
        # self.sync_events_changed.emit()
        # self._stream_data = None
        # self._stream_ts = None
        # self._xdf_markers = []
        # self._channel_names = []
        # self._channels = {}
        # self._is_aligned = False
        self._clear_timeline_tracks()

    def on_disabled(self) -> None:
        self._clear_timeline_tracks()

    def _set_available_sync_events(self, event_names: list[str]) -> None:
        unique_names = sorted({name for name in event_names if name})
        selected = self._sync_event if self._sync_event in unique_names else ""
        if not selected and unique_names:
            selected = unique_names[0]

        changed = (unique_names != self._available_sync_events) or (selected != self._sync_event)
        self._available_sync_events = unique_names
        self._sync_event = selected
        self.sync_events_changed.emit()

        if changed:
            self.changed.emit()

    def _get_xdf_cache_file(self) -> Path:
        return self.get_cache_path() / "xdf_file_cache.json"

    def _get_marker_stream_cache_file(self, stream_uid: str) -> Path:
        return self.get_cache_path() / f"xdf_marker_stream_{stream_uid}.json"

    def _get_data_stream_cache_file(self, stream_uid: str) -> Path:
        return self.get_cache_path() / f"xdf_data_stream_{stream_uid}.npy"

    def _reset_loaded_xdf_state(self) -> None:
        self._available_data_streams = []
        self._available_marker_streams = []
        self.data_stream_uid = None
        self.marker_stream_uid = None
        self._channels = {}

    def _update_channel_selection(self) -> None:
        if not self.data_stream.loaded:
            self.channels = {}
            return

        previous_channels = dict(self._channels)
        new_channels: dict[str, bool] = {}
        for idx, ch_name in enumerate(self.data_stream.channel_names):
            new_channels[ch_name] = previous_channels.get(ch_name, idx == 0)
        self.channels = new_channels

    def load_xdf(self) -> None:
        # Prevent attempts to load XDF while the plugin state is not fully
        # initialized - i.e., file path is set but stream names are not
        if not self._state_initialized:
            return

        # Fast path: if this xdf stream is already cached, load it instantly.
        if self._attempt_load_xdf_from_cache(log_missing=False):
            return

        logging.info("Could not load XDF data from cache, re-building the cache")
        self._reset_loaded_xdf_state()

        # In headless mode, either load the cached data or proceed with the
        # requested background job directly
        if self.headless:
            return

        self._xdf_load_job = self.job_manager.run_background_action(
            "Loading XDF streams",
            "XDFMultimodalPlugin._bg_load_xdf",
            self._xdf_path,
        )
        self._xdf_load_job.finished.connect(self._on_xdf_load_finished)

    def _on_xdf_load_finished(self) -> None:
        self._xdf_load_job = None
        self._attempt_load_xdf_from_cache()

    def _bg_load_xdf(self, xdf_path: str) -> Iterator[ProgressUpdate]:
        try:
            xdf_file = Path(xdf_path)
            logging.info("Loading XDF in background: %s", xdf_file)
            yield ProgressUpdate(0.1)

            logging.getLogger("pyxdf").setLevel(logging.INFO)
            streams, _ = pyxdf.load_xdf(str(xdf_file))
            xdf_streams = []
            for s in streams:
                xdf_stream = XDFStream.from_xdf_dict(s)
                try:
                    xdf_stream.raise_not_valid()
                    xdf_streams.append(xdf_stream)
                except XDFStreamError as e:
                    logging.warning(
                        f"Skipping XDF stream {xdf_stream.name} ({xdf_stream.xdf_id}) "
                        f"due to invalid data. Reason: {str(e)}"
                    )

            n_streams = len(xdf_streams)
            data_streams = []
            marker_streams = []
            for idx, stream in enumerate(xdf_streams):
                if stream.is_marker_stream:
                    self._prepare_marker_stream_cache(stream)
                    marker_streams.append(stream.to_dict())
                else:
                    self._prepare_data_stream_cache(stream)
                    data_streams.append(stream.to_dict())

                yield ProgressUpdate(0.2 + (0.7 * (idx + 1) / n_streams))

            meta_payload = {
                "cache_version": self._XDF_CACHE_VERSION,
                "source_path": str(xdf_file.resolve()),
                "data_streams": data_streams,
                "marker_streams": marker_streams,
            }

            meta_cache_file = self._get_xdf_cache_file()
            meta_cache_file.parent.mkdir(parents=True, exist_ok=True)
            with meta_cache_file.open("w", encoding="utf-8") as meta_fp:
                json.dump(meta_payload, meta_fp)
        except Exception:
            logging.exception("Failed to load XDF in background")
            yield ProgressUpdate(1.0)
            return

        logging.info(f"Created cache files for {n_streams} XDF streams")
        yield ProgressUpdate(1.0)

    def _prepare_marker_stream_cache(self, stream: XDFStream) -> None:
        info_cache_file = self._get_marker_stream_cache_file(stream.uid)
        with info_cache_file.open("w", encoding="utf-8") as stream_meta_fp:
            json.dump(stream.markers, stream_meta_fp)

    def _prepare_data_stream_cache(self, stream: XDFStream) -> None:
        data_cache_file = self._get_data_stream_cache_file(stream.uid)
        data_cache_file.parent.mkdir(parents=True, exist_ok=True)

        # One NPY per data stream. First column is timestamps.
        stream_matrix = np.column_stack((stream.timestamps, stream.data))
        np.save(str(data_cache_file), stream_matrix.astype(np.float32))

    def _attempt_load_xdf_from_cache(self, *, log_missing: bool = True) -> bool:
        meta_cache_file = self._get_xdf_cache_file()
        if not meta_cache_file.exists():
            logging.debug("XDF cache metadata file not found: %s", meta_cache_file)
            return False

        try:
            if not self._load_xdf_from_cache(log_missing=log_missing):
                return False
        except Exception as e:
            logging.exception(f"Failed to load XDF from cache. Error: {str(e)}")
            return False

        self._update_events()
        self.align_with_recording()
        return True

    def _clear_cache(self) -> None:
        shutil.rmtree(self.get_cache_path())

    def _load_data_stream_from_cache(self) -> None:
        data_cache_file = self._get_data_stream_cache_file(self._data_stream.uid)
        try:
            self.data_stream.load_data(data_cache_file)
            self._update_channel_selection()
        except XDFStreamError as e:
            logging.error(
                f"Failed to load cached data for the XDF stream "
                f"{self.data_stream.display_name}"
            )
            self.data_stream_uid = None

    def _load_marker_stream_from_cache(self) -> None:
        marker_cache_file = self._get_marker_stream_cache_file(self._marker_stream.uid)
        try:
            self._marker_stream.load_markers(marker_cache_file)
        except XDFStreamError as e:
            logging.error(
                f"Failed to load cached markers for the XDF stream "
                f"{self._marker_stream.display_name}"
            )
            self._marker_stream = None
            return False

    def _load_xdf_from_cache(self, *, log_missing: bool = True) -> bool:
        meta_cache_file = self._get_xdf_cache_file()
        with meta_cache_file.open("r", encoding="utf-8") as meta_fp:
            meta_payload = json.load(meta_fp)

        source_path = meta_payload.get("source_path", "")
        if meta_payload.get("cache_version") != self._XDF_CACHE_VERSION:
            logging.debug(
                "Cached data needs to be re-built due to an outdated format"
            )
            self._clear_cache()
            return False

        if source_path != str(self._xdf_path.resolve()):
            logging.debug(
                "Cached data corresponds to a different XDF file, so the "
                "cache has to be re-built"
            )
            self._clear_cache()
            return False

        cached_data_streams = {}
        for cached_stream in meta_payload.get("data_streams", []):
            xdf_stream = XDFStream.from_dict(cached_stream)
            cached_data_streams[xdf_stream.uid] = xdf_stream
        if self.data_stream and self.data_stream.uid not in cached_data_streams:
            self.data_stream_uid = None
        self.available_data_streams = list(cached_data_streams.values())

        cached_marker_streams = {}
        for cached_stream in meta_payload.get("marker_streams", []):
            xdf_stream = XDFStream.from_dict(cached_stream)
            cached_marker_streams[xdf_stream.uid] = xdf_stream
        if self.marker_stream and self.marker_stream.uid not in cached_marker_streams:
            self.marker_stream_uid = None
        self.available_marker_streams = list(cached_marker_streams.values())

        if self.data_stream_uid:
            self._load_data_stream_from_cache()
            if not self.data_stream:
                return False

        if self.marker_stream_uid:
            self._load_marker_stream_from_cache()
            if not self.marker_stream:
                return False

        return True

    def _update_events(self) -> None:
        if not self.marker_stream or not self.marker_stream.markers:
            self.available_sync_events = []

        ep = Plugin.get_instance_by_name("EventsPlugin")
        if ep:
            self._neon_events = ep.events
        else:
            _, self._neon_events = _load_events_from_recording(self.recording)

        neon_event_names = set(self._neon_events.keys())
        xdf_marker_names = set(self.marker_stream.markers.keys())
        common_names = neon_event_names & xdf_marker_names
        self.available_sync_events = common_names

    def _reset_aligned_state(self) -> None:
        self._is_aligned = False
        self.update_timeline()

    def align_with_recording(self) -> None:
        if not self.recording:
            return

        if not self._sync_event:
            logging.warning("No common sync event selected. Skipping alignment.")
            self._reset_aligned_state()
            return

        neon_timestamps = self._neon_events[self._sync_event]
        xdf_timestamps = self.marker_stream.markers[self._sync_event]

        if len(neon_timestamps) > 1:
            logging.warning(
                f"Found multiple occurrences of the '{self._sync_event}' event "
                f"in Neon Player. Using the first one for alignment."
            )
        neon_timestamp = neon_timestamps[0]

        if len(xdf_timestamps) > 1:
            logging.warning(
                f"Found multiple occurrences of the '{self._sync_event}' marker "
                f"in XDF data. Using the first one for alignment."
            )
        xdf_timestamp = int(xdf_timestamps[0] * 1e9)

        self._offset_ns = neon_timestamp - xdf_timestamp
        logging.info(
            f"Aligned Neon and XDF data using the `{self._sync_event}` event.\n"
            f"\tXDF timestamp:  {xdf_timestamp} ns\n"
            f"\tNeon timestamp: {neon_timestamp} ns\n"
            f"\tOffset:         {self._offset_ns} ns."
        )
        self._is_aligned = True
        self.update_timeline()

    def update_timeline(self):
        if self.headless or not self.recording:
            return

        self._clear_timeline_tracks()
        self._update_timeline_markers()
        self._update_timeline_data()

    def _clear_timeline_tracks(self) -> None:
        if self.headless:
            return

        timeline = self.get_timeline()
        was_sorting_enabled = timeline.disable_plot_sorting()

        for row_name in self._active_timeline_row_names:
            timeline.remove_timeline_plot(row_name)

        self._active_timeline_row_names.clear()

        if was_sorting_enabled:
            timeline.enable_plot_sorting()

    def _get_data_stream_group_title(self) -> str:
        stream_type = self.data_stream.type.strip()
        return f"XDF - {stream_type}" if stream_type else "XDF - Data Stream"

    def _update_timeline_markers(self) -> None:
        if not self.marker_stream or not self.marker_stream.markers:
            return

        timeline = self.get_timeline()
        was_sorting_enabled = timeline.disable_plot_sorting()

        # Filter and plot markers that fall within the recording
        for name, timestamps in self.marker_stream.markers.items():
            offset_timestamps = apply_offset(timestamps, self._offset_ns)
            timeline_row_name = f"XDF Markers - {name}"
            plot_item = timeline.get_timeline_plot(timeline_row_name, True)
            if not plot_item.items:
                timeline.add_timeline_scatter(timeline_row_name, [])

            y = np.zeros_like(offset_timestamps)
            plot_item.items[0].setData(offset_timestamps, y)
            self._active_timeline_row_names.add(timeline_row_name)

        if was_sorting_enabled:
            timeline.enable_plot_sorting()

    def _update_timeline_data(self) -> None:
        if not self._data_stream or not self._is_aligned:
            return

        timeline = self.get_timeline()

        # 1. Convert XDF timestamps to Neon clock (nanoseconds)
        offset_ts = apply_offset(self._data_stream.timestamps, self._offset_ns)

        # 2. Filter data to fit within the recording bounds
        mask = (offset_ts >= self.recording.start_time) & (offset_ts <= self.recording.stop_time)

        if not np.any(mask):
            logging.warning("No data found within recording bounds.")
            return

        plot_ts = offset_ts[mask]
        plot_data = self.data_stream.data[mask]

        # Update the status bars on the timeline
        data_stream_row_name = "XDF - Data Stream"
        timeline.add_timeline_broken_bar(
            data_stream_row_name,
            [(plot_ts[0], plot_ts[-1])],
            color="#00FFFF",
        )
        self._active_timeline_row_names.add(data_stream_row_name)

        selected_names = [
            n for n in self.data_stream.channel_names
            if self._channels.get(n, False)
        ]

        if not selected_names:
            return

        indices = [self.data_stream.channel_names.index(n) for n in selected_names]

        # Extract and clean data
        data = plot_data[:, indices].astype(np.float32)
        data = np.nan_to_num(data, nan=0.0)

        # Optional EEG-style filter for streams where that makes sense.
        if self.apply_bandpass and self.data_stream.fs > 0:
            nyq = 0.5 * self.data_stream.fs
            low, high = 1.0 / nyq, 30.0 / nyq
            # Simple bandpass 1-30Hz
            b, a = butter(4, [max(0.001, low), min(0.999, high)], btype='band')
            data = filtfilt(b, a, data, axis=0)
        elif self.apply_bandpass and self.data_stream.fs <= 0:
            logging.warning(
                f"Bandpass filtering is enabled, but sample rate is invalid "
                f"({self.data_stream.fs:.3f} Hz). Skipping filter."
            )

        # 4. Normalize (Center the data around 0)
        data = data - np.nanmean(data, axis=0)

        # 6. Plot each channel in its own subplot under the XDF group prefix.
        plotted_row_names: set[str] = set()
        for i, name in enumerate(selected_names):
            channel_data = data[:, i]
            plot_data_matrix = np.column_stack((plot_ts, channel_data))
            row_name = f"{self._get_data_stream_group_title()} - {name}"
            plotted_row_names.add(row_name)

            plot_item = timeline.add_timeline_plot(
                timeline_row_name=row_name,
                data=plot_data_matrix,
                plot_name="",
            )
            if plot_item is not None:
                plot_item.preferred_height_2d = 60
                plot_item.adjust_size()
                plot_item.getViewBox().enableAutoRange(y=True)

        self._active_timeline_row_names |= plotted_row_names

    @action
    @action_params(compact=True, icon=QIcon(str(neon_player.asset_path("export.svg"))))
    def export(self, destination: Path = Path()) -> None:
        if not self._data_stream or not self._is_aligned:
            logging.warning("Cannot export: no aligned data available.")
            return

        offset_ts = apply_offset(self.data_stream.timestamps, self._offset_ns)
        start_time, stop_time = self.app.get_export_window()
        mask = (offset_ts >= start_time) & (offset_ts <= stop_time)

        if not np.any(mask):
            logging.warning("No data in the export window.")
            return

        export_ts = offset_ts[mask]
        export_data = self.data_stream.data[mask]

        df = pd.DataFrame({
            "recording id": self.recording.id,
            "timestamp [ns]": export_ts
        })
        for i, name in enumerate(self.data_stream.channel_names):
            df[name] = export_data[:, i].astype(np.float32)

        export_file = destination / "xdf_stream.csv"
        df.to_csv(export_file, index=False)
        logging.info(f"Exported {len(df)} samples of XDF data to {export_file}")

    @action
    @action_params(compact=True, icon=QIcon.fromTheme("help-about"))
    def debug_events(self) -> None:
        if not self.recording:
            return

        logging.info("--- NEON EVENTS ---")
        evs = []
        for event_name, timestamps in self._neon_events.items():
            for ts in timestamps:
                evs.append((event_name, ts))
        for name, ts in evs:
            logging.info(f"Neon: '{name}' @ {ts}")
        logging.info("--- XDF MARKERS ---")
        evs = []
        for marker_name, timestamps in self.marker_stream.markers.items():
            offset_timestamps = apply_offset(timestamps, self._offset_ns)
            for ts, offset_ts in zip(timestamps, offset_timestamps):
                evs.append((marker_name, ts, offset_ts))
        for name, ts, offset_ts in evs:
            logging.info(f"XDF: '{name}' @ {offset_ts} (ts={ts})")

    @action
    @action_params(compact=True, icon=QIcon.fromTheme("edit-select-all"))
    def enable_all_channels(self) -> None:
        if not self.data_stream.loaded:
            return

        self.channels = {name: True for name in self.data_stream.channel_names}

    @action
    @action_params(compact=True, icon=QIcon.fromTheme("edit-clear"))
    def disable_all_channels(self) -> None:
        if not self.data_stream.loaded:
            return

        self.channels = {name: False for name in self.data_stream.channel_names}
