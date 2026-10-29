import json
import logging
import shutil
import uuid
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyxdf
from pupil_labs.neon_recording import NeonRecording
from PySide6.QtCore import Signal
from PySide6.QtGui import QIcon
from qt_property_widgets.utilities import (
    FilePath,
    PersistentPropertiesMixin,
    action_params,
    property_params,
)
from qt_property_widgets.widgets import DynamicComboWidget
from scipy.signal import butter, filtfilt

from pupil_labs import neon_player
from pupil_labs.neon_player import Plugin, action, asset_path
from pupil_labs.neon_player.job_manager import BackgroundJob, ProgressUpdate
from pupil_labs.neon_player.plugins.events import (
    EventsPlugin,
    _load_events_from_recording,
)

logger = logging.getLogger(__name__)


def first(xdf_dict: Mapping[str, Any], key: str, default: Any) -> Any:  # noqa: ANN401
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


def parse_event_name(text: str) -> str:
    """Parse event names from strings.

    For now, just strip whitespace. Future improvements could parse JSON
    or other structured formats.
    """
    return text.strip()


def parse_markers(xdf_dict: dict[str, Any]) -> dict[str, list[float]]:
    """Parse and group markers from an XDF stream."""
    if "time_stamps" not in xdf_dict or "time_series" not in xdf_dict:
        return {}

    markers: dict[str, list[float]] = {}
    for ts, marker in zip(
        xdf_dict["time_stamps"], xdf_dict["time_series"], strict=False
    ):
        marker_to_parse = marker[0] if isinstance(marker, list) else str(marker)
        name = parse_event_name(marker_to_parse)
        if name not in markers:
            markers[name] = []

        timestamp = float(ts)
        markers[name].append(timestamp)
    return markers


def get_fs_from_timestamps(timestamps: npt.NDArray[np.float64]) -> float:
    if len(timestamps) < 2:
        return 0.0

    dt = np.diff(timestamps)
    return 1.0 / float(np.median(dt)) if len(dt) > 0 else 0.0


type XDFNumericArray = npt.NDArray[
    np.float32 | np.float64 | np.int8 | np.int16 | np.int32 | np.int64
]


def parse_stream_data(
    xdf_dict: dict[str, Any],
) -> tuple[XDFNumericArray | None, npt.NDArray[np.float64] | None]:
    """Parse XDF stream data."""
    if "time_series" not in xdf_dict or "time_stamps" not in xdf_dict:
        return None, None

    # NOTE: pyxdf returns a data array that has its dtype set according
    # to the declared channel format, timestamps should be float64
    data = xdf_dict["time_series"]
    timestamps = xdf_dict["time_stamps"]
    if not isinstance(data, np.ndarray) or not isinstance(timestamps, np.ndarray):
        raise XDFStreamError("XDF samples and timestamps must be NumPy arrays")
    if data.dtype not in (
        np.dtype(np.float32),
        np.dtype(np.float64),
        np.dtype(np.int8),
        np.dtype(np.int16),
        np.dtype(np.int32),
        np.dtype(np.int64),
    ) or timestamps.dtype != np.dtype(np.float64):
        raise XDFStreamError("XDF samples or timestamps have an unsupported dtype")

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    return data, timestamps


def parse_channel_names(info: dict[str, Any]) -> list[str]:
    """Parses channel names for data streams.

    Expected location: info > desc > channels > channel (one entry per channel).
    """
    desc = first(info, "desc", {})
    channels = first(desc, "channels", {})
    ch_list = channels.get("channel", [])
    if not ch_list:
        return []

    channel_names = []
    for i, ch in enumerate(ch_list):
        channel_names.append(str(first(ch, "label", f"Ch{i + 1}")))
    return channel_names


def fallback_channel_names(channel_count: int) -> list[str]:
    return [f"Ch{i + 1}" for i in range(max(0, channel_count))]


def estimate_offset_ns(xdf_timestamp: float, neon_timestamp: int) -> int:
    """Estimate the offset that aligns XDF timestamps with the Neon ones."""
    return neon_timestamp - int(xdf_timestamp * 1e9)


def apply_offset(
    xdf_timestamps: npt.NDArray[np.float64], offset_ns: int
) -> npt.NDArray[np.int64]:
    """Apply offset to align XDF timestamps with the Neon ones."""
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

    # Keep the persisted XDF property name compatible with existing caches.
    @type.setter  # noqa: A003
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
        condition: Callable[[dict[str, Any]], bool] | None = None,
        recursive: bool = False,
    ) -> dict[str, Any]:
        state = super().to_dict(include_class_name, condition, recursive)

        # NOTE: class name distinguishes marker streams from data streams
        state["__class__"] = self.__class__.__name__
        return state

    def parse_info(self, info: dict[str, Any]) -> None:
        self.xdf_id = int(first(info, "stream_id", -1))
        self.uid = str(first(info, "uid", uuid.uuid4()))
        # NOTE: according to the specification, the 'name' attribute is not mandatory
        # but should be present most of the times for interpretability purposes
        self.name = str(first(info, "name", "<No name>"))
        self.type = str(first(info, "type", "")).strip()
        self.fs = float(first(info, "nominal_srate", 0.0))
        self.channel_count = int(first(info, "channel_count", -1))
        self.channel_format = str(first(info, "channel_format", "")).strip().lower()

    @classmethod
    def from_xdf_dict(cls, xdf_dict: dict[str, Any]) -> "XDFStream":
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
            raise XDFStreamError("Marker cache file does not exist")

        try:
            with open(marker_cache_file) as f:
                self.markers = json.load(f)
            self._loaded = True
        except Exception as exc:
            raise XDFStreamError("Failed to load markers from cache") from exc

    def save_markers(self, marker_cache_file: Path) -> None:
        try:
            with marker_cache_file.open("w", encoding="utf-8") as stream_meta_fp:
                json.dump(self.markers, stream_meta_fp)
        except Exception:
            logger.warning(
                f"Failed to save markers for stream {self.display_name}", exc_info=True
            )

    @classmethod
    def from_xdf_dict(cls, xdf_dict: dict[str, Any]) -> "MarkerXDFStream":
        stream = cls()
        stream.parse_info(xdf_dict.get("info", {}))
        stream.markers = parse_markers(xdf_dict)
        stream._loaded = True
        return stream


class DataXDFStream(XDFStream):
    def __init__(self) -> None:
        super().__init__()
        self._is_marker_stream = False

        # Expected format: samples x channels
        self.data: XDFNumericArray | None = None
        self.timestamps: npt.NDArray[np.float64] | None = None
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
            raise XDFStreamError("Data cache file does not exist")

        with np.load(str(data_cache_file)) as cache:
            self.data = cache["data"]
            self.timestamps = cache["timestamps"]
        self._loaded = True

    def save_data(self, data_cache_file: Path) -> None:
        if self.data is None or self.timestamps is None:
            logger.warning(f"Stream {self.display_name} has no data to be saved")
            return

        np.savez(str(data_cache_file), data=self.data, timestamps=self.timestamps)

    def raise_not_valid(self) -> None:
        super().raise_not_valid()
        if self.data is None:
            raise XDFStreamError("Failed to parse stream data")

    @classmethod
    def from_xdf_dict(cls, xdf_dict: dict[str, Any]) -> "DataXDFStream":
        info = xdf_dict.get("info", {})

        stream = cls()
        stream.parse_info(info)
        stream.data, stream.timestamps = parse_stream_data(xdf_dict)
        stream.channel_names = parse_channel_names(info)

        if stream.data is None or stream.timestamps is None:
            stream._loaded = True
            return stream

        if stream.timestamps is not None and np.any(np.diff(stream.timestamps) < 0):
            logger.warning(
                f"Timestamps of the XDF stream {stream.display_name} are not "
                f"increasingly monotonically, sorting the timestamps"
            )
            order = stream.timestamps.argsort()
            stream.timestamps = stream.timestamps[order]
            stream.data = stream.data[order, :]

        if not np.isfinite(stream.fs) or stream.fs <= 0:
            stream.fs = get_fs_from_timestamps(stream.timestamps)

        if not stream.channel_names:
            stream.channel_names = fallback_channel_names(stream.data.shape[1])
        elif len(set(stream.channel_names)) != len(stream.channel_names):
            logger.warning(
                f"Channel names of the XDF stream {stream.display_name} "
                f"contain duplicates: {stream.channel_names}"
            )

        stream._loaded = True
        return stream


class XDFPlugin(Plugin):
    label = "XDF Stream"
    _XDF_CACHE_VERSION = 4
    streams_changed = Signal()
    sync_events_changed = Signal()

    def __init__(self) -> None:
        super().__init__()

        self._state_initialized = False
        self._xdf_path: Path = Path("")
        self._available_data_streams: list[DataXDFStream] = []
        self._available_marker_streams: list[MarkerXDFStream] = []
        self._data_streams_by_uid: dict[str, DataXDFStream] = {}
        self._data_stream_uid: str | None = None
        self._data_stream: DataXDFStream | None = None
        self._marker_streams_by_uid: dict[str, MarkerXDFStream] = {}
        self._marker_stream_uid: str | None = None
        self._marker_stream: MarkerXDFStream | None = None
        self._available_sync_events: list[str] = []
        self._neon_events: dict[str, list[int]] = {}
        self._sync_event: str = ""
        self._apply_bandpass: bool = False
        self._channels: dict[str, bool] = {}  # channel name -> enabled

        self._offset_ns: int = 0
        self._is_aligned: bool = False
        self._xdf_load_job: BackgroundJob | None = None
        self._active_timeline_row_names: dict[str, set[str]] = {
            "markers": set(),
            "data": set(),
        }

    def _reset_loaded_xdf_state(self) -> None:
        self.data_stream_uid = None
        self.marker_stream_uid = None
        self.available_data_streams = []
        self.available_marker_streams = []
        self._rebuild_stream_uid_mapping()
        self._reset_aligned_state()

    def _reset_aligned_state(self) -> None:
        self._is_aligned = False
        self._offset_ns = 0
        self.update_timeline()

    @property
    @property_params(
        label="File Path (.xdf)",
        dialog_title="Open an XDF file",
        file_filter="XDF files (*.xdf)",
    )
    def file_path(self) -> FilePath:
        # Returning None keeps FilePathWidget visually empty.
        return FilePath(self._xdf_path) if self._xdf_path != Path("") else None  # type: ignore[return-value]  # Optional getters break persisted union conversion.

    @file_path.setter
    def file_path(self, value: FilePath | None) -> None:
        new_path = Path("") if value is None else value
        if new_path == self._xdf_path:
            return

        self._xdf_path = new_path
        self._reset_loaded_xdf_state()
        if self.file_path_valid:
            self.load_xdf()

    @property
    @property_params(widget=None, dont_encode=True)
    def file_path_valid(self) -> bool:
        return self._xdf_path.exists() and self._xdf_path.is_file()

    def _rebuild_stream_uid_mapping(self) -> None:
        self._data_streams_by_uid = {s.uid: s for s in self.available_data_streams}
        self._marker_streams_by_uid = {s.uid: s for s in self.available_marker_streams}

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
        widget=DynamicComboWidget,  # type: ignore[arg-type]  # property_params omits supported widget classes.
        options_source="available_data_stream_options",
        options_changed_signal="streams_changed",
    )
    def data_stream_uid(self) -> str:
        return self._data_stream_uid  # type: ignore[return-value]  # Optional getters break persisted union conversion.

    @data_stream_uid.setter
    def data_stream_uid(self, value: str | None) -> None:
        if self._data_stream_uid == value:
            return

        self._data_stream_uid = value
        self._data_stream = (
            self._data_streams_by_uid.get(self._data_stream_uid)
            if self._data_stream_uid is not None
            else None
        )
        should_load = (
            self._state_initialized and self.file_path_valid and self._data_stream
        )
        if not should_load:
            self._update_channel_selection()
            return

        if not self._load_data_stream_from_cache():
            logger.warning("Could not load XDF data from cache, re-building the cache")
            self._clear_cache()
            self.load_xdf()

        self._update_channel_selection()

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
    def available_marker_stream_options(self) -> list[tuple[str, str]]:
        return [(s.display_name, s.uid) for s in self.available_marker_streams]

    @property
    @property_params(
        label="Marker Stream",
        widget=DynamicComboWidget,  # type: ignore[arg-type]  # property_params omits supported widget classes.
        options_source="available_marker_stream_options",
        options_changed_signal="streams_changed",
    )
    def marker_stream_uid(self) -> str:
        return self._marker_stream_uid  # type: ignore[return-value]  # Optional getters break persisted union conversion.

    @marker_stream_uid.setter
    def marker_stream_uid(self, value: str | None) -> None:
        if self._marker_stream_uid == value:
            return

        self._marker_stream_uid = value
        self._marker_stream = (
            self._marker_streams_by_uid.get(self._marker_stream_uid)
            if self._marker_stream_uid is not None
            else None
        )
        should_load = (
            self._state_initialized and self.file_path_valid and self._marker_stream
        )
        if not should_load:
            self._update_events()
            self._reset_aligned_state()
            return

        if not self._load_marker_stream_from_cache():
            logger.warning(
                "Could not load marker data from cache, re-building the cache"
            )
            self._clear_cache()
            self.load_xdf()

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
        self._available_sync_events = sorted(value)
        self.sync_events_changed.emit()

    @property
    @property_params(
        label="Sync Event",
        widget=DynamicComboWidget,  # type: ignore[arg-type]  # property_params omits supported widget classes.
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
            self.update_timeline(markers=False)

    @property
    @property_params(label="Channel Selection")
    def channels(self) -> dict[str, bool]:
        return self._channels

    @channels.setter
    def channels(self, value: dict[str, bool]) -> None:
        self._channels = value
        if self._state_initialized:
            self.update_timeline(markers=False)

    def on_recording_loaded(self, _recording: NeonRecording) -> None:
        # State is fully initialized by now, allow loading XDF data from now on
        self._state_initialized = True

        # Keep persisted file_path from settings. If it is still valid, reload it
        # automatically so the XDF opens together with the recording.
        if self.file_path_valid:
            self.load_xdf()
            return

        self.file_path = None
        self._reset_loaded_xdf_state()

    def on_disabled(self) -> None:
        self.clear_timeline()

    def _set_available_sync_events(self, event_names: list[str]) -> None:
        unique_names = sorted({name for name in event_names if name})
        selected = self._sync_event if self._sync_event in unique_names else ""
        if not selected and unique_names:
            selected = unique_names[0]

        changed = (unique_names != self._available_sync_events) or (
            selected != self._sync_event
        )
        self._available_sync_events = unique_names
        self._sync_event = selected
        self.sync_events_changed.emit()

        if changed:
            self.changed.emit()

    def _get_xdf_cache_file(self) -> Path:
        return self.require_cache_path() / "xdf_file_cache.json"

    def _get_marker_stream_cache_file(self, stream_uid: str) -> Path:
        return self.require_cache_path() / f"xdf_marker_stream_{stream_uid}.json"

    def _get_data_stream_cache_file(self, stream_uid: str) -> Path:
        return self.require_cache_path() / f"xdf_data_stream_{stream_uid}.npz"

    def _update_channel_selection(self) -> None:
        if not self.data_stream or not self.data_stream.loaded:
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

        # Prevent launching multiple background jobs
        if self._xdf_load_job is not None:
            self._xdf_load_job.finished.disconnect()
            self._xdf_load_job.cancel()

        # Fast path: if this xdf stream is already cached, load it instantly.
        if self._attempt_load_xdf_from_cache():
            return

        logger.info("Could not load XDF data from cache")
        self._reset_loaded_xdf_state()

        # In headless mode, either load the cached data or proceed with the
        # requested background job directly
        if self.headless:
            return

        logger.info("Re-building the cache in the background")
        self._xdf_load_job = self.job_manager.run_background_action(
            "Loading XDF streams",
            "XDFPlugin._bg_load_xdf",
            self._xdf_path,
        )
        assert self._xdf_load_job is not None, "XDF loading must start a background job"
        self._xdf_load_job.finished.connect(self._on_xdf_load_finished)

    def _on_xdf_load_finished(self) -> None:
        self._xdf_load_job = None
        if not self._attempt_load_xdf_from_cache():
            logger.error(
                "Failed to load the XDF data from cache after the "
                "background job completed, clearing the cache"
            )
            self._clear_cache()
            self.file_path = None

    def _bg_load_xdf(self, xdf_path: str) -> Iterator[ProgressUpdate]:
        try:
            xdf_file = Path(xdf_path)
            logger.info(f"Loading XDF in background: {xdf_file}")
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
                    logger.warning(
                        f"Skipping XDF stream {xdf_stream.name} ({xdf_stream.xdf_id}) "
                        f"due to invalid data. Reason: {e!s}"
                    )

            n_streams = len(xdf_streams)
            data_streams = []
            marker_streams = []
            for idx, stream in enumerate(xdf_streams):
                if isinstance(stream, MarkerXDFStream):
                    self._prepare_marker_stream_cache(stream)
                    marker_streams.append(stream.to_dict())
                elif isinstance(stream, DataXDFStream):
                    self._prepare_data_stream_cache(stream)
                    data_streams.append(stream.to_dict())
                else:
                    logger.warning(f"Skipping unexpected stream type: {type(stream)}")

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
            logger.exception("Failed to load XDF in background")
            yield ProgressUpdate(1.0)
            return

        logger.info(f"Created cache files for {n_streams} XDF streams")
        yield ProgressUpdate(1.0)

    def _prepare_marker_stream_cache(self, stream: MarkerXDFStream) -> None:
        marker_cache_file = self._get_marker_stream_cache_file(stream.uid)
        marker_cache_file.parent.mkdir(parents=True, exist_ok=True)
        stream.save_markers(marker_cache_file)

    def _prepare_data_stream_cache(self, stream: DataXDFStream) -> None:
        data_cache_file = self._get_data_stream_cache_file(stream.uid)
        data_cache_file.parent.mkdir(parents=True, exist_ok=True)
        stream.save_data(data_cache_file)

    def _attempt_load_xdf_from_cache(self) -> bool:
        meta_cache_file = self._get_xdf_cache_file()
        if not meta_cache_file.exists():
            logger.debug(f"XDF cache metadata file not found: {meta_cache_file}")
            return False

        try:
            if not self._load_xdf_from_cache():
                return False

            self._update_events()
            self.align_with_recording()
        except Exception:
            logger.exception("Failed to load XDF from cache")
            return False

        return True

    def _clear_cache(self) -> None:
        cache_folder = self.get_cache_path()
        if cache_folder is None or not cache_folder.exists():
            return

        shutil.rmtree(cache_folder)

    def _load_data_stream_from_cache(self) -> bool:
        if not self.data_stream:
            return False

        data_cache_file = self._get_data_stream_cache_file(self.data_stream.uid)
        try:
            self.data_stream.load_data(data_cache_file)
            return True
        except XDFStreamError as e:
            logger.error(
                f"Failed to load cached data for the XDF stream "
                f"{self.data_stream.display_name}. Reason: {e!s}"
            )
            return False

    def _load_marker_stream_from_cache(self) -> bool:
        if not self.marker_stream:
            return False

        marker_cache_file = self._get_marker_stream_cache_file(self.marker_stream.uid)
        try:
            self.marker_stream.load_markers(marker_cache_file)
            return True
        except XDFStreamError as e:
            logger.error(
                f"Failed to load cached markers for the XDF stream "
                f"{self.marker_stream.display_name}. Reason: {e!s}"
            )
            return False

    def _load_xdf_from_cache(self) -> bool:
        meta_cache_file = self._get_xdf_cache_file()
        with meta_cache_file.open("r", encoding="utf-8") as meta_fp:
            meta_payload = json.load(meta_fp)

        if meta_payload.get("cache_version") != self._XDF_CACHE_VERSION:
            logger.debug("Cached data needs to be re-built due to an outdated format")
            self._clear_cache()
            return False

        source_path = meta_payload.get("source_path", "")
        if source_path != str(self._xdf_path.resolve()):
            logger.debug(
                "Cached data corresponds to a different XDF file, so the "
                "cache has to be re-built"
            )
            self._clear_cache()
            return False

        cached_data_streams = {}
        for cached_stream in meta_payload.get("data_streams", []):
            xdf_stream = DataXDFStream.from_dict(cached_stream)
            cached_data_streams[xdf_stream.uid] = xdf_stream
        if self.data_stream_uid and self.data_stream_uid not in cached_data_streams:
            self.data_stream_uid = None
        self.available_data_streams = list(cached_data_streams.values())

        cached_marker_streams = {}
        for cached_stream in meta_payload.get("marker_streams", []):
            xdf_stream = MarkerXDFStream.from_dict(cached_stream)
            cached_marker_streams[xdf_stream.uid] = xdf_stream
        if (
            self.marker_stream_uid
            and self.marker_stream_uid not in cached_marker_streams
        ):
            self.marker_stream_uid = None
        self.available_marker_streams = list(cached_marker_streams.values())

        if self.data_stream_uid and not self._load_data_stream_from_cache():
            return False

        return not (
            self.marker_stream_uid and not self._load_marker_stream_from_cache()
        )

    def _update_events(self) -> None:
        recording = self.recording
        if not self.marker_stream or not self.marker_stream.markers:
            self.available_sync_events = []
            return

        ep = Plugin.get_instance_by_name("EventsPlugin")
        if ep and isinstance(ep, EventsPlugin):
            self._neon_events = ep.events
        elif recording is not None:
            _, self._neon_events = _load_events_from_recording(recording)
        else:
            self._neon_events = {}

        neon_event_names = set(self._neon_events.keys())
        xdf_marker_names = set(self.marker_stream.markers.keys())
        common_names = neon_event_names & xdf_marker_names
        self.available_sync_events = list(common_names)

    def align_with_recording(self) -> None:
        recording = self.recording
        if not recording:
            return

        if not self.marker_stream or not self._sync_event:
            self._reset_aligned_state()
            return

        neon_timestamps = self._neon_events[self._sync_event]
        xdf_timestamps = self.marker_stream.markers[self._sync_event]

        if len(neon_timestamps) > 1:
            logger.warning(
                f"Found multiple occurrences of the '{self._sync_event}' event "
                f"in Neon Player. Using the first one for alignment."
            )
        neon_timestamp = neon_timestamps[0]

        if len(xdf_timestamps) > 1:
            logger.warning(
                f"Found multiple occurrences of the '{self._sync_event}' marker "
                f"in XDF data. Using the first one for alignment."
            )
        xdf_timestamp = xdf_timestamps[0]

        self._offset_ns = estimate_offset_ns(xdf_timestamp, neon_timestamp)
        logger.info(
            f"Aligned Neon and XDF data using the `{self._sync_event}` event.\n"
            f"\tXDF timestamp:  {int(xdf_timestamp * 1e9)} ns\n"
            f"\tNeon timestamp: {neon_timestamp} ns\n"
            f"\tOffset:         {self._offset_ns} ns."
        )
        self._is_aligned = True
        self.update_timeline()

    def update_timeline(self, markers: bool = True, data: bool = True) -> None:
        recording = self.recording
        if self.headless or not recording:
            return

        timeline = self.get_timeline()
        was_sorting_enabled = timeline.disable_plot_sorting()

        if markers:
            self._clear_timeline_tracks(kind="markers")
            self._update_timeline_markers()
        if data:
            self._clear_timeline_tracks(kind="data")
            self._update_timeline_data()

        if was_sorting_enabled:
            timeline.enable_plot_sorting()

    def clear_timeline(self) -> None:
        self._clear_timeline_tracks(kind="markers")
        self._clear_timeline_tracks(kind="data")

    def _clear_timeline_tracks(self, kind: Literal["markers", "data"]) -> None:
        timeline = self.get_timeline()
        for row_name in self._active_timeline_row_names[kind]:
            timeline.remove_timeline_plot(row_name)

        self._active_timeline_row_names[kind].clear()

    def _get_data_stream_group_title(self) -> str:
        stream_type = self.data_stream.type.strip() if self.data_stream else ""
        return f"XDF - {stream_type}" if stream_type else "XDF - Data Stream"

    def _update_timeline_markers(self) -> None:
        if not self.marker_stream or not self.marker_stream.markers:
            return

        timeline = self.get_timeline()

        # Filter and plot markers that fall within the recording
        for name, timestamps in self.marker_stream.markers.items():
            offset_timestamps = apply_offset(np.array(timestamps), self._offset_ns)
            timeline_row_name = f"XDF Markers - {name}"
            plot_item = timeline.get_timeline_plot(timeline_row_name, True)
            if not plot_item:
                logger.warning(f"Failed to create timeline plot {timeline_row_name}")
                continue

            if not plot_item.items:
                timeline.add_timeline_scatter(timeline_row_name, [])

            y = np.zeros_like(offset_timestamps)
            plot_item.items[0].setData(offset_timestamps, y)
            self._active_timeline_row_names["markers"].add(timeline_row_name)

    def _update_timeline_data(self) -> None:
        recording = self.recording
        if not recording:
            return

        if not self.data_stream or not self.data_stream.loaded or not self._is_aligned:
            return

        assert self.data_stream.data is not None
        assert self.data_stream.timestamps is not None
        timeline = self.get_timeline()

        # Convert XDF timestamps to Neon clock (nanoseconds)
        offset_ts = apply_offset(self.data_stream.timestamps, self._offset_ns)

        # Filter data to fit within the recording bounds
        mask = (offset_ts >= recording.start_time) & (offset_ts <= recording.stop_time)

        if not np.any(mask):
            logger.warning("No data found within recording bounds.")
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
        self._active_timeline_row_names["data"].add(data_stream_row_name)

        selected_names = [
            n for n in self.data_stream.channel_names if self.channels.get(n, False)
        ]

        if not selected_names:
            return

        indices = [
            idx
            for idx, n in enumerate(self.data_stream.channel_names)
            if self.channels.get(n, False)
        ]

        # Extract and clean data
        data = plot_data[:, indices]
        data = np.nan_to_num(data, nan=0.0)

        # Optional EEG-style filter for streams where that makes sense.
        if self.apply_bandpass and self.data_stream.fs > 0:
            nyq = 0.5 * self.data_stream.fs
            low, high = 1.0 / nyq, 30.0 / nyq
            # Simple bandpass 1-30Hz
            b, a = butter(4, [max(0.001, low), min(0.999, high)], btype="band")
            data = filtfilt(b, a, data, axis=0)
        elif self.apply_bandpass and self.data_stream.fs <= 0:
            logger.warning(
                f"Bandpass filtering is enabled, but sample rate is invalid "
                f"({self.data_stream.fs:.3f} Hz). Skipping filter."
            )

        # Plot each channel in its own subplot under the XDF group prefix.
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

        self._active_timeline_row_names["data"] |= plotted_row_names

    @action
    @action_params(compact=True, icon=QIcon(str(neon_player.asset_path("export.svg"))))
    def export(self, destination: Path = Path()) -> None:
        recording = self.recording
        if not recording:
            return

        if not self.data_stream or not self.data_stream.loaded or not self._is_aligned:
            logger.warning("Cannot export: no aligned data available.")
            return

        assert self.data_stream.data is not None
        assert self.data_stream.timestamps is not None

        offset_ts = apply_offset(self.data_stream.timestamps, self._offset_ns)
        start_time, stop_time = self.app.get_export_window()
        mask = (offset_ts >= start_time) & (offset_ts <= stop_time)

        if not np.any(mask):
            logger.warning("No data in the export window.")
            return

        export_ts = offset_ts[mask]
        export_data = self.data_stream.data[mask]

        df = pd.DataFrame({"recording id": recording.id, "timestamp [ns]": export_ts})
        for i, name in enumerate(self.data_stream.channel_names):
            df[name] = export_data[:, i].astype(np.float32)

        export_file = destination / "xdf_stream.csv"
        df.to_csv(export_file, index=False)
        logger.info(f"Exported {len(df)} samples of XDF data to {export_file}")

    @action
    @action_params(compact=True, icon=QIcon(str(asset_path("info.svg"))))
    def debug_events(self) -> None:
        recording = self.recording
        if not recording:
            return

        logger.info("--- NEON EVENTS ---")
        neon_events: list[tuple[str, int]] = []
        for event_name, neon_timestamps in self._neon_events.items():
            neon_events.extend((event_name, neon_ts) for neon_ts in neon_timestamps)
        for name, neon_ts in neon_events:
            logger.info(f"Neon: '{name}' @ {neon_ts}")

        logger.info("--- XDF MARKERS ---")
        if not self.marker_stream or not self.marker_stream.markers:
            logger.info("No XDF markers are present")
            return

        xdf_events: list[tuple[str, float, int]] = []
        for marker_name, xdf_timestamps in self.marker_stream.markers.items():
            offset_timestamps = apply_offset(np.array(xdf_timestamps), self._offset_ns)
            for xdf_ts, offset_ts in zip(
                xdf_timestamps, offset_timestamps, strict=False
            ):
                xdf_events.append((marker_name, xdf_ts, offset_ts))
        for name, xdf_ts, offset_ts in xdf_events:
            logger.info(f"XDF: '{name}' @ {offset_ts} (ts={xdf_ts})")

    @action
    @action_params(compact=True, icon=QIcon(str(asset_path("done_all.svg"))))
    def enable_all_channels(self) -> None:
        if not self.data_stream or not self.data_stream.loaded:
            return

        self.channels = dict.fromkeys(self.data_stream.channel_names, True)

    @action
    @action_params(compact=True, icon=QIcon(str(asset_path("cancel.svg"))))
    def disable_all_channels(self) -> None:
        if not self.data_stream or not self.data_stream.loaded:
            return

        self.channels = dict.fromkeys(self.data_stream.channel_names, False)
