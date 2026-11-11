from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from pupil_labs.neon_recording import NeonRecording
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QGraphicsProxyWidget, QLineEdit
from pytestqt.qtbot import QtBot

from pupil_labs.neon_player.plugins.events import EventsPlugin, EventType
from pupil_labs.neon_player.plugins.xdf import (
    DataXDFStream,
    MarkerXDFStream,
    XDFPlugin,
    XDFStream,
    apply_offset,
    estimate_offset_ns,
    fallback_channel_names,
    first,
    parse_channel_names,
    parse_markers,
)
from pupil_labs.neon_player.ui.timeline_dock import TimeLineDock
from pupil_labs.neon_player.ui.timeline_dock_components import (
    TimelineRowKey,
    TimelineRowRole,
)
from tests.conftest import MockNeonPlayerApp
from tests.mocks import mock_event_timeseries

# Original values are wrapped in the list in the output of pyxdf.load_xdf
MOCK_XDF_DATA_STREAM_INFO = {
    "stream_id": 1,
    "uid": ["uid"],
    "name": ["Mock XDF stream"],
    "type": ["Gaze"],
    "nominal_srate": ["250.0"],
    "channel_count": ["10"],
    "channel_format": ["float32"],
}
MOCK_XDF_MARKER_STREAM_INFO = {
    "stream_id": 1,
    "uid": ["uid"],
    "name": ["Mock XDF stream"],
    "type": ["Markers"],
    "nominal_srate": ["0.0"],
    "channel_count": ["1"],
    "channel_format": ["string"],
}
MOCK_XDF_MARKERS = {
    "time_stamps": np.array([1.0, 2.0, 3.0], dtype=np.float64),
    "time_series": [["first"], ["second"], ["first"]],
}


@pytest.mark.parametrize(
    ("stream_info", "stream_type", "fs", "channel_count", "channel_format"),
    [
        (MOCK_XDF_DATA_STREAM_INFO, "Gaze", 250.0, 10, "float32"),
        (MOCK_XDF_MARKER_STREAM_INFO, "Markers", 0.0, 1, "string"),
    ],
)
def test_xdf_stream_parse_info(
    stream_info: dict[str, int | list[str]],
    stream_type: str,
    fs: float,
    channel_count: int,
    channel_format: str,
) -> None:
    xdf_stream = XDFStream()
    xdf_stream.parse_info(stream_info)
    assert xdf_stream.xdf_id == 1
    assert xdf_stream.uid == "uid"
    assert xdf_stream.name == "Mock XDF stream"
    assert xdf_stream.type == stream_type
    assert xdf_stream.fs == fs
    assert xdf_stream.channel_count == channel_count
    assert xdf_stream.channel_format == channel_format


def test_xdf_stream_from_xdf_dict_parses_markers() -> None:
    xdf_dict = {"info": MOCK_XDF_MARKER_STREAM_INFO, **MOCK_XDF_MARKERS}
    xdf_stream = XDFStream.from_xdf_dict(xdf_dict)
    assert isinstance(xdf_stream, MarkerXDFStream)
    assert len(xdf_stream.markers) == 2


def test_xdf_stream_from_xdf_dict_parse_data_out_of_order_ts() -> None:
    xdf_dict = {
        "info": MOCK_XDF_DATA_STREAM_INFO,
        "time_stamps": np.array([1.0, 3.0, 2.0]),
        "time_series": np.array([0.0, 0.0, 1.0]),
    }
    xdf_stream = XDFStream.from_xdf_dict(xdf_dict)
    assert isinstance(xdf_stream, DataXDFStream)
    assert np.allclose(xdf_stream.timestamps, [1.0, 2.0, 3.0])
    assert np.allclose(np.squeeze(xdf_stream.data), [0.0, 1.0, 0.0])


def test_xdf_stream_from_xdf_dict_parse_data_fallback_channel_names() -> None:
    xdf_dict = {
        "info": MOCK_XDF_DATA_STREAM_INFO,
        "time_stamps": np.array([1.0, 3.0, 2.0]),
        "time_series": np.array([0.0, 0.0, 1.0]),
    }
    xdf_stream = XDFStream.from_xdf_dict(xdf_dict)
    assert isinstance(xdf_stream, DataXDFStream)
    assert xdf_stream.channel_names == ["Ch1"]


def test_marker_xdf_stream_load_save(tmp_path: Path) -> None:
    xdf_dict = {"info": MOCK_XDF_MARKER_STREAM_INFO, **MOCK_XDF_MARKERS}
    xdf_stream = MarkerXDFStream.from_xdf_dict(xdf_dict)

    cache_path = tmp_path / "xdf_markers.json"
    xdf_stream.save_markers(cache_path)
    xdf_stream.load_markers(cache_path)
    assert len(xdf_stream.markers) == 2


def test_data_xdf_stream_load_save(tmp_path: Path) -> None:
    xdf_dict = {
        "info": MOCK_XDF_DATA_STREAM_INFO,
        "time_stamps": np.array([1.0, 2.0, 3.0]),
        "time_series": np.array([0.0, 1.0, 0.0]),
    }
    xdf_stream = DataXDFStream.from_xdf_dict(xdf_dict)

    cache_path = tmp_path / "xdf_data.npz"
    xdf_stream.save_data(cache_path)
    xdf_stream.load_data(cache_path)
    assert xdf_stream.data.shape == (3, 1)
    assert xdf_stream.timestamps.size == 3


@pytest.mark.parametrize(
    ("info", "key", "default", "expected"),
    [({}, "key", 1, 1), ({"key": 2}, "key", 1, 2), ({"key": [3, 4]}, "key", 1, 3)],
)
def test_first(
    info: dict[str, int | list[int]], key: str, default: int, expected: int
) -> None:
    assert first(info, key, default) == expected


def test_parse_markers() -> None:
    markers = parse_markers(MOCK_XDF_MARKERS)
    assert len(markers) == 2
    assert markers["first"] == [1.0, 3.0]
    assert markers["second"] == [2.0]


def test_parse_channel_names() -> None:
    info_dict = {
        "desc": [
            {
                "channels": [
                    {
                        "channel": [
                            {
                                "label": "A1",
                            },
                            {
                                "label": "A2",
                            },
                            {"label": "A3"},
                        ]
                    }
                ]
            }
        ]
    }
    channel_names = parse_channel_names(info_dict)
    assert channel_names == ["A1", "A2", "A3"]


def test_fallback_channel_names() -> None:
    assert fallback_channel_names(3) == ["Ch1", "Ch2", "Ch3"]


def test_estimate_apply_offset() -> None:
    xdf_timestamp_s = 1.1
    neon_timestamp_ns = int(1.2 * 1e9)

    offset_ns = estimate_offset_ns(xdf_timestamp_s, neon_timestamp_ns)
    assert offset_ns == 100000000

    assert apply_offset(xdf_timestamp_s, offset_ns) == neon_timestamp_ns


def test_xdf_plugin_restores_nonempty_persisted_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_stream = DataXDFStream()
    data_stream.uid = "data-uid"
    marker_stream = MarkerXDFStream()
    marker_stream.uid = "marker-uid"
    plugin = XDFPlugin()
    monkeypatch.setattr(plugin, "load_xdf", lambda: None)
    monkeypatch.setattr(plugin, "update_timeline", lambda: None)
    monkeypatch.setattr(plugin, "_update_events", lambda: None)
    plugin.__setstate__(
        {
            "file_path": str(tmp_path / "recording.xdf"),
            "available_data_streams": [data_stream.to_dict()],
            "available_marker_streams": [marker_stream.to_dict()],
            "data_stream_uid": "data-uid",
            "marker_stream_uid": "marker-uid",
        }
    )
    assert plugin.file_path == tmp_path / "recording.xdf"
    assert plugin.data_stream_uid == "data-uid"
    assert plugin.marker_stream_uid == "marker-uid"
    restored_state = plugin.to_dict()
    assert restored_state["data_stream_uid"] == "data-uid"
    assert restored_state["marker_stream_uid"] == "marker-uid"


@pytest.fixture
def xdf_gui(
    qtbot: QtBot,
    qapp: MockNeonPlayerApp,
    monkeypatch: pytest.MonkeyPatch,
    mock_neon_recording: Callable[..., NeonRecording],
) -> Iterator[tuple[XDFPlugin, TimeLineDock]]:
    monkeypatch.setattr(qapp, "recording", None)
    monkeypatch.setattr(qapp, "args", SimpleNamespace(job=None))
    monkeypatch.setattr(qapp, "playback_speed_options", [-1.0, 1.0], raising=False)
    timeline = TimeLineDock()
    qtbot.addWidget(timeline)
    monkeypatch.setattr(
        qapp,
        "recording",
        mock_neon_recording(
            start_time=0,
            stop_time=5_000_000_000,
            events=mock_event_timeseries(
                {"recording.begin": [0], "recording.end": [5_000_000_000]}
            ),
        ),
    )
    window = SimpleNamespace(
        timeline=timeline,
        register_action=Mock(),
        unregister_action=Mock(),
        sort_action_menu=Mock(),
        remove_menu_if_empty=Mock(),
    )
    monkeypatch.setattr(qapp, "main_window", window, raising=False)
    plugin = XDFPlugin()
    yield plugin, timeline
    plugin.on_disabled()
    plugin.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def marker_source(uid: str, markers: dict[str, list[float]]) -> MarkerXDFStream:
    stream = MarkerXDFStream()
    stream.uid = uid
    stream.markers = markers
    return stream


def search_field(timeline: TimeLineDock, group_name: str = "XDF Markers") -> QLineEdit:
    fields = [
        field
        for item in timeline.graphics_view.scene().items()
        if isinstance(item, QGraphicsProxyWidget) and item.widget() is not None
        for field in item.widget().findChildren(QLineEdit)
        if field.accessibleName() == f"Search {group_name}"
    ]
    assert len(fields) == 1
    return fields[0]


def test_xdf_markers_contribute_searchable_scoped_rows(
    xdf_gui: tuple[XDFPlugin, TimeLineDock], qtbot: QtBot
) -> None:
    plugin, timeline = xdf_gui
    names = ["Search", "", "XDF Markers", "Trial Begin", "trial end"]
    stream = marker_source("markers", {name: [1.0, 2.0] for name in names})
    plugin.available_marker_streams = [stream]
    global_plot = timeline.get_timeline_plot("Search", True)
    plugin.marker_stream_uid = stream.uid
    group = timeline.get_timeline_group("XDF Markers")
    assert group is not None
    assert group.searchable
    assert group.child_row_names == set(names)
    field = search_field(timeline)
    assert field.placeholderText() == "Search"
    qtbot.keyClicks(field, "tRiAl")
    plots = {name: timeline.get_timeline_plot(name, group=group) for name in names}
    assert [name for name, plot in plots.items() if plot.isVisible()] == [
        "Trial Begin",
        "trial end",
    ]
    assert global_plot.isVisible()
    for name, plot in plots.items():
        assert plot.row_key == TimelineRowKey(
            "XDF Markers", name, TimelineRowRole.CUSTOM
        )
        assert list(plot.items[0].getOriginalDataset()[0]) == [
            1_000_000_000,
            2_000_000_000,
        ]
        assert timeline.get_timeline_plot(f"XDF Markers - {name}") is None
    field.setText("missing")
    assert not any(plot.isVisible() for plot in plots.values())
    search_plot = timeline.timeline_plots[
        TimelineRowKey("XDF Markers", "Search", TimelineRowRole.SEARCH)
    ]
    assert search_plot.isVisible()
    assert search_plot.items == []
    timeline.timeline_group_controls[group.name].click()
    field.clear()
    assert not search_plot.isVisible()
    assert not any(plot.isVisible() for plot in plots.values())
    timeline.timeline_group_controls[group.name].click()
    assert all(plot.isVisible() for plot in plots.values())


def test_xdf_alignment_and_marker_refresh_preserve_query_and_series(
    xdf_gui: tuple[XDFPlugin, TimeLineDock],
) -> None:
    plugin, timeline = xdf_gui
    stream = marker_source(
        "markers", {"recording.begin": [1.0], "Trial": [1.2, 1.5], "Other": [2.0]}
    )
    plugin.available_marker_streams = [stream]
    plugin.marker_stream_uid = stream.uid
    group = timeline.get_timeline_group("XDF Markers")
    field = search_field(timeline)
    field.setText("trial")
    plot = timeline.get_timeline_plot("Trial", group=group)
    series = plot.items[0]
    plugin.sync_event = "recording.begin"
    plugin.align_with_recording()
    assert timeline.get_timeline_group("XDF Markers") is group
    assert search_field(timeline) is field
    assert field.text() == "trial"
    assert timeline.get_timeline_plot("Trial", group=group) is plot
    assert plot.items[0] is series
    assert list(series.getOriginalDataset()[0]) == [200_000_000, 500_000_000]
    assert stream.markers["Trial"] == [1.2, 1.5]
    assert plot.isVisible()
    assert not timeline.get_timeline_plot("Other", group=group).isVisible()
    timeline.timeline_group_controls[group.name].click()
    stream.markers.pop("Other")
    stream.markers["Later Trial"] = [2.5]
    stream.markers["No match"] = [3.0]
    plugin.update_timeline(data=False)
    assert timeline.get_timeline_group("XDF Markers") is group
    assert group.collapsed
    assert field.text() == "trial"
    assert timeline.get_timeline_plot("Other", group=group) is None
    assert not timeline.get_timeline_plot("Later Trial", group=group).isVisible()
    timeline.timeline_group_controls[group.name].click()
    assert timeline.get_timeline_plot("Later Trial", group=group).isVisible()
    assert not timeline.get_timeline_plot("No match", group=group).isVisible()
    later = timeline.get_timeline_plot("Later Trial", group=group)
    assert list(later.items[0].getOriginalDataset()[0]) == [1_500_000_000]


def test_xdf_marker_source_replacement_resets_search_before_deferred_deletion(
    xdf_gui: tuple[XDFPlugin, TimeLineDock],
) -> None:
    plugin, timeline = xdf_gui
    first_stream = marker_source("first", {"Trial": [1.0]})
    second_stream = marker_source("second", {"Trial": [2.0]})
    plugin.available_marker_streams = [first_stream, second_stream]
    plugin.marker_stream_uid = first_stream.uid
    old_group = timeline.get_timeline_group("XDF Markers")
    old_field = search_field(timeline)
    old_field.setText("missing")
    timeline.timeline_group_controls[old_group.name].click()
    plugin.marker_stream_uid = first_stream.uid
    assert search_field(timeline) is old_field
    plugin.marker_stream_uid = second_stream.uid
    new_group = timeline.get_timeline_group("XDF Markers")
    assert new_group is not old_group
    field = search_field(timeline)
    assert field.text() == ""
    assert not new_group.collapsed
    old_field.setText("still missing")
    plot = timeline.get_timeline_plot("Trial", group=new_group)
    assert plot.isVisible()
    assert list(plot.items[0].getOriginalDataset()[0]) == [2_000_000_000]
    destroyed = Mock()
    old_field.destroyed.connect(destroyed)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    destroyed.assert_called_once()
    assert search_field(timeline) is field
    plugin.marker_stream_uid = None
    assert timeline.get_timeline_group("XDF Markers") is None
    plugin.marker_stream_uid = first_stream.uid
    assert search_field(timeline).text() == ""


def test_headless_xdf_markers_never_require_timeline_widgets(
    qapp: MockNeonPlayerApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(qapp, "recording", None)
    monkeypatch.delattr(qapp, "main_window", raising=False)
    before = set(qapp.allWidgets())
    plugin = XDFPlugin()
    assert plugin.headless
    stream = marker_source("markers", {"Trial": [1.0]})
    plugin.available_marker_streams = [stream]
    plugin.marker_stream_uid = stream.uid
    plugin.update_timeline()
    plugin.clear_timeline()
    plugin.on_disabled()
    assert set(qapp.allWidgets()) == before
    assert plugin._marker_timeline_group is None
    plugin.deleteLater()


def test_events_and_xdf_search_independently_with_identical_names(
    xdf_gui: tuple[XDFPlugin, TimeLineDock],
    qapp: MockNeonPlayerApp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin, timeline = xdf_gui
    events = EventsPlugin()
    monkeypatch.setattr(qapp, "plugins_by_class", {"EventsPlugin": events})
    monkeypatch.setattr(events, "save_cached_json", Mock())
    for name in ("Trial", "Search"):
        events.add_event_type(EventType.from_name(name))
        events.add_events({name: [1_000_000_000]})
    stream = marker_source("markers", {"Trial": [1.0], "Search": [2.0]})
    plugin.available_marker_streams = [stream]
    plugin.marker_stream_uid = stream.uid
    global_plot = timeline.get_timeline_plot("Search", True)
    events_field = search_field(timeline, "Events")
    xdf_field = search_field(timeline)
    events_field.setText("Trial")
    xdf_field.setText("SEARCH")
    assert timeline.get_timeline_plot("Trial", group="Events").isVisible()
    assert not timeline.get_timeline_plot("Search", group="Events").isVisible()
    assert timeline.get_timeline_plot("Search", group="XDF Markers").isVisible()
    assert not timeline.get_timeline_plot("Trial", group="XDF Markers").isVisible()
    assert global_plot.isVisible()
    plugin.clear_timeline()
    assert timeline.get_timeline_group("XDF Markers") is None
    assert events_field.text() == "Trial"
    assert timeline.get_timeline_plot("Trial", group="Events").isVisible()
    events.on_disabled()
    events.deleteLater()


def test_xdf_data_channels_stay_global_during_marker_search_and_refresh(
    xdf_gui: tuple[XDFPlugin, TimeLineDock],
) -> None:
    plugin, timeline = xdf_gui
    markers = marker_source("markers", {"recording.begin": [1.0], "Trial": [2.0]})
    data = DataXDFStream.from_xdf_dict(
        {
            "info": {**MOCK_XDF_DATA_STREAM_INFO, "channel_count": ["2"]},
            "time_stamps": np.array([1.0, 2.0]),
            "time_series": np.array([[1.0, 10.0], [2.0, 20.0]]),
        }
    )
    plugin.available_data_streams = [data]
    plugin.available_marker_streams = [markers]
    plugin.data_stream_uid = data.uid
    plugin.marker_stream_uid = markers.uid
    plugin.sync_event = "recording.begin"
    plugin.align_with_recording()
    field = search_field(timeline)
    field.setText("no match")
    channel = timeline.get_timeline_plot("XDF - Gaze - Ch1")
    status = timeline.get_timeline_plot("XDF - Data Stream")
    assert channel is not None
    assert channel.row_key.group_name is None
    assert channel.isVisible()
    assert status is not None
    assert status.row_key.group_name is None
    assert status.isVisible()
    assert list(channel.items[0].getOriginalDataset()[0]) == [0, 1_000_000_000]
    assert list(channel.items[0].getOriginalDataset()[1]) == [1.0, 2.0]
    plugin.channels = {"Ch1": False, "Ch2": True}
    plugin.update_timeline(markers=False)
    assert timeline.get_timeline_plot("XDF - Gaze - Ch1") is None
    second = timeline.get_timeline_plot("XDF - Gaze - Ch2")
    assert second.isVisible()
    assert second.row_key.group_name is None
    assert list(second.items[0].getOriginalDataset()[1]) == [10.0, 20.0]
    assert search_field(timeline) is field
    assert field.text() == "no match"
    plugin.clear_timeline()
    assert timeline.get_timeline_plot("XDF - Gaze - Ch2") is None
    assert timeline.get_timeline_plot("XDF - Data Stream") is None


def test_xdf_disable_and_reload_without_recording_start_fresh(
    xdf_gui: tuple[XDFPlugin, TimeLineDock],
    qapp: MockNeonPlayerApp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin, timeline = xdf_gui
    stream = marker_source("markers", {"Trial": [1.0]})
    plugin.available_marker_streams = [stream]
    plugin.marker_stream_uid = stream.uid
    group = timeline.get_timeline_group("XDF Markers")
    old_field = search_field(timeline)
    old_field.setText("missing")
    recording = qapp.recording
    monkeypatch.setattr(qapp, "recording", None)
    plugin.on_disabled()
    timeline.on_recording_unloaded()
    assert timeline.get_timeline_group("XDF Markers") is None
    assert plugin._marker_timeline_group is None
    fresh = XDFPlugin()
    fresh.available_marker_streams = [stream]
    fresh.marker_stream_uid = stream.uid
    monkeypatch.setattr(qapp, "recording", recording)
    fresh.update_timeline()
    new_group = timeline.get_timeline_group("XDF Markers")
    assert new_group is not group
    assert not new_group.collapsed
    old_field.setText("another stale query")
    assert search_field(timeline).text() == ""
    assert timeline.get_timeline_plot("Trial", group=new_group).isVisible()
    fresh.on_disabled()
    fresh.deleteLater()


def test_xdf_file_replacement_discards_marker_group(
    xdf_gui: tuple[XDFPlugin, TimeLineDock],
    tmp_path: Path,
) -> None:
    plugin, timeline = xdf_gui
    stream = marker_source("markers", {"Trial": [1.0]})
    plugin.available_marker_streams = [stream]
    plugin.marker_stream_uid = stream.uid
    old_group = timeline.get_timeline_group("XDF Markers")
    search_field(timeline).setText("missing")
    plugin.file_path = tmp_path / "replacement.xdf"
    assert timeline.get_timeline_group("XDF Markers") is None
    plugin.available_marker_streams = [stream]
    plugin.marker_stream_uid = stream.uid
    assert timeline.get_timeline_group("XDF Markers") is not old_group
    assert search_field(timeline).text() == ""


def test_xdf_reuses_searchable_group_and_keeps_empty_source_search_available(
    xdf_gui: tuple[XDFPlugin, TimeLineDock],
) -> None:
    plugin, timeline = xdf_gui
    group = timeline.add_timeline_group("XDF Markers", searchable=True)
    field = search_field(timeline)
    field.setText("trial")
    stream = marker_source("markers", {"Trial": [1.0]})
    plugin.available_marker_streams = [stream]
    plugin.marker_stream_uid = stream.uid
    assert timeline.get_timeline_group("XDF Markers") is group
    assert search_field(timeline) is field
    stream.markers.clear()
    plugin.update_timeline(data=False)
    assert not group.child_row_names
    assert timeline.get_timeline_group("XDF Markers") is group
    assert search_field(timeline) is field
    assert field.text() == "trial"
    assert timeline.timeline_plots[
        TimelineRowKey("XDF Markers", "Search", TimelineRowRole.SEARCH)
    ].isVisible()


def test_xdf_disable_does_not_remove_an_external_replacement_group(
    xdf_gui: tuple[XDFPlugin, TimeLineDock],
) -> None:
    plugin, timeline = xdf_gui
    stream = marker_source("markers", {"Trial": [1.0]})
    plugin.available_marker_streams = [stream]
    plugin.marker_stream_uid = stream.uid
    timeline.remove_timeline_group("XDF Markers")
    replacement = timeline.add_timeline_group("XDF Markers", searchable=True)
    plot = timeline.get_timeline_plot("Keep", True, group=replacement)
    field = search_field(timeline)
    field.setText("Keep")
    plugin.on_disabled()
    assert timeline.get_timeline_group("XDF Markers") is replacement
    assert timeline.get_timeline_plot("Keep", group=replacement) is plot
    assert field.text() == "Keep"
    assert plugin._marker_timeline_group is None


def test_xdf_cached_marker_source_switch_loads_and_aligns_fresh_rows(
    xdf_gui: tuple[XDFPlugin, TimeLineDock],
    qapp: MockNeonPlayerApp,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin, timeline = xdf_gui
    plugin.on_recording_loaded(qapp.recording)
    xdf_path = tmp_path / "recording.xdf"
    plugin.file_path = xdf_path
    xdf_path.touch()
    monkeypatch.setattr(plugin, "require_cache_path", lambda: tmp_path)
    streams = [
        marker_source("first", {"recording.begin": [1.0], "Trial": [1.5]}),
        marker_source("second", {"recording.begin": [2.0], "Trial": [3.0]}),
    ]
    for stream in streams:
        stream.save_markers(tmp_path / f"xdf_marker_stream_{stream.uid}.json")
    plugin.available_marker_streams = streams
    plugin.marker_stream_uid = streams[0].uid
    plugin.sync_event = "recording.begin"
    old_group = timeline.get_timeline_group("XDF Markers")
    search_field(timeline).setText("missing")
    plugin.marker_stream_uid = streams[1].uid
    group = timeline.get_timeline_group("XDF Markers")
    assert group is not old_group
    assert search_field(timeline).text() == ""
    plot = timeline.get_timeline_plot("Trial", group=group)
    assert plot.isVisible()
    assert list(plot.items[0].getOriginalDataset()[0]) == [1_000_000_000]
