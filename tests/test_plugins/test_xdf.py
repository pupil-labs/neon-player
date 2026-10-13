import numpy as np
import pytest

from pupil_labs.neon_player.plugins.xdf import (
    DataXDFStream,
    MarkerXDFStream,
    XDFStream,
    apply_offset,
    estimate_offset_ns,
    fallback_channel_names,
    first,
    parse_markers,
    parse_channel_names
)


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
    "time_series": [["first"], ["second"], ["first"]]
}


@pytest.mark.parametrize(
    "stream_info, stream_type, fs, channel_count, channel_format", [
        (MOCK_XDF_DATA_STREAM_INFO, "Gaze", 250.0, 10, "float32"),
        (MOCK_XDF_MARKER_STREAM_INFO, "Markers", 0.0, 1, "string")
    ]
)
def test_xdf_stream_parse_info(stream_info, stream_type, fs, channel_count, channel_format):
    xdf_stream = XDFStream()
    xdf_stream.parse_info(stream_info)
    assert xdf_stream.xdf_id == 1
    assert xdf_stream.uid == "uid"
    assert xdf_stream.name == "Mock XDF stream"
    assert xdf_stream.type == stream_type
    assert xdf_stream.fs == fs
    assert xdf_stream.channel_count == channel_count
    assert xdf_stream.channel_format == channel_format


def test_xdf_stream_from_xdf_dict_parses_markers():
    xdf_dict = {
        "info": MOCK_XDF_MARKER_STREAM_INFO,
        **MOCK_XDF_MARKERS
    }
    xdf_stream = XDFStream.from_xdf_dict(xdf_dict)
    assert isinstance(xdf_stream, MarkerXDFStream)
    assert len(xdf_stream.markers) == 2


def test_xdf_stream_from_xdf_dict_parse_data_out_of_order_ts():
    xdf_dict = {
        "info": MOCK_XDF_DATA_STREAM_INFO,
        "time_stamps": np.array([1.0, 3.0, 2.0]),
        "time_series": np.array([0.0, 0.0, 1.0]),
    }
    xdf_stream = XDFStream.from_xdf_dict(xdf_dict)
    assert isinstance(xdf_stream, DataXDFStream)
    assert np.allclose(xdf_stream.timestamps, [1.0, 2.0, 3.0])
    assert np.allclose(np.squeeze(xdf_stream.data), [0.0, 1.0, 0.0])


def test_xdf_stream_from_xdf_dict_parse_data_fallback_channel_names():
    xdf_dict = {
        "info": MOCK_XDF_DATA_STREAM_INFO,
        "time_stamps": np.array([1.0, 3.0, 2.0]),
        "time_series": np.array([0.0, 0.0, 1.0]),
    }
    xdf_stream = XDFStream.from_xdf_dict(xdf_dict)
    assert isinstance(xdf_stream, DataXDFStream)
    assert xdf_stream.channel_names == ["Ch1"]


def test_marker_xdf_stream_load_save(tmp_path):
    xdf_dict = {
        "info": MOCK_XDF_MARKER_STREAM_INFO,
        **MOCK_XDF_MARKERS
    }
    xdf_stream = MarkerXDFStream.from_xdf_dict(xdf_dict)

    cache_path = tmp_path / "xdf_markers.json"
    xdf_stream.save_markers(cache_path)
    xdf_stream.load_markers(cache_path)
    assert len(xdf_stream.markers) == 2


def test_data_xdf_stream_load_save(tmp_path):
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
    "info, key, default, expected", [
        ({}, "key", 1, 1),
        ({"key": 2}, "key", 1, 2),
        ({"key": [3, 4]}, "key", 1, 3)
    ]
)
def test_first(info, key, default, expected):
    assert first(info, key, default) == expected


def test_parse_markers():
    markers = parse_markers(MOCK_XDF_MARKERS)
    assert len(markers) == 2
    assert markers["first"] == [1.0, 3.0]
    assert markers["second"] == [2.0]


def test_parse_channel_names():
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
                            {
                                "label": "A3"
                            }
                        ]
                    }
                ]
            }
        ]
    }
    channel_names = parse_channel_names(info_dict)
    assert channel_names == ["A1", "A2", "A3"]


def test_fallback_channel_names():
    assert fallback_channel_names(3) == ["Ch1", "Ch2", "Ch3"]


def test_estimate_apply_offset():
    xdf_timestamp_s = 1.1
    neon_timestamp_ns = int(1.2 * 1e9)

    offset_ns = estimate_offset_ns(xdf_timestamp_s, neon_timestamp_ns)
    assert offset_ns == 100000000

    assert apply_offset(xdf_timestamp_s, offset_ns) == neon_timestamp_ns
