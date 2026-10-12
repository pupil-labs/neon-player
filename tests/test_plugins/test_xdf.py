import pytest

from pupil_labs.neon_player.plugins.xdf import (
    DataXDFStream,
    MarkerXDFStream,
    XDFStream,
    apply_offset,
    estimate_offset_ns,
    first
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


@pytest.mark.parametrize(
    "stream_info, expected_type", [
        (MOCK_XDF_DATA_STREAM_INFO, DataXDFStream),
        (MOCK_XDF_MARKER_STREAM_INFO, MarkerXDFStream)
    ]
)
def test_xdf_stream_from_xdf_dict_reroute(stream_info, expected_type):
    xdf_dict = {"info": stream_info}
    assert isinstance(XDFStream.from_xdf_dict(xdf_dict), expected_type)


@pytest.mark.parametrize(
    "info, key, default, expected", [
        ({}, "key", 1, 1),
        ({"key": 2}, "key", 1, 2),
        ({"key": [3, 4]}, "key", 1, 3)
    ]
)
def test_first(info, key, default, expected):
    assert first(info, key, default) == expected


def test_estimate_apply_offset():
    xdf_timestamp_s = 1.1
    neon_timestamp_ns = int(1.2 * 1e9)

    offset_ns = estimate_offset_ns(xdf_timestamp_s, neon_timestamp_ns)
    assert offset_ns == 100000000

    assert apply_offset(xdf_timestamp_s, offset_ns) == neon_timestamp_ns
