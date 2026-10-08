import pytest

from pupil_labs.neon_player.plugins.xdf import (
    XDFStream,
    apply_offset,
    estimate_offset_ns,
    first
)


# All values are wrapped in the list in the output of pyxdf.load_xdf
MOCK_XDF_DATA_STREAM_INFO = {
    "name": ["Mock XDF stream"],
    "type": ["Gaze"],
    "channel_format": ["float32"],
}
MOCK_XDF_MARKER_STREAM_INFO = {
    "name": ["Mock XDF stream"],
    "type": ["Markers"],
    "channel_format": ["string"],
}


@pytest.mark.parametrize(
    "stream_info, stream_type, is_marker_stream", [
        (MOCK_XDF_DATA_STREAM_INFO, "Gaze", False),
        (MOCK_XDF_MARKER_STREAM_INFO, "Markers", True)
    ]
)
def test_xdf_stream_from_xdf_dict(stream_info, stream_type, is_marker_stream):
    stream = {"info": stream_info}
    parsed_stream = XDFStream.from_xdf_dict(stream)
    assert parsed_stream.name == "Mock XDF stream"
    assert parsed_stream.type == stream_type
    assert parsed_stream.is_marker_stream == is_marker_stream


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
