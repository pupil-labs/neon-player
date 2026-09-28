import pytest

from pupil_labs.neon_player.plugins.synchronized_xdf import XDFStream, first


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
        (MOCK_XDF_DATA_STREAM_INFO, "gaze", False),
        (MOCK_XDF_MARKER_STREAM_INFO, "markers", True)
    ]
)
def test_xdf_stream_from_dict(stream_info, stream_type, is_marker_stream):
    stream = {"info": stream_info}
    parsed_stream = XDFStream.from_dict(stream)
    assert parsed_stream.name == "Mock XDF stream"
    assert parsed_stream.type == stream_type
    assert parsed_stream.is_marker_stream == is_marker_stream
    assert parsed_stream.is_data_stream == (not is_marker_stream)


@pytest.mark.parametrize(
    "info, key, default, expected", [
        ({}, "key", 1, 1),
        ({"key": 2}, "key", 1, 2),
        ({"key": [3, 4]}, "key", 1, 3)
    ]
)
def test_first(info, key, default, expected):
    assert first(info, key, default) == expected
