import fractions
import numpy as np

from collections import namedtuple
from pupil_labs.neon_player.plugins.audio import _prepare_audio_frame


MockAudioStream = namedtuple("MockAudioStream", ["layout", "rate", "time_base"])
MockAudioLayout = namedtuple("MockAudioLayout", ["name"])


def test_prepare_audio_frame__correct_data():
    data = np.zeros((1, 1024), dtype=np.int16)
    mock_layout = MockAudioLayout(name="mono")
    mock_stream = MockAudioStream(
        layout=mock_layout, rate=44100, time_base=fractions.Fraction(1, 44100)
    )
    frame = _prepare_audio_frame(data, format="s16", time=0.5, stream=mock_stream)

    assert np.allclose(frame.to_ndarray(), data)
    assert frame.pts == 22050
    assert frame.dts == 22050


def test_prepare_audio_frame__uses_stream_params():
    data = np.zeros((1, 1024), dtype=np.int16)
    mock_layout = MockAudioLayout(name="mono")
    mock_stream = MockAudioStream(
        layout=mock_layout, rate=44100, time_base=fractions.Fraction(1, 44100)
    )
    frame = _prepare_audio_frame(data, format="s16", time=0.5, stream=mock_stream)

    assert frame.layout.name == mock_stream.layout.name
    assert frame.sample_rate == mock_stream.rate
    assert frame.time_base == mock_stream.time_base
