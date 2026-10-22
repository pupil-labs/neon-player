import fractions
import numpy as np

from collections import namedtuple
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from scipy.io import wavfile
from scipy.signal import resample_poly

from pupil_labs.neon_player.plugins.audio import (
    AudioPlugin,
    _prepare_audio_frame,
    _resample_audio,
)
from tests.mocks import mock_av_timeseries


MockAudioStream = namedtuple("MockAudioStream", ["layout", "rate", "time_base"])
MockAudioLayout = namedtuple("MockAudioLayout", ["name"])


def mock_audio_plugin():
    plugin = AudioPlugin()
    plugin.player = MagicMock()
    plugin.on_speed_changed = MagicMock()
    plugin.on_playback_state_changed = MagicMock()
    return plugin


def test_prepare_audio_frame_correct_data():
    data = np.random.default_rng().integers(-32768, 32767, size=(1, 1024), dtype=np.int16)
    mock_layout = MockAudioLayout(name="mono")
    mock_stream = MockAudioStream(
        layout=mock_layout, rate=44100, time_base=fractions.Fraction(1, 44100)
    )
    frame = _prepare_audio_frame(data, format="s16", time=0.5, stream=mock_stream)

    assert np.allclose(frame.to_ndarray(), data)
    assert frame.pts == 22050
    assert frame.dts == 22050


def test_prepare_audio_frame_uses_stream_params():
    data = np.random.default_rng().integers(-32768, 32767, size=(1, 1024), dtype=np.int16)
    mock_layout = MockAudioLayout(name="mono")
    mock_stream = MockAudioStream(
        layout=mock_layout, rate=44100, time_base=fractions.Fraction(1, 44100)
    )
    frame = _prepare_audio_frame(data, format="s16", time=0.5, stream=mock_stream)

    assert frame.layout.name == mock_stream.layout.name
    assert frame.sample_rate == mock_stream.rate
    assert frame.time_base == mock_stream.time_base


def test_resample_audio_resamples_and_generates_timestamps(tmp_path):
    sample_rate = 800
    first_timestamp = 1_700_000_000_123_456_789
    audio_data = np.arange(800, dtype=np.int16)
    cache_file = tmp_path / "audio.wav"
    wavfile.write(cache_file, sample_rate, audio_data)
    data = _resample_audio(cache_file, first_timestamp)

    expected_samples = resample_poly(audio_data, up=1, down=4)
    expected_timestamps = first_timestamp + np.arange(len(expected_samples)) * 5_000_000
    np.testing.assert_array_equal(data[:, 0], expected_timestamps.astype(data.dtype))
    np.testing.assert_allclose(data[:, 1], expected_samples)


def test_load_audio_creates_timestamped_waveform_cache_when_missing(
    qapp, mock_neon_recording
):
    first_timestamp = 1_700_000_000_123_456_789
    expected_data = np.array(
        [[first_timestamp, 0.25], [first_timestamp + 5_000_000, -0.25]],
        dtype=np.float64,
    )
    recording = mock_neon_recording(audio=mock_av_timeseries([first_timestamp]))
    qapp.recording = recording
    plugin = mock_audio_plugin()
    plugin.cache_file.parent.mkdir(parents=True, exist_ok=True)
    plugin.cache_file.touch()
    with patch(
        "pupil_labs.neon_player.plugins.audio._resample_audio",
        return_value=expected_data
    ):
        plugin.load_audio()

    assert plugin.first_timestamp == first_timestamp
    assert plugin.recording_has_audio
    np.testing.assert_array_equal(
        np.load(plugin.resampled_audio_cache_file), expected_data
    )


def test_load_audio_sets_first_timestamp_from_cached_waveform(
    qapp, mock_neon_recording
):
    plugin = mock_audio_plugin()
    plugin.cache_file.parent.mkdir(parents=True, exist_ok=True)
    plugin.cache_file.touch()

    waveform_timestamp = 987_654_321
    waveform = np.array(
        [[waveform_timestamp, 0.25], [waveform_timestamp + 5_000_000, -0.25]],
        dtype=np.float64,
    )
    np.save(plugin.resampled_audio_cache_file, waveform)

    plugin.load_audio()
    assert plugin.first_timestamp == waveform_timestamp
