import json
from unittest.mock import MagicMock

import numpy as np
import pytest
from qt_property_widgets.utilities import ComplexEncoder

from pupil_labs.neon_player.plugins.surface_tracking import SurfaceTrackingPlugin
from pupil_labs.neon_player.plugins.surface_tracking.surface_tracking import (
    LEGACY_DETECTOR_CONFIG,
    TagFamily,
)


@pytest.fixture
def plugin(qapp, mock_neon_recording):
    qapp.recording = mock_neon_recording()
    qapp.headless = True
    plugin = SurfaceTrackingPlugin()
    plugin._load_marker_cache = MagicMock()
    plugin.redetect_markers = MagicMock()
    plugin._start_marker_detection = MagicMock()
    yield plugin
    qapp.headless = True


def _write_cache(plugin, config=None):
    plugin.marker_cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(plugin.marker_cache_file, np.array([[]], dtype=object))
    if config is not None:
        plugin.marker_config_file.write_text(json.dumps(config))


def test_detector_settings_roundtrip(plugin):
    plugin.tag_family = TagFamily.tag25h9
    plugin.min_decision_margin = 50
    state = json.loads(json.dumps(plugin.to_dict(), cls=ComplexEncoder))
    assert state["tag_family"] == "tag25h9"
    state.pop("surfaces")  # surface restore needs the full app

    restored = SurfaceTrackingPlugin.from_dict(state)
    assert restored.tag_family is TagFamily.tag25h9
    assert restored.min_decision_margin == 50.0
    assert restored.detector_config()["tag_family"] == "tag25h9"


def test_restoring_settings_does_not_start_detection(plugin):
    # settings are applied before the recording is ready
    plugin.tag_family = TagFamily.tag16h5
    assert not plugin._redetect_timer.isActive()


def test_setting_change_schedules_redetect(qapp, plugin):
    qapp.headless = False
    plugin._recording_ready = True
    plugin.tag_family = TagFamily.tag25h9
    assert plugin._redetect_timer.isActive()
    plugin._redetect_timer.stop()


def test_legacy_cache_matches_default_settings(qapp, plugin):
    qapp.headless = False
    _write_cache(plugin)  # old cache without config file
    plugin.attempt_marker_cache_load()
    plugin._load_marker_cache.assert_called_once()
    plugin.redetect_markers.assert_not_called()


def test_cache_from_other_settings_is_redetected(qapp, plugin):
    qapp.headless = False
    _write_cache(plugin, LEGACY_DETECTOR_CONFIG)
    plugin._tag_family = TagFamily.tag25h9
    plugin.attempt_marker_cache_load()
    plugin.redetect_markers.assert_called_once()
    plugin._load_marker_cache.assert_not_called()


def test_cache_with_matching_settings_is_loaded(qapp, plugin):
    qapp.headless = False
    plugin._tag_family = TagFamily.tag25h9
    _write_cache(plugin, plugin.detector_config())
    plugin.attempt_marker_cache_load()
    plugin._load_marker_cache.assert_called_once()


def test_missing_cache_starts_detection(qapp, plugin):
    qapp.headless = False
    plugin.attempt_marker_cache_load()
    plugin._start_marker_detection.assert_called_once()


def test_headless_never_redetects(plugin):
    _write_cache(plugin, LEGACY_DETECTOR_CONFIG)
    plugin._tag_family = TagFamily.tag25h9
    plugin.attempt_marker_cache_load()
    plugin.redetect_markers.assert_not_called()
    plugin._start_marker_detection.assert_not_called()
