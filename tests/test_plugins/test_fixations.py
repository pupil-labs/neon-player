from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest
from pupil_labs.neon_recording import NeonRecording

from pupil_labs.neon_player import Plugin
from pupil_labs.neon_player.plugins.fixations import (
    FixationCircleViz,
    FixationsPlugin,
    ScanpathViz,
)
from pupil_labs.neon_player.plugins.gaze import GazeDataPlugin

if TYPE_CHECKING:
    from tests.conftest import MockNeonPlayerApp


def test_fixations_plugin__to_dict__viz_have_class_names(
    qapp: "MockNeonPlayerApp",
    mock_neon_recording: Callable[..., NeonRecording],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qapp, "recording", mock_neon_recording())

    plugin = FixationsPlugin()
    plugin.visualizations = [ScanpathViz(), FixationCircleViz()]
    state = plugin.to_dict(recursive=True)

    assert state["visualizations"][0]["__class__"] == "ScanpathViz"
    assert state["visualizations"][1]["__class__"] == "FixationCircleViz"


def test_export_data_without_recording(
    qapp: "MockNeonPlayerApp", monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(qapp, "recording", None)
    plugin = FixationsPlugin()
    with pytest.raises(RuntimeError, match="A loaded recording is required"):
        plugin.get_export_fixations()
    with pytest.raises(RuntimeError, match="A loaded recording is required"):
        plugin.get_export_saccades()


def test_gaze_offset_requires_gaze_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = FixationsPlugin()
    monkeypatch.setattr(Plugin, "get_instance_by_name", lambda _name: None)
    assert plugin.get_gaze_offset() == (0.0, 0.0)
    monkeypatch.setattr(Plugin, "get_instance_by_name", lambda _name: Plugin())
    assert plugin.get_gaze_offset() == (0.0, 0.0)
    gaze = GazeDataPlugin()
    gaze.offset_x = 0.125
    gaze.offset_y = -0.25
    monkeypatch.setattr(Plugin, "get_instance_by_name", lambda _name: gaze)
    assert plugin.get_gaze_offset() == (0.125, -0.25)
