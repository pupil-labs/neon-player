from collections.abc import Callable
from typing import TYPE_CHECKING

from pupil_labs.neon_recording import NeonRecording

from pupil_labs.neon_player.plugins.fixations import (
    FixationCircleViz,
    FixationsPlugin,
    ScanpathViz,
)

if TYPE_CHECKING:
    from tests.conftest import MockNeonPlayerApp


def test_fixations_plugin__to_dict__viz_have_class_names(
    qapp: "MockNeonPlayerApp", mock_neon_recording: Callable[..., NeonRecording]
) -> None:
    qapp.recording = mock_neon_recording()

    plugin = FixationsPlugin()
    plugin.visualizations = [ScanpathViz(), FixationCircleViz()]
    state = plugin.to_dict(recursive=True)

    assert state["visualizations"][0]["__class__"] == "ScanpathViz"
    assert state["visualizations"][1]["__class__"] == "FixationCircleViz"
