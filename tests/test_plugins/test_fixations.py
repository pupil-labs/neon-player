from pupil_labs.neon_player.plugins.fixations import (
    FixationsPlugin, ScanpathViz, FixationCircleViz
)


def test_fixations_plugin__to_dict__viz_have_class_names(qapp, mock_neon_recording):
    qapp.recording = mock_neon_recording()

    plugin = FixationsPlugin()
    plugin.visualizations = [ScanpathViz(), FixationCircleViz()]
    state = plugin.to_dict(recursive=True)

    assert state["visualizations"][0]["__class__"] == "ScanpathViz"
    assert state["visualizations"][1]["__class__"] == "FixationCircleViz"
