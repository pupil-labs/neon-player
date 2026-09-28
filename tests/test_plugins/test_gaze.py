from pupil_labs.neon_player.plugins.gaze import GazeDataPlugin, CircleViz, CrosshairViz


def test_circle_viz_parameter_capping_not_required():
    viz = CircleViz()
    viz.radius = 30
    viz.stroke_width = 10   # < 2 * radius
    assert viz._applied_radius == viz.radius
    assert viz._applied_stroke_width == viz.stroke_width


def test_circle_viz_parameter_capping_required():
    viz = CircleViz()
    viz.radius = 30
    viz.stroke_width = 80   # > 2 * radius
    assert viz._applied_radius == 35
    assert viz._applied_stroke_width == 70


def test_gaze_plugin__to_dict__viz_have_class_names(qapp, mock_neon_recording):
    qapp.recording = mock_neon_recording()

    plugin = GazeDataPlugin()
    plugin.visualizations = [CircleViz(), CrosshairViz()]
    state = plugin.to_dict(recursive=True)

    assert state["visualizations"][0]["__class__"] == "CircleViz"
    assert state["visualizations"][1]["__class__"] == "CrosshairViz"
