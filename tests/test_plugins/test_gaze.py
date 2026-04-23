from pupil_labs.neon_player.plugins.gaze import CircleViz


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
