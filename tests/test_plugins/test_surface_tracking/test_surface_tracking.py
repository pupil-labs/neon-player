from pupil_labs.neon_player.plugins.gaze import GazeDataPlugin
from pupil_labs.neon_player.plugins.surface_tracking import SurfaceTrackingPlugin
from pupil_labs.neon_player.plugins.surface_tracking.tracked_surface import TrackedSurface


def test_surface_tracking_plugin__to_dict__recursive(qapp, mock_neon_recording):
    qapp.recording = mock_neon_recording()
    qapp.plugins_by_class["GazeDataPlugin"] = GazeDataPlugin()

    plugin = SurfaceTrackingPlugin()
    # Set the surfaces directly to a private field to skip surface initialization
    plugin._surfaces = [TrackedSurface(), TrackedSurface()]
    plugin_dict = plugin.to_dict(recursive=True)

    for el in plugin_dict["surfaces"]:
        assert isinstance(el, dict), \
            f"Expected surface to be serialized to dict, but got {type(el)}"
        assert not el["edit"], \
            f"Expected surface to be serialized with edit=False, but got {el['edit']}"

        assert isinstance(el["preview_options"], dict), \
            f"Expected surface preview_options to be serialized to dict"
