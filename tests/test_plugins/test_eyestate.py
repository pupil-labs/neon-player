from pupil_labs.neon_player.plugins.eyestate import EyestatePlugin


def test_eyestate_plugin__global_properties__to_dict__includes_class_name():
    plugin = EyestatePlugin()
    plugin_dict = plugin.global_properties.to_dict()
    assert plugin_dict["__class__"] == "EyestatePluginGlobalProps"
