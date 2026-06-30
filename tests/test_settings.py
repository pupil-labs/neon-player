from qt_property_widgets.utilities import property_params

from pupil_labs.neon_player import Plugin
from pupil_labs.neon_player.settings import (
    get_property_scopes, merge_plugin_states
)


class ExamplePlugin(Plugin):
    @property
    @property_params()
    def default_property(self) -> int:
        return 10

    @property
    @property_params(scope="recording")
    def recording_property(self) -> int:
        return 20

    @property
    @property_params(scope="workspace")
    def workspace_property(self) -> int:
        return 30

    @property
    @property_params(scope=["recording", "workspace"])
    def general_property(self) -> int:
        return 40


def prepare_plugin_states():
    saved_states = {
        "ExamplePlugin": {
            "default_property": 0,
            "recording_property": 0,
            "workspace_property": 0,
            "general_property": 0
        }
    }

    current_states = {
        "ExamplePlugin": {
            "default_property": 1,
            "recording_property": 1,
            "workspace_property": 1,
            "general_property": 1
        }
    }

    return saved_states, current_states


def test_settings_get_property_scopes():
    scopes = get_property_scopes(ExamplePlugin)

    assert len(scopes) == 4
    assert scopes["default_property"] == ["workspace"]
    assert scopes["recording_property"] == ["recording"]
    assert scopes["workspace_property"] == ["workspace"]
    assert scopes["general_property"] == ["recording", "workspace"]


def test_settings_merge_plugin_states_no_condition():
    saved_states, current_states = prepare_plugin_states()
    scopes = {
        "ExamplePlugin": get_property_scopes(ExamplePlugin)
    }

    merged_states = merge_plugin_states(
        saved_states,
        current_states,
        scopes,
        overwrite_condition=None
    )

    for prop in saved_states["ExamplePlugin"]:
        assert merged_states["ExamplePlugin"][prop] == 1


def test_settings_merge_plugin_states_overwrite_recording_only():
    saved_states, current_states = prepare_plugin_states()
    scopes = {
        "ExamplePlugin": get_property_scopes(ExamplePlugin)
    }

    merged_states = merge_plugin_states(
        saved_states,
        current_states,
        scopes,
        overwrite_condition=lambda scopes: "recording" in scopes
    )

    plugin_states = merged_states["ExamplePlugin"]
    assert plugin_states["default_property"] == 0
    assert plugin_states["recording_property"] == 1, plugin_states
    assert plugin_states["workspace_property"] == 0
    assert plugin_states["general_property"] == 1


def test_settings_merge_plugin_states_overwrite_workspace_only():
    saved_states, current_states = prepare_plugin_states()
    scopes = {
        "ExamplePlugin": get_property_scopes(ExamplePlugin)
    }

    merged_states = merge_plugin_states(
        saved_states,
        current_states,
        scopes,
        overwrite_condition=lambda scopes: "workspace" in scopes
    )

    plugin_states = merged_states["ExamplePlugin"]
    assert plugin_states["default_property"] == 1
    assert plugin_states["recording_property"] == 0
    assert plugin_states["workspace_property"] == 1
    assert plugin_states["general_property"] == 1
