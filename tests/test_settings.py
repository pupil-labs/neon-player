import pytest
from qt_property_widgets.utilities import property_params

from pupil_labs.neon_player import Plugin
from pupil_labs.neon_player.settings import (
    get_property_scopes, merge_plugin_states, get_custom_plugin_state
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
    @property_params(scope="custom")
    def custom_property(self) -> int:
        return 40

    def custom_property_for_scope(self, scope: str) -> int:
        if scope == "recording":
            return 50

        return 60


def prepare_plugin_states():
    saved_states = {
        "ExamplePlugin": {
            "default_property": 0,
            "recording_property": 0,
            "workspace_property": 0,
            "custom_property": 0
        }
    }

    current_states = {
        "ExamplePlugin": {
            "default_property": 1,
            "recording_property": 1,
            "workspace_property": 1,
            "custom_property": 1
        }
    }

    return saved_states, current_states


def test_settings_get_property_scopes():
    scopes, custom_scope_properties = get_property_scopes(ExamplePlugin)

    assert len(scopes) == 4
    assert scopes["default_property"] == "workspace"
    assert scopes["recording_property"] == "recording"
    assert scopes["workspace_property"] == "workspace"
    assert scopes["custom_property"] == "custom"
    assert custom_scope_properties == ["custom_property"]


@pytest.mark.parametrize("overwrite_scope, expected_states", [
    (None, {
        "default_property": 1,
        "recording_property": 1,
        "workspace_property": 1,
        "custom_property": 1
    }),
    ("recording", {
        "default_property": 0,
        "recording_property": 1,
        "workspace_property": 0,
        "custom_property": 0
    }),
    ("workspace", {
        "default_property": 1,
        "recording_property": 0,
        "workspace_property": 1,
        "custom_property": 0
    }),
    ("custom", {
        "default_property": 0,
        "recording_property": 0,
        "workspace_property": 0,
        "custom_property": 1
    }),
], ids=[
    "overwrite_none", "overwrite_recording", "overwrite_workspace", "overwrite_custom"
])
def test_settings_merge_plugin_states(overwrite_scope, expected_states):
    saved_states, current_states = prepare_plugin_states()
    example_scopes, _ = get_property_scopes(ExamplePlugin)
    scopes = {
        "ExamplePlugin": example_scopes
    }

    merged_states = merge_plugin_states(
        saved_states,
        current_states,
        scopes,
        overwrite_scope=overwrite_scope
    )

    for prop in saved_states["ExamplePlugin"]:
        assert merged_states["ExamplePlugin"][prop] == expected_states[prop]


@pytest.mark.parametrize("scope, expected_value", [
    ("recording", 50),
    ("workspace", 60),
])
def test_settings_get_custom_plugin_state(scope, expected_value):
    plugin = ExamplePlugin()
    custom_scope_properties = ["custom_property"]

    custom_state = get_custom_plugin_state(plugin, custom_scope_properties, scope)
    assert len(custom_state) == 1
    assert custom_state["custom_property"] == expected_value
