from qt_property_widgets.utilities import property_params

from pupil_labs.neon_player import Plugin
from pupil_labs.neon_player.settings import PluginSettingsDispatcher


class MyPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__()
        self._var = "var"
        self._prop1 = ""
        self._prop2 = ""

    @property
    @property_params(scope=["recording"])
    def prop1(self) -> str:
        return self._prop1

    @prop1.setter
    def prop1(self, value: str) -> None:
        self._prop1 = value

    @property
    def prop2(self) -> str:
        return self._prop2

    @prop2.setter
    def prop2(self, value: str) -> None:
        self._prop2 = value


def test__toggle_plugin__enable_from_scratch(qapp):
    qapp.plugins_by_class = {}

    state = {"prop1": "new_value1", "prop2": "new_value2"}
    qapp.toggle_plugin(MyPlugin, True, state)

    plugin = qapp.plugins_by_class[MyPlugin.__name__]
    assert plugin.prop1 == "new_value1"
    assert plugin.prop2 == "new_value2"


def test__toggle_plugin__enable_from_existing(qapp):
    plugin = MyPlugin()
    plugin.prop1 = "old_value1"
    plugin.prop2 = "old_value2"

    class MockSettingsDispatcher:
        def __init__(self):
            self.property_scopes = {
                MyPlugin.__name__: {
                    "prop1": ["recording"],
                    "prop2": ["workspace"]
                }
            }

    qapp.plugins_by_class = {MyPlugin.__name__: plugin}
    qapp.session_settings = MockSettingsDispatcher()

    state = {"prop1": "new_value1", "prop2": "new_value2"}
    qapp.toggle_plugin(MyPlugin, True, state)

    plugin = qapp.plugins_by_class[MyPlugin.__name__]
    assert plugin.prop1 == "new_value1"
    assert plugin.prop2 == "old_value2"
