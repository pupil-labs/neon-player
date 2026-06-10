import json
import logging
import typing as T

from pathlib import Path
from PySide6.QtCore import QObject, Signal
from qt_property_widgets.utilities import (
    PersistentPropertiesMixin, property_params, ComplexEncoder, get_properties
)

from pupil_labs import neon_player
from pupil_labs.neon_player import GlobalPluginProperties, Plugin
from pupil_labs.neon_recording import NeonRecording


def plugin_label_lookup(cls_name: str) -> str:
    try:
        cls = Plugin.get_class_by_name(cls_name)
        if cls and hasattr(cls, "label"):
            return cls.label
        else:
            return cls_name
    except ValueError:
        pass

    return f"{cls_name} (missing?)"


def get_property_scopes(plugin_cls: type) -> dict[str, list[str]]:
    properties = get_properties(plugin_cls)
    scopes = {}

    for prop_name, prop in properties.items():
        if not prop.fget:
            continue

        params = getattr(prop.fget, "parameters", {})
        encode_ok = not params.get("dont_encode", False)
        if not encode_ok:
            continue

        scope = params.get("scope", "workspace")
        if isinstance(scope, str):
            scope = [scope]

        scopes[prop_name] = scope

    return scopes


def merge_plugin_states(
    saved_states: dict[str, dict],
    new_states: dict[str, dict],
    scopes: dict[str, dict] | None = None,
    overwrite_condition: T.Callable[[list[str]], bool] | None = None,
) -> dict[str, dict]:
    """
    Merge plugin states, optionally overwriting only properties that match
    a condition based on their scope.

    :param saved_states: Current plugin states, will be used by default.
    :param new_states: New plugin states, will overwrite saved states if the condition is met.
    :param scopes: Optional mapping of plugin properties to their scopes, used
    to filter which properties to merge based on the provided condition.
    :param overwrite: Optional function that takes property scopes and returns
    a boolean indicating whether to overwrite the property. If None, all properties
    will be merged.
    :return: Merged plugin states.
    """
    condition_present = overwrite_condition is not None

    merged = saved_states.copy()
    for plugin_name, props in new_states.items():
        if plugin_name not in merged:
            merged[plugin_name] = {}

        plugin_scopes = scopes.get(plugin_name, {}) if scopes else {}
        for prop_name, prop_value in props.items():
            prop_scopes = plugin_scopes.get(prop_name, [])

            if condition_present and not overwrite_condition(prop_scopes):
                continue

            merged[plugin_name][prop_name] = prop_value

    return merged


class GeneralSettings(PersistentPropertiesMixin, QObject):
    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._skip_gray_frames_on_load = True
        self._show_fps = False

        plugin_names = [k.__name__ for k in Plugin.known_classes]
        plugin_names.sort()
        self._default_plugins = dict.fromkeys(plugin_names, False)
        self._default_plugins.update({
            "GazeDataPlugin": True,
            "AudioPlugin": True,
            "SceneRendererPlugin": True,
            "EventsPlugin": True,
            "ExportAllPlugin": True,
        })

    @property
    def skip_gray_frames_on_load(self) -> bool:
        return self._skip_gray_frames_on_load

    @skip_gray_frames_on_load.setter
    def skip_gray_frames_on_load(self, value: bool) -> None:
        self._skip_gray_frames_on_load = value

    @property
    def show_fps(self) -> bool:
        return self._show_fps

    @show_fps.setter
    def show_fps(self, value: bool) -> None:
        self._show_fps = value

    @property
    def default_plugins(self) -> dict[str, bool]:
        for cls in Plugin.known_classes:
            if cls.__name__ not in self._default_plugins:
                self._default_plugins[cls.__name__] = False

        return self._default_plugins

    @default_plugins.setter
    def default_plugins(self, value: dict[str, bool]) -> None:
        self._default_plugins = value.copy()

    @property
    @property_params(widget=None)
    def plugin_globals(self) -> dict[str, GlobalPluginProperties]:
        value = {}
        for cls in Plugin.known_classes:
            if cls.global_properties is not None:
                value[cls.__name__] = cls.global_properties

        return value

    @plugin_globals.setter
    def plugin_globals(self, value: dict[str, GlobalPluginProperties]) -> None:
        for k, v in value.items():
            for cls in Plugin.known_classes:
                if k == cls.__name__:
                    cls.global_properties = v


class SessionSettings(PersistentPropertiesMixin):
    def __init__(self) -> None:
        super().__init__()
        self._enabled_plugins = neon_player.instance().settings.default_plugins.copy()
        self._plugin_states: dict[str, dict] = {}
        self._export_window: tuple[int, int] = ()
        self.property_scopes: dict[str, dict] = {}

    @property
    @property_params(widget=None, scope="recording")
    def export_window(self) -> tuple[int, int]:
        return self._export_window

    @export_window.setter
    def export_window(self, value: tuple[int, int]) -> None:
        self._export_window = value

    @property
    @property_params(label_lookup=plugin_label_lookup)
    def enabled_plugins(self) -> dict[str, bool]:
        for cls in Plugin.known_classes:
            if cls.__name__ not in self._enabled_plugins:
                self._enabled_plugins[cls.__name__] = False

        return self._enabled_plugins

    @enabled_plugins.setter
    def enabled_plugins(self, value: dict[str, bool]) -> None:
        self._enabled_plugins = value.copy()

    def _update_plugin_states(self) -> None:
        app = neon_player.instance()

        attached_to_recording = app.session_settings.recording_settings == self
        attached_to_workspace = app.session_settings.workspace_settings == self
        if not attached_to_recording and not attached_to_workspace:
            return

        current_states = {
            class_name: p.to_dict()
            for class_name, p in app.plugins_by_class.items()
        }

        overwrite_condition = None
        if app.batch_mode_enabled:
            target_scope = (
                "workspace" if attached_to_workspace else "recording"
            )
            overwrite_condition = lambda scopes: target_scope in scopes

        # Overwrite only properties that are relevant for the current
        # scope (recording or workspace) when batch mode is enabled
        plugin_states = merge_plugin_states(
            self._plugin_states,
            current_states,
            scopes=self.property_scopes,
            overwrite_condition=overwrite_condition
        )

        self._plugin_states = {k: v for k, v in plugin_states.items() if v}

    @property
    @property_params(widget=None, scope=["recording", "workspace"])
    def plugin_states(self) -> dict[str, dict]:
        self._update_plugin_states()

        return self._plugin_states

    @plugin_states.setter
    def plugin_states(self, value: dict[str, dict]) -> None:
        self._plugin_states = value.copy()

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        for kls in Plugin.known_classes:
            if kls.__name__ not in state["enabled_plugins"]:
                self._enabled_plugins[kls.__name__] = False


class PluginSettingsDispatcher(QObject):
    changed = Signal()
    export_window_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.recording_settings = SessionSettings()
        self.workspace_settings = SessionSettings()
        self._batch_mode_enabled: bool = False

        self.property_scopes: dict[str, dict] = {}
        for plugin_cls in Plugin.known_classes:
            scopes = get_property_scopes(plugin_cls)
            self.property_scopes[plugin_cls.__name__] = scopes

    def load_recording_settings(
        self, settings_path: Path, recording: NeonRecording
    ) -> None:
        try:
            if settings_path.exists():
                logging.info(f"Loading recording settings from {settings_path}")
                self.recording_settings = SessionSettings.from_dict(
                    json.loads(settings_path.read_text())
                )

                if len(self.recording_settings.export_window) != 2:
                    logging.warning("Invalid export window in settings")
                    self.recording_settings.export_window = [
                        recording.start_time,
                        recording.stop_time,
                    ]

            else:
                logging.info(f"Recording settings file not found, using defaults")
                self.recording_settings = SessionSettings()
                self.recording_settings.export_window = [
                    recording.start_time,
                    recording.stop_time,
                ]

        except Exception:
            logging.exception("Failed to load recording settings")
            self.recording_settings = SessionSettings()
            self.recording_settings.export_window = [
                recording.start_time,
                recording.stop_time,
            ]

        self.recording_settings.property_scopes = self.property_scopes
        logging.info("Recording settings loaded")

    def load_workspace_settings(self, settings_path: Path) -> None:
        try:
            if settings_path.exists():
                logging.info(f"Loading workspace settings from {settings_path}")
                self.workspace_settings = SessionSettings.from_dict(
                    json.loads(settings_path.read_text())
                )
            else:
                logging.info(f"Workspace settings file not found, using defaults")
                self.workspace_settings = SessionSettings()

        except Exception:
            logging.exception("Failed to load workspace settings")
            self.workspace_settings = SessionSettings()

        self.workspace_settings.property_scopes = self.property_scopes
        logging.info("Workspace settings loaded")

    @staticmethod
    def _save_settings_data(data: dict, settings_path: Path) -> None:
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        with settings_path.open("w") as f:
            json.dump(data, f, cls=ComplexEncoder)

    def save_recording_settings(self, settings_path: Path) -> None:
        data = self.recording_settings.to_dict()
        self._save_settings_data(data, settings_path)

    def save_workspace_settings(self, settings_path: Path) -> None:
        data = self.workspace_settings.to_dict()
        self._save_settings_data(data, settings_path)

    @property
    def batch_mode_enabled(self) -> bool:
        return self._batch_mode_enabled

    @batch_mode_enabled.setter
    def batch_mode_enabled(self, batch_mode_enabled: bool) -> None:
        self._batch_mode_enabled = batch_mode_enabled
        if not self._batch_mode_enabled:
            self.workspace_settings = SessionSettings()

    @property
    def default_source(self) -> SessionSettings:
        if not self.batch_mode_enabled:
            return self.recording_settings

        return self.workspace_settings

    @property
    def export_window(self) -> tuple[int, int]:
        return self.recording_settings.export_window

    @export_window.setter
    def export_window(self, value: tuple[int, int]) -> None:
        self.recording_settings.export_window = value
        self.export_window_changed.emit()
        self.changed.emit()

    @property
    @property_params(label_lookup=plugin_label_lookup)
    def enabled_plugins(self) -> dict[str, bool]:
        return self.default_source.enabled_plugins

    @enabled_plugins.setter
    def enabled_plugins(self, value: dict[str, bool]) -> None:
        self.default_source.enabled_plugins = value

    @property
    def plugin_states(self) -> dict[str, dict]:
        if not self.batch_mode_enabled:
            return self.recording_settings.plugin_states

        workspace_states = self.workspace_settings.plugin_states
        recording_states = self.recording_settings.plugin_states

        # Overwrite values of recording-specific properties
        combined_states = merge_plugin_states(
            workspace_states,
            recording_states,
            scopes=self.property_scopes,
            overwrite_scope="recording"
        )

        return combined_states
