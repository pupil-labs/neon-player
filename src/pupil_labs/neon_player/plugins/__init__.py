import json
import typing as T
from pathlib import Path

from numpyencoder import NumpyEncoder
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QMessageBox
from qt_property_widgets.utilities import PersistentPropertiesMixin, property_params

from pupil_labs import neon_player
from pupil_labs.neon_player.ui import QtShortcutType
from pupil_labs.neon_player.ui.timeline_dock import TimeLineDock
from pupil_labs.neon_recording import NeonRecording

if T.TYPE_CHECKING:
    from pupil_labs.neon_player.app import NeonPlayerApp
    from pupil_labs.neon_player.job_manager import JobManager


class GlobalPluginProperties(PersistentPropertiesMixin):
    _known_types: T.ClassVar[list[type["GlobalPluginProperties"]]] = []

    def __init_subclass__(cls) -> None:
        GlobalPluginProperties._known_types.append(cls)
        return super().__init_subclass__()

    def to_dict(
        self,
        include_class_name: bool = False,
        condition: T.Callable[[dict], bool] | None = None,
        recursive: bool = False
    ) -> dict:
        state = super().to_dict(
            include_class_name=include_class_name,
            condition=condition,
            recursive=recursive
        )

        # NOTE: Class name is required to correctly select the properties class
        # when creating global settings form
        state["__class__"] = self.__class__.__name__
        return state


class Plugin(PersistentPropertiesMixin, QObject):
    """Base class for all Neon Player plugins."""

    changed = Signal()
    known_classes: T.ClassVar[list] = []
    global_properties: T.ClassVar[GlobalPluginProperties | None] = None

    def __init__(self) -> None:
        super().__init__()
        self.render_layer = 1
        self._enabled = False

        app = neon_player.instance()
        if app is None:
            return

        app.aboutToQuit.connect(self.on_disabled)

    def register_action(
        self, name: str, shortcut: QtShortcutType, func: T.Callable
    ) -> None:
        return self.app.main_window.register_action(name, shortcut, func)

    def unregister_action(self, name: str) -> None:
        return self.app.main_window.unregister_action(name)

    def register_timeline_action(
        self, name: str, shortcut: QtShortcutType, func: T.Callable
    ) -> None:
        return self.app.main_window.register_action(f"Timeline/{name}", shortcut, func)

    def unregister_timeline_action(self, name: str) -> None:
        return self.app.main_window.unregister_action(f"Timeline/{name}")

    def register_data_point_action(
        self, event_name: str, action_name: str, callback: T.Callable
    ) -> None:
        self.app.main_window.timeline.register_data_point_action(
            event_name, action_name, callback
        )

    def unregister_data_point_action(
        self, event_name: str, action_name: str
    ):
        self.app.main_window.timeline.unregister_data_point_action(
            event_name, action_name
        )

    def add_dynamic_action(self, name: str, func: T.Callable) -> None:
        my_prop_form = self.app.main_window.settings_panel.plugin_class_expanders[
            self.__class__.__name__
        ].content_widget
        my_prop_form.add_action(name, func)

    @classmethod
    def __init_subclass__(cls: type["Plugin"], **kwargs: dict) -> None:  # type: ignore
        super().__init_subclass__(**kwargs)
        if cls.__name__ not in [c.__name__ for c in Plugin.known_classes]:
            Plugin.known_classes.append(cls)

    def on_recording_loaded(self, recording: NeonRecording) -> None:
        """This method is called for all plugins while the recording is loading.

        Override this method to finalize the initialization of the plugin for a given
        recording: e.g., create necessary timeline plots or load data from cache.
        Note that values of all properties that aren't explicitly marked with the
        `dont_encode=True` parameter are already restored from the last session
        by the time `on_recording_loaded` is called.

        Args:
            recording: Neon recording that is currently being opened in Neon Player.
        """
        pass

    def on_disabled(self) -> None:
        """This method is called whenever a plugin is disabled.

        Override this method to clean up any UI elements added by the plugin (e.g.,
        timeline plots). The corresponding section in the plugin settings sidebar
        will be removed automatically.
        """
        pass

    def render(self, painter: QPainter, time_in_recording: int) -> None:
        """This method is called whenever the scene video viewport is rendered.

        Override this method to add overlays on top of the scene video.

        Args:
            painter: The painter object the can be used to display elements on
                the canvas. Note that the painter is transformed to the coordinate
                system of the scene video (i.e., has the resolution of 1600x1200px).
            time_in_recording: The current timestamp of the playhead.
        """
        pass

    def trigger_scene_update(self) -> None:
        """Request an update of the scene video viewport.

        If the video playback is paused, changes to the rendered content may not
        be displayed immediately. Use this method to trigger a scene update explicitly.
        """
        self.app.main_window.video_widget.update()

    def get_timeline(self) -> TimeLineDock:
        """Access the timeline.

        Returns:
            timeline: The timeline controller that allows adding / removing plots.
        """
        return self.app.main_window.timeline

    def user_confirm(self, title: str, message: str) -> bool:
        reply = QMessageBox.question(None, title, message)
        return reply == QMessageBox.StandardButton.Yes

    def get_cache_path(self) -> Path:
        """Get the path to the plugin cache folder.

        The returned location is the recommended place to store cache files (e.g.,
        the result of long-running tasks) without interfering with the original
        recording files or cache of other plugins.

        Returns:
            path: Path to the plugin cache folder.
        """
        if self.recording is None:
            return None

        cache_dir = self.recording._rec_dir / ".neon_player" / "cache"
        return cache_dir / self.__class__.__name__

    def load_cached_json(self, filename: str) -> T.Any:
        """Load a JSON file from the cache folder.

        Args:
            filename: Name of the file in the cache folder.

        Returns:
            contents: File contents.
        """
        if self.recording is None:
            return None

        cache_file = self.get_cache_path() / filename

        if not cache_file.exists():
            return None

        with cache_file.open("r") as f:
            return json.load(f)

    def save_cached_json(self, filename: str, data: T.Any) -> None:
        """Save data to a JSON file in the cache folder.

        Args:
            filename: Name of the file in the cache folder.
            data: Data to be saved.
        """
        if self.recording is None:
            return

        cache_file = self.get_cache_path() / filename
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        with cache_file.open("w") as f:
            json.dump(data, f, cls=NumpyEncoder)

    def get_scene_idx_for_time(
        self,
        t: int = -1,
        method: T.Literal["nearest", "backward", "forward"] = "backward",
        tolerance: int | None = None,
    ) -> int:
        return self.app.get_scene_idx_for_time(t, method, tolerance)

    def is_time_gray(self, t: int = -1) -> bool:
        if t == -1:
            t = self.app.current_ts

        if t < self.recording.scene.time[0]:
            return True

        closest_idx = self.get_scene_idx_for_time(t, method="backward")
        if closest_idx < 0:
            return True

        # Consider gaps above 0.5 s in the scene stream as breaks to be filled
        # with gray frames since such gaps are unlikely to happen naturally
        closest_time = self.recording.scene.time[closest_idx]
        scene_stream_ended = t > self.recording.scene.time[-1]
        threshold = 1 / 30 if scene_stream_ended else 1 / 2

        return abs(t - closest_time) / 1e9 > threshold

    @property
    @property_params(widget=None, dont_encode=True)
    def recording(self) -> NeonRecording | None:
        """Access the currently opened Neon recording.

        Returns:
            recording: Currently opened recording. If no recording is opened,
                `None` is returned.
        """
        return neon_player.instance().recording

    @property
    @property_params(widget=None, dont_encode=True)
    def app(self) -> "NeonPlayerApp":
        """Access the Neon Player application instance.

        Returns:
            app: The Neon Player application instance, if one is launched.
        """
        return neon_player.instance()

    @property
    @property_params(widget=None, dont_encode=True)
    def headless(self) -> bool:
        """Detect if Neon Player is running in headless mode.

        Headless mode is used for background jobs and/or CLI-based usage
        of Neon Player.

        Returns:
            headless: `True` if running in headless mode.
        """
        return self.app is None or self.app.headless

    @property
    @property_params(widget=None, dont_encode=True)
    def job_manager(self) -> "JobManager":
        """Access the background job manager.

        Job manager allows launching long-running tasks in the background
        to avoid blocking the UI while the task is running.

        Returns:
            job_manager: The job manager instance.
        """
        return neon_player.instance().job_manager

    @staticmethod
    def get_class_by_name(name: str) -> type["Plugin"]:
        for cls in Plugin.known_classes:
            if cls.__name__ == name:
                return cls

        raise ValueError(f"Plugin class {name} not found")

    @staticmethod
    def get_instance_by_name(name: str) -> T.Union["Plugin", None]:
        return neon_player.instance().plugins_by_class.get(name)

    @classmethod
    def get_label(cls: type["Plugin"]) -> str:
        if hasattr(cls, "label"):
            return cls.label

        return cls.__name__
