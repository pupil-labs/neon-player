import json
import typing as T
from pathlib import Path

from numpyencoder import NumpyEncoder
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QAction, QPainter
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
    """Base class for plugin settings that are shared across recordings.

    Subclass this to define plugin settings (using the [`@property`][property]
    decorator) that are applied for each recording by default.
    Such settings appear and can be configured in `File > Global settings`
    under a section with the plugin's name.
    """

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
    """Base class for all Neon Player plugins.

    To create a plugin, subclass this class and, depending on the intended
    goal, add:

    - plugin settings (via [`@property`][property] and
        [`@property_params`](properties.md)
        decorators)
    - launchable actions (via [`@action`][pupil_labs.neon_player.action] and
        [`@action_params`](actions.md)
        decorators)
    - timeline plots (via [`get_timeline`][get_timeline])
    - overlays on top of the scene video (via [`render`][])
    """

    changed = Signal()
    known_classes: T.ClassVar[list] = []
    global_properties: T.ClassVar[GlobalPluginProperties | None] = None
    """Optional default settings used by this plugin for all recordings."""

    def __init__(self) -> None:
        super().__init__()
        self.render_layer = 1
        """Set the render layer of the plugin.

        Plugins render their content on top of the scene video according
        to the ascending order of the `render_layer` parameter."""

        self._enabled = False

        app = neon_player.instance()
        if app is None:
            return

        app.aboutToQuit.connect(self.on_disabled)

    def register_action(
        self, name: str, shortcut: QtShortcutType, func: T.Callable
    ) -> QAction:
        """Add an action to the application menu and optional keyboard shortcut.

        Args:
            name: Label shown for the action. Use '/' to indicate nesting
                (e.g., `"Help/About"`).
            shortcut: Keyboard shortcut, or ``None`` for no shortcut (default).
            func: Function to call when the action is selected.

        Returns:
            action: The created action.
        """
        return self.app.main_window.register_action(name, shortcut, func)

    def unregister_action(self, name: str) -> None:
        """Remove an application-menu action previously added by this plugin.

        Args:
            name: Label used when the action was registered.
        """
        return self.app.main_window.unregister_action(name)

    def register_timeline_action(
        self, name: str, shortcut: QtShortcutType, func: T.Callable
    ) -> None:
        """Add an action to the "Timeline" menu.

        Actions added to the "Timeline" menu are displayed when the user
        right-clicks the timeline area if no data points lie under the cursor.

        Args:
            name: Label shown for the action. Use '/' to indicate nesting
                (e.g., `"Add event/Event 1"`).
            shortcut: Keyboard shortcut, or ``None`` for no shortcut (default).
            func: Function to call when the action is selected.
        """
        return self.app.main_window.register_action(f"Timeline/{name}", shortcut, func)

    def unregister_timeline_action(self, name: str) -> None:
        """Remove a timeline-menu action previously added by this plugin.

        Args:
            name: Label used when the action was registered.
        """
        return self.app.main_window.unregister_action(f"Timeline/{name}")

    def register_data_point_action(
        self, row_name: str, action_name: str, callback: T.Callable
    ) -> None:
        """Add an data point action to the context menu of the timeline.

        This context menu appears when a data point displayed in the
        specified timeline row is right-clicked.

        Args:
            row_name: Name of the timeline row containing the points.
            action_name: Label shown in the context menu.
            callback: Function called with the selected point. The callback
                receives the clicked data point as a ``(timestamp, value)``
                pair. Timestamps are in nanoseconds.
        """
        self.app.main_window.timeline.register_data_point_action(
            row_name, action_name, callback
        )

    def unregister_data_point_action(
        self, row_name: str, action_name: str
    ) -> None:
        """Remove a data-point action previously registered by this plugin.

        Args:
            row_name: Name of the timeline row containing the points.
            action_name: Label used when the action was registered.
        """
        self.app.main_window.timeline.unregister_data_point_action(
            row_name, action_name
        )

    def add_dynamic_action(self, name: str, func: T.Callable) -> None:
        """Add a button to this plugin's section in the settings sidebar.

        Args:
            name: Text displayed on the button.
            func: Function to call when the button is pressed.
        """
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
        """Prepare this plugin after a recording has been opened.

        Override this method to load plugin data from cache or add timeline content.
        Saved plugin settings are restored before this method runs.

        Args:
            recording: The recording that was opened.
        """
        pass

    def on_disabled(self) -> None:
        """Clean up plugin content when the plugin is disabled.

        Override this method to remove items added to the timeline or other
        plugin-created content. The settings section is removed automatically.
        """
        pass

    def render(self, painter: QPainter, time_in_recording: int) -> None:
        """Draw an overlay on the scene video.

        Override this method to paint on top of the current scene frame.

        Args:
            painter: Qt painter for drawing the overlay. The painter uses scene-video
                pixel coordinates.
            time_in_recording: Timestamp of the frame being drawn (in nanoseconds).
        """
        pass

    def trigger_scene_update(self) -> None:
        """Ask Neon Player to redraw the scene video.

        Call this after changing overlay data while playback is paused.
        """
        self.app.main_window.video_widget.update()

    def get_timeline(self) -> TimeLineDock:
        """Access the timeline to add or remove its plots.

        Returns:
            timeline: The shared timeline component.
        """
        return self.app.main_window.timeline

    def user_confirm(self, title: str, message: str) -> bool:
        """Ask the user a Yes/No question and get their response.

        Args:
            title: Title shown on the question dialog.
            message: Question shown to the user.

        Returns:
            confirmed: ``True`` when the user selects Yes; otherwise ``False``.
        """
        reply = QMessageBox.question(None, title, message)
        return reply == QMessageBox.StandardButton.Yes

    def get_cache_path(self) -> Path:
        """Get this plugin's cache folder for the current recording.

        Use this folder for files that can be recreated, such as results from a
        long-running calculation. Returns ``None`` when no recording is open.

        Returns:
            Path to the cache folder, or ``None`` if there is no open recording.
        """
        if self.recording is None:
            return None

        cache_dir = self.recording._rec_dir / ".neon_player" / "cache"
        return cache_dir / self.__class__.__name__

    def load_cached_json(self, filename: str) -> T.Any:
        """Read a JSON file from this plugin's cache folder.

        Args:
            filename: Name of the file to read.

        Returns:
            contents: The decoded JSON value, or ``None`` if the file is
                missing or no recording is currently open.
        """
        if self.recording is None:
            return None

        cache_file = self.get_cache_path() / filename

        if not cache_file.exists():
            return None

        with cache_file.open("r") as f:
            return json.load(f)

    def save_cached_json(self, filename: str, data: T.Any) -> None:
        """Save a value as JSON in this plugin's cache folder.

        Args:
            filename: Name of the file to write.
            data: Value to encode as JSON.
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
        """Find the scene frame closest to a timestamp.

        Args:
            t: Timestamp in nanoseconds. ``-1`` uses the current playhead time.
            method: Whether to choose the nearest frame or only look backward
                or forward in time.
            tolerance: Maximum allowed difference in nanoseconds, or ``None``
                for no limit.

        Returns:
            The frame's index, or ``-1`` when no frame matches.
        """
        return self.app.get_scene_idx_for_time(t, method, tolerance)

    def is_time_gray(self, t: int = -1) -> bool:
        """Check whether the scene video has no usable frame at a timestamp.

        A recording must be open.

        Args:
            t: Timestamp in nanoseconds. ``-1`` checks the current playhead time.

        Returns:
            ``True`` when the video should show a gray frame at that time.
        """
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

        Returns `None` if no recording is opened.
        """
        return neon_player.instance().recording

    @property
    @property_params(widget=None, dont_encode=True)
    def app(self) -> "NeonPlayerApp":
        """Access the main Neon Player application."""
        return neon_player.instance()

    @property
    @property_params(widget=None, dont_encode=True)
    def headless(self) -> bool:
        """Report whether Neon Player is running without its normal user interface.

        Background jobs and command-line tasks run in this mode.
        """
        return self.app is None or self.app.headless

    @property
    @property_params(widget=None, dont_encode=True)
    def job_manager(self) -> "JobManager":
        """Access the service for starting long-running background jobs.

        See the [Job Manager API reference][pupil_labs.neon_player.job_manager.JobManager]
        for the supported method and the returned job's progress and completion signals.
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
