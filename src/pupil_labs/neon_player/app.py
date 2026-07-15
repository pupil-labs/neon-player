import argparse
import importlib.util
import json
import logging
import os
import sys
import time
import typing as T
from importlib.metadata import version
from pathlib import Path

import numpy as np
from pupil_labs.neon_recording.sample import match_ts
from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import QAction, QIcon, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QMessageBox,
    QSystemTrayIcon,
)
from qt_property_widgets.utilities import ComplexEncoder, create_action_object

from pupil_labs import neon_player
from pupil_labs import neon_recording as nr
from pupil_labs.neon_player import Plugin
from pupil_labs.neon_player.ipc_logger import IPCLogger
from pupil_labs.neon_player.job_manager import JobManager
from pupil_labs.neon_player.plugin_management import (
    check_dependencies_for_plugin,
    install_dependencies,
)
from pupil_labs.neon_player.plugins import (
    audio,  # noqa: F401
    blinks,  # noqa: F401
    events,  # noqa: F401
    export_all,  # noqa: F401
    eye_overlay,  # noqa: F401
    eyestate,  # noqa: F401
    fixations,  # noqa: F401
    gaze,  # noqa: F401
    imu,  # noqa: F401
    scene_renderer,  # noqa: F401
    surface_tracking,  # noqa: F401
    video_exporter,  # noqa: F401
)
from pupil_labs.neon_player.history import RecordingHistory
from pupil_labs.neon_player.settings import (
    GeneralSettings, PluginSettingsDispatcher
)
from pupil_labs.neon_player.ui.main_window import MainWindow
from pupil_labs.neon_player.ui.plugin_installation_dialog import (
    PluginInstallationDialog,
)
from pupil_labs.neon_player.utilities import SlotDebouncer, clone_menu
from pupil_labs.neon_player.workspace import Workspace, check_if_neon_recording
from pupil_labs.neon_recording import NeonRecording


class NeonPlayerApp(QApplication):
    export_window_changed = Signal()
    playback_state_changed = Signal(bool)
    position_changed = Signal(object)
    seeked = Signal(object)
    speed_changed = Signal(float)
    recording_loaded = Signal(object)
    recording_unloaded = Signal()
    workspace_loaded = Signal()
    workspace_unloaded = Signal()

    def __init__(self, argv: list[str]) -> None:
        self._initializing = True
        super().__init__(argv)

        try:
            app_version = version("pupil_labs.neon_player")
        except Exception:
            app_version = "?"

        self.setApplicationName("Neon Player")
        self.setApplicationVersion(app_version)
        self.setWindowIcon(QIcon(str(neon_player.asset_path("neon-player.svg"))))
        self.setStyle("Fusion")

        self.tray_icon = QSystemTrayIcon()
        self.tray_icon.setIcon(self.windowIcon())
        self.tray_icon.setToolTip("Neon Player")

        self.plugins_by_class: dict[str, Plugin] = {}
        self.plugins: list[Plugin] = []
        self.recording: nr.NeonRecording | None = None
        self.workspace: Workspace = Workspace()
        self.batch_mode_enabled: bool = False
        self.playback_start_anchor = 0
        self.current_ts = 0
        self.playback_speed_options = [
            -4.0, -2.0, -1.0, -0.5, -0.25, -0.125, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0
        ]
        self.playback_speed = 1.0

        self.settings = GeneralSettings()
        self.loading_recording = False
        self.recording_history = RecordingHistory()

        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(1000 / 30)
        self.refresh_timer.timeout.connect(self.poll)

        self.job_manager = JobManager()

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "recording", nargs="?", default=None,
            help="Path to a folder with one or more recordings to load on startup"
        )
        parser.add_argument(
            "--workspace", action="store_true",
            help="Whether to load parent folder as workspace"
        )
        parser.add_argument(
            "--recording-settings", type=str, default=None,
            help="Path to recording settings JSON file to use on load"
        )
        parser.add_argument(
            "--workspace-settings", type=str, default=None,
            help="Path to workspace settings JSON file to use on load"
        )
        parser.add_argument("--progress-ipc-name", type=str, default=None)
        parser.add_argument(
            "--job",
            nargs="+",
            default=None,
        )

        self.args = parser.parse_args()

        self.progress_ipc_name = self.args.progress_ipc_name

        self.main_window = MainWindow()

        self.ipc_logger = IPCLogger()
        logging.info(
            f"{self.applicationName()} v{self.applicationVersion()} starting up"
        )

        if self.args.job and self.args.job[0] == "install_packages":
            packages = self.args.job[1:]
            self.job_manager.work_job(install_dependencies(packages))
            self.quit()
            return

        # Iterate through all modules within plugins and register them
        plugin_search_path = Path.home() / "Pupil Labs" / "Neon Player" / "plugins"
        if plugin_search_path.exists():
            self.find_plugins(plugin_search_path)

        # NOTE: plugins should be loaded before initializing the dispatcher
        # to ensure that property scopes are collected correctly
        self.session_settings = PluginSettingsDispatcher()
        self.session_settings.export_window_changed.connect(
            self.export_window_changed.emit
        )
        self.session_settings.changed.connect(self.toggle_plugins_by_settings)
        SlotDebouncer.debounce(self.session_settings.changed, self.save_settings)

        # Use custom paths to settings files if provided
        self._recording_settings_path = None
        if self.args.recording_settings:
            self._recording_settings_path = Path(self.args.recording_settings)

        self._workspace_settings_path = None
        if self.args.workspace_settings:
            self._workspace_settings_path = Path(self.args.workspace_settings)

        # Load global settings and recording history
        try:
            self.settings = GeneralSettings.from_dict(self.load_global_settings())
        except FileNotFoundError:
            logging.warning("Settings file not found")
        except Exception:
            logging.exception("Failed to load settings")

        try:
            self.recording_history = RecordingHistory.from_dict(self.load_recording_history())
        except FileNotFoundError:
            logging.warning("Recording history file not found")
        except Exception:
            logging.exception("Failed to load recording history")
        self.recording_history.changed.connect(self.save_history)

        if self.args.job and self.args.recording:
            path_to_load = Path(self.args.recording)
            if self.args.workspace:
                self.set_batch_mode(True)
                self.load_workspace(path_to_load.parent)
                self.workspace_loaded.emit()
                self.load_recording(path_to_load)
            else:
                self.load(path_to_load)
        elif self.args.recording:
            QTimer.singleShot(1, lambda: self.load(Path(self.args.recording)))
        else:
            self._initializing = False
            os.chdir(Path.home())

        self.aboutToQuit.connect(self.unload)

    def run_jobs(self, job):
        plugin_name, action_name = job[0].split(".")
        job_args = job[1:]

        for plugin in self.plugins:
            if plugin.__class__.__name__ == plugin_name:
                # use an action object to provide type conversion for arguments
                if action_name not in plugin._action_objects:
                    action_obj = create_action_object(
                        getattr(plugin, action_name), plugin
                    )
                else:
                    action_obj = plugin._action_objects[action_name]

                keys = list(action_obj.args.keys())
                if len(keys) > 0 and keys[0] == "self":
                    keys = keys[1:]

                args = dict(zip(keys, job_args))
                action_obj.__setstate__(args)
                self.job_manager.work_job(action_obj())

                break
        else:
            logging.error(f"Could not find plugin action method: {job[0]}")

        self.quit()

    @property
    def history_path(self) -> Path:
        return Path.home() / "Pupil Labs" / "Neon Player" / "history.json"

    @property
    def settings_path(self) -> Path:
        return Path.home() / "Pupil Labs" / "Neon Player" / "settings.json"

    def recording_settings_path(self, recording: NeonRecording | None = None) -> Path | None:
        if recording is None:
            if self._recording_settings_path is not None:
                return self._recording_settings_path
            recording = self.recording

        if recording is None:
            return None

        return recording._rec_dir / ".neon_player" / "settings.json"

    @property
    def workspace_settings_path(self) -> Path | None:
        if self._workspace_settings_path is not None:
            return self._workspace_settings_path

        if self.workspace.path is None:
            return None

        return self.workspace.path / ".neon_player" / "workspace-settings.json"

    def load_global_settings(self) -> T.Any:
        logging.info(f"Loading settings from {self.settings_path}")
        return json.loads(self.settings_path.read_text())

    def load_recording_history(self) -> T.Any:
        logging.info(f"Loading recording history from {self.history_path}")
        return json.loads(self.history_path.read_text())

    def save_settings(self, force: bool = False) -> None:
        if self._initializing and not force:
            return

        try:
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            data = self.settings.to_dict()
            with self.settings_path.open("w") as f:
                json.dump(data, f, cls=ComplexEncoder)

            if self.recording:
                self.session_settings.save_recording_settings(
                    self.recording_settings_path()
                )

            if self.batch_mode_enabled:
                self.session_settings.save_workspace_settings(
                    self.workspace_settings_path
                )

            logging.info("Settings saved")
        except Exception:
            logging.exception("Failed to save settings")
            raise

    def save_history(self) -> None:
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            data = self.recording_history.recent_recordings
            with self.history_path.open("w") as f:
                json.dump(data, f, cls=ComplexEncoder)

            logging.info("History saved")
        except Exception:
            logging.exception("Failed to save history")

    def find_plugins(self, path: Path) -> None:
        sys.path.append(str(path))
        sys.path.append(str(path / "site-packages"))
        logging.info(f"Searching for plugins in {path}")
        for d in path.iterdir():
            if d.is_file() and d.suffix != ".py":
                continue

            if d.name in ["__pycache__", "site-packages"]:
                continue

            try:
                if d.is_dir():
                    spec = importlib.util.spec_from_file_location(
                        d.stem, d / "__init__.py"
                    )
                else:
                    spec = importlib.util.spec_from_file_location(d.stem, d)

                if spec is None:
                    continue

                missing_dependencies = check_dependencies_for_plugin(d)

                if missing_dependencies:
                    dialog = PluginInstallationDialog(missing_dependencies, d.name)
                    dialog.exec()

                logging.info(f"Importing plugin module {d}")

                module = importlib.util.module_from_spec(spec)
                sys.modules[d.stem] = module
                if spec.loader:
                    spec.loader.exec_module(module)

            except Exception:
                logging.exception(f"Failed to import plugin module {d}")

    def toggle_plugin(
        self,
        kls: type[Plugin] | str,
        enabled: bool,
        state: dict | None = None,
        delete: bool = True
    ) -> Plugin | None:
        if isinstance(kls, str):
            try:
                kls = Plugin.get_class_by_name(kls)
            except ValueError:
                if enabled:
                    logging.warning(f"Couldn't enable plugin class: {kls}")
                return None

        plugin_exists = kls.__name__ in self.plugins_by_class
        if plugin_exists and not isinstance(self.plugins_by_class[kls.__name__], kls):
            raise RuntimeError(f"Invalid instance of plugin found: {kls.__name__}")
        currently_enabled = plugin_exists and self.plugins_by_class[kls.__name__]._enabled

        if enabled and not currently_enabled:
            logging.info(f"Enabling plugin: {kls.__name__}")
            try:
                if state is None:
                    state = self.session_settings.plugin_states.get(kls.__name__, {})

                if plugin_exists:
                    plugin = self.plugins_by_class[kls.__name__]
                else:
                    plugin: Plugin = kls.from_dict(state)
                    self.plugins_by_class[kls.__name__] = plugin

                    plugin.changed.connect(self.main_window.video_widget.update)
                    SlotDebouncer.debounce(plugin.changed, self.save_settings)

                self.main_window.settings_panel.add_plugin_settings(plugin)
                if self.recording:
                    plugin.on_recording_loaded(self.recording)

                plugin._enabled = True
            except Exception:
                logging.exception(f"Failed to enable plugin {kls}")
                return None

        elif not enabled and currently_enabled:
            plugin = self.plugins_by_class[kls.__name__]

            plugin.on_disabled()
            plugin._enabled = False
            self.main_window.settings_panel.remove_plugin_settings(kls.__name__)
            if delete:
                plugin.on_deleted()
                plugin.deleteLater()
                del self.plugins_by_class[kls.__name__]

            logging.info(f"Disabled plugin: {kls.__name__}")

        self.plugins = [p for p in self.plugins_by_class.values() if p._enabled]
        self.plugins.sort(key=lambda p: p.render_layer)

        self.main_window.video_widget.update()

    def run(self) -> int:
        if not self.headless:
            self.main_window.show()
            self.tray_icon.show()
            menu = self.main_window.get_menu("File", auto_create=False)
            context_menu = clone_menu(menu)
            self.tray_icon.setContextMenu(context_menu)

        else:
            if sys.platform == "darwin":
                # hide dock icon for background jobs
                from AppKit import NSApplication, NSApplicationActivationPolicyProhibited

                NSApplication.sharedApplication().setActivationPolicy_(
                    NSApplicationActivationPolicyProhibited
                )

        return self.exec()

    def show_notification(
        self,
        title: str,
        message: str,
        icon: QSystemTrayIcon.MessageIcon
        | QIcon = QSystemTrayIcon.MessageIcon.Information,
        duration: int = 10000,
    ) -> None:
        self.tray_icon.showMessage(title, message, icon, duration)

    @property
    def headless(self) -> bool:
        return self.args.job is not None

    def unload(self) -> None:
        if self.recording:
            self.unload_recording(delete_plugins=True)

        self.workspace.clear()
        self.workspace_unloaded.emit()

    def unload_recording(self, delete_plugins: bool = False) -> None:
        self.set_playback_state(False)
        self.save_settings(force=True)
        class_names = list(self.plugins_by_class.keys())

        timeline = self.main_window.timeline
        timeline.disable_plot_sorting()
        for plugin_class_name in class_names:
            self.toggle_plugin(plugin_class_name, False, delete=delete_plugins)
        timeline.enable_plot_sorting()

        if self.recording is not None:
            self.recording.close()
        self.recording = None
        self.recording_unloaded.emit()

    def load(self, path: Path) -> None:
        """
        Opens the Neon recording in the provided folder (if native data
        format is detected) or opens the folder as a workplace, detecting
        recordings in subfolders automatically and loading the first one.
        """
        self.unload()
        is_neon_recording = check_if_neon_recording(path)
        self.set_batch_mode(not is_neon_recording)

        if is_neon_recording:
            # Only include this recording in the workspace
            recording_path = path
            self.workspace.add_recording(recording_path)
        else:
            # Load all recordings that appear as first-level subfolders
            self.load_workspace(path)
            if not self.workspace.recordings:
                QMessageBox.critical(
                    self.main_window,
                    "No recordings found!",
                    "Found no recordings in the provided folder. "
                    "Please ensure that Native Recording Data is used or "
                    "select a different folder."
                )
                return

            recording_path = self.workspace.recordings[0]._rec_dir

        self.workspace_loaded.emit()
        self.load_recording(recording_path)

    def set_batch_mode(self, enabled: bool) -> None:
        self.batch_mode_enabled = enabled
        self.session_settings.batch_mode_enabled = enabled

    def load_workspace(self, path: Path) -> None:
        self.workspace.load_recording_list(path)
        self.session_settings.load_workspace_settings(
            self.workspace_settings_path
        )

    def load_recording(self, path: Path) -> None:
        """Load a recording from the given path."""
        self.loading_recording = True
        self._initializing = True
        self.unload_recording()
        logging.info("Opening recording at path: %s", path)
        self.recording = nr.load(path)
        self.recording_history.add_recording(path, self.recording)

        os.chdir(path)
        self.playback_start_anchor = 0

        self.main_window.on_recording_loaded(self.recording)
        self.session_settings.load_recording_settings(
            self.recording_settings_path(), self.recording
        )

        if self.settings.skip_gray_frames_on_load:
            self.seek_to(self.recording.scene[0].time)
        else:
            self.seek_to(self.recording.start_time)

        QTimer.singleShot(0, self.toggle_plugins_by_settings)
        QTimer.singleShot(10, self.on_recording_load_complete)

    def on_recording_load_complete(self) -> None:
        self.loading_recording = False
        self.recording_loaded.emit(self.recording)
        logging.info(f"Loaded `{self.recording._rec_dir}`")

    def toggle_plugins_by_settings(self) -> None:
        timeline = self.main_window.timeline
        timeline.disable_plot_sorting()

        for cls_name, enabled in self.session_settings.enabled_plugins.items():
            state = self.session_settings.plugin_states.get(cls_name, {})
            self.toggle_plugin(cls_name, enabled, state)

        timeline.enable_plot_sorting()
        self._initializing = False

        if self.args.job:
            logging.info("Running jobs")
            self.run_jobs(self.args.job)

    def get_action(self, action_path: str) -> QAction:
        return self.main_window.get_action(action_path)

    def toggle_play(self) -> None:
        if self.recording is None:
            return

        if self.current_ts >= self.recording.stop_time:
            self.current_ts = self.recording.start_time

        if self.refresh_timer.isActive():
            self.refresh_timer.stop()

        else:
            self._reset_start_anchor()
            self.refresh_timer.start()

        self.playback_state_changed.emit(self.refresh_timer.isActive())

    def set_playback_speed(self, speed: float) -> None:
        self.playback_speed = speed
        self._reset_start_anchor()
        self.speed_changed.emit(speed)

    def switch_playback_speed(self, by: int) -> None:
        current_idx = self.playback_speed_options.index(self.playback_speed)
        new_idx = current_idx + by
        if new_idx < 0 or new_idx >= len(self.playback_speed_options):
            return

        new_speed = self.playback_speed_options[new_idx]
        self.set_playback_speed(new_speed)

    def _reset_start_anchor(self) -> None:
        if self.playback_speed == 0:
            return

        now = time.time_ns()
        elapsed_time = (
            self.current_ts - self.recording.start_time
        ) / self.playback_speed
        self.playback_start_anchor = now - elapsed_time

    def set_playback_state(self, playing: bool) -> None:
        if self.is_playing != playing:
            self.toggle_play()

    def poll(self) -> None:
        if self.recording is None:
            return

        if self.playback_speed == 0:
            return

        now = time.time_ns()
        elapsed_time = (now - self.playback_start_anchor) * self.playback_speed
        target_ts = int(elapsed_time + self.recording.start_time)

        if self.recording.start_time <= target_ts <= self.recording.stop_time:
            self.current_ts = target_ts
            self.main_window.set_time_in_recording(self.current_ts)

        else:
            self.current_ts = min(
                max(target_ts, self.recording.start_time), self.recording.stop_time
            )
            self.main_window.set_time_in_recording(self.current_ts)

            self.refresh_timer.stop()
            self.playback_state_changed.emit(self.refresh_timer.isActive())

        self.position_changed.emit(self.current_ts)

    def seek_to(self, ts: int) -> None:
        if self.recording is None:
            return

        restart_playback = self.is_playing
        if restart_playback:
            self.set_playback_state(False)

        ts = min(max(int(ts), self.recording.start_time), self.recording.stop_time)

        now = time.time_ns()
        self.current_ts = ts
        self.playback_start_anchor = now - (ts - self.recording.start_time)
        self.main_window.set_time_in_recording(ts)

        self.position_changed.emit(self.current_ts)
        self.seeked.emit(self.current_ts)

        if restart_playback:
            QTimer.singleShot(5, lambda: self.set_playback_state(True))

    def seek_by(self, ns: int) -> None:
        self.seek_to(self.current_ts + ns)

    def seek_by_frame(self, frame_delta: int) -> None:
        if self.recording is None:
            return

        scene_idx = self.get_scene_idx_for_time()
        scene_idx = max(0, min(scene_idx + frame_delta, len(self.recording.scene) - 1))
        self.seek_to(self.recording.scene[scene_idx].time)

    def get_scene_idx_for_time(
        self,
        t: int = -1,
        method: T.Literal["nearest", "backward", "forward"] = "backward",
        tolerance: int | None = None,
    ) -> int:
        if t < 0:
            t = self.current_ts

        scene_idx = match_ts([t], self.recording.scene.time, method, tolerance)[0]
        return -1 if np.isnan(scene_idx) else scene_idx

    def render_to(self, painter: QPainter, ts: int | None = None) -> None:
        if ts is None:
            ts = self.current_ts

        brush = painter.brush()
        pen = painter.pen()
        font = painter.font()
        painter.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )

        for plugin in self.plugins:
            plugin.render(painter, ts)
            painter.setBrush(brush)
            painter.setPen(pen)
            painter.setFont(font)
            painter.setOpacity(1.0)

    def export_all(self, export_path: Path) -> None:
        for plugin in self.plugins:
            if hasattr(plugin, "export"):
                try:
                    plugin.export(Path(export_path))
                except Exception:
                    logging.exception(f"Exception while exporting plugin {plugin}")

    @property
    def is_playing(self) -> bool:
        return self.refresh_timer.isActive()

    def get_export_window(self) -> tuple[int, int] | None:
        if self.recording is None:
            return None

        return self.session_settings.export_window

    def set_export_window(self, export_window: tuple[int, int]) -> None:
        if self.recording is None:
            return

        if not isinstance(export_window, tuple) or len(export_window) != 2:
            raise ValueError(
                "Export window must be a tuple with two integer timestamps (start, end)"
            )

        self.session_settings.export_window = export_window
        self.main_window.timeline.set_export_window(export_window)
