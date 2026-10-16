"""Copy into the custom-plugin directory and enable Panels Demo for a recording.

The static widget never subscribes to playback. The position widget subscribes
only after registration supplies playback. Discovery uses synchronous GUI-thread
connect-then-snapshot; observers must not mutate panels during notifications.
"""

from contextlib import ExitStack

from pupil_labs.neon_recording import NeonRecording
from PySide6.QtWidgets import QLabel

from pupil_labs.neon_player import Panel, Plugin, action
from pupil_labs.neon_player.managers.playback import PlaybackPosition


class PlaybackLabel(QLabel):
    def bind_playback(self, playback: PlaybackPosition) -> None:
        playback.changed.connect(self.show_position)
        self.show_position(playback.time)

    def show_position(self, timestamp: int | None) -> None:
        self.setText(
            "No recording" if timestamp is None else f"Recording time: {timestamp} ns"
        )


class PanelsDemo(Plugin):
    label = "Panels Demo"

    def __init__(self) -> None:
        super().__init__()
        self._owned_panels: dict[str, Panel] = {}
        self._known_panels: dict[str, Panel] = {}
        self._observing = False

    def on_recording_loaded(self, recording: NeonRecording) -> None:
        if not self._observing:
            self.panel_manager.panel_added.connect(self._on_panel_added)
            self.panel_manager.panel_removed.connect(self._on_panel_removed)
            self._observing = True
            for panel in self.panel_manager.panels:
                self._on_panel_added(panel)
        self.open_panels()

    @action
    def open_panels(self) -> None:
        """Create missing demo panels; closed panels reopen as fresh instances."""
        if "demo.static" not in self._owned_panels:
            self._add_panel(
                Panel(
                    "demo.static",
                    "Static analysis",
                    QLabel("Static content: playback does not recompute this widget."),
                )
            )
        if "demo.playback" not in self._owned_panels:
            widget = PlaybackLabel()
            panel = Panel("demo.playback", "Playback position", widget)
            self._add_panel(panel)
            widget.bind_playback(panel.playback)

    def _add_panel(self, panel: Panel) -> None:
        with ExitStack() as rollback:
            rollback.callback(panel.deleteLater)
            rollback.callback(panel.widget.deleteLater)
            self.add_panel(panel)
            rollback.pop_all()
        self._owned_panels[panel.panel_id] = panel

    def _on_panel_added(self, panel: Panel) -> None:
        self._known_panels[panel.panel_id] = panel

    def _on_panel_removed(self, panel: Panel) -> None:
        self._known_panels.pop(panel.panel_id, None)
        if self._owned_panels.get(panel.panel_id) is panel:
            del self._owned_panels[panel.panel_id]

    def on_disabled(self) -> None:
        # Optional explicit removal while discovery is still connected.
        for panel in tuple(self._owned_panels.values()):
            self.remove_panel(panel)
        if self._observing:
            self.panel_manager.panel_added.disconnect(self._on_panel_added)
            self.panel_manager.panel_removed.disconnect(self._on_panel_removed)
            self._observing = False
        self._known_panels.clear()
