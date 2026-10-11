"""Video panel registration and ownership shared by Neon Player plugins."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from PySide6.QtCore import QRectF, QTimer
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import QWidget
from pyside6_docking import DockFeatures, DockManager, DockPosition, DockWidget


@dataclass
class _Panel:
    dock: DockWidget
    on_close: Callable[[], None]
    initial_overlay: QRectF | None = None


class VideoWorkspace(DockManager):
    def __init__(self, scene_video: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.scene_panel = DockWidget(
            "scene-video", "Scene video", scene_video, features=DockFeatures.NONE
        )
        self.add_dock(self.scene_panel)
        self._panels: dict[str, _Panel] = {}
        self.dock_removed.connect(self._on_panel_removed)

    def register_panel(
        self,
        panel_id: str,
        title: str,
        content: QWidget,
        *,
        on_close: Callable[[], None],
        features: DockFeatures = DockFeatures.MOVABLE | DockFeatures.CLOSABLE,
        position: DockPosition = DockPosition.RIGHT,
        relative_to: str = "scene-video",
        overlay: QRectF | None = None,
    ) -> DockWidget:
        """Register plugin-owned content; overlay rectangles are parent fractions."""
        if self.get_dock(panel_id) is not None:
            raise ValueError(f"Panel already registered: {panel_id}")
        target = self.get_dock(relative_to)
        if target is None:
            raise ValueError(f"Unknown placement target: {relative_to}")
        dock = DockWidget(panel_id, title, content, features=features)
        self.add_dock(dock, position, relative_to=target)
        self._panels[panel_id] = _Panel(dock, on_close, overlay)
        if overlay is not None:
            self.overlay_dock(panel_id, relative_to)
            if self.isVisible():
                self._apply_initial_overlays()
        return dock

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        # Hidden docks have staging dimensions. Wait for the real layout before
        # resolving the requested fractions into the package's pixel geometry.
        QTimer.singleShot(0, self, self._apply_initial_overlays)

    def _apply_initial_overlays(self) -> None:
        if not self.isVisible():
            return
        pending = {panel_id: panel for panel_id, panel in self._panels.items()
                   if panel.initial_overlay is not None}
        if not pending:
            return
        # Use the public schema-2 restore contract; plugin code never needs to
        # manipulate framework-created widgets or share layout ownership.
        layout = self.save_layout()
        overlays = cast(list[dict[str, object]], layout["overlays"])
        for placement in overlays:
            panel = pending.get(str(placement["dock_id"]))
            if panel is not None and panel.initial_overlay is not None:
                rect = panel.initial_overlay
                placement.update(x=rect.x(), y=rect.y(),
                                 width=rect.width(), height=rect.height())
        self.restore_layout(layout)
        for panel in pending.values():
            panel.initial_overlay = None

    def unregister_panel(self, panel_id: str) -> None:
        """Remove a plugin panel without interpreting teardown as user close."""
        panel = self._panels.pop(panel_id, None)
        if panel is None:
            return
        if self.get_dock(panel_id) is not None:
            self.remove_dock(panel_id)
        panel.dock.dispose()
        panel.dock.deleteLater()

    def _on_panel_removed(self, panel_id: str) -> None:
        panel = self._panels.get(panel_id)
        if panel is None:
            return

        def notify_owner() -> None:
            if self._panels.get(panel_id) is panel:
                self.unregister_panel(panel_id)
                panel.on_close()

        # Docking callbacks cannot synchronously mutate the manager. The identity
        # check also prevents an old close from disabling a replacement panel.
        QTimer.singleShot(0, self, notify_owner)

    def dispose(self) -> None:
        self._panels.clear()
        super().dispose()
