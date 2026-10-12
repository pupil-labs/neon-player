"""Video panel registration and ownership shared by Neon Player plugins."""

from collections.abc import Callable
from copy import deepcopy
import logging
from dataclasses import dataclass
from typing import cast

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import QWidget
from pyside6_docking import (
    DockError,
    DockFeatures,
    DockManager,
    DockPosition,
    DockWidget,
)

from .workspace_layout import layout_ids, reconcile_layout


@dataclass
class _Panel:
    dock: DockWidget
    on_close: Callable[[], None]
    initial_overlay: QRectF | None = None
    position: DockPosition = DockPosition.RIGHT
    relative_to: str = "scene-video"
    default_overlay: QRectF | None = None


class VideoWorkspace(DockManager):
    def __init__(self, scene_video: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.scene_panel = DockWidget(
            "scene-video", "Scene video", scene_video, features=DockFeatures.NONE
        )
        self.add_dock(self.scene_panel)
        self._panels: dict[str, _Panel] = {}
        self._remembered: dict = {}
        self._changing = False
        self._restore_pending = False
        self._preference_disposed = False
        self.dock_removed.connect(self._on_panel_removed)
        self.layout_changed.connect(self._remember_layout)

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
        self._remember_layout()
        self._changing = True
        try:
            dock = DockWidget(panel_id, title, content, features=features)
            self.add_dock(dock, position, relative_to=target)
            self._panels[panel_id] = _Panel(
                dock, on_close, overlay, position, relative_to, overlay
            )
            if overlay is not None:
                self.overlay_dock(panel_id, relative_to)
        finally:
            self._changing = False
        self._restore_pending = True
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
        pending = {
            panel_id: panel
            for panel_id, panel in self._panels.items()
            if panel.initial_overlay is not None
        }
        if not pending:
            self._restore_preference()
            return
        # Use the public schema-2 restore contract; plugin code never needs to
        # manipulate framework-created widgets or share layout ownership.
        layout = self.save_layout()
        overlays = cast(list[dict[str, object]], layout["overlays"])
        for placement in overlays:
            panel = pending.get(str(placement["dock_id"]))
            if panel is not None and panel.initial_overlay is not None:
                rect = panel.initial_overlay
                placement.update(
                    x=rect.x(), y=rect.y(), width=rect.width(), height=rect.height()
                )
        self._changing = True
        try:
            self.restore_layout(layout)
            for panel in pending.values():
                panel.initial_overlay = None
        finally:
            self._changing = False
        self._restore_preference()

    def load_preference(self, layout: dict) -> None:
        """Accept global settings without activating any plugin."""
        self._remembered = deepcopy(layout)
        self._restore_pending = True
        if self.isVisible():
            self._apply_initial_overlays()

    def save_preference(self) -> dict:
        """Capture live geometry while retaining temporarily absent panels."""
        if not self._preference_disposed:
            self._remember_layout()
        return deepcopy(self._remembered)

    def _remember_layout(self) -> None:
        if self._changing or self._restore_pending or self._preference_disposed:
            return
        live = self.save_layout()
        try:
            self._remembered = (
                reconcile_layout(
                    live,
                    self._remembered,
                    layout_ids(live) | layout_ids(self._remembered),
                )
                if self._remembered
                else live
            )
        except (
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            OverflowError,
            RecursionError,
        ):
            self._remembered = live

    def _restore_preference(self) -> None:
        if not self._restore_pending:
            return
        self._changing = True
        try:
            if self._remembered:
                defaults = self.save_layout()
                restored = reconcile_layout(
                    self._remembered, defaults, layout_ids(defaults)
                )
                self.restore_layout(restored)
        except (
            DockError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            OverflowError,
            RecursionError,
        ) as error:
            logging.warning("Using default video workspace layout: %s", error)
            self._remembered = {}
        finally:
            self._changing = False
            self._restore_pending = False
        self._remember_layout()

    def reset_layout(self) -> None:
        """Reset live panels and discard preferences for absent panels."""
        self._remembered = {}
        self._restore_pending = False
        self._changing = True
        try:
            for panel_id, panel in self._panels.items():
                self.dock(panel_id, panel.position, relative_to=panel.relative_to)
                if panel.default_overlay is not None:
                    self.overlay_dock(panel_id, panel.relative_to)
                    panel.initial_overlay = panel.default_overlay
        finally:
            self._changing = False
        self._apply_initial_overlays()
        self._remember_layout()

    def unregister_panel(self, panel_id: str) -> None:
        """Remove a plugin panel without interpreting teardown as user close."""
        self._remember_layout()
        panel = self._panels.pop(panel_id, None)
        if panel is None:
            return
        self._changing = True
        try:
            if self.get_dock(panel_id) is not None:
                self.remove_dock(panel_id)
            panel.dock.dispose()
            panel.dock.deleteLater()
        finally:
            self._changing = False

    def reveal_panel(self, panel_id: str) -> None:
        """Reveal and focus a panel without changing its placement or tab order."""
        dock = self.get_dock(panel_id)
        if dock is None:
            raise ValueError(f"Unknown panel: {panel_id}")
        self.show_dock(panel_id)
        # The docking API exposes tab activation through its layout snapshot.
        layout = cast(dict, self.save_layout())
        target_id = next(
            (entry["parent_id"] for entry in layout.get("overlays", [])
             if entry["dock_id"] == panel_id),
            panel_id,
        )
        nodes = [layout["main"], *(entry["root"] for entry in layout["floating"])]
        while nodes:
            node = nodes.pop()
            if node is None:
                continue
            if node["type"] == "tabs":
                if target_id in node["docks"]:
                    node["active"] = target_id
                    break
            else:
                nodes.extend(node["children"])
        self.restore_layout(layout)
        window = dock.window()
        window.raise_()
        window.activateWindow()
        dock.content.setFocus(Qt.FocusReason.OtherFocusReason)

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
        if not self._preference_disposed:
            self._remember_layout()
        self._preference_disposed = True
        self._panels.clear()
        super().dispose()
