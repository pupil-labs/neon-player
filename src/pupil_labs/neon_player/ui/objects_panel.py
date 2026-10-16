from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem, QWidget

from pupil_labs.neon_player.managers.panel import Panel, PanelManager
from pupil_labs.neon_player.overlays import PanelOverlay
from pupil_labs.neon_player.ui.panel_widget import PanelWidget


class ObjectsPanel(QTreeWidget):
    def __init__(
        self, panel_manager: PanelManager, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setColumnCount(1)
        self.setHeaderHidden(True)
        self.setSelectionMode(QTreeWidget.SelectionMode.NoSelection)
        self.setEditTriggers(QTreeWidget.EditTrigger.NoEditTriggers)
        self.setDragDropMode(QTreeWidget.DragDropMode.NoDragDrop)
        self.setSortingEnabled(False)

        self._panel_items: dict[str, QTreeWidgetItem] = {}
        self._overlay_items: dict[
            tuple[Panel, int, PanelOverlay], QTreeWidgetItem
        ] = {}
        self._panel_manager = panel_manager
        panel_manager.panel_added.connect(self._on_panel_added)
        panel_manager.panel_removed.connect(self._on_panel_removed)
        panel_manager.overlay_added.connect(self._on_overlay_added)
        panel_manager.overlay_removed.connect(self._on_overlay_removed)
        for panel in panel_manager.panels:
            self._on_panel_added(panel)

    def _on_panel_added(self, panel: Panel) -> None:
        item = QTreeWidgetItem([panel.title])
        item.setToolTip(0, panel.panel_id)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        self._panel_items[panel.panel_id] = item
        self.addTopLevelItem(item)
        hosts = self._panel_manager.get_hosts(panel)
        for host in hosts:
            host_id = id(host)
            for overlay in host.overlays:
                self._add_overlay_item(
                    panel,
                    host_id,
                    overlay,
                    self._panel_manager.get_overlay_owner(overlay),
                )

    def _on_panel_removed(self, panel: Panel) -> None:
        self._remove_overlay_items(panel)
        item = self._panel_items.pop(panel.panel_id)
        self.takeTopLevelItem(self.indexOfTopLevelItem(item))

    def _on_overlay_added(
        self,
        panel: Panel,
        host: PanelWidget,
        overlay: PanelOverlay,
        owner: object | None,
    ) -> None:
        if self._panel_manager.get_panel(panel.panel_id) is not panel:
            return
        host_id = id(host)
        self._add_overlay_item(panel, host_id, overlay, owner)

    def _add_overlay_item(
        self,
        panel: Panel,
        host_id: int,
        overlay: PanelOverlay,
        owner: object | None,
    ) -> None:
        key = (panel, host_id, overlay)
        if key in self._overlay_items:
            return
        if self._panel_manager.get_panel(panel.panel_id) is not panel:
            return
        parent = self._panel_items.get(panel.panel_id)
        if parent is None:
            return
        if owner is None:
            owner_identity = "Host-owned"
        else:
            owner_identity = f"Plugin {type(owner).__qualname__}@{id(owner):x}"
        overlay_identity = f"{type(overlay).__qualname__}@{id(overlay):x}"
        item = QTreeWidgetItem([overlay.display_name])
        item.setToolTip(0, f"{owner_identity}\nOverlay {overlay_identity}")
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        parent.addChild(item)
        self._overlay_items[key] = item

    def _on_overlay_removed(
        self,
        panel: Panel,
        host: PanelWidget,
        overlay: PanelOverlay,
        _owner: object | None,
    ) -> None:
        host_id = id(host)
        self._remove_overlay_item(panel, host_id, overlay)

    def _remove_overlay_item(
        self, panel: Panel, host_id: int, overlay: PanelOverlay
    ) -> None:
        item = self._overlay_items.pop((panel, host_id, overlay), None)
        if item is not None:
            parent = self._panel_items.get(panel.panel_id)
            if parent is not None:
                parent.removeChild(item)

    def _remove_overlay_items(
        self, panel: Panel, host_id: int | None = None
    ) -> None:
        for key in tuple(self._overlay_items):
            if key[0] is panel and (host_id is None or key[1] == host_id):
                self._remove_overlay_item(panel, key[1], key[2])
