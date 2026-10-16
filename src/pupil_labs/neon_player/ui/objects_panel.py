from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem, QWidget

from pupil_labs.neon_player.managers.panel import Panel, PanelManager


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
        panel_manager.panel_added.connect(self._on_panel_added)
        panel_manager.panel_removed.connect(self._on_panel_removed)
        for panel in panel_manager.panels:
            self._on_panel_added(panel)

    def _on_panel_added(self, panel: Panel) -> None:
        item = QTreeWidgetItem([panel.title])
        item.setToolTip(0, panel.panel_id)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        self._panel_items[panel.panel_id] = item
        self.addTopLevelItem(item)

    def _on_panel_removed(self, panel: Panel) -> None:
        item = self._panel_items.pop(panel.panel_id)
        self.takeTopLevelItem(self.indexOfTopLevelItem(item))
