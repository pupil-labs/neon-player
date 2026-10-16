from collections.abc import Iterator

from PySide6.QtCore import QModelIndex, Qt
from PySide6.QtWidgets import (
    QLabel,
    QScrollArea,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qt_property_widgets.utilities import PersistentPropertiesMixin
from qt_property_widgets.widgets import PropertyForm

from pupil_labs.neon_player.managers.panel import Panel, PanelManager
from pupil_labs.neon_player.overlays import InteractivePanelOverlay2D, PanelOverlay
from pupil_labs.neon_player.ui.panel_widget import PanelWidget


class InspectorPanel(QWidget):
    def __init__(
        self, panel_manager: PanelManager, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.splitter = QSplitter(Qt.Orientation.Vertical, self)
        layout.addWidget(self.splitter)

        self.tree_widget = QTreeWidget(self.splitter)
        self.tree_widget.setColumnCount(1)
        self.tree_widget.setHeaderHidden(True)
        self.tree_widget.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self.tree_widget.setEditTriggers(QTreeWidget.EditTrigger.NoEditTriggers)
        self.tree_widget.setDragDropMode(QTreeWidget.DragDropMode.NoDragDrop)
        self.tree_widget.setSortingEnabled(False)
        self.tree_widget.currentItemChanged.connect(self._on_tree_selection_changed)

        self.properties_scroll_area = QScrollArea(self.splitter)
        self.properties_scroll_area.setWidgetResizable(True)
        self.properties_widget = QWidget()
        self._properties_layout = QVBoxLayout(self.properties_widget)
        self._properties_layout.setContentsMargins(6, 6, 6, 6)
        self.empty_properties_label = QLabel(
            "Properties of the clicked overlay will be displayed here."
        )
        self._properties_layout.addWidget(self.empty_properties_label)
        self._properties_layout.addStretch()
        self.properties_scroll_area.setWidget(self.properties_widget)

        self.splitter.addWidget(self.tree_widget)
        self.splitter.addWidget(self.properties_scroll_area)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setSizes([100, 200])

        self._panel_items: dict[str, QTreeWidgetItem] = {}
        self._overlay_items: dict[
            tuple[Panel, int, PanelOverlay], QTreeWidgetItem
        ] = {}
        self._panel_manager = panel_manager
        self.property_form: PropertyForm | None = None
        self.selected_object_label: QLabel | None = None
        panel_manager.panel_added.connect(self._on_panel_added)
        panel_manager.panel_removed.connect(self._on_panel_removed)
        panel_manager.overlay_added.connect(self._on_overlay_added)
        panel_manager.overlay_removed.connect(self._on_overlay_removed)
        panel_manager.selection_changed.connect(self._on_selection_changed)
        for panel in panel_manager.panels:
            self._on_panel_added(panel)
        self._on_selection_changed(panel_manager.selected_object)

    def _on_panel_added(self, panel: Panel) -> None:
        item = QTreeWidgetItem([panel.title])
        item.setToolTip(0, panel.panel_id)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        item.setData(0, Qt.ItemDataRole.UserRole, panel.widget)
        self._panel_items[panel.panel_id] = item
        self.tree_widget.addTopLevelItem(item)
        item.setExpanded(True)
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
        self.tree_widget.takeTopLevelItem(self.tree_widget.indexOfTopLevelItem(item))

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
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        item.setData(0, Qt.ItemDataRole.UserRole, overlay)
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

    def _on_tree_selection_changed(
        self, current: QTreeWidgetItem | None, _previous: QTreeWidgetItem | None
    ) -> None:
        selected_object = (
            None
            if current is None
            else current.data(0, Qt.ItemDataRole.UserRole)
        )
        self._panel_manager.select_object(selected_object)

    def _on_selection_changed(self, selected_object: object | None) -> None:
        selected_item = next(
            (
                item
                for item in self._iter_items()
                if item.data(0, Qt.ItemDataRole.UserRole) is selected_object
            ),
            None,
        )
        if self.tree_widget.currentItem() is not selected_item:
            if selected_item is None:
                self.tree_widget.setCurrentIndex(QModelIndex())
            else:
                self.tree_widget.setCurrentItem(selected_item)

        self._clear_property_form()
        self.empty_properties_label.setVisible(selected_object is None)
        if not isinstance(selected_object, PersistentPropertiesMixin) or not isinstance(
            selected_object, InteractivePanelOverlay2D
        ):
            return

        self.selected_object_label = QLabel(selected_object.display_name)
        label_font = self.selected_object_label.font()
        label_font.setBold(True)
        label_font.setPointSizeF(max(label_font.pointSizeF() + 1.0, 10.0))
        self.selected_object_label.setFont(label_font)
        self._properties_layout.insertWidget(0, self.selected_object_label)
        self.property_form = PropertyForm(selected_object)
        property_form_layout = self.property_form.layout()
        if property_form_layout is None:
            raise RuntimeError("PropertyForm has no layout")
        property_form_layout.setContentsMargins(12, 4, 0, 0)
        self._properties_layout.insertWidget(1, self.property_form)

    def _clear_property_form(self) -> None:
        if self.selected_object_label is not None:
            self._properties_layout.removeWidget(self.selected_object_label)
            self.selected_object_label.deleteLater()
            self.selected_object_label = None
        if self.property_form is None:
            return
        self._properties_layout.removeWidget(self.property_form)
        self.property_form.deleteLater()
        self.property_form = None

    def _iter_items(self) -> Iterator[QTreeWidgetItem]:
        for index in range(self.tree_widget.topLevelItemCount()):
            parent = self.tree_widget.topLevelItem(index)
            if parent is None:
                continue
            yield parent
            for child_index in range(parent.childCount()):
                child = parent.child(child_index)
                if child is not None:
                    yield child
