from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QDockWidget, QMainWindow, QVBoxLayout, QWidget

from pupil_labs.neon_player.managers.panel import Panel, PanelLayoutBackend, _require_gui_thread


class _PanelDock(QDockWidget):
    def __init__(self, panel: Panel, backend: PanelLayoutBackend) -> None:
        super().__init__(panel.title, backend.widget)
        self._panel_id = panel.panel_id
        self._closable = panel.closable
        self._backend = backend
        self.setObjectName(panel.panel_id)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            if panel.closable
            else QDockWidget.DockWidgetFeature.NoDockWidgetFeatures
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        event.ignore()
        if self._closable:
            self._backend.close_requested.emit(self._panel_id)


class QtPanelLayoutBackend(PanelLayoutBackend):
    """Default adapter respecting panel user-close policy.

    Floating and arrangement are not enabled yet.
    """

    def __init__(self) -> None:
        _require_gui_thread()
        super().__init__()
        self._host = QMainWindow()
        self._docks: dict[str, _PanelDock] = {}

    @property
    def widget(self) -> QWidget:
        return self._host

    def attach(self, panel: Panel) -> None:
        _require_gui_thread()
        if panel.panel_id in self._docks:
            raise ValueError(f"Duplicate panel ID: {panel.panel_id}")
        dock = _PanelDock(panel, self)
        original_parent = panel.widget.parentWidget()
        original_flags = panel.widget.windowFlags()
        was_visible = panel.widget.isVisible()
        try:
            dock.setWidget(panel.widget)
            self._host.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        except Exception:
            panel.widget.setParent(original_parent, original_flags)
            panel.widget.setVisible(was_visible)
            self._host.removeDockWidget(dock)
            dock.hide()
            dock.deleteLater()
            raise
        self._docks[panel.panel_id] = dock

    def detach(self, panel_id: str) -> None:
        _require_gui_thread()
        dock = self._docks[panel_id]
        widget = dock.widget()
        if widget is None:
            raise RuntimeError(f"Panel dock has no visualization widget: {panel_id}")
        dock_was_visible = dock.isVisible()
        widget_was_visible = widget.isVisible()
        try:
            self._host.removeDockWidget(dock)
            widget.hide()
            widget.setParent(None)
            dock.hide()
        except Exception:
            dock.setWidget(widget)
            self._host.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
            dock.setVisible(dock_was_visible)
            widget.setVisible(widget_was_visible)
            raise
        del self._docks[panel_id]
        dock.deleteLater()


class PanelContainer(QWidget):
    """Host widget owned by the main window, separate from manager lifetime.

    The owner must shut down the manager before destroying this container.
    Hiding the container does not remove its panels or shut down the manager.
    """

    def __init__(
        self, backend: PanelLayoutBackend, parent: QWidget | None = None
    ) -> None:
        _require_gui_thread()
        if backend.widget.parent() is not None:
            raise ValueError("Layout host already belongs to a container")
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(backend.widget)
