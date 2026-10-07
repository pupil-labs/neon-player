from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QWidget, QTabWidget, QVBoxLayout, QDockWidget


class PanelContainer(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self.area = QMainWindow(self)
        self.area.setDockNestingEnabled(True)
        self.area.setTabPosition(Qt.DockWidgetArea.AllDockWidgetAreas, QTabWidget.TabPosition.North)
        self.area.setWindowFlags(Qt.WindowType.Widget)

        assert not self.area.isWindow()
        self.area.setDockOptions(
            QMainWindow.DockOption.AllowNestedDocks
            | QMainWindow.DockOption.AllowTabbedDocks
            | QMainWindow.DockOption.AnimatedDocks
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.area)

    def add_panel(self, widget: QWidget, title: str):
        dock = QDockWidget(title, self.area)
        dock.setObjectName(title.lower().replace(" ", "_"))
        dock.setWidget(widget)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        dock.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self.area.addDockWidget(
            Qt.DockWidgetArea.LeftDockWidgetArea, dock
        )
        dock.show()
