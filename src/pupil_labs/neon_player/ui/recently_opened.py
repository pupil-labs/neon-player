from PySide6.QtCore import QRect, Qt, Signal, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QLabel,
    QSizePolicy,
    QHeaderView,
    QGridLayout,
)

from pathlib import Path
from pupil_labs import neon_player
from pupil_labs.neon_player import asset_path
from pupil_labs.neon_player.ui.components import create_heading_with_icon, HoverRowTable


class RecentWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("RecentWidget")
        self.setStyleSheet("#content { background: #121212; }")

        recording_heading = create_heading_with_icon(
            "Recently opened",
            str(asset_path("recent.svg"))
        )
        self.no_recording_history_label = self._create_empty_history_label("recordings")
        self.recording_columns = {
            "Recording name": "name",
            "Wearer": "wearer",
            "Last opened": "last_opened",
            "Recorded": "recorded",
            "Path": "path"
        }
        self.recording_table = self._create_history_table(self.recording_columns)
        self.recording_table.cellClicked.connect(self.on_recording_table_cell_clicked)

        workspace_heading = create_heading_with_icon(
            "Workspaces",
            str(asset_path("workspace.svg"))
        )
        self.no_workspace_history_label = self._create_empty_history_label("workspaces")
        self.workspace_columns = {
            "Workspace name": "name",
            "Last opened": "last_opened",
            "Path": "path"
        }
        self.workspace_table = self._create_history_table(self.workspace_columns)
        self.workspace_table.cellClicked.connect(self.on_workspace_table_cell_clicked)

        self.container = QWidget(self)
        self.container.setObjectName("content")

        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(50, 50, 50, 50)
        layout.addLayout(recording_heading)
        layout.addWidget(self.no_recording_history_label)
        layout.addWidget(self.recording_table)
        layout.addSpacing(20)
        layout.addLayout(workspace_heading)
        layout.addWidget(self.no_workspace_history_label)
        layout.addWidget(self.workspace_table)

        self.grid_layout = QGridLayout(self)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.addWidget(self.container, 0, 0, 1, 1)
        self.setLayout(self.grid_layout)

        app = neon_player.instance()
        app.load_history.changed.connect(self.update_load_history)

    def fit_rect(self, rect: QRect) -> None:
        w, h = rect.width(), rect.height()
        self.setGeometry(QRect(0, 0, w // 2, h))

    def _slide_animation(self, kind: str, rect: QRect, duration: int = 250) -> QPropertyAnimation:
        w, h = rect.width(), rect.height()
        half_w = w // 2

        curve_type = QEasingCurve.Type.OutCubic if kind == "in" else QEasingCurve.Type.InCubic
        start_x = -half_w if kind == "in" else 0
        end_x = 0 if kind == "in" else -half_w

        anim = QPropertyAnimation(self, b"geometry")
        anim.setDuration(duration)
        anim.setEasingCurve(curve_type)
        anim.setStartValue(QRect(start_x, 0, half_w, h))
        anim.setEndValue(QRect(end_x, 0, half_w, h))
        return anim

    def slide_in_animation(self, rect: QRect, duration: int = 250) -> QPropertyAnimation:
        return self._slide_animation("in", rect, duration)

    def slide_out_animation(self, rect: QRect, duration: int = 250) -> QPropertyAnimation:
        return self._slide_animation("out", rect, duration)

    def update_load_history(self) -> None:
        app = neon_player.instance()

        recent_recordings = app.load_history.recent_recordings.items()
        self._update_recent_items(
            self.recording_table,
            self.recording_columns,
            recent_recordings,
            self.no_recording_history_label
        )

        recent_workspaces = app.load_history.recent_workspaces.items()
        self._update_recent_items(
            self.workspace_table,
            self.workspace_columns,
            recent_workspaces,
            self.no_workspace_history_label
        )

    @staticmethod
    def _update_recent_items(
        table: QTableWidget,
        columns: dict[str, str],
        recent_items: list[tuple[str, dict]],
        no_history_label: QLabel,
    ) -> None:
        """
        Update the table displaying recently opened recordings or workspaces.
        If there are no recent items, hide the table and show a label instead.
        """
        table.setSortingEnabled(False)
        table.clearContents()

        if not recent_items:
            table.setVisible(False)
            no_history_label.setVisible(True)
            return

        no_history_label.setVisible(False)
        table.setVisible(True)
        table.setRowCount(len(recent_items))
        for row, (path, info) in enumerate(recent_items):
            for col, field_name in enumerate(columns.values()):
                if field_name == "path":
                    item_path = QTableWidgetItem(path)
                    item_path.setForeground(QColor("#666"))
                    item_path.setToolTip(path)
                    table.setItem(row, col, item_path)
                    continue

                item_text = info.get(field_name, "-")
                item = QTableWidgetItem(item_text)
                item.setForeground(QColor("#ededef"))
                if field_name == "name":
                    item.setData(Qt.ItemDataRole.UserRole, path)
                    item.setForeground(QColor("#6d7be0"))
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                table.setItem(row, col, item)

        last_opened_col_index = list(columns.keys()).index("Last opened")
        table.setSortingEnabled(True)
        table.sortByColumn(last_opened_col_index, Qt.SortOrder.DescendingOrder)

    def on_recording_table_cell_clicked(self, row: int, column: int) -> None:
        self._on_table_cell_clicked(self.recording_table, row, column)

    def on_workspace_table_cell_clicked(self, row: int, column: int) -> None:
        self._on_table_cell_clicked(self.workspace_table, row, column)

    @staticmethod
    def _on_table_cell_clicked(table: QTableWidget, row: int, column: int) -> None:
        item = table.item(row, 0)
        if item is None:
            return

        path_str = item.data(Qt.ItemDataRole.UserRole)
        if not path_str:
            return

        neon_player.instance().load(Path(path_str))

    @staticmethod
    def _create_empty_history_label(entity: str) -> QLabel:
        label = QLabel(f"Recently opened {entity} will appear here.")
        label.setAlignment(Qt.AlignmentFlag.AlignTop)
        label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )
        label.setVisible(False)
        return label

    def _create_history_table(self, columns: dict[str, str]) -> QTableWidget:
        num_columns = len(columns)

        table = HoverRowTable(self)
        table.setColumnCount(num_columns)
        table.setHorizontalHeaderLabels(list(columns.keys()))

        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        table.setShowGrid(False)
        table.setWordWrap(False)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        horiz_header = table.horizontalHeader()
        horiz_header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        horiz_header.setSectionResizeMode(num_columns - 1, QHeaderView.ResizeMode.Stretch)
        horiz_header.setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        horiz_header.setCursor(Qt.CursorShape.PointingHandCursor)

        vert_header = table.verticalHeader()
        vert_header.setVisible(False)

        return table


class RecentWidgetBackdrop(QWidget):
    """
    This widget is used as a backdrop for the RecentWidget, providing a semi-transparent
    overlay effect and catching mouse events to prevent interaction with underlying widgets.
    """
    clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 0.5);")

    def fit_rect(self, rect: QRect) -> None:
        self.setGeometry(rect)

    def mousePressEvent(self, event) -> None:
        # Override mouse press event to prevent interaction with underlying widgets
        event.accept()

        if event.button() != Qt.MouseButton.LeftButton:
            return

        self.clicked.emit()
