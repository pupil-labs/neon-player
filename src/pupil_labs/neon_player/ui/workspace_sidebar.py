from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QToolButton, QWidget, QVBoxLayout, QLabel, QTableWidgetItem, QAbstractItemView, QHeaderView
)

from pupil_labs import neon_player
from pupil_labs.neon_player import asset_path
from pupil_labs.neon_player.workspace import RecordingMetadata
from pupil_labs.neon_player.ui.components import HoverRowTable, create_heading_with_icon
from pupil_labs.neon_recording import NeonRecording


class WorkspaceSidebar(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("WorkspaceSidebar")

        self._column_field_mapping = {
            "Recording name": ("name", lambda name: f"  {name}"),
            "Duration": ("duration", lambda dur: str(dur)),
            "Wearer": ("wearer", None)
        }
        self._column_names = list(self._column_field_mapping.keys())

        self.icon_path_collapsed = str(asset_path("chevron_forward.svg"))
        self.icon_path_expanded = str(asset_path("chevron_back.svg"))
        self.width_collapsed = 50
        self.width_expanded = 400

        self._collapsed = False
        self.toggle_button = QToolButton(self)
        self.toggle_button.setObjectName("WorkspaceSidebarToggle")
        self.toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_button.setIcon(QIcon(self.icon_path_expanded))
        self.toggle_button.setIconSize(QSize(24, 24))
        self.toggle_button.clicked.connect(self.toggle_collapse)

        self.recordings_table = HoverRowTable(self)
        self.recordings_table.setColumnCount(len(self._column_names))
        self.recordings_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.recordings_table.setFocusPolicy(Qt.NoFocus)
        self.recordings_table.setHorizontalHeaderLabels(self._column_names)
        self.recordings_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.recordings_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.recordings_table.setShowGrid(False)
        self.recordings_table.setIconSize(QSize(80, 40))
        self.recordings_table.cellClicked.connect(self.on_table_cell_clicked)

        horiz_header = self.recordings_table.horizontalHeader()
        horiz_header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        horiz_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        horiz_header.setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        horiz_header.setCursor(Qt.CursorShape.PointingHandCursor)

        vert_header = self.recordings_table.verticalHeader()
        vert_header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        vert_header.setVisible(False)

        workspace_layout = create_heading_with_icon(
            "Workspace",
            str(asset_path("workspace.svg")),
            heading_level=3,
            icon_size=(16, 16)
        )
        self.workspace_heading = QWidget(self)
        self.workspace_heading.setContentsMargins(0, 0, 0, 0)
        self.workspace_heading.setLayout(workspace_layout)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.addWidget(self.toggle_button, alignment=Qt.AlignmentFlag.AlignRight)
        self.main_layout.addWidget(self.workspace_heading)
        self.main_layout.addWidget(self.recordings_table)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.setLayout(self.main_layout)
        self.setMinimumSize(self.width_expanded, 100)

    def on_recording_loaded(self, recording: NeonRecording) -> None:
        recording_names = [
            self.recordings_table.item(row, 0).text().strip()
            for row in range(self.recordings_table.rowCount())
        ]
        recording_index = recording_names.index(recording._rec_dir.name)
        self.recordings_table.setCurrentCell(recording_index, 0)

    def on_table_cell_clicked(self, row: int, column: int) -> None:
        app = neon_player.instance()
        recording_name = self.recordings_table.item(row, 0).text()
        recording_path = app.workspace.get_recording_path(recording_name)
        app.load_recording(recording_path)

    def update_recording_table(self, recording_list: list[RecordingMetadata]) -> None:
        self.recordings_table.clearContents()
        self.recordings_table.setSortingEnabled(False)

        self.recordings_table.setRowCount(len(recording_list))
        for i_row, recording in enumerate(recording_list):
            for i_col, (field, formatter) in enumerate(
                self._column_field_mapping.values()
            ):
                value = getattr(recording, field)
                if formatter is not None:
                    value = formatter(value)

                item = QTableWidgetItem(str(value))
                if field == "name" and recording.thumbnail_path.exists():
                    item.setIcon(QIcon(str(recording.thumbnail_path)))
                self.recordings_table.setItem(i_row, i_col, item)

        self.recordings_table.resizeColumnsToContents()
        self.recordings_table.setSortingEnabled(True)
        self.recordings_table.sortByColumn(0, Qt.SortOrder.AscendingOrder)

    def toggle_collapse(self) -> None:
        if self._collapsed:
            self.expand()
        else:
            self.collapse()

    def collapse(self) -> None:
        if self._collapsed:
            return

        self.toggle_button.setIcon(QIcon(self.icon_path_collapsed))
        self.workspace_heading.hide()
        self.recordings_table.hide()
        self.main_layout.addStretch()
        self.setMinimumWidth(self.width_collapsed)
        self.setMaximumWidth(self.width_collapsed)
        self._collapsed = True

    def expand(self) -> None:
        if not self._collapsed:
            return

        self.toggle_button.setIcon(QIcon(self.icon_path_expanded))
        self.workspace_heading.show()
        self.recordings_table.show()
        self.main_layout.takeAt(self.main_layout.count() - 1)
        self.setMinimumWidth(self.width_expanded)
        self._collapsed = False
