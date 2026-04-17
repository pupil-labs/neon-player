from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTableWidgetItem, QAbstractItemView, QHeaderView
)

from pupil_labs import neon_player
from pupil_labs.neon_player.workspace import RecordingDescription
from pupil_labs.neon_player.ui.components import HoverRowTable
from pupil_labs.neon_recording import NeonRecording


class ProjectSidebar(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self._columns = dict(
            name="Recording name",
            duration="Duration",
            recorded="Recorded",
            wearer="Wearer"
        )

        self.recordings_table = HoverRowTable(self)
        self.recordings_table.setColumnCount(len(self._columns))
        self.recordings_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.recordings_table.setFocusPolicy(Qt.NoFocus)
        self.recordings_table.setHorizontalHeaderLabels(self._columns.values())
        self.recordings_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.recordings_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.recordings_table.cellClicked.connect(self.on_table_cell_clicked)

        horiz_header = self.recordings_table.horizontalHeader()
        horiz_header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        horiz_header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        horiz_header.setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        horiz_header.setCursor(Qt.CursorShape.PointingHandCursor)

        vert_header = self.recordings_table.verticalHeader()
        vert_header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        vert_header.setVisible(False)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(QLabel("Workspace"))
        main_layout.addWidget(self.recordings_table)
        main_layout.setContentsMargins(10, 10, 10, 10)
        self.setLayout(main_layout)
        self.setMinimumSize(400, 100)

    def on_recording_loaded(self, recording: NeonRecording) -> None:
        pass

    def on_table_cell_clicked(self, row: int, column: int) -> None:
        app = neon_player.instance()
        recording_name = self.recordings_table.item(row, 0).text()
        recording_path = app.workspace.get_recording_path(recording_name)
        app.load(recording_path)

    def update_recording_table(self, recording_list: list[RecordingDescription]) -> None:
        self.recordings_table.clearContents()
        self.recordings_table.setSortingEnabled(False)

        self.recordings_table.setRowCount(len(recording_list))
        for i_row, recording in enumerate(recording_list):
            for i_col, field in enumerate(self._columns):
                value = str(getattr(recording, field))
                self.recordings_table.setItem(i_row, i_col, QTableWidgetItem(value))

        self.recordings_table.resizeColumnsToContents()
        self.recordings_table.setSortingEnabled(True)
