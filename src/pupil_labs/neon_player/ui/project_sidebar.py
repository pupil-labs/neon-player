from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QLabel, QTableWidgetItem, QAbstractItemView
)

from pupil_labs import neon_player
from pupil_labs.neon_player.project import RecordingDescription, get_recording_list
from pupil_labs.neon_recording import NeonRecording


class ProjectSidebar(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self._columns = dict(
            name="Recording Name",
            duration="Duration",
            wearer="Wearer"
        )

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 0, 0, 0)
        self.setLayout(main_layout)
        self.setMinimumSize(400, 100)

        main_layout.addWidget(QLabel("Recordings"))
        self.recordings_table = QTableWidget(self)
        self.recordings_table.setColumnCount(len(self._columns))
        self.recordings_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.recordings_table.setHorizontalHeaderLabels(self._columns.values())
        self.recordings_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.recordings_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.recordings_table.verticalHeader().setVisible(False)
        self.recordings_table.itemSelectionChanged.connect(self.on_selection_changed)
        main_layout.addWidget(self.recordings_table)

    def on_recording_loaded(self, recording: NeonRecording) -> None:
        pass

    def on_selection_changed(self) -> None:
        selected_items = self.recordings_table.selectedItems()
        if not selected_items:
            return

        app = neon_player.instance()
        recording_name = selected_items[0].text()
        recording_path = app.project.get_recording_path(recording_name)
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
