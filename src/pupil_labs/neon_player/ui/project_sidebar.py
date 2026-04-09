from pathlib import Path

from PySide6.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QLabel, QTableWidgetItem

from pupil_labs.neon_player.project import RecordingDescription, get_recording_list


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
        self.recordings_table.setHorizontalHeaderLabels(self._columns.values())
        main_layout.addWidget(self.recordings_table)

    def update_recording_table(self, recording_list: list[RecordingDescription]) -> None:
        self.recordings_table.clearContents()
        self.recordings_table.setSortingEnabled(False)

        self.recordings_table.setRowCount(len(recording_list))
        for i_row, recording in enumerate(recording_list):
            for i_col, field in enumerate(self._columns):
                value = str(getattr(recording, field))
                self.recordings_table.setItem(i_row, i_col, QTableWidgetItem(value))

        self.recordings_table.setSortingEnabled(True)
