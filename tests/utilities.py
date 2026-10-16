from PySide6.QtWidgets import QApplication, QMessageBox


def retrieve_messagebox() -> QMessageBox | None:
    top_widgets = QApplication.topLevelWidgets()
    return next((w for w in top_widgets if isinstance(w, QMessageBox)), None)
    