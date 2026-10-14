from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


class ReleaseNotesDialog(QDialog):
    def __init__(self, parent: QWidget | None = None, markdown: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Release Notes")
        self.resize(800, 600)

        self.markdown_view = QTextBrowser(self)
        self.markdown_view.setMarkdown(markdown)
        self.markdown_view.setOpenExternalLinks(True)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(self.markdown_view)
        layout.addWidget(buttons)

    def set_markdown(self, markdown: str) -> None:
        self.markdown_view.setMarkdown(markdown)
