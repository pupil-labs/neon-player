from PySide6.QtGui import QTextCursor
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
        self.markdown_view.document().setDocumentMargin(16)
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
        self._update_block_margins()

    def _update_block_margins(self):
        block = self.markdown_view.document().begin()
        while block.isValid():
            block_format = block.blockFormat()
            if block_format.headingLevel() > 0:
                block_format.setTopMargin(24)
                block_format.setBottomMargin(12)
                QTextCursor(block).setBlockFormat(block_format)
            block = block.next()
