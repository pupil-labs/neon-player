import os
from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QIcon, QCursor
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton


class UpdatePill(QFrame):
    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.release_url = ""
        self.setObjectName("UpdatePill")
        self.setFixedHeight(28)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        self.setStyleSheet(
            """
            #UpdatePill {
                background-color: #6d7be0;
                border-radius: 12px;
                border: none;
                margin-top: 2px;
                margin-bottom: 2px;
            }
            #UpdatePill:hover {
                background-color: #5a66b9;
            }
        """
        )

        self.layout_ = QHBoxLayout(self)
        self.layout_.setContentsMargins(12, 0, 12, 0)
        self.layout_.setSpacing(8)

        # Package icon
        self.icon_label = QLabel()
        icon_path = os.path.join(os.path.dirname(__file__), "..", "assets", "package.svg")
        self.icon_label.setPixmap(QIcon(icon_path).pixmap(14, 14))
        self.icon_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.layout_.addWidget(self.icon_label)

        # Text label (matching autoupdate branch style)
        self.label = QLabel("Update available")
        self.label.setStyleSheet("color: white; font-size: 11pt; background: transparent;")
        self.label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.layout_.addWidget(self.label)

        self.dismiss_btn = QPushButton("✕")
        self.dismiss_btn.setFixedSize(16, 16)
        self.dismiss_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.dismiss_btn.setStyleSheet(
            "QPushButton { background: transparent; color: rgba(255, 255, 255, 0.7); "
            "border: none; font-size: 10pt; padding: 0px; margin: 0px; }"
            "QPushButton:hover { color: white; }"
        )
        self.dismiss_btn.clicked.connect(self.hide)
        self.layout_.addWidget(self.dismiss_btn)

        self.clicked.connect(self._on_action_clicked)
        self.hide()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
        else:
            super().mousePressEvent(event)

    def show_update(self, version_tag: str, release_url: str):
        self.release_url = release_url
        self.label.setText(f"There is a new update available {version_tag}")
        tooltip = f"Open {version_tag} on GitHub ({release_url})"
        self.setToolTip(tooltip)
        self.show()

    def _on_action_clicked(self):
        if self.release_url:
            QDesktopServices.openUrl(QUrl(self.release_url))
