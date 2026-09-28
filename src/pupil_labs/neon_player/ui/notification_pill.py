import os
from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QIcon, QCursor
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton


class NotificationPill(QFrame):
    clicked = Signal()
    dismissed = Signal()

    def __init__(self, color="#6d7be0", hover_color="#5a66b9", icon_path=None, parent=None):
        super().__init__(parent)
        self.action_url = ""
        # Create a unique object name so styles apply independently if needed
        self.setObjectName("NotificationPill")
        self.setFixedHeight(28)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        self.setStyleSheet(
            f"""
            QFrame[objectName="NotificationPill"] {{
                background-color: {color};
                border-radius: 12px;
                border: none;
                margin-top: 2px;
                margin-bottom: 2px;
            }}
            QFrame[objectName="NotificationPill"]:hover {{
                background-color: {hover_color};
            }}
        """
        )

        self.layout_ = QHBoxLayout(self)
        self.layout_.setContentsMargins(12, 0, 12, 0)
        self.layout_.setSpacing(8)

        # Package icon
        self.icon_label = QLabel()
        if icon_path:
            self.icon_label.setPixmap(QIcon(icon_path).pixmap(14, 14))
        self.icon_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.layout_.addWidget(self.icon_label)

        # Text label
        self.label = QLabel("")
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
        self.dismiss_btn.clicked.connect(self._on_dismiss)
        self.layout_.addWidget(self.dismiss_btn)

        self.clicked.connect(self._on_action_clicked)
        self.hide()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
        else:
            super().mousePressEvent(event)

    def show_pill(self, text: str, url: str, tooltip: str = ""):
        self.action_url = url
        self.label.setText(text)
        if tooltip:
            self.setToolTip(tooltip)
        self.show()

    def _on_action_clicked(self):
        if self.action_url:
            QDesktopServices.openUrl(QUrl(self.action_url))

    def _on_dismiss(self):
        self.hide()
        self.dismissed.emit()


class UpdatePill(NotificationPill):
    def __init__(self, parent=None):
        icon_path = os.path.join(os.path.dirname(__file__), "..", "assets", "package.svg")
        super().__init__(color="#6d7be0", hover_color="#5a66b9", icon_path=icon_path, parent=parent)

    def show_update(self, version_tag: str, release_url: str):
        # We also want it accessible via self.release_url for backwards compat
        self.release_url = release_url
        text = f"There is a new update available {version_tag}"
        tooltip = f"Open {version_tag} on GitHub ({release_url})"
        self.show_pill(text, release_url, tooltip)


class WhatsNewPill(NotificationPill):
    def __init__(self, parent=None):
        icon_path = os.path.join(os.path.dirname(__file__), "..", "assets", "sparkles.svg")
        super().__init__(color="#6d7be0", hover_color="#5a66b9", icon_path=icon_path, parent=parent)
        self.label.setText("See what's new!")
