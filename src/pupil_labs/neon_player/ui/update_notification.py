import webbrowser

from pathlib import Path
from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QIcon, QCursor
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QPushButton,
    QToolButton,
    QWidget,
)

from pupil_labs.neon_player import asset_path


class NotificationPill(QFrame):
    clicked = Signal()
    dismissed = Signal()

    def __init__(
        self,
        color: str = "#6d7be0",
        hover_color: str = "#5a66b9",
        icon_path: str | Path | None = None,
        parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.action_url = ""
        self.setObjectName("NotificationPill")
        self.setFixedHeight(28)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        self.setStyleSheet(
            f"""
            QFrame[objectName="NotificationPill"] {{
                background-color: {color};
                border-radius: 12px;
                border: none;
            }}
        """
        )

        self.layout_ = QHBoxLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.layout_.setSpacing(0)

        self.action_btn = QPushButton()
        self.action_btn.setObjectName("NotificationAction")
        self.action_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.action_btn.setFixedHeight(self.height())
        self.action_btn.setStyleSheet(
            f"""
            QPushButton[objectName="NotificationAction"] {{
                background: transparent;
                border: none;
                border-top-left-radius: 12px;
                border-bottom-left-radius: 12px;
                color: white;
                font-size: 11pt;
                padding: 0px;
                padding-left: 12px;
                padding-right: 6px;
                margin: 0px;
            }}
            QPushButton[objectName="NotificationAction"]:hover {{
                background-color: {hover_color};
            }}
            """
        )
        if icon_path:
            self.action_btn.setIcon(QIcon(str(icon_path)))
        self.action_btn.clicked.connect(self.clicked.emit)
        self.layout_.addWidget(self.action_btn, 1)

        self.dismiss_btn = QToolButton()
        self.dismiss_btn.setFixedSize(28, self.height())
        self.dismiss_btn.setIcon(QIcon(str(asset_path("close_white.svg"))))
        self.dismiss_btn.setIconSize(QSize(20, 20))
        self.dismiss_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.dismiss_btn.setStyleSheet(
            f"""
            QToolButton {{
                background: transparent;
                color: rgba(255, 255, 255, 0.7);
                border: none;
                border-top-right-radius: 12px;
                border-bottom-right-radius: 12px;
                font-size: 10pt;
                padding: 0px;
                margin: 0px;
            }}
            QToolButton:hover {{
                color: white;
                background-color: {hover_color};
            }}
            """
        )
        self.dismiss_btn.clicked.connect(self._on_dismiss)
        self.dismiss_btn.setToolTip("Hide this notification")
        self.layout_.addWidget(self.dismiss_btn)

        self.clicked.connect(self._on_action_clicked)
        self.hide()

    def show_pill(self, text: str, url: str, tooltip: str = "") -> None:
        self.action_url = url
        self.action_btn.setText("  " + text)
        if tooltip:
            self.action_btn.setToolTip(tooltip)
        self.show()

    def _on_action_clicked(self) -> None:
        if self.action_url:
            webbrowser.open(self.action_url)

    def _on_dismiss(self) -> None:
        self.hide()
        self.dismissed.emit()


class UpdatePill(NotificationPill):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(icon_path=asset_path("package.svg"), parent=parent)
        self.version_tag = ""

    def show_update(self, version_tag: str, release_url: str) -> None:
        self.version_tag = version_tag
        text = f"Update available: {version_tag}"
        tooltip = f"Open v{version_tag} release on GitHub ({release_url})"
        self.show_pill(text, release_url, tooltip)
