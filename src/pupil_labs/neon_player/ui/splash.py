from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import (
    QDragEnterEvent,
    QDragLeaveEvent,
    QDropEvent,
    QIcon,
    QPixmap,
)
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from pupil_labs import neon_player
from pupil_labs.neon_player import asset_path


class SplashWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self.setObjectName("Splash")
        self.resize(1147, 738)
        self.setWindowTitle("Form")
        self.setStyleSheet(
            "#content { background: #000000; }\n"
            "#dropbox { border: 1px dotted #ccc; border-radius: 5px; }"
        )
        self.setAcceptDrops(True)

        root_layout = QGridLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        content = QWidget(self)
        content.setObjectName("content")
        content.setMinimumHeight(300)
        content_layout = QVBoxLayout(content)
        content_layout.addStretch(1)

        welcome = QLabel("Welcome to Neon Player!", content)
        welcome.setStyleSheet(
            "font-size: 30pt; font-weight: bold; margin-bottom: 20px;"
        )
        welcome.setAlignment(
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter
        )
        content_layout.addWidget(welcome)

        recording_row = QHBoxLayout()
        recording_row.addStretch(1)

        self.dropbox = QWidget(content)
        self.dropbox.setObjectName("dropbox")
        self.dropbox.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        self.dropbox.setMaximumWidth(870)
        self.dropbox.setStyleSheet(
            "#dropbox { background: #080808; border: 1px dotted #ccc; "
            "border-radius: 5px; }"
        )
        dropbox_layout = QVBoxLayout(self.dropbox)
        dropbox_layout.setSpacing(10)
        dropbox_layout.setContentsMargins(10, 30, 10, 30)

        dropbox_layout.addWidget(
            QLabel("Drop a recording folder here or", self.dropbox),
            alignment=Qt.AlignmentFlag.AlignCenter,
        )

        self.browse_button = QPushButton(self.dropbox)
        self.browse_button.setObjectName("browse_button")
        self.browse_button.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed
        )
        self.browse_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.browse_button.setText(" Choose a recording folder")
        self.browse_button.setIcon(QIcon.fromTheme("folder"))
        dropbox_layout.addWidget(
            self.browse_button, alignment=Qt.AlignmentFlag.AlignHCenter
        )
        recording_row.addWidget(self.dropbox, 2)
        recording_row.addStretch(1)
        content_layout.addLayout(recording_row)

        recent_row = QHBoxLayout()
        recent_row.setContentsMargins(0, 20, 0, 0)
        recent_row.addStretch()
        self.recent_button = QPushButton(" Show recently opened recordings", content)
        self.recent_button.setObjectName("RecentButton")
        self.recent_button.setIconSize(QSize(24, 24))
        self.recent_button.setFlat(True)
        self.recent_button.setIcon(QIcon(str(asset_path("recent.svg"))))
        self.recent_button.setCursor(Qt.CursorShape.PointingHandCursor)
        recent_row.addWidget(self.recent_button)
        recent_row.addStretch()
        content_layout.addLayout(recent_row)

        self.logo = QLabel(content)
        self.logo.setPixmap(QPixmap(asset_path("Primary-White-76px.png")))
        self.logo.setObjectName("logo")
        self.logo.setAlignment(
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter
        )

        content_layout.addStretch(1)
        content_layout.addWidget(self.logo)
        root_layout.addWidget(content)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1 and urls[0].isLocalFile() and urls[0].toLocalFile():
                path = Path(urls[0].toLocalFile())
                if path.is_dir():
                    event.acceptProposedAction()
                    self.dropbox.setStyleSheet("#dropbox { background: #141414 }")
                    return
        event.ignore()

    def dragLeaveEvent(self, _event: QDragLeaveEvent) -> None:
        self.dropbox.setStyleSheet("#dropbox { background: #080808 }")

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            path = Path(urls[0].toLocalFile())
            if path.is_dir():
                self.dropbox.setStyleSheet("#dropbox { background: #080808 }")
                neon_player.instance().load(path)
                event.acceptProposedAction()
                return
        event.ignore()
