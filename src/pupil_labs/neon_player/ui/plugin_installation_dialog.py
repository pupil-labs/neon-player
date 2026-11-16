import logging

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from pupil_labs import neon_player

logger = logging.getLogger(__name__)


class DependencyInstallationDialog(QDialog):
    def __init__(
        self,
        dependencies_to_install: list[str],
        title: str,
        base_message: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.dependencies_to_install = dependencies_to_install

        self.setWindowTitle(title)
        self.setMinimumWidth(450)

        layout = QVBoxLayout()
        layout.setSpacing(10)
        self.setLayout(layout)

        self.label = QLabel(
            base_message + " Additional dependencies may be installed as well."
        )
        self.label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        self.label.setWordWrap(True)
        layout.addWidget(self.label)

        self.dependency_bullets = QLabel(
            "<html><ul>"
            + "\n".join([f"<li>{dep}</li>" for dep in dependencies_to_install])
            + "</ul></html>"
        )
        layout.addWidget(self.dependency_bullets)

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            background: transparent;
            border: none;
        """)
        layout.addStretch()
        layout.addWidget(self.progress_bar)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Install")
        self.button_box.accepted.connect(self.start_installation)
        self.button_box.rejected.connect(self.cancel_installation)
        layout.addWidget(self.button_box)

    def start_installation(self) -> None:
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.button_box.setEnabled(False)

        # NOTE: package installation job is started before settings are fully
        # initialized, so it should not save settings before starting
        self.job = neon_player.instance().job_manager.run_background_action(
            "Install dependencies",
            "install_packages",
            *self.dependencies_to_install,
            save_settings=False,
        )
        self.job.finished.connect(self.close)

    def cancel_installation(self) -> None:
        logger.debug("Dependency installation was cancelled")
        self.reject()
