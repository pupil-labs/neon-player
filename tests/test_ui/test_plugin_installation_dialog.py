from unittest.mock import Mock

import pytest
from PySide6.QtWidgets import QDialogButtonBox
from pytestqt.qtbot import QtBot

from pupil_labs.neon_player.ui import plugin_installation_dialog
from pupil_labs.neon_player.ui.plugin_installation_dialog import (
    DependencyInstallationDialog,
)


def test_dialog_message(qtbot: QtBot) -> None:
    dialog = DependencyInstallationDialog(
        ["example==2.0"], "Dependency title", "Dependencies need updating."
    )
    qtbot.addWidget(dialog)

    assert dialog.windowTitle() == "Dependency title"
    assert dialog.label.text() == (
        "Dependencies need updating. Additional dependencies may be installed as well."
    )
    assert "example==2.0" in dialog.dependency_bullets.text()


def test_start_installation(qtbot: QtBot, monkeypatch: pytest.MonkeyPatch) -> None:
    app = Mock()
    monkeypatch.setattr(plugin_installation_dialog.neon_player, "instance", lambda: app)
    dialog = DependencyInstallationDialog(
        ["example==2.0"], "Dependency title", "Dependencies need updating."
    )
    qtbot.addWidget(dialog)

    dialog.button_box.button(QDialogButtonBox.StandardButton.Ok).click()

    app.job_manager.run_background_action.assert_called_once_with(
        "Install dependencies", "install_packages", "example==2.0", save_settings=False
    )
    assert dialog.progress_bar.minimum() == 0
    assert dialog.progress_bar.maximum() == 0
    assert not dialog.button_box.isEnabled()
    app.job_manager.run_background_action.return_value.finished.connect.assert_called_once_with(
        dialog.close
    )
