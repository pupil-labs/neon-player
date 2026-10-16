import pytest

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QMessageBox

from pupil_labs.neon_player.ui.main_window import MainWindow
from pupil_labs.neon_player.ui.update_notification import DismissedUpdatePrompt

from tests.utilities import retrieve_messagebox


def test_main_window_on_update_available(qtbot):
    main_window = MainWindow()
    qtbot.addWidget(main_window)

    assert main_window.statusBar().isHidden()
    assert main_window.update_pill.isHidden()

    with qtbot.waitExposed(main_window):
        def shows_update_pill():
            assert main_window.statusBar().isVisible()
            assert main_window.update_pill.isVisible()

        main_window.show()
        main_window.on_update_available("7.0.0", "url")
        qtbot.waitUntil(shows_update_pill)


def test_main_window_dismissed_update_sets_last_displayed_update_version(qtbot, qapp):
    main_window = MainWindow()
    qtbot.addWidget(main_window)

    main_window.update_pill.version_tag = "7.0.0"
    qapp.settings.last_displayed_update_version = "6.0.0"
    qapp.settings.show_prompt_on_dismissed_update = False

    main_window.prompt_dismiss_updates()
    assert qapp.settings.last_displayed_update_version == "7.0.0"


@pytest.mark.parametrize("set_checkbox", [True, False])
def test_main_window_dismissed_update_not_show_message_again(set_checkbox, qtbot, qapp):
    main_window = MainWindow()
    qtbot.addWidget(main_window)

    main_window.update_pill.version_tag = "7.0.0"
    qapp.settings.last_displayed_update_version = "6.0.0"

    def check_messagebox():
        msg_box = retrieve_messagebox()
        assert isinstance(msg_box, DismissedUpdatePrompt)

        if set_checkbox:
            msg_box.cb.setChecked()
        qtbot.mouseClick(
            msg_box.button(QMessageBox.StandardButton.Ok), 
            Qt.MouseButton.LeftButton
        )
        assert qapp.settings.show_prompt_on_dismissed_update ^ set_checkbox

    main_window.prompt_dismiss_updates()
    QTimer.singleShot(100, check_messagebox)
