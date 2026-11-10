import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QMenu
from pytestqt.qtbot import QtBot

from pupil_labs.neon_player.ui.main_window import MainWindow
from tests.conftest import MockNeonPlayerApp


def test_main_window_on_update_available(qtbot: QtBot) -> None:
    main_window = MainWindow()
    qtbot.addWidget(main_window)

    assert main_window.statusBar().isHidden()
    assert main_window.update_pill.isHidden()

    with qtbot.waitExposed(main_window):

        def shows_update_pill() -> None:
            assert main_window.statusBar().isVisible()
            assert main_window.update_pill.isVisible()

        main_window.show()
        main_window.on_update_available("7.0.0", "url")
        qtbot.waitUntil(shows_update_pill)


def test_main_window_dismissed_update_records_version(
    qtbot: QtBot,
    qapp: MockNeonPlayerApp,
) -> None:
    main_window = MainWindow()
    qtbot.addWidget(main_window)

    qapp.settings.last_displayed_update_version = "6.0.0"
    with qtbot.waitExposed(main_window):
        main_window.show()
        main_window.on_update_available("7.0.0", "url")

    qtbot.mouseClick(main_window.update_pill.dismiss_btn, Qt.MouseButton.LeftButton)

    assert main_window.update_pill.isHidden()
    assert qapp.settings.last_displayed_update_version == "7.0.0"


def test_menu_lookup_keeps_missing_and_existing_menus_distinct(qtbot: QtBot) -> None:
    window = MainWindow.__new__(MainWindow)
    QMainWindow.__init__(window)
    qtbot.addWidget(window)
    assert window.get_menu("Timeline", auto_create=False) is None
    menu = window.get_menu("Timeline")
    assert isinstance(menu, QMenu)
    assert window.get_menu("Timeline", auto_create=False) is menu


def test_status_log_handler_removed_when_label_is_destroyed(qtbot: QtBot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    handler = window.log_status_handler
    root_logger = logging.getLogger()
    assert handler in root_logger.handlers

    window.status_label.deleteLater()
    qtbot.waitUntil(lambda: handler not in root_logger.handlers)
    root_logger.warning("Logging remains usable after status label destruction")
