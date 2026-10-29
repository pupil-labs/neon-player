from PySide6.QtWidgets import QMainWindow, QMenu
from pytestqt.qtbot import QtBot

from pupil_labs.neon_player.ui.main_window import MainWindow


def test_menu_lookup_keeps_missing_and_existing_menus_distinct(qtbot: QtBot) -> None:
    window = MainWindow.__new__(MainWindow)
    QMainWindow.__init__(window)
    qtbot.addWidget(window)
    assert window.get_menu("Timeline", auto_create=False) is None
    menu = window.get_menu("Timeline")
    assert isinstance(menu, QMenu)
    assert window.get_menu("Timeline", auto_create=False) is menu
