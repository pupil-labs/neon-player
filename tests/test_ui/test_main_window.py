import pytest
from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtWidgets import QMainWindow, QMenu
from pytestqt.qtbot import QtBot

from pupil_labs.neon_player.ui.main_window import MainWindow
from pupil_labs.neon_player.ui.timeline_dock import TimeLineDock
from tests.conftest import MockNeonPlayerApp


def test_menu_lookup_keeps_missing_and_existing_menus_distinct(qtbot: QtBot) -> None:
    window = MainWindow.__new__(MainWindow)
    QMainWindow.__init__(window)
    qtbot.addWidget(window)
    assert window.get_menu("Timeline", auto_create=False) is None
    menu = window.get_menu("Timeline")
    assert isinstance(menu, QMenu)
    assert window.get_menu("Timeline", auto_create=False) is menu


def test_timeline_group_menu_actions_and_separate_shortcuts(
    qtbot: QtBot, qapp: MockNeonPlayerApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(qapp, "recording", None)
    monkeypatch.setattr(qapp, "playback_speed_options", [-1.0, 1.0], raising=False)
    window = MainWindow.__new__(MainWindow)
    QMainWindow.__init__(window)
    qtbot.addWidget(window)
    window.timeline = TimeLineDock()
    window.setCentralWidget(window.timeline)
    window._register_timeline_actions()
    first = window.timeline.add_timeline_group("First")
    second = window.timeline.add_timeline_group("Second", collapsed=True)
    expand = window.get_action("Timeline/Expand all groups")
    collapse = window.get_action("Timeline/Collapse all groups")
    assert expand.shortcut().toString() == "Ctrl+E"
    assert collapse.shortcut().toString() == "Ctrl+Shift+E"
    window.resize(800, 400)
    window.show()
    window.activateWindow()
    QCoreApplication.processEvents()
    qtbot.keyClick(window, Qt.Key.Key_E, Qt.KeyboardModifier.ControlModifier)
    assert not first.collapsed
    assert not second.collapsed
    qtbot.keyClick(window, Qt.Key.Key_E, Qt.KeyboardModifier.ControlModifier)
    assert not first.collapsed
    assert not second.collapsed
    qtbot.keyClick(
        window,
        Qt.Key.Key_E,
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
    )
    assert first.collapsed
    assert second.collapsed
    qtbot.keyClick(
        window,
        Qt.Key.Key_E,
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
    )
    assert first.collapsed
    assert second.collapsed
    expand.trigger()
    assert not first.collapsed
    assert not second.collapsed
    window.timeline.timeline_group_controls[first.name].click()
    collapse.trigger()
    assert first.collapsed
    assert second.collapsed
    window.unregister_action("Timeline/Expand all groups")
    with pytest.raises(ValueError, match="not found"):
        window.get_action("Timeline/Expand all groups")
