import logging
import threading
from unittest.mock import patch

import pytest
from PySide6.QtCore import QCoreApplication, Qt, QThread
from PySide6.QtWidgets import QLabel, QMainWindow, QMenu
from pytestqt.qtbot import QtBot

from pupil_labs.neon_player.ui.main_window import MainWindow, StatusBarLogHandler
from pupil_labs.neon_player.ui.timeline_dock import TimeLineDock
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


def test_status_log_handler_removed_when_label_is_destroyed(qtbot: QtBot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    handler = window.log_status_handler
    root_logger = logging.getLogger()
    assert handler in root_logger.handlers

    window.status_label.deleteLater()
    qtbot.waitUntil(lambda: handler not in root_logger.handlers)
    root_logger.warning("Logging remains usable after status label destruction")


def test_status_log_handler_updates_on_main_thread(qtbot: QtBot) -> None:
    label = QLabel()
    qtbot.addWidget(label)
    handler = StatusBarLogHandler(label)
    update_threads: list[QThread] = []
    process_record = handler.process_record

    def capture_update(record: logging.LogRecord) -> None:
        update_threads.append(QThread.currentThread())
        process_record(record)

    handler._dispatcher.process_record = capture_update
    try:
        record = logging.makeLogRecord({"msg": "worker status", "levelname": "INFO"})
        worker = threading.Thread(
            target=handler.handle,
            args=(record,),
        )
        worker.start()
        worker.join(timeout=1)
        assert not worker.is_alive()
        assert update_threads == []
        qtbot.waitUntil(lambda: label.text() == "worker status")
        assert update_threads == [label.thread()]

        handler.handle(logging.makeLogRecord({"msg": "pending", "levelname": "INFO"}))
        handler.close()
        with patch.object(label, "setText") as set_text:
            qtbot.wait(10)
        set_text.assert_not_called()
    finally:
        handler.close()
