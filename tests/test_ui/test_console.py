import logging

from PySide6.QtWidgets import QTextEdit
from pytestqt.qtbot import QtBot

from pupil_labs.neon_player.ui.console import QTextEditLogger


def test_console_logger_detaches_when_text_edit_is_destroyed(qtbot: QtBot) -> None:
    text_edit = QTextEdit()
    qtbot.addWidget(text_edit)
    handler = QTextEditLogger()
    handler.set_text_edit(text_edit)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.warning("Queue scrolling before destruction")

    text_edit.deleteLater()
    qtbot.waitUntil(lambda: handler not in root_logger.handlers)

    handler._scroll_to_bottom()
    assert handler._text_edit is None
