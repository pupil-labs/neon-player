import logging
import sys
import threading
from unittest.mock import patch

import pytest
from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import QTextEdit
from pytestqt.qtbot import QtBot

from pupil_labs.neon_player.ui.console import QTextEditLogger


def test_console_logger_queues_worker_records_on_main_thread(
    qtbot: QtBot, caplog: pytest.LogCaptureFixture
) -> None:
    text_edit = QTextEdit()
    qtbot.addWidget(text_edit)
    handler = QTextEditLogger()
    handler.set_text_edit(text_edit)
    handler.setFormatter(
        logging.Formatter("%(pathname)s:%(lineno)d %(funcName)s: %(message)s")
    )
    update_threads: list[QThread] = []
    caller_lines: list[int] = []
    append_text = handler._append_text

    def capture_update(text: str, color: Qt.GlobalColor | None = None) -> None:
        update_threads.append(QThread.currentThread())
        append_text(text, color)

    def log_from_worker() -> None:
        caller_lines.append(sys._getframe().f_lineno + 1)
        worker_logger.warning("message %s", "from worker")

    worker_logger = logging.getLogger("test.console.worker")
    worker_logger.addHandler(handler)
    try:
        with (
            caplog.at_level(logging.WARNING, logger=worker_logger.name),
            patch.object(handler, "_append_text", capture_update),
        ):
            worker = threading.Thread(target=log_from_worker)
            worker.start()
            worker.join(timeout=1)
            assert not worker.is_alive()
            assert update_threads == []
            qtbot.waitUntil(lambda: len(update_threads) == 1)

        assert update_threads == [text_edit.thread()]
        assert text_edit.toPlainText() == (
            f"{__file__}:{caller_lines[0]} log_from_worker: message from worker"
        )
    finally:
        worker_logger.removeHandler(handler)
        handler.close()


def test_console_logger_buffers_records_until_widget_is_set(qtbot: QtBot) -> None:
    handler = QTextEditLogger()
    handler.setFormatter(logging.Formatter("%(message)s"))
    record = logging.makeLogRecord({"msg": "buffered message"})
    try:
        handler.handle(record)
        qtbot.waitUntil(lambda: handler._buffer == ["buffered message"])
        text_edit = QTextEdit()
        qtbot.addWidget(text_edit)
        handler.set_text_edit(text_edit)
        assert text_edit.toPlainText() == "buffered message"
        assert handler._buffer == []
    finally:
        handler.close()


def test_console_logger_ignores_queued_records_after_close(qtbot: QtBot) -> None:
    handler = QTextEditLogger()
    text_edit = QTextEdit()
    qtbot.addWidget(text_edit)
    handler.set_text_edit(text_edit)
    handler.handle(logging.makeLogRecord({"msg": "queued before close"}))
    handler.close()
    with patch.object(handler, "_append_text") as append_text:
        qtbot.wait(10)
    append_text.assert_not_called()
    assert handler._buffer == []


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
