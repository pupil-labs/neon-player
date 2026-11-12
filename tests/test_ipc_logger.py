import logging
import pickle
import sys
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from pytestqt.qtbot import QtBot

from pupil_labs.neon_player import ipc_logger
from pupil_labs.neon_player.ipc_logger import IPCLogger


@pytest.fixture
def ipc_loggers(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[tuple[IPCLogger, IPCLogger]]:
    monkeypatch.setattr(ipc_logger, "IPC_APP_NAME", f"neon-test-{uuid4().hex}")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = logging.getLogger()
    handlers = root.handlers[:]
    level = root.level
    server = IPCLogger()
    client = IPCLogger()
    root.removeHandler(client)
    qtbot.waitUntil(lambda: len(server._client_sockets) == 1)
    try:
        yield server, client
    finally:
        for handler in root.handlers[:]:
            if handler not in handlers:
                root.removeHandler(handler)
                handler.close()
        root.setLevel(level)
        client.log_socket.abort()
        for socket in server._client_sockets:
            socket.abort()
        assert server.server is not None
        server.server.close()
        ipc_logger.QLocalServer.removeServer(ipc_logger.IPC_APP_NAME)


@pytest.mark.parametrize(
    "exception_args",
    [("useful exception details",), ("useful exception details", lambda: None)],
    ids=["ordinary", "unpicklable"],
)
def test_exception_log_roundtrip(
    exception_args: tuple[object, ...],
    ipc_loggers: tuple[IPCLogger, IPCLogger],
    qtbot: QtBot,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _, client = ipc_loggers
    try:
        raise ValueError(*exception_args)
    except ValueError:
        record = logging.getLogger("ipc-test").makeRecord(
            "ipc-test",
            logging.ERROR,
            __file__,
            1,
            "failed %s",
            ("operation",),
            sys.exc_info(),
        )
    client.handle(record)
    qtbot.waitUntil(lambda: any(r.name == "ipc-test" for r in caplog.records))
    assert "[BG] failed operation" in caplog.text
    assert "ValueError" in caplog.text
    assert "useful exception details" in caplog.text


def test_fragmented_log_roundtrip(
    ipc_loggers: tuple[IPCLogger, IPCLogger],
    qtbot: QtBot,
    caplog: pytest.LogCaptureFixture,
) -> None:
    server, client = ipc_loggers
    record = logging.getLogger("ipc-fragment-test").makeRecord(
        "ipc-fragment-test",
        logging.INFO,
        __file__,
        1,
        "fragmented %s",
        ("message",),
        None,
    )
    payload = pickle.dumps(record)
    assert client.log_stream is not None
    client.log_stream.writeUInt32(len(payload))
    client.log_stream.writeRawData(payload[:7])
    with qtbot.waitSignal(server._client_sockets[0].readyRead):
        client.log_socket.flush()
    assert not any(r.name == "ipc-fragment-test" for r in caplog.records)
    client.log_stream.writeRawData(payload[7:])
    client.log_socket.flush()
    qtbot.waitUntil(lambda: any(r.name == "ipc-fragment-test" for r in caplog.records))
    records = [r for r in caplog.records if r.name == "ipc-fragment-test"]
    assert [r.getMessage() for r in records] == ["[BG] fragmented message"]
