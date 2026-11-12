from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QMouseEvent
from pytestqt.qtbot import QtBot

from pupil_labs.neon_player.ui import GUIEventNotifier


def test_event_notifier_emits_mouse_click(qtbot: QtBot) -> None:
    widget = GUIEventNotifier()
    qtbot.addWidget(widget)
    event = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(1.25, 2.5),
        QPointF(1.25, 2.5),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    with qtbot.waitSignal(widget.mouse_pressed) as signal:
        widget.mousePressEvent(event)
    assert signal.args[0].position() == QPointF(1.25, 2.5)
    with qtbot.waitSignal(widget.mouse_clicked):
        widget.mouseReleaseEvent(event)
