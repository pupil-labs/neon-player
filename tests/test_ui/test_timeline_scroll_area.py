from unittest.mock import MagicMock

import pytest

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QWidget

from pupil_labs.neon_player.ui.timeline_dock_components import TimelineScrollArea


@pytest.mark.parametrize(
    "modifiers",
    [
        Qt.KeyboardModifier.ShiftModifier,
        Qt.KeyboardModifier.ControlModifier,
        Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier,
    ],
)
def test_timeline_scroll_area__modified_wheel_event_is_ignored(qapp, modifiers):
    scroll_area = TimelineScrollArea()
    content = QWidget()
    content.setMinimumSize(100, 600)
    scroll_area.setWidget(content)
    scroll_area.resize(100, 100)
    scroll_area.show()
    qapp.processEvents()

    scrollbar = scroll_area.verticalScrollBar()
    assert scrollbar.maximum() > 0
    starting_value = scrollbar.value()
    event = QWheelEvent(
        QPointF(10, 10),
        QPointF(10, 10),
        QPoint(),
        QPoint(0, -120),
        Qt.MouseButton.NoButton,
        modifiers,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )

    scroll_area.wheelEvent(event)

    assert scrollbar.value() == starting_value
    assert not event.isAccepted()


@pytest.mark.parametrize(
    "modifiers",
    [
        Qt.KeyboardModifier.NoModifier,
        Qt.KeyboardModifier.MetaModifier,
    ],
)
def test_timeline_scroll_area__unhandled_modifier_wheel_event_scrolls(
    qapp, modifiers
):
    scroll_area = TimelineScrollArea()
    content = QWidget()
    content.setMinimumSize(100, 600)
    scroll_area.setWidget(content)
    scroll_area.resize(100, 100)
    scroll_area.show()
    qapp.processEvents()

    scrollbar = scroll_area.verticalScrollBar()
    assert scrollbar.maximum() > 0
    event = QWheelEvent(
        QPointF(10, 10),
        QPointF(10, 10),
        QPoint(),
        QPoint(0, -120),
        Qt.MouseButton.NoButton,
        modifiers,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )

    scroll_area.wheelEvent(event)

    assert scrollbar.value() > 0
    assert event.isAccepted()
