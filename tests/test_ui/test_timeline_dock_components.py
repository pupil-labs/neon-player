import pytest
from PySide6.QtCore import Qt
from pytestqt.qtbot import QtBot

from pupil_labs.neon_player.ui.timeline_dock_components import TimelinePlotGroupControl


@pytest.mark.parametrize("collapsed", [False, True])
def test_group_control_presents_current_state(qtbot: QtBot, collapsed: bool) -> None:
    control = TimelinePlotGroupControl("Events", collapsed)
    qtbot.addWidget(control)
    assert control.text() == ("▸" if collapsed else "▾")
    assert control.toolTip() == ("Expand Events" if collapsed else "Collapse Events")
    assert control.accessibleName() == control.toolTip()
    assert control.cursor().shape() == Qt.CursorShape.PointingHandCursor
    assert control.autoRaise()
    assert control.width() == 20
    assert control.height() == 20
    control.set_collapsed(not collapsed)
    assert control.text() == ("▾" if collapsed else "▸")
    assert control.toolTip() == ("Collapse Events" if collapsed else "Expand Events")
    assert control.accessibleName() == control.toolTip()


def test_group_control_emits_click_without_mutating_its_state(qtbot: QtBot) -> None:
    control = TimelinePlotGroupControl("Events")
    qtbot.addWidget(control)
    with qtbot.waitSignal(control.clicked):
        control.click()
    assert control.text() == "▾"
