from unittest.mock import patch

from PySide6.QtCore import QPoint, Qt

from pupil_labs.neon_player.ui.notification_pill import UpdatePill


def test_update_pill_hidden_by_default(qtbot):
    pill = UpdatePill()
    qtbot.addWidget(pill)

    assert pill.isHidden()


def test_update_pill_show_update(qtbot):
    pill = UpdatePill()
    qtbot.addWidget(pill)

    url = "https://github.com/pupil-labs/neon-player/releases/tag/v2.0.0"
    pill.show_update("v2.0.0", url)
    assert pill.isVisible()
    assert "v2.0.0" in pill.action_btn.text()
    assert pill.action_url == url


def test_update_pill_dismiss(qtbot):
    pill = UpdatePill()
    qtbot.addWidget(pill)

    pill.show_update(
        "v2.0.0", "https://github.com/pupil-labs/neon-player/releases/tag/v2.0.0"
    )
    assert pill.isVisible()

    with qtbot.waitSignal(pill.dismissed):
        qtbot.mouseClick(pill.dismiss_btn, Qt.MouseButton.LeftButton)
    assert pill.isHidden()


@patch("pupil_labs.neon_player.ui.notification_pill.webbrowser.open")
def test_update_pill_action_click_opens_url(mock_open_url, qtbot):
    pill = UpdatePill()
    qtbot.addWidget(pill)

    url = "https://github.com/pupil-labs/neon-player/releases/tag/v2.0.0"
    pill.show_update("v2.0.0", url)

    qtbot.mouseClick(pill.action_btn, Qt.MouseButton.LeftButton)
    mock_open_url.assert_called_once_with(url)
    mock_open_url.reset_mock()

    qtbot.mouseClick(pill.dismiss_btn, Qt.MouseButton.LeftButton)
    mock_open_url.assert_not_called()
