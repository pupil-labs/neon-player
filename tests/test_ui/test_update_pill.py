from unittest.mock import patch

from PySide6.QtCore import Qt
from PySide6.QtGui import QDesktopServices

from pupil_labs.neon_player.ui.update_pill import UpdatePill


def test_update_pill_initial_state(qtbot):
    pill = UpdatePill()
    qtbot.addWidget(pill)

    assert pill.isHidden()
    assert pill.dismiss_btn is not None
    assert pill.label is not None


def test_update_pill_show_update(qtbot):
    pill = UpdatePill()
    qtbot.addWidget(pill)

    url = "https://github.com/pupil-labs/neon-player/releases/tag/v2.0.0"
    pill.show_update("v2.0.0", url)
    assert not pill.isHidden()
    assert "v2.0.0" in pill.label.text()
    assert pill.release_url == url


def test_update_pill_dismiss(qtbot):
    pill = UpdatePill()
    qtbot.addWidget(pill)

    pill.show_update(
        "v2.0.0", "https://github.com/pupil-labs/neon-player/releases/tag/v2.0.0"
    )
    assert not pill.isHidden()

    qtbot.mouseClick(pill.dismiss_btn, Qt.MouseButton.LeftButton)
    assert pill.isHidden()


@patch.object(QDesktopServices, "openUrl")
def test_update_pill_click_opens_url(mock_open_url, qtbot):
    pill = UpdatePill()
    qtbot.addWidget(pill)

    url = "https://github.com/pupil-labs/neon-player/releases/tag/v2.0.0"
    pill.show_update("v2.0.0", url)

    qtbot.mouseClick(pill, Qt.MouseButton.LeftButton)
    mock_open_url.assert_called_once()
    called_url = mock_open_url.call_args[0][0].toString()
    assert called_url == url
