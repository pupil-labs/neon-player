import pytest

from unittest.mock import PropertyMock, patch

from pupil_labs.neon_player.ui.main_window import MainWindow
from pupil_labs.neon_player.ui.update_notification import DismissedUpdatePrompt


def test_main_window_on_update_available(qtbot):
    main_window = MainWindow()
    qtbot.addWidget(main_window)

    assert main_window.statusBar().isHidden()
    assert main_window.update_pill.isHidden()

    with qtbot.waitExposed(main_window):
        def shows_update_pill():
            assert main_window.statusBar().isVisible()
            assert main_window.update_pill.isVisible()

        main_window.show()
        main_window.on_update_available("7.0.0", "url")
        qtbot.waitUntil(shows_update_pill)


def test_main_window_dismissed_update_sets_last_displayed_update_version(qtbot, qapp):
    main_window = MainWindow()
    qtbot.addWidget(main_window)

    main_window.update_pill.version_tag = "7.0.0"
    qapp.settings.last_displayed_update_version = "6.0.0"
    qapp.settings.show_prompt_on_dismissed_update = False

    main_window.prompt_dismiss_updates()
    assert qapp.settings.last_displayed_update_version == "7.0.0"


@pytest.mark.parametrize("user_confirmed", [True, False])
def test_main_window_dismissed_update_not_show_message_again(user_confirmed, qtbot, qapp):
    main_window = MainWindow()
    qtbot.addWidget(main_window)

    main_window.update_pill.version_tag = "7.0.0"
    qapp.settings.last_displayed_update_version = "6.0.0"
    qapp.settings.show_prompt_on_dismissed_update = True

    with patch.object(
        DismissedUpdatePrompt, "user_confirmed", new_callable=PropertyMock
    ) as mock_confirmed:
        mock_confirmed.return_value = user_confirmed
        main_window.prompt_dismiss_updates()
        assert qapp.settings.show_prompt_on_dismissed_update or user_confirmed
