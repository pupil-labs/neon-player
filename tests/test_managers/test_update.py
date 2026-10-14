import requests
from unittest.mock import patch

from packaging import version
from pupil_labs.neon_player.clients.github import GithubAPIClient
from pupil_labs.neon_player.managers.update import CheckUpdateThread


@patch.object(
    GithubAPIClient,
    "get_latest_release",
    return_value={
        "tag_name": "v2.0.0",
        "html_url": "https://github.com/pupil-labs/neon-player/releases/tag/v2.0.0",
    }
)
@patch(
    "pupil_labs.neon_player.managers.update.app_version",
    return_value=version.parse("1.0.0")
)
def test_check_update_thread_update_available(mock_get_version, mock_get_release, qtbot):
    thread = CheckUpdateThread("1.0.0")

    with qtbot.waitSignal(thread.update_available, timeout=1000) as blocker:
        thread.run()

    mock_get_version.assert_called_once()
    mock_get_release.assert_called_once()

    tag_name, release_url = blocker.args
    assert tag_name == "v2.0.0"
    assert (
        release_url == "https://github.com/pupil-labs/neon-player/releases/tag/v2.0.0"
    )


@patch.object(
    GithubAPIClient,
    "get_latest_release",
    return_value={
        "tag_name": "v2.0.0",
        "html_url": "https://github.com/pupil-labs/neon-player/releases/tag/v2.0.0",
    }
)
@patch(
    "pupil_labs.neon_player.managers.update.app_version",
    return_value=version.parse("2.0.0")
)
def test_check_update_thread_no_update(mock_get_version, mock_get_release, qtbot):
    thread = CheckUpdateThread("2.0.0")
    with qtbot.assertNotEmitted(thread.update_available):
        thread.run()

    mock_get_version.assert_called_once()
    mock_get_release.assert_called_once()


@patch.object(
    GithubAPIClient,
    "get_latest_release",
    return_value={
        "tag_name": "v2.0.1",
        "html_url": "https://github.com/pupil-labs/neon-player/releases/tag/v2.0.0",
    }
)
@patch(
    "pupil_labs.neon_player.managers.update.app_version",
    return_value=version.parse("2.0.0")
)
def test_check_update_thread_update_already_shown(mock_get_version, mock_get_release, qtbot):
    thread = CheckUpdateThread("2.0.1")
    with qtbot.assertNotEmitted(thread.update_available):
        thread.run()

    mock_get_version.assert_called_once()
    mock_get_release.assert_called_once()


@patch.object(
    GithubAPIClient,
    "get_latest_release",
    side_effect=requests.ConnectionError("Connection refused")
)
@patch(
    "pupil_labs.neon_player.managers.update.app_version",
    return_value=version.parse("1.0.0")
)
def test_check_update_thread_offline(mock_get_version, mock_get_release, qtbot):
    thread = CheckUpdateThread("1.0.0")
    with qtbot.assertNotEmitted(thread.update_available):
        thread.run()

    mock_get_release.assert_called_once()
    mock_get_version.assert_not_called()
