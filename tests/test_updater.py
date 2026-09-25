import json
import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch

from pupil_labs.neon_player.updater import CheckUpdateThread


@patch("urllib.request.urlopen")
@patch("pupil_labs.neon_player.updater.get_version")
def test_check_update_thread_update_available(mock_get_version, mock_urlopen, qtbot):
    mock_get_version.return_value = "1.0.0"

    release_data = {
        "tag_name": "v2.0.0",
        "html_url": "https://github.com/pupil-labs/neon-player/releases/tag/v2.0.0",
    }
    cm = MagicMock()
    cm.read.return_value = json.dumps(release_data).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = cm

    thread = CheckUpdateThread(repo="dummy/repo")

    with qtbot.waitSignal(thread.update_available, timeout=1000) as blocker:
        thread.run()

    tag_name, release_url, release_notes = blocker.args
    assert tag_name == "v2.0.0"
    assert (
        release_url == "https://github.com/pupil-labs/neon-player/releases/tag/v2.0.0"
    )
    assert release_notes == ""


@patch("urllib.request.urlopen")
@patch("pupil_labs.neon_player.updater.get_version")
def test_check_update_thread_no_update(mock_get_version, mock_urlopen):
    mock_get_version.return_value = "2.0.0"

    release_data = {
        "tag_name": "v1.5.0",
        "html_url": "https://github.com/pupil-labs/neon-player/releases/tag/v1.5.0",
    }
    cm = MagicMock()
    cm.read.return_value = json.dumps(release_data).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = cm

    thread = CheckUpdateThread(repo="dummy/repo")
    thread.run()


@patch("urllib.request.urlopen")
def test_check_update_thread_offline(mock_urlopen):
    mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
    thread = CheckUpdateThread(repo="dummy/repo")
    thread.run()


def test_check_update_thread_mock_flag(qtbot):
    import sys

    thread = CheckUpdateThread(repo="dummy/repo")
    with (
        patch.object(sys, "argv", ["neon-player", "--mock-update"]),
        qtbot.waitSignal(thread.update_available, timeout=1000) as blocker,
    ):
        thread.run()

    tag_name, release_url, release_notes = blocker.args
    assert tag_name == "v99.99.99"
    assert "releases" in release_url
    assert "Mock Release" in release_notes


def test_general_settings_check_for_updates():
    from pupil_labs.neon_player.settings import GeneralSettings

    settings = GeneralSettings()
    assert settings.check_for_updates is True

    settings.check_for_updates = False
    assert settings.check_for_updates is False
    assert settings.to_dict()["check_for_updates"] is False

    loaded = GeneralSettings.from_dict({"check_for_updates": False})
    assert loaded.check_for_updates is False
