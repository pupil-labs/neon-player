import pytest
import requests
from unittest.mock import patch

from packaging import version
from pupil_labs.neon_player.clients.github import GithubAPIClient
from pupil_labs.neon_player.managers.update import (
    CheckUpdateThread,
    UpdateManager,
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


def test_update_manager_load_release_notes(tmp_path, qapp):
    mock_release_notes = {
        "6.0.0": "New release",
        "6.0.1": "Even newer release"
    }
    for version_tag, notes in mock_release_notes.items():
        with open(tmp_path / f"v{version_tag}.md", "w+") as f:
            f.write(notes)

    with patch(
        "pupil_labs.neon_player.managers.update.changelog_path",
        return_value=tmp_path
    ):
        update_manager = UpdateManager(qapp)
        update_manager.load_release_notes()

    assert len(update_manager._release_notes) == 2
    for version_tag, expected_notes in mock_release_notes.items():
        actual_notes = update_manager._release_notes[version.parse(version_tag)]
        assert actual_notes == expected_notes


def _setup_mock_update_manager(qapp):
    mock_release_notes = {
        "6.0.0": "New release",
        "6.0.1": "Even newer release"
    }
    update_manager = UpdateManager(qapp)
    for version_tag, notes in mock_release_notes.items():
        update_manager._release_notes[version.parse(version_tag)] = notes

    return update_manager


def test_update_manager_get_new_releases(qapp):
    update_manager = _setup_mock_update_manager(qapp)
    assert len(update_manager._get_new_releases()) == 2
    assert not update_manager._get_new_releases(version.parse("6.0.2"))


def test_update_manager_get_release_notes_uses_title(qapp):
    update_manager = _setup_mock_update_manager(qapp)
    notes = update_manager.get_release_notes(title="Cool title")
    assert "Cool title" in notes


def test_update_manager_get_release_notes_uses_since_version(qapp):
    update_manager = _setup_mock_update_manager(qapp)
    notes = update_manager.get_release_notes(since_version=version.parse("6.0.0"))
    assert "New release" not in notes
    assert "Even newer release" in notes


@pytest.mark.parametrize("last_read_version, has_unread", [
    ("6.0.0", True),
    ("6.0.1", False)
])
def test_update_manager_check_unread_release_notes(
    last_read_version, has_unread, qtbot, qapp
):
    update_manager = _setup_mock_update_manager(qapp)

    check_fn = qtbot.waitSignal if has_unread else qtbot.assertNotEmitted
    with check_fn(update_manager.notes_available):
        update_manager.check_unread_release_notes(last_read_version)
