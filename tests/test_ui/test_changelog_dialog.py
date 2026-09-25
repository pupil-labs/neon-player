import json
import urllib.error
from unittest.mock import MagicMock, patch

from pupil_labs.neon_player.ui.changelog_dialog import (
    ChangelogDialog,
    FetchChangelogThread,
)


def test_changelog_dialog_with_provided_markdown(qtbot):
    markdown = "## v2.0.0\n\n- Awesome new feature!"
    dlg = ChangelogDialog(releases_markdown=markdown)
    qtbot.addWidget(dlg)

    assert "v2.0.0" in dlg.browser.toPlainText()
    assert "Awesome new feature!" in dlg.browser.toPlainText()


@patch("urllib.request.urlopen")
def test_fetch_changelog_thread_success(mock_urlopen, qtbot):
    releases = [
        {"tag_name": "v2.0.0", "body": "Release 2.0 notes"},
        {"tag_name": "v1.9.0", "body": "Release 1.9 notes"},
    ]
    cm = MagicMock()
    cm.read.return_value = json.dumps(releases).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = cm

    thread = FetchChangelogThread(repo="dummy/repo")
    with qtbot.waitSignal(thread.changelog_ready, timeout=1000) as blocker:
        thread.run()

    result = blocker.args[0]
    assert "## v2.0.0" in result
    assert "Release 2.0 notes" in result
    assert "## v1.9.0" in result


@patch("urllib.request.urlopen")
def test_fetch_changelog_thread_offline(mock_urlopen, qtbot):
    mock_urlopen.side_effect = urllib.error.URLError("No network")

    thread = FetchChangelogThread(repo="dummy/repo")
    with qtbot.waitSignal(thread.changelog_ready, timeout=1000) as blocker:
        thread.run()

    result = blocker.args[0]
    assert "Offline" in result
