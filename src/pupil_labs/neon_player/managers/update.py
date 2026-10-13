import logging
import urllib.error
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version

from packaging import version
from pupil_labs.neon_player.clients.github import GithubAPIClient
from PySide6.QtCore import QObject, QThread, Signal

logger = logging.getLogger(__name__)


GITHUB_OWNER = "pupil-labs"
GITHUB_REPO = "neon-player"


class CheckUpdateThread(QThread):
    update_available = Signal(str, str, str)  # tag_name, release_url, release_notes

    def __init__(self):
        super().__init__()
        self.github_client = GithubAPIClient()

    def run(self):
        try:
            data = self.github_client.get_latest_release(GITHUB_OWNER, GITHUB_REPO)
        except urllib.error.URLError as e:
            logger.info(
                "Could not check for updates (offline or unreachable: %s)", e.reason
            )
            return
        except Exception:
            logger.warning("Failed to check for updates", exc_info=True)
            return

        try:
            current_version = version.parse(get_version("pupil-labs-neon-player"))
        except (PackageNotFoundError, version.InvalidVersion):
            current_version = version.parse("0.0.0")
        logging.debug(f"Currently installed NP: {current_version}")

        tag_name = data.get("tag_name", "v0.0.0")
        try:
            latest_version = version.parse(tag_name.lstrip("v"))
        except version.InvalidVersion:
            logger.warning("Invalid release tag version: %s", tag_name)
            return
        logging.debug(f"Latest available NP: {latest_version}")

        if latest_version <= current_version:
            logger.info(
                "App is up to date (current: %s, latest: %s)",
                current_version,
                latest_version,
            )
            return

        logger.info(f"A new update is available: {latest_version}")
        release_url = data.get(
            "html_url", f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases"
        )
        release_notes = data.get("body", "No release notes found.")
        self.update_available.emit(tag_name, release_url, release_notes)


class UpdateManager(QObject):
    update_available = Signal(str, str, str)  # tag_name, release_url, release_notes

    def __init__(self, parent=None):
        super().__init__(parent)
        self.thread = CheckUpdateThread()
        self.thread.update_available.connect(self.update_available.emit)

    def check_for_updates(self):
        self.thread.start()
