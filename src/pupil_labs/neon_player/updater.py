import json
import logging
import urllib.error
import urllib.request
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version

from packaging import version
from pupil_labs.neon_player.clients.github import GithubAPIClient
from PySide6.QtCore import QObject, QThread, Signal

logger = logging.getLogger(__name__)


class CheckUpdateThread(QThread):
    update_available = Signal(str, str, str)  # tag_name, release_url, release_notes

    def __init__(self, repo="pupil-labs/neon-player"):
        super().__init__()
        self.github_client = GithubAPIClient(repo)

    def run(self):
        try:
            import sys

            if "--mock-update" in sys.argv:
                logger.info("Forcing mock update notification via --mock-update flag")
                self.update_available.emit(
                    "v99.99.99",
                    f"https://github.com/{self.github_client.repo}/releases",
                    "### Mock Release v99.99.99\n\n"
                    "- Example changelog entry.\n"
                    "- Real-time Markdown rendering in Neon Player!",
                )
                return

            try:
                curr_ver_str = get_version("pupil-labs-neon-player")
            except PackageNotFoundError:
                try:
                    curr_ver_str = get_version("pupil_labs.neon_player")
                except PackageNotFoundError:
                    curr_ver_str = "0.0.0"

            try:
                current_version = version.parse(curr_ver_str.lstrip("v"))
            except version.InvalidVersion:
                current_version = version.parse("0.0.0")

            data = self.github_client.get_latest_release()

            tag_name = data.get("tag_name", "v0.0.0")
            try:
                latest_version = version.parse(tag_name.lstrip("v"))
            except version.InvalidVersion:
                logger.warning("Invalid release tag version: %s", tag_name)
                return

            if latest_version <= current_version:
                logger.info(
                    "App is up to date (current: %s, latest: %s)",
                    current_version,
                    latest_version,
                )
                return

            release_url = data.get(
                "html_url", f"https://github.com/{self.github_client.repo}/releases"
            )
            release_notes = data.get("body", "")
            self.update_available.emit(tag_name, release_url, release_notes)
        except urllib.error.URLError as e:
            logger.info(
                "Could not check for updates (offline or unreachable: %s)", e.reason
            )
        except Exception:
            logger.warning("Failed to check for updates", exc_info=True)


class UpdateManager(QObject):
    update_available = Signal(str, str, str)  # tag_name, release_url, release_notes

    def __init__(self, repo="pupil-labs/neon-player", parent=None):
        super().__init__(parent)
        self.thread = CheckUpdateThread(repo=repo)
        self.thread.update_available.connect(self.update_available.emit)

    def check_for_updates(self):
        self.thread.start()
