import json
import logging
import urllib.error
import urllib.request
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version

from packaging import version
from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)


class CheckUpdateThread(QThread):
    update_available = Signal(str, str, str)  # tag_name, release_url, release_notes

    def __init__(self, repo="pupil-labs/neon-player"):
        super().__init__()
        self.repo = repo

    def run(self):
        try:
            import sys

            if "--mock-update" in sys.argv:
                logger.info("Forcing mock update notification via --mock-update flag")
                self.update_available.emit(
                    "v99.99.99",
                    f"https://github.com/{self.repo}/releases",
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

            url = f"https://api.github.com/repos/{self.repo}/releases/latest"
            req = urllib.request.Request(
                url, headers={"User-Agent": "Neon-Player-Updater"}
            )
            with urllib.request.urlopen(req, timeout=10) as response:  # noqa: S310
                data = json.loads(response.read().decode())

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
                "html_url", f"https://github.com/{self.repo}/releases"
            )
            release_notes = data.get("body", "")
            self.update_available.emit(tag_name, release_url, release_notes)
        except urllib.error.URLError as e:
            logger.info(
                "Could not check for updates (offline or unreachable: %s)", e.reason
            )
        except Exception:
            logger.warning("Failed to check for updates", exc_info=True)
