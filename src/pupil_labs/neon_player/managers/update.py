import logging
import urllib.error
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version

from packaging import version
from pupil_labs.neon_player import changelog_path
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
        self._release_notes: dict[version.Version, str] = {}
        self.thread = CheckUpdateThread()
        self.thread.update_available.connect(self.update_available.emit)

    def check_for_updates(self):
        self.thread.start()

    def get_release_notes(
        self,
        title: str = "Release notes",
        min_version: version.Version | None = None
    ) -> str:
        all_releases = sorted(
            self._release_notes.items(), key=lambda el: el[0], reverse=True
        )
        result = f"# {title}\n"
        for version_tag, notes in all_releases:
            result += f"## v{str(version_tag)}\n{notes}\n"

        return result

    def load_release_notes(self) -> None:
        for path in changelog_path().iterdir():
            print(path, path.suffix)
            if not path.is_file() or not path.suffix == ".md":
                continue

            version_tag = version.parse(path.stem.lstrip("v"))
            with open(path, "r") as f:
                notes = f.read()

            self._release_notes[version_tag] = notes

        print(self._release_notes)
