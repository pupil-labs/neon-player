import logging
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version

from packaging import version
from pupil_labs.neon_player import changelog_path
from pupil_labs.neon_player.clients.github import GithubAPIClient
from PySide6.QtCore import QObject, QThread, Signal

logger = logging.getLogger(__name__)


GITHUB_OWNER = "pupil-labs"
GITHUB_REPO = "neon-player"


def app_version() -> version.Version:
    try:
        v = version.parse(get_version("pupil-labs-neon-player"))
    except (PackageNotFoundError, version.InvalidVersion):
        v = version.parse("0.0.0")

    return v


class CheckUpdateThread(QThread):
    update_available = Signal(str, str)

    def __init__(self):
        super().__init__()
        self.github_client = GithubAPIClient()

    def run(self):
        try:
            data = self.github_client.get_latest_release(GITHUB_OWNER, GITHUB_REPO)
        except ConnectionError:
            logger.info(
                "Could not check for updates (the app is offline, or "
                "the GitHub release API is unreachable)"
            )
            return
        except Exception:
            logger.warning("Failed to check for updates", exc_info=True)
            return

        current_version = app_version()
        logging.debug(f"Currently installed NP: v{current_version}")

        tag_name = data.get("tag_name", "v0.0.0")
        try:
            latest_version = version.parse(tag_name.lstrip("v"))
        except version.InvalidVersion:
            logger.warning("Invalid release tag version: %s", tag_name)
            return
        logging.debug(f"Latest available NP: v{latest_version}")

        if latest_version <= current_version:
            logger.info("Neon Player is up to date")
            return

        logger.info(f"A new release of Neon Player is available: v{latest_version}")
        release_url = data.get(
            "html_url", f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases"
        )
        self.update_available.emit(tag_name, release_url)


class UpdateManager(QObject):
    update_available = Signal(str, str)
    notes_available = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.app = parent
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
        releases = sorted(
            self._release_notes.items(), key=lambda el: el[0], reverse=True
        )
        if min_version is not None:
            releases = list(filter(lambda el: el[0] > min_version, releases))

        if not releases:
            return ""

        result = f"# {title}\n"
        for version_tag, notes in releases:
            result += f"## v{str(version_tag)}\n{notes}\n"

        return result

    def load_release_notes(self, latest_read_version: str) -> None:
        for path in changelog_path().iterdir():
            if not path.is_file() or not path.suffix == ".md":
                continue

            version_tag = version.parse(path.stem.lstrip("v"))
            with open(path, "r") as f:
                notes = f.read()

            self._release_notes[version_tag] = notes

        read_version = version.parse(latest_read_version)
        unread_notes = self.get_release_notes(
            title="What's Changed",
            min_version=read_version
        )
        if not unread_notes:
            return

        logging.debug(f"Release notes were not read since {read_version}")
        self.notes_available.emit(unread_notes)
