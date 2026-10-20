import logging
import re
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version
from pathlib import Path
from typing import TYPE_CHECKING

from packaging import version
from pupil_labs.neon_player.clients.github import GithubAPIClient
from PySide6.QtCore import QObject, QThread, Signal

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pupil_labs.neon_player.app import NeonPlayerApp


GITHUB_OWNER = "pupil-labs"
GITHUB_REPO = "neon-player"


def release_notes_path() -> Path:
    return Path.home() / "Pupil Labs" / "Neon Player" / "release_notes"


def _process_release_notes(notes: str) -> str:
    lines = notes.splitlines()
    if lines and "what's changed" in lines[0].lower():
        lines = lines[1:]

    notes = re.sub(r"(?m)^([ \t]*)\*(?=[ \t])", r"\1-", "\n".join(lines))
    return notes.lstrip("\n")


def app_version() -> version.Version:
    try:
        v = version.parse(get_version("pupil-labs-neon-player"))
    except (PackageNotFoundError, version.InvalidVersion):
        v = version.parse("0.0.0")

    return v


class CheckUpdateThread(QThread):
    update_available = Signal(str, str)

    def __init__(self, last_displayed_update_version: str) -> None:
        super().__init__()
        self.last_displayed_update_version = version.parse(last_displayed_update_version)
        self.github_client = GithubAPIClient()

    def run(self) -> None:
        try:
            data = self.github_client.get_latest_release(GITHUB_OWNER, GITHUB_REPO)
        except GithubAPIClient.FailedRequest as e:
            logger.warning(
                "Could not check for updates (the app is offline, or "
                "the request to the GitHub release API has failed). "
                f"Error: {str(e)}"
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

        if latest_version <= self.last_displayed_update_version:
            logger.debug(f"Update v{latest_version} was already presented")
            return

        logger.info(f"A new release of Neon Player is available: v{latest_version}")
        release_url = data.get(
            "html_url", f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases"
        )
        self.update_available.emit(tag_name, release_url)


class LoadReleaseNotesThread(QThread):
    releases_loaded = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.github_client = GithubAPIClient()

    def run(self) -> None:
        try:
            releases = self.github_client.get_releases(GITHUB_OWNER, GITHUB_REPO)
        except GithubAPIClient.FailedRequest as e:
            logger.warning("Could not load release notes from GitHub: %s", e)
            self.releases_loaded.emit(None)
            return
        except Exception:
            logger.warning("Failed to load release notes from GitHub", exc_info=True)
            self.releases_loaded.emit(None)
            return

        self.releases_loaded.emit(releases)


class UpdateManager(QObject):
    """Loads GitHub release notes and checks for new application releases."""

    update_available = Signal(str, str)
    notes_available = Signal(str)

    def __init__(self, parent: "NeonPlayerApp") -> None:
        super().__init__(parent)
        self.app = parent
        self._release_notes: dict[version.Version, str] = {}
        self._release_notes_thread: LoadReleaseNotesThread | None = None
        self._loading_release_notes = False
        self._pending_read_version: str | None = None
        self.update_thread = CheckUpdateThread(
            self.app.settings.last_displayed_update_version
        )
        self.update_thread.update_available.connect(self.update_available.emit)

    def check_for_updates(self) -> None:
        self.update_thread.start()

    def _get_new_releases(
        self, since_version: version.Version | None = None
    ) -> list[tuple[version.Version, str]]:
        releases = sorted(
            self._release_notes.items(), key=lambda el: el[0], reverse=True
        )
        if since_version is None:
            return releases

        return list(filter(lambda el: el[0] > since_version, releases))

    def get_release_notes(
        self,
        title: str = "Release notes",
        since_version: version.Version | None = None
    ) -> str:
        releases = self._get_new_releases(since_version)
        if not releases:
            return ""

        result = f"# {title}\n"
        for version_tag, notes in releases:
            result += f"## v{str(version_tag)}\n{notes}\n"

        return result

    def _load_release_notes_from_cache(self) -> None:
        cache_path = release_notes_path()
        if not cache_path.exists() or not cache_path.is_dir():
            return

        for path in cache_path.glob("v*.md"):
            try:
                version_tag = version.parse(path.stem.lstrip("v"))
            except version.InvalidVersion:
                logger.warning("Ignoring invalid cached release note: %s", path)
                continue

            try:
                notes = path.read_text(encoding="utf-8")
                self._release_notes[version_tag] = _process_release_notes(notes)
            except (OSError, UnicodeError):
                logger.exception("Failed to read cached release note: %s", path)

    def load_release_notes(self) -> None:
        self._load_release_notes_from_cache()
        max_cached_version = max(self._release_notes, default=None)
        if max_cached_version is not None and app_version() <= max_cached_version:
            logging.debug("No new release notes to display")
            return

        if self._loading_release_notes:
            return

        self._loading_release_notes = True
        self._release_notes_thread = LoadReleaseNotesThread()
        self._release_notes_thread.releases_loaded.connect(self._on_releases_loaded)
        self._release_notes_thread.start()

    def _on_releases_loaded(self, releases: object) -> None:
        self._loading_release_notes = False
        fetched_notes: dict[version.Version, str] = {}
        for release in releases:
            if not isinstance(release, dict):
                continue

            tag_name = release.get("tag_name")
            if not isinstance(tag_name, str):
                continue

            try:
                version_tag = version.parse(tag_name.lstrip("v"))
            except version.InvalidVersion:
                logger.warning("Ignoring invalid release tag version: %s", tag_name)
                continue

            body = release.get("body")
            notes = body if isinstance(body, str) else ""
            fetched_notes[version_tag] = _process_release_notes(notes)

        self._release_notes.update(fetched_notes)
        try:
            cache_path = release_notes_path()
            cache_path.mkdir(parents=True, exist_ok=True)
            for version_tag, notes in fetched_notes.items():
                cache_file = cache_path / f"v{version_tag}.md"
                cache_file.write_text(notes, encoding="utf-8")
        except OSError:
            logger.exception("Failed to cache GitHub release notes")

        pending_read_version = self._pending_read_version
        self._pending_read_version = None
        if pending_read_version is not None:
            self._emit_unread_release_notes(pending_read_version)

    def check_unread_release_notes(self, latest_read_version: str) -> None:
        if self._loading_release_notes:
            self._pending_read_version = latest_read_version
            return

        self._emit_unread_release_notes(latest_read_version)

    def _emit_unread_release_notes(self, latest_read_version: str) -> None:
        read_version = version.parse(latest_read_version)
        unread_notes = self.get_release_notes(
            title="What's Changed",
            since_version=read_version
        )
        if not unread_notes:
            return

        logging.debug(f"Release notes were not read since {read_version}")
        self.notes_available.emit(unread_notes)
