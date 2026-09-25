import json
import logging
import urllib.error
import urllib.request

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QTextBrowser,
    QVBoxLayout,
)

logger = logging.getLogger(__name__)


class FetchChangelogThread(QThread):
    changelog_ready = Signal(str)

    def __init__(self, repo="pupil-labs/neon-player", parent=None):
        super().__init__(parent)
        self.repo = repo

    def run(self):
        try:
            url = f"https://api.github.com/repos/{self.repo}/releases"
            req = urllib.request.Request(url, headers={"User-Agent": "Neon-Player"})
            with urllib.request.urlopen(req, timeout=8) as response:  # noqa: S310
                releases = json.loads(response.read().decode())[:10]

            if not releases:
                self.changelog_ready.emit("No release notes found.")
                return

            md_lines = []
            for r in releases:
                tag = r.get("tag_name") or r.get("name", "Unknown Version")
                body = (r.get("body") or "").strip() or "No release notes provided."
                md_lines.append(f"## {tag}\n\n{body}\n\n---")

            self.changelog_ready.emit("\n\n".join(md_lines))
        except urllib.error.URLError as e:
            logger.info("Could not fetch changelog (offline: %s)", e.reason)
            self.changelog_ready.emit(
                "### Offline\n\n"
                "Unable to load release history from GitHub while offline."
            )
        except Exception:
            logger.warning("Failed to fetch changelog", exc_info=True)
            self.changelog_ready.emit(
                "### Error\n\nFailed to fetch release history from GitHub."
            )


class ChangelogDialog(QDialog):
    def __init__(self, releases_markdown: str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Release History")
        self.resize(650, 480)

        layout = QVBoxLayout(self)

        self.browser = QTextBrowser(self)
        self.browser.setOpenExternalLinks(True)
        layout.addWidget(self.browser)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok, self)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)

        if releases_markdown:
            self.browser.setMarkdown(releases_markdown)
        else:
            self.browser.setMarkdown("*Loading release history from GitHub...*")
            self._fetch_thread = FetchChangelogThread(parent=self)
            self._fetch_thread.changelog_ready.connect(self.browser.setMarkdown)
            self._fetch_thread.start()
