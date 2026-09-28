import json
import urllib.request


class GithubAPIClient:
    def __init__(self, repo: str):
        self.repo = repo

    def get_latest_release(self) -> dict:
        url = f"https://api.github.com/repos/{self.repo}/releases/latest"
        req = urllib.request.Request(url, headers={"User-Agent": "Neon-Player-Updater"})
        with urllib.request.urlopen(req, timeout=10) as response:  # ruff: ignore[suspicious-url-open-usage]
            return json.loads(response.read().decode())
