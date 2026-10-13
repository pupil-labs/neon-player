import logging
import requests

from typing import Any


class GithubAPIClient:
    def __init__(self, timeout_seconds: float = 10) -> None:
        self.base_url = "https://api.github.com"
        self.timeout_seconds = timeout_seconds
        self.headers = {"User-Agent": "Neon-Player-Updater"}

    def _get_json(self, url: str) -> Any:
        logging.debug(f"Retrieving {url}")
        response = requests.get(
            url,
            headers=self.headers,
            timeout=self.timeout_seconds
        )
        return response.json()

    def get_latest_release(self, owner: str, repo: str) -> dict:
        return self._get_json(f"{self.base_url}/repos/{owner}/{repo}/releases/latest")
