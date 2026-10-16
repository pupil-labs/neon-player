import logging
import requests

from typing import Any


class GithubAPIClient:
    class FailedRequest(Exception): ...

    def __init__(self, timeout_seconds: float = 10) -> None:
        self.base_url = "https://api.github.com"
        self.timeout_seconds = timeout_seconds
        self.headers = {"User-Agent": "Neon-Player-GithubAPIClient"}

    def _get_json(self, url: str) -> Any:
        logging.debug(f"Retrieving {url}")
        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout_seconds
            )
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            raise GithubAPIClient.FailedRequest(f"Request failed: {str(e)}") from e
        except requests.ConnectionError as e:
            raise GithubAPIClient.FailedRequest(f"Connection error: {str(e)}") from e

    def get_latest_release(self, owner: str, repo: str) -> Any:
        return self._get_json(f"{self.base_url}/repos/{owner}/{repo}/releases/latest")
