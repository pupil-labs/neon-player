import logging
from collections.abc import Iterator
from typing import Any

import requests
from requests import Response

logger = logging.getLogger(__name__)


class GithubAPIClient:
    class FailedRequest(Exception): ...

    def __init__(self, timeout_seconds: float = 10) -> None:
        self.base_url = "https://api.github.com"
        self.timeout_seconds = timeout_seconds
        self.headers = {"User-Agent": "Neon-Player-GithubAPIClient"}

    def _get(self, url: str, **params: str | int) -> requests.Response:
        logger.debug(f"Retrieving {url}")
        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout_seconds,
                params={k: str(v) for k, v in params.items()} or None,
            )
            response.raise_for_status()
            return response
        except requests.HTTPError as e:
            raise GithubAPIClient.FailedRequest(f"Request failed: {e!s}") from e
        except requests.ConnectionError as e:
            raise GithubAPIClient.FailedRequest(f"Connection error: {e!s}") from e
        except requests.Timeout as e:
            raise GithubAPIClient.FailedRequest(f"Request timed out: {e!s}") from e
        except Exception as e:
            raise GithubAPIClient.FailedRequest(f"Request failed: {e!s}") from e

    @staticmethod
    def _json(response: Response) -> object:
        try:
            return response.json()
        except requests.JSONDecodeError as e:
            raise GithubAPIClient.FailedRequest(f"Failed to decode JSON: {e!s}") from e

    def _get_json(self, url: str, **params: str | int) -> object:
        return self._json(self._get(url, **params))

    def _paginate(self, url: str, **params: str | int) -> Iterator[Response]:
        next_page: str | None = url
        while next_page:
            response = self._get(next_page, **params)
            yield response

            next_page = response.links.get("next", {}).get("url")

    def get_latest_release(self, owner: str, repo: str) -> dict[str, Any]:
        release = self._get_json(
            f"{self.base_url}/repos/{owner}/{repo}/releases/latest"
        )
        if not isinstance(release, dict):
            raise GithubAPIClient.FailedRequest(
                "Unexpected response from GitHub latest release API"
            )
        return release

    def get_releases(self, owner: str, repo: str) -> list[dict[str, Any]]:
        releases: list[dict[str, Any]] = []
        url = f"{self.base_url}/repos/{owner}/{repo}/releases"
        for page in self._paginate(url, per_page=100):
            page_releases = self._json(page)
            if not isinstance(page_releases, list) or any(
                not isinstance(release, dict) for release in page_releases
            ):
                raise GithubAPIClient.FailedRequest(
                    "Unexpected response from GitHub releases API"
                )

            releases.extend(page_releases)

        return releases
