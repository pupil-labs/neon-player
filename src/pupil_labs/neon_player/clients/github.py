import logging
import requests

from collections.abc import Iterator
from requests import Response
from typing import Any


class GithubAPIClient:
    class FailedRequest(Exception): ...

    def __init__(self, timeout_seconds: float = 10) -> None:
        self.base_url = "https://api.github.com"
        self.timeout_seconds = timeout_seconds
        self.headers = {"User-Agent": "Neon-Player-GithubAPIClient"}

    def _get(self, url: str, **params: Any) -> requests.Response:
        logging.debug(f"Retrieving {url}")
        try:
            request_options: dict[str, Any] = {
                "headers": self.headers,
                "timeout": self.timeout_seconds,
            }
            if params:
                request_options["params"] = {k: str(v) for k, v in params.items()}

            response = requests.get(url, **request_options)
            response.raise_for_status()
            return response
        except requests.HTTPError as e:
            raise GithubAPIClient.FailedRequest(f"Request failed: {str(e)}") from e
        except requests.ConnectionError as e:
            raise GithubAPIClient.FailedRequest(f"Connection error: {str(e)}") from e
        except requests.Timeout as e:
            raise GithubAPIClient.FailedRequest(f"Request timed out: {str(e)}") from e
        except Exception as e:
            raise GithubAPIClient.FailedRequest(f"Request failed: {str(e)}") from e

    @staticmethod
    def _json(response: Response) -> Any:
        try:
            return response.json()
        except requests.JSONDecodeError as e:
            raise GithubAPIClient.FailedRequest(
                f"Failed to decode JSON: {str(e)}"
            ) from e

    def _get_json(self, url: str, **params: Any) -> Any:
        return self._json(self._get(url, **params))

    def _paginate(self, url: str, **params: Any) -> Iterator[Response]:
        next_page: str | None = url
        while next_page:
            response = self._get(next_page, **params)
            yield response

            next_page = response.links.get("next", {}).get("url")

    def get_latest_release(self, owner: str, repo: str) -> Any:
        return self._get_json(f"{self.base_url}/repos/{owner}/{repo}/releases/latest")

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
