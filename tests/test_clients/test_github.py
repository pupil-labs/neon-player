import json

import pytest
import requests

from unittest.mock import patch

from pupil_labs.neon_player.clients.github import GithubAPIClient


def mock_response(body, link=None):
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps(body).encode("utf-8")
    if link:
        response.headers["Link"] = link

    return response


@patch("requests.get", side_effect=requests.HTTPError)
def test_github_api_client_bad_status_code(mock_get):
    client = GithubAPIClient()
    with pytest.raises(GithubAPIClient.FailedRequest, match="Request failed"):
        client.get_latest_release("test-owner", "test-repo")

    mock_get.assert_called_once()


@patch("requests.get", side_effect=requests.ConnectionError)
def test_github_api_client_failed_connection(mock_get):
    client = GithubAPIClient()
    with pytest.raises(GithubAPIClient.FailedRequest, match="Connection error"):
        client.get_latest_release("test-owner", "test-repo")

    mock_get.assert_called_once()


@patch("requests.get", side_effect=requests.Timeout("Request timed out"))
def test_github_api_client_request_timeout(mock_get):
    client = GithubAPIClient()
    with pytest.raises(GithubAPIClient.FailedRequest, match="Request timed out"):
        client.get_latest_release("test-owner", "test-repo")

    mock_get.assert_called_once()


@patch.object(GithubAPIClient, "_get_json", return_value=[])
def test_github_api_client_get_latest_release(mock_get_json):
    client = GithubAPIClient()
    client.get_latest_release("test-owner", "test-repo")

    mock_get_json.assert_called_once_with(
        "https://api.github.com/repos/test-owner/test-repo/releases/latest"
    )


@patch("requests.get")
def test_github_api_client_get_releases(mock_get):
    mock_get.return_value = mock_response([])
    client = GithubAPIClient()
    assert client.get_releases("test-owner", "test-repo") == []

    mock_get.assert_called_once()
    assert mock_get.call_args.args[0] == (
        "https://api.github.com/repos/test-owner/test-repo/releases"
    )
    assert mock_get.call_args.kwargs["params"] == {"per_page": "100"}


@pytest.mark.parametrize("page_count", [1, 2])
def test_github_api_client_get_releases_uses_next_link(page_count):
    url = "https://api.github.com/repos/test-owner/test-repo/releases"
    pages = [
        [{"tag_name": f"v6.0.{page * 100 + index}"} for index in range(100)]
        for page in range(page_count)
    ]
    responses = []
    for index, page in enumerate(pages):
        link = None
        if page_count > 1:
            link = (
                f'<{url}?per_page=100&page={page_count}>; rel="last", '
                f'<{url}?per_page=100&page=1>; rel="first"'
            )
            if index > 0:
                link += f', <{url}?per_page=100&page={index}>; rel="prev"'
            if index + 1 < page_count:
                link += f', <{url}?per_page=100&page={index + 2}>; rel="next"'
        responses.append(mock_response(page, link))

    with patch("requests.get", side_effect=responses) as mock_get:
        releases = GithubAPIClient().get_releases("test-owner", "test-repo")

    assert len(releases) == 100 * page_count
    assert mock_get.call_count == page_count
    if page_count == 2:
        assert mock_get.call_args_list[1].args[0] == (
            f"{url}?per_page=100&page=2"
        )
