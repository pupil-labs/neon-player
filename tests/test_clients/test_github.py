import pytest
import requests

from unittest.mock import patch

from pupil_labs.neon_player.clients.github import GithubAPIClient


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


@patch.object(GithubAPIClient, "_get_json")
def test_github_api_client_get_latest_release(mock_get_json):
    client = GithubAPIClient()
    client.get_latest_release("test-owner", "test-repo")

    mock_get_json.assert_called_with(
        "https://api.github.com/repos/test-owner/test-repo/releases/latest"
    )
