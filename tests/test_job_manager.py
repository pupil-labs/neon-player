import sys

from pathlib import Path

from pupil_labs.neon_player.job_manager import prepare_command


def test_prepare_command():
    recording_path = Path("/path/to/recording")
    action_name = "Plugin.action"
    server_name = "neon-player-server-0"

    expected_cmd = [
        sys.executable,
        "-m",
        "pupil_labs.neon_player",
        str(recording_path),
        "--progress-ipc-name",
        server_name,
        "--job",
        action_name,
    ]

    cmd = prepare_command(
        recording_path,
        action_name,
        args=[],
        server_name=server_name,
        batch_mode_enabled=False,
        is_frozen=False,
    )

    assert cmd == expected_cmd


def test_prepare_command_batch_mode():
    recording_path = Path("/path/to/recording")
    action_name = "Plugin.action"
    server_name = "neon-player-server-0"

    cmd = prepare_command(
        recording_path,
        action_name,
        args=[],
        server_name=server_name,
        batch_mode_enabled=True,
        is_frozen=False,
    )

    assert "--workspace" in cmd


def test_prepare_command_is_frozen():
    recording_path = Path("/path/to/recording")
    action_name = "Plugin.action"
    server_name = "neon-player-server-0"

    cmd = prepare_command(
        recording_path,
        action_name,
        args=[],
        server_name=server_name,
        batch_mode_enabled=False,
        is_frozen=True,
    )

    assert "pupil_labs.neon_player" not in cmd


def test_prepare_command_job_args_are_forwarded():
    recording_path = Path("/path/to/recording")
    action_name = "Plugin.action"
    server_name = "neon-player-server-0"
    job_args = ["arg1", "arg2"]

    cmd = prepare_command(
        recording_path,
        action_name,
        args=job_args,
        server_name=server_name,
        batch_mode_enabled=True,
        is_frozen=False,
    )

    for arg in job_args:
        assert arg in cmd


def test_prepare_command_custom_settings_paths():
    recording_path = Path("/path/to/recording")
    action_name = "Plugin.action"
    server_name = "neon-player-server-0"
    recording_settings_path = Path("/custom/recording_settings.json")
    workspace_settings_path = Path("/custom/workspace_settings.json")

    cmd = prepare_command(
        recording_path,
        action_name,
        args=[],
        server_name=server_name,
        batch_mode_enabled=True,
        is_frozen=False,
        recording_settings_path=recording_settings_path,
        workspace_settings_path=workspace_settings_path,
    )

    assert "--recording-settings" in cmd
    assert str(recording_settings_path) in cmd
    assert "--workspace-settings" in cmd
    assert str(workspace_settings_path) in cmd
