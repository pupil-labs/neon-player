from unittest.mock import Mock

import pytest

from pupil_labs.neon_player.app import NeonPlayerApp


def test_job_manager_saves_settings_by_default(
    qapp: NeonPlayerApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(qapp, "save_settings", Mock(side_effect=ValueError("stop")))
    monkeypatch.setattr(type(qapp), "headless", False)

    with pytest.raises(ValueError, match="stop"):
        qapp.job_manager.run_background_action("Mock job", "MockPlugin.mock_action")
    qapp.save_settings.assert_called_once_with(force=True)


def test_job_manager_does_not_save_settings_when_requested(
    qapp: NeonPlayerApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(qapp, "save_settings", Mock(side_effect=ValueError("stop")))
    monkeypatch.setattr(type(qapp), "headless", False)

    qapp.job_manager.run_background_action(
        "Mock job", "MockPlugin.mock_action", save_settings=False
    )
    qapp.save_settings.assert_not_called()
