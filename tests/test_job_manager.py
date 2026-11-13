from unittest.mock import Mock

import pytest

from pupil_labs.neon_player.app import NeonPlayerApp


@pytest.mark.parametrize("save_settings", [True, False])
def test_job_manager_saves_settings_when_requested(
    save_settings: bool, qapp: NeonPlayerApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(qapp, "save_settings", Mock(side_effect=ValueError("stop")))
    monkeypatch.setattr(type(qapp), "headless", False)
    with pytest.raises(ValueError, match="stop"):
        qapp.job_manager.run_background_action(
            "Mock job", "MockPlugin.mock_action", save_settings=save_settings
        )

    qapp.save_settings.assert_called_once_with(force=save_settings)
