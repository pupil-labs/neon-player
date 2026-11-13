from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pupil_labs.neon_player import app as app_module
from pupil_labs.neon_player.app import NeonPlayerApp


@pytest.mark.parametrize("plugin_name", [None, "example.py"])
def test_install_dependencies__interactive(
    monkeypatch: pytest.MonkeyPatch, plugin_name: str | None
) -> None:
    app = SimpleNamespace(headless=False)
    dialog = Mock()
    monkeypatch.setattr(app_module, "DependencyInstallationDialog", dialog)

    NeonPlayerApp._install_dependencies(app, ["example==2.0"], plugin_name)

    dialog.assert_called_once()
    dependencies, title, message = dialog.call_args.args
    assert dependencies == ["example==2.0"]
    if plugin_name is None:
        assert title == "Neon Player - Dependency Compatibility"
        assert "compatibility with the current version of Neon Player" in message
    else:
        assert title == "Neon Player - Plugin Requirements"
        assert "<b>example.py</b> plugin file requires" in message
        assert "1 package(s)" in message
    dialog.return_value.exec.assert_called_once_with()


def test_install_dependencies__headless(monkeypatch: pytest.MonkeyPatch) -> None:
    app = SimpleNamespace(headless=True, job_manager=Mock())
    installer = Mock()
    dialog = Mock()
    monkeypatch.setattr(app_module, "install_dependencies", installer)
    monkeypatch.setattr(app_module, "DependencyInstallationDialog", dialog)

    NeonPlayerApp._install_dependencies(app, ["example==2.0"])

    installer.assert_called_once_with(["example==2.0"])
    app.job_manager.run_in_foreground.assert_called_once_with(installer.return_value)
    dialog.assert_not_called()


@pytest.mark.parametrize("headless", [False, True])
def test_install_dependencies__empty(
    monkeypatch: pytest.MonkeyPatch, headless: bool
) -> None:
    app = SimpleNamespace(headless=headless, job_manager=Mock())
    installer = Mock()
    dialog = Mock()
    monkeypatch.setattr(app_module, "install_dependencies", installer)
    monkeypatch.setattr(app_module, "DependencyInstallationDialog", dialog)

    NeonPlayerApp._install_dependencies(app, [])

    installer.assert_not_called()
    dialog.assert_not_called()
    app.job_manager.run_in_foreground.assert_not_called()
