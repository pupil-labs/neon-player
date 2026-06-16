from pathlib import Path
from unittest.mock import patch

import pytest

from pupil_labs.neon_player.plugin_management.dependencies import (
    _check_dependencies_for_plugin_script,
    is_dependency_installed,
    get_installed_packages,
)


def _plugin_script(deps: list[str]) -> str:
    dep_lines = "\n".join(f'#     "{d}",' for d in deps)
    return f"""\
# /// script
# requires-python = ">=3.12"
# dependencies = [
{dep_lines}
# ]
# ///
import os
"""


def test_is_dependency_installed__installed():
    installed_packages = {"numpy": "2.0"}
    assert is_dependency_installed("numpy", installed_packages)


def test_is_dependency_installed__not_installed():
    installed_packages = {"numpy": "2.0"}
    assert not is_dependency_installed("pandas", installed_packages)


def test_is_dependency_installed__installed_version_satisfies_specifier():
    installed_packages = {"numpy": "2.0"}
    assert is_dependency_installed("numpy>=2.0", installed_packages)


def test_is_dependency_installed__installed_version_does_not_satisfy_specifier():
    installed_packages = {"numpy": "2.0"}
    assert not is_dependency_installed("numpy<2.0", installed_packages)


def test_check_dependencies_for_plugin_script__all_deps_installed():
    script = _plugin_script(["numpy", "pandas"])
    with patch(
        "pupil_labs.neon_player.plugin_management.dependencies.get_installed_packages",
        return_value={"numpy": "2.0", "pandas": "1.5"},
    ):
        assert not _check_dependencies_for_plugin_script(script, "test_plugin")


def test_check_dependencies_for_plugin_script__missing_dep():
    script = _plugin_script(["numpy", "some-missing-lib"])
    with patch(
        "pupil_labs.neon_player.plugin_management.dependencies.get_installed_packages",
        return_value={"numpy": "2.0"},
    ):
        result = _check_dependencies_for_plugin_script(script, "test_plugin")
        assert len(result) == 1
        assert "some-missing-lib" in result[0]
